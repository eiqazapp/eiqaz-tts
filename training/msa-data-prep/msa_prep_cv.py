#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msa_prep_cv.py — تجهيز صوت أنثى فصحى من مرآة Common Voice 18 العربي
(MohamedRashad/common-voice-18-arabic — CC0-1.0) كوحدات تدريب متوافقة
مع خط NileTTS:

المراحل (قابلة للاستئناف):
  scan    — تنزيل parquet للتقسيمات النظيفة (train/validation/test) +
            مسح الإناث: sf.info لكل مقطع (مدة) → ترتيب client_id بالساعات
            → اختيار أفضل متحدثتين
  process — للمتحدثات المختارات فقط: فك mp3 → 22050Hz mono PCM16 ذروة<=0.95
            + نص: ، → , / ؟ → ? ثم تشكيل catt محلي (كاش json) + toks_ms
            (مسار فصحى — توكن q!) + فلاتر NileTTS (>=3s، >=10 حروف عربية،
            لا ASCII alpha، <=180 توكن)

الاستخدام:
  python3 msa_prep_cv.py --phase scan
  python3 msa_prep_cv.py --phase process
"""
import argparse
import io
import json
import os
import random
import re
import sys
import time

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

BASE = '/home/z/my-project/work'
OUT = f'{BASE}/msa_prep'
TMP = f'{BASE}/msa_tmp'
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
DATASET = 'MohamedRashad/common-voice-18-arabic'
SPLITS = ['train', 'validation', 'test']
TARGET_SR = 22050
MIN_DUR_S = 3.0
MAX_DUR_S = 30.0
MIN_AR_CHARS = 10
WRITE_TOK_CAP = 180
TRAIN_TOK_CAP = 160
N_SPEAKERS = 2

AR_CHARS = re.compile(r'[\u0621-\u064A]')
ASCII_ALPHA = re.compile(r'[A-Za-z]')
TASHKEEL = re.compile(r'[\u064B-\u0652\u0670]')

SCAN_PATH = f'{OUT}/cv_female_scan.json'
CATT_CACHE = f'{OUT}/catt_cache.json'
PROG_PATH = f'{OUT}/cv_progress.json'
REC_PATH = f'{OUT}/records/cv_records.json'


def shard_list():
    r = requests.get(
        f'https://datasets-server.huggingface.co/parquet?dataset={DATASET}',
        headers=UA, timeout=60)
    r.raise_for_status()
    return [(f['url'], f['size'], f['split'])
            for f in r.json()['parquet_files'] if f['split'] in SPLITS]


def local_path(split, url):
    return f'{TMP}/cv_{split}_{url.rsplit("/", 1)[-1]}'


def ensure_downloaded():
    files = []
    for url, size, split in shard_list():
        lp = local_path(split, url)
        if os.path.exists(lp) and os.path.getsize(lp) == size:
            files.append((split, lp))
            continue
        print(f'downloading {split} {os.path.basename(lp)} '
              f'({size/1e6:.0f}MB)...', flush=True)
        t0 = time.time()
        r = requests.get(url, headers=UA, timeout=(15, 900), stream=True)
        r.raise_for_status()
        with open(lp + '.part', 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(lp + '.part', lp)
        print(f'  done in {time.time()-t0:.0f}s', flush=True)
        files.append((split, lp))
    return files


def phase_scan():
    import pyarrow.parquet as pq
    files = ensure_downloaded()
    stats = {}   # client_id -> {clips, seconds, splits}
    t0 = time.time()
    for split, lp in files:
        pf = pq.ParquetFile(lp)
        n_fem = 0
        for batch in pf.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if not str(row.get('gender') or '').startswith('female'):
                    continue
                n_fem += 1
                aud = row.get('audio') or {}
                data = aud.get('bytes')
                if not data:
                    continue
                try:
                    info = sf.info(io.BytesIO(data))
                    dur = info.frames / float(info.samplerate)
                except Exception:
                    continue
                st = stats.setdefault(row['client_id'],
                                       {'clips': 0, 'seconds': 0.0})
                st['clips'] += 1
                st['seconds'] += dur
        print(f'{split}: scanned rows, females={n_fem}, '
              f'elapsed={time.time()-t0:.0f}s', flush=True)
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]['seconds'])
    top = [{
        'client_id': cid,
        'clips': v['clips'],
        'hours': round(v['seconds'] / 3600, 2),
    } for cid, v in ranked[:N_SPEAKERS]]
    scan = {
        'scanned_at': time.strftime('%Y-%m-%d %H:%M'),
        'n_female_clients': len(stats),
        'top_speakers': top,
        'next_8_for_reference': [
            {'client_id': cid, 'clips': v['clips'],
             'hours': round(v['seconds'] / 3600, 2)}
            for cid, v in ranked[2:10]
        ],
    }
    tmp = SCAN_PATH + '.tmp'
    json.dump(scan, open(tmp, 'w'), ensure_ascii=False, indent=1)
    os.replace(tmp, SCAN_PATH)
    print(json.dumps(scan, ensure_ascii=False, indent=1))


def get_catt():
    sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference')
    sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')
    import infer
    return infer.catt_vocalize


class CattVocalizer:
    """catt مع كاش دائم على القرص (استئناف بعد الانقطاع)."""

    def __init__(self):
        self.cache = {}
        if os.path.exists(CATT_CACHE):
            try:
                self.cache = json.load(open(CATT_CACHE, encoding='utf-8'))
            except Exception:
                self.cache = {}
        self.fn = None
        self.n_calls = 0

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        if self.fn is None:
            print('loading catt (infer import)...', flush=True)
            self.fn = get_catt()
        v = self.fn(text) or text
        self.cache[text] = v
        self.n_calls += 1
        if self.n_calls % 200 == 0:
            self._save()
        return v

    def _save(self):
        tmp = CATT_CACHE + '.tmp'
        json.dump(self.cache, open(tmp, 'w', encoding='utf-8'),
                  ensure_ascii=False)
        os.replace(tmp, CATT_CACHE)

    def flush(self):
        self._save()


def clean_text(t):
    return t.replace('\u060C', ',').replace('\u061F', '?').strip()


def resample_to_target(x, sr):
    if sr == TARGET_SR:
        return x
    from math import gcd
    g = gcd(int(sr), TARGET_SR)
    return resample_poly(x, TARGET_SR // g, int(sr) // g)


def phase_process():
    import pyarrow.parquet as pq
    sys.path.insert(0, '/home/z/my-project/work/tts_arabic_pkg')
    from tts_arabic.text import (arabic_to_buckwalter, buckwalter_to_phonemes,
                                 phonemes_to_tokens)

    def toks_ms(text):
        return phonemes_to_tokens(
            buckwalter_to_phonemes(arabic_to_buckwalter(text)))

    scan = json.load(open(SCAN_PATH, encoding='utf-8'))
    selected = {s['client_id'] for s in scan['top_speakers']}
    spk_label = {cid: f"cvf_{cid[:8]}" for cid in selected}
    print('selected speakers:', {spk_label[c]: c for c in selected})

    files = ensure_downloaded()
    os.makedirs(f'{OUT}/cv_female/wavs', exist_ok=True)
    os.makedirs(f'{OUT}/records', exist_ok=True)

    prog = {'seq': 0, 'done_keys': []}
    if os.path.exists(PROG_PATH):
        prog = json.load(open(PROG_PATH, encoding='utf-8'))
    done_keys = set(prog['done_keys'])
    seq = prog['seq']

    records = []
    if os.path.exists(REC_PATH):
        records = json.load(open(REC_PATH, encoding='utf-8'))
    vocal = CattVocalizer()

    drops = {}
    t0 = time.time()
    for split, lp in files:
        pf = pq.ParquetFile(lp)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                cid = row.get('client_id')
                if cid not in selected:
                    continue
                key = f"{cid}|{row.get('path')}"
                if key in done_keys:
                    continue
                aud = row.get('audio') or {}
                data = aud.get('bytes')
                if not data:
                    drops['no_audio'] = drops.get('no_audio', 0) + 1
                    done_keys.add(key)
                    continue
                try:
                    x, sr = sf.read(io.BytesIO(data), dtype='float64')
                except Exception:
                    drops['decode_error'] = drops.get('decode_error', 0) + 1
                    done_keys.add(key)
                    continue
                if x.ndim == 2:
                    x = x.mean(axis=1)
                dur = len(x) / sr
                if dur < MIN_DUR_S:
                    drops['short_audio'] = drops.get('short_audio', 0) + 1
                    done_keys.add(key)
                    continue
                if dur > MAX_DUR_S:
                    drops['too_long'] = drops.get('too_long', 0) + 1
                    done_keys.add(key)
                    continue

                raw = clean_text(row.get('sentence') or '')
                if len(AR_CHARS.findall(raw)) < MIN_AR_CHARS:
                    drops['short_text'] = drops.get('short_text', 0) + 1
                    done_keys.add(key)
                    continue
                if ASCII_ALPHA.search(raw):
                    drops['ascii_alpha'] = drops.get('ascii_alpha', 0) + 1
                    done_keys.add(key)
                    continue
                diac = vocal(raw)
                try:
                    toks = toks_ms(diac)
                except Exception:
                    drops['tokenize_error'] = drops.get('tokenize_error', 0) + 1
                    done_keys.add(key)
                    continue
                if len(toks) > WRITE_TOK_CAP:
                    drops['over_tokens'] = drops.get('over_tokens', 0) + 1
                    done_keys.add(key)
                    continue

                y = resample_to_target(x, sr)
                m = float(np.abs(y).max()) if y.size else 0.0
                if m > 0.95:
                    y = y * (0.95 / m)
                pcm = (np.clip(y.astype(np.float32), -1.0, 1.0)
                       * 32767.0).astype(np.int16)

                utt = f'cvf_{seq:05d}'
                wav_rel = f'cv_female/wavs/{utt}.wav'
                sf.write(f'{OUT}/{wav_rel}', pcm, TARGET_SR,
                         subtype='PCM_16')

                rnd = random.Random(hash(key) & 0xffffffff)
                records.append({
                    'utt': utt, 'wav': wav_rel,
                    'split': 'train' if rnd.random() < 0.95 else 'eval',
                    'dialect': 'msa', 'speaker': spk_label[cid],
                    'text': diac, 'text_raw': row.get('sentence') or '',
                    'n_tokens': len(toks),
                    'train_pool_ok': len(toks) <= TRAIN_TOK_CAP,
                    'n_q': sum(1 for t in toks if t == 'q'),
                    'tashkeel_density': round(
                        len(TASHKEEL.findall(diac)) /
                        max(1, len(AR_CHARS.findall(diac))), 3),
                    'duration_s': round(dur, 2), 'sr_orig': sr,
                    'clip_id': row.get('path') or '',
                    'cv_split': split,
                    'age': row.get('age') or '',
                    'accent': row.get('accent') or '',
                    'source': DATASET, 'license': 'CC0-1.0',
                })
                seq += 1
                done_keys.add(key)
                if len(done_keys) % 100 == 0:
                    _persist(prog, done_keys, seq, records)
                    el = time.time() - t0
                    print(f'  {len(done_keys)} processed, '
                          f'{len(records)} kept, elapsed={el:.0f}s',
                          flush=True)
        # حفظ بعد كل ملف تقسيم
        _persist(prog, done_keys, seq, records)
    vocal.flush()
    _persist(prog, done_keys, seq, records)
    print(f'CV DONE: kept={len(records)} drops={drops} '
          f'catt_cached={len(vocal.cache)}')
    for split, lp in files:
        try:
            os.remove(lp)
        except OSError:
            pass


def _persist(prog, done_keys, seq, records):
    prog['seq'] = seq
    prog['done_keys'] = sorted(done_keys)
    tmp = PROG_PATH + '.tmp'
    json.dump(prog, open(tmp, 'w'))
    os.replace(tmp, PROG_PATH)
    tmp = REC_PATH + '.tmp'
    json.dump(records, open(tmp, 'w', encoding='utf-8'),
              ensure_ascii=False)
    os.replace(tmp, REC_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='scan', choices=['scan', 'process'])
    args = ap.parse_args()
    if args.phase == 'scan':
        phase_scan()
    else:
        phase_process()


if __name__ == '__main__':
    main()
