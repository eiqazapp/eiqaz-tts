#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msa_prep_clartts.py — تجهيز ClArTTS (ذكر فصحى كلاسيكي، مشكول يدويًا،
CC-BY-4.0، 12h) كوحدات تدريب متوافقة تمامًا مع خط NileTTS:

  - صوت: 22050Hz mono PCM16، ذروة <= 0.95 (قاعدة NileTTS-prep نفسها)
  - نص: تحويل الترقيم ، → , / ؟ → ? (اصطلاح ph2 حرفيًا)، تشكيل كامل محفوظ
  - فلاتر الوحدة: مدة >= 3s، >= 10 حروف عربية، لا ASCII alpha،
    توكنات <= 180 (كتابة) مع علم <= 160 (أهلية train-pool)
  - التوكنات عبر toks_ms (مسار فصحى — توكن q محفوظ!) + إحصاء تعرض القاف

يعالج شظايا parquet واحدة تلو الأخرى (قرص محدود): تنزيل → معالجة →
حذف. قابل للاستئناف: ملف records لكل شظية + قائمة done.
الاستخدام:
  python3 msa_prep_clartts.py [--max-shards N] [--split train|test|all]
"""
import argparse
import json
import os
import re
import sys
import time
from math import gcd

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, '/home/z/my-project/work/tts_arabic_pkg')
from tts_arabic.text import (arabic_to_buckwalter, buckwalter_to_phonemes,
                             phonemes_to_tokens)

BASE = '/home/z/my-project/work'
OUT = f'{BASE}/msa_prep'
TMP = f'{BASE}/msa_tmp'
REC_DIR = f'{OUT}/records'
DONE_PATH = f'{TMP}/clartts_done.json'
TARGET_SR = 22050
MIN_DUR_S = 3.0
MIN_AR_CHARS = 10
WRITE_TOK_CAP = 180
TRAIN_TOK_CAP = 160

AR_CHARS = re.compile(r'[\u0621-\u064A]')
ASCII_ALPHA = re.compile(r'[A-Za-z]')
TASHKEEL = re.compile(r'[\u064B-\u0652\u0670]')

UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}


def get_tokenizer():
    def toks_ms(text):
        return phonemes_to_tokens(
            buckwalter_to_phonemes(arabic_to_buckwalter(text)))
    return toks_ms


TOKS = get_tokenizer()


def clean_text(t):
    """اصطلاح ph2 حرفيًا: فقط ، → , و ؟ → ? — كل شيء آخر يبقى كما هو."""
    return t.replace('\u060C', ',').replace('\u061F', '?').strip()


def resample_to_target(x, sr):
    if sr == TARGET_SR:
        return x
    g = gcd(int(sr), TARGET_SR)
    return resample_poly(x, TARGET_SR // g, int(sr) // g)


def normalize_audio(x):
    """float64 → float32 في [-1,1]؛ دفاعي ضد مقاييس int16."""
    m = float(np.abs(x).max()) if x.size else 0.0
    if m > 2.0:                      # مقياس int16 مسطح
        x = x / 32768.0
        m /= 32768.0
    elif m > 1.0:                    # تجاوز طفيف
        x = x / m
        m = 1.0
    if m > 0.95:                     # قاعدة NileTTS: ذروة <= 0.95
        x = x * (0.95 / m)
    return x.astype(np.float32)


def process_row(row, seq, split):
    """يعيد dict السجل أو (None, reason)."""
    text_raw = row.get('text') or ''
    text = clean_text(text_raw)
    if len(AR_CHARS.findall(text)) < MIN_AR_CHARS:
        return None, 'short_text'
    if ASCII_ALPHA.search(text):
        return None, 'ascii_alpha'
    try:
        toks = TOKS(text)
    except Exception:
        return None, 'tokenize_error'
    n_tok = len(toks)
    if n_tok > WRITE_TOK_CAP:
        return None, 'over_tokens'

    audio = row.get('audio')
    if audio is None or len(audio) == 0:
        return None, 'no_audio'
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:                  # استريو → مونو
        x = x.mean(axis=1)
    sr = int(row.get('sampling_rate') or 0)
    if sr <= 0:
        return None, 'bad_sr'
    dur = len(x) / sr
    if dur < MIN_DUR_S:
        return None, 'short_audio'
    if dur > 30.0:
        return None, 'too_long'

    y = normalize_audio(resample_to_target(x, sr))
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)

    utt = f'clt_{seq:05d}'
    wav_rel = f'clartts/wavs/{utt}.wav'
    wav_path = f'{OUT}/{wav_rel}'
    sf.write(wav_path, pcm, TARGET_SR, subtype='PCM_16')

    n_q = sum(1 for t in toks if t == 'q')
    dens = len(TASHKEEL.findall(text)) / max(1, len(AR_CHARS.findall(text)))
    return {
        'utt': utt, 'wav': wav_rel, 'split': split, 'dialect': 'msa',
        'speaker': 'clartts_male',
        'text': text, 'text_raw': text_raw.strip(),
        'n_tokens': n_tok, 'train_pool_ok': n_tok <= TRAIN_TOK_CAP,
        'n_q': n_q, 'tashkeel_density': round(dens, 3),
        'duration_s': round(dur, 2), 'sr_orig': sr,
        'clip_id': row.get('file') or '',
        'source': 'MBZUAI/ClArTTS', 'license': 'CC-BY-4.0',
    }, None


def shard_list():
    r = requests.get(
        'https://datasets-server.huggingface.co/parquet?dataset=MBZUAI/ClArTTS',
        headers=UA, timeout=60)
    r.raise_for_status()
    return [(f['url'], f['size']) for f in r.json()['parquet_files']]


def process_shard(url, split_tag):
    import pyarrow.parquet as pq
    name = url.rsplit('/', 1)[-1]
    key = f'{split_tag}/{name}'
    local = f'{TMP}/clartts_{split_tag}_{name}'
    t0 = time.time()
    r = requests.get(url, headers=UA, timeout=(15, 600), stream=True)
    r.raise_for_status()
    with open(local, 'wb') as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    dl_s = time.time() - t0

    t1 = time.time()
    pf = pq.ParquetFile(local)
    seq_base = next_seq()
    recs, drops = [], {}
    n = 0
    for batch in pf.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            n += 1
            seq = seq_base + len(recs)
            rec, why = process_row(row, seq, split_tag)
            if rec is None:
                drops[why] = drops.get(why, 0) + 1
            else:
                recs.append(rec)
    # ملف السجل لهذه الشظية — ذري
    rec_path = f'{REC_DIR}/clartts_{split_tag}_{name.replace(".parquet", "")}.json'
    tmp = rec_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False)
    os.replace(tmp, rec_path)
    try:
        os.remove(local)
    except OSError:
        pass
    return {'key': key, 'rows': n, 'kept': len(recs), 'drops': drops,
            'dl_s': round(dl_s, 1), 'proc_s': round(time.time() - t1, 1)}


def next_seq():
    """يعيد أول تسلسل حر — مسح فعلي للدليل في كل استدعاء
    (تصحيح: الكاش الثابت كان يسبب كتابة الشظايا فوق بعضها)."""
    mx = -1
    d = f'{OUT}/clartts/wavs'
    if os.path.isdir(d):
        for fn in os.listdir(d):
            try:
                mx = max(mx, int(fn[4:9]))
            except ValueError:
                pass
    return mx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-shards', type=int, default=0)
    ap.add_argument('--split', default='all',
                    choices=['all', 'train', 'test'])
    args = ap.parse_args()

    for p in (f'{OUT}/clartts/wavs', REC_DIR, TMP):
        os.makedirs(p, exist_ok=True)
    done = set()
    if os.path.exists(DONE_PATH):
        done = set(json.load(open(DONE_PATH)))

    shards = shard_list()
    if args.split != 'all':
        shards = [s for s in shards if f'/{args.split}/' in s[0]]
    def key_of(u):
        tag = 'test' if '/test/' in u else 'train'
        return f'{tag}/{u.rsplit("/", 1)[-1]}'

    todo = [s for s in shards if key_of(s[0]) not in done]
    # ترتيب: test أولاً (صغير — نتيجة سريعة) ثم train
    todo.sort(key=lambda s: (0 if '/test/' in s[0] else 1, s[0]))
    if args.max_shards:
        todo = todo[:args.max_shards]

    print(f'shards total={len(shards)} done={len(done)} todo={len(todo)}')
    t0 = time.time()
    for i, (url, size) in enumerate(todo):
        split_tag = 'test' if '/test/' in url else 'train'
        info = process_shard(url, split_tag)
        done.add(info['key'])
        json.dump(sorted(done), open(DONE_PATH, 'w'))
        el = time.time() - t0
        print(f"[{i+1}/{len(todo)}] {info['key']} rows={info['rows']} "
              f"kept={info['kept']} drops={info['drops']} "
              f"dl={info['dl_s']}s proc={info['proc_s']}s "
              f"elapsed={el:.0f}s", flush=True)
    print('DONE clartts stage')


if __name__ == '__main__':
    main()
