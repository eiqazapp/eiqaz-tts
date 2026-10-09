#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
توليد عينات صوتية قابلة للاستماع للمقارنة بين النسخ
====================================================
لنصوص ذهبية مختارة، يولد WAV لكل مسار:
  - baseline: المسار الأصلي PyTorch (infer.synthesize — مرجع الجودة)
  - fp32 / fp16 / int8dyn / int8static: نفس التوكنز عبر جلسات ONNX

المخرجات في docs/samples/ (تُرفع للمستودع — أدلة استماع مباشرة).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(INF_DIR)
MODELS = os.path.join(REPO, 'web-exp', 'models')
OUT = os.path.join(REPO, 'docs', 'samples')

sys.path.insert(0, INF_DIR)
sys.path.insert(0, os.path.join(INF_DIR, 'lib'))
sys.path.insert(0, os.path.join(INF_DIR, 'lib', 'mixer_repo'))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402

import infer  # noqa: E402

SAMPLE_IDS = ['egy_short_01', 'marker_qaf_01', 'msa_01']


def main():
    with open(os.path.join(HERE, 'golden_texts.json'), encoding='utf-8') as f:
        gold = json.load(f)
    items = {g['id']: g for g in gold['items']}

    os.makedirs(OUT, exist_ok=True)
    ckpt = infer.resolve_checkpoint(None)
    model, it = infer.load_model(ckpt)
    print(f'[samples] checkpoint: {os.path.basename(ckpt)} (iter {it})')

    toks_ms, toks_egy, ids_of = infer.get_tokenizer()
    sessions = {}
    for vid, fname in [('fp32', 'mixertts_fp32.onnx'),
                       ('fp16', 'mixertts_fp16.onnx'),
                       ('int8dyn', 'mixertts_int8dyn.onnx'),
                       ('int8static', 'mixertts_int8static.onnx')]:
        p = os.path.join(MODELS, fname)
        sessions[vid] = ort.InferenceSession(
            p, providers=['CPUExecutionProvider'])
    vocos = ort.InferenceSession(os.path.join(MODELS, 'vocos22.onnx'),
                                 providers=['CPUExecutionProvider'])

    manifest = []
    for sid in SAMPLE_IDS:
        item = items[sid]
        res = infer.prepare_text_rich(item['text'], 'manual', item['dialect'])
        toks = toks_ms(res['text']) if item['dialect'] == 'msa' \
            else toks_egy(res['text'])
        ids = ids_of(toks)

        # 1) المرجع: المسار الأصلي بالكامل (mel_to_wav الإنتاجي)
        base_wav = os.path.join(OUT, f'{sid}_baseline_pytorch.wav')
        t0 = time.perf_counter()
        infer.synthesize(model, res['text'], item['dialect'], 0, 1.0,
                         base_wav, 0.005)
        base_s = time.perf_counter() - t0
        manifest.append({'id': sid, 'variant': 'baseline_pytorch',
                         'file': os.path.basename(base_wav),
                         'gen_s': round(base_s, 3)})
        print(f'  {sid} baseline: {base_s:.2f}s')

        # 2) نسخ ONNX — نفس التوكنز ثم vocos
        for vid, sess in sessions.items():
            t0 = time.perf_counter()
            mel = sess.run(None, {
                'text': np.array([ids], dtype=np.int64),
                'pace': np.array(1.0, dtype=np.float32),
                'speaker': np.array(0, dtype=np.int64),
            })[0]
            wave = vocos.run(None, {
                'mel_spec': mel.astype('float32'),
                'denoise': np.array([0.005], dtype='float32')})[0][0]
            gen_s = time.perf_counter() - t0
            out_wav = os.path.join(OUT, f'{sid}_{vid}.wav')
            sf.write(out_wav, wave, 22050, subtype='PCM_16')
            manifest.append({'id': sid, 'variant': vid,
                             'file': os.path.basename(out_wav),
                             'gen_s': round(gen_s, 3),
                             'audio_s': round(len(wave) / 22050, 2)})
            print(f'  {sid} {vid}: {gen_s:.2f}s '
                  f'({len(wave)/22050:.1f}s audio)')

    with open(os.path.join(OUT, 'samples_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'checkpoint': os.path.basename(ckpt),
                   'iter': str(it),
                   'note_ar': ('عينات استماع للمقارنة — نفس النص والتوكنز '
                               'والمتحدث وpace لكل النسخ؛ baseline هو مسار '
                               'PyTorch الإنتاجي كما هو'),
                   'samples': manifest}, f, ensure_ascii=False, indent=2)
    print(f'[samples] written: {OUT}')


if __name__ == '__main__':
    main()
