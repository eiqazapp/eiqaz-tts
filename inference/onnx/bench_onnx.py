#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
قياس أداء نسخ ONNX على الخادم (Python) — TTFA / RTF / تحميل
==============================================================
لكل نسخة صوتية في web-exp/models (من manifest):
  - زمن إنشاء الجلسة أول مرة (باردًا)
  - لكل نص ذهبي: زمن mel + زمن المُصوِّت + مدة الصوت + RTF
  - محاكاة بث: TTFA = زمن أول مقطع جاهز (تقسيم النص عند حدود الجمل
    وتوليد أول مقطع فقط — المتبقي يصل لاحقًا في الواقع)

النتائج JSON في inference/onnx/bench_onnx_results.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(INF_DIR)
MODELS = os.path.join(REPO, 'web-exp', 'models')

sys.path.insert(0, INF_DIR)
sys.path.insert(0, os.path.join(INF_DIR, 'lib', 'mixer_repo'))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

import infer  # noqa: E402


def run_mel(sess, ids, pace=1.0, speaker=0):
    return sess.run(None, {
        'text': np.array([ids], dtype=np.int64),
        'pace': np.array(pace, dtype=np.float32),
        'speaker': np.array(speaker, dtype=np.int64),
    })[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', default=os.path.join(HERE, 'golden_texts.json'))
    ap.add_argument('--out', default=os.path.join(HERE,
                                                  'bench_onnx_results.json'))
    ap.add_argument('--repeat', type=int, default=2)
    args = ap.parse_args()

    with open(os.path.join(MODELS, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.gold, encoding='utf-8') as f:
        gold = json.load(f)

    # توكنز ذهبية + تقسيم جمل (كما يفعل المتصفح)
    toks_ms, toks_egy, ids_of = infer.get_tokenizer('auto')
    cases = []
    for item in gold['items']:
        try:
            res = infer.prepare_text_rich(item['text'], 'never',
                                          item['dialect'], 'auto')
            acts = res['qaf_actions'] or None
            nat = frozenset(res['qaf_native']) or None
            if item['dialect'] == 'msa':
                toks = infer.get_msa_synthesis_tokens(res['text'], 'auto',
                                                      acts)
            else:
                _, te, ids_of = infer.get_tokenizer('auto', acts, nat)
                toks = te(res['text'])
            ids = ids_of(toks)
            if 2 <= len(ids) <= 160:
                cases.append((item['id'], item['text'], ids))
        except Exception:                                  # noqa: BLE001
            continue
    print(f'[bench] حالات ذهبية: {len(cases)}')

    vocos_path = os.path.join(MODELS, 'vocos22_fp16.onnx')
    if not os.path.exists(vocos_path):
        vocos_path = os.path.join(MODELS, 'vocos22.onnx')
    vocos = ort.InferenceSession(vocos_path,
                                 providers=['CPUExecutionProvider'])

    results = []
    for v in manifest['acoustic_variants']:
        if v.get('status') == 'failed' or 'file' not in v:
            continue
        path = os.path.join(MODELS, v['file'])
        try:
            t0 = time.perf_counter()
            sess = ort.InferenceSession(path,
                                        providers=['CPUExecutionProvider'])
            create_s = time.perf_counter() - t0

            rows = []
            n_err = 0
            for rep in range(args.repeat):
                for gid, text, ids in cases:
                    try:
                        t0 = time.perf_counter()
                        mel = run_mel(sess, ids)
                        t_mel = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        wave = vocos.run(None, {
                            'mel_spec': mel.astype('float32'),
                            'denoise': np.array([0.005], dtype='float32'),
                        })[0][0]
                        t_voc = time.perf_counter() - t0
                        audio_s = len(wave) / 22050.0
                        gen = t_mel + t_voc
                        rows.append({
                            'id': gid, 'rep': rep, 'n_tokens': len(ids),
                            'mel_s': round(t_mel, 4),
                            'vocos_s': round(t_voc, 4),
                            'gen_s': round(gen, 4),
                            'audio_s': round(audio_s, 3),
                            'rtf': round(gen / audio_s, 4) if audio_s > 0 else None,
                        })
                    except Exception as e:              # noqa: BLE001
                        # مدد صفورية (أوزان عشوائية) — نفس قيد النموذج
                        # الأصلي في بايثون (kernel > input) — سجل ولا تُفشل البقية
                        n_err += 1
                        rows.append({'id': gid, 'rep': rep, 'error':
                                     f'{type(e).__name__}'})

            # محاكاة TTFA: أول جملة حقيقية من نص البث الطويل (تقسيم نصي
            # عند نهاية الجملة — لا قطع توكنز اعتباطي يفسد المدد)
            ttfa_sim_ms = None
            try:
                stream_text = next(g['text'] for g in gold['items']
                                   if g['id'] == 'stream_long_01')
                first_sent = stream_text.split('.')[0].strip() + '.'
                res1 = infer.prepare_text_rich(first_sent, 'never', 'egy',
                                               'auto')
                _, te1, _ = infer.get_tokenizer('auto', res1['qaf_actions']
                                                or None, None)
                ids1 = infer.get_tokenizer('auto')[2](te1(res1['text']))
                t0 = time.perf_counter()
                mel1 = run_mel(sess, ids1)
                vocos.run(None, {'mel_spec': mel1.astype('float32'),
                                 'denoise': np.array([0.005], dtype='float32')})
                ttfa_sim_ms = round((time.perf_counter() - t0) * 1000, 1)
            except Exception:                              # noqa: BLE001
                ttfa_sim_ms = None

            gen_all = [r['gen_s'] for r in rows]
            rtf_all = [r['rtf'] for r in rows if r['rtf']]
            results.append({
                'id': v['id'], 'file': v['file'],
                'size_bytes': v['size_bytes'],
                'session_create_s': round(create_s, 3),
                'gen_median_s': round(sorted(gen_all)[len(gen_all) // 2], 4),
                'rtf_median': round(sorted(rtf_all)[len(rtf_all) // 2], 4),
                'ttfa_first_chunk_sim_ms': ttfa_sim_ms,
                'n_runs': len(rows),
                'n_run_errors': n_err,
                'parity_verdict': v.get('parity', {}).get('verdict'),
                'rows': rows,
            })
            print(f"  {v['id']:<11} {v['size_bytes']/1e6:6.1f}MB "
                  f"create={create_s:.2f}s rtf={results[-1]['rtf_median']} "
                  f"ttfa_sim={ttfa_sim_ms}ms err={n_err}")
        except Exception as e:                              # noqa: BLE001
            results.append({'id': v['id'], 'error': f'{type(e).__name__}: {e}'})
            print(f"  {v['id']:<11} FAILED: {e}")

    out = {
        'kind': 'onnx_server_bench',
        'vocoder': os.path.basename(vocos_path),
        'repeat': args.repeat,
        'note_ar': ('قِست بنماذج sanity (أوزان عشوائية) — المدد الصوتية '
                    'ضئيلة فالـRTF متضخم زائفًا؛ أعِد التشغيل بالأوزان '
                    'الحقيقية على Windows للأرقام الرسمية'),
        'results': results,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'[bench] written: {args.out}')


if __name__ == '__main__':
    main()
