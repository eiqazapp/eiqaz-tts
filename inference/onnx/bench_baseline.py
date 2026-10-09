#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
قياس خط الأساس — المسار الأصلي PyTorch (مرجع الجودة R1F)
============================================================
يقيس على النصوص الذهبية (golden_texts.json) بنفس طريقة infer.py تمامًا:
زمن تحميل النموذج، زمن التوليد، مدة الصوت، TTFA (زمن أول صوت جاهز)،
RTF (زمن التوليد ÷ مدة الصوت)، والأخطاء — ويكتب WAV لكل تجربة.

يُشغَّل مرتين:
  1) في بيئة التطوير بـ checkpoint عشوائي (states_sanity_random.pth)
     → تحقق ميكانيكي فقط (الأرقام تعكس البنية لا الجودة).
  2) على جهاز Windows لدى المستخدم بـ checkpoint الإنتاج الحقيقي
     (states_79590.pth) → أرقام خط الأساس الرسمية للجودة والسرعة.

الاستخدام:
    python bench_baseline.py --checkpoint states_79590.pth --out baseline_results.json
    python bench_baseline.py --checkpoint states_sanity_random.pth --tag sanity --out baseline_sanity.json
"""
import argparse
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)              # inference/
sys.path.insert(0, INF_DIR)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import infer  # noqa: E402  (الوحدة المرجعية نفسها — بلا أي تعديل)


def bench_one(model, text, dialect, speaker, pace, out_wav, qaf_mode='auto'):
    """قياس توليد واحد — نفس نداءات webapp.run_job (prepare_text_rich + synthesize)."""
    t0 = time.perf_counter()
    res = infer.prepare_text_rich(text, 'auto', dialect, qaf_mode)
    prep_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    n_tokens, n_secs = infer.synthesize(
        model, res['text'], dialect, speaker, pace, out_wav, 0.005,
        qaf_mode=qaf_mode, qaf_word_actions=res['qaf_actions'] or None,
        qaf_native_skel=res['qaf_native'] or None, peak_normalize=True,
        verbose=False)
    gen_s = time.perf_counter() - t1
    # في المسار الأصلي (غير المتدفق) أول صوت جاهز = اكتمال التوليد
    ttfa_ms = (time.perf_counter() - t0) * 1000.0
    rtf = gen_s / n_secs if n_secs > 0 else None
    return {
        'n_tokens': n_tokens, 'audio_secs': round(n_secs, 3),
        'prep_ms': round(prep_ms, 1), 'gen_s': round(gen_s, 3),
        'ttfa_ms': round(ttfa_ms, 1),
        'rtf': round(rtf, 4) if rtf is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description='قياس خط الأساس PyTorch')
    ap.add_argument('--checkpoint', default=None,
                    help='اسم checkpoint في checkpoints/ (افتراضي: أحدث states_*.pth)')
    ap.add_argument('--gold', default=os.path.join(HERE, 'golden_texts.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'baseline_results.json'))
    ap.add_argument('--wav-dir', default=os.path.join(HERE, 'baseline_wavs'))
    ap.add_argument('--tag', default=None,
                    help='وسم النتائج (مثال: sanity للوضع العشوائي)')
    ap.add_argument('--repeat', type=int, default=1,
                    help='تكرار كل جملة (تجارب دافئة)')
    args = ap.parse_args()

    ckpt = infer.resolve_checkpoint(args.checkpoint)
    ckpt_name = os.path.basename(ckpt)
    print(f'[baseline] checkpoint: {ckpt_name}')

    t0 = time.perf_counter()
    model, it = infer.load_model(ckpt)
    load_s = time.perf_counter() - t0
    print(f'[baseline] model loaded in {load_s:.2f}s (iter {it})')

    with open(args.gold, encoding='utf-8') as f:
        gold = json.load(f)

    os.makedirs(args.wav_dir, exist_ok=True)
    results = []
    for item in gold['items']:
        rep = None
        err = None
        try:
            for r in range(args.repeat):
                wav = os.path.join(args.wav_dir,
                                   f"{item['id']}_spk0_r{r}.wav")
                rep = bench_one(model, item['text'], item['dialect'], 0,
                                1.0, wav)
        except SystemExit as e:
            err = str(e.code)
        except Exception as e:                              # noqa: BLE001
            err = f'{type(e).__name__}: {e}'
        row = {'id': item['id'], 'dialect': item['dialect'],
               'tags': item['tags']}
        if rep:
            row.update(rep)
        if err:
            row['error'] = err
        results.append(row)
        status = 'ERR: ' + err if err else (
            f"ttfa={rep['ttfa_ms']}ms rtf={rep['rtf']}")
        print(f"  {item['id']:<20} {status}")

    summary = {
        'kind': 'pytorch_baseline',
        'tag': args.tag or ('sanity' if 'sanity' in ckpt_name else 'production'),
        'checkpoint': ckpt_name,
        'torch': __import__('torch').__version__,
        'python': platform.python_version(),
        'machine': platform.processor() or platform.machine(),
        'model_iter': str(it),
        'model_load_s': round(load_s, 3),
        'n_items': len(results),
        'n_errors': sum(1 for r in results if 'error' in r),
        'results': results,
    }
    ok = [r for r in results if 'rtf' in r and r['rtf'] is not None]
    if ok:
        summary['rtf_median'] = round(sorted(r['rtf'] for r in ok)
                                      [len(ok) // 2], 4)
        summary['ttfa_median_ms'] = round(sorted(r['ttfa_ms'] for r in ok)
                                          [len(ok) // 2], 1)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[baseline] written: {args.out} "
          f"(rtf_median={summary.get('rtf_median')} "
          f"ttfa_median={summary.get('ttfa_median_ms')}ms)")


if __name__ == '__main__':
    main()
