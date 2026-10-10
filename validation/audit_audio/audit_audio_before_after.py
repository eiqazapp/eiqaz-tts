#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_audio_before_after.py — توليد أصوات قبل/بعد للمقارنة الأمينة.

«قبل» = نص مخرج main (نفس الأوزان والإعدادات — الكود الصوتي متطابق
حرفيًا بين الفرعين؛ الفرق الوحيد نص التشكيل).
«بعد» = نص مخرج فرع الإصلاح.

لكل حالة: ملفان WAV + التوكنات + المدة + الفروق. تحذير صريح: اختلاف
الصوت ليس دليل جودة — التقييم النهائي بشرية مصرية.
"""
import json
import os
import sys

REPO = '/home/z/my-project/work/github_repo/eiqaz-tts'
INF = os.path.join(REPO, 'inference')
for p in (INF, os.path.join(INF, 'lib')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(INF)

import infer  # noqa: E402

OUT_DIR = '/home/z/my-project/download/audit_audio'
os.makedirs(OUT_DIR, exist_ok=True)
CKPT = os.path.join(INF, 'checkpoints', 'states_cont_180516.pth')

BEFORE = json.load(open('/tmp/main_texts.json', encoding='utf-8'))
AFTER_REPORT = json.load(
    open(os.path.join(OUT_DIR, 'audio_integration_report.json'),
         encoding='utf-8'))


def main():
    model, it = infer.load_model(CKPT)
    _, toks_egy, _ = infer.get_tokenizer()
    rows = []
    for row in AFTER_REPORT['cases']:
        cid = row['id']
        b = BEFORE.get(cid)
        if not b:
            continue
        before_text = b['main_text']
        after_text = row['final_text']
        wav_b = os.path.join(OUT_DIR, f'{cid}_before_main.wav')
        wav_a = os.path.join(OUT_DIR, f'{cid}_after_fix.wav')
        n_b, d_b = infer.synthesize(
            model, before_text, row['dialect'], speaker=0, pace=1.0,
            out_path=wav_b, denoise=0.005, peak_normalize=True)
        n_a, d_a = infer.synthesize(
            model, after_text, row['dialect'], speaker=0, pace=1.0,
            out_path=wav_a, denoise=0.005, peak_normalize=True)
        t_b, t_a = list(toks_egy(before_text)), list(toks_egy(after_text))
        r = {
            'id': cid, 'input': row['input'],
            'before_text': before_text, 'after_text': after_text,
            'before_tokens': t_b, 'after_tokens': t_a,
            'tokens_changed': t_b != t_a,
            'before_n_tokens': n_b, 'after_n_tokens': n_a,
            'before_duration_sec': round(d_b, 3),
            'after_duration_sec': round(d_a, 3),
            'wav_before': wav_b, 'wav_after': wav_a,
        }
        rows.append(r)
        print(f"[{cid}] {row['input']}")
        print(f"  قبل: {before_text} ({n_b} توكن، {d_b:.2f}ث)")
        print(f"  بعد: {after_text} ({n_a} توكن، {d_a:.2f}ث)")
        print(f"  التوكنات تغيرت: {t_b != t_a}")

    report = {
        'method': 'نفس الأوزان والإعدادات والمتحدث (0) وpace=1.0 — '
                  'الفرق الوحيد: نص التشكيل (مخرج main مقابل مخرج '
                  'فرع الإصلاح). الكود الصوتي متطابق بين الفرعين.',
        'settings': AFTER_REPORT['settings'],
        'cases': rows,
        'disclaimer': 'اختلاف التوكنات/الصوت لا يُعد دليلًا على نطق '
                      'أفضل — المراجعة البشرية المصرية ضرورية للحسم.',
    }
    out = os.path.join(OUT_DIR, 'audio_before_after_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\nReport: {out}')


if __name__ == '__main__':
    main()
