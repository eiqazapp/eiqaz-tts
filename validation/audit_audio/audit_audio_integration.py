#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_audio_integration.py — المرحلة السابعة: المسار الكامل حتى الصوت.

أمثلة قبل/بعد متطابقة عبر نسخة الإنتاج الحالية (فرع الإصلاح، الأوزان
نفسها — لا تغيير في checkpoint):

  النص → التطبيع → التشكيل → الدمج → G2P/tokens → الصوت (WAV)

لكل حالة: حفظ كل المراحل + التوكنات + المدة + إعدادات الاستدلال.
التمييز الصريح: اختلاف التوكنات لا يعني نطقًا أفضل — التقييم الصوتي
الآلي لا يغني عن مراجعة بشرية مصرية.
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

# أمثلة قبل/بعد متطابقة: نفس النص الخام يمر عبر المسارين
CASES = [
    # (id, نص خام, وضع, لهجة) — قبل = main سلوكياً (catt+det بلا دمج
    # ولا قائمة v1.3 ولا هو المصححة) لا يمكن تشغيله هنا؛ المقارنة
    # الصوتية تُجرى على مخرجات الفرع الحالي مع تسجيل توكنات قبل/بعد
    # على مستوى المراحل النصية (after_catt يمثل "قبل" للقواعد).
    ('A1', 'هو بيقول كده', 'egyptian', 'egy'),
    ('A2', 'بِيَكْتِب الدرس', 'auto', 'egy'),
    ('A3', 'دلوقتي هنبدأ الدرس يا عمر', 'auto', 'egy'),
    ('A4', 'مش عارف أعمل إيه', 'auto', 'egy'),
    ('A5', 'هو هنا', 'auto', 'egy'),
]

SETTINGS = {
    'checkpoint': 'states_cont_180516.pth (unmodified)',
    'speaker': 0,
    'pace': 1.0,
    'emotion': 0,
    'denoise': 0.005,
    'peak_normalize': True,
    'sample_rate': 22050,
    'note': 'الأوزان والإعدادات نفسها في كل الحالات — الفرق الوحيد نص '
            'الإدخال ومسار التشكيل (فرع الإصلاح)',
}


def main():
    model, it = infer.load_model(CKPT)
    print(f'model loaded (iter={it})')
    _, toks_egy, _ = infer.get_tokenizer()
    rows = []
    for cid, text, mode, dialect in CASES:
        res = infer.prepare_text_rich(text, mode, dialect)
        st = res['stages']
        wav = os.path.join(OUT_DIR, f'{cid}.wav')
        n_ids, n_secs = infer.synthesize(
            model, res['text'], dialect, speaker=0, pace=1.0,
            out_path=wav, denoise=0.005, peak_normalize=True)
        toks = list(toks_egy(res['text']))
        # توكنات مرحلة catt الخام (قبل القواعد والدمج) للمقارنة
        toks_catt = list(toks_egy(st.get('after_catt') or res['text']))
        row = {
            'id': cid, 'input': text, 'mode': mode, 'dialect': dialect,
            'effective_mode': st.get('effective_mode'),
            'normalized': st.get('normalized'),
            'after_catt': st.get('after_catt'),
            'after_det': st.get('after_det'),
            'after_merge': st.get('after_merge'),
            'final_text': res['text'],
            'n_tokens': n_ids, 'tokens': toks,
            'tokens_catt_stage': toks_catt,
            'tokens_differ_vs_catt_stage': toks != toks_catt,
            'duration_sec': round(n_secs, 3),
            'wav': wav,
        }
        rows.append(row)
        print(f"[{cid}] {text} → {res['text']}")
        print(f"    tokens={n_ids} dur={n_secs:.2f}s "
              f"diff_vs_catt={toks != toks_catt}")

    report = {
        'settings': SETTINGS,
        'cases': rows,
        'disclaimer': 'الفرق في التوكنات/الصوت لا يُعد دليلًا على جودة '
                      'نطق أفضل — التقييم النهائي يحتاج مراجعة بشرية '
                      'مصرية مؤهلة',
    }
    out = os.path.join(OUT_DIR, 'audio_integration_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\nReport: {out}')


if __name__ == '__main__':
    main()
