# -*- coding: utf-8 -*-
"""qaf_diag_user_sentences.py — تشخيص حي لجملة القاف عبر مسار الإنتاج.

PATCH 11 (2026-10-02): أجاب عن «لماذا هذا ينطق بالهمزة دائمًا؟» — يعرض
لكل كلمة قاف: العلامة، الهيكل، قرار fix_qaf التوكني (أصيلة خام/جيم/كاف/
همزة خام {ء}). عدّل الجملتين أو أضف جملك تحت ثم شغّل من مجلد inference:

    python ../validation/qaf_diag_user_sentences.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.join(os.path.dirname(HERE), 'inference')
sys.path.insert(0, INF)                              # → inference/
os.chdir(INF)

import infer  # noqa: E402

SENTENCES = [
    ('جملة المستخدم (القرآن/قصص/نقرأه — PATCH 11)',
     'القرآن{ق} كتاب عظيم، وكل قرآن{ق} نقرأه{ق} يذكرنا بآيات القرآن، '
     'وفي القرآن{ق} قصص{ق} وحكم كثيرة'),
    ('جملة الأشكال الثلاثة (تعمل)',
     'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا'),
]


def diagnose(label, raw):
    print('=' * 72)
    print(label)
    print('=' * 72)
    clean, actions = infer.parse_qaf_markers(raw)
    print('1) العلامات:', actions or 'لا شيء')
    voc = ' '.join(infer.catt_vocalize(clean).split())
    print('2) بعد catt :', voc)
    standalone = len(voc.split()) <= 2
    new_voc, planted, native, un_deep = infer._apply_qaf_text_layer(
        voc, 'auto', 'egy', actions, standalone)
    print('3) بعد الزرع:', new_voc)
    print('   مزروع:', planted or '—')
    print('   native :', sorted(native) or '—')
    print('   نزع ء  :', un_deep or '—')

    from tts_arabic.text import arabic_to_buckwalter
    print('4) قرار fix_qaf لكل كلمة قاف:')
    for w_ar in new_voc.split():
        if 'ق' not in w_ar:
            continue
        buck = arabic_to_buckwalter(w_ar)
        skel = ''.join(c for c in buck if c not in infer._QAF_DIAC)
        after = infer.fix_qaf(buck, 'auto', 'egy', actions, native)
        act = infer._lookup_qaf_action(skel, actions)
        if 'q' in after:
            verdict = ('أصيلة خام \'<\' (مزروع/بيئة deep — PATCH 9/11)'
                       if act == 'q' else 'همزة خام (خارج كل قائمة)')
        elif 'j' in after:
            verdict = 'جيم [g]'
        elif 'k' in after:
            verdict = 'كاف [k] تقريب (لا شكل مدروس — PATCH 11)'
        else:
            verdict = '?'
        print(f'   {w_ar:16s} skel={skel:10s} act={act!s:5s} '
              f'=> {verdict}')
    print()


for label, sent in SENTENCES:
    diagnose(label, sent)
