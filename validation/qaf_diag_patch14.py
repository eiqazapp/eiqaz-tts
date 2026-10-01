# -*- coding: utf-8 -*-
"""qaf_diag_patch14.py — تشخيص تقرير المستخدم الرابع (بعد PATCH 13).

الشكوى:
  «جميع الكلمات التي يجب نطق بها ق فصحي مثل يوم القيامة {ق} وغيرها لا يتم
   نطق القاف اصليه، حتى الكلمات التي كانت تعمل سابقا مثل القرآن {ق} أصبحت
   جميعها تنطق همزة»

يتتبع مسار الإنتاج الحرفي لكل كلمة قاف موسومة:
  raw → bind_markers → parse_qaf_markers → catt → _apply_qaf_text_layer
  → fix_qaf → EGY_TOKEN_MAP (التوكن النهائي الذي يسمعه النموذج)

الهدف: تحديد أي كلمة موسومة تنتهي بتوكن '<' (همزة) مقابل 'v' ([g])
مقابل خام عميق (زرع داخل التوزيع).
"""
import os
import sys

REPO = '/home/z/my-project/work/github_repo/eiqaz-tts/inference'
sys.path.insert(0, REPO)
os.chdir(REPO)

import infer  # noqa: E402
from tts_arabic.text import arabic_to_buckwalter, phonemes_to_tokens  # noqa: E402

CASES = [
    # (اسم الحالة، النص الخام)
    ('يوم القيامة (مسافة قبل الوسم)', 'يوم القيامة {ق}'),
    ('يوم القيامة (ملتصق)', 'يوم القيامة{ق}'),
    ('القرآن بالتعريف', 'القرآن{ق} كتاب عظيم'),
    ('قرآن مجردة', 'قرآن{ق} كتاب عظيم'),
    ('قصص', 'في القرآن{ق} قصص{ق} وحكم كثيرة'),
    ('نقرأه', 'وكل قرآن{ق} نقرأه{ق} يذكرنا'),
    ('قرآننا', 'لأن قرآننا{ق} مصدر هداية'),
    ('يقرأ', 'وكل من يقرأ{ق} القرآن{ق} يتعلم'),
    ('قرآناً تنوين نصب', 'أن يتدبر قرآنا{ق} كريما'),
    ('الجملة العاملة سابقا (قطعة/رقم/قال)',
     'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا'),
    ('قيامة مجردة', 'قيامة{ق}'),
    ('قطعة (كانت تعمل)', 'قطعة{ق}'),
    ('القطعة بالتعريف', 'القطعة{ق}'),
    ('الحقيقة (TRUST)', 'الحقيقة{ق}'),
    ('القيمة (corpus معرفة)', 'القيمة{ق}'),
    ('القصة (corpus معرفة)', 'القصة{ق}'),
]


def trace_case(name, raw):
    print('=' * 90)
    print(f'الحالة: {name}')
    print(f'النص الخام: {raw}')
    raw_b = infer.webapp_bind = None
    # مسار الويب: ربط الوسم المنفصل أولاً
    try:
        bound = __import__('webapp', fromlist=['bind_markers_to_words']) \
            .bind_markers_to_words(raw)
    except Exception:
        bound = raw
    print(f'بعد الربط : {bound}')

    clean, actions = infer.parse_qaf_markers(bound)
    print(f'بعد نزع الوسم: {clean}')
    print(f'الأفعال: {actions or "—"}')

    voc = ' '.join(infer.catt_vocalize(clean).split())
    print(f'بعد catt   : {voc}')

    standalone = len(voc.split()) <= 2
    new_voc, planted, native, un_deep = infer._apply_qaf_text_layer(
        voc, 'auto', 'egy', actions, standalone)
    print(f'بعد الزرع  : {new_voc}')
    for p in planted:
        print(f"  مزروع: {p['was']} ← {p['now']} (هيكل {p['skel']})")
    if native:
        print(f'  native  : {sorted(native)}')
    if un_deep:
        for u in un_deep:
            print(f"  نزع عمق{{ء}}: {u['was']} ← {u['now']}")

    # قرار التوكنات: ماذا يرى النموذج في موضع كل قاف؟
    print('  قرار التوكن لكل كلمة قاف:')
    for w_ar in new_voc.split():
        if 'ق' not in w_ar:
            continue
        buck = arabic_to_buckwalter(w_ar)
        after = infer.fix_qaf(buck, 'auto', 'egy', actions, native)
        toks = phonemes_to_tokens(
            __import__('tts_arabic.text', fromlist=['buckwalter_to_phonemes'])
            .buckwalter_to_phonemes(after))
        final_toks = [infer.EGY_TOKEN_MAP.get(t, t) for t in toks]
        # توكن موضع القاف: قبل/بعد الاستبدال
        q_idx = [i for i, t in enumerate(toks) if t == 'q']
        v_idx = [i for i, t in enumerate(final_toks) if t == 'v']
        if any(t == 'q' for t in toks):
            verdict = "q→'<' همزة (خام)"
        elif v_idx and 'q' not in after and 'j' in after:
            verdict = "q→j→'v' جيم [g]"
        elif any(t == 'k' for t in toks):
            verdict = "'k' كاف (محظور!)"
        else:
            verdict = '؟'
        # هل الزرع داخل التوزيع (corpus form للشكل السطحي)؟
        skel = ''.join(c for c in buck if c not in infer._QAF_DIAC)
        in_corpus = skel in infer.QAF_Q_CORPUS_FORMS
        print(f'    {w_ar:18s} buck={after:16s} {verdict}'
              f'{" [شكل corpus: داخل التوزيع → عميق]" if in_corpus else ""}')
    print()


for nm, txt in CASES:
    trace_case(nm, txt)
