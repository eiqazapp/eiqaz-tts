# -*- coding: utf-8 -*-
"""qaf_diag_patch12.py — تشخيص حي لجملة المستخدم الجديدة (PATCH 12) عبر مسار الإنتاج.

يجيب حرفياً عن تقرير المستخدم:
«كلمة القرآن هنا تُنطق في الأغلب بالهمزة ولكن في آخر الجملة تنطق ق فصحى —
اكتشف السبب ليس لكلمة القرآن فقط ولكن بشكل عام»

يحاكي مسار الويب كاملاً: split_into_chunks → prepare_text_rich لكل مقطع →
قرار fix_qaf لكل كلمة قاف. يكشف:
  1) خلل اللصق: «قرآن {ق}نقرأه{ق}» (مسافة قبل الوسم، بلا مسافة بعده)
  2) زرع المعرفة OOD (الْقُرْآن — صفر مواضع في corpus §3-أ) → إمرار خام → همزة
  3) المجردة قُرْآنٍ (شكل corpus الحرفي) → خام → قاف عميقة (آخر الجملة!)
"""
import os
import sys

REPO = '/home/z/my-project/work/github_repo/eiqaz-tts/inference'
sys.path.insert(0, REPO)
os.chdir(REPO)

import infer  # noqa: E402

# جملة المستخدم الجديدة حرفيًا (بالمسافات كما أرسلها)
USER = ('القرآن {ق} كتاب عظيم، وكل قرآن {ق}نقرأه{ق} يذكرنا بآيات القرآن{ق}، '
        'وفي القرآن{ق} قصص{ق} وحكم كثيرة، ونحب أن نقرأ القرآن{ق} كل يوم، '
        'لأن قرآننا{ق} مصدر هداية، وكل من يقرأ{ق} القرآن{ق} يتعلم من آيات '
        'القرآن{ق}، ويعود إلى القرآن{ق} كلما أراد أن يتدبر قرآنا{ق} كريما.')


def token_verdict(w_buck, actions, native):
    """قرار fix_qaf التوكني لكلمة واحدة + تصنيفه الصوتي المتوقع."""
    after = infer.fix_qaf(w_buck, 'auto', 'egy', actions, native)
    skel = ''.join(c for c in w_buck if c not in infer._QAF_DIAC)
    act = infer._lookup_qaf_action(skel, actions)
    if 'q' in after:
        # خام: النموذج يقرر حسب تعلمه من الكلمة — شكل corpus الحرفي → قاف
        # عميقة؛ شكل OOD (التعريف) → التحقيق الافتراضي = همزة
        note = 'خام ← قاف عميقة إن كان الشكل مدرَّبًا / همزة إن كان OOD'
    elif 'k' in after:
        note = 'تقريب [k] — كاف ثقيلة (لا همزة أبدًا)'
    elif 'j' in after:
        note = '[g] جيم'
    else:
        note = '؟'
    return act, after, note


def diagnose_chunk(idx, chunk_raw):
    print('=' * 78)
    print(f'المقطع {idx}: {chunk_raw}')
    print('=' * 78)
    clean, actions = infer.parse_qaf_markers(chunk_raw)
    print('1) بعد نزع العلامات:', clean)
    print('   العلامات:', {k: v for k, v in actions.items()} or 'لا شيء')
    voc = ' '.join(infer.catt_vocalize(clean).split())
    print('2) بعد catt     :', voc)
    standalone = len(voc.split()) <= 2
    new_voc, planted, native, un_deep = infer._apply_qaf_text_layer(
        voc, 'auto', 'egy', actions, standalone)
    print('3) بعد الزرع    :', new_voc)
    for p in planted:
        print(f"   مزروع: {p['was']} ← {p['now']}  (هيكل {p['skel']})")
    print('   native :', sorted(native) or '—')
    from tts_arabic.text import arabic_to_buckwalter
    print('4) قرار fix_qaf لكل كلمة قاف:')
    for w_ar in new_voc.split():
        if 'ق' not in w_ar:
            continue
        buck = arabic_to_buckwalter(w_ar)
        act, after, note = token_verdict(buck, actions, native)
        print(f'   {w_ar:16s} act={act!s:5s} after={after:18s} => {note}')
    print()


print('### محاكاة مسار الويب: split_into_chunks أولاً ثم prepare لكل مقطع')
print()
tok_fns = infer.get_tokenizer('auto')
chunks = [USER]  # بدون تقسيم أولاً — الجملة الواحدة
# محاكاة التقسيم الفعلي (المستخدم قد يفعّل التقسيم)
try:
    from webapp import split_into_chunks
    chunks_split = split_into_chunks(USER, 'egy', tok_fns)
    if chunks_split:
        chunks = chunks_split
except Exception as e:  # noqa: BLE001
    print('(تعذر استيراد webapp — تشخيص بلا تقسيم:', e, ')')

print(f'عدد المقاطع بعد التقسيم: {len(chunks)}')
for i, c in enumerate(chunks, 1):
    diagnose_chunk(i, c)

# ── اختبار خلل اللصق منعزلًا ────────────────────────────────────────────────
print('=' * 78)
print('اختبار خلل اللصق منعزلًا (مسافة قبل الوسم + بلا مسافة بعده):')
print('=' * 78)
GLUE = 'وكل قرآن {ق}نقرأه{ق} يذكرنا'
clean, actions = infer.parse_qaf_markers(GLUE)
print('النص الأصلي :', GLUE)
print('بعد نزع الوسم:', clean)
words = clean.split()
bad = [w for w in words if len(w) > 10 and 'ق' in w]
print('كلمات ملتصقة مشتبه بها:', bad or 'لا شيء ✓')
