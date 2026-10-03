# -*- coding: utf-8 -*-
"""qaf_corpus_scan.py — مسح كلمات القاف في corpus تدريب NileTTS (21,880 وحدة).

الأسئلة:
  1. ما عائلات القاف التي رآها النموذج في التدريب، وبأي تكرار؟
  2. عائلة قطع (قطعة/قطع/يقطع...) — كم مرة وفي أي سياقات؟
  3. قرآن/القرآن — كم مرة (التفسير المقترح لعدم تماثل نطقهما)؟
  4. أكثر كلمات القاف الابتدائية تكرارًا (مرشحات «السلوك الأصلي»).
"""
import json
import re
from collections import Counter

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,!?;:()"«»\u060C\u061F\u061B\u2026-—'

UNITS = json.load(open(
    '/home/z/my-project/work/github_repo/eiqaz-tts/tashkeel-pipeline/'
    'state/ph2_input_full.json', encoding='utf-8'))


def words_of(text):
    return [TASH.sub('', w).strip(PUNCT) for w in text.split()]


def skel(w):
    return TASH.sub('', w).strip(PUNCT)


# 1) كل الكلمات فيها ق + تكراراتها
qword_counter = Counter()
qword_units = {}
for u in UNITS:
    for w in words_of(u['text']):
        if 'ق' in w and len(w) >= 2:
            qword_counter[w] += 1
            qword_units.setdefault(w, []).append(u['id'])

print(f'وحدات: {len(UNITS)} | كلمات قاف فريدة: {len(qword_counter)} | '
      f'مجموع المواضع: {sum(qword_counter.values())}')

# 2) عائلات مطلوبة بعينها
FAMILIES = {
    'قطع': lambda w: 'قطع' in w or 'قطئ' in w,
    'قرآن': lambda w: 'قرآن' in w or 'قران' in w,
    'قسم': lambda w: 'قسم' in w,
    'رقم': lambda w: 'رقم' in w,
    'قيم': lambda w: 'قيم' in w or 'قيمة' in w,
    'قاس': lambda w: 'قاس' in w or 'قياس' in w or 'مقياس' in w,
    'قعد': lambda w: 'قعد' in w,
    'قلب': lambda w: 'قلب' in w,
    'قمر': lambda w: 'قمر' in w,
    'قرش': lambda w: 'قرش' in w,
    'قوة': lambda w: 'قوة' in w or 'قوى' in w,
}
print('\n== العائلات المستهدفة (تكرار | مثال سياق) ==')
for fam, pred in FAMILIES.items():
    hits = {w: n for w, n in qword_counter.items() if pred(w)}
    total = sum(hits.values())
    top = sorted(hits.items(), key=lambda kv: -kv[1])[:6]
    print(f'  {fam}: {total} موضعًا | {top}')

# 3) قرآن — كل المواضع بسياقها
print('\n== كل مواضع قرآن/القرآن بسياق كامل ==')
for u in UNITS:
    for w in words_of(u['text']):
        if 'قرآن' in w or 'قران' in w:
            print(f"  [{u['id']} | {u['split']}] {w!r} في: {u['text'][:90]}")

# 4) أعلى 40 كلمة قاف (خام — بق ال التعريف كما هي)
print('\n== أعلى 40 كلمة قاف في الـcorpus ==')
for w, n in qword_counter.most_common(40):
    print(f'  {w:<18} {n:>5}')

# 5) مواضع قطع بسياق (حتى 12)
print('\n== سياقات قطع (حتى 12) ==')
n_shown = 0
for u in UNITS:
    if n_shown >= 12:
        break
    for w in words_of(u['text']):
        if 'قطع' in w:
            print(f"  [{u['id']}] {w!r} في: {u['text'][:90]}")
            n_shown += 1
            break

# 6) حفظ
out = {
    'n_units': len(UNITS),
    'n_distinct_qwords': len(qword_counter),
    'n_q_positions': sum(qword_counter.values()),
    'top100': qword_counter.most_common(100),
    'families': {fam: {w: n for w, n in qword_counter.items() if pred(w)}
                 for fam, pred in FAMILIES.items()},
}
with open('/home/z/my-project/work/qaf_research/corpus_qword_scan.json',
          'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('\nحُفظ: work/qaf_research/corpus_qword_scan.json')
