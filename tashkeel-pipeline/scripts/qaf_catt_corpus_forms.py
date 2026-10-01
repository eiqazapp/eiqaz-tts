# -*- coding: utf-8 -*-
"""qaf_catt_corpus_forms.py — أي تشكيلات لعائلة قطع رآها النموذج في التدريب؟

فرضية: catt (مُشكِّل التدريب) أعطى في السياق أشكالًا مختلفة (كسرة+تنوين)
عن المنفرد (فتحة) — والنموذج رأى التوزيع الفعلي.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference')
sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')

import infer

UNITS = json.load(open(
    '/home/z/my-project/work/github_repo/eiqaz-tts/tashkeel-pipeline/'
    'state/ph2_input_full.json', encoding='utf-8'))

TASH = re.compile(r'[\u064B-\u0652]')


def qte_forms(sent):
    """استخراج أشكال عائلة قطع المشكولة من جملة مشكولة."""
    out = []
    for w in sent.split():
        sk = TASH.sub('', w).strip('.,!?؛،:')
        if 'قطع' in sk:
            out.append(w.strip('.,!?؛،:'))
    return out


n_sent = 0
forms = Counter()
print('== أشكال عائلة قطع كما شكّلها catt في جمل التدريب ==')
for u in UNITS:
    has_qte = any('قطع' in TASH.sub('', w).strip('.,!?؛،:')
                  for w in u['text'].split())
    if not has_qte:
        continue
    n_sent += 1
    if n_sent > 25:
        continue
    voc = infer.catt_vocalize(u['text'])
    fs = qte_forms(voc)
    for f in fs:
        forms[f] += 1
    if fs and n_sent <= 14:
        print(f"  [{u['id']}] {', '.join(fs)}  ← {u['text'][:60]}")

print(f'\nجمل فيها قطع: {n_sent} (فُحصت أول 25)')
print('توزيع الأشكال المشكولة:')
for f, n in forms.most_common(15):
    print(f'  {f:<16} {n}')

# مقارنة: المنفرد
print('\n== للمقارنة: قطعة منفردة ==')
print('  catt(قطعة) =', infer.catt_vocalize('قطعة'))
print('  catt(قطعة حلوى) =', infer.catt_vocalize('قطعة حلوى'))
print('  catt(عشرين قطعة حلوى) =',
      infer.catt_vocalize('عشرين قطعة حلوى'))
