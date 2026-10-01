# -*- coding: utf-8 -*-
"""build_corpus_forms_dict.py — توليد معجم QAF_Q_CORPUS_FORMS بمفاتيح باكوالتير.

من work/qaf_results/corpus_forms_expanded.json (أشكال corpus المُشكَّلة الحرفية)
→ dict {هيكل_باكوالتير: شكل_عربي_مُشكَّل} جاهز للتضمين في infer.py (PATCH 13).
تُستبعد الهياكل الموجودة أصلًا في QAF_Q_STUDY_FORMS (المعجم المدروس يفوز).
"""
import json
import re
import sys

sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')

from tts_arabic.text import arabic_to_buckwalter  # noqa: E402

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
QAF_DIAC = frozenset('auiFNK~o')

src = json.load(open('/home/z/my-project/work/qaf_results/'
                     'corpus_forms_expanded.json', encoding='utf-8'))
study = {
    '$qyq', 'AnqsAm', 'Hqwq', 'mnTq', 'mstqym', 'qSp', 'qTAE', 'qTAEAt',
    'qTE', 'qTEp', 'qTEtyn', 'qTb', 'qTbyn', 'qTr', 'qdrAt', 'qdrp', 'qlwb',
    'qsm', 'qsmp', 'qsmyn', 'qwY', 'qwp', 'qyAm', 'qyAs', 'qymp', 'tEqyd',
    'trqym', 'twqyt', 'yqys', 'qr|n', 'qrAn', 'qr|ny', 'qr|nyp', 'qrAny',
    'qSS', 'qr|nA',
}

out = {}
collisions = []
for ar_skel, info in src['expanded_deep'].items():
    bare = TASH.sub('', ar_skel)
    try:
        b = arabic_to_buckwalter(bare)
    except Exception:
        continue
    sk = ''.join(c for c in b if c not in QAF_DIAC)
    if sk.endswith('h') and len(sk) > 2:      # ه →ة (رسم مصري — مطابق _ar_skel)
        sk = sk[:-1] + 'p'
    if not sk or 'q' not in sk:
        continue
    if sk in study:
        collisions.append((sk, ar_skel))
        continue                     # المعجم المدروس يفوز
    out[sk] = info['form']

print(f'هياكل corpus جديدة (بعد استبعاد المدروس): {len(out)}')
print(f'تعارضات (المدروس يفوز): {len(collisions)}')

# توليد كود python جاهز للتضمين مرتبًا أبجديًا
lines = []
for sk in sorted(out):
    form = out[sk]
    lines.append(f"    '{sk}': '{form}',")
code = 'QAF_Q_CORPUS_FORMS = {\n' + '\n'.join(lines) + '\n}'

with open('/home/z/my-project/work/qaf_results/corpus_forms_dict_code.py',
          'w', encoding='utf-8') as f:
    f.write(code + '\n')
print('حُفظ: work/qaf_results/corpus_forms_dict_code.py')
print('\nعينة (أول 25):')
for sk in sorted(out)[:25]:
    print(f"  '{sk}': '{out[sk]}'")
