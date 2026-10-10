#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mode_matrix.py — مصفوفة الأوضاع manual/egyptian/fusha/auto على نسخة الكود المحددة.

استخدام: python3 mode_matrix.py <INF_DIR> <OUT_JSON>

ثلاث جمل (عاري / جزئي التشكيل / مكتمل التشكيل) × أربعة أوضاع.
تُشغَّل على main worktree وعلى فرع الإصلاح ثم تُقارن الخليتان —
الانحدار المقبول: manual يجب ألا يتغير إطلاقًا؛ الفروق الأخرى تُوثَّق.
"""
import json
import os
import subprocess
import sys

INF_DIR = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
for p in (INF_DIR, os.path.join(INF_DIR, 'lib')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(INF_DIR)

import infer  # noqa: E402

SENTS = [
    ('bare', 'هو بيقول كده'),
    ('partial', 'بِيَكْتِب الدرس'),
    ('full', 'هُوَ بِيِقُولْ كِدَهْ'),
]
MODES = ['manual', 'egyptian', 'fusha', 'auto']


def main():
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                  cwd=INF_DIR, text=True).strip()
    print(f'INF={INF_DIR} sha={sha[:10]}')
    rows = {}
    for skey, text in SENTS:
        for mode in MODES:
            res = infer.prepare_text_rich(text, mode, 'egy')
            st = res.get('stages') or {}
            rows[f'{skey}|{mode}'] = {
                'input': text, 'mode': mode,
                'final_text': res['text'],
                'effective_mode': st.get('effective_mode'),
            }
            print(f"[{skey:7s}|{mode:8s}] {text} → {res['text']}")
    out = {'git_sha': sha, 'rows': rows}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
