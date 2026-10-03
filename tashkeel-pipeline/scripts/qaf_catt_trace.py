# -*- coding: utf-8 -*-
"""تتبع سلوك catt + G2P لكلمات المستخدم التجريبية — فرضيات قرآن/القرآن/قطعة.

فرضيات للتحقق:
  H1: catt يعيد إملاء/تشكيل «القرآن» بشكل يكسر مطابقة B (→ تفسير الهمزة)
  H2: catt يشكّل «قطعة» منفردة ≠ داخل جملة (→ تفسير اختلاف النطق)
  H3: خريطة قرار القاف لكل كلمة في درس المستخدم عبر الأوضاع الأربعة
"""
import sys

sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference')
sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')

import infer
from tts_arabic.text import arabic_to_buckwalter

def trace(word, note=''):
    catt = infer.catt_vocalize(word)
    buck = arabic_to_buckwalter(catt)
    print(f'--- {word!r} {note}')
    print(f'    catt : {catt}')
    print(f'    buck : {buck}')
    for mode in ('auto', 'qaf', 'hamza', 'g'):
        fixed = infer.fix_qaf(buck, mode, 'egy')
        tag = '=> معدّل' if fixed != buck else '=> كما هو (نموذج يقرر)'
        # أي حرف أصبحت القاف؟
        out_ch = set()
        for w_in, w_out in zip(buck.split(), fixed.split()):
            if 'q' in w_in:
                for ci, co in zip(w_in, w_out):
                    if ci == 'q':
                        out_ch.add(co)
        print(f'    [{mode:<5s}] {fixed}  {tag}  قاف→{sorted(out_ch) if out_ch else "?"}')

print('=' * 70)
print('H1: قرآن / القرآن / قراءات — هل يكسر catt مطابقة B؟')
print('=' * 70)
trace('قرآن')
trace('القرآن')
trace('القرآن الكريم')
trace('قرآن')
trace('في القرآن')

print()
print('=' * 70)
print('H2: قطعة منفردة vs في سياقات مختلفة')
print('=' * 70)
trace('قطعة')
trace('قطعة حلوى')
trace('عشرين قطعة حلوى')
trace('قطع')
trace('يقطع')
trace('قطعت')

print()
print('=' * 70)
print('كلمات المستخدم الأخرى المنفردة')
print('=' * 70)
for w in ['رقم', 'الرقم', 'قسمة', 'القسمة', 'قانون', 'القانون',
          'قيمة', 'القيمة', 'قياس', 'تقريب', 'قوة', 'القوة']:
    trace(w)
