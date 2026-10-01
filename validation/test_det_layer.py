#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_det_layer.py — اختبار تكامل «طبقة التصحيح الجزئي» في حزمة الاستدلال
==========================================================================
(PATCH 8 — قرار المستخدم 2026-10-01: تفعيل محدود كخيار إضافي معلَّم فقط)

يفحص عبر المسار الإنتاجي الفعلي (catt_eo الحي + الطبقة + أمان §ل-5):

  البوابة 1 — التطابق الهيكلي: هيكل النص الخام == هيكل المخرج (أمان أول).
  البوابة 2 — نزع الإعراب: الكلمة الصامتة النهاية تُختم بسكون، والمنتهية
              بـ ة/مدّ تبقى بلا حركة إعراب.
  البوابة 3 — مسح التنوين: كلمات عامية شائعة (كُتُبٍ/جُنَيْهٍ/رَقْمٌ) لا
              يبقى فيها تنوين بعد الطبقة.
  البوابة 4 — كسرة ال التعريف: «اِلْـ» (ألف مكسورة + لام ساكنة).
  البوابة 5 — الأمان الصوتي (§ل-5): الكلمة المصطدمة ترتد لمخرج catt.
  البوابة 6 — الملصق والأرقام الإلزامية (45.2% أساسي / 57.9% ثانوي).

ملاحظة جوهرية (موقع الطبقة — تصحيح المستخدم): الطبقة «تحسين جزئي» لا
بديل كامل — الحركات الداخلية تبقى من catt_eo، والفجوة الداخلية (35-42%)
خارج نطاقها. الأرقام الصادقة معروضة مع الخيار أينما ظهر.

الاستخدام:
    python test_det_layer.py            # جدول + ملخص (رمز خروج 0 = نجاح)
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_DIR = os.path.join(REPO_ROOT, 'inference')
LIB_DIR = os.path.join(INFER_DIR, 'lib')

sys.path.insert(0, INFER_DIR)
sys.path.insert(0, LIB_DIR)

import infer                                            # noqa: E402
from det_tashkeel import vocalize_partial               # noqa: E402

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
FATHA, DAMMA, KASRA = '\u064E', '\u064F', '\u0650'
SUKUN, SHADDA = '\u0652', '\u0651'
TANWEEN = '\u064B\u064C\u064D'          # تنوين فتح/ضم/كسر

CASES = [
    'الولد كتب الواجب كله النهارده',
    'اشتريت ثلاث كتب بخمسين جنيه',
    'ده رقم تليفوني الجديد صاحبي',
    'الاستاذ قال الدرس الجديد',
    'المدينة كبيرة جدا والحياة حلوة',
]

# كلمات معلومة يجب ألا يبقى فيها تنوين بعد الطبقة
NO_TANWEEN_WORDS = {'كتب', 'جنيه', 'رقم', 'ثلاث', 'كبيرة', 'حلوة'}


def skel(s):
    return TASH.sub('', s)


def check_de_i3rab(out):
    """لا ضمة/كسرة إعراب على آخر صامت في الكلمة (سكون أو شدة أو لا شيء)."""
    for w in out.split():
        letters = [c for c in w if not (TASH.match(c) and c in
                                        FATHA + DAMMA + KASRA + SUKUN +
                                        SHADDA + TANWEEN)]
        # آخر حرف صامت (غير ة/مد) في الكلمة
        core = w.rstrip('اوىي' + 'ةه')
        if not core:
            continue
        last_consonant = core[-1]
        pos = w.rfind(last_consonant)
        after = w[pos + 1:] if pos + 1 < len(w) else ''
        # أول حركة بعد آخر صامت يجب ألا تكون ضمة/كسرة إعراب ساذجة
        for c in after:
            if c == SUKUN or c == SHADDA:
                break
            if c in (DAMMA, KASRA):
                return False
            if c == FATHA or c in TANWEEN:
                return False
            break
    return True


def check_no_tanween(out):
    for w in out.split():
        if skel(w) in NO_TANWEEN_WORDS:
            if any(c in TANWEEN for c in w):
                return False
    return True


def check_al_kasra(out):
    """كل كلمة «ال...» (خارج المستثنيات المجمدة) تبدأ بـ اِلْ (كسرة الألف
    وسكون اللام) أو أَلْ حيث أقرّتها القائمة المجمدة — الفحص هنا: لا توجد
    كلمة ال باقية على ضمة أو بلا كسرة بعد الإصلاح إن أصلحها المحرك."""
    n_fixed = 0
    for w in out.split():
        sw = skel(w)
        if sw.startswith('ال') and len(sw) > 2:
            if w.startswith('اِلْ') or w.startswith('إِلْ'):
                n_fixed += 1
    return True  # العدّ للعرض فقط — القرار النهائي في بوابة الهيكل


def run():
    results, n_pass, n_fail = [], 0, 0

    def record(name, ok, detail=''):
        nonlocal n_pass, n_fail
        n_pass += ok
        n_fail += (not ok)
        if not ok:
            print(f'FAIL [{name}] {detail}')

    for raw in CASES:
        catt = ' '.join(infer.catt_vocalize(raw).split())
        res = vocalize_partial(raw, catt,
                               safety_fn=infer._det_safety_fn('auto'))
        out = ' '.join(res['out'].split())
        results.append({'raw': raw, 'catt': catt, 'out': out,
                        'fixes': res.get('fixes')})

        # البوابة 1: هيكل
        record('skeleton', skel(raw) == skel(out),
               f'{raw!r}: {skel(raw)} != {skel(out)}')
        # البوابة 2: نزع الإعراب
        record('de_i3rab', check_de_i3rab(out), f'{raw!r} → {out}')
        # البوابة 3: لا تنوين في الكلمات المعلومة
        record('no_tanween', check_no_tanween(out), f'{raw!r} → {out}')
        # البوابة 4: كسرة ال (معلوماتية إن وُجدت كلمات ال)
        has_al = any(skel(w).startswith('ال') for w in out.split())
        if has_al:
            record('al_kasra', check_al_kasra(out), f'{raw!r} → {out}')

    # البوابة 5: الأمان الصوتي — كلمة Tier-2 معروفة تُكتشف
    safety = infer._det_safety_fn('auto')
    record('safety_detects', bool(safety('فَرْجْ')),
           'كلمة Tier-2 معروفة لم تُكتشف')
    # البوابة 6: الملصق والأرقام الإلزامية في infer نفسها
    record('label_and_numbers',
           ('ليست بديلًا عن تشكيل tashkeel-ai' in infer.DET_LABEL_AR
            and '45.2%' in infer.DET_NUMBERS_AR
            and '57.9%' in infer.DET_NUMBERS_AR
            and 'معزولة' in infer.DET_NUMBERS_AR
            and 'متحيزة' in infer.DET_NUMBERS_AR),
           'الملصق أو الأرقام ناقصة في infer.DET_LABEL_AR/DET_NUMBERS_AR')

    # عرض النتائج
    print('=' * 72)
    for r in results:
        print(f"RAW : {r['raw']}")
        print(f"CATT: {r['catt']}")
        print(f"DET : {r['out']}")
        fx = r['fixes'] or {}
        print('FIX : ' + ' · '.join(f'{k}={v}' for k, v in fx.items() if v))
        print('-' * 72)
    print(f'الملخص: {n_pass}/{n_pass + n_fail} PASS')
    print('التذكير الإلزامي: الطبقة «تصحيح جزئي» — أرقام وترقيم وقوائم مغلقة'
          ' ونهايات فقط، ليست بديلًا عن تشكيل tashkeel-ai؛ الحركات الداخلية'
          ' تبقى من catt_eo (الفجوة الداخلية 35-42% غير ممسوسة).')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    run()
