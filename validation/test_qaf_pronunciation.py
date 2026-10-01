#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_qaf_pronunciation.py — اختبار ظاهرة فقدان القاف في NileTTS/Eiqaz TTS
==========================================================================

يشخّص ويراقب (regression) ظاهرة: «كلمات القاف التي يُتوقع فيها نطق قاف
فموي ([g]/[q]) تُنطق همزة/ألف».

== السبب الجذري المُثبَت (انظر تقرير التحقيق) ==
خريطة التوكنز المصرية EGY_TOKEN_MAP = {'j':'v','q':'<','^':'t','*':'d'}
تُحوِّل توكن القاف 'q' إلى توكن الهمزة '<' **بلا أي شرط** في كل من:
  - التدريب:  training/kaggle-kernels/niletts-4h-train.py  (دالة toks_egy
              في _worker_process — كل 19,721 مقطع تدريب)
  - الاستدلال: inference/infer.py (دالة synthesize عند dialect='egy')
وبذلك لا يصل النموذج توكن 'q' إطلاقًا (0 توكن من 1,590,209 في corpus
التدريب كاملًا)، وتنهار هوية القاف عند هذه المرحلة تحديدًا.

== الإصلاح المطبق — PATCH 6 (inference/infer.py) ==
معجم QAF_G_SKELETONS (هياكل باكوالتير بلا حركات) + دالة fix_qaf_g:
الكلمات التي قافها [g] فقط تُستبدل فيها ق→ج قبل الترميز، فتصل النموذج
توكن 'v' — وهو التوكن الذي تعلّمه النموذج [g] من كل جيمات corpus
التدريب المصري (ج المصرية = [g]). باقي الكلمات تظل على q→'<' (الصحيح
للقاف القاهرية [ʔ]). لا مساس بالنموذج ولا بالأوزان ولا باللهجة الفصحى.

== PATCH 6b (مراجعة المستخدم 2026-10-01: النطق الثابت/المتغير) ==
تدقيق كامل لكل مواضع corpus (work/qaf_review/qaf_audit_report.md):
  1. الفئة B فقط تبقى في المعجم — نطق ثابت [g] في كل السياقات.
  2. الفئة C (حقيقة/دقيق/قراءة/قرأ وعائلاتها) حُذفت من المعجم كاملة
     وعادت للافتراضي [ʔ] — حفاظًا على السلوك المتعلّم فعليًا من التدريب
     (874 سماعًا لعائلة حقيقة = أعلى عائلة قاف رسمية) بدل تخمين نصي.
  3. حارس مستقل للصيغة التقدمية العامية «بقرأ» (وأخواتها بسابقة الباء
     على قرأ): بقرأ/بتقرأ/بيقرأ/بيتقرأ = [ʔ] دائمًا بغض النظر عن أي
     معجم — إصلاح خطأ منطق نزع السوابق التلقائي.

== PATCH 7 (2026-10-01: أوضاع نطق القاف + الفئة Q فصحى المصطلحات) ==
وضع نصي/واجهة لكل توليد (--qaf / خيار الواجهة): auto (الافتراضي —
السلوك المعتمد الساري دون أي تغيير) | qaf (فصحى المصطلحات التعليمية:
قائمة Q مغلقة 81 هيكلًا → قاف فصحى تقريبية [k] — ليست كل كلمة) |
hamza (فرض الهمزة شاملًا) | g (فرض الجيم شاملًا). حارس بقرأ يعمل في
كل الأوضاع. مسار الفصحى msa أُصلح أيضًا (توكن q غير المدرّب 0/1.59M
كان يُمرَّر خامًا — الآن قاف فصحى تقريبية [k] لكل الكلمات).

== أساس التصنيف (ليس تخمينًا من شكل التوكن) ==
فئة النطق المستهدف لكل كلمة مستندة إلى:
  1. اللهجة القاهرية القياسية: الافتراض ق→[ʔ] (همزة)، مع استثناءات
     معجمية تنطق [g] (دخائل مثل قانون/قرش/قيراط + كلمات مثل رقم —
     قِيست صوتيًا [g] في تسجيلات NileTTS نفسها).
  2. سجل "العامية المثقفة"/الديني: كلمات مثل القرآن/المقام يُتوقع
     فيها قاف فموية [g]/[q].
  3. قياس صوتي على تسجيلات التدريب الأصلية (انظر التقرير): رقم=[g]
     (إغلاق 40ms + انفجار)، قانون=[g] بحركة خلفية، دلوقتي=[ʔ] (لا انفجار).
  4. إحصاء corpus التدريب: القرآن=0 تكرار، مقام≈2، رقم=79، بينما
     كلمات [ʔ] العامية (بقى 1134، قوي 686، دلوقتي 274...) تهيمن —
     لذا تعلّم توكن '<' نطق [ʔ] الغالب رغم أمثلة [g] القليلة.

== الحكم ==
لكل حالة، PASS تعني: وصل النموذج التوكن المطابق للنطق المستهدف:
  الفئة A (هدف [ʔ]): يجب أن تصبح القاف '<' — تعمل قبل الإصلاح وبعده.
  الفئة B (هدف [g]): يجب أن تصبح القاف 'v' (التوكن الذي تعلّمه النموذج
                     [g] من جيمات corpus التدريب) — تعمل بعد PATCH 6.
  الفئة C (نطق متغير — PATCH 6b): عادت للافتراضي القاهري '<' [ʔ]
                     بقرار مراجعة المستخدم 2026-10-01، وتشمل حارس
                     «بقرأ» التقدمية.

== PATCH 9 (2026-10-01: الأشكال المدروسة + علامات النص) ==
زرع الشكل المطابق للتدريب يستدعي النطق المتعلم العميق (قطعة/قِطْعَةً)
+ علامات {ق}/{ء}/{ج} بعد الكلمة تخلط الأشكال الثلاثة في جملة واحدة.

== PATCH 10 (2026-10-02: ترقية الطبقات بنتائج القياس run1) ==
قياس آلي على جهاز المستخدم (speaker 0): قصة/قطب/قطبين قاست q? →
verified؛ قطر/يقيس قاست ء عند الزرع → tier2؛ صيغة قطر → قُطْرًا
(الشكل العميق المقيس). معايرة حاسمة: قطعة قاست g وأذن المستخدم
أكدتها أصيلة — المصنّف numpy لا يقيس مكان النطق، فقياس g على
المزروع لا ينزل من auto؛ فقط ء (ضرر مؤكد) ينزل و q? يرقّي.

== الاستخدام ==
    python test_qaf_pronunciation.py            # جدول + ملخص
    python test_qaf_pronunciation.py --json out.json
    python test_qaf_pronunciation.py --quiet    # خروج صامت: 0=نجاح كامل

رمز الخروج: 0 إذا كانت كل الحالات PASS، 1 إذا وُجد FAIL — صالح كبوابة
regression دائمة بعد PATCH 6 (وأي تعديل مستقبلي على الترميز).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_DIR = os.path.join(REPO_ROOT, 'inference')
LIB_DIR = os.path.join(INFER_DIR, 'lib')

sys.path.insert(0, INFER_DIR)
sys.path.insert(0, LIB_DIR)

# ---------------------------------------------------------------- test set --
# (input, class, rationale)
# class: A = هدف [ʔ] قاهري | B = هدف [g] | C = متغير → افتراضي [ʔ] (PATCH 6b)
TEST_CASES = [
    # -- الحالات المؤكدة من التقرير --
    ('القرآن', 'B', 'سجل ديني/تعليمي — 0 تكرار في corpus التدريب'),
    ('رقم', 'B', 'قياس صوتي: [g] في تسجيلات التدريب (79 تكرارًا)'),
    ('المقام', 'B', 'سجل ديني — ~0 تكرار فعلي في corpus التدريب'),
    # -- قاف جيمية معجمية --
    ('قانون', 'B', 'قياس صوتي: وقفة فموية بحركة خلفية في التسجيلات'),
    ('القانون', 'B', 'ال+ قاف جيمية'),
    ('قرش', 'B', 'دخيلة عثمانية [girš]'),
    ('قيراط', 'B', 'دخيلة يونانية'),
    ('قنطار', 'B', 'دخيلة'),
    ('قرآن', 'B', 'سجل ديني (تكراران فقط)'),
    # -- امتدادات عائلية للمعجم (PATCH 6) --
    ('أرقام', 'B', 'جمع رقم (قياس عائلة رقم=[g] في التسجيلات)'),
    ('قوانين', 'B', 'جمع قانون'),
    ('قرآني', 'B', 'نسبة قرآن'),
    ('مقامات', 'B', 'جمع مقام'),
    ('الرقم', 'B', 'ال+ رقم'),
    ('برقم', 'B', 'بـ+ رقم (سابقة اتصال)'),
    ('ترقيم', 'B', 'اشتقاق رقم'),
    ('بالقرآن', 'B', 'بـ+ال+ قرآن — الباء الجارة تُنزع، القرآن تبقى [g]'),
    # -- الفئة C بعد PATCH 6b: نطق متغير → عادت للافتراضي [ʔ] --
    ('قراءة', 'C', 'قرار المراجعة 2026-10-01: عادت للافتراضي [ʔ]'),
    ('القراءة', 'C', 'ال+ قراءة — [ʔ]'),
    ('قراءات', 'C', 'جمع قراءة — [ʔ]'),
    ('حقيقة', 'C', '874 سماعًا في التدريب — سلوك متعلّم [ʔ]'),
    ('الحقيقة', 'C', 'ال+ حقيقة — [ʔ]'),
    ('حقائق', 'C', 'جمع حقيقة — [ʔ]'),
    ('دقيقة', 'C', '251 موضعًا (81% عامي) — [ʔ]'),
    ('دقيق', 'C', 'عائلة دقيقة — [ʔ]'),
    ('دقائق', 'C', 'جمع دقيقة — [ʔ]'),
    ('قرأ', 'C', 'الصيغة المجردة — [ʔ] افتراضي (حارس بقرأ أدناه مستقل عنها)'),
    # -- حارس «بقرأ» التقدمية (PATCH 6b — قاعدة مستقلة) --
    ('بقرأ', 'C', 'التقدمية العامية — [ʔ] دائمًا بغض النظر عن أي معجم'),
    ('بتقرأ', 'C', 'بـ+ت قراءة مخاطبة — [ʔ]'),
    ('بيقرأ', 'C', 'بـ+ي تقدمية — [ʔ]'),
    ('بيتقرأ', 'C', 'بـ+يت تقدمية — [ʔ]'),
    ('بقرأ الكتاب', 'C', 'التقدمية داخل جملة — [ʔ]'),
    ('وبقرأ', 'C', 'و+بقرأ — الحارس يعمل بعد و/ف أيضًا — [ʔ]'),
    # -- الفئة A: الافتراض القاهري [ʔ] (يجب أن تظل تعمل) --
    ('قمر', 'A', 'بداية كلمة'), ('قلم', 'A', 'بداية كلمة'),
    ('قلب', 'A', 'بداية كلمة (47)'), ('قطة', 'A', 'بداية كلمة'),
    ('قريب', 'A', 'بداية كلمة (54)'), ('قديم', 'A', 'بداية كلمة (28)'),
    ('قارئ', 'A', 'بداية كلمة'), ('قواعد', 'A', 'بداية كلمة (33)'),
    ('قوة', 'A', 'بداية كلمة + شدة'), ('قبل', 'A', '264 في التدريب'),
    ('مقال', 'A', 'وسط (7)'), ('مقارنة', 'A', 'وسط (55)'),
    ('مقدار', 'A', 'وسط (0)'), ('مقدمة', 'A', 'وسط (1) + شدة'),
    ('مقعد', 'A', 'وسط (0)'), ('مقبول', 'A', 'وسط (9)'),
    ('طريقة', 'A', 'وسط (161)'), ('طريق', 'A', 'وسط (103)'),
    ('فريق', 'A', 'وسط (88)'), ('صديق', 'A', 'وسط (8)'),
    ('رفيق', 'A', 'وسط (3)'), ('طاقة', 'A', 'وسط (47)'),
    ('ورقة', 'A', 'وسط (31)'), ('حديقة', 'A', 'وسط (1)'),
    ('شقة', 'A', 'وسط — عامية (217)'), ('نقطة', 'A', 'وسط — عامية (477)'),
    ('بقى', 'A', 'عامية (1134)'), ('قوي', 'A', 'عامية (686)'),
    ('تلاقي', 'A', 'عامية (124)'), ('وقت', 'A', 'نهاية (274)'),
    ('فوق', 'A', 'نهاية (65)'), ('يقول', 'A', 'نهاية (61)'),
    ('يقرأ', 'A', 'نهاية (7) — عامية الاستعمال: [ʔ]'),
    ('حق', 'A', 'نهاية — عامية'),
    ('صدق', 'A', 'نهاية'), ('أعمق', 'A', 'نهاية (135)'),
    ('القيمة', 'A', 'ال+وسط'), ('المنطقة', 'A', 'ال+وسط'),
    ('القاهرة', 'A', 'ال+اسم علم (30) — قاهري [ʔ]'),
    ('الحقّ', 'A', 'قاف مشدودة نهائية'), ('حقّقنا', 'A', 'قاف مشدودة وسطية'),
    ('قول', 'A', 'قاف+مد واوي'), ('قيل', 'A', 'قاف+مد يائي'),
    ('قال', 'A', 'قاف+مد ألفي'),
    # -- جمل سياقية (نطق الكلمة داخل سياق) --
    ('هذا رقم كبير', 'B', 'رقم في جملة'),
    ('رقم التليفون خمسة', 'B', 'رقم بداية جملة'),
    ('القرآن كتاب عظيم', 'B', 'القرآن وحدها صاحبة قاف [g] في الجملة'),
    ('هذا هو المقام', 'B', 'المقام في جملة'),
    ('القانون يحكم الدولة', 'B', 'القانون فاعل'),
    ('قرأت هذا الكتاب', 'C', 'قرأت بلا سابقة باء — عادت للافتراضي [ʔ]'),
    ('دي حقيقة مهمة', 'C', 'حقيقة عامية في جملة — [ʔ]'),
    ('هذا قمر', 'A', 'قمر في جملة'),
    ('هذا قلم', 'A', 'قلم في جملة'),
    ('دي طريقة كويسة', 'A', 'طريقة عامية في جملة'),
]

EXPECTED = {'A': '<', 'B': 'v', 'C': '<'}
CLASS_DESC = {
    'A': '[ʔ] قاهري افتراضي — الخريطة الأصلية صحيحة لهذه الفئة',
    'B': '[g] قاف جيمية ثابتة (فئة B المعتمدة) — PATCH 6 يحوّلها إلى توكن v',
    'C': 'نطق متغير → الافتراضي [ʔ] (قرار مراجعة المستخدم 2026-10-01 — '
         'PATCH 6b) — يشمل حارس «بقرأ» التقدمية',
}

# ------------------------------------------------------------------ PATCH 7 --
# حالات أوضاع القاف: (المدخل، الوضع، التوكن المتوقع، الملاحظة)
# المتوقع: 'v' = جيم [g] | 'k' = قاف فصحى تقريبية | '<' = همزة [ʔ]
MODE_CASES = [
    # -- الوضع الافتراضي auto = السلوك المعتمد (بلا أي تغيير) --
    ('رقم', 'auto', 'v', 'B سارية في auto — قرار معتمد دون تغيير'),
    ('قانون', 'auto', 'v', 'B سارية في auto'),
    ('القرآن', 'auto', 'v', 'B سارية في auto (قرار مُعاد تأكيده 2026-10-01)'),
    ('قيمة', 'auto', '<', 'خارج B — همزة في auto'),
    ('قسمة', 'auto', '<', 'خارج B — همزة في auto'),
    # -- وضع qaf: أمثلة المستخدم + قائمة Q --
    ('رقم', 'qaf', 'k', 'مثال المستخدم — Q تفوز على B في وضع الفصحى'),
    ('أرقام', 'qaf', 'k', 'مثال المستخدم'),
    ('قسمة', 'qaf', 'k', 'مثال المستخدم'),
    ('يقسم', 'qaf', 'k', 'مثال المستخدم'),
    ('تقريب', 'qaf', 'k', 'مثال المستخدم'),
    ('تقريبًا', 'qaf', 'k', 'ظرف شائع في الشرح'),
    ('قياس', 'qaf', 'k', 'منهج أساسي'),
    ('مقياس', 'qaf', 'k', 'مقياس الرسم — جغرافيا/هندسة'),
    ('قيمة', 'qaf', 'k', 'القيمة المكانية'),
    ('قاعدة', 'qaf', 'k', 'قواعد الحساب'),
    ('قواعد', 'qaf', 'k', 'قواعد اللغة'),
    ('قانون', 'qaf', 'k', 'تعارض Q/B — Q تفوز في وضع الفصحى فقط'),
    ('قطر', 'qaf', 'k', 'قطر الدائرة (والدولة — نفس الهيكل)'),
    ('المقام', 'qaf', 'k', 'مقام الكسر — تعارض مع «في المقام الأول» B: auto=g / qaf=k'),
    ('مستقيم', 'qaf', 'k', 'الخط المستقيم'),
    ('قيمة العدد تسعة هي تسعة', 'qaf', 'k', 'جملة رياضيات — القيمة كلمة القاف الوحيدة'),
    ('المطلقة', 'qaf', '<', 'خارج قائمة Q — تبقى همزة (كلمات بعينها فقط)'),
    ('قوة', 'qaf', 'k', 'فيزياء — القوة'),
    ('طاقة', 'qaf', 'k', 'فيزياء — الطاقة'),
    ('حقوق', 'qaf', 'k', 'تربية مدنية'),
    ('القسمة', 'qaf', 'k', 'ال+ قسمة'),
    ('ويقسم', 'qaf', 'k', 'و+ يقسم'),
    # -- وضع qaf: ليست كل كلمة قاف! --
    ('قمر', 'qaf', '<', 'خارج القائمتين — تبقى همزة (كلمات بعينها فقط)'),
    ('قال', 'qaf', '<', 'خارج القائمتين — همزة'),
    ('حقيقة', 'qaf', '<', 'C تبقى همزة حتى في وضع الفصحى'),
    ('قراءة', 'qaf', '<', 'C تبقى همزة'),
    ('القرآن', 'qaf', 'v', 'يبقى على قرار B [g] — لا قاف فصحى للقرآن في الوضع التعليمي'),
    ('قرش', 'qaf', 'v', 'B غير المتقاطعة تبقى [g] في وضع الفصحى'),
    ('قيراط', 'qaf', 'v', 'B غير المتقاطعة'),
    # -- فرض الهمزة شاملًا --
    ('رقم', 'hamza', '<', 'فرض الهمزة يتجاوز B'),
    ('القرآن', 'hamza', '<', 'فرض الهمزة يتجاوز B'),
    ('قانون', 'hamza', '<', 'فرض الهمزة يتجاوز B'),
    # -- فرض الجيم شاملًا --
    ('قمر', 'g', 'v', 'فرض الجيم شامل'),
    ('قال', 'g', 'v', 'فرض الجيم شامل'),
    ('حقيقة', 'g', 'v', 'فرض الجيم يتجاوز C'),
    # -- حارس بقرأ في كل الأوضاع (PATCH 6b مستقل) --
    ('بقرأ', 'auto', '<', 'الحارس في auto'),
    ('بقرأ', 'qaf', '<', 'الحارس في وضع الفصحى'),
    ('بقرأ', 'g', '<', 'الحارس يمنع فرض الجيم عن بقرأ (خطأ صوتي ثابت)'),
    ('بيقرأ', 'g', '<', 'الحارس بعد سابقة الباء'),
]

# حالات مسار الفصحى msa (PATCH 7 — إصلاح التوكن غير المدرّب):
# (المدخل، الوضع، القاف المتوقعة في باكوالتير بعد fix_qaf)
MSA_CASES = [
    ('القرآن كتاب عظيم', 'auto', 'k', 'msa: كل قاف فصحى تقريبية — كان q خام'),
    ('قال المعلم', 'auto', 'k', 'msa auto: قاف فصحى'),
    ('رقم خمسة', 'hamza', '<', 'msa فرض همزة'),
    ('قال المعلم', 'g', 'j', 'msa فرض جيم'),
]


# ---------------------------------------------------------------- pipeline --
def load_pipeline():
    """يحمّل مسار الترميز الإنتاجي من inference/ نفسه (المصدر الوحيد للحقيقة)."""
    try:
        import infer  # noqa: F401  (من inference/)
        from tts_arabic.text import (
            arabic_to_buckwalter, buckwalter_to_phonemes,
            phonemes_to_tokens, tokens_to_ids)
    except ImportError as e:
        raise SystemExit(
            f'[خطأ] تعذر استيراد مسار الترميز من inference/: {e}\n'
            f'شغّل السكربت من داخل المستودع: python validation/test_qaf_pronunciation.py')
    import importlib
    infer = importlib.import_module('infer')
    egy_map = infer.EGY_TOKEN_MAP
    return (infer, arabic_to_buckwalter, buckwalter_to_phonemes,
            phonemes_to_tokens, tokens_to_ids, egy_map)


def trace(raw, infer, a2b, b2p, p2t, ids_of, egy_map, vocalize='auto',
          qaf_mode='auto'):
    """يتبع النص عبر المسار الإنتاجي: تنظيف → تشكيل catt → ترميز → معرفات.

    الترميز المصري يُستخرج من infer.toks_egy نفسها (المكان الفعلي لـPATCH 6
    + أوضاع PATCH 7 — تُبنى بـ get_tokenizer(qaf_mode))، والتوكنز الفصحى
    من infer.toks_ms لتحديد مواضع q المرجعية (المرجع الخام بلا إصلاح)."""
    text = ' '.join(raw.split())
    clean = infer.keep_arabic_only(text)
    density, _ = infer.diacritic_density(clean)
    do_voc = (vocalize == 'always' or
              (vocalize == 'auto' and density < 0.30))
    if do_voc:
        processed = ' '.join(infer.catt_vocalize(clean).split())
    else:
        processed = clean
    toks_ms_fn, toks_egy_fn, _ids = infer.get_tokenizer(qaf_mode)
    toks_ms_l = toks_ms_fn(processed)
    toks_egy_l = toks_egy_fn(processed)
    entry = {
        'raw': raw, 'normalized': clean, 'vocalized': do_voc,
        'processed': processed, 'buckwalter': a2b(processed),
        'phonemes_ms': b2p(a2b(processed)),
        'tokens_ms': toks_ms_l, 'tokens_egy': toks_egy_l,
        'ids_egy': ids_of(toks_egy_l) if len(toks_ms_l) == len(toks_egy_l) else [],
        'q_positions': [i for i, t in enumerate(toks_ms_l) if t == 'q'],
    }
    if len(toks_ms_l) != len(toks_egy_l):
        entry['structure_mismatch'] = len(toks_ms_l) - len(toks_egy_l)
    return entry


# ---------------------------------------------------------------- verdicts --
def evaluate(entry, cls):
    if entry.get('structure_mismatch') is not None:
        return 'FAIL', (f"اختلاف بنية التوكنز بين msa/egy "
                        f"({entry['structure_mismatch']}) — خرق لأمان PATCH 6")
    if not entry['q_positions']:
        return 'SUSPICIOUS', 'لا يوجد توكن q للمقارنة (راجع يدويًا)'
    got = [entry['tokens_egy'][i] for i in entry['q_positions']]
    want = EXPECTED[cls]
    if all(t == want for t in got):
        return 'PASS', f"القاف → '{want}' كما هو متوقع للفئة {cls}"
    uniq = '/'.join(dict.fromkeys(got))
    return ('FAIL',
            f"القاف → '{uniq}' بينما الفئة {cls} تتوقع '{want}'")


def evaluate_mode(entry, want):
    """حكم حالات أوضاع PATCH 7 — نفس منطق evaluate بمعيار الوضع."""
    if entry.get('structure_mismatch') is not None:
        return 'FAIL', ('اختلاف بنية التوكنز — خرق لأمان PATCH 6/7')
    if not entry['q_positions']:
        return 'SUSPICIOUS', 'لا يوجد توكن q للمقارنة (راجع يدويًا)'
    got = [entry['tokens_egy'][i] for i in entry['q_positions']]
    if all(t == want for t in got):
        return 'PASS', f"القاف → '{want}' كما يتوقع الوضع"
    uniq = '/'.join(dict.fromkeys(got))
    return 'FAIL', f"القاف → '{uniq}' بينما الوضع يتوقع '{want}'"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--json', default=None, help='حفظ النتائج JSON')
    ap.add_argument('--quiet', action='store_true',
                    help='بلا جدول — رمز خروج فقط')
    args = ap.parse_args()

    (infer, a2b, b2p, p2t, ids_of, egy_map) = load_pipeline()

    rows, n_pass, n_fail, n_susp = [], 0, 0, 0
    for raw, cls, note in TEST_CASES:
        e = trace(raw, infer, a2b, b2p, p2t, ids_of, egy_map)
        verdict, why = evaluate(e, cls)
        n_pass += verdict == 'PASS'
        n_fail += verdict == 'FAIL'
        n_susp += verdict == 'SUSPICIOUS'
        rows.append({'input': raw, 'class': cls, 'class_desc': CLASS_DESC[cls],
                     'note': note, 'verdict': verdict, 'why': why,
                     'expected_qaf_token': EXPECTED[cls],
                     'got_qaf_tokens': [e['tokens_egy'][i]
                                        for i in e['q_positions']],
                     'normalized': e['normalized'],
                     'vocalized': e['vocalized'], 'processed': e['processed'],
                     'buckwalter': e['buckwalter'], 'phonemes': e['phonemes_ms'],
                     'tokens_ms': e['tokens_ms'], 'tokens_egy': e['tokens_egy'],
                     'token_ids_egy': e['ids_egy']})
        if not args.quiet:
            print(f'{raw[:22]:<24s} {cls} {verdict:<11s} {why[:64]}')

    # ------------------ PATCH 7: حالات الأوضاع ------------------
    mode_rows = []
    for raw, mode, want, note in MODE_CASES:
        e = trace(raw, infer, a2b, b2p, p2t, ids_of, egy_map,
                  vocalize='never', qaf_mode=mode)
        verdict, why = evaluate_mode(e, want)
        n_pass += verdict == 'PASS'
        n_fail += verdict == 'FAIL'
        n_susp += verdict == 'SUSPICIOUS'
        mode_rows.append({'input': raw, 'mode': mode, 'note': note,
                          'verdict': verdict, 'why': why,
                          'expected': want,
                          'got': [e['tokens_egy'][i] for i in e['q_positions']]})
        if not args.quiet:
            print(f'{raw[:22]:<24s} [{mode:<5s}] {verdict:<11s} {why[:58]}')

    # ------------------ PATCH 7: مسار الفصحى msa ------------------
    msa_rows = []
    for raw, mode, want, note in MSA_CASES:
        buck_fixed = infer.fix_qaf(a2b(raw), mode, 'msa')
        verdict = 'PASS' if 'q' not in buck_fixed and want in buck_fixed else 'FAIL'
        # البديل المرفوض: توكن q خام (غير مدرّب) في أي وضع
        got_q = 'q' in buck_fixed
        why = (f"باكوالتير بعد fix_qaf: {buck_fixed[:40]}…"
               if not got_q else 'توكن q خام غير مدرّب ما زال موجودًا!')
        n_pass += verdict == 'PASS'
        n_fail += verdict == 'FAIL'
        msa_rows.append({'input': raw, 'mode': mode, 'note': note,
                         'verdict': verdict, 'why': why,
                         'buck_fixed': buck_fixed})
        if not args.quiet:
            print(f'{raw[:22]:<24s} [msa/{mode:<5s}] {verdict:<11s} {why[:56]}')

    # ------------------ PATCH 9 (2026-10-01): العلامات + زرع الأشكال --------
    p9_cases = []

    def p9(name, ok, why=''):
        p9_cases.append((name, bool(ok), why))

    # 1) تحليل علامات النص
    clean_m, acts_m = infer.parse_qaf_markers(
        'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وقال{ء} شكرًا')
    p9('parse: نزع العلامات من النص', '{' not in clean_m and '}' not in clean_m,
       clean_m)
    p9('parse: قطعة=q', acts_m.get('qTEp') == 'q', str(acts_m))
    p9('parse: رقم=g', acts_m.get('rqm') == 'g', str(acts_m))
    p9('parse: قال=h', acts_m.get('qAl') == 'h', str(acts_m))
    clean_m2, acts_m2 = infer.parse_qaf_markers('القسمة{q} وقال {ء} وبقطعة{g}')
    p9('parse: لاتيني q + مسافة + سابقة',
       acts_m2.get('qsmp') == 'q' and acts_m2.get('qAl') == 'h'
       and acts_m2.get('bqTEp') == 'g', str(acts_m2))

    # 2) زرع الأشكال المدروسة (سلوك الإنتاج)
    r = infer.prepare_text_rich('قطعة', 'always', 'egy', 'auto')
    p9('plant: قطعة منفردة → الشكل المدروس', r['text'] == 'قِطْعَةً', r['text'])
    p9('plant: native=qTEp', r['qaf_native'] == frozenset({'qTEp'}),
       str(r['qaf_native']))
    r = infer.prepare_text_rich('قطع', 'always', 'egy', 'auto')
    p9('plant: قطع منفردة → الشكل المدروس', r['text'] == 'قِطَعٍ', r['text'])
    r = infer.prepare_text_rich('رقم', 'always', 'egy', 'auto')
    p9('plant: رقم (B) لا يُزرع في auto', not r['qaf_planted'], r['text'])
    r = infer.prepare_text_rich('قيمة', 'never', 'egy', 'qaf')
    p9('plant: قيمة خام في qaf تُزرع (tier2)', r['text'] == 'قِيمَةُ', r['text'])
    r = infer.prepare_text_rich('قيمة', 'never', 'egy', 'auto')
    p9('plant: قيمة خام في auto لا زرع (tier2)', not r['qaf_planted'], r['text'])
    r = infer.prepare_text_rich('قطعة', 'never', 'egy', 'g')
    p9('plant: وضع g بلا زرع تلقائي', not r['qaf_planted'], r['text'])
    r = infer.prepare_text_rich('قطعة{ق}', 'always', 'egy', 'g')
    p9('plant: العلامة {ق} تفوز على وضع g', r['text'] == 'قِطْعَةً', r['text'])
    r = infer.prepare_text_rich('قِطْعَةً{ء}', 'never', 'egy', 'auto')
    p9('plant: {ء} تنزع كسرة القاف', r['text'] == 'قَطْعَةً', r['text'])
    r = infer.prepare_text_rich('قسمنا قطع قماش كتير', 'always', 'egy', 'auto')
    p9('sentence: قطع الملتبسة بفعل تُتخطى داخل الجملة',
       not r['qaf_planted'], r['text'])

    # 3) fix_qaf مع أفعال العلامات و native
    p9('fix: {ء} تجاوز B للكلمة المحددة',
       'j' not in infer.fix_qaf('Alrqm', 'auto', 'egy', {'rqm': 'h'}))
    p9('fix: {ج} تجبر الجيم',
       'j' in infer.fix_qaf('qism', 'auto', 'egy', {'qsm': 'g'}))
    p9('fix: {ق} تمر القاف خامًا',
       infer.fix_qaf('qiTEpF', 'auto', 'egy', {'qTEp': 'q'}) == 'qiTEpF')
    p9('fix: qaf + native → خام (أصيلة)',
       infer.fix_qaf('qiTEpF', 'qaf', 'egy', None, {'qTEp'}) == 'qiTEpF')
    p9('fix: qaf بلا native → k', 'k' in infer.fix_qaf('qAEdp', 'qaf', 'egy'))
    p9('fix: B في auto → j', 'j' in infer.fix_qaf('Alrqm', 'auto', 'egy'))
    p9('fix: حارس بقرأ قبل العلامات',
       'j' not in infer.fix_qaf('bqr>', 'g', 'egy', {'bqr>': 'g'}))
    p9('fix: msa {ء} → همزة',
       infer.fix_qaf('qAl', 'auto', 'msa', {'qAl': 'h'}) == '<Al')

    # 4) المسار الكامل: جملة الأشكال الثلاثة
    mixed = 'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا'
    r = infer.prepare_text_rich(mixed, 'always', 'egy', 'auto')
    p9('full: قطعة مزروعة بالعلامة داخل الجملة',
       any(pl['skel'] == 'qTEp' for pl in r['qaf_planted']),
       str(r['qaf_planted']))
    _, toks_fn_p9, _ = infer.get_tokenizer(
        'auto', r['qaf_actions'] or None, r['qaf_native'] or None)
    toks_p9 = toks_fn_p9(r['text'])
    p9('full: لا توكن q خام غير مدرّب في المخرج', 'q' not in toks_p9,
       str(toks_p9[:24]))
    buck_p9 = a2b(r['text'])
    fixed_p9 = infer.fix_qaf(buck_p9, 'auto', 'egy', r['qaf_actions'])
    p9('full: رقم → جيم في باكوالتير المُصلَح', 'rajomi' in fixed_p9,
       fixed_p9[:60])
    p9('full: علامات النص لم تصل النص النهائي',
       '{' not in r['text'] and '}' not in r['text'], r['text'][:60])

    n_pass += sum(1 for _n, _ok, _w in p9_cases if _ok)
    n_fail += sum(1 for _n, _ok, _w in p9_cases if not _ok)
    p9_rows = [{'name': _n, 'verdict': 'PASS' if _ok else 'FAIL', 'why': _w}
               for _n, _ok, _w in p9_cases]
    if not args.quiet:
        print()
        for _n, _ok, _w in p9_cases:
            print(f'{("[P9] " + _n):<34s} {"PASS" if _ok else "FAIL":<11s} '
                  f'{_w[:52]}')

    # ------------------ PATCH 10 (2026-10-02): ترقية run1 للطبقات --------
    p10_cases = []

    def p10(name, ok, why=''):
        p10_cases.append((name, bool(ok), why))

    # 1) الترقيات الثلاث → verified (زرع في auto حتى بلا علامة)
    r = infer.prepare_text_rich('قَصَّةَ', 'never', 'egy', 'auto')
    p10('قصة (ترقية q?) تُزرع في auto', r['text'] == 'قِصَّةِ', r['text'])
    p10('native يشمل qSp', 'qSp' in r['qaf_native'], str(r['qaf_native']))
    r = infer.prepare_text_rich('قَطْب', 'never', 'egy', 'auto')
    p10('قطب (ترقية q?) تُزرع في auto', r['text'] == 'قُطْبٌ', r['text'])
    r = infer.prepare_text_rich('قَطْبَيْن', 'never', 'egy', 'auto')
    p10('قطبين (ترقية q?) تُزرع في auto', r['text'] == 'قُطْبَيْنِ', r['text'])

    # 2) التنزيلان → tier2 (لا زرع في auto)
    r = infer.prepare_text_rich('قَطَرَ', 'never', 'egy', 'auto')
    p10('قطر (تنزيل ء) لا تُزرع في auto', not r['qaf_planted'], r['text'])
    r = infer.prepare_text_rich('يقيس', 'never', 'egy', 'auto')
    p10('يقيس (تنزيل ء) لا يُزرع في auto', not r['qaf_planted'], r['text'])

    # 3) صيغة قطر المحدثة قُطْرًا (الشكل العميق المقيس في run1)
    r = infer.prepare_text_rich('قَطَرَ', 'never', 'egy', 'qaf')
    p10('قطر في qaf تُزرع بصيغة قُطْرًا', r['text'] == 'قُطْرًا', r['text'])
    r = infer.prepare_text_rich('قَطَرَ{ق}', 'never', 'egy', 'auto')
    p10('علامة {ق} تزرع قُطْرًا رغم tier2', r['text'] == 'قُطْرًا', r['text'])
    p10('fix_qaf: native قطر خام في qaf',
        infer.fix_qaf('qaTr', 'qaf', 'egy', None, {'qTr'}) == 'qaTr')

    # 4) قطعة تظل verified رغم قياس g (نقطة المعايرة — أذن المستخدم
    #    مقدّمة على المصنّف numpy الذي لا يقيس مكان النطق)
    r = infer.prepare_text_rich('قَطْعَةَ', 'never', 'egy', 'auto')
    p10('قطعة تظل مزروعة (معايرة الأذن)', r['text'] == 'قِطْعَةً', r['text'])

    n_pass += sum(1 for _n, _ok, _w in p10_cases if _ok)
    n_fail += sum(1 for _n, _ok, _w in p10_cases if not _ok)
    p10_rows = [{'name': _n, 'verdict': 'PASS' if _ok else 'FAIL', 'why': _w}
                for _n, _ok, _w in p10_cases]
    if not args.quiet:
        print()
        for _n, _ok, _w in p10_cases:
            print(f'{("[P10] " + _n):<34s} {"PASS" if _ok else "FAIL":<11s} '
                  f'{_w[:52]}')

    total = len(rows)
    by_class = {}
    for c in 'ABC':
        sub = [r for r in rows if r['class'] == c]
        by_class[c] = {'n': len(sub),
                       'fail': sum(r['verdict'] == 'FAIL' for r in sub)}
    summary = {
        'total': total, 'pass': n_pass, 'fail': n_fail, 'suspicious': n_susp,
        'by_class': by_class,
        'n_mode_cases': len(mode_rows),
        'n_msa_cases': len(msa_rows),
        'n_p9_cases': len(p9_rows),
        'n_p10_cases': len(p10_rows),
        'p10_cases_fail': sum(r['verdict'] == 'FAIL' for r in p10_rows),
        'p9_cases_fail': sum(r['verdict'] == 'FAIL' for r in p9_rows),
        'mode_cases_fail': sum(r['verdict'] == 'FAIL' for r in mode_rows),
        'msa_cases_fail': sum(r['verdict'] == 'FAIL' for r in msa_rows),
        'policy': ("A→'<' (قاهري [ʔ])، B→'v' ([g] ثابت)، "
                   "C→'<' (متغير → افتراضي [ʔ] — PATCH 6b)"),
        'policy_p9': (
            "PATCH 9 — زرع الأشكال المدروسة (29 هيكلا) + علامات النص "
            "{ق}/{ء}/{ج} تخلط الأشكال الثلاثة في الجملة الواحدة؛ الملتبسة "
            "بأفعال تُزرع منفردة/بعلامة فقط؛ قرار B ساري في auto"),
        'policy_p10': (
            "PATCH 10 (run1 2026-10-02، speaker 0) — ترقية الطبقات بالقياس "
            "الآلي: verified=5 (قطعة/قطع أذن + قصة/قطب/قطبين قياس q?) "
            "tier1=12 tier2=12؛ قطر/يقيس نزلتا tier2 (زرعهما قاس ء) وصيغة "
            "قطر تحدثت إلى قُطْرًا (قاست q? عميقًا 46ms). معايرة: قياس g على "
            "المزروع لا ينزل من auto (قطعة قاست g وأذن المستخدم أصيلة — "
            "مجهور مؤخر)؛ فقط ء ينزل و q? يرقّي"),
        'policy_modes': (
            "PATCH 7 — auto: السلوك المعتمد دون تغيير | qaf: قائمة Q (81 "
            "هيكلًا) → 'k' قاف فصحى تقريبية، الباقي كما auto | hamza: كل "
            "قاف '<' | g: كل قاف 'v' | حارس بقرأ يعمل في كل الأوضاع | msa: "
            "كل قاف 'k' تقريبًا (كانت توكن خام غير مدرّب)"),
        'policy_decision': (
            "مراجعة المستخدم 2026-10-01: الفئة B فقط ثابتة [g]؛ حقيقة/دقيق/"
            "قراءة/قرأ وعائلاتها عادت للافتراضي [ʔ] حفاظًا على السلوك المتعلّم "
            "(874 سماعًا لعائلة حقيقة)؛ حارس مستقل لصيغة «بقرأ» التقدمية؛ "
            "وفي القرار نفسه: أوضاع نطق القاف (auto/qaf/hamza/g) + قائمة Q "
            "لفصحى المصطلحات التعليمية (تقريب [k] — القاف الأصيلة مؤجلة "
            "كمشروع إعادة تدريب مستقل)"),
        'root_cause': ("EGY_TOKEN_MAP q→'<' غير مشروطة في toks_egy "
                       "(infer.py + نواة التدريب) — عولجت في الاستدلال بـ"
                       "PATCH 6 (QAF_G_SKELETONS + fix_qaf_g في infer.py)، "
                       "ثم عُمّمت بـPATCH 7 (fix_qaf بأوضاع auto/qaf/"
                       "hamza/g + إصلاح مسار msa)"),
    }
    if not args.quiet:
        print('\n' + '=' * 76)
        print(f"الملخص: {n_pass}/{n_pass + n_fail + n_susp} PASS | "
              f"{n_fail} FAIL | {n_susp} SUSPICIOUS")
        for c in 'ABC':
            s = by_class[c]
            print(f"  الفئة {c} ({CLASS_DESC[c][:38]}...): "
                  f"{s['n'] - s['fail']}/{s['n']} PASS")
        print(f"  أوضاع PATCH 7 (auto/qaf/hamza/g): "
              f"{len(mode_rows) - summary['mode_cases_fail']}/{len(mode_rows)}"
              f" PASS")
        print(f"  PATCH 9 (علامات + زرع الأشكال): "
              f"{len(p9_rows) - summary['p9_cases_fail']}/{len(p9_rows)}"
              f" PASS")
        print(f"  PATCH 10 (ترقية run1 للطبقات): "
              f"{len(p10_rows) - summary['p10_cases_fail']}/{len(p10_rows)}"
              f" PASS")
        print(f"  مسار msa المُصلَح: "
              f"{len(msa_rows) - summary['msa_cases_fail']}/{len(msa_rows)}"
              f" PASS")
        print('\nملاحظة: فئة C = نطق متغير — عادت للافتراضي [ʔ] بقرار مراجعة '
              'المستخدم (2026-10-01)، وتشمل حارس «بقرأ» التقدمية (PATCH 6b). '
              'أوضاع PATCH 7 إضافية فوق السلوك المعتمد — الافتراض auto بلا '
              'أي تغيير؛ وضع qaf قائمة مغلقة (81 هيكلًا) ليست كل كلمة قاف؛ '
              'القاف الفصحى تقريب [k] (توكن q غير مدرّب — 0/1.59M).')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'rows': rows,
                       'p9_rows': p9_rows,
                       'mode_rows': mode_rows, 'msa_rows': msa_rows}, f,
                      ensure_ascii=False, indent=1)
        if not args.quiet:
            print(f'\nحُفظت النتائج: {args.json}')

    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
