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
# المتوقع: 'v' = جيم [g] | '<' = خام/همزة [ʔ] (لا 'k' — حظر الكاف PATCH 13)
MODE_CASES = [
    # -- الوضع الافتراضي auto = السلوك المعتمد (بلا أي تغيير) --
    ('رقم', 'auto', 'v', 'B سارية في auto — قرار معتمد دون تغيير'),
    ('قانون', 'auto', 'v', 'B سارية في auto'),
    ('القرآن', 'auto', 'v', 'B سارية في auto (قرار مُعاد تأكيده 2026-10-01)'),
    ('قيمة', 'auto', '<', 'خارج B — همزة في auto'),
    ('قسمة', 'auto', '<', 'خارج B — همزة في auto'),
    # -- وضع qaf: أمثلة المستخدم + قائمة Q --
    ('رقم', 'qaf', '<', 'مثال المستخدم — Q تفوز على B في وضع الفصحى'),
    ('أرقام', 'qaf', '<', 'مثال المستخدم'),
    ('قسمة', 'qaf', '<', 'مثال المستخدم'),
    ('يقسم', 'qaf', '<', 'مثال المستخدم'),
    ('تقريب', 'qaf', '<', 'مثال المستخدم'),
    ('تقريبًا', 'qaf', '<', 'ظرف شائع في الشرح'),
    ('قياس', 'qaf', '<', 'منهج أساسي'),
    ('مقياس', 'qaf', '<', 'مقياس الرسم — جغرافيا/هندسة'),
    ('قيمة', 'qaf', '<', 'القيمة المكانية'),
    ('قاعدة', 'qaf', '<', 'قواعد الحساب'),
    ('قواعد', 'qaf', '<', 'قواعد اللغة'),
    ('قانون', 'qaf', '<', 'تعارض Q/B — Q تفوز في وضع الفصحى فقط'),
    ('قطر', 'qaf', '<', 'قطر الدائرة (والدولة — نفس الهيكل)'),
    ('المقام', 'qaf', '<', 'مقام الكسر — تعارض مع «في المقام الأول» B: auto=g / qaf=k'),
    ('مستقيم', 'qaf', '<', 'الخط المستقيم'),
    ('قيمة العدد تسعة هي تسعة', 'qaf', '<', 'جملة رياضيات — القيمة كلمة القاف الوحيدة'),
    ('المطلقة', 'qaf', '<', 'خارج قائمة Q — تبقى همزة (كلمات بعينها فقط)'),
    ('قوة', 'qaf', '<', 'فيزياء — القوة'),
    ('طاقة', 'qaf', '<', 'فيزياء — الطاقة'),
    ('حقوق', 'qaf', '<', 'تربية مدنية'),
    ('القسمة', 'qaf', '<', 'ال+ قسمة'),
    ('ويقسم', 'qaf', '<', 'و+ يقسم'),
    # -- وضع qaf: ليست كل كلمة قاف! --
    ('قمر', 'qaf', '<', 'خارج القائمتين — تبقى همزة (كلمات بعينها فقط)'),
    ('قال', 'qaf', '<', 'خارج القائمتين — همزة'),
    ('حقيقة', 'qaf', '<', 'C تبقى همزة حتى في وضع الفصحى'),
    ('قراءة', 'qaf', '<', 'C تبقى همزة'),
    ('القرآن', 'qaf', '<', 'PATCH 11: عائلة قرآن انضمت لقائمة Q — qaf يمرّها خامًا (لا كاف — PATCH 13)'),
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

# حالات مسار الفصحى msa (PATCH 7 + 13 — إصلاح التوكن غير المدرّب، بلا كاف):
# (المدخل، الوضع، القاف المتوقعة في باكوالتير بعد fix_qaf)
MSA_CASES = [
    ('القرآن كتاب عظيم', 'auto', '<', 'msa: قاف خام طبيعي — PATCH 13 (كانت k محظورة)'),
    ('قال المعلم', 'auto', '<', 'msa auto: قاف خام طبيعي'),
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
    p9('fix: {ق} مع native → خام',
       infer.fix_qaf('qiTEpF', 'auto', 'egy', {'qTEp': 'q'},
                     {'qTEp'}) == 'qiTEpF')
    p9('fix: {ق} بلا native → خام طبيعي (PATCH 13 — لا كاف)',
       infer.fix_qaf('qiTEpF', 'auto', 'egy',
                     {'qTEp': 'q'}) == 'qiTEpF')
    p9('fix: qaf + native → خام (أصيلة)',
       infer.fix_qaf('qiTEpF', 'qaf', 'egy', None, {'qTEp'}) == 'qiTEpF')
    p9('fix: qaf بلا native → خام (PATCH 13)',
       infer.fix_qaf('qAEdp', 'qaf', 'egy') == 'qAEdp')
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

    # ------------- PATCH 11 (2026-10-02): جملة المستخدم الفاشلة ---------
    p11_cases = []

    def p11(name, ok, why=''):
        p11_cases.append((name, bool(ok), why))

    # 1) جملة المستخدم حرفيًا — كل كلمة {ق} تخرج من الهمزة العرضية
    fail_sent = ('القرآن{ق} كتاب عظيم، وكل قرآن{ق} نقرأه{ق} يذكرنا بآيات '
                 'القرآن، وفي القرآن{ق} قصص{ق} وحكم كثيرة')
    r = infer.prepare_text_rich(fail_sent, 'always', 'egy', 'auto')
    # PATCH 12: المعرفة (القرآن) لا تُزرع — زرعها OOD (صفر مواضع بالتعريف
    # في corpus §3-أ) كان يُسمع همزة؛ الآن [k] حتمي (بوابة QAF_Q_AFFIX_OK)
    p11('full: القرآن بالتعريف تُزرع أفضل-جهد الآن (PATCH 13)',
       any(pl['skel'] == 'Alqr|n' and pl['was'].startswith('ال')
           for pl in r['qaf_planted']),
       str(r['qaf_planted']))
    p11('full: قصص مزروعة قِصَصٍ',
       any(pl['skel'] == 'qSS' and pl['now'] == 'قِصَصٍ'
           for pl in r['qaf_planted']), str(r['qaf_planted']))
    p11('full: native يشمل qr|n و qSS',
       {'qr|n', 'qSS'} <= set(r['qaf_native']), str(r['qaf_native']))
    _, toks_p11, _ = infer.get_tokenizer(
        'auto', r['qaf_actions'] or None, r['qaf_native'] or None)
    toks_p11 = toks_p11(r['text'])
    p11('full: لا توكن q خام في المخرج النهائي', 'q' not in toks_p11,
       str(toks_p11[:30]))

    # 2) نقرأه{ق} — العلامة تصل التوكنات (إصلاح ه↔ة) وتمر خامًا طبيعيًا
    clean_p11, acts_p11 = infer.parse_qaf_markers('نقرأه{ق}')
    p11('parse: نقرأه{ق} → nqr>p', acts_p11.get('nqr>p') == 'q',
       str(acts_p11))
    p11('fix: نقرأه{ق} → خام طبيعي (PATCH 13 — لا كاف)',
       infer.fix_qaf('naqora>ahu', 'auto', 'egy',
                     {'nqr>p': 'q'}) == 'naqora>ahu')
    p11('fix: نقرأه{ء} يصل أيضًا (ه↔ة)',
       infer.fix_qaf('naqora>ahu', 'auto', 'egy',
                     {'nqr>p': 'h'}) == 'naqora>ahu')

    # 3) السياسة (PATCH 13): {ق} = أصيلة مزروعة أو خام طبيعي — لا كاف أبدًا
    # (سكّل باكوالتير qbArp: الحركات aui تُنزع قبل المطابقة)
    p11('fix: {ق} لكلمة بلا معجم → خام (لا كاف)',
       infer.fix_qaf('qubArp', 'auto', 'egy',
                     {'qbArp': 'q'}) == 'qubArp')
    p11('fix: {ق} مجردة native → خام (لا مسبوقة)',
       infer.fix_qaf('quro|nK', 'auto', 'egy', {'qr|n': 'q'},
                     {'qr|n'}) == 'quro|nK')
    p11('fix: رقمه (ه↔ة) يطابق B الآن → j',
       'j' in infer.fix_qaf('Alrqmh', 'auto', 'egy'))
    p11('fix: رقمه{ء} يجاوز B الآن (وصل الهيكل)',
       'j' not in infer.fix_qaf('Alrqmh', 'auto', 'egy', {'rqmp': 'h'}))

    # 4) قرار B ساري بلا علامة — لا انحدار في auto
    r = infer.prepare_text_rich('القرآن كتاب عظيم', 'always', 'egy', 'auto')
    p11('auto بلا علامة: القرآن B → [g] (لا انحدار)',
       not r['qaf_planted'] and 'j' in infer.fix_qaf(
           a2b(r['text']), 'auto', 'egy'), r['text'])
    r = infer.prepare_text_rich('قصص كثيرة', 'always', 'egy', 'auto')
    p11('auto بلا علامة: قصص لا تُزرع (tier2)', not r['qaf_planted'],
       r['text'])
    r = infer.prepare_text_rich('قصص كثيرة', 'always', 'egy', 'qaf')
    p11('qaf بلا علامة: قصص native (catt قِصَصٌ deep → تخطٍّ PATCH 9)',
       'qSS' in r['qaf_native'],
       f"planted={r['qaf_planted']} native={r['qaf_native']} {r['text']}")
    p11('qaf بلا علامة (plain): قصص{ق} العلامة تزرع قِصَصٍ رغم deep',
       (lambda rr: rr['text'].split()[0] == 'قِصَصٍ')(
           infer.prepare_text_rich('قصص{ق} كثيرة', 'always', 'egy', 'qaf')), '')
    p11('qaf: القرآن المجردة native خام (Q تفوق B)',
       infer.fix_qaf('quro|nK', 'qaf', 'egy', None,
                     {'qr|n'}) == 'quro|nK')

    # 5) msa: {ق} = خام '<' في كل وضع (العلامة تتفوق — qur|n = قُرْآن بالمدة)
    p11('msa: {ق} → خام (PATCH 13 — لا كاف)',
       infer.fix_qaf('qur|n', 'hamza', 'msa',
                     {'qr|n': 'q'}) == '<ur|n')

    n_pass += sum(1 for _n, _ok, _w in p11_cases if _ok)
    n_fail += sum(1 for _n, _ok, _w in p11_cases if not _ok)
    p11_rows = [{'name': _n, 'verdict': 'PASS' if _ok else 'FAIL', 'why': _w}
               for _n, _ok, _w in p11_cases]
    if not args.quiet:
        print()
        for _n, _ok, _w in p11_cases:
            print(f'{("[P11] " + _n):<34s} {"PASS" if _ok else "FAIL":<11s} '
                  f'{_w[:52]}')

    # ------------- PATCH 12 (2026-10-02): العلامات تعم كل كلمة -------------
    p12_cases = []

    def p12(name, ok, why=''):
        p12_cases.append((name, bool(ok), why))

    # 1) إصلاح خلل اللصق — الوسم يُستبدل بمسافة (كل أنماط المسافات)
    c, a = infer.parse_qaf_markers('وكل قرآن {ق}نقرأه{ق} يذكرنا')
    p12('لصق: «قرآن {ق}نقرأه{ق}» لا تلتصق الكلمتان',
       c == 'وكل قرآن نقرأه يذكرنا', c)
    p12('لصق: العلامتان تصلان (qr|n + nqr>p)',
       a.get('qr|n') == 'q' and a.get('nqr>p') == 'q', str(a))
    c, _ = infer.parse_qaf_markers('قال{ء}شكرًا')
    p12('لصق: «قال{ء}شكرًا» بلا مسافة → تُفصل بمسافة', c == 'قال شكرًا', c)
    c, _ = infer.parse_qaf_markers('قطعة{ق} قماش')
    p12('لصق: الملتحقة تعمل كما كانت', c == 'قطعة قماش', c)
    c, _ = infer.parse_qaf_markers('القرآن {ق} كتاب')
    p12('لصق: مسافة قبل وبعد → نظيفة', c == 'القرآن كتاب', c)
    c, _ = infer.parse_qaf_markers('القرآن{ق}، وفي')
    p12('لصق: الترقيم يعود للتصاقه (جمالي)', c == 'القرآن، وفي', c)

    # 2) المسبوقة المعرفة — PATCH 13: زرع أفضل-جهد (لا كاف أبدًا)
    r = infer.prepare_text_rich('القرآن{ق} كتاب عظيم', 'always', 'egy', 'auto')
    p12('زرع أفضل-جهد: القرآن{ق} (معرفة) تُزرع الْقُرْآن',
       any(pl['skel'] == 'Alqr|n' for pl in r['qaf_planted'])
       and 'Aloquro|n' in infer.fix_qaf(
           a2b(r['text']), 'auto', 'egy', r['qaf_actions'] or None,
           r['qaf_native'] or None), r['text'])
    p12('حارس التسريب: المسبوقة غير المزروعة خام (لا كاف — P13)',
       infer.fix_qaf('Aloquro|n', 'auto', 'egy', {'qr|n': 'q'},
                     {'qr|n'}) == 'Aloquro|n')
    p12('حارس التسريب: المسبوقة في qaf خام أيضًا (P13)',
       infer.fix_qaf('Aloquro|n', 'qaf', 'egy', None,
                     {'qr|n'}) == 'Aloquro|n')
    p12('زرع: القطعة{ق} (معرفة بتعرض corpus) تُزرع',
       (lambda rr: any(pl['skel'] == 'AlqTEp' and pl['was'].startswith('ال')
                       for pl in rr['qaf_planted']))(
           infer.prepare_text_rich('القطعة{ق} كبيرة', 'always', 'egy',
                                   'auto')), '')
    p12('زرع: القسم{ق} (معرفة qsm∈AFFIX_OK) خام',
       infer.fix_qaf('Aloqisomi', 'auto', 'egy', {'qsm': 'q'},
                     {'Alqsm'}) == 'Aloqisomi')

    # 3) قرآنا (qr|nA) — ألف التنوين تطابق المعجم وتُزرع قُرْآنَ
    c, a = infer.parse_qaf_markers('قرآنا{ق}')
    p12('ألف التنوين: قرآنا{ق} → qr|nA + qr|n',
       a.get('qr|nA') == 'q' and a.get('qr|n') == 'q', str(a))
    r = infer.prepare_text_rich('يتدبر قرآنا{ق} كريما', 'always', 'egy',
                                'auto')
    p12('ألف التنوين: قرآنا{ق} تُزرع قُرْآنَ (شكل corpus)',
       any(pl['now'] == 'قُرْآنَ' and pl['skel'] == 'qr|n'
           for pl in r['qaf_planted']), str(r['qaf_planted']))
    p12('ألف التنوين: المزروعة خام (هيكل ما بعد الزرع)',
       infer.fix_qaf('quro|na', 'auto', 'egy', {'qr|n': 'q'},
                     {'qr|n'}) == 'quro|na')
    p12('ألف التنوين: قانونا بلا علامة يطابق B → j',
       'j' in infer.fix_qaf('qAnuwnaA', 'auto', 'egy'))

    # 4) TRUST — الموثوقة خامًا (حقيقة: 197 موضعًا + قياس run1)
    r = infer.prepare_text_rich('حقيقة{ق} مؤلمة', 'always', 'egy', 'auto')
    p12('TRUST: حقيقة{ق} خام (لا [k])', 'Hqyqp' in r['qaf_native'],
       f"native={r['qaf_native']}")
    p12('TRUST: الحقيقة{ق} (معرفة) خام أيضًا',
       (lambda rr: 'AlHqyqp' in rr['qaf_native'])(
           infer.prepare_text_rich('الحقيقة{ق} مؤلمة', 'always', 'egy',
                                   'auto')), '')
    p12('TRUST: fix_qaf خام لا k',
       infer.fix_qaf('Haqiyqap', 'auto', 'egy', {'Hqyqp': 'q'},
                     {'Hqyqp'}) == 'Haqiyqap')

    # 5) ربط الوسم بالكلمة قبل التقسيم (webapp)
    try:
        from webapp import bind_markers_to_words, split_into_chunks
        p12('webapp: «كلمة {ق}» تُربط قبل التقسيم',
           bind_markers_to_words('كلمة {ق} بعدها') == 'كلمة{ق} بعدها')
        long_txt = ('جملة طويلة جدا تحتاج تقطيع عند حدود الكلمات ' * 6
                    + 'وفي آخرها القرآن {ق} عظيم')
        chunks = split_into_chunks(long_txt, 'egy', infer.get_tokenizer('auto'))
        p12('webapp: الوسم لا ينفصل عن كلمته بعد التقسيم',
           all(' {ق}' not in ch for ch in chunks)
           and any('القرآن{ق}' in ch for ch in chunks),
           str([ch[-25:] for ch in chunks]))
    except Exception as e:  # noqa: BLE001
        p12('webapp: ربط الوسم قبل التقسيم (تخطي — استيراد)', False, str(e))

    # 6) جملة المستخدم الثانية حرفيًا (بمسافاتها) — التكامل الكامل
    user2 = ('القرآن {ق} كتاب عظيم، وكل قرآن {ق}نقرأه{ق} يذكرنا بآيات '
             'القرآن{ق}، وفي القرآن{ق} قصص{ق} وحكم كثيرة، ونحب أن نقرأ '
             'القرآن{ق} كل يوم، لأن قرآننا{ق} مصدر هداية، وكل من يقرأ{ق} '
             'القرآن{ق} يتعلم من آيات القرآن{ق}، ويعود إلى القرآن{ق} كلما '
             'أراد أن يتدبر قرآنا{ق} كريما.')
    r = infer.prepare_text_rich(user2, 'always', 'egy', 'auto')
    fixed = infer.fix_qaf(a2b(r['text']), 'auto', 'egy',
                          r['qaf_actions'] or None, r['qaf_native'] or None)
    p12('جملة المستخدم 2: كل معرفة مزروعة خام (لا كاف — P13)',
       'Alokuro|n' not in fixed and 'Aloquro|n' in fixed, fixed[:60])
    p12('جملة المستخدم 2: قصص/قرآنا مزروعتان native',
       {'qSS', 'qr|n'} <= set(r['qaf_native']),
       f"native={sorted(r['qaf_native'])}")
    p12('جملة المستخدم 2: نقرأه/يقرأ/قرآننا → خام طبيعي (P13)',
       'naqora>uhu' in fixed and 'yaqora>u' in fixed
       and 'quro|nanaA' in fixed, '')
    p12('جملة المستخدم 2: بلا لصق (كل الكلمات سليمة)',
       'قرآننقرأه' not in r['text'] and 'قُرْآنٍنَقْرَأَهُ' not in r['text'],
       r['text'][:50])

    # 7) لا انحدار: الجملة العاملة القديمة (الأشكال الثلاثة)
    r = infer.prepare_text_rich(
        'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا',
        'always', 'egy', 'auto')
    fixed = infer.fix_qaf(a2b(r['text']), 'auto', 'egy',
                          r['qaf_actions'] or None, r['qaf_native'] or None)
    p12('لا انحدار: قطعة{ق} خام أصيلة + رقم{ج} جيم + قال{ء} خام',
       'qiToEapF' in fixed and 'rajomi' in fixed and 'qaAla' in fixed,
       fixed[:60])

    n_pass += sum(1 for _n, _ok, _w in p12_cases if _ok)
    n_fail += sum(1 for _n, _ok, _w in p12_cases if not _ok)
    p12_rows = [{'name': _n, 'verdict': 'PASS' if _ok else 'FAIL', 'why': _w}
                for _n, _ok, _w in p12_cases]
    if not args.quiet:
        print()
        for _n, _ok, _w in p12_cases:
            print(f'{("[P12] " + _n):<34s} {"PASS" if _ok else "FAIL":<11s} '
                  f'{_w[:52]}')


    # ------------- PATCH 13 (2026-10-02): لا كاف أبدًا + قيامة + توحيد -----
    p13_cases = []

    def p13(name, ok, why=''):
        p13_cases.append((name, bool(ok), why))

    # 1) يوم القيامة{ق} — حالة المستخدم الحرفية (كانت «الكيامة»)
    r = infer.prepare_text_rich('ننتظر يوم القيامة{ق} بفارغ الصبر',
                                'always', 'egy', 'auto')
    p13('قيامة: يوم القيامة{ق} تُزرع الْقِيَامَة (كسرة قاف)',
       any(pl['skel'] == 'AlqyAmp' and 'قِيَامَة' in pl['now']
           for pl in r['qaf_planted']), str(r['qaf_planted']))
    fixed13 = infer.fix_qaf(a2b(r['text']), 'auto', 'egy',
                            r['qaf_actions'] or None, r['qaf_native'] or None)
    p13('قيامة: لا كاف (توكن القاف q خام)',
       'AloqiyaAmap' in fixed13 and 'AlokiyaAmap' not in fixed13, fixed13[:60])
    r = infer.prepare_text_rich('القيامة{ق} الكبرى', 'always', 'egy', 'auto')
    p13('قيامة: المعرفة منفردة تُزرع أيضًا',
       any(pl['skel'] == 'AlqyAmp' for pl in r['qaf_planted']),
       str(r['qaf_planted']))

    # 2) حارس الالتباس المعجمي: قلب{ق} لا تصبح كلبًا أبدًا
    p13('حارس الالتباس: قلب{ق} → qalobi خام (لا كلب!)',
       infer.fix_qaf('qalobi', 'auto', 'egy',
                     {'qlb': 'q'}) == 'qalobi')
    p13('حارس الالتباس: قلبي{ق} → qalobiy خام',
       infer.fix_qaf('qalobiy', 'auto', 'egy',
                     {'qlby': 'q'}) == 'qalobiy')
    p13('حارس الالتباس: msa قلب → <alob (لا كلب)',
       infer.fix_qaf('qalob', 'auto', 'msa') == '<alob')

    # 3) معجم corpus الموسع (148 هيكلاً — أشكال حرفية من بيانات التدريب)
    r = infer.prepare_text_rich('القصة{ق} جميلة', 'always', 'egy', 'auto')
    p13('corpus: القصة{ق} منفردة تُزرع شكل corpus (اِلْقِصَّة)',
       any(pl['skel'] == 'AlqSp' and 'قِصَّة' in pl['now']
           for pl in r['qaf_planted']), str(r['qaf_planted']))
    r = infer.prepare_text_rich('الحقيقة{ق} المؤلمة', 'always', 'egy', 'auto')
    p13('corpus: الحقيقة{ق} (معرفة 115 موضعًا) خام TRUST',
       'AlHqyqp' in r['qaf_native'], f"native={r['qaf_native']}")
    r = infer.prepare_text_rich('بيقول{ق} كلام كتير', 'always', 'egy', 'auto')
    p13('corpus: بيقول{ق} (شكل corpus الموثق) يُزرع',
       any(pl['skel'] == 'byqwl' for pl in r['qaf_planted']),
       str(r['qaf_planted']))

    # 4) حظر الكاف الشامل — مسح كل المسارات
    sweep = [
        ('naqora>ahu', 'auto', 'egy', {'nqr>p': 'q'}, None),   # علامة OOD
        ('raqom', 'qaf', 'egy', None, None),                    # وضع qaf
        ('Aloquro|n', 'qaf', 'egy', None, {'qr|n'}),            # مسبوقة qaf
        ('qAEdp', 'qaf', 'egy', None, None),                    # Q بلا شكل
    ]
    no_k = all('k' not in infer.fix_qaf(b, m, d, a, n2)
               for b, m, d, a, n2 in sweep)
    p13('الحظر الشامل: لا كاف في أي مسار مصري', no_k,
       str([infer.fix_qaf(b, m, d, a, n2)[:20]
            for b, m, d, a, n2 in sweep]))
    msa_no_k = all(
        'k' not in infer.fix_qaf(a2b(t), 'auto', 'msa')
        for t in ('قال المعلم قلبًا صادقًا', 'رقم خمسة وقيمة عشرة'))
    p13('الحظر الشامل: لا كاف في مسار msa', msa_no_k, '')

    # 5) msa: {ق} خام + الأوضاع الثلاثة
    p13('msa: {ق} → خام <',
       infer.fix_qaf('qur|n', 'auto', 'msa',
                     {'qr|n': 'q'}) == '<ur|n')
    p13('msa: فرض g → j', infer.fix_qaf('qur|n', 'g', 'msa') == 'jur|n')
    p13('msa: فرض hamza → <', infer.fix_qaf('qur|n', 'hamza', 'msa') == '<ur|n')

    # 6) لا انحدار auto: رقم بلا علامة = B جيم، قرآن بلا علامة = B
    p13('لا انحدار: رقم auto B → جيم',
       'j' in infer.fix_qaf('raqom', 'auto', 'egy'))
    p13('لا انحدار: قرآن auto B → جيم',
       'j' in infer.fix_qaf('quro|n', 'auto', 'egy'))
    p13('لا انحدار: قمر auto → خام همزة',
       infer.fix_qaf('qamar', 'auto', 'egy') == 'qamar')

    # 7) توحيد النبرة (webapp) — تعادل RMS + تطبيع واحد
    try:
        from webapp import _unify_chunks_tone, CHUNK_GAP_S
        import numpy as np
        sr = 22050
        w1 = 0.02 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype('float32')
        w2 = 0.10 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype('float32')
        final_w, tone_info = _unify_chunks_tone([w1.copy(), w2.copy()])
        rms = [float(np.sqrt(np.mean(np.square(w))))
               for w in (w1, w2)]
        gains13 = tone_info['rms_gains']
        post13 = [rms[i] * gains13[i] for i in range(len(rms))]
        p13('توحيد النبرة: تعادل RMS بين المقطعين',
           min(post13) > 0.74 * max(post13),
           f"gains={gains13} pre={rms} post={post13}")
        p13('توحيد النبرة: تطبيع ذروة واحد للنص',
           abs(float(np.abs(final_w).max()) - 0.9) < 0.01,
           f"peak={float(np.abs(final_w).max()):.3f}")
        p13('توحيد النبرة: طول النص = مقطعان + فاصل',
           abs(len(final_w) - (2 * sr + int(CHUNK_GAP_S * sr))) < 3,
           f"len={len(final_w)}")
        # مقطع واحد: لا تعادل + تطبيع واحد (سلوك متسق)
        f1, t1 = _unify_chunks_tone([w2.copy()])
        p13('توحيد النبرة: مقطع واحد بلا تعادل RMS',
           t1['rms_gains'] is None and
           abs(float(np.abs(f1).max()) - 0.9) < 0.01, '')
    except Exception as e:  # noqa: BLE001
        p13('توحيد النبرة (تخطي — استيراد webapp)', False, str(e))

    n_pass += sum(1 for _n, _ok, _w in p13_cases if _ok)
    n_fail += sum(1 for _n, _ok, _w in p13_cases if not _ok)
    p13_rows = [{'name': _n, 'verdict': 'PASS' if _ok else 'FAIL', 'why': _w}
                for _n, _ok, _w in p13_cases]
    if not args.quiet:
        print()
        for _n, _ok, _w in p13_cases:
            print(f'{("[P13] " + _n):<44s} {"PASS" if _ok else "FAIL":<11s} '
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
        'n_p11_cases': len(p11_rows),
        'n_p12_cases': len(p12_rows),
        'n_p13_cases': len(p13_rows),
        'p12_cases_fail': sum(r['verdict'] == 'FAIL' for r in p12_rows),
        'p13_cases_fail': sum(r['verdict'] == 'FAIL' for r in p13_rows),
        'p11_cases_fail': sum(r['verdict'] == 'FAIL' for r in p11_rows),
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
        'policy_p11': (
            "PATCH 11 (2026-10-02، جملة المستخدم «القرآن{ق}... قصص{ق}») — "
            "+عائلة قرآن (5 هياكل) + قصص للمعجم tier2 بأشكال corpus "
            "موثقة (قُرْآنٍ/قِصَصٍ)؛ إصلاح ه↔ة بين طبقتي النص والتوكنات "
            "(علامات نقرأه{ق}/رقمه{ء} كانت تضيع)؛ السياسة الحتمية: علامة "
            "{ق} = أصيلة مزروعة (معجم) أو تقريب [k] — لا همزة عرضية أبدًا "
            "(كانت {ق} تخطي B ثم تُنطق همزة خام)"),
        'policy_p12': (
            "PATCH 12 (2026-10-02، جملة المستخدم الثانية «القرآن {ق}… "
            "قرآنا{ق}») — العلامات تعم كل كلمة: إصلاح خلل اللصق (الوسم "
            "يُستبدل بمسافة — لا التصاق مهما كانت مسافاته)؛ بوابة السوابق "
            "QAF_Q_AFFIX_OK بأدلة corpus (المسبوقة OOD مثل الْقُرْآن صفر "
            "مواضع → [k] لا زرع يُسمع همزة، ولا ركوب native العائلة)؛ ألف "
            "تنوين النصب تطابق المعجم (قرآنا → قُرْآنَ شكل corpus)؛ QAF_Q_TRUST "
            "للموثوقة خامًا (حقيقة 197 موضعًا + قياس run1)؛ ربط الوسم بكلمته "
            "قبل تقسيم الويب"),
        'policy_modes': (
            "PATCH 7 + 13 — auto: السلوك المعتمد دون تغيير | qaf: قائمة Q "
            "مزروعة خامًا أصيلة (المعجمان)، الباقي خام طبيعي — لا كاف أبدًا "
            "| hamza: كل قاف '<' | g: كل قاف 'v' | حارس بقرأ يعمل في كل "
            "الأوضاع | msa: كل قاف خام '<' (توكن q غير مدرّب — كان تقريب k "
            "محظورًا PATCH 13)"),
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
        print(f"  PATCH 11 (جملة المستخدم + ه↔ة + ضمان العلامة): "
              f"{len(p11_rows) - summary['p11_cases_fail']}/{len(p11_rows)}"
              f" PASS")
        print(f"  PATCH 12 (العلامات تعم كل كلمة + بوابة السوابق): "
              f"{len(p12_rows) - summary['p12_cases_fail']}/{len(p12_rows)}"
              f" PASS")
        print(f"  PATCH 13 (لا كاف أبدًا + قيامة + معجم corpus + النبرة): "
              f"{len(p13_rows) - summary['p13_cases_fail']}/{len(p13_rows)}"
              f" PASS")
        print(f"  مسار msa المُصلَح: "
              f"{len(msa_rows) - summary['msa_cases_fail']}/{len(msa_rows)}"
              f" PASS")
        print('\nملاحظة: فئة C = نطق متغير — عادت للافتراضي [ʔ] بقرار مراجعة '
              'المستخدم (2026-10-01)، وتشمل حارس «بقرأ» التقدمية (PATCH 6b). '
              'أوضاع PATCH 7 إضافية فوق السلوك المعتمد — الافتراض auto بلا '
              'أي تغيير؛ وضع qaf قائمة مغلقة ليست كل كلمة قاف؛ القاف '
              'الفصحى [q] الحقيقية تحتاج إعادة تدريب (توكن q غير مدرّب — '
              '0/1.59M) — لا تقريب كاف بعد PATCH 13.')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'rows': rows,
                       'p9_rows': p9_rows, 'p10_rows': p10_rows,
                       'p11_rows': p11_rows, 'p12_rows': p12_rows,
                       'p13_rows': p13_rows,
                       'mode_rows': mode_rows, 'msa_rows': msa_rows}, f,
                      ensure_ascii=False, indent=1)
        if not args.quiet:
            print(f'\nحُفظت النتائج: {args.json}')

    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
