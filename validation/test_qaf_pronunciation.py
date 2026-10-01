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


def trace(raw, infer, a2b, b2p, p2t, ids_of, egy_map, vocalize='auto'):
    """يتبع النص عبر المسار الإنتاجي: تنظيف → تشكيل catt → ترميز → معرفات.

    الترميز المصري يُستخرج من infer.toks_egy نفسها (المكان الفعلي لـPATCH 6)،
    والتوكنز الفصحى من infer.toks_ms لتحديد مواضع q المرجعية. بنية الطولين
    متطابقة بالتصميم (فرق q/j لا يغيّر بنية التوكنز)."""
    text = ' '.join(raw.split())
    clean = infer.keep_arabic_only(text)
    density, _ = infer.diacritic_density(clean)
    do_voc = (vocalize == 'always' or
              (vocalize == 'auto' and density < 0.30))
    if do_voc:
        processed = ' '.join(infer.catt_vocalize(clean).split())
    else:
        processed = clean
    toks_ms_fn, toks_egy_fn, _ids = infer.get_tokenizer()
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

    total = len(rows)
    by_class = {}
    for c in 'ABC':
        sub = [r for r in rows if r['class'] == c]
        by_class[c] = {'n': len(sub),
                       'fail': sum(r['verdict'] == 'FAIL' for r in sub)}
    summary = {
        'total': total, 'pass': n_pass, 'fail': n_fail, 'suspicious': n_susp,
        'by_class': by_class,
        'policy': ("A→'<' (قاهري [ʔ])، B→'v' ([g] ثابت)، "
                   "C→'<' (متغير → افتراضي [ʔ] — PATCH 6b)"),
        'policy_decision': (
            "مراجعة المستخدم 2026-10-01: الفئة B فقط ثابتة [g]؛ حقيقة/دقيق/"
            "قراءة/قرأ وعائلاتها عادت للافتراضي [ʔ] حفاظًا على السلوك المتعلّم "
            "(874 سماعًا لعائلة حقيقة)؛ حارس مستقل لصيغة «بقرأ» التقدمية"),
        'root_cause': ("EGY_TOKEN_MAP q→'<' غير مشروطة في toks_egy "
                       "(infer.py + نواة التدريب) — عولجت في الاستدلال بـ"
                       "PATCH 6 (QAF_G_SKELETONS + fix_qaf_g في infer.py)"),
    }
    if not args.quiet:
        print('\n' + '=' * 76)
        print(f"الملخص: {n_pass}/{total} PASS | {n_fail} FAIL | {n_susp} SUSPICIOUS")
        for c in 'ABC':
            s = by_class[c]
            print(f"  الفئة {c} ({CLASS_DESC[c][:38]}...): "
                  f"{s['n'] - s['fail']}/{s['n']} PASS")
        print('\nملاحظة: فئة C = نطق متغير — عادت للافتراضي [ʔ] بقرار مراجعة '
              'المستخدم (2026-10-01)، وتشمل حارس «بقرأ» التقدمية (PATCH 6b).')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'rows': rows}, f,
                      ensure_ascii=False, indent=1)
        if not args.quiet:
            print(f'\nحُفظت النتائج: {args.json}')

    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
