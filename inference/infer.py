#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NileTTS 4h — حزمة استدلال مستقلة (CPU فقط) — Standalone CPU-only inference
============================================================================
سكريبت التوليد المستخرج من نواة "NileTTS 4h Train" (مرحلة generate فقط،
منفصل تمامًا عن حلقة التدريب). يحوّل نصًا عربيًا إلى ملف صوتي WAV عبر:

    نص خام/مشكول → (تشكيل catt_eo اختياري) → ترميز مصري (EGY_TOKEN_MAP)
    → MixerTTS (من checkpoint التدريب) → ميل سبكتروجرام → vocos22.onnx → WAV

مسار المعالجة مطابق حرفيًا لمسار التوليد في نواة التدريب (synth_set /
gen_probe): نفس دوال الترميز، نفس استدعاء model.infer(x, pace, speaker,
emotion=0)، نفس معاملات المُصوِّت (denoise=0.005، 22050Hz، PCM_16).

الاستخدام:
    python infer.py --checkpoint states_79590.pth --text "السلام عليكم" --out out.wav
    python infer.py --text-file input.txt --out out.wav --speaker 1
    python infer.py --text "إِزَّيْك يَا صَاحِبِي" --vocalize never

المتحدثون: 0 = SPEAKER_01 (ذكر) ، 1 = SPEAKER_02 (أنثى)

ملاحظات CPU: كل عمليات torch تُنفَّذ على المعالج صراحة (device='cpu'،
torch.load(map_location='cpu')) ولا يوجد أي استدعاء .cuda() أو دقة نصفية
(amp/half). المُصوِّت والمُشكِّل (onnx) يعملان بمزوّد CPUExecutionProvider فقط.
"""

import argparse
import os
import re
import sys
import time

# --- Windows console safety: never crash on non-UTF8 consoles -------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, 'lib')                          # → tts_arabic
MIXER_REPO_DIR = os.path.join(HERE, 'lib', 'mixer_repo')     # → models.*
CKPT_DIR = os.path.join(HERE, 'checkpoints')
WEIGHTS_DIR = os.path.join(HERE, 'weights')
VOCOS_ONNX = os.path.join(WEIGHTS_DIR, 'vocos22.onnx')

for _p in (LIB_DIR, MIXER_REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# نفس خريطة التوكنز المصرية من نواة التدريب (بدون أي تعديل)
EGY_TOKEN_MAP = {'j': 'v', 'q': '<', '^': 't', '*': 'd'}

# ============================================================================
# إصلاح القاف الجيمية (PATCH 6) — انظر PATCHES.md والتحقيق الكامل في
# validation/test_qaf_pronunciation.py
# ============================================================================
# السبب الجذري المُثبَت: الخريطة أعلاه تحوّل q→'<' بلا شرط في التدريب
# والاستدلال معًا، فالنموذج لم يرَ توكن 'q' إطلاقًا (0 من 1,590,209 توكن
# في corpus التدريب كاملًا). النتيجة: الكلمات التي تُنطق قافها [g]
# (رقم، قانون، القرآن، مقام...) تُقرأ همزة [ʔ] أو مدًّا يشبه الألف.
#
# الإصلاح العام: لهذه الكلمات فقط (معجم هياكل مُثبَت)، تُستبدل ق→ج في
# نص باكوالتير قبل الترميز، فتتولّد التوكن 'v' — وهو التوكن الذي تعلّمه
# النموذج نطقًا [g] من كل جيمات corpus التدريب المصري (ج المصرية = [g]).
# كل ما عداها يبقى على السلوك الأصلي q→'<' (الصحيح للقاف القاهرية [ʔ]).
#
# أمان التغيير: q و j يولّدان بنية توكنز متطابقة تمامًا (الفرق الوحيد
# تلوين الحركات المُطبَقة AA/aa — والتوكنان يتوحدان إلى 'aa' في الترميز)،
# فلا يتغير طول التسلسل ولا مواضع التوكنز، ولا يُمَس النموذج أو أوزانه.
#
# ── PATCH 6b (مراجعة المستخدم 2026-10-01: تدقيق النطق الثابت/المتغير) ──
# قرارات مراجعة كاملة لكل مواضع corpus (work/qaf_review/qaf_audit_report.md):
# 1. الفئة B فقط تبقى في المعجم (نطق ثابت [g] بكل السياقات).
# 2. الفئة C (حقيقة/دقيق/قراءة/قرأ وعائلاتها) حُذفت كاملة وعادت للافتراضي
#    [ʔ]: عائلة حقيقة وحدها 874 سماعًا في التدريب (أعلى عائلة قاف رسمية)
#    — فرض [g] عليها كان يستبدل نطقًا متعلّمًا قويًا بتخمين نصي غير مؤكد.
# 3. حارس مستقل لـ«بقرأ» التقدمية العامة (انظر _QAF_PROGRESSIVE_BAQR):
#    سابقة الباء التقدمية لا تُنزع أبدًا لمطابقة قرأ — خطأ منطق النزع
#    التلقائي — والحارس يعمل بغض النظر عن محتوى المعجم الحالي.
_QAF_DIAC = frozenset('auiFNK~o')       # حركات باكوالتير (تُنزع لهيكل المطابقة)
_QAF_CLITICS = ('w', 'f', 'b', 'l', 'k')  # سوابق اتصال شائعة قبل الكلمة/ال

# هياكل باكوالتير (بلا حركات) لكلمات قاف=[g]. «ال» والسوابق تُنزع تلقائيًا
# عند المطابقة فلا حاجة لإدراج صيغها. '|' = مدّة آ.
# الفئة B فقط (نطق ثابت [g] معتمد من التحقيق والمراجعة الثنائية):
# قياس صوتي (رقم/قانون)، دخائل (قرش/قيراط/قنطار)، أسماء مجمدة (القرآن)،
# ومقام («في المقام الأول» في كل مواضعها). كل قاف أخرى = الافتراضي [ʔ].
QAF_G_SKELETONS = frozenset({
    'rqm', 'rqmp', 'rqmnp', '>rqAm', 'trqym',            # رقم/رقمة/رقمنة/أرقام/ترقيم
    'qr|n', 'qr|ny', 'qr|nyp', 'qrAn', 'qrAny',          # قرآن/قرآني/قرآنية (+رسم بلا مدة)
    'mqAm', 'mqAmAt',                                    # مقام/مقامات
    'qAnwn', 'qAnwny', 'qAnwnyp', 'qwAnyn',            # قانون/قانوني/قانونية/قوانين
    'qr$', 'qrw$',                                       # قرش/قروش
    'qyrAT', 'qrAryT',                                   # قيراط/قراريط
    'qnTAr', 'qnATyr',                                   # قنطار/قناطير
})

# ============================================================================
# PATCH 7 (2026-10-01) — أوضاع نطق القاف + الفئة Q (سجل فصحى/تعليمي)
# ============================================================================
# طلب المستخدم (حالة «معلم الأطفال»): مصطلحات المناهج (رقم، أرقام، قسمة،
# يقسم، تقريب، قياس، قيمة، قاعدة، قانون...) تُنطق بالقاف الفصحى [q] عند
# الشرح التعليمي، بينما عامية الكلام تبقى همزة/جيم — «كلمات بعينها» لا كل
# ما فيه قاف. التوثيق الكامل للأصل والمبررات: validation/qaf_q_list.json
# (81 هيكلًا؛ أمثلة المستخدم + مصطلحات مناهج مصر + إحصاء corpus نيلتس:
# أعلى 60 هيكل قاف كلها عامية بالهمزة، فالقائمة سجلُّ تعليم لا عامية).
#
# قيد النموذج الحالي (from-scratch): توكن q غير مدرّب إطلاقًا (0/1.59M في
# corpus التدريب) — استخدامه مباشرة يُخرج صوتًا غير معرف. لذا تُقرَّب القاف
# الفصحى بأقرب صوت مدرَّب: كاف ثقيلة [k] (وقفة صماء — نفس الطريقة والجهر
# كالقاف، مخرج أقرب قليلًا؛ وتُقرأ كصوت «قاف مشوبة» لا كصوت غريب).
# القاف الأصيلة الحقيقية uvular /q/ تتطلب إعادة تدريب — المشروع المؤجل.
#
# سياسة التعارض مع B (رقم/قانون/مقام...): في الوضع الافتراضي auto يبقى
# قرار B ساريًا دون أي تغيير (قرار معتمد 2026-10-01: «يبقى ساريًا بلا
# تغيير»)؛ وفي وضع qaf (فصحى المصطلحات) تفوز Q بالقاف الفصحى للسجل
# التعليمي. الوضعان منفصلان تمامًا — لا تبادل أثر.
#
# الأوضاع (كلها متاحة من CLI والواجهة — الافتراض auto دائمًا):
#   auto  : السلوك المعتمد الساري — B→[g]، كل ما علاها→[ʔ]
#   qaf   : فصحى المصطلحات — قائمة Q→[q] تقريبًا [k]؛ B غير المتقاطعة→[g]؛
#           الباقي→[ʔ] (ليست كل كلمة فيها قاف — قائمة مغلقة فقط)
#   hamza : فرض شامل — كل قاف→[ʔ] (تجاوز B)
#   g     : فرض شامل — كل قاف→[g] (تجاوز B)؛ حارس بقرأ يظل يعمل دائمًا
QAF_Q_SKELETONS = frozenset({
    '<qlym', '>qAlym', '>qTAr', '>qsAm', '>rqAm',
    'AnqsAm', 'AqtSAd', 'AqtSAdy', 'AstqAmp', "AstqSA'",
    'AstqlAl', "AstqrA'", 'ElAqAt', 'ElAqp', 'Hqwq',
    'TAqAt', 'TAqp', '^qAfp', '^qAfy', 'lqAH',
    'mnTq', 'mqAm', 'mqAmAt', 'mqAwmAt', 'mqAwmp',
    'mqyAs', 'mstqym', 'nqsm', 'qAEdp', 'qAbl',
    'qAnwn', 'qAnwny', 'qAnwnyp', 'qArAt', 'qArp',
    'qA}m', 'qA}mp', 'qDAyA', 'qDyp', 'qTAE',
    'qTAEAt', 'qTb', 'qTbyn', 'qTr', 'qdr',
    'qdrAt', 'qdrp', 'qlb', 'qlwb', 'qnAp',
    'qnwAt', 'qrb', 'qsm', 'qsmp', 'qsmt',
    'qsmyn', 'qwAEd', 'qwAnyn', 'qwY', 'qwp',
    'qyAs', 'qym', 'qymp', 'rqm', 'rqmp',
    'rqmyn', 'tqdym', 'tqdyr', 'tqdyry', 'tqryb',
    'tqrybA', 'tqryby', 'tqrybyp', 'tqsm', 'tqsym',
    'tqys', 'trqym', 'twqyt', 'ynqsm', 'yqsm',
    'yqys',
})

QAF_MODES = ('auto', 'qaf', 'hamza', 'g')

QAF_MODE_DESC = {
    'auto': ('تلقائي معتمد: كلمات B→[g] (رقم/قانون/القرآن...) وكل ما علاها'
             '→[ʔ] — قرار 2026-10-01 الساري بلا تغيير'),
    'qaf': ('فصحى المصطلحات (تعليمي): قائمة مغلقة من 81 هيكلًا (رقم/أرقام/'
            'قسمة/يقسم/تقريب/قياس/قيمة/قاعدة/قانون/قطر/مقام...) → قاف فصحى'
            ' تقريبية [k]؛ الباقي كما في auto — ليست كل كلمة فيها قاف'),
    'hamza': 'فرض شامل: كل قاف→همزة [ʔ] (تجاوز قائمة B)',
    'g': 'فرض شامل: كل قاف→جيم [g] كنطق جيم العامية (تجاوز قائمة B)',
}

# PATCH 6b: الصيغة العامية التقدمية «بقرأ» (وأخواتها بسابقة الباء على قرأ
# بعد و/ف): بقرأ/بتقرأ/بيقرأ/بيتقرأ — كلها [ʔ] العامة دائمًا، ولا تُطبّق
# عليها أي معالجة قاف-جيمية أبدًا، حتى لو أُعيدت قرأ للمعجم مستقبلًا.
# (بدل نطق السابقة: بـ=مضارع تقدمي لا جرّ/تعريف — نزعها خطأ صرفي.)
_QAF_PROGRESSIVE_BAQR = re.compile(r'^[wf]?b(?:yt|y|t)?qr>')


def _qaf_skel_variants(s):
    """صيغ المطابقة المحتملة لهيكل الكلمة: كما هو، أو بعد نزع «ال»،
    أو سابقة اتصال (و/ف/ب/ل/ك) [+ «ال»]."""
    out = {s}
    if s.startswith('Al') and len(s) > 3:
        out.add(s[2:])
    for c in _QAF_CLITICS:
        if s.startswith(c) and len(s) > 2:
            out.add(s[1:])
            if s[1:3] == 'Al' and len(s) > 4:
                out.add(s[3:])
    return out


def fix_qaf(buck, mode='auto', dialect='egy'):
    """PATCH 7 — التحكم الكامل بنطق القاف على مستوى باكوالتير حسب الوضع:

    auto  (مصري): B→ج [g] (PATCH 6 الساري)، والباقي q→'<' عبر EGY_TOKEN_MAP
    qaf   (مصري): قائمة Q→ك [k] (قاف فصحى تقريبية — أولاً قبل B للمتقاطع)،
                  ثم B غير المتقاطعة→ج، والباقي همزة
    hamza (مصري): لا تعديل — كل قاف تذهب لخريطة '<' [ʔ] (تجاوز B)
    g     (مصري): كل قاف→ج [g] شاملًا (تجاوز B) — حارس بقرأ يظل يعمل

    فصحى (msa): القاف الفصحى للجميع تقريبًا [k] في auto/qaf (كانت توكن q
    غير مدرّب — إصلاح خلل قديم)، و'<' عند فرض الهمزة، و'j' عند فرض الجيم.

    لا يمس حارس «بقرأ» التقدمية في أي وضع (PATCH 6b — مستقل دائمًا).
    """
    if 'q' not in buck:
        return buck
    if dialect == 'msa':
        if mode == 'hamza':
            return buck.replace('q', '<')
        if mode == 'g':
            return buck.replace('q', 'j')
        return buck.replace('q', 'k')      # auto/qaf: قاف فصحى تقريبية
    if mode == 'hamza':
        return buck                         # EGY_TOKEN_MAP: q→'<' تلقائيًا
    words = buck.split(' ')
    hit = False
    for i, w in enumerate(words):
        if 'q' not in w:
            continue
        skel = ''.join(c for c in w if c not in _QAF_DIAC)
        if _QAF_PROGRESSIVE_BAQR.match(skel):
            continue                        # بقرأ التقدمية = [ʔ] دائمًا
        if mode == 'g':
            words[i] = w.replace('q', 'j')
            hit = True
            continue
        variants = _qaf_skel_variants(skel)
        if mode == 'qaf' and any(v in QAF_Q_SKELETONS for v in variants):
            words[i] = w.replace('q', 'k')  # فصحى المصطلحات — Q قبل B
            hit = True
        elif any(v in QAF_G_SKELETONS for v in variants):
            words[i] = w.replace('q', 'j')  # B المعتمدة
            hit = True
    return ' '.join(words) if hit else buck


def fix_qaf_g(buck):
    """اسم قِدَم PATCH 6 — الآن غلاف ثابت للوضع الافتراضي auto (مصري).
    محفوظة للتوافق مع الاختبارات والتوثيق القائم."""
    return fix_qaf(buck, mode='auto', dialect='egy')

# نفس تجاوزات الإعدادات المستخدمة في التدريب — تُستخدم فقط كاحتياط إذا
# كان الـcheckpoint قديمًا لا يخزّن net_config داخله. الأصل: قراءة
# net_config من الـcheckpoint نفسه (أدق وأكثر موثوقية).
NET_CONFIG_FALLBACK = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}

# الحروف العربية + الحركات + المسافة فقط (يطابق ما تراه النموذج في التدريب:
# catt كان يقص كل ما ليس حرفًا عربيًا — ترقيم وأرقام ولاتيني — قبل التشكيل)
_AR_LETTERS = re.compile(r'[\u0621-\u063A\u0641-\u064A]')
_DIACRITICS = '\u064B-\u0652'          # تنوين + حركات + شدة + سكون
_SAFE_CHAR = re.compile(r'[^\u0621-\u063A\u0641-\u064A' + _DIACRITICS + r' ]')
_TATWEEL = '\u0640'
TRAIN_MAX_TOKENS = 160                  # سقف التدريب الفعلي للنموذج


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1) تحضير النص — يطابق مسار نواة التدريب
# ============================================================================
def strip_tatweel(text):
    return text.replace(_TATWEEL, '')


def keep_arabic_only(text):
    """إبقاء الحروف العربية والحركات والمسافات فقط، مع توحيد المسافات.

    هذا مطابق لسلوك بيانات التدريب: catt_eo كان يطبق remove_non_arabic
    (قص الترقيم والأرقام واللاتيني والتطويل) قبل التشكيل، فلم يرَ النموذج
    أثناء التدريب أي رمز ترقيم/رقم إطلاقًا (صفوف embedding الخاصة بها غير
    مدرَّبة). أي نص مُدخل يُنظَّف بنفس الطريقة هنا سلفًا."""
    text = strip_tatweel(text)
    text = _SAFE_CHAR.sub(' ', text)
    return ' '.join(text.split())


def diacritic_density(text):
    letters = _AR_LETTERS.findall(text)
    if not letters:
        return 0.0, 0
    n_d = sum(1 for c in text if '\u064B' <= c <= '\u0652')
    return n_d / len(letters), n_d


def get_tokenizer(qaf_mode='auto'):
    """نفس دالة get_tokenizer في نواة التدريب حرفيًا + إصلاح القاف الجيمية
    (PATCH 6) على مسار اللهجة المصرية فقط + أوضاع نطق القاف (PATCH 7).

    toks_ms تبقى مرجعًا خامًا (توكن q في مواضعه) لتوطين مواضع القاف —
    مسار التركيب الفصحى الفعلي يمر عبر get_msa_synthesis_tokens."""
    from tts_arabic.text import (
        arabic_to_buckwalter, tokens_to_ids, phonemes_to_tokens,
        buckwalter_to_phonemes)

    def toks_ms(text):
        return phonemes_to_tokens(buckwalter_to_phonemes(arabic_to_buckwalter(text)))

    def toks_egy(text):
        buck = fix_qaf(arabic_to_buckwalter(text), qaf_mode, 'egy')
        toks = phonemes_to_tokens(buckwalter_to_phonemes(buck))
        return [EGY_TOKEN_MAP.get(t, t) for t in toks]

    return toks_ms, toks_egy, tokens_to_ids


def get_msa_synthesis_tokens(text, qaf_mode='auto'):
    """PATCH 7 — توكنز مسار الفصحى للتركيب (ليست المرجع الخام).

    خلل قديم مُصلَح: كان toks_ms يمرر توكن q للنموذج وهو غير مدرّب إطلاقًا
    (0 من 1,590,209 توكن في corpus التدريب — الخريطة حولته في التدريب)،
    فكان أي نص فصحى فيه قاف يولّد صوتًا غير معرف. الآن: القاف الفصيحة
    للجميع تقريبًا [k] (أو فرض '<'/'j' حسب الوضع)."""
    from tts_arabic.text import (
        arabic_to_buckwalter, phonemes_to_tokens, buckwalter_to_phonemes)
    buck = fix_qaf(arabic_to_buckwalter(text), qaf_mode, 'msa')
    return phonemes_to_tokens(buckwalter_to_phonemes(buck))


# ============================================================================
# 2) المُشكِّل النصي catt_eo (onnx — CPU فقط) — للتشكيل التلقائي
# + 2-ب) PATCH 8 — طبقة التصحيح الجزئي det_tashkeel (خيار إضافي، غير افتراضي)
# ============================================================================
# قرار المستخدم (2026-10-01) بعد الاطلاع على الأرقام: تفعيل محدود كخيار
# إضافي معلَّم فقط — ليس الخيار الافتراضي في أي مكان، وليس مصدر بيانات
# للتدريب ولا بديلًا نهائيًا عن مسار tashkeel-ai. استخدامه المقصود: تحسين
# سريع مؤقت لنص بلا تشكيل يدوي/GLM متاح.
#
# الأرقام الصادقة المعروضة مع الخيار أينما ظهر:
#   45.2% تطابق كلمة-بكلمة كامل الحركات (عينة دخان معزولة — المقياس
#         الأساسي) | 57.9% (عينة الـ50 متحيزة — رقم ثانوي موضَّح؛ ساهمت في
#         اشتقاق قواعد §ز فتبدو متفائلة) | البوابات الميكانيكية وحدها ~96%
#   الفجوة الباقية داخلية بامتياز (حركات داخل الكلمة تحتاج فهم سياق).
_catt = None


def catt_vocalize(text):
    """تشكيل تلقائي عبر catt_eo.onnx (المُضمَّن في lib/tts_arabic/data)."""
    global _catt
    if _catt is None:
        import onnxruntime as ort
        from tts_arabic.vocalizer.models.core import get_model
        # get_model يقرأ الملف من داخل الحزمة المحلية — لا أي تنزيل
        _catt = get_model('catt_eo')
    return _catt.predict(text)


DET_LABEL_AR = ('تصحيح جزئي: أرقام وترقيم وقوائم مغلقة ونهايات فقط — '
                'ليست بديلًا عن تشكيل tashkeel-ai')
DET_NUMBERS_AR = ('تطابق كلمة-بكلمة كامل الحركات: 45.2% (عينة معزولة — '
                  'المقياس الأساسي) · 57.9% (عينة متحيزة — رقم ثانوي)')

_det_safety_cache = {}


def _det_safety_fn(qaf_mode):
    """دالة فحص الأمان (§ل-5) — تُبنى مرة لكل وضع قاف وتُخزَّن."""
    if qaf_mode not in _det_safety_cache:
        from safety_check import make_safety_fn
        _, toks_egy, _ = get_tokenizer(qaf_mode)
        _det_safety_cache[qaf_mode] = make_safety_fn(toks_egy)
    return _det_safety_cache[qaf_mode]


def apply_det_partial(raw, catt_out, qaf_mode='auto'):
    """تطبيق طبقة التصحيح الجزئي على مخرج catt_eo الخام.

    يعيد النص المُحسَّن. سجل التقدم/الأعلام يُطبع في الكونسول فقط — الملصق
    والأرقام تُعرض في الواجهة (web/index.html) وفي مستندات الحزمة."""
    from det_tashkeel import vocalize_partial
    res = vocalize_partial(raw, catt_out, safety_fn=_det_safety_fn(qaf_mode))
    out = ' '.join(res['out'].split())
    fixes = res.get('fixes') or {}
    log(f'[det] التصحيح الجزئي مفعّل: {res["label_ar"]}')
    log(f'[det] {DET_NUMBERS_AR}')
    if fixes:
        log(f'[det] إصلاحات: ' + ' · '.join(
            f'{k}={v}' for k, v in fixes.items() if v))
    if res.get('flags'):
        log(f'[det] أعلام: ' + ', '.join(map(str, res['flags'][:6])))
    return out


# ============================================================================
# 3) نموذج MixerTTS من الـcheckpoint (torch — CPU فقط)
# ============================================================================
def load_model(checkpoint_path):
    """بناء النموذج من net_config المخزّن داخل الـcheckpoint وتحميل الأوزان.

    device='cpu' صراحة في كل خطوة (torch.load + .to())."""
    import torch
    from models.mixer_tts.mixer_tts import MixerTTSModel

    st = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(st, dict) or 'model' not in st:
        raise SystemExit(
            f'[خطأ] الملف {checkpoint_path} ليس checkpoint صالحًا لنموذج '
            'MixerTTS (لا يحتوي مفتاح "model").')

    net_config = dict(st.get('net_config') or {})
    if 'num_tokens' not in net_config:            # checkpoint قديم جدًا
        net_config.update(NET_CONFIG_FALLBACK)
    model = MixerTTSModel(**net_config)
    model.load_state_dict(st['model'], strict=True)
    model.eval()
    model.to('cpu')                                # صراحة — CPU فقط

    pitch_mean = st.get('pitch_mean')
    pitch_std = st.get('pitch_std')
    if pitch_mean is not None and pitch_std:
        model.pitch_mean = float(pitch_mean)
        model.pitch_std = float(pitch_std)
    it = st.get('iter', '?')
    return model, it


# ============================================================================
# 4) المُصوِّت vocos22 (onnx — CPU فقط) — مطابق لـ mel_to_wav في النواة
# ============================================================================
_vocos = None
def vocos_session():
    global _vocos
    if _vocos is None:
        import onnxruntime as ort
        if not os.path.exists(VOCOS_ONNX):
            raise SystemExit(f'[خطأ] ملف المُصوِّت غير موجود: {VOCOS_ONNX}')
        _vocos = ort.InferenceSession(
            VOCOS_ONNX, providers=['CPUExecutionProvider'])
    return _vocos


def mel_to_wav(mel_80_T, denoise=0.005):
    """نفس دالة mel_to_wav في نواة التدريب (بدون أي تعديل)."""
    import numpy as np
    sess = vocos_session()
    wave = sess.run(None, {
        'mel_spec': mel_80_T[None].astype('float32'),
        'denoise': np.array([denoise], dtype='float32'),
    })[0].astype('float32')[0]
    return 0.9 * wave / (np.abs(wave).max() + 1e-5)


# ============================================================================
# 5) خط التوليد الكامل
# ============================================================================
def resolve_checkpoint(arg_value):
    """يقرأ الـcheckpoint من مجلد checkpoints/ — بلا أي مسار Kaggle."""
    if arg_value:
        cand = [arg_value,
                os.path.join(CKPT_DIR, arg_value),
                os.path.join(CKPT_DIR, os.path.basename(arg_value))]
        for c in cand:
            if os.path.exists(c):
                return os.path.abspath(c)
        raise SystemExit(
            f'[خطأ] لم أجد الـcheckpoint: {arg_value}\n'
            f'  البحث شمل: مجلد التشغيل الحالي + {CKPT_DIR}\n'
            '  ضع ملفات states_*.pth (من مخرجات Kaggle) داخل مجلد checkpoints/.')
    # بدون وسيط: خذ أحدث states_*.pth في checkpoints/
    import glob
    snaps = glob.glob(os.path.join(CKPT_DIR, 'states_*.pth'))
    rolling = os.path.join(CKPT_DIR, 'states.pth')
    if not snaps and os.path.exists(rolling):
        return os.path.abspath(rolling)
    if not snaps:
        raise SystemExit(
            f'[خطأ] لا يوجد أي checkpoint في {CKPT_DIR}\n'
            '  ضع ملف states_79590.pth (أو أي states_*.pth) هناك، أو مرّر '
            '--checkpoint بمسار صريح.')
    it_of = lambda p: int(re.search(r'states_(\d+)\.pth$', p).group(1))
    return max(snaps, key=it_of)


def prepare_text(raw_text, vocalize_mode, dialect, qaf_mode='auto',
                 det_partial=False):
    """تنظيف → (تشكيل اختياري [+ طبقة التصحيح الجزئي إن فُعِلت]) → تنظيف نهائي.

    PATCH 7: qaf_mode — وضع نطق القاف (auto/qaf/hamza/g — الافتراضي auto).
    PATCH 8: det_partial — تفعيل طبقة التصحيح الجزئي فوق مخرج catt_eo
    (خيار إضافي معلَّل غير افتراضي — انظر DET_LABEL_AR/DET_NUMBERS_AR).
    يرجع (نص المعالجة، تم_التشكيل)."""
    text = ' '.join(raw_text.split())
    if not _AR_LETTERS.search(keep_arabic_only(text)):
        raise SystemExit('[خطأ] النص لا يحتوي حروفًا عربية.')

    density, _ = diacritic_density(keep_arabic_only(text))
    do_vocalize = (vocalize_mode == 'always' or
                   (vocalize_mode == 'auto' and density < 0.30))

    if do_vocalize:
        voc = ' '.join(catt_vocalize(text).split())   # catt يقص غير العربي بنفسه
        if det_partial:
            voc = apply_det_partial(text, voc, qaf_mode)
        return voc, True
    else:
        text = keep_arabic_only(text)      # الحفاظ على التشكيل الموجود
        if not _AR_LETTERS.search(text):
            raise SystemExit('[خطأ] النص لا يحتوي حروفًا عربية بعد التنظيف.')
        return text, False


def synthesize(model, text, dialect, speaker, pace, out_path, denoise,
               qaf_mode='auto', verbose=False):
    """من نص مُعالَج إلى ملف WAV. يعيد (عدد التوكنز، مدة الصوت بالثواني).

    PATCH 7: dialect='msa' يمر الآن عبر get_msa_synthesis_tokens (قاف فصحى
    تقريبية [k] — إصلاح توكن q غير المدرّب) بدل toks_ms الخام."""
    import torch
    import soundfile as sf

    toks_ms, toks_egy, ids_of = get_tokenizer(qaf_mode)
    if dialect == 'msa':
        toks = get_msa_synthesis_tokens(text, qaf_mode)
    else:
        toks = toks_egy(text)
    ids = ids_of(toks)
    if len(ids) < 2:
        raise SystemExit('[خطأ] النص قصير جدًا بعد الترميز (توكنز < 2).')
    if len(ids) > TRAIN_MAX_TOKENS:
        log(f'[تحذير] عدد التوكنز {len(ids)} يتجاوز سقف التدريب '
            f'({TRAIN_MAX_TOKENS}). الجودة قد تتأثر — يُفضّل تقسيم النص.')

    if verbose:
        log(f'[tokens] {len(ids)}: {toks}')

    x = torch.LongTensor([ids])            # CPU (الافتراضي) — بلا أي .cuda()
    t0 = time.time()
    with torch.inference_mode():
        mel = model.infer(x, pace=pace, speaker=speaker, emotion=0)
    m = mel.transpose(1, 2)[0].cpu().numpy()          # [80, T]
    t_mel = time.time() - t0

    wave = mel_to_wav(m, denoise=denoise)
    t_all = time.time() - t0

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, wave, 22050, subtype='PCM_16')

    n_secs = len(wave) / 22050.0
    if verbose:
        log(f'[timing] mel: {t_mel:.2f}s | total: {t_all:.2f}s')
    return len(ids), n_secs


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='NileTTS 4h — توليد صوت عربي من نص (CPU فقط)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='أمثلة:\n'
               '  python infer.py --checkpoint states_79590.pth '
               '--text "السلام عليكم" --out out.wav\n'
               '  python infer.py --text-file article.txt --out out.wav '
               '--speaker 1 --pace 0.9\n')
    ap.add_argument('--checkpoint', default=None,
                    help='اسم أو مسار ملف checkpoint (يُبحث في checkpoints/ '
                         'أولًا). بدون قيمة: يُختار أحدث states_*.pth تلقائيًا.')
    ap.add_argument('--text', default=None, help='النص العربي مباشرة')
    ap.add_argument('--text-file', default=None,
                    help='قراءة النص من ملف (UTF-8)')
    ap.add_argument('--out', default='out.wav', help='مسار ملف WAV الناتج')
    ap.add_argument('--speaker', type=int, default=0,
                    help='0 = SPEAKER_01 ذكر (افتراضي) ، 1 = SPEAKER_02 أنثى')
    ap.add_argument('--dialect', choices=['egy', 'msa'], default='egy',
                    help='egy: ترميز مصري بخريطة EGY_TOKEN_MAP (افتراضي) | '
                         'msa: ترميز فصحى')
    ap.add_argument('--pace', type=float, default=1.0,
                    help='سرعة الكلام (1.0 = طبيعي؛ أقل = أبطأ)')
    ap.add_argument('--vocalize', choices=['auto', 'always', 'never'],
                    default='auto',
                    help='auto: تشكيل catt_eo إذا كان النص غير مشكول | '
                         'always: دائمًا | never: استخدام التشكيل الموجود')
    ap.add_argument('--qaf', choices=list(QAF_MODES), default='auto',
                    help='نطق القاف (PATCH 7 — الافتراضي auto): auto = '
                         'السلوك المعتمد (B→[g] والباقي همزة) | qaf = فصحى '
                         'المصطلحات التعليمية (قائمة مغلقة → قاف فصحى '
                         'تقريبية [k]) | hamza = فرض الهمزة شاملًا | '
                         'g = فرض الجيم شاملًا')
    ap.add_argument('--det-vocalize', action='store_true',
                    help='تفعيل طبقة التصحيح الجزئي فوق تشكيل catt_eo '
                         '(PATCH 8 — غير مفعّل افتراضيًا): تصحيح جزئي: '
                         'أرقام وترقيم وقوائم مغلقة ونهايات فقط — ليست '
                         'بديلًا عن تشكيل tashkeel-ai')
    ap.add_argument('--denoise', type=float, default=0.005,
                    help='معامل تنقية المُصوِّت vocos (افتراضي 0.005)')
    ap.add_argument('--threads', type=int, default=None,
                    help='عدد خيوط CPU لـ torch (الافتراضي: تلقائي)')
    ap.add_argument('--verbose', action='store_true', help='تفاصيل إضافية')
    args = ap.parse_args()

    # ---------------- CPU-only enforcement --------------------------------
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    # لا نستخدم CUDA إطلاقًا حتى لو كان متاحًا — كل شيء على المعالج.
    if torch.cuda.is_available():
        log('[ملاحظة] تم اكتشاف GPU لكن الحزمة مضبوطة على CPU فقط '
            '(device="cpu") — سيُستخدم المعالج.')

    # ---------------- input text -------------------------------------------
    if args.text and args.text_file:
        raise SystemExit('[خطأ] استخدم --text أو --text-file وليس كليهما.')
    if args.text_file:
        if not os.path.exists(args.text_file):
            raise SystemExit(f'[خطأ] ملف النص غير موجود: {args.text_file}')
        with open(args.text_file, encoding='utf-8') as f:
            raw = f.read().strip()
    elif args.text:
        raw = args.text.strip()
    else:
        raise SystemExit('[خطأ] مرّر النص عبر --text "..." أو --text-file.')

    ckpt = resolve_checkpoint(args.checkpoint)
    log(f'[1/4] checkpoint: {ckpt}')

    # ---------------- model -------------------------------------------------
    model, it = load_model(ckpt)
    log(f'[2/4] النموذج جاهز (iter {it}) — CPU')

    # ---------------- text pipeline -----------------------------------------
    text, did_vocalize = prepare_text(raw, args.vocalize, args.dialect,
                                      args.qaf, args.det_vocalize)
    tag = 'تشكيل تلقائي catt_eo' if did_vocalize else 'تشكيل النص كما هو'
    if did_vocalize and args.det_vocalize:
        tag += ' + طبقة التصحيح الجزئي (خيار معلَّل — ليس بديلًا عن tashkeel-ai)'
    log(f'[3/4] معالجة النص ({tag} | قاف: {args.qaf}):')
    log(f'      {text}')

    # ---------------- synth ---------------------------------------------------
    n_tokens, n_secs = synthesize(
        model, text, args.dialect, args.speaker, args.pace, args.out,
        args.denoise, qaf_mode=args.qaf, verbose=args.verbose)
    log(f'[4/4] تم: {args.out} — {n_tokens} توكن | {n_secs:.1f} ثانية صوت '
        f'| متحدث {args.speaker} | لهجة {args.dialect} | قاف {args.qaf} '
        f'| pace {args.pace}')


if __name__ == '__main__':
    main()
