#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eiqaz TTS — حزمة استدلال مستقلة (CPU فقط) — مواءمة كاملة مع checkpoint
states_cont_180516.pth (التدريب النهائي: eiqaz-train-v1 ثم eiqaz-train-cont-v1)
============================================================================

مسار التوليد (مواءَم حرفيًا مع معالجة نصوص التدريب النهائي):

    نص خام → eqz_text.normalize_text
        (أرقام → كلمات عربية منطوقة · ، → , و ؟ → ? · نقحرة لاتينية)
    → تشكيل تلقائي (وضع مصري: catt_eo + القواعد الصارمة det_tashkeel
       من دليل التشكيل المصري v2.1 · وضع فصحى: catt_eo فقط · يدوي: كما هو)
    → علامات القاف {ق}/{ج}/{ء} تُحلل عند الترميز (نظام Eiqaz محفوظ)
    → eqz_tokens (سياسة القاف v1 نفسها التي درّب النموذج النهائي بها:
       FORCED_Q → 'q' قاف أصيلة · FORCED_G → 'v' جيم مصرية [g] ·
       الافتراضي → '<' همزة · EGY_SOUND_MAP {j→v, ^→t, *→d} بلا q)
    → MixerTTS (states_cont_180516.pth) → ميل → vocos22.onnx → WAV 22050Hz

لماذا هذه المواءمة (تشخيص جنائي 2026-10-06):
  * EGY_TOKEN_MAP القديمة {'j':'v','q':'<',...} كانت تحوّل ق→همزة قسرًا
    بينما التدريب النهائي درّب توكن 'q' خامًا (396,773 تعرضًا؛ haqiqa
    q/g/h متميزة) — أُزيلت الخريطة القديمة وحلّ محلها eqz_tokens.
  * keep_arabic_only القديمة كانت تحذف الترقيم كله قبل النموذج بينما
    التدريب رأى . , ? ! توكنات حقيقية (ids 5-8) — صارت محفوظة.
  * الأرقام كانت تختفي صامتة (572 نصًا) — صارت كلمات منطوقة.
  * التشكيل التلقائي العام (catt فقط) صار مصريًا عبر القواعد الصارمة
    الموجودة (det_tashkeel — دليل v2.1: نهايات/تنوين قائمة مغلقة/
    ال التعريف/هو-هي/عبارات مجمدة).

المتحدثون (خريطة speaker_map_v1 من index.json التدريب):
    0 = egy_SPEAKER_01 ذكر مصري (افتراضي)
    1 = egy_SPEAKER_02 أنثى مصرية
    2-6 = متحدثو MSA (ClArTTS/Common Voice) — للتقييم المتقدم
    7-15 = غير مدرّبين (drift=0.0 منذ التهيئة) — ممنوعون

الاستخدام:
    python infer.py --text "السلام عليكم" --out out.wav
    python infer.py --text "عندي 5 كتب." --speaker 1
    python infer.py --text "قسّمنا قطعة{ق} على رقم{ج} وقال{ء} شكرًا"
    python infer.py --text-file article.txt --diacritize fusha --dialect msa

ملاحظات CPU: كل عمليات torch على المعالج صراحة (device='cpu') ولا يوجد
أي .cuda() أو دقة نصفية. المُصوِّت والمُشكِّل (onnx) CPUExecutionProvider.
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

# ── مصدر الحقيقة الوحيد للمعالجة القبلية والترميز (توحيد Web/CLI/API) ─────
import eqz_text                       # التطبيع الموحد (أرقام/ترقيم/نقحرة)
import eqz_tokens                     # سياسة توكن التدريب النهائي (نسخة حرفية)

# ── checkpoint الإنتاج المعتمد — الافتراض دائمًا، بلا fallback صامت ──────
DEFAULT_CHECKPOINT = 'states_cont_180516.pth'

SPEAKERS = {
    0: 'ذكر مصري (egy_SPEAKER_01)',
    1: 'أنثى مصرية (egy_SPEAKER_02)',
    2: 'MSA ذكر كلاسيكي (clartts)', 3: 'MSA أنثى (cv)',
    4: 'MSA', 5: 'MSA', 6: 'MSA',
}
EGYPTIAN_SPEAKERS = (0, 1)
ALLOWED_SPEAKERS = tuple(SPEAKERS)     # 0-6 مدرّبون فقط (7-15 drift=0)

DIACRITIZE_MODES = ('auto', 'egyptian', 'fusha', 'manual')

# نفس تجاوزات الإعدادات المستخدمة في التدريب — احتياط لـcheckpoint قديم جدًا
# لا يخزّن net_config. الأصل: قراءة net_config من الـcheckpoint نفسه.
NET_CONFIG_FALLBACK = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}

_AR_LETTERS = re.compile(r'[\u0621-\u063A\u0641-\u064A]')
_DIACRITICS = '\u064B-\u0652'          # تنوين + حركات + شدة + سكون
TRAIN_MAX_TOKENS = 160                  # سقف التدريب الفعلي للنموذج


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1) الترميز — eqz_tokens (سياسة التدريب النهائي نفسها — نسخة حرفية)
# ============================================================================
_TOK_CACHE = {}


def get_tokenizer():
    """(toks_ms, toks_egy, ids_of) — من eqz_tokens مباشرة.

    toks_egy: علامات {ق}/{ج}/{ء} + fix_qaf_v1 (FORCED_Q/FORCED_G/همزة
    افتراضية) + EGY_SOUND_MAP {'j':'v','^':'t','*':'d'} — بلا 'q' في
    الخريطة: القاف الأصيلة تمر خامًا كما درّبها eiqaz-train-cont-v1.
    toks_ms: الفصحى — q خام مدربة، ج='j' [dʒ]، والعلامات تعمل."""
    if 'fns' not in _TOK_CACHE:
        _TOK_CACHE['fns'] = eqz_tokens.get_tokenizers([LIB_DIR])
    toks_ms, toks_egy, ids_of, _parse = _TOK_CACHE['fns']
    return toks_ms, toks_egy, ids_of


def tokenize(text, dialect='egy'):
    """توكنات النص (بعد أي معالجة) وفق سياسة التدريب. واجهة مختصرة."""
    toks_ms, toks_egy, _ = get_tokenizer()
    return toks_ms(text) if dialect == 'msa' else toks_egy(text)


def count_tokens(text, dialect='egy'):
    """عدد توكنات النص الخام بعد التطبيع الموحد (لتقدير أحجام المقاطع).

    للنص غير المشكول: تقدير ما بعد التشكيل بعامل 1.5 (قياس corpus التدريب
    نفسه: وسيط التوكنات المشكولة/غير المشكولة = 1.488) — وإلا فالمقطع
    الذي يبدو 160 توكن سيتضخم إلى ~240 بعد التشكيل فيتجاوز سقف التدريب."""
    _, _, ids_of = get_tokenizer()
    norm = eqz_text.normalize_text(text, dialect)
    cleaned = eqz_text.keep_arabic_and_punct(norm)
    if not _AR_LETTERS.search(cleaned):
        return 0
    density, _ = diacritic_density(cleaned)
    try:
        n = len(ids_of(tokenize(cleaned, dialect)))
    except Exception:                                   # noqa: BLE001
        return 0
    if density < 0.30:
        n = round(n * 1.5)
    return n


# ============================================================================
# 2) المُشكِّل النصي catt_eo + القواعد الصارمة det_tashkeel
# ============================================================================
_catt = None


def catt_vocalize(text):
    """تشكيل تلقائي عبر catt_eo.onnx (المُضمَّن في lib/tts_arabic/data).

    ملاحظة: catt يقص الترقيم والأرقام واللاتيني (remove_non_arabic) —
    الترقيم يُستعاد بعدها عبر محاذاة det (مصري) أو restore_punctuation
    (فصحى)؛ الأرقام واللاتيني عولجا قبلها في eqz_text.normalize_text."""
    global _catt
    if _catt is None:
        from tts_arabic.vocalizer.models.core import get_model
        # get_model يقرأ الملف من داخل الحزمة المحلية — لا أي تنزيل
        _catt = get_model('catt_eo')
    return _catt.predict(text)


DET_LABEL_AR = ('تصحيح جزئي: أرقام وترقيم وقوائم مغلقة ونهايات فقط — '
                'ليست بديلًا عن تشكيل tashkeel-ai')
DET_NUMBERS_AR = ('تطابق كلمة-بكلمة كامل الحركات: 45.2% (عينة معزولة — '
                  'المقياس الأساسي) · 57.9% (عينة متحيزة — رقم ثانوي)')

_det_safety_cache = {}


def _det_safety_fn():
    """دالة فحص الأمان الصوتي (§ل-5) — تُبنى مرة وتُخزَّن (نفس toks_egy)."""
    if 'fn' not in _det_safety_cache:
        from safety_check import make_safety_fn
        _, toks_egy, _ = get_tokenizer()
        _det_safety_cache['fn'] = make_safety_fn(toks_egy)
    return _det_safety_cache['fn']


def apply_det_partial(raw, catt_out):
    """القواعد الصارمة الموجودة (det_tashkeel — دليل التشكيل المصري v2.1)
    فوق مخرج catt_eo: نهايات/تنوين قائمة مغلقة/ال التعريف/هو-هي/عبارات
    مجمدة + استرجاع حروف الخام والترقيم بالمحاذاة."""
    from det_tashkeel import vocalize_partial
    res = vocalize_partial(raw, catt_out, safety_fn=_det_safety_fn())
    out = ' '.join(res['out'].split())
    fixes = res.get('fixes') or {}
    log(f'[det] القواعد الصارمة مفعّلة: {res["label_ar"]}')
    if fixes:
        applied = {k: v for k, v in fixes.items() if v}
        if applied:
            log(f'[det] إصلاحات: ' + ' · '.join(
                f'{k}={v}' for k, v in applied.items()))
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


def mel_to_wav(mel_80_T, denoise=0.005, normalize=True):
    """نفس دالة mel_to_wav في نواة التدريب.

    normalize=True (CLI أحادي المقطع): تطبيع ذروة 0.9 لكل موجة.
    normalize=False (webapp متعدد المقاطع): الخام — تعادل الجهارة وتطبيع
    الذروة الواحد يتم في webapp._unify_chunks_tone على النص كاملاً."""
    import numpy as np
    sess = vocos_session()
    wave = sess.run(None, {
        'mel_spec': mel_80_T[None].astype('float32'),
        'denoise': np.array([denoise], dtype='float32'),
    })[0].astype('float32')[0]
    if normalize:
        return 0.9 * wave / (np.abs(wave).max() + 1e-5)
    return wave


# ============================================================================
# 5) checkpoint — الافتراض المعتمد states_cont_180516.pth (بلا fallback)
# ============================================================================
def resolve_checkpoint(arg_value=None):
    """يقرر checkpoint التوليد.

    بلا وسيط: states_cont_180516.pth (checkpoint الإنتاج المعتمد المضمّن
    في المستودع) — وإن غاب فخطأ صريح (لا fallback صامت لملف قديم).
    بوسيط: اسم/مسار يُبحث في مجلد التشغيل وcheckpoints/."""
    pinned = os.path.join(CKPT_DIR, DEFAULT_CHECKPOINT)
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
            f'  checkpoint الإنتاج الافتراضي: {DEFAULT_CHECKPOINT} '
            '(يأتي مع المستودع داخل checkpoints/).')
    if os.path.exists(pinned):
        return os.path.abspath(pinned)
    raise SystemExit(
        f'[خطأ] checkpoint الإنتاج {DEFAULT_CHECKPOINT} غير موجود في '
        f'{CKPT_DIR}\n'
        '  هذا الملف يأتي مع المستودع (clone نظيف يجده في مكانه). إن '
        'حُذف بالخطأ فاستعده من Git أو مرّر --checkpoint بمسار صريح.')


# ============================================================================
# 6) تحضير النص — المسار الموحد (Web/CLI/API كلهم من هنا)
# ============================================================================
def diacritic_density(text):
    letters = _AR_LETTERS.findall(text)
    if not letters:
        return 0.0, 0
    n_d = sum(1 for c in text if '\u064B' <= c <= '\u0652')
    return n_d / len(letters), n_d


def keep_arabic_only(text):
    """(توافق قديم مع webapp) — نفس eqz_text.keep_arabic_and_punct:
    يُبقي العربية والحركات والمسافات والترقيم المدرب (. , ? !) فقط."""
    return eqz_text.keep_arabic_and_punct(text)


def effective_diacritize_mode(raw_text, mode, dialect='egy'):
    """توحيد قرار التشكيل (كما يحسمه auto): كثافة حركات < 0.30 → تشكيل،
    وإلا تشكيل النص كما هو. الفصحى بلا قواعد det المصرية أبدًا."""
    if mode in ('egyptian', 'fusha', 'manual'):
        return mode
    density, _ = diacritic_density(
        eqz_text.keep_arabic_and_punct(raw_text))
    if density < 0.30:
        return 'fusha' if dialect == 'msa' else 'egyptian'
    return 'manual'


def prepare_text_rich(raw_text, diacritize_mode='auto', dialect='egy',
                      verbose=False):
    """تحضير النص بخروج غني — المسار الموحد الكامل.

    1) eqz_text.normalize_text: أرقام → كلمات منطوقة (مصري/فصحى) ·
       ، → , و ؟ → ? · نقحرة الكلمات اللاتينية (meeting → ميتينج).
    2) علامات القاف {ق}/{ج}/{ء} تُفك عن الكلمات قبل التشكيل (catt يقص
       غير العربي فتضيع) وتُعاد لحقًا بالكلمات نفسها بعد التشكيل —
       العدد محفوظ فالمحاذاة موضعية مضمونة.
    3) التشكيل حسب الوضع:
       egyptian: catt_eo + القواعد الصارمة det_tashkeel (دليل v2.1
                 الموجود في المشروع) — استرجاع الترقيم بالمحاذاة.
       fusha:    catt_eo + استرجاع الترقيم فقط (بلا قواعد مصرية).
       manual:   تشكيل النص كما هو.
    4) تنظيف نهائي: العربية + الحركات + المسافات + الترقيم المدرب
       (. , ? !) + أقواس العلامات {} — كل ما عداها يُسقطه G2P كما في
       التدريب تمامًا.

    يعيد dict: {text, did_vocalize, diacritize, normalized, numbers,
    translit, markers}."""
    # ---------- 1) التطبيع الموحد ----------
    norm = eqz_text.normalize_text(raw_text, dialect)
    had_digits = bool(re.search(r'[0-9\u0660-\u0669]', raw_text))
    had_latin = bool(re.search(r'[A-Za-z]', raw_text))

    if not _AR_LETTERS.search(norm):
        raise SystemExit('[خطأ] النص لا يحتوي حروفًا عربية.')

    # ---------- 2) فك علامات القاف قبل التشكيل ----------
    words_tags = eqz_text._split_markers(norm)
    tags = [t for _, t in words_tags]
    stripped = ' '.join(w for w, _ in words_tags)
    has_markers = any(tags)

    # ---------- 3) التشكيل ----------
    mode = effective_diacritize_mode(stripped, diacritize_mode, dialect)
    if mode in ('egyptian', 'fusha'):
        voc = ' '.join(catt_vocalize(stripped).split())
        if mode == 'egyptian':
            voc = apply_det_partial(stripped, voc)
        else:
            voc = eqz_text.restore_punctuation(stripped, voc)
    else:
        voc = eqz_text.keep_arabic_and_punct(stripped)
        if not _AR_LETTERS.search(voc):
            raise SystemExit('[خطأ] النص لا يحتوي حروفًا عربية بعد التنظيف.')

    # ---------- 4) إعادة العلامات للكلمات نفسها ----------
    voc_words = voc.split()
    if has_markers and len(voc_words) == len(words_tags):
        voc = ' '.join((w + t if t else w)
                       for w, t in zip(voc_words, tags))

    # ---------- 5) التنظيف النهائي (يحفظ {} لعلامات القاف) ----------
    text = _final_clean(voc)

    res = {
        'text': text,
        'did_vocalize': mode in ('egyptian', 'fusha'),
        'diacritize': mode,
        'normalized': norm,
        'numbers': had_digits,
        'translit': had_latin,
        'markers': has_markers,
    }
    if verbose:
        log(f'[prep] mode={mode} | numbers={had_digits} | '
            f'translit={had_latin} | markers={has_markers}')
        log(f'[prep] normalized: {norm}')
        log(f'[prep] final:      {text}')
    return res


_FINAL_KEEP = re.compile(
    r'[^\u0621-\u063A\u0641-\u064A\u064B-\u0652 .,?!{}]')


def _final_clean(text):
    """إبقاء العربية + الحركات + الترقيم المدرب (. , ? !) + أقواس العلامات."""
    text = text.replace('\u0640', '')               # التطويل
    text = _FINAL_KEEP.sub(' ', text)
    return ' '.join(text.split())


def prepare_text(raw_text, diacritize_mode='auto', dialect='egy'):
    """(توافق قديم) تحضير النص — يعيد (نص المعالجة، تم_التشكيل)."""
    res = prepare_text_rich(raw_text, diacritize_mode, dialect)
    return res['text'], res['did_vocalize']


# ============================================================================
# 7) خط التوليد الكامل
# ============================================================================
def validate_speaker(speaker):
    """التحقق من هوية المتحدث — 0/1 مصريان (الافتراض)، 2-6 فصحى (تقييم)."""
    try:
        speaker = int(speaker)
    except (TypeError, ValueError):
        raise SystemExit(f'[خطأ] رقم متحدث غير صالح: {speaker!r}')
    if speaker not in ALLOWED_SPEAKERS:
        raise SystemExit(
            f'[خطأ] المتحدث {speaker} غير مدرّب (7-15 غير مُعرَّفة في '
            f'checkpoint). المتاح: 0={SPEAKERS[0]} · 1={SPEAKERS[1]} '
            '(2-6 فصحى للتقييم المتقدم).')
    if speaker >= 2:
        log(f'[تحذير] المتحدث {speaker} ({SPEAKERS[speaker]}) فصحى وليس '
            'أحد الصوتين المصريين المعتمدين (0 ذكر / 1 أنثى).')
    return speaker


def synthesize(model, text, dialect, speaker, pace, out_path, denoise,
               peak_normalize=True, verbose=False):
    """من نص مُعالَج (مخرج prepare_text_rich) إلى ملف WAV.

    الترميز عبر eqz_tokens (سياسة التدريب النهائي نفسها). يعيد
    (عدد التوكنز، مدة الصوت بالثواني)."""
    import torch
    import soundfile as sf

    toks_ms, toks_egy, ids_of = get_tokenizer()
    toks = toks_ms(text) if dialect == 'msa' else toks_egy(text)
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

    wave = mel_to_wav(m, denoise=denoise, normalize=peak_normalize)
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
        description='Eiqaz TTS — توليد صوت عربي من نص (CPU فقط) — '
                    f'checkpoint الإنتاج: {DEFAULT_CHECKPOINT}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='أمثلة:\n'
               '  python infer.py --text "السلام عليكم" --out out.wav\n'
               '  python infer.py --text "عندي 5 كتب." --speaker 1\n'
               '  python infer.py --text "قسّمنا قطعة{ق} قماش على رقم{ج} '
               'أطفال وكل واحد قال{ء} شكرًا"\n'
               '      (علامات القاف: {ق}=قاف أصيلة · {ء}=همزة · {ج}=جيم '
               'مصرية [g] — تصل النموذج فعليًا)\n'
               '  python infer.py --text-file article.txt --out long.wav\n')
    ap.add_argument('--checkpoint', default=None,
                    help=f'اسم أو مسار checkpoint (يُبحث في checkpoints/ '
                         f'أولًا). الافتراضي: {DEFAULT_CHECKPOINT} '
                         '(المعتمد — يأتي مع المستودع).')
    ap.add_argument('--text', default=None, help='النص العربي مباشرة')
    ap.add_argument('--text-file', default=None,
                    help='قراءة النص من ملف (UTF-8)')
    ap.add_argument('--out', default='out.wav', help='مسار ملف WAV الناتج')
    ap.add_argument('--speaker', type=int, default=0,
                    help='0 = ذكر مصري (افتراضي) · 1 = أنثى مصرية · '
                         '2-6 = فصحى (تقييم متقدم)')
    ap.add_argument('--dialect', choices=['egy', 'msa'], default='egy',
                    help='egy: ترميز مصري بسياسة eqz_tokens (افتراضي) | '
                         'msa: ترميز فصحى')
    ap.add_argument('--pace', type=float, default=1.0,
                    help='سرعة الكلام (1.0 = طبيعي؛ أقل = أبطأ)')
    ap.add_argument('--diacritize', choices=list(DIACRITIZE_MODES),
                    default='auto',
                    help='التشكيل التلقائي (الافتراضي auto): auto = حسب '
                         'كثافة التشكيل | egyptian = catt + القواعد الصارمة '
                         'المصرية (دليل v2.1) | fusha = catt فقط | '
                         'manual = تشكيل النص كما هو')
    ap.add_argument('--vocalize', choices=['auto', 'always', 'never'],
                    default=None,
                    help='[مهمل — توافق قديم] always→تشكيل تلقائي · '
                         'never→يدوي · auto→حسب الكثافة. استخدم '
                         '--diacritize.')
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
    if torch.cuda.is_available():
        log('[ملاحظة] تم اكتشاف GPU لكن الحزمة مضبوطة على CPU فقط '
            '(device="cpu") — سيُستخدم المعالج.')

    # ---------------- توافق --vocalize القديم -----------------------------
    if args.vocalize:
        args.diacritize = {'always': 'auto', 'never': 'manual',
                           'auto': 'auto'}[args.vocalize]

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

    speaker = validate_speaker(args.speaker)

    ckpt = resolve_checkpoint(args.checkpoint)
    log(f'[1/4] checkpoint: {ckpt}')

    # ---------------- model -------------------------------------------------
    model, it = load_model(ckpt)
    log(f'[2/4] النموذج جاهز (iter {it}) — CPU')

    # ---------------- text pipeline -----------------------------------------
    res = prepare_text_rich(raw, args.diacritize, args.dialect,
                            verbose=args.verbose)
    text = res['text']
    tag = {'egyptian': 'تشكيل مصري (catt + القواعد الصارمة det)',
           'fusha': 'تشكيل فصحى (catt)',
           'manual': 'تشكيل النص كما هو'}[res['diacritize']]
    extras = []
    if res['numbers']:
        extras.append('أرقام → كلمات')
    if res['translit']:
        extras.append('نقحرة لاتينية')
    if res['markers']:
        extras.append('علامات قاف نشطة')
    log(f'[3/4] معالجة النص ({tag}'
        + (f' | {" · ".join(extras)}' if extras else '') + '):')
    log(f'      {text}')

    # ---------------- synth ---------------------------------------------------
    n_tokens, n_secs = synthesize(
        model, text, args.dialect, speaker, args.pace, args.out,
        args.denoise, verbose=args.verbose)
    log(f'[4/4] تم: {args.out} — {n_tokens} توكن | {n_secs:.1f} ثانية صوت '
        f'| متحدث {speaker} ({SPEAKERS[speaker]}) | لهجة {args.dialect} | '
        f'تشكيل {res["diacritize"]} | pace {args.pace}')


if __name__ == '__main__':
    main()
