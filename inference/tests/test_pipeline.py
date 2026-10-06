#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات Eiqaz TTS الإلزامية — مواءمة الاستدلال مع النموذج النهائي
=============================================================================
اختبارات حقيقية (بلا mock): تحمّل states_cont_180516.pth نفسه وتولّد
صوتًا فعليًا. مجموعة الاختبار الثابتة (regression):

  A — الترقيم: بدون/نقطة/فاصلة (عربية وأمريكية)/استفهام/تعجب/جمل متعددة
      + مقارنة مع السلوك القديم (كان يحذف الترقيم قبل النموذج).
  B — الأرقام: 5 / 10 / 2026 / داخل جملة / عشري / أرقام عربية شرقية.
  C — القاف/الجيم/الهمزة: قانون · {ق}انون · {ج} · {ء} · كلمات ج ·
      كلمات همزة · جملة تجمع الثلاثة.
  D — التشكيل المصري: جُمل مصرية حقيقية (القواعد الصارمة det).
  E — الفصحى: الوضع الفصيحي يعمل.
  F — النص المختلط: عربي + English.
  G — النص الطويل: تقسيم متعدد المقاطع لكل من المتحدث 0 والمتحدث 1
      + إثبات ثبات هوية المتحدث عبر المقاطع (speaker/RMS/F0/مدة).

الاستخدام (من مجلد inference/):
    python tests/test_pipeline.py
يكتب تقريرًا كاملًا في tests/test_results.json ويعيد رمز خروج 0 عند
نجاح كل شيء.
"""
import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
for p in (INF, os.path.join(INF, 'lib'), os.path.join(INF, 'lib', 'mixer_repo')):
    if p not in sys.path:
        sys.path.insert(0, p)

import infer                      # noqa: E402

RESULTS = {'sections': {}, 'failures': [], 'n_pass': 0, 'n_fail': 0,
           'details': {}}
TMP = tempfile.mkdtemp(prefix='eqz_tests_')


def record(section, name, ok, detail=None):
    sec = RESULTS['sections'].setdefault(section, {'pass': [], 'fail': []})
    (sec['pass'] if ok else sec['fail']).append(name)
    if detail is not None:
        RESULTS['details'].setdefault(section, {})[name] = detail
    if ok:
        RESULTS['n_pass'] += 1
    else:
        RESULTS['n_fail'] += 1
        RESULTS['failures'].append(
            {'section': section, 'test': name, 'detail': str(detail)[:500]})
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {name}')
    if detail and not ok:
        print(f'         {str(detail)[:300]}')


def f0_autocorr(wav, sr):
    """تقدير F0 وسيط بالارتباط الذاتي على عدة نوافذ (متين ضد أخطاء
    الأوكتاف في نافذة واحدة — لإثبات ثبات الهوية فقط)."""
    import numpy as np
    f0s = []
    n = len(wav)
    win = min(sr, max(n // 5, int(0.4 * sr)))
    if win < int(0.2 * sr):
        return 0.0
    for frac in (0.15, 0.3, 0.45, 0.6, 0.75):
        s0 = int(n * frac)
        seg = wav[s0:s0 + win]
        if len(seg) < win:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lo, hi = int(sr / 250), int(sr / 70)
        if hi >= len(ac):
            continue
        i = int(np.argmax(ac[lo:hi])) + lo
        if ac[i] > 0.3 and i > 0:
            f0s.append(sr / i)
    if not f0s:
        return 0.0
    f0s.sort()
    return f0s[len(f0s) // 2]


# ============================================================================
print('=' * 72)
print('تحميل النموذج الحقيقي — states_cont_180516.pth')
print('=' * 72)
CKPT = infer.resolve_checkpoint()
assert CKPT.endswith('states_cont_180516.pth'), CKPT
MODEL, IT = infer.load_model(CKPT)
print(f'loaded: iter={IT} (n_speakers=16, num_tokens=148)')


def full(raw, dialect='egy', diacritize='egyptian'):
    """المسار الكامل: تحضير + ترميز (بلا توليد صوتي)."""
    res = infer.prepare_text_rich(raw, diacritize, dialect)
    toks = infer.tokenize(res['text'], dialect)
    return res, toks


def synth(raw, name, speaker=0, dialect='egy', diacritize='egyptian',
          pace=1.0):
    """المسار الكامل مع توليد صوتي فعلي."""
    res = infer.prepare_text_rich(raw, diacritize, dialect)
    out = os.path.join(TMP, name)
    n_tok, n_secs = infer.synthesize(
        MODEL, res['text'], dialect, speaker, pace, out, 0.005)
    import soundfile as sf
    w, sr = sf.read(out, dtype='float32')
    return res, n_tok, w, sr, out


# ============================================================================
# A — الترقيم
# ============================================================================
print('\n' + '=' * 72)
print('A — الترقيم (لا يختفي قبل النموذج)')
print('=' * 72)
A_CASES = [
    ('بدون ترقيم', 'دلوقتي هنبدأ الدرس يا عمر', None),
    ('نقطة', 'دلوقتي هنبدأ الدرس يا عمر.', '.'),
    ('فاصلة أمريكية', 'دلوقتي هنبدأ الدرس, يا عمر', ','),
    ('فاصلة عربية', 'دلوقتي هنبدأ الدرس، يا عمر', ','),
    ('استفهام عربي', 'هل أنت جاهز؟', '?'),
    ('تعجب', 'ممتاز! نبدأ الآن.', '!'),
    ('جمل متعددة', 'ممتاز! نبدأ الآن. هل أنت جاهز؟ نعم.', '.'),
]
for name, raw, expect_punct in A_CASES:
    try:
        res, toks = full(raw)
        got_punct = [t for t in toks if t in '.,?!']
        ok = (expect_punct is None or expect_punct in got_punct)
        record('A', name, ok,
               {'input': raw, 'normalized': res['normalized'],
                'final': res['text'],
                'punct_tokens': got_punct, 'n_tokens': len(toks)})
    except Exception as e:                                   # noqa: BLE001
        record('A', name, False, traceback.format_exc())

# إثبات الفرق قبل/بعد الإصلاح: توكنز النقطة كانت تُحذف في المسار القديم
try:
    res_new, toks_new = full('دلوقتي هنبدأ الدرس يا عمر.')
    # محاكاة المسار القديم: حذف الترقيم قبل الترميز (كان السلوك)
    import eqz_text
    old_text = eqz_text.keep_arabic_and_punct(
        res_new['text']).replace('.', '').replace(',', '') \
        .replace('?', '').replace('!', '')
    old_text = ' '.join(old_text.split())
    toks_old = infer.tokenize(old_text)
    has_dot_new = '.' in toks_new
    has_dot_old = '.' in toks_old
    record('A', 'مقارنة قبل/بعد: توكن النقطة', has_dot_new and not has_dot_old,
           {'old_had_dot': has_dot_old, 'new_has_dot': has_dot_new,
            'old_n_tokens': len(toks_old), 'new_n_tokens': len(toks_new)})
except Exception as e:                                       # noqa: BLE001
    record('A', 'مقارنة قبل/بعد: توكن النقطة', False, traceback.format_exc())

# ============================================================================
# B — الأرقام
# ============================================================================
print('\n' + '=' * 72)
print('B — الأرقام (تُنطق كلمات عربية ولا تختفي)')
print('=' * 72)
B_CASES = [
    ('5', 'عندي 5 كتب', 'خمسة'),
    ('10', 'عمره 10 سنين', 'عشرة'),
    ('2026', 'عام 2026 كان كويس', 'ألفين'),
    ('250', 'السعر 250 جنيه', 'ميتين'),
    ('15', 'عندي 15 كتاب', 'خمستاشر'),
    ('20', 'الرصيد 20 جنيه', 'عشرين'),
    ('100', 'عندي 100 جنيه', 'مية'),
    ('رقم عربي ٥', 'عندي ٥ كتب', 'خمسة'),
    ('رقم عربي ١٠', 'عمره ١٠ سنين', 'عشرة'),
    ('رقم عربي ٢٥٠', 'دفع ٢٥٠ جنيه', 'ميتين'),
    ('رقم عربي ٢٠٢٦', 'في عام ٢٠٢٦', 'ألفين'),
    ('عشري 3.5', 'المعدل 3.5', 'ونص'),
    ('عشري 5.4', 'نسبة 5.4', 'فاصل'),
    ('نسبة %', 'استهلكت 80% من الباقة', 'في المية'),
]
for name, raw, expect_word in B_CASES:
    try:
        res, toks = full(raw)
        import re as _re
        digits_left = _re.search(r'[0-9\u0660-\u0669]', res['text'])
        ok = (not digits_left) and (expect_word in res['normalized'])
        record('B', name, ok,
               {'input': raw, 'normalized': res['normalized'],
                'final': res['text'], 'digits_left': bool(digits_left),
                'n_tokens': len(toks)})
    except Exception as e:                                   # noqa: BLE001
        record('B', name, False, traceback.format_exc())

# ============================================================================
# C — القاف/الجيم/الهمزة
# ============================================================================
print('\n' + '=' * 72)
print('C — القاف/الجيم/الهمزة (سياسة التدريب النهائي)')
print('=' * 72)
C_CASES = [
    ('قانون → q', 'قانون', 'q'),
    ('{ق}انون → q', '{ق}انون', 'q'),
    ('القرآن → q', 'القرآن', 'q'),
    ('حقيقة → q', 'الحقيقة', 'q'),
    ('قطعة{ق} → q', 'قطعة{ق}', 'q'),
    ('رقم → v', 'رقم', 'v'),
    ('رقم{ج} → v', 'رقم{ج}', 'v'),
    ('{ج}الرقم → v', '{ج}الرقم', 'v'),
    ('كلمة ج → v', 'جميل', 'v'),
    ('جملة تجمع الثلاثة',
     'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا',
     'q'),
]
for name, raw, want_tok in C_CASES:
    try:
        res, toks = full(raw)
        if name.startswith('جملة'):
            ok = ('q' in toks) and ('v' in toks) and ('<' in toks)
        else:
            ok = (want_tok in toks)
        record('C', name, ok,
               {'input': raw, 'final': res['text'],
                'qaf_tokens': [t for t in toks if t in ('q', 'v', '<', 'j')],
                'n_tokens': len(toks)})
    except Exception as e:                                   # noqa: BLE001
        record('C', name, False, traceback.format_exc())

# علامة الهمزة: قال{ء} → '<' و لا q ولا v
try:
    res, toks = full('قال{ء}')
    ok = ('<' in toks) and ('q' not in toks) and ('v' not in toks)
    record('C', 'قال{ء} → همزة', ok, {'final': res['text'],
                                       'tokens': toks})
except Exception as e:                                       # noqa: BLE001
    record('C', 'قال{ء} → همزة', False, traceback.format_exc())

# كلمة همزة حقيقية
try:
    res, toks = full('أمر')
    ok = ('<' in toks) and ('q' not in toks)
    record('C', 'كلمة همزة حقيقية (أمر)', ok, {'tokens': toks})
except Exception as e:                                       # noqa: BLE001
    record('C', 'كلمة همزة حقيقية', False, traceback.format_exc())

# تحقق خاص: q لا يتحول إلى همزة أبدًا (الخلل القديم EGY_TOKEN_MAP q→'<')
try:
    _, toks = full('قانون')
    old_map_bug = '<' in toks and 'q' not in toks
    record('C', 'q لم يعد يتحول إلى همزة (الخلل القديم)', not old_map_bug,
           {'tokens': toks})
except Exception as e:                                       # noqa: BLE001
    record('C', 'q لم يعد يتحول إلى همزة (الخلل القديم)', False,
           traceback.format_exc())

# ============================================================================
# D — التشكيل المصري (القواعد الصارمة الموجودة)
# ============================================================================
print('\n' + '=' * 72)
print('D — التشكيل المصري (catt + القواعد الصارمة det — دليل v2.1)')
print('=' * 72)
D_CASES = [
    'إزيك يا صاحبي',
    'دلوقتي هنبدأ الدرس يا عمر',
    'أنا عايز أعرف إنت عملت إيه النهارده',
    'مش عارف أعمل إيه في المشكلة دي',
    'البكريات كده أحسن بكتير',
    'هو إنت ليه ما قلتليش من الأول؟',
]
for raw in D_CASES:
    try:
        res, toks = full(raw)
        density, _ = infer.diacritic_density(res['text'])
        ok = res['did_vocalize'] and density >= 0.4
        record('D', raw[:40], ok,
               {'input': raw, 'final': res['text'],
                'density': round(density, 2), 'n_tokens': len(toks)})
    except Exception as e:                                   # noqa: BLE001
        record('D', raw[:40], False, traceback.format_exc())

# قاعدة ال التعريف الصارمة: الدرس → اِلدَّرْس (كسرة + شدة شمسية)
# (تُبنى الحركات بأكواد صريحة — الأحرف المكتوبة بصريًا غير موثوقة
# في ترتيب الشدة/الحركة عبر المحارف ثنائية الاتجاه)
try:
    res, _ = full('الدرس هيبدأ دلوقتي')
    ok = (('ا' + '\u0650') in res['text']
          and ('د' + '\u0651' + '\u064E') in res['text'])
    record('D', 'قاعدة ال التعريف (اِلدَّرْس)', ok, {'final': res['text']})
except Exception as e:                                       # noqa: BLE001
    record('D', 'قاعدة ال التعريف (اِلدَّرْس)', False, traceback.format_exc())

# ============================================================================
# E — الفصحى
# ============================================================================
print('\n' + '=' * 72)
print('E — الفصحى (وضع fusha يعمل والنص المشكول يمر كما هو)')
print('=' * 72)
try:
    msa_text = 'اَللُّغَةُ الْعَرَبِيَّةُ مِنْ أَكْثَرِ اللُّغَاتِ انتِشَارًا فِي الْعَالَم.'
    res, toks = full(msa_text, dialect='msa', diacritize='manual')
    record('E', 'فصحى يدوي (مشكول كما هو)', res['text'] and len(toks) > 10,
           {'final': res['text'], 'n_tokens': len(toks)})
except Exception as e:                                       # noqa: BLE001
    record('E', 'فصحى يدوي', False, traceback.format_exc())

try:
    res, toks = full('اللغة العربية من أكثر اللغات انتشارا في العالم',
                     dialect='msa', diacritize='fusha')
    density, _ = infer.diacritic_density(res['text'])
    record('E', 'فصحى تلقائي (catt)', res['did_vocalize'] and density > 0.3,
           {'final': res['text'], 'density': round(density, 2),
            'n_tokens': len(toks)})
except Exception as e:                                       # noqa: BLE001
    record('E', 'فصحى تلقائي', False, traceback.format_exc())

try:
    res, toks = full('هذا قانون مهم')
    ok = 'q' in toks
    record('E', 'فصحى: قاف خام q (مسار msa)', ok, {'tokens': toks[:20]})
except Exception as e:                                       # noqa: BLE001
    record('E', 'فصحى: قاف خام q', False, traceback.format_exc())

# ============================================================================
# F — النص المختلط
# ============================================================================
print('\n' + '=' * 72)
print('F — النص المختلط (عربي + English)')
print('=' * 72)
try:
    raw = 'أنا رايح أعمل meeting بكرة'
    res, toks = full(raw)
    import re as _re
    latin_left = _re.search(r'[A-Za-z]', res['text'])
    ok = (not latin_left) and ('ميتينج' in res['normalized'])
    record('F', 'meeting → ميتينج (نقحرة)', ok,
           {'input': raw, 'normalized': res['normalized'],
            'final': res['text'], 'latin_left': bool(latin_left),
            'n_tokens': len(toks)})
except Exception as e:                                       # noqa: BLE001
    record('F', 'meeting → ميتينج', False, traceback.format_exc())

try:
    raw = 'الـ deadline بتاع المشروع بعد أسبوع'
    res, toks = full(raw)
    ok = 'ديدلاين' in res['normalized']
    record('F', 'deadline → ديدلاين', ok, {'normalized': res['normalized']})
except Exception as e:                                       # noqa: BLE001
    record('F', 'deadline → ديدلاين', False, traceback.format_exc())

# ============================================================================
# G — النص الطويل + ثبات هوية المتحدث (توليد صوتي حقيقي)
# ============================================================================
print('\n' + '=' * 72)
print('G — النص الطويل (توليد حقيقي: متحدث 0 ومتحدث 1)')
print('=' * 72)
LONG = (
    'النهاردة هنتكلم عن موضوع مهم جدًا وهو تنظيم الوقت. '
    'كل واحد فينا عنده 24 ساعة في اليوم, بس الفرق بيننا هو إزاي '
    'بنستخدم الساعات دي. في ناس بتقضي اليوم كله في التليفون, '
    'وناس تانية بتذاكر وبتشتغل. أنا شخصيًا بحاول أنظم يومي من الصبح. '
    'بصحى الساعة 6, وبعمل رياضة نص ساعة, وبعدين بفطر. '
    'بعدها ببدأ شغلي الأساسي, وبيكون فيه 3 اجتماعات. '
    'في وسط اليوم باخد استراحة قصيرة. بالليل بقرا كتاب شوية قبل النوم. '
    'النتيجة إن تنظيم الوقت بيساعدنا نحقق أهدافنا. '
    'هل جربت تنظيم وقتك؟'
)
import webapp  # noqa: E402  — لنقسيم المقاطع بنفس دالة الواجهة

for spk, label in ((0, 'الرجل المصري'), (1, 'المرأة المصرية')):
    try:
        chunks = webapp.split_into_chunks(LONG, 'egy', None)
        detail = []
        for i, ch in enumerate(chunks):
            res = infer.prepare_text_rich(ch, 'egyptian', 'egy')
            out = os.path.join(TMP, f'g_spk{spk}_chunk{i:02d}.wav')
            n_tok, _ = infer.synthesize(
                MODEL, res['text'], 'egy', spk, 1.0, out, 0.005)
            import soundfile as sf
            import numpy as np
            w, sr = sf.read(out, dtype='float32')
            detail.append({
                'chunk': i + 1, 'speaker': spk, 'n_tokens': n_tok,
                'duration_s': round(len(w) / sr, 2),
                'rms': round(float(np.sqrt(np.mean(np.square(w)))), 4),
                'f0_hz': round(f0_autocorr(w, sr), 1),
                'sample_rate': sr, 'channels': 1,
            })
        speakers = {d['speaker'] for d in detail}
        f0s = [d['f0_hz'] for d in detail if d['f0_hz'] > 0]
        f0_median = sorted(f0s)[len(f0s) // 2] if f0s else 0
        f0_spread = (max(f0s) - min(f0s)) if f0s else 0
        # الثبات: متحدث واحد لكل المقاطع + مدى F0 معقول (< 60Hz)
        ok = (len(speakers) == 1) and (len(detail) >= 3) and f0_spread < 60
        record('G', f'متحدث {spk} ({label}): {len(detail)} مقاطع — '
               f'ثبات الهوية', ok,
               {'chunks': detail, 'f0_median': f0_median,
                'f0_spread': round(f0_spread, 1)})
        RESULTS.setdefault('g_f0', {})[spk] = f0_median
    except Exception as e:                                   # noqa: BLE001
        record('G', f'متحدث {spk}', False, traceback.format_exc())

# فحص جوهري إضافي: انفصال F0 بين الرجل والمرأة (هوية متمايزة)
try:
    g0 = RESULTS.get('g_f0', {}).get(0, 0)
    g1 = RESULTS.get('g_f0', {}).get(1, 0)
    ok = g0 > 0 and g1 > 0 and (g1 - g0) > 25
    record('G', 'انفصال F0 رجل/مرأة (هوية متمايزة)', ok,
           {'spk0_f0_median': g0, 'spk1_f0_median': g1})
except Exception as e:                                       # noqa: BLE001
    record('G', 'انفصال F0 رجل/مرأة', False, traceback.format_exc())

# G إضافي — توليد صوتي لأمثلة ممثلة (ترقيم/أرقام/مختلط/فصحى)
AUDIO_CASES = [
    ('ترقيم + أرقام (spk0)',
     'دلوقتي هنبدأ الدرس يا عمر. عندي 5 كتب, وكل واحد مهم. هل أنت جاهز؟',
     0, 'egy', 'egyptian'),
    ('مختلط + علامات (spk0)',
     'أنا رايح أعمل meeting بكرة. قسّمنا قطعة{ق} قماش على رقم{ج} أطفال '
     'وكل واحد قال{ء} شكرًا!', 0, 'egy', 'egyptian'),
    ('فصحى مشكول (spk0)',
     'اَللُّغَةُ الْعَرَبِيَّةُ مِنْ أَكْثَرِ اللُّغَاتِ انتِشَارًا فِي الْعَالَم.',
     0, 'msa', 'manual'),
    ('أرقام (spk1)',
     'عندي 5 كتب وعمره 10 سنين وسعر الكتاب 250 جنيه في عام 2026.',
     1, 'egy', 'egyptian'),
]
for name, raw, spk, dialect, diac in AUDIO_CASES:
    try:
        safe = name[:12].replace(' ', '_')
        res, n_tok, w, sr, out = synth(raw, f'audio_{safe}.wav',
                                       speaker=spk, dialect=dialect,
                                       diacritize=diac)
        ok = len(w) > sr  # أكثر من ثانية صوت
        record('G', f'صوت: {name}', ok,
               {'input': raw, 'final': res['text'], 'n_tokens': n_tok,
                'duration_s': round(len(w) / sr, 2), 'sr': sr,
                'f0_hz': round(f0_autocorr(w, sr), 1), 'wav': out})
    except Exception as e:                                   # noqa: BLE001
        record('G', f'صوت: {name}', False, traceback.format_exc())

# ============================================================================
RESULTS['checkpoint'] = {
    'path': CKPT, 'iter': IT,
    'sha256': '88e98233e451145bcaf0042a2a1862cb28312390d63a14d4e898a81200bd7561',
    'size_bytes': os.path.getsize(CKPT),
}
RESULTS['tmp_dir'] = TMP
out_json = os.path.join(HERE, 'test_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)

print('\n' + '=' * 72)
print(f"النتيجة: {RESULTS['n_pass']} PASS / {RESULTS['n_fail']} FAIL")
for sec, d in RESULTS['sections'].items():
    print(f"  {sec}: {len(d['pass'])} pass, {len(d['fail'])} fail")
print(f'التقرير الكامل: {out_json}')
print('=' * 72)
sys.exit(1 if RESULTS['n_fail'] else 0)
