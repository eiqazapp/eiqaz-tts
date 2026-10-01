#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qaf_experiment.py — عدة تجربة «اكتشاف القاف الأصيلة» في النموذج
==========================================================================
(اكتشاف المستخدم 2026-10-01: قطعة/قطع تُنطق قافًا أصيلة في سياق الدرس،
لكن همزةً منفردة؛ وقرآن ≠ القرآن — «ربما هناك حل ما»)

الأدلة الموجّهة لهذه العدة (موثقة في validation/QAF_NATIVE_DISCOVERY.md):
  1. corpus التدريب: عائلة قطع = 108 مواضع (شكّلها catt بالكسرة في السياق
     قِطْعَةً/ِ/ٍ في ~16/18)، بينما المنفردة تُشكَّل بالفتحة قَطْعَةَ — شكل
     خارج توزيع التدريب. القرآن بالتعريف = 0 مواضع؛ قرآن المجردة = 2.
  2. تحليل مخرجات النموذج الحقيقية (بلا أي تصحيح قاف): مواضع قاف بإغلاق
     صامت + VOT طويل (42-100ms) = انفجار عميق [q]-الشكل، لا همزة ولا جيم.
  3. إذن: توكن q غير مدرَّب (حقيقة توكنية) لكن الصوت الانفجاري العميق
     موجود كتحقق متعلّم لتوكن '<' في كلمات بعينها — والتشكيل (فتحة/كسرة)
     يبدو مفتاح التحكم.

الفرضية القابلة للاختبار: زرع الشكل المطابق للتدريب (كسرة+تنوين) يجعل
النموذج يُخرج نطقه المتعلّم الأصلي حتى خارج السياق — «حل بلا إعادة تدريب».

التجربة: كل كلمة تُولَّد في 5 تصييرات:
  خام_منفرد  : egy + hamza + catt(منفرد)  → سلوك النموذج الأصلي على الشكل المنفرد
  خام_مدروس  : egy + hamza + شكل مزروع (كسرة/تنوين كما في التدريب، بلا catt)
  كاف_k      : msa (كل قاف→[k] تقريبًا)   → الخيار الحالي في وضع qaf
  جيم_g      : egy + g (كل قاف→جيم [g])
  افتراضي    : egy + auto (السلوك المعتمد: B→[g] والباقي '<')

التصنيف الصوتي الآلي لكل ملف (كلمة قاف-ابتدائية = موضع معلوم):
  لا انفجار في موضع القاف → 'ء' (همزة)
  انفجار بإغلاق مجهور أو فتحة قصيرة (<25ms) → 'g' (مجهور)
  انفجار صامت الإغلاق بفتحة طويلة (>=30ms) → 'q?' (انفجار عميق!)
  (البburst يُكشف في أول 0.8 ثانية — الكلمات مفردة فالقاف في البداية)

التشغيل (على جهازك، من مجلد inference بعد وضع checkpoint في checkpoints/):
    python ..\\validation\\qaf_experiment.py            # كل الكلمات
    python ..\\validation\\qaf_experiment.py --words 10  # أول 10 كلمات
    python ..\\validation\\qaf_experiment.py --speaker 1
    python ..\\validation\\qaf_experiment.py --skip-listen  # بلا جمل السياق

المخرجات:
    validation/qaf_experiment_output/*.wav   (استمع بنفسك وقارن مع التصنيف)
    validation/qaf_experiment_output/results.json + جدول في الكونسول

⚠️ تذكير بالملصق: هذه تجربة تشخيصية — نتيجتها تقرر تصميم «وضع القاف
الفصحى». الطبقة الحتمية (إن فعلتها) تبقى «تصحيح جزئي» معلَّمًا كما هي.
"""
import argparse
import json
import os
import sys
import time

# --- Windows console safety (نمط infer.py) ---------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_DIR = os.path.join(REPO_ROOT, 'inference')
LIB_DIR = os.path.join(INFER_DIR, 'lib')
for _p in (INFER_DIR, LIB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# الكلمات: (التسمية، المجردة، الشكل المدروس المزروع، الفئة)
MEASURE_WORDS = [
    # مرشحات «القاف الأصيلة» (تقارير المستخدم)
    ('قطعة', 'قطعة', 'قِطْعَةً', 'مرشح أصيلة'),
    ('قطع', 'قطع', 'قِطَعٍ', 'مرشح أصيلة'),
    ('قرآن', 'قرآن', 'قُرْآنًا', 'مرشح أصيلة (مجردة)'),
    # فئة B (قياس صوتي سابق: [g])
    ('رقم', 'رقم', 'رَقْمٍ', 'B مرجع [g]'),
    ('قانون', 'قانون', 'قَانُونًا', 'B مرجع [g]'),
    ('قرش', 'قرش', 'قِرْشًا', 'B مرجع [g]'),
    # منهجية (قائمة Q)
    ('قسمة', 'قسمة', 'قِسْمَةً', 'منهجية'),
    ('يقسم', 'يقسم', 'يَقْسِمُ', 'منهجية'),
    ('تقريب', 'تقريب', 'تَقْرِيبًا', 'منهجية'),
    ('قياس', 'قياس', 'قِيَاسًا', 'منهجية'),
    ('مقياس', 'مقياس', 'مِقْيَاسًا', 'منهجية'),
    ('قيمة', 'قيمة', 'قِيمَةً', 'منهجية'),
    ('قاعدة', 'قاعدة', 'قَاعِدَةً', 'منهجية'),
    ('قطر', 'قطر', 'قُطْرًا', 'منهجية'),
    ('مقام', 'مقام', 'مَقَامًا', 'منهجية'),
    ('مستقيم', 'مستقيم', 'مُسْتَقِيمًا', 'منهجية'),
    ('قوة', 'قوة', 'قُوَّةً', 'منهجية'),
    ('طاقة', 'طاقة', 'طَاقَةً', 'منهجية'),
    ('حقوق', 'حقوق', 'حُقُوقًا', 'منهجية'),
    ('قطاع', 'قطاع', 'قِطَاعًا', 'منهجية'),
    # ضوابط (متوقع: همزة خام)
    ('قمر', 'قمر', 'قَمَرًا', 'ضابط ء'),
    ('قال', 'قال', 'قَالَ', 'ضابط ء'),
    ('حقيقة', 'حقيقة', 'حَقِيقَةً', 'ضابط ء'),
    ('قراءة', 'قراءة', 'قِرَاءَةً', 'ضابط ء'),
]

# (اسم التصيير، اللهجة، وضع القاف، وضع التشكيل، مصدر النص)
RENDERINGS = [
    ('خام_منفرد', 'egy', 'hamza', 'always', 'bare'),
    ('خام_مدروس', 'egy', 'hamza', 'never', 'planted'),
    ('كاف_k', 'msa', 'auto', 'always', 'bare'),
    ('جيم_g', 'egy', 'g', 'always', 'bare'),
    ('افتراضي', 'egy', 'auto', 'always', 'bare'),
]

# جمل استماع سياقية (من درس المستخدم + قرآن/القرآن)
LISTEN_SENTENCES = [
    ('درس-قطعة', 'طيب، لو معانا عشرين قطعة حلوى، وعايزين نوزعهم على خمسة أطفال بالتساوي، كل طفل هياخد كام قطعة؟', 'egy', 'auto'),
    ('درس-قطع', 'يعني كل طفل هياخد أربع قطع.', 'egy', 'auto'),
    ('درس-قسمة', 'النهارده هنتكلم عن القسمة. القسمة ببساطة هي إننا نوزع حاجة بالتساوي.', 'egy', 'auto'),
    ('قرآن-مجردة-auto', 'قرآن', 'egy', 'auto'),
    ('قرآن-معرفة-auto', 'القرآن', 'egy', 'auto'),
    ('قرآن-مجردة-خام', 'قرآن', 'egy', 'hamza'),
    ('قرآن-معرفة-خام', 'القرآن', 'egy', 'hamza'),
    ('قرآن-مدروس-خام', 'قُرْآنًا', 'egy', 'hamza'),
    ('قطعة-مدروسة-خام', 'قِطْعَةً', 'egy', 'hamza'),
]

SR = 22050


# ============================================================================
# التصنيف الصوتي (numpy فقط — بلا scipy: الحزمة لا تتطلبها)
# ============================================================================
def _bands(x, frame=0.008, hop=0.002):
    """مسارات طاقة: جهر 60-300 / انفجار 300-5000 / الكل."""
    import numpy as np
    L = int(frame * SR)
    w = np.hanning(L)
    n = int((len(x) - L) / (hop * SR)) + 1
    freqs = np.fft.rfftfreq(L, 1 / SR)
    m_v = (freqs >= 60) & (freqs < 300)
    m_b = (freqs >= 300) & (freqs < 5000)
    ev = np.zeros(n)
    eb = np.zeros(n)
    step = int(hop * SR)
    for i in range(n):
        s = i * step
        seg = x[s:s + L] * w
        mag = np.abs(np.fft.rfft(seg))
        ev[i] = float(np.sum(mag[m_v]))
        eb[i] = float(np.sum(mag[m_b]))
    return ev, eb, hop


def classify_word_wav(path, t_max=0.9):
    """تصنيف القاف الابتدائية في ملف كلمة مفردة.

    يعيد dict: {verdict, detail} — verdict ∈ {'ء', 'g', 'q?', 'ambiguous',
    'no_speech'}."""
    import numpy as np
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype='float32')
    except Exception as e:
        return {'verdict': 'error', 'detail': str(e)}
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    x = x / (np.abs(x).max() + 1e-9)
    if len(x) < int(0.25 * SR):
        return {'verdict': 'no_speech', 'detail': 'قصير جدًا'}

    ev, eb, hop = _bands(x)
    n = len(ev)
    tot = ev + eb + 1e-12
    # مستوى الكلام (للتمييز عن الصمت البادئ)
    speech = np.percentile(eb, 85)
    if speech < 1e-6:
        return {'verdict': 'no_speech', 'detail': 'بلا كلام'}

    # أول إطار كلام
    i0 = next((i for i in range(n) if eb[i] > 0.15 * speech), 0)
    # كشف أول انفجار في نافذة الكلام الأولى: قفزة eb قوية بعد إطار هادئ
    burst_i = None
    for i in range(i0 + 3, min(n - 2, int(t_max / hop))):
        cur = eb[i]
        pre = np.min(eb[max(i0, i - 18):i - 1]) if i - 1 > i0 else cur
        if cur > 5 * (pre + 1e-12) and cur > 0.4 * speech:
            burst_i = i
            break
    if burst_i is None:
        return {'verdict': 'ء',
                'detail': 'لا انفجار فموي في موضع القاف (بداية صوتية مباشرة)'}

    # جهر الإغلاق: نسبة نطاق الجهر في 40ms قبل الانفجار
    pre = slice(max(i0, burst_i - int(0.040 / hop)), burst_i - 1)
    closure_voice = float(np.mean(ev[pre]) / (np.mean(tot[pre]) + 1e-12)) \
        if burst_i - 1 > pre.start else 0.0
    # VOT: حتى نسبة الجهر > 0.4 (مستقرة إطارين)
    vot_ms = None
    for j in range(burst_i, min(n - 2, burst_i + int(0.12 / hop))):
        if ev[j] / (tot[j] + 1e-12) > 0.40 and all(
                ev[k] / (tot[k] + 1e-12) > 0.30 for k in (j + 1, j + 2)):
            vot_ms = round((j - burst_i) * hop * 1000)
            break
    det = (f'انفجار عند {(burst_i * hop):.2f}s | جهر_الإغلاق='
           f'{closure_voice:.2f} | VOT={vot_ms}ms')
    if closure_voice > 0.30 or (vot_ms is not None and vot_ms < 25):
        return {'verdict': 'g', 'detail': det + ' → مجهور/فتحة قصيرة'}
    if vot_ms is not None and vot_ms >= 30:
        return {'verdict': 'q?',
                'detail': det + ' → صامت الإغلاق بفتحة طويلة (انفجار عميق!)'}
    return {'verdict': 'ambiguous',
            'detail': det + ' → غير حاسم (راجع بالسمع)'}


# ============================================================================
# التوليد
# ============================================================================
def main():
    import infer

    ap = argparse.ArgumentParser(
        description='عدة تجربة اكتشاف القاف الأصيلة (انظر رأس الملف)')
    ap.add_argument('--words', type=int, default=0,
                    help='عدد الكلمات المفحوصة (0 = الكل)')
    ap.add_argument('--speaker', type=int, default=0, choices=(0, 1))
    ap.add_argument('--skip-listen', action='store_true',
                    help='توليد جمل الاستماع السياقية')
    ap.add_argument('--out', default=None, help='مجلد المخرجات')
    args = ap.parse_args()

    out_dir = args.out or os.path.join(HERE, 'qaf_experiment_output')
    os.makedirs(out_dir, exist_ok=True)

    words = MEASURE_WORDS if not args.words else MEASURE_WORDS[:args.words]

    ckpt = infer.resolve_checkpoint(None)
    print(f'[1/4] checkpoint: {ckpt}')
    model, it = infer.load_model(ckpt)
    print(f'[2/4] النموذج جاهز (iter {it}) — متحدث {args.speaker}')

    # إحماء catt
    infer.catt_vocalize('اختبار')
    print('[3/4] توليد الكلمات × التصييرات ...')

    results = {}
    t0 = time.time()
    n_files = 0
    for label, bare, planted, kind in words:
        row = {'kind': kind, 'renderings': {}}
        for rname, dialect, qaf, voc, src in RENDERINGS:
            text_src = planted if src == 'planted' else bare
            wav = os.path.join(
                out_dir, f'{label}_{rname}.wav').replace(' ', '_')
            try:
                text, _ = infer.prepare_text(text_src, voc, dialect, qaf,
                                             det_partial=False)
                infer.synthesize(model, text, dialect, args.speaker, 1.0,
                                 wav, 0.005, qaf_mode=qaf)
                cls = classify_word_wav(wav)
                row['renderings'][rname] = {
                    'text_used': text, 'wav': os.path.basename(wav),
                    'verdict': cls['verdict'], 'detail': cls['detail']}
                n_files += 1
            except SystemExit as e:
                row['renderings'][rname] = {'error': str(e.code)}
            except Exception as e:                     # noqa: BLE001
                row['renderings'][rname] = {
                    'error': f'{type(e).__name__}: {e}'}
        results[label] = row
        # سطر جدولي مختصر
        cells = []
        for rname, *_ in [(r[0],) for r in RENDERINGS]:
            v = row['renderings'].get(rname, {})
            cells.append(v.get('verdict', v.get('error', '—'))[:10])
        print(f'  {label:<10} [{kind:<14}] ' + ' | '.join(cells))

    listen = {}
    if not args.skip_listen:
        print('[4/4] جمل الاستماع السياقية ...')
        for label, sent, dialect, qaf in LISTEN_SENTENCES:
            wav = os.path.join(out_dir, f'listen_{label}.wav')
            try:
                text, _ = infer.prepare_text(sent, 'always', dialect, qaf,
                                             det_partial=False)
                infer.synthesize(model, text, dialect, args.speaker, 1.0,
                                 wav, 0.005, qaf_mode=qaf)
                listen[label] = {'sentence': sent, 'qaf': qaf,
                                 'wav': os.path.basename(wav),
                                 'vocalized_text': text}
            except Exception as e:                     # noqa: BLE001
                listen[label] = {'error': str(e)}

    summary = {
        'n_words': len(words), 'n_files': n_files,
        'speaker': args.speaker,
        'elapsed_s': round(time.time() - t0, 1),
        'verdicts_count': {},
    }
    for label, row in results.items():
        for rname, v in row['renderings'].items():
            if 'verdict' in v:
                key = f'{rname}:{v["verdict"]}'
                summary['verdicts_count'][key] = \
                    summary['verdicts_count'].get(key, 0) + 1

    with open(os.path.join(out_dir, 'results.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'summary': summary, 'words': results,
                   'listen': listen}, f, ensure_ascii=False, indent=1)

    print('\n' + '=' * 78)
    print(f'تم: {n_files} ملف صوت في {summary["elapsed_s"]}s → {out_dir}')
    print('اقرأ الجدول أعلاه ثم استمع للملفات وقارن أذنك مع التصنيف الآلي.')
    print('دلائل «q?» في خام_مدروس (وليس خام_منفرد) = تأكيد فرضية زرع الشكل.')
    print('النتائج الكاملة: results.json — أرسلها أو أخبرني بالخلاصة.')
    print('التصنيفات: ء = لا انفجار | g = مجهور/فتحة قصيرة | '
          'q? = صامت بفتحة طويلة (انفجار عميق) | ambiguous = راجع سمعيًا')


if __name__ == '__main__':
    main()
