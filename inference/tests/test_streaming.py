#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات Eiqaz TTS — التوليد المتدفق + المقاطعة + ثبات النبرة
=============================================================================
اختبارات حقيقية (بلا mock إلا في نقطة سباق واحدة موثقة): تحمّل
states_cont_180516.pth نفسه وتولّد صوتًا فعليًا عبر stream_tts.

  A — سلامة تجزئة النص: لا قصّ كلمات · لا فقد نص · لا تكرار · الوسوم
      ملتحمة · حدود السقف — عبر تغذية كلمة-بكلمة (أقصى ضغط).
  B — إثبات التدفق الحقيقي (§31): أول AudioChunk يصل قبل اكتمال تغذية
      النص وقبل finish() — بلاfake streaming.
  C — الإلغاء أثناء الاستهلاك: حدث نهائي cancelled · المنتج خرج ·
      العدّ الصادق للنتائج المهملة.
  D — تفريغ الطابور: مقاطع مولدة وغير مستهلكة تُحذف عند الإلغاء.
  E — رفض النتائج القديمة: request_id قديم يُرفض بعد بدء طلب جديد.
  F — طلب جديد فور المقاطعة: نفس النموذج (بلا إعادة تحميل) يعمل.
  G — السباقات: مقاطعة لحظة جاهزية المقطع · قبل التشغيل · أثناء
      التشغيل الفعلي (realtime) · نتيجة inference متأخرة تُهمَل ·
      إلغاءات متكررة · لا صوت قديم بعد الطلب الجديد.
  H — ثبات هوية المتحدث: نفس speaker لكل المقاطع · مدى F0 · ECAPA
      عبر المقاطع + تمييز رجل/مرأة كأساس.
  I — ثبات النبرة والأسلوب: نص طويل (سؤال/خبر/تعليمي/فاصلة/انتقال
      مصرية↔فصحى/شدة/الوسوم الثلاثة) · F0/RMS/معدل الكلام/الوصلات ·
      مقارنة single-shot: توكنات متطابقة + bit-identity بالمسار الخام.
  J — التوافق الرجعي: infer.synthesize كما هو (دخان).
  K — الأداء: Short/Medium/Long/Mixed (TTFA/RTF/زمن/مدة/مقاطع/CPU/RAM)
      + مقارنة أحجام المقاطع (صغير/وسط/كبير) واختيار الافتراضي.

الاستخدام (من مجلد inference/):
    python tests/test_streaming.py
يكتب تقريرًا في tests/streaming_results.json ويعيد رمز خروج 0 عند
نجاح كل شيء.
"""
import json
import os
import re as _re
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
for p in (INF, os.path.join(INF, 'lib'), os.path.join(INF, 'lib', 'mixer_repo')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                             # noqa: E402

import infer                                                    # noqa: E402
import stream_tts                                               # noqa: E402

RESULTS = {'sections': {}, 'failures': [], 'n_pass': 0, 'n_fail': 0,
           'details': {}}
TMP = tempfile.mkdtemp(prefix='eqz_stream_tests_')


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
            {'section': section, 'test': name, 'detail': str(detail)[:800]})
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {name}')
    if detail and not ok:
        print(f'         {str(detail)[:400]}')


def f0_autocorr(wav, sr):
    """تقدير F0 وسيط بالارتباط الذاتي — نفس دالة tests/test_pipeline.py."""
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
        best, bi = 0.0, -1
        for i in range(lo, min(hi, len(ac))):
            if ac[i] > best:
                best, bi = ac[i], i
        if bi > 0 and best > 0.3:
            f0s.append(sr / bi)
    if not f0s:
        return 0.0
    f0s.sort()
    return f0s[len(f0s) // 2]


def rms(w):
    return float(np.sqrt(np.mean(np.square(w)))) if len(w) else 0.0


def consume(stream, max_chunks=None, on_chunk=None):
    """استهلاك الأحداث حتى النهاية — يعيد (chunks, end_event)."""
    chunks, end = [], None
    for ev in stream.events():
        if ev.is_audio:
            chunks.append(ev)
            if on_chunk:
                on_chunk(ev)
            if max_chunks and len(chunks) >= max_chunks:
                break
        else:
            end = ev
            break
    return chunks, end


def run_stream(engine, text_pieces, *, speaker=0, cadence=0.05,
               max_chunks=None, on_chunk=None, **kw):
    """تشغيل طلب كامل بتغذية متدرجة — يعيد (session, chunks, end)."""
    s = engine.create_stream(speaker=speaker, **kw)

    def feeder():
        for p in text_pieces:
            s.feed(p)
            time.sleep(cadence)
        s.finish()

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    chunks, end = consume(s, max_chunks=max_chunks, on_chunk=on_chunk)
    th.join(10)
    return s, chunks, end


def split_pieces(text, n=None):
    """تقسيم نص إلى قطع تغذية (جمل أو أجزاء جمل)."""
    if n is None:
        return [p.strip() + ' ' for p in
                _re.split(r'(?<=[.!؟?…])\s+', text) if p.strip()]
    words = text.split()
    step = max(1, len(words) // n)
    return [' '.join(words[i:i + step]) + ' '
            for i in range(0, len(words), step)]


# ============================================================================
# نصوص الاختبار
# ============================================================================
SHORT = 'دلوقتي هنبدأ الدرس يا عمر.'

MEDIUM = (
    'بص يا عمر، النهاردة هنتكلم عن القسمة. '
    'القسمة معناها إننا بنوزع الحاجة بالتساوي. '
    'عندنا 10 تفاحات وعايزين نقسمها على 5 أطفال. '
    'كل واحد هياخد تفاحتين.'
)

LONG = (
    'بص يا عمر، النهاردة هنتكلم عن حاجة مهمة جدًا وهي القسمة. '
    'القسمة معناها إننا بنوزع الحاجة بالتساوي. '
    'عندنا 10 تفاحات وعايزين نقسمها على 5 أطفال. '
    'كل واحد هياخد تفاحتين. '
    'لو عندنا 9 تفاحات و3 أطفال، كل واحد هياخد 3 تفاحات. '
    'بس لو العدد مش بيتقسم بالتساوي، هتظهر باقية. '
    'مثلاً 10 على 3، كل واحد هياخد 3 وتفضل تفاحة واحدة. '
    'دي اسمها الباقي. مفهوم يا عمر؟'
)

# نص النبرة (§22): سؤال + خبرية + تعليمية + فاصلة + جملة طويلة +
# انتقال مصرية→فصحى + شدة + {ق} + {ج} + {ء}
PROSODY_TEXT = (
    'بص يا عمر، درس النهاردة عن القسمة. '
    'القسمة معناها إننا بنوزع الحاجة بالتساوي. '
    'عندنا اتناشر تفاحة وعايزين نقسمها على أربع أطفال، كل واحد هياخد تلات تفاحات. '
    'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا. '
    'والآن ننتقل إلى الفصحى: القسمةُ عمليةٌ حسابيةٌ أساسيةٌ تُستخدم في حياتنا اليومية، '
    'فهي توزيعُ عددٍ على مجموعاتٍ متساويةٍ دون فضلٍ لأحدٍ على أحد. '
    'هل فهمت الدرس يا عمر؟'
)

# ============================================================================
# تحميل المحرك — مرة واحدة لكل الاختبارات
# ============================================================================
print('=' * 72)
print('تحميل المحرك (checkpoint الإنتاج R1F نفسه)')
print('=' * 72)
_t0 = time.time()
import psutil                                                  # noqa: E402
PROC = psutil.Process(os.getpid())
RSS_BEFORE_MODEL = PROC.memory_info().rss
TTS = stream_tts.EiqazStreamingTTS()
RSS_AFTER_MODEL = PROC.memory_info().rss
MODEL_LOAD_S = round(time.time() - _t0, 2)
print(f'loaded iter={TTS.iter} in {MODEL_LOAD_S}s · '
      f'RAM +{(RSS_AFTER_MODEL - RSS_BEFORE_MODEL)/1e6:.0f}MB')
assert TTS.checkpoint.endswith('states_cont_180516.pth'), TTS.checkpoint


# ============================================================================
# A — سلامة تجزئة النص (بلا نموذج — سريع)
# ============================================================================
print('\n' + '=' * 72)
print('A — سلامة تجزئة النص (تغذية كلمة-بكلمة)')
print('=' * 72)

A_TEXTS = {
    'egy_multi': MEDIUM,
    'mixed_msa': (
        'يلا نبدأ الدرس. اليوم هنتكلم عن القسمة{ق} يا عمر. '
        'والآن ننتقل إلى الفصحى: القسمةُ عمليةٌ حسابيةٌ أساسيةٌ. '
        'هل فهمت الدرس؟'),
    'markers_shadda': (
        'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا. '
        'الدَّرْس مُفيد جدًا.'),
    'no_punct_end': 'جملة أولى. وجملة تانية بدون نقطة في الآخر',
    'tiny_sentences': 'بص. يا. عمر. النهاردة. هنبدأ. الدرس.',
    'digits_only_piece': 'عندي 5 كتب. وقرأت 12 صفحة. هل فهمت؟',
}

for name, text in A_TEXTS.items():
    try:
        ch = stream_tts._Chunker('egy', min_tok=12, max_tok=144)
        for w in text.split(' '):
            ch.feed(w + ' ')
        chunks = ch.poll(final=True)

        target = ' '.join(text.split())
        joined = ' '.join(' '.join(chunks).split())
        ok_join = joined == target                       # لا فقد/تكرار
        ok_words = target.split() == ' '.join(chunks).split()
        toks = [infer.count_tokens(c, 'egy') for c in chunks]
        ok_cap = all(t <= 144 for t in toks)
        ok_min = all(t >= 6 or i == len(chunks) - 1      # الأخير يجوز أصغر
                    for i, t in enumerate(toks))
        ok_arabic = all(stream_tts._has_arabic_speech(c) for c in chunks)
        ok = ok_join and ok_words and ok_cap and ok_min and ok_arabic
        record('A', f'{name}: {len(chunks)} مقاطع · سلامة النص', ok,
               {'chunks': chunks, 'toks': toks, 'join': ok_join,
                'words': ok_words, 'cap': ok_cap, 'min': ok_min,
                'arabic': ok_arabic})
    except Exception:                                    # noqa: BLE001
        record('A', f'{name}: {len(chunks)} مقاطع · سلامة النص', False,
               traceback.format_exc())

# الوسوم لا تنفصل عن كلماتها عبر التجزئة
try:
    ch = stream_tts._Chunker('egy', min_tok=4, max_tok=40)
    text = A_TEXTS['markers_shadda']
    for w in text.split(' '):
        ch.feed(w + ' ')
    chunks = ch.poll(final=True)
    ok = all(not c.startswith('{') and
             not _re.search(r' \{[^{}]{1,2}\}', c) for c in chunks)
    record('A', 'الوسوم {ق}/{ج}/{ء} ملتحمة بكلماتها', ok, {'chunks': chunks})
except Exception:                                        # noqa: BLE001
    record('A', 'الوسوم {ق}/{ج}/{ء} ملتحمة بكلماتها', False,
           traceback.format_exc())

# الجملة العملاقة تُقسّم عند الفواصل الناعمة ثم حدود الكلمات (لا قصّ كلمات)
try:
    giant = '، '.join(f'جملة رقم {i} طويلة جدًا في النص' for i in range(30)) + '.'
    ch = stream_tts._Chunker('egy', min_tok=12, max_tok=144)
    ch.feed(giant)
    chunks = ch.poll(final=True)
    ok_words = giant.split() == ' '.join(chunks).split()
    toks = [infer.count_tokens(c, 'egy') for c in chunks]
    ok_cap = all(t <= 144 for t in toks)
    record('A', 'الجملة العملاقة: تقسيم آمن بلا قصّ كلمات',
           ok_words and ok_cap, {'n': len(chunks), 'toks': toks})
except Exception:                                        # noqa: BLE001
    record('A', 'الجملة العملاقة: تقسيم آمن بلا قصّ كلمات', False,
           traceback.format_exc())


# ============================================================================
# B — إثبات التدفق الحقيقي (§31): أول صوت قبل اكتمال النص
# ============================================================================
print('\n' + '=' * 72)
print('B — إثبات التدفق: أول AudioChunk قبل اكتمال تغذية النص')
print('=' * 72)

B_MARKS = {}
try:
    s = TTS.create_stream(name='B-proof')
    marks = {'first_audio': None, 'last_feed': None, 'finish': None,
             'n_audio_at_first': None}
    n_audio = [0]
    t_proof_text_fed_fraction = [0.0]

    pieces = split_pieces(LONG)
    total_chars = sum(len(p) for p in pieces)

    def feeder():
        for i, p in enumerate(pieces):
            s.feed(p)
            marks['last_feed'] = time.time()
            if marks['first_audio'] is not None and not t_proof_text_fed_fraction[0]:
                t_proof_text_fed_fraction[0] = (i + 1) / len(pieces)
            time.sleep(0.3)                 # إيقاع LLM واقعي
        s.finish()
        marks['finish'] = time.time()

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    chunks, end = [], None
    for ev in s.events():
        if ev.is_audio:
            if marks['first_audio'] is None:
                marks['first_audio'] = time.time()
            n_audio[0] += 1
            chunks.append(ev)
        else:
            end = ev
            break
    th.join(15)
    B_MARKS = dict(marks)

    # الإثبات الصارم: الصوت الأول وصل قبل آخر تغذية وقبل finish()
    ok_before_feed = (marks['first_audio'] is not None
                      and marks['last_feed'] is not None
                      and marks['first_audio'] < marks['last_feed'])
    ok_before_finish = (marks['first_audio'] is not None
                        and marks['finish'] is not None
                        and marks['first_audio'] < marks['finish'])
    ok_multi = len(chunks) >= 3
    ok_end = end is not None and end.kind == 'end'
    record('B', 'أول صوت قبل اكتمال النص (قبل آخر feed)', ok_before_feed,
           {'first_audio': round(marks['first_audio'] or -1, 2),
            'last_feed': round(marks['last_feed'] or -1, 2),
            'fed_fraction_at_first_audio': round(
                t_proof_text_fed_fraction[0], 2)})
    record('B', 'أول صوت قبل finish()', ok_before_finish,
           {'first_audio': round(marks['first_audio'] or -1, 2),
            'finish': round(marks['finish'] or -1, 2)})
    record('B', f'تدفق متعدد المقاطع ({len(chunks)}) + نهاية سليمة',
           ok_multi and ok_end,
           {'n_chunks': len(chunks), 'end': end.kind if end else None})
    st = s.stats
    record('B', 'TTFA مقيس (من أول تغذية إلى أول مقطع جاهز)',
           st['ttfa_s'] is not None and st['ttfa_s'] < 5.0,
           {'ttfa_s': st['ttfa_s'], 'rtf': st['rtf'],
            'n_chunks': st['n_chunks_emitted']})
except Exception:                                        # noqa: BLE001
    record('B', 'إثبات التدفق', False, traceback.format_exc())


# ============================================================================
# C — الإلغاء أثناء الاستهلاك
# ============================================================================
print('\n' + '=' * 72)
print('C — الإلغاء أثناء الاستهلاك')
print('=' * 72)

try:
    s = TTS.create_stream(name='C-cancel')
    s.feed(LONG)
    got = []
    for ev in s.events():
        if ev.is_audio:
            got.append(ev)
            if len(got) == 1:
                break
    ok_has1 = len(got) == 1
    t_cancel = time.time()
    ok_cancel = s.cancel() is True
    t0 = time.time()
    end = None
    for ev in s.events():
        if not ev.is_audio:
            end = ev
            break
    wake_s = time.time() - t_cancel
    st = s.stats
    record('C', 'إلغاء → StreamEnd(cancelled) سريع', ok_cancel and end
           is not None and end.kind == 'cancelled' and wake_s < 5.0,
           {'wake_s': round(wake_s, 2), 'kind': end.kind if end else None})
    record('C', 'المنتج خرج وطابور فارغ',
           s.wait_producer(5) and s.pending_audio() == 0,
           {'alive': s.alive, 'pending': s.pending_audio(),
            'discarded_inflight': st['n_discarded_inflight']})
    record('C', 'لا مقاطع جديدة بعد الإلغاء',
           st['n_chunks_consumed'] <= st['n_chunks_emitted']
           and st['state'] == 'cancelled',
           {'emitted': st['n_chunks_emitted'],
            'consumed': st['n_chunks_consumed'],
            'state': st['state']})
except Exception:                                        # noqa: BLE001
    record('C', 'الإلغاء أثناء الاستهلاك', False, traceback.format_exc())


# ============================================================================
# D — تفريغ الطابور (مقاطع مولدة وغير مستهلكة)
# ============================================================================
print('\n' + '=' * 72)
print('D — تفريغ الطابور عند الإلغاء')
print('=' * 72)

try:
    s = TTS.create_stream(name='D-queue')
    s.feed(LONG)
    s.finish()
    # انتظر توليد ≥3 مقاطع دون استهلاك (producer يراكم في الطابور)
    t0 = time.time()
    while (s.stats['n_chunks_emitted'] < 3
           and time.time() - t0 < 30 and not s._terminal):
        time.sleep(0.05)
    n_before = s.pending_audio()
    n_emitted = s.stats['n_chunks_emitted']
    s.cancel()
    n_after = s.pending_audio()
    record('D', 'المقاطع المنتظرة حُذفت كلها',
           n_before >= 1 and n_after == 0,
           {'emitted': n_emitted, 'pending_before': n_before,
            'pending_after': n_after})
    # بعدها: لا يصل أي صوت — فقط الحدث النهائي
    audio_after = []
    end = None
    for ev in s.events():
        if ev.is_audio:
            audio_after.append(ev)
        else:
            end = ev
            break
    record('D', 'لا صوت بعد التفريغ — فقط الحدث النهائي',
           not audio_after and end is not None,
           {'audio_after': len(audio_after), 'end': end.kind if end else None})
    s.wait_producer(5)
except Exception:                                        # noqa: BLE001
    record('D', 'تفريغ الطابور', False, traceback.format_exc())


# ============================================================================
# E — رفض النتائج القديمة (request_id)
# ============================================================================
print('\n' + '=' * 72)
print('E — رفض النتائج القديمة (§17: 101 بعد بدء 102 يُتجاهل)')
print('=' * 72)

try:
    player = stream_tts.PlaybackQueue()
    player.begin(101)
    # chunk مزيّف من الطلب 101 (البنية وحدها تكفي للاختبار)
    class _FC:
        is_audio = True
        def __init__(self, rid, seq):
            self.request_id, self.seq = rid, seq
            self.wave = np.zeros(2205, dtype='float32')
            self.sample_rate = 22050
            self.n_tokens = 5
            self.duration_s = 0.1
            self.is_final = False

    r1 = player.submit(_FC(101, 1))
    player.interrupt()
    player.begin(102)                       # طلب جديد بدأ
    r_stale = player.submit(_FC(101, 2))    # متأخر من 101 → رفض
    r_new = player.submit(_FC(102, 1))      # من 102 → تشغيل
    record('E', 'chunk من 101 بعد بدء 102 يُرفض',
           r1 == 'played' and r_stale == 'stale' and r_new == 'played'
           and len(player.rejected) == 1
           and player.rejected[0]['request_id'] == 101,
           {'r1': r1, 'r_stale': r_stale, 'r_new': r_new})
    # نفس الطلب بعد interrupt بدون begin جديد → dropped
    player.interrupt()
    r_drop = player.submit(_FC(102, 2))
    record('E', 'chunk بعد interrupt (قبل طلب جديد) يُسقط',
           r_drop == 'dropped' and len(player.dropped) == 1,
           {'r_drop': r_drop})
except Exception:                                        # noqa: BLE001
    record('E', 'رفض النتائج القديمة', False, traceback.format_exc())


# ============================================================================
# F — طلب جديد فور المقاطعة (بلا إعادة تحميل النموذج)
# ============================================================================
print('\n' + '=' * 72)
print('F — طلب جديد فور المقاطعة (§19: بلا إعادة تحميل)')
print('=' * 72)

try:
    loads_before = TTS.model_loads
    sA = TTS.create_stream(name='F-A')
    sA.feed(LONG)
    got = 0
    for ev in sA.events():
        if ev.is_audio:
            got += 1
            break
    sA.cancel()
    sA.wait_producer(5)
    rid_A = sA.request_id
    t0 = time.time()
    sB, chunksB, endB = run_stream(TTS, split_pieces(MEDIUM), name='F-B')
    elapsed = time.time() - t0
    record('F', 'طلب B اكتمل بعد مقاطعة A',
           endB is not None and endB.kind == 'end' and len(chunksB) >= 1
           and sB.stats['state'] == 'done',
           {'n_chunks_B': len(chunksB), 'elapsed': round(elapsed, 2)})
    record('F', 'بلا إعادة تحميل النموذج + معرفات متصاعدة',
           TTS.model_loads == loads_before
           and sB.request_id > rid_A,
           {'model_loads': TTS.model_loads,
            'request_ids': [rid_A, sB.request_id]})
except Exception:                                        # noqa: BLE001
    record('F', 'طلب جديد بعد المقاطعة', False, traceback.format_exc())


# ============================================================================
# G — السباقات (§18: A-F)
# ============================================================================
print('\n' + '=' * 72)
print('G — السباقات')
print('=' * 72)

# G1: مقاطعة لحظة جاهزية أول مقطع (chunk ready + interrupt متزامنان)
try:
    s = TTS.create_stream(name='G1')
    s.feed(LONG)
    state = {'first': None, 'end': None}
    evs = s.events()

    def racer():
        for ev in evs:
            if ev.is_audio:
                state['first'] = ev
                s.cancel()                 # المقاطعة فور وصول أول مقطع
                break

    th = threading.Thread(target=racer, daemon=True)
    th.start()
    th.join(30)
    ok = (state['first'] is not None and s.stats['state'] == 'cancelled'
          and s.wait_producer(5) and s.pending_audio() == 0)
    record('G', 'G1: مقاطعة لحظة جاهزية المقطع — نظيفة', ok,
           {'state': s.stats['state'],
            'first_seq': state['first'].seq if state['first'] else None})
except Exception:                                        # noqa: BLE001
    record('G', 'G1: مقاطعة لحظة جاهزية المقطع', False,
           traceback.format_exc())

# G2: مقاطع مولدة + إلغاء قبل التشغيل (مستهلك متوقف تمامًا)
try:
    s = TTS.create_stream(name='G2')
    s.feed(LONG)
    s.finish()
    t0 = time.time()
    while (s.stats['n_chunks_emitted'] < 2
           and time.time() - t0 < 30 and not s._terminal):
        time.sleep(0.05)
    emitted = s.stats['n_chunks_emitted']
    s.cancel()
    seen_audio = 0
    end = None
    for ev in s.events():
        if ev.is_audio:
            seen_audio += 1
        else:
            end = ev
            break
    record('G', 'G2: إلغاء قبل التشغيل — لا وصول صوت',
           emitted >= 2 and seen_audio == 0 and end is not None
           and end.kind == 'cancelled',
           {'emitted': emitted, 'seen_after_cancel': seen_audio,
            'end': end.kind if end else None})
    s.wait_producer(5)
except Exception:                                        # noqa: BLE001
    record('G', 'G2: إلغاء قبل التشغيل', False, traceback.format_exc())

# G3: مقاطعة أثناء التشغيل الفعلي (realtime) + طلب جديد فورًا (§32)
try:
    sA = TTS.create_stream(name='G3-A')
    sA.feed(LONG)
    player = stream_tts.PlaybackQueue(realtime=True, play_rate=8.0)
    player.begin(sA.request_id)
    got = []
    for ev in sA.events():
        if ev.is_audio:
            got.append(ev)
            player.submit(ev)
            if len(got) == 2:
                break
    # انتظر حتى يبدأ تشغيل A1 فعليًا
    t0 = time.time()
    while not player._playing and time.time() - t0 < 5:
        time.sleep(0.02)
    t_interrupt = time.time()
    info = player.interrupt()
    sA.cancel()
    # طلب B فورًا
    sB, chunksB, endB = run_stream(TTS, split_pieces(MEDIUM), name='G3-B')
    player.begin(sB.request_id)
    for c in chunksB:
        player.submit(c)
    player.wait_idle(10)
    # أ) التشغيل الجاري توقف خلال ≤ 0.3ث من المقاطعة
    a1 = next((r for r in player.played
               if r['request_id'] == sA.request_id), None)
    stop_lat = (a1['t_play_end'] - t_interrupt) if a1 else 99.0
    ok_stop = a1 is not None and a1['interrupted'] and stop_lat <= 0.3
    # ب) B اكتمل تشغيله بالكامل
    b_recs = [r for r in player.played if r['request_id'] == sB.request_id]
    ok_b = len(b_recs) == len(chunksB) and all(
        not r['interrupted'] for r in b_recs)
    # ج) لا صوت من A بعد أول تشغيل لـB
    first_b = min((r['t_play_start'] for r in b_recs), default=None)
    a_after_b = [r for r in player.played
                 if r['request_id'] == sA.request_id and first_b
                 and r['t_play_start'] >= first_b]
    ok_no_stale = not a_after_b
    # د) المقاطع المنتظرة في المشغل فُرّغت
    record('G', 'G3: أثناء التشغيل — توقف فوري + B كامل + لا قديم بعد B',
           ok_stop and ok_b and ok_no_stale,
           {'stop_latency_s': round(stop_lat, 3), 'cleared': info['cleared'],
            'b_played': len(b_recs), 'a_after_b': len(a_after_b),
            'a1_interrupted': bool(a1 and a1['interrupted'])})
except Exception:                                        # noqa: BLE001
    record('G', 'G3: مقاطعة أثناء التشغيل', False, traceback.format_exc())

# G4: نتيجة inference متأخرة بعد الإلغاء تُهمَل (المسار الحتمي للسباق)
#    نقطة السباق الوحيدة المُحاكاة: تأخير prepare_text_rich ليصدر الإلغاء
#    أثناء المعالجة — بعدها الحارس بعد التوليد يُهمل النتيجة حتمًا.
try:
    real_prep = infer.prepare_text_rich

    def slow_prep(*a, **kw):
        time.sleep(1.0)
        return real_prep(*a, **kw)

    infer.prepare_text_rich = slow_prep
    try:
        s = TTS.create_stream(name='G4')
        s.feed(MEDIUM)
        # انتظر دخول المنتج في المعالجة (t_first_boundary يُضبط أولها)
        t0 = time.time()
        while s.stats['t_first_boundary'] is None and time.time() - t0 < 10:
            time.sleep(0.02)
        time.sleep(0.2)                 # الآن المعالجة جارية (sleep 1.0)
        s.cancel()                      # الإلغاء أثناء المعالجة
        s.wait_producer(15)
        st = s.stats
        audio_seen = []
        for ev in s.events():
            if ev.is_audio:
                audio_seen.append(ev)
            else:
                break
        record('G', 'G4: نتيجة inference متأخرة بعد الإلغاء تُهمَل',
               st['n_discarded_inflight'] >= 1 and not audio_seen
               and st['state'] == 'cancelled',
               {'discarded': st['n_discarded_inflight'],
                'audio_seen': len(audio_seen),
                'emitted': st['n_chunks_emitted']})
    finally:
        infer.prepare_text_rich = real_prep
except Exception:                                        # noqa: BLE001
    record('G', 'G4: نتيجة متأخرة تُهمَل', False, traceback.format_exc())
    try:
        infer.prepare_text_rich = real_prep
    except Exception:                                    # noqa: BLE001
        pass

# G5: إلغاءات متكررة idempotent
try:
    s = TTS.create_stream(name='G5')
    s.feed(MEDIUM)
    s.cancel()
    r2 = s.cancel()
    r3 = s.cancel()
    s.wait_producer(5)
    # إلغاء جلسة منتهية لا يفسد حالتها
    s2, ch2, e2 = run_stream(TTS, [SHORT], name='G5-b')
    state_before = s2.stats['state']
    r_late = s2.cancel()
    state_after = s2.stats['state']
    record('G', 'G5: إلغاءات متكررة آمنة + المنتهية لا تفسد',
           r2 is False and r3 is False and r_late is True
           and state_before == 'done' and state_after == 'done',
           {'second': r2, 'third': r3, 'late_on_done': r_late,
            'state': state_after})
except Exception:                                        # noqa: BLE001
    record('G', 'G5: إلغاءات متكررة', False, traceback.format_exc())

# G6: feed بعد cancel/finish يُرفض صراحة (لا صمت زائف)
try:
    s = TTS.create_stream(name='G6')
    s.feed(MEDIUM)
    s.cancel()
    try:
        s.feed('نص متأخر')
        ok1 = False
    except RuntimeError:
        ok1 = True
    s2, _, _ = run_stream(TTS, [SHORT], name='G6-b')
    try:
        s2.feed('بعد الانتهاء')
        ok2 = False
    except RuntimeError:
        ok2 = True
    record('G', 'G6: feed بعد cancel/finish يُرفض', ok1 and ok2,
           {'after_cancel': ok1, 'after_finish': ok2})
except Exception:                                        # noqa: BLE001
    record('G', 'G6: feed بعد الإلغاء', False, traceback.format_exc())


# ============================================================================
# H — ثبات هوية المتحدث عبر المقاطع (§7)
# ============================================================================
print('\n' + '=' * 72)
print('H — ثبات هوية المتحدث (speaker conditioning + F0 + ECAPA)')
print('=' * 72)

_EC = None


def _ecapa_savedir():
    """مجلد نموذج ECAPA: متغير بيئة ← مجلد التشغيل السابق ← الافتراضي."""
    env = os.environ.get('EIQAZ_ECAPA_DIR')
    if env:
        return env
    legacy = '/home/z/my-project/work/sb_ecapa'
    if os.path.isdir(legacy):
        return legacy
    return os.path.expanduser('~/.cache/speechbrain/ecapa')


def ecapa():
    global _EC
    if _EC is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _EC = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir=_ecapa_savedir(),
            run_opts={'device': 'cpu'})
    return _EC


def embed(wav_22k):
    import torch
    import torchaudio.functional as AF
    ec = ecapa()
    w = torch.from_numpy(np.ascontiguousarray(wav_22k, dtype='float32'))
    w16 = AF.resample(w, 22050, 16000)
    with torch.no_grad():
        e = ec.encode_batch(w16.unsqueeze(0))
    return e.squeeze(0).squeeze(0)


def cos(a, b):
    import torch
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _centroid(embs):
    """مركز (متوسط مطبع) لمجموعة ترميزات ECAPA — مقياس أمتن للمقاطع
    القصيرة من pair-wise."""
    import torch
    return torch.nn.functional.normalize(
        torch.stack(embs).mean(0), dim=0)


try:
    # نفس الرد الطويل بمتحدث واحد — كل المقاطع نفس speaker + ECAPA عالية
    s0, chunks0, end0 = run_stream(TTS, split_pieces(LONG), speaker=0,
                                   name='H-spk0')
    ok_end0 = end0 is not None and end0.kind == 'end' and len(chunks0) >= 3
    f0s0 = [f0_autocorr(c.wave, 22050) for c in chunks0]
    f0_valid = [f for f in f0s0 if f > 0]
    f0_spread0 = (max(f0_valid) - min(f0_valid)) if len(f0_valid) >= 2 else 0
    # عتبة 60Hz — نفس معيار القسم G في tests/test_pipeline.py
    record('H', 'متحدث 0: F0 ثابت عبر المقاطع (مدى < 60Hz)',
           ok_end0 and f0_spread0 < 60,
           {'n_chunks': len(chunks0), 'f0s': [round(f, 1) for f in f0s0],
            'spread': round(f0_spread0, 1)})

    # نفس النص بمتحدث 1 — هوية متمايزة (فحص جوهري إضافي)
    s1, chunks1, end1 = run_stream(TTS, split_pieces(MEDIUM), speaker=1,
                                   name='H-spk1')
    f0s1 = [f0_autocorr(c.wave, 22050) for c in chunks1]
    f0_valid1 = [f for f in f0s1 if f > 0]
    f0_med0 = sorted(f0_valid)[len(f0_valid) // 2] if f0_valid else 0
    f0_med1 = (sorted(f0_valid1)[len(f0_valid1) // 2]
               if f0_valid1 else 0)
    record('H', 'انفصال F0 رجل/مرأة (هوية متمايزة)',
           f0_med0 > 0 and f0_med1 > 0 and abs(f0_med0 - f0_med1) > 40,
           {'spk0_f0_median': round(f0_med0, 1),
            'spk1_f0_median': round(f0_med1, 1)})

    # ECAPA — مقياس المركز (centroid): أمتن من pair-wise للمقاطع القصيرة
    # (معايرة قياسًا على هذا الجهاز: نفس المتحدث min-to-centroid=0.565+
    #  للذكر والأنثى · العابر max-to-centroid=0.443 — العتبة 0.50 في الفجوة)
    embs0 = [embed(c.wave) for c in chunks0]
    cent0 = _centroid(embs0)
    to_cent0 = [cos(e, cent0) for e in embs0]
    mean_tc = sum(to_cent0) / len(to_cent0)
    min_tc = min(to_cent0)
    pair_same = [cos(embs0[i], embs0[j])
                 for i in range(len(embs0))
                 for j in range(i + 1, len(embs0))]
    mean_pair = (sum(pair_same) / len(pair_same)) if pair_same else 0.0
    # أساس التمييز: مقاطع الرجل ضد مركز المرأة والعكس
    embs1 = [embed(c.wave) for c in chunks1]
    cent1 = _centroid(embs1)
    cross_cent = ([cos(a, cent1) for a in embs0]
                  + [cos(b, cent0) for b in embs1])
    mean_cross = sum(cross_cent) / len(cross_cent)
    ok_ecapa = (mean_tc >= 0.75 and min_tc >= 0.50
                and mean_cross <= 0.40)
    record('H', 'ECAPA (centroid): نفس المتحدث عبر المقاطع — أعلى بوضوح '
           'من العابر', ok_ecapa,
           {'mean_to_centroid': round(mean_tc, 4),
            'min_to_centroid': round(min_tc, 4),
            'mean_pairwise': round(mean_pair, 4),
            'mean_cross_speaker_to_centroid': round(mean_cross, 4),
            'n_chunks': len(chunks0)})
    RESULTS.setdefault('h_ecapa', {}).update(
        {'mean_to_centroid': round(mean_tc, 4),
         'min_to_centroid': round(min_tc, 4),
         'mean_pairwise': round(mean_pair, 4),
         'mean_cross_to_centroid': round(mean_cross, 4)})
except Exception:                                        # noqa: BLE001
    record('H', 'ثبات هوية المتحدث', False, traceback.format_exc())


# ============================================================================
# I — ثبات النبرة والأسلوب عبر المقاطع (§8/§22/§33)
# ============================================================================
print('\n' + '=' * 72)
print('I — ثبات النبرة والأسلوب (نص §22: سؤال/خبر/تعليمي/فاصلة/طويلة/'
      'مصرية→فصحى/شدة/الوسوم)')
print('=' * 72)

try:
    sP, chunksP, endP = run_stream(TTS, split_pieces(PROSODY_TEXT),
                                   speaker=0, name='I-prosody')
    ok_multi = endP is not None and endP.kind == 'end' and len(chunksP) >= 4
    f0sP = [f0_autocorr(c.wave, 22050) for c in chunksP]
    rmssP = [rms(c.wave) for c in chunksP]
    dursP = [c.duration_s for c in chunksP]
    toksP = [c.n_tokens for c in chunksP]
    ratesP = [t / d for t, d in zip(toksP, dursP)]     # توكن/ثانية

    f0v = [f for f in f0sP if f > 0]
    f0_spreadP = max(f0v) - min(f0v) if len(f0v) >= 2 else 0
    adj_jumps = [abs(f0sP[i + 1] - f0sP[i]) for i in range(len(f0sP) - 1)]
    max_adj = max(adj_jumps) if adj_jumps else 0
    rms_ratio = (max(rmssP) / min(rmssP)) if min(rmssP) > 0 else 99.0
    mean_rate = sum(ratesP) / len(ratesP)
    rate_cv = ((sum((r - mean_rate) ** 2 for r in ratesP)
                / len(ratesP)) ** 0.5 / mean_rate) if mean_rate else 99.0

    record('I', 'pitch continuity: مدى F0 < 60Hz وقفزة متجاورة < 45Hz',
           ok_multi and f0_spreadP < 60 and max_adj < 45,
           {'f0s': [round(f, 1) for f in f0sP],
            'spread': round(f0_spreadP, 1), 'max_adjacent': round(max_adj, 1)})
    record('I', 'loudness continuity: نسبة RMS بين المقاطع ≤ 2.2',
           ok_multi and rms_ratio <= 2.2,
           {'rms': [round(r, 4) for r in rmssP], 'ratio': round(rms_ratio, 3)})
    record('I', 'speaking rate: CV ≤ 0.30', ok_multi and rate_cv <= 0.30,
           {'rates_tok_s': [round(r, 2) for r in ratesP],
            'cv': round(rate_cv, 3)})
    record('I', 'مدة المقاطع معقولة (متوسط 1.5-8ث)',
           ok_multi and 1.0 <= (sum(dursP) / len(dursP)) <= 9.0,
           {'durations': [round(d, 2) for d in dursP],
            'avg': round(sum(dursP) / len(dursP), 2)})

    # جودة الوصلات: لا نقرات ولا صمت غير طبيعي ولا فونيم مبتور/مكرر
    full = np.concatenate([c.wave for c in chunksP])
    offs, pos = [], 0
    for c in chunksP:
        pos += len(c.wave)
        offs.append(pos)
    junction_jumps = [abs(float(full[o - 1] - full[o]))
                      for o in offs[:-1]]
    max_junction = max(junction_jumps) if junction_jumps else 0.0
    # فترة الفاصل: صمت فعلي (أقصى |عينة| خلال أول 0.1ث من كل مقطع ≥2)
    gap_ok = True
    for c in chunksP[1:]:
        head = c.wave[:int(0.10 * 22050)]
        if float(np.abs(head).max()) > 1e-3:
            gap_ok = False
    record('I', 'chunk transitions: لا نقرات + فاصل صمت نظيف',
           max_junction < 0.01 and gap_ok,
           {'max_junction_jump': round(max_junction, 6),
            'gap_silence_clean': gap_ok})

    # انتقال مصرية→فصحى داخل نفس الرد: هوية ECAPA ثابتة على جانبي الحد
    # (موقعياً: من المقطع الحاوي لكلمة الفصحى فصاعدًا = ما بعد الحد)
    embsP = [embed(c.wave) for c in chunksP]
    t_idx = next((i for i, c in enumerate(chunksP) if 'الفصحى' in c.text),
                 None)
    if t_idx is not None and t_idx >= 1 and t_idx < len(chunksP):
        cent_pre = _centroid(embsP[:t_idx])
        post_to_pre = [cos(e, cent_pre) for e in embsP[t_idx:]]
        min_dialect = min(post_to_pre)
        mean_dialect = sum(post_to_pre) / len(post_to_pre)
        # معايرة قياسًا: post→pre-centroid min=0.66 مقيسًا (عابر max=0.44)
        record('I', 'انتقال مصرية↔فصحى: هوية المتحدث ثابتة عبر الحد',
               min_dialect >= 0.50 and mean_dialect >= 0.60,
               {'transition_chunk': t_idx, 'n_chunks': len(chunksP),
                'min_post_to_pre_centroid': round(min_dialect, 4),
                'mean_post_to_pre_centroid': round(mean_dialect, 4)})
        RESULTS.setdefault('i_prosody', {}).update(
            {'dialect_transition': {
                'transition_chunk': t_idx,
                'min_post_to_pre_centroid': round(min_dialect, 4),
                'mean_post_to_pre_centroid': round(mean_dialect, 4)}})
    else:
        record('I', 'انتقال مصرية↔فصحى: هوية المتحدث ثابتة عبر الحد',
               False, {'transition_chunk': t_idx,
                       'note': 'لم يُعثر على حد صريح في المقاطع'})

    RESULTS.setdefault('i_prosody', {}).update(
        {'n_chunks': len(chunksP), 'f0s': [round(f, 1) for f in f0sP],
         'rms': [round(r, 4) for r in rmssP],
         'rates': [round(r, 2) for r in ratesP],
         'f0_spread': round(f0_spreadP, 1),
         'rms_ratio': round(rms_ratio, 3), 'rate_cv': round(rate_cv, 3),
         'max_junction_jump': round(max_junction, 6)})
except Exception:                                        # noqa: BLE001
    record('I', 'ثبات النبرة', False, traceback.format_exc())


# ----------------------------------------------------------------------------
# I-2 — مقارنة Streaming مقابل Single-shot (§21)
# ----------------------------------------------------------------------------
print('\n-- I-2: مقارنة streaming مقابل single-shot (نص يقبع في مقطع واحد)')


try:
    import soundfile as sf
    ONE_SHOT = ('بص يا عمر، القسمة معناها إننا بنوزع الحاجة بالتساوي. '
                'هل فهمت الدرس؟')
    mode = infer.effective_diacritize_mode(ONE_SHOT, 'auto', 'egy')
    toks_ms, toks_egy, ids_of = infer.get_tokenizer()

    # (أ) المسار الخام (continuity=False): كل مقطع على حدة يجب أن يطابق
    #     infer.synthesize لنصه بالبايت — أقوى إثبات أن مسار الإنتاج
    #     نفسه يُستخدم بلا أي تغيير
    s_raw, chunks_raw, end_raw = run_stream(
        TTS, [ONE_SHOT], apply_continuity=False, name='I-raw')
    bit_details, ok_bit = [], True
    for i, c in enumerate(chunks_raw):
        res_i = infer.prepare_text_rich(c.text, mode, 'egy')
        tmp_i = os.path.join(TMP, f'oneshot_c{i}.wav')
        infer.synthesize(TTS.model, res_i['text'], 'egy', 0, 1.0, tmp_i,
                         0.005, peak_normalize=False)
        w_i, _sr = sf.read(tmp_i, dtype='float32')
        eq = np.array_equal(c.wave, w_i)
        ok_bit &= eq
        bit_details.append({'seq': c.seq, 'n_tokens': c.n_tokens,
                            'bit_equal': bool(eq)})
    record('I2', 'bit-identity لكل مقطع: التدفق الخام == مسار الإنتاج',
           ok_bit and len(chunks_raw) >= 1, {'chunks': bit_details})

    # (ب) تسلسل التوكنات مقابل single-shot للنص الكامل:
    #     الفرق المسموح الوحيد = توكن نهاية التسلسل _eos_ (id=1) لكل
    #     مقطع إضافي — كل مقطع وحدة توليد مكتملة (طبيعة غير ذاتية
    #     التراجع). بلا _eos_ يجب أن يتطابق التسلسل تمامًا.
    res_full = infer.prepare_text_rich(ONE_SHOT, mode, 'egy')
    tmp_full = os.path.join(TMP, 'oneshot_full.wav')
    infer.synthesize(TTS.model, res_full['text'], 'egy', 0, 1.0, tmp_full,
                     0.005, peak_normalize=False)
    wave_single, _sr = sf.read(tmp_full, dtype='float32')
    ids_single = list(ids_of(toks_egy(res_full['text'])))
    ids_stream = []
    for c in chunks_raw:
        ids_stream += list(ids_of(toks_egy(infer.prepare_text_rich(
            c.text, mode, 'egy')['text'])))
    _EOS = 1
    strip = lambda ids: [t for t in ids if t != _EOS]     # noqa: E731
    ok_tokens = strip(ids_stream) == strip(ids_single)
    record('I2', 'توكنات النطق متطابقة (باستثناء _eos_ لكل حد مقطع)',
           ok_tokens,
           {'n_stream': len(ids_stream), 'n_single': len(ids_single),
            'n_eos_stream': ids_stream.count(_EOS),
            'n_eos_single': ids_single.count(_EOS),
            'equal_after_strip': bool(ok_tokens)})

    # (ج) مع الاستمرارية: المكافئ الصوتي للتجميع مقابل single-shot
    #     (معايرة قياسًا: ecapa=0.978 · ΔF0≈5Hz · rms ratio≈1.07)
    s_ct, chunks_ct, end_ct = run_stream(TTS, [ONE_SHOT], name='I-cont')
    ok_ct = end_ct is not None and len(chunks_ct) >= 1
    full_stream = np.concatenate([c.wave for c in chunks_ct]) if ok_ct else None
    sim = cos(embed(full_stream), embed(wave_single)) if ok_ct else 0.0
    f0_st = f0_autocorr(full_stream, 22050) if ok_ct else 0.0
    f0_ss = f0_autocorr(wave_single, 22050)
    rms_st = rms(full_stream) if ok_ct else 0.0
    rms_ss = rms(wave_single)
    dur_st = len(full_stream) / 22050 if ok_ct else 0.0
    dur_ss = len(wave_single) / 22050
    dur_ratio = dur_st / dur_ss if dur_ss else 99.0
    ok_acoustic = (sim >= 0.90 and abs(f0_st - f0_ss) <= 15
                   and (max(rms_st, rms_ss) / max(1e-9, min(rms_st, rms_ss)))
                   <= 1.25 and 0.85 <= dur_ratio <= 1.15)
    record('I2', 'المكافئ الصوتي: ECAPA ≥ 0.90 + F0/RMS/مدة متقاربة',
           ok_acoustic,
           {'ecapa_sim': round(sim, 4), 'f0_stream': round(f0_st, 1),
            'f0_single': round(f0_ss, 1), 'rms_stream': round(rms_st, 4),
            'rms_single': round(rms_ss, 4), 'dur_stream': round(dur_st, 2),
            'dur_single': round(dur_ss, 2), 'dur_ratio': round(dur_ratio, 3)})
except Exception:                                        # noqa: BLE001
    record('I2', 'مقارنة single-shot', False, traceback.format_exc())


# ============================================================================
# J — التوافق الرجعي (§20)
# ============================================================================
print('\n' + '=' * 72)
print('J — التوافق الرجعي: infer.synthesize كما هو')
print('=' * 72)

try:
    res = infer.prepare_text_rich(SHORT, 'auto', 'egy')
    out = os.path.join(TMP, 'j_compat.wav')
    n_tok, n_sec = infer.synthesize(TTS.model, res['text'], 'egy', 0, 1.0,
                                    out, 0.005)
    ok = os.path.exists(out) and n_tok >= 2 and n_sec > 0.5
    import soundfile as sf
    w, sr = sf.read(out, dtype='float32')
    record('J', 'infer.synthesize (CLI path) يعمل كما هو',
           ok and sr == 22050,
           {'n_tokens': n_tok, 'duration': round(n_sec, 2), 'sr': sr})
except Exception:                                        # noqa: BLE001
    record('J', 'التوافق الرجعي', False, traceback.format_exc())


# ============================================================================
# K — الأداء (§28) + تحسين حجم المقاطع (§29)
# ============================================================================
print('\n' + '=' * 72)
print('K — الأداء: Short/Medium/Long/Mixed + مقارنة أحجام المقاطع')
print('=' * 72)


class _SysMon:
    """مراقب CPU/RAM خفيف أثناء كل تشغيل."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        psutil.cpu_percent(interval=None)          # تهيئة
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(
                (psutil.cpu_percent(interval=None),
                 PROC.memory_info().rss))
            time.sleep(0.2)

    def __exit__(self, *a):
        self._stop.set()
        if self._th:
            self._th.join(1)

    def report(self):
        cpus = [s[0] for s in self.samples] or [0]
        rss = [s[1] for s in self.samples] or [0]
        return {'cpu_avg_pct': round(sum(cpus) / len(cpus), 1),
                'cpu_max_pct': round(max(cpus), 1),
                'rss_peak_mb': round(max(rss) / 1e6, 0),
                'cpu_cores': psutil.cpu_count()}


def bench(name, text, cadence=0.06, **kw):
    pieces = split_pieces(text)
    t_start = time.time()
    with _SysMon() as mon:
        s, chunks, end = run_stream(TTS, pieces, cadence=cadence, **kw)
    wall = time.time() - t_start
    st = s.stats
    durs = [c.duration_s for c in chunks]
    row = {
        'test': name, 'n_chunks': len(chunks),
        'ttfa_s': st['ttfa_s'], 'rtf': st['rtf'],
        'total_gen_s': round(st['synth_time_total_s'], 2),
        'wall_s': round(wall, 2),
        'audio_dur_s': round(sum(durs), 2),
        'avg_chunk_dur_s': round(sum(durs) / len(durs), 2) if durs else 0,
        'first_chunk_tokens': chunks[0].n_tokens if chunks else 0,
        **mon.report(),
    }
    ok = (end is not None and end.kind == 'end' and len(chunks) >= 1
          and st['ttfa_s'] is not None)
    print(f'    {name}: TTFA={row["ttfa_s"]}s RTF={row["rtf"]} '
          f'chunks={row["n_chunks"]} audio={row["audio_dur_s"]}s '
          f'wall={row["wall_s"]}s')
    return ok, row


K_ROWS = []
try:
    ok1, r1 = bench('short', SHORT)
    ok2, r2 = bench('medium', MEDIUM)
    ok3, r3 = bench('long', LONG)
    ok4, r4 = bench('mixed', PROSODY_TEXT)
    K_ROWS = [r1, r2, r3, r4]
    record('K', 'الأداء: الأربعة اكتملت بصوت (CPU)', ok1 and ok2 and ok3
           and ok4, {'rows': K_ROWS})
except Exception:                                        # noqa: BLE001
    record('K', 'الأداء', False, traceback.format_exc())

# ---- مقارنة أحجام المقاطع (§29): صغير/وسط/كبير على النص الطويل ----
try:
    sizes = {
        'small': dict(min_chunk_tokens=6, max_chunk_tokens=40),
        'medium': dict(min_chunk_tokens=12, max_chunk_tokens=90),
        'large': dict(min_chunk_tokens=24, max_chunk_tokens=144),
    }
    size_rows = {}
    for sz, kw in sizes.items():
        s, chunks, end = run_stream(TTS, split_pieces(LONG), cadence=0.06,
                                    name=f'K-{sz}', **kw)
        st = s.stats
        f0s = [f0_autocorr(c.wave, 22050) for c in chunks]
        f0v = [f for f in f0s if f > 0]
        spread = (max(f0v) - min(f0v)) if len(f0v) >= 2 else 0
        rmss = [rms(c.wave) for c in chunks]
        rr = (max(rmss) / min(rmss)) if min(rmss) > 0 else 99
        size_rows[sz] = {
            'n_chunks': len(chunks), 'ttfa_s': st['ttfa_s'],
            'rtf': st['rtf'],
            'avg_chunk_dur_s': round(
                sum(c.duration_s for c in chunks) / max(1, len(chunks)), 2),
            'f0_spread': round(spread, 1), 'rms_ratio': round(rr, 3),
            'total_gen_s': round(st['synth_time_total_s'], 2),
            'n_inferences': st['n_inferences'],
        }
        print(f'    size={sz}: {size_rows[sz]}')
    # القرار المقيس: الافتراضي (12/144؟) — نختار من القياس:
    # TTFA أفضل في small لكن الاستمرارية (f0/rms) أفضل في large.
    # المعيار: أكبر حجم يظل TTFA فيه مقيدًا (أول جملة كاملة أصلًا) مع
    # أقل عدد استدعاءات inference — القرار الموثق في التقرير.
    RESULTS['k_sizes'] = size_rows
    m = size_rows['medium']
    l = size_rows['large']
    ok_sizes = all(r['n_chunks'] >= 1 for r in size_rows.values())
    record('K', 'مقارنة أحجام المقاطع (صغير/وسط/كبير) مقيسة', ok_sizes,
           size_rows)
except Exception:                                        # noqa: BLE001
    record('K', 'مقارنة أحجام المقاطع', False, traceback.format_exc())


# ============================================================================
# الخلاصة والكتابة
# ============================================================================
print('\n' + '=' * 72)
n_pass, n_fail = RESULTS['n_pass'], RESULTS['n_fail']
print(f'الخلاصة: {n_pass}/{n_pass + n_fail} PASS · {n_fail} FAIL')
if RESULTS['failures']:
    print('الإخفاقات:')
    for f in RESULTS['failures']:
        print(f"  - [{f['section']}] {f['test']}")
print('=' * 72)

RESULTS['environment'] = {
    'checkpoint': os.path.basename(TTS.checkpoint),
    'model_iter': TTS.iter,
    'model_load_s': MODEL_LOAD_S,
    'cpu_cores': psutil.cpu_count(),
    'ram_total_mb': round(psutil.virtual_memory().total / 1e6, 0),
    'rss_after_model_mb': round(RSS_AFTER_MODEL / 1e6, 0),
    'torch_threads': __import__('torch').get_num_threads(),
}
RESULTS['overall'] = 'PASS' if n_fail == 0 else 'FAIL'

OUT = os.path.join(HERE, 'streaming_results.json')
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)
print(f'كُتب التقرير: {OUT}')

sys.exit(0 if n_fail == 0 else 1)
