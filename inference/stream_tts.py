#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eiqaz TTS — واجهة التوليد المتدفق (Streaming) + المقاطعة + ثبات النبرة
============================================================================
طبقة واجهة فوق مسار الإنتاج R1F نفسه (infer.py) — بلا أي تعديل في النموذج
أو الأوزان أو الترميز أو قواعد النطق:

    نص متدفق (LLM)
        ↓  stream.feed(...)
    مخزن نص آمن (Text Buffer)
        ↓
    مجزّع واعٍ بالنبرة (Prosody-aware Chunker)
        حدود جملة ← فواصل ناعمة ← حدود كلمات كملاذ أخير
        + دمج الجمل القصيرة (فلسفة PATCH 14 نفسها)
        ↓
    معالجة R1F الموحدة (infer.prepare_text_rich — بلا مساس)
        ↓
    R1F (MixerTTS · states_cont_180516.pth) — نفس speaker/emotion/pace
        ↓
    المُصوِّت vocos22.onnx (نفس denoise)
        ↓
    استمرارية سببية (trim + مرساة RMS جارية + تلاشي قصير + فاصل)
        ↓
    Audio Queue (لكل طلب)
        ↓
    المستهلك (Consumer): سطر أوامر / سطح مكتب / متصفح — أي منصة

لماذا مستوى المقاطع لا التدفّق داخل المقطع؟
  MixerTTS غير ذاتي التراجع (non-autoregressive): ينظر إلى توكنات المقطع
  كاملة ثم يتنبأ بالمدة والنبرة ويولّد الميل دفعة واحدة. لا يوجد أي
  state صوتي أو latent يُحفظ بين النداءات — لا يوجد في المعمارية ما يمكن
  «مواصلته». لذلك وحدة التدفق الطبيعية = مقطع جُملي كامل، وهو ما دُرّب
  عليه النموذج أصلًا (وحدات متعددة الجمل حتى 160 توكن).

ثبات هوية المتحدث:
  نفس speaker ID (conditioning متطابق عبر speaker_emb) + emotion=0 +
  نفس pace/لهجة/تشكيل لكل مقطع — الصوت واحد بالبناء، ويثبت بالقياس
  (F0 + ECAPA عبر المقاطع) في tests/test_streaming.py.

ثبات النبرة/الأسلوب:
  (1) نفس الـconditioning أعلاه · (2) تجزئة عند حدود جملية طبيعية مع دمج
  الجمل القصيرة حتى 144 توكن (90% من سقف التدريب — داخل توزيع التدريب
  تمامًا كما في PATCH 14) · (3) استمرارية صوتية سببية على مستوى الموجة:
  قصّ الصمت الحدي، مرساة جهارة RMS جارية (بلا معرفة المستقبل — بديل
  الوسيط غير السببي في PATCH 13)، تلاشي 8ms عند الحواف، فاصل صمت ثابت.
  النبرة داخل كل مقطع تبقى نبرة النموذج الطبيعية للجملة (سؤال/خبر...)
  والانتقال بين المقاطع عند حدود جمل = إعادة ضبط إعراب طبيعية وليس
  «تسجيلات ملصقة» — الحدود المقبولة موثقة في STREAMING.md.

الإلغاء (Interrupt) — بلا ادعاء زائف:
  cancel() يوقف التوليد القادم فورًا (أعلام تُفحص قبل/بعد كل مرحلة)،
  يفرّغ الطابور حالًا، ويرفض أي نتيجة متأخرة (request_id قديم). جارٍ
  inference داخل torch لا يمكن إلغاؤه فعليًا من بايثون — يُستكمل ثم
  يُهمَل الناتج (discard) ويُحصى في stats.n_discarded_inflight — هذا هو
  الفرق الموثق بين «إلغاء حقيقي» و«إبطال نتيجة».

توافق رجعي كامل: infer.py / webapp.py / CLI لم تُمس. الوحدة جديدة تمامًا
وتستورد مسار الإنتاج كما هو (prepare_text_rich + synthesize عبر ملف WAV
مؤقت — نفس نمط webapp.run_job بالحرف).

الاستخدام:

    from stream_tts import EiqazStreamingTTS, PlaybackQueue

    tts = EiqazStreamingTTS()                    # يحمّل R1F مرة واحدة
    stream = tts.create_stream(speaker=0)        # طلب جديد (request_id)
    stream.feed('بص يا عمر،')                    # نص LLM متدفق
    stream.feed(' القسمة معناها إننا بنوزع الحاجة بالتساوي.')
    stream.finish()
    for ev in stream:                            # AudioChunk ثم StreamEnd
        if ev.is_audio:
            play(ev.wave)                        # المستهلك — أي منصة
    ...
    stream.cancel()                              # مقاطعة في أي لحظة

اختبار الإلغاء الكامل (طلب جديد فور المقاطعة بلا إعادة تحميل):

    player = PlaybackQueue(realtime=True)
    player.begin(stream.request_id)
    player.submit(chunk)
    player.interrupt()                           # يوقف التشغيل ويفرّغ الطابور
"""

import collections
import itertools
import os
import re
import sys
import tempfile
import threading
import time

# --- Windows console safety: never crash on non-UTF8 consoles -------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import infer                       # مسار الإنتاج R1F — مصدر الحقيقة الوحيد

SAMPLE_RATE = 22050                # نفس infer.synthesize

__all__ = ['AudioChunk', 'StreamEnd', 'StreamSession',
           'EiqazStreamingTTS', 'PlaybackQueue']


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1) حدود التجزيع — نفس أنماط webapp المُثبتة (PATCH 12/14)
# ============================================================================
_SENT_END = re.compile(r'(?<=[.!؟?…])\s+')       # نهاية جملة + فراغ تالٍ
_SOFT_SPLIT = re.compile(r'(?<=[,،؛;:])\s+')      # فواصل ناعمة
_TAG_BIND_RE = re.compile(                        # PATCH 12: لصق الوسم بكلمته
    r'([\u0621-\u063A\u0641-\u064A\u064B-\u0652])\s+(\{[^{}]{1,2}\})')


def _bind_markers(text):
    """PATCH 12 — الوسم المنفصل يلتحق بكلمته (لا ينفصل عند التجزئة)."""
    return _TAG_BIND_RE.sub(r'\1\2', text)


def _has_arabic_speech(text, dialect='egy'):
    """هل يحمل النص كلامًا عربيًا منطوقًا؟ (بعد التطبيع — «5» تصير «خمسة»)

    الفحص على النص المطبع لا الخام حتى لا تضيع جملة رقمية خالصة
    (تحسين صغير على مرشح webapp الخام — قرار عزل فقط، لا يمس النطق)."""
    try:
        norm = infer.eqz_text.normalize_text(text, dialect)
    except Exception:                                   # noqa: BLE001
        norm = text
    return bool(infer._AR_LETTERS.search(norm))


# ============================================================================
# 2) الأحداث — AudioChunk / StreamEnd
# ============================================================================
class AudioChunk:
    """مقطع صوتي جاهز للتشغيل فور وصوله.

    wave: np.float32 أحادية القناة بمعدل sample_rate (22050).
    request_id/seq: هوية الطلب وترتيب المقطع داخله — لرفض نتائج قديمة.
    is_final: آخر مقطع في الطلب (لا فاصل صمت يُلحق به).
    """
    is_audio = True

    def __init__(self, request_id, seq, wave, sample_rate, n_tokens,
                 duration_s, synth_time_s, t_ready, text, is_final,
                 gain_applied, trimmed_s):
        self.request_id = request_id
        self.seq = seq
        self.wave = wave
        self.sample_rate = sample_rate
        self.n_tokens = n_tokens
        self.duration_s = duration_s
        self.synth_time_s = synth_time_s
        self.t_ready = t_ready
        self.text = text
        self.is_final = is_final
        self.gain_applied = gain_applied
        self.trimmed_s = trimmed_s

    def __repr__(self):
        return (f'<AudioChunk req={self.request_id} seq={self.seq} '
                f'{self.duration_s:.2f}s {self.n_tokens}tok '
                f'final={self.is_final}>')


class StreamEnd:
    """حدث نهاية الطلب — يُسلَّم مرة واحدة بالضبط بعد آخر AudioChunk."""
    is_audio = False

    def __init__(self, request_id, kind, stats, error=None):
        self.request_id = request_id
        self.kind = kind          # 'end' | 'cancelled' | 'error'
        self.stats = stats
        self.error = error

    def __repr__(self):
        return f'<StreamEnd req={self.request_id} kind={self.kind}>'


class _CancelledError(Exception):
    pass


# ============================================================================
# 3) المجزّع المتدفق — مخزن نص تزايدي + استخراج عند حدود آمنة
# ============================================================================
class _Chunker:
    """مخزن نص آمن + مجزّع واعٍ بالنبرة.

    السياسة (مواءمة مقصودة مع webapp.split_into_chunks المُثبتة):
      * الحدود: نهاية جملة (. ! ؟ ? …) ← فواصل ناعمة (، ؛ ; :) ←
        حدود كلمات كملاذ أخير — لا تُقصّ كلمة أبدًا (الحدود كلها عند
        فراغات، والفراغ التالي لعلامة نهاية الجملة شرط للحدّ).
      * دمج الجمل القصيرة في مقطع واحد حتى بلوغ min_tok (فلسفة PATCH 14:
        كل حد مقطع يصفّر إعراب الجملة — مقاطع أطول = صفريات نبرة أقل)
        بشرط ألا يتجاوز المجموع max_tok (90% من سقف التدريب 160).
      * الجملة العملاقة وحدها > max_tok تُقسَّم عند الفواصل الناعمة ثم
        حدود الكلمات (كما في webapp).
      * finish(): يشطف كل الباقي — آخر مقطع يجوز أن يكون أصغر من min_tok.
    """

    def __init__(self, dialect, min_tok=12, max_tok=144):
        self.dialect = dialect
        self.min_tok = max(2, int(min_tok))
        self.max_tok = max(self.min_tok, int(max_tok))
        self._pending = ''         # نص لم يُستخرج (جملة ربما غير مكتملة)
        self._carry = ''           # جمل مدموجة تنتظر بلوغ min_tok
        self._carry_tok = 0
        self._tok_cache = {}
        self.n_pieces = 0          # جمل مكتملة شوهدت
        self.n_emitted = 0         # مقاطع أُطلقت
        self.n_dropped = 0         # قطع بلا كلام عربي (فراغات/رموز)

    # ---- عدّ التوكنات بالمسار الموحد (مصدر الحقيقة: infer.count_tokens)
    def _tok(self, text):
        n = self._tok_cache.get(text)
        if n is None:
            n = infer.count_tokens(text, self.dialect)
            if len(self._tok_cache) < 8192:
                self._tok_cache[text] = n
        return n

    def feed(self, text):
        if text:
            self._pending = _bind_markers(self._pending + text)

    # ---- استخراج المقاطع الجاهزة الآن
    def poll(self, final=False):
        out = []
        parts = _SENT_END.split(self._pending)
        complete, self._pending = parts[:-1], parts[-1]
        for p in complete:
            p = p.strip(' \t\r')
            if p:
                self._absorb(p, out)
        if final:
            tail = self._pending.strip()
            self._pending = ''
            if tail:
                self._absorb(tail, out)
            self._emit_carry(out)          # الشطف: الأخير يجوز أصغر من min
        return out

    def _absorb(self, piece, out):
        n = self._tok(piece)
        self.n_pieces += 1
        # تفادي فيض السقف: أطلق الـcarry الحالي قبل إضافة قطعة ستفعمه
        if self._carry and self._carry_tok + n > self.max_tok:
            self._emit_carry(out)
        self._carry = (self._carry + ' ' + piece).strip()
        self._carry_tok += n
        if self._carry_tok > self.max_tok:
            # جملة عملاقة وحدها — قسّمها ثم أعد امتصاص أجزائها
            oversized, self._carry, self._carry_tok = self._carry, '', 0
            subs = self._split_oversized(oversized)
            for s in subs:
                self._absorb(s, out)
            return
        if self._carry_tok >= self.min_tok:
            self._emit_carry(out)

    def _split_oversized(self, text):
        """تقسيم جملة أطول من السقف — فواصل ناعمة أولًا ثم كلمات."""
        subs = [s.strip() for s in _SOFT_SPLIT.split(text) if s.strip()]
        if len(subs) < 2:
            subs = [text]
        final_subs = []
        for s in subs:
            if self._tok(s) <= self.max_tok:
                final_subs.append(s)
                continue
            words = s.split()
            cur, cur_n = [], 0
            for w in words:
                wn = self._tok(w)
                if cur and cur_n + wn > self.max_tok:
                    final_subs.append(' '.join(cur))
                    cur, cur_n = [w], wn
                else:
                    cur.append(w)
                    cur_n += wn
            if cur:
                final_subs.append(' '.join(cur))
        return final_subs

    def _emit_carry(self, out):
        text = ' '.join(self._carry.split())
        self._carry, self._carry_tok = '', 0
        if not text:
            return
        if not _has_arabic_speech(text, self.dialect):
            self.n_dropped += 1
            return
        out.append(text)
        self.n_emitted += 1


# ============================================================================
# 4) الاستمرارية الصوتية السببية — نسخة streaming من PATCH 13
# ============================================================================
class _Continuity:
    """معالجة موجة كل مقطع للحفاظ على استمرارية الاستماع — بلا معرفة
    المستقبل (الفرق الجوهري عن PATCH 13 الذي رأى كل المقاطع قبل الوصل):

      1) قصّ صمت البداية/النهاية (عتبة نسبية من ذروة المقطع نفسه) —
         صمت النموذج الحدي متغير، وبدون القص يتراكم مع الفاصل الثابت
         فترة صمت غير طبيعية.
      2) مرساة جهارة RMS جارية: أول مقطع يضع المرساة، وكل مقطع يُسحب
         نحوها بكسب محدود [min,max]، والمرساة نفسها تتكيف ببطء (EMA)
         فتبقى الجهارة متماسكة عبر الرد كله دون قفزات.
      3) تلاشي قصير 8ms عند حافتي الكلام نفسه — يمنع نقرات الوصل
         بالبناء (الوصل يتم عبر صمت الفاصل: صفر ← صمت ← صفر).
      4) فاصل صمت ثابت يُسبق به كل مقطع عدا الأول — إيقاع جُملي ثابت
         ولا فاصل زائد بعد آخر مقطع أبدًا.
      5) أمان الذروة: قصّ لين إن تجاوزت العينة 0.98.

    enabled=False يعيد الموجة الخام كما خرجت من المُصوِّت — لإثبات
    bit-identity مع مسار الإنتاج في الاختبارات.
    """

    def __init__(self, sr=SAMPLE_RATE, gap_s=0.20, fade_s=0.008,
                 gain_range=(0.5, 2.0), trim_rel=0.02, trim_pad_s=0.04,
                 enabled=True):
        import numpy as np
        self.np = np
        self.sr = sr
        self.gap = max(0, int(gap_s * sr))
        self.fade = max(1, int(fade_s * sr))
        self.gain_range = gain_range
        self.trim_rel = trim_rel
        self.trim_pad = max(1, int(trim_pad_s * sr))
        self.enabled = enabled
        self.anchor_rms = None
        self.n_processed = 0            # الفاصل يُسبق به كل مقطع عدا الأول
        self.stats = {'n': 0, 'gains': [], 'trimmed_s': []}

    def _trim(self, w):
        np = self.np
        peak = float(np.abs(w).max()) if len(w) else 0.0
        if peak <= 1e-6 or len(w) < int(0.25 * self.sr):
            return w, 0.0
        thr = peak * self.trim_rel
        idx = np.nonzero(np.abs(w) > thr)[0]
        if len(idx) == 0:
            return w, 0.0
        a = max(0, int(idx[0]) - self.trim_pad)
        b = min(len(w), int(idx[-1]) + self.trim_pad)
        trimmed = (len(w) - (b - a)) / self.sr
        return w[a:b].copy(), trimmed

    def process(self, wave):
        """معالجة موجة مقطع — يعيد (الموجة، الكسب المطبق، الثواني المقصوصة).

        الفاصل يُسبق به كل مقطع عدا الأول (لا فاصل بعد آخر مقطع أبدًا —
        حتى لو أُطلق آخر مقطع أثناء التدفق قبل finish)."""
        np = self.np
        prepend_gap = self.n_processed > 0 and self.gap > 0
        if not self.enabled:
            self.n_processed += 1
            return np.asarray(wave, dtype='float32'), 1.0, 0.0
        w = np.asarray(wave, dtype='float32')
        w, trimmed = self._trim(w)
        if len(w) == 0:
            self.n_processed += 1
            return w, 1.0, trimmed

        rms = float(np.sqrt(np.mean(np.square(w)))) if len(w) else 0.0
        gain = 1.0
        if rms > 1e-6:
            if self.anchor_rms is None:
                self.anchor_rms = rms
            else:
                lo, hi = self.gain_range
                gain = min(max(self.anchor_rms / rms, lo), hi)
                # المرساة تتكيف ببطء (EMA 0.15) — تتبع انزياح الجهارة
                # الحقيقي دون قفزات، وتسع الفروق التعبيرية الطبيعية
                self.anchor_rms = (0.85 * self.anchor_rms + 0.15 * rms)
            w = (w * gain).astype('float32')

        # تلاشي قصير عند حافتي الكلام نفسه (منع النقرات بالبناء)
        if len(w) > 2 * self.fade:
            ramp = np.linspace(0.0, 1.0, self.fade, dtype='float32')
            w[:self.fade] *= ramp
            w[-self.fade:] *= ramp[::-1]

        # الفاصل قبل المقطع (عدا الأول) — الوصل: صفر ← صمت ← صفر
        if prepend_gap:
            w = np.concatenate([np.zeros(self.gap, dtype='float32'), w])

        # أمان الذروة
        peak = float(np.abs(w).max()) if len(w) else 0.0
        if peak > 0.98:
            w = (w * (0.98 / peak)).astype('float32')

        self.n_processed += 1
        self.stats['n'] += 1
        self.stats['gains'].append(round(gain, 3))
        self.stats['trimmed_s'].append(round(trimmed, 3))
        return w, gain, trimmed


# ============================================================================
# 5) جلسة الطلب — Producer thread + Audio Queue + إلغاء + إحصاءات
# ============================================================================
class StreamSession:
    """طلب توليد متدفق واحد (request_id واحد).

    دورة الحياة:
        feed() × N  →  finish()  →  التكرار حتى StreamEnd('end')
        أو في أي لحظة: cancel() → StreamEnd('cancelled')

    الخيط المنتج (Producer): يراقب المخزن، يستخرج المقاطع عند حدود آمنة،
    يولّد كل مقطع عبر مسار الإنتاج نفسه (prepare_text_rich + synthesize)
    ثم يدفع AudioChunk إلى الطابور — المستهلك يسحب فور توفر أول مقطع
    بينما يستمر الخيط في توليد البقية.

    الإلغاء: علم _cancelled يُفحص قبل كل مرحلة وبعدها؛ ما يجري داخل
    torch لا يمكن قطعه فعليًا فيُستكمل ثم يُهمَل الناتج (يُحصى في
    stats['n_discarded_inflight']) — توثيق صادق لا ادعاء إلغاء زائف.
    """

    def __init__(self, engine, request_id, speaker, dialect, diacritize,
                 pace, denoise, chunker, continuity, name=None, verbose=False):
        self._engine = engine
        self.request_id = request_id
        self.name = name or f'req-{request_id}'
        self.speaker = speaker
        self.dialect = dialect
        self.diacritize = diacritize        # الوضع المطلوب (auto/egyptian/...)
        self.pace = pace
        self.denoise = denoise
        self._chunker = chunker
        self._cont = continuity
        self.verbose = verbose

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._audio_q = collections.deque()      # AudioChunk ثم StreamEnd
        #   (deque تحت القفل: تفريغ ذري مع الإلغاء — بلا سباق مع المستهلك)
        self._feed_done = False
        self._final_polled = False
        self._cancelled = False
        self._terminal = None                    # StreamEnd بعد صدوره
        self._thread = None

        # ---- الإحصاءات (تُقرأ عبر .stats — نسخة) ----
        self._stats = {
            'request_id': request_id, 'name': self.name,
            'speaker': speaker, 'dialect': dialect, 'pace': pace,
            'diacritize_requested': diacritize,
            'diacritize_resolved': None,
            't_created': time.time(),
            't_first_feed': None,
            't_first_boundary': None,
            't_first_chunk_ready': None,
            'feed_pieces': 0, 'feed_chars': 0,
            'n_chunks_emitted': 0, 'n_chunks_consumed': 0,
            'n_discarded_inflight': 0,
            'n_inferences': 0,
            'synth_time_total_s': 0.0,
            'audio_duration_total_s': 0.0,
            'chunks': [],
            'state': 'open',          # open→finishing→done | cancelled | error
        }
        self._tmpdir = tempfile.mkdtemp(prefix='eqz_stream_')

    # ------------------------------------------------------------------ API
    def feed(self, text):
        """تغذية نص متدفّق (incremental) — قد يُستدعى من خيط LLM/الشبكة.

        لا يولّد صوتًا بنفسه: يضيف للمخزن ويوقظ المنتج — التجزئة قرار
        المجزّع وحده. يرفض التغذية بعد finish/cancel."""
        if not isinstance(text, str):
            raise TypeError('feed() يتوقع نصًا (str)')
        with self._cond:
            if self._cancelled:
                raise RuntimeError('الطلب ملغى — لا تغذية بعد الإلغاء.')
            if self._feed_done:
                raise RuntimeError('انتهت التغذية (finish) — لا تغذية بعدها.')
            if self._stats['t_first_feed'] is None:
                self._stats['t_first_feed'] = time.time()
            self._stats['feed_pieces'] += 1
            self._stats['feed_chars'] += len(text)
            self._chunker.feed(text)
            self._cond.notify_all()

    def finish(self):
        """إعلان انتهاء النص — يشطف المجزّع ويولّد آخر مقطع ثم ينهي."""
        with self._cond:
            if self._cancelled:
                return False
            if not self._feed_done:
                self._feed_done = True
                if self._stats['state'] == 'open':
                    self._stats['state'] = 'finishing'
                self._cond.notify_all()
            return True

    def cancel(self):
        """المقاطعة: أوقف التوليد القادم، فرّغ الطابور، ارفض المتأخر.

        - تُهمل نتيجة أي inference جارٍ (يُحصى في n_discarded_inflight).
        - idempotent — استدعاءات متعددة آمنة، وإلغاء طلب منتهٍ لا يفسد
          حالته (يمنع فقط تسليم ما تبقى من صوته المنتظر).
        - المستهلك المحجوز يستيقظ على StreamEnd('cancelled') خلال ≤
          زمن inference الجاري (لا يمكن قطع torch من بايثون)."""
        with self._cond:
            if self._cancelled:
                return False
            ended = self._stats['state'] in ('done', 'error')
            # تفريغ ذري: كل AudioChunk يُحذف — StreamEnd (إن وُجد) يبقى
            self._audio_q = collections.deque(
                e for e in self._audio_q if not e.is_audio)
            self._cancelled = True
            if not ended:
                self._stats['state'] = 'cancelled'
            self._cond.notify_all()
        return True

    # -------------------------------------------------------------- أحداث
    def __iter__(self):
        return self.events()

    def events(self, timeout=None):
        """مولّد الأحداث: AudioChunk* ثم StreamEnd واحد بالضبط.

        timeout: مهلة كل انتظار (None = بلا حد) — عند انقضائها بلا
        أحداث يتوقف المولّد (StopIteration) — مفيد للمستهلك الذي يريد
        نخّات (heartbeat). مستهلك واحد لكل جلسة (عقد الواجهة)."""
        t_deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._cond:
                while not self._audio_q:
                    if self._terminal is not None:
                        return              # لا مزيد من الأحداث أبدًا
                    if t_deadline is not None and time.time() >= t_deadline:
                        return
                    wait_s = 0.25
                    if t_deadline is not None:
                        wait_s = max(0.0, min(wait_s,
                                              t_deadline - time.time()))
                    if wait_s <= 0:
                        return
                    self._cond.wait(wait_s)
                ev = self._audio_q.popleft()
                if ev.is_audio:
                    self._stats['n_chunks_consumed'] += 1
            yield ev
            if not ev.is_audio:
                return                      # الحدث النهائي سُلّم — انتهى

    def pending_audio(self):
        """عدد المقاطع الصوتية المنتظرة في الطابور الآن."""
        with self._lock:
            return sum(1 for e in self._audio_q if e.is_audio)

    @property
    def stats(self):
        with self._lock:
            snap = dict(self._stats)
            snap['chunks'] = list(self._stats['chunks'])
            if self._cont is not None:
                snap['continuity'] = dict(self._cont.stats)
            return snap

    @property
    def alive(self):
        t = self._thread
        return bool(t and t.is_alive())

    def wait_producer(self, timeout=10.0):
        """انتظار خروج الخيط المنتج (للاختبارات)."""
        t = self._thread
        if t:
            t.join(timeout)
        return not (t and t.is_alive())

    def close(self):
        """تحرير الموارد (المجلد المؤقت). آمن للاستدعاء المتكرر."""
        import shutil
        with self._lock:
            d, self._tmpdir = self._tmpdir, None
        if d:
            shutil.rmtree(d, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cancel()
        self.wait_producer(5.0)
        self.close()
        return False

    # ------------------------------------------------------------- تشغيل
    def start(self):
        self._thread = threading.Thread(
            target=self._producer, name=f'eqz-stream-{self.request_id}',
            daemon=True)
        self._thread.start()
        return self

    # ---- حلقة المنتج --------------------------------------------------
    def _producer(self):
        error = None
        try:
            while True:
                chunks = []
                with self._cond:
                    while True:
                        if self._cancelled:
                            raise _CancelledError()
                        chunks = self._chunker.poll(final=False)
                        if chunks:
                            break
                        if not self._feed_done:
                            self._cond.wait(0.05)
                            continue
                        # التغذية انتهت
                        if not self._final_polled:
                            chunks = self._chunker.poll(final=True)
                            self._final_polled = True
                            if chunks:
                                break
                            continue
                        return                    # لا مزيد — نهاية طبيعية
                for raw in chunks:
                    self._produce_one(raw)
        except _CancelledError:
            pass
        except SystemExit as e:                   # من مسار الإنتاج نفسه
            error = str(e.code if e.code is not None
                        else 'خطأ غير معروف في مسار الإنتاج')
        except Exception as e:                    # noqa: BLE001
            error = f'{type(e).__name__}: {e}'
        finally:
            self._finish_stream(error)

    def _produce_one(self, raw_text):
        """توليد مقطع واحد: مسار الإنتاج بالحرف ثم دفعه للطابور.

        فحوص الإلغاء: قبل المعالجة وبعد التوليد (قبل الدفع) — جارٍ
        torch لا يُقطع؛ ناتجه يُهمَل ويُحصى في n_discarded_inflight."""
        import soundfile as sf

        if self._stats['t_first_boundary'] is None:
            self._stats['t_first_boundary'] = time.time()

        # فحص الإلغاء قبل بدء المعالجة
        with self._lock:
            if self._cancelled:
                raise _CancelledError()

        # توحيد وضع التشكيل مرة واحدة لكل الطلب (بلا قلب أنماط بين
        # المقاطع) — مصدر القرار: أول نص وصل
        if self._stats['diacritize_resolved'] is None:
            self._stats['diacritize_resolved'] = (
                infer.effective_diacritize_mode(
                    raw_text, self.diacritize, self.dialect))

        seq = self._stats['n_chunks_emitted'] + 1
        t0 = time.time()
        wave = None
        n_tok = 0
        with self._engine._infer_lock:        # توليد واحد في كل ملة (CPU)
            res = infer.prepare_text_rich(
                raw_text, self._stats['diacritize_resolved'],
                self.dialect)
            tmp_wav = os.path.join(self._tmpdir, f'chunk_{seq:03d}.wav')
            n_tok, _ = infer.synthesize(
                self._engine.model, res['text'], self.dialect,
                self.speaker, self.pace, tmp_wav, self.denoise,
                peak_normalize=False)
            wave, _sr = sf.read(tmp_wav, dtype='float32')
            os.remove(tmp_wav)
        t_synth = time.time() - t0

        # فحص الإلغاء بعد التوليد — ناتج inference جارٍ يُهمَل لا يُدفع
        with self._cond:
            if self._cancelled:
                self._stats['n_discarded_inflight'] += 1
                raise _CancelledError()

        wave, gain, trimmed = self._cont.process(wave)
        with self._cond:
            now = time.time()
            dur = len(wave) / float(SAMPLE_RATE)
            is_final = (self._feed_done and self._final_polled
                        and self._chunker.n_emitted == seq
                        and not self._chunker._carry
                        and not self._chunker._pending.strip())
            chunk = AudioChunk(
                request_id=self.request_id, seq=seq, wave=wave,
                sample_rate=SAMPLE_RATE, n_tokens=n_tok,
                duration_s=dur, synth_time_s=t_synth, t_ready=now,
                text=raw_text, is_final=is_final,
                gain_applied=round(float(gain), 3),
                trimmed_s=round(float(trimmed), 3))
            self._audio_q.append(chunk)
            if self._stats['t_first_chunk_ready'] is None:
                self._stats['t_first_chunk_ready'] = now
            self._stats['n_chunks_emitted'] = seq
            self._stats['n_inferences'] += 1
            self._stats['synth_time_total_s'] += t_synth
            self._stats['audio_duration_total_s'] += dur
            self._stats['chunks'].append({
                'seq': seq, 'n_tokens': n_tok,
                'duration_s': round(dur, 2),
                'synth_time_s': round(t_synth, 2),
                'gain': chunk.gain_applied,
                'trimmed_s': chunk.trimmed_s,
                'text': raw_text})
            self._cond.notify_all()
        if self.verbose:
            log(f'[stream {self.name}] chunk {seq}: {n_tok} توكن · '
                f'{dur:.2f}ث صوت · {t_synth:.2f}ث توليد')

    def _finish_stream(self, error):
        """إصدار الحدث النهائي الوحيد وتحديث الحالة."""
        with self._cond:
            if self._terminal is not None:
                return
            if error is not None:
                self._stats['state'] = 'error'
                kind, err = 'error', error
            elif self._cancelled:
                kind, err = 'cancelled', None
            else:
                self._stats['state'] = 'done'
                kind, err = 'end', None
            # مقاييس مشتقة
            t_ff = self._stats['t_first_feed']
            t_fc = self._stats['t_first_chunk_ready']
            self._stats['ttfa_s'] = (
                round(t_fc - t_ff, 3)
                if (t_ff is not None and t_fc is not None) else None)
            self._stats['ttfa_from_create_s'] = (
                round(t_fc - self._stats['t_created'], 3)
                if t_fc is not None else None)
            d_tot = self._stats['audio_duration_total_s']
            s_tot = self._stats['synth_time_total_s']
            self._stats['rtf'] = round(s_tot / d_tot, 4) if d_tot > 0 else None
            terminal = StreamEnd(self.request_id, kind, None, err)
            self._terminal = terminal
            self._audio_q.append(terminal)
            self._cond.notify_all()
        # لقطة الإحصاءات خارج القفل (stats يكتسب القفل نفسه)
        terminal.stats = self.stats
        self.close()


# ============================================================================
# 6) المحرك — تحميل R1F مرة واحدة وطلبات متعددة بلا إعادة تحميل
# ============================================================================
class EiqazStreamingTTS:
    """محرك التدفق: يحمّل checkpoint الإنتاج (R1F) مرة واحدة ويخدم أي عدد
    من الطلبات المتعاقبة — المقاطعة ثم طلب جديد لا تعيدان تحميل النموذج.

    create_stream() هو نقطة الدخول الوحيدة — نفس معاملات المسار الموحد
    (speaker/لهجة/تشكيل/pace/denoise) + معاملات التجزيع والاستمرارية.
    """

    def __init__(self, checkpoint=None, threads=None):
        self.checkpoint = infer.resolve_checkpoint(checkpoint)
        import torch
        if threads:
            torch.set_num_threads(int(threads))
        self.model, self.iter = infer.load_model(self.checkpoint)
        self.model_loads = 1
        self._infer_lock = threading.Lock()
        self._req_ids = itertools.count(1)
        self.active_request_id = None
        self.n_requests = 0

    def create_stream(self, speaker=0, dialect='egy', diacritize='auto',
                      pace=1.0, denoise=0.005, *,
                      min_chunk_tokens=12, max_chunk_tokens=144,
                      chunk_gap_s=0.20, edge_fade_s=0.008,
                      rms_gain_range=(0.5, 2.0), trim_rel_threshold=0.02,
                      trim_pad_s=0.04, apply_continuity=True,
                      cancel_previous=False, name=None, verbose=False):
        """إنشاء طلب تدفق جديد.

        cancel_previous=True: أي طلب سابق لا يزال حيًّا يُلغى فورًا
        (نمط المحادثة: الطلب الجديد يزيح القديم) — أو دع الأوركسترا
        تستدعي old.cancel() بنفسها (الافتراضي: تحكم صريح).
        """
        speaker = infer.validate_speaker(speaker)
        if dialect not in ('egy', 'msa'):
            raise ValueError("dialect: 'egy' أو 'msa' فقط")
        if diacritize not in infer.DIACRITIZE_MODES:
            raise ValueError(
                f"diacritize: أحد {infer.DIACRITIZE_MODES}")
        if pace <= 0:
            raise ValueError('pace يجب أن يكون موجبًا')
        if not (2 <= min_chunk_tokens <= infer.TRAIN_MAX_TOKENS):
            raise ValueError(
                f'min_chunk_tokens يجب أن يكون بين 2 و '
                f'{infer.TRAIN_MAX_TOKENS}')
        cap = int(infer.TRAIN_MAX_TOKENS * 0.9)     # PATCH 14: 90% من السقف
        if not (min_chunk_tokens <= max_chunk_tokens <= cap):
            raise ValueError(
                f'max_chunk_tokens بين {min_chunk_tokens} و {cap} '
                '(90% من سقف التدريب 160 — سياسة PATCH 14)')

        request_id = next(self._req_ids)
        chunker = _Chunker(dialect, min_tok=min_chunk_tokens,
                           max_tok=max_chunk_tokens)
        continuity = _Continuity(
            sr=SAMPLE_RATE, gap_s=chunk_gap_s, fade_s=edge_fade_s,
            gain_range=tuple(rms_gain_range), trim_rel=trim_rel_threshold,
            trim_pad_s=trim_pad_s, enabled=bool(apply_continuity))
        session = StreamSession(
            self, request_id, speaker, dialect, diacritize, pace, denoise,
            chunker, continuity, name=name, verbose=verbose)

        if cancel_previous and self.active_request_id is not None:
            old = getattr(self, '_active_session', None)
            if old is not None and old.alive:
                old.cancel()
        self._active_session = session
        self.active_request_id = request_id
        self.n_requests += 1
        return session.start()

    def shutdown(self, timeout=5.0):
        """إلغاء أي جلسة حية وانتظار خروج خيوطها (للإقفال النظيف)."""
        s = getattr(self, '_active_session', None)
        if s is not None:
            s.cancel()
            s.wait_producer(timeout)
            s.close()


# ============================================================================
# 7) مستهلك مرجعي — طابور تشغيل بمقاطعة ورفض نتائج قديمة
# ============================================================================
class PlaybackQueue:
    """مستهلك مرجعي (UI-agnostic) لعميل التشغيل: سطح مكتب / متصفح /
    أندرويد — نفس العقد: submit(chunk) + interrupt().

    * epoch (request_id): كل chunk بمعرّف غير النشط يُرفض فورًا —
      «صوت قديم بعد إجابة جديدة» مستحيل بالبناء.
    * generation counter: interrupt() يرفع الجيل — التشغيل الجاري
      يتوقف خلال ≤ tick ويبقى متوقفًا حتى لو بدأ طلب جديد بـbegin()
      في نفس اللحظة (لا سباق: إلغاء التوقف لا يُلغي مقاطعة جارية).
    * interrupt(): يوقف التشغيل الجاري ويفرّغ الـpending فورًا.
    * realtime=False: وضع الاختبار الفوري — التشغيل لحظي بلا انتظار.
    * play_rate: تسريع التشغيل في الاختبارات (الزمن الحقيقي ÷ المعدل).
    """

    def __init__(self, realtime=False, play_rate=1.0, tick_s=0.05):
        self.realtime = bool(realtime)
        self.play_rate = max(0.01, float(play_rate))
        self.tick_s = tick_s
        self._lock = threading.Lock()
        self._q = collections.deque()
        self._epoch = None
        self._stopped = False          # بعد interrupt وقبل begin الجديد
        self._gen = 0                  # جيل التشغيل — interrupt يرفعه
        self._thread = None
        self._playing = False
        self.played = []           # سجلات التشغيل الفعلي
        self.rejected = []         # chunks قديمة رُفضت
        self.dropped = []          # chunks سقطت بعد interrupt (نفس الطلب)

    def begin(self, request_id):
        """بدء استقبال طلب جديد — يثبّت الـepoch ويستأنف القبول.

        (لا يرفع الجيل — مقاطعة التشغيل الجاري تبقى نافذة حتى تنقطع.)"""
        with self._lock:
            self._epoch = request_id
            self._stopped = False
        return request_id

    def submit(self, chunk):
        """استقبال AudioChunk — يعيد 'stale' | 'dropped' | 'played' '"'queued'."""
        with self._lock:
            if self._epoch is None or chunk.request_id != self._epoch:
                self.rejected.append(
                    {'request_id': chunk.request_id, 'seq': chunk.seq,
                     'reason': 'stale-epoch',
                     'epoch': self._epoch, 't': time.time()})
                return 'stale'
            if self._stopped:
                self.dropped.append(
                    {'request_id': chunk.request_id, 'seq': chunk.seq,
                     'reason': 'interrupted', 't': time.time()})
                return 'dropped'
            self._q.append(chunk)
        if not self.realtime:
            self._play_now(chunk)
            return 'played'
        self._ensure_thread()
        return 'queued'

    def interrupt(self):
        """المقاطعة: أوقف التشغيل الجاري (خلال ≤ tick) وفرّغ الـpending."""
        with self._lock:
            self._stopped = True
            self._gen += 1
            n = len(self._q)
            self._q.clear()
        return {'cleared': n, 'generation': self._gen}

    def wait_idle(self, timeout=10.0):
        """انتظار فراغ الطابور وانتهاء التشغيل الجاري (للاختبارات)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                idle = (not self._q) and not self._playing
            if idle:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------ داخل
    def _ensure_thread(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._player, daemon=True,
                    name='eqz-playback')
                self._thread.start()

    def _player(self):
        while True:
            with self._lock:
                if self._stopped:
                    return
                if not self._q:
                    return
                chunk = self._q.popleft()
                gen = self._gen
            self._play_now(chunk, gen)

    def _play_now(self, chunk, gen=None):
        rec = {'request_id': chunk.request_id, 'seq': chunk.seq,
               'duration_s': round(chunk.duration_s, 2),
               't_play_start': time.time(), 't_play_end': None,
               'interrupted': False}
        with self._lock:
            self._playing = True
            my_gen = self._gen if gen is None else gen
        try:
            if self.realtime and chunk.duration_s > 0:
                wait = chunk.duration_s / self.play_rate
                played = 0.0
                while played < wait:
                    if self._gen != my_gen:      # جيل جديد = مقاطعة نافذة
                        rec['interrupted'] = True
                        break
                    dt = min(self.tick_s, wait - played)
                    time.sleep(dt)
                    played += dt
            rec['t_play_end'] = time.time()
            with self._lock:
                self.played.append(rec)
        finally:
            with self._lock:
                self._playing = False
