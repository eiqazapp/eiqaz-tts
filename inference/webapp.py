#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NileTTS 4h — واجهة ويب محلية للتوليد (صفحة واحدة في المتصفح)
============================================================================
خادم محلي خفيف يفتح صفحة ويب تكتب فيها النص وتضبط كل معاملات التوليد
(المتحدث، اللهجة، السرعة، التشكيل، التنقية، تقسيم النص الطويل...) ثم
يحوّل النص إلى صوت وتستمع إليه وتنزّله — كل ذلك محليًا على جهازك.

مبني بـمكتبات بايثون المدمجة فقط (http.server / json / threading) —
لا يحتاج تثبيت أي مكتبة جديدة، ولا أي اتصال بالإنترنت.

مسار التوليد مطابق حرفيًا لـinfer.py: يستورد نفس الدوال المُختبرة
(prepare_text + synthesize) من نفس الملف — بلا أي نسخ أو تعديل للمسار
الحسابي. عند "التقسيم التلقائي" يُقسَّم النص إلى جمل ويُولَّد كل جزء
بنفس الدالة ثم تُجمَّع الموجات مع فاصل صمت قصير (تجميع مخرجات فقط —
لا يمس التوليد نفسه).

التشغيل:
    python webapp.py                 ← يفتح المتصفح تلقائيًا
    python webapp.py --port 9000     ← منفذ مختلف
    python webapp.py --no-browser    ← دون فتح المتصفح
    python webapp.py --threads 4     ← تقييد أنوية المعالج

الخادم يستمع على 127.0.0.1 فقط (جهازك — غير مكشوف للشبكة).

البث التدريجي (Streaming) — PATCH 18:
    نقطة نهاية WebSocket‏ (/ws) فوق نفس الخادم بلا أي مكتبة جديدة
    (تنفيذ RFC 6455 مبسّط بمكتبات بايثون المدمجة). تربط محرك التدفق
    EiqazStreamingTTS (stream_tts.py — فوق مسار الإنتاج R1F نفسه)
    بالمتصفح: النص يُغذّى تدريجيًا (feed) والمقاطع الصوتية تعود فور
    جاهزيتها (chunk + PCM16 ثنائي) فيشغّل المتصفح أول مقطع قبل اكتمال
    النص. المقاطعة (interrupt) توقف التوليد وتفرّغ الطابور وترفض
    المتأخر — وطلب جديد يزيح القديم فورًا. النموذج مشترك بين المسارين
    الدفعي والمتدفق (تحميل واحد). المسار الدفعي القديم (/api/generate)
    لم يُمس.
"""

import argparse
import base64
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import struct
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --- Windows console safety -------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# استيراد مسار التوليد المُختبر من infer.py نفسه (بلا أي نسخ للكود)
import infer

WEB_DIR = os.path.join(HERE, 'web')
INDEX_HTML = os.path.join(WEB_DIR, 'index.html')
SAVE_DIR = os.path.join(HERE, 'web_outputs')          # نسخة من كل توليد
MAX_TEXT_CHARS = 10000
MAX_CHUNKS = 60
CHUNK_GAP_S = 0.22                                     # صمت بين المقاطع
SAMPLE_RATE = 22050
_EDGE_FADE_S = 0.008           # PATCH 13: تلاشي 8ms عند حواف المقطع

# ============================================================================
# حالة التطبيق: النماذج + المهام
# ============================================================================
_models = {}                     # مسار checkpoint -> (model, iter)
_models_lock = threading.Lock()
_MODEL_CACHE_MAX = 3
_last_used = {}                  # مسار -> timestamp (لإزالة الأقدم)

GEN_LOCK = threading.Lock()      # توليد واحد في كل مرة (CPU)
JOBS = {}                        # job_id -> dict
JOBS_LOCK = threading.Lock()
JOB_TTL_S = 30 * 60              # الاحتفاظ بالمهمة 30 دقيقة
MAX_JOBS_KEPT = 15


def log(msg):
    print(msg, flush=True)


_BUILD = None


def server_build():
    """BUILD قصير (git rev-parse --short HEAD) — يُعرض في الواجهة.

    PATCH 19 (2026-10-09، شكوى المستخدم: «الصوت لا يبدأ قبل اكتمال تحويل
    كامل النص — شريط جاري التوليد بالكامل»): التشخيص كان أن الجهاز يشغّل
    خادمًا قديمًا (عملية webapp.py بدأت قبل سحب PATCH 18) و/أو صفحة قديمة
    من كاش المتصفح — فتتراجع الواجهة للمسار الدفعي بصمت ويظن المستخدم أن
    البث لا يعمل. الحل: هوية بناء واضحة تفحصها الصفحة فور التحميل عبر
    /api/build فتكشف الخادم القديم برسالة إصلاح صريحة، وCache-Control:
    no-store على الصفحة نفسها فلا تُقدّم نسخة قديمة مخبأة أبدًا.
    لا ينهار أبدًا: بلا git (نسخة zip) يبقى 'unknown'."""
    global _BUILD
    if _BUILD is None:
        _BUILD = 'unknown'
        try:
            import subprocess
            out = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE,
                capture_output=True, timeout=3)
            if out.returncode == 0:
                s = out.stdout.decode('ascii', 'replace').strip()
                if s:
                    _BUILD = s
        except Exception:                                   # noqa: BLE001
            pass
    return _BUILD


def get_model(ckpt_path):
    """تحميل نموذج من checkpoint (مع كاش صغير) — نفس دالة infer.load_model."""
    with _models_lock:
        if ckpt_path in _models:
            _last_used[ckpt_path] = time.time()
            return _models[ckpt_path]
        model, it = infer.load_model(ckpt_path)
        _models[ckpt_path] = (model, it)
        _last_used[ckpt_path] = time.time()
        # إزالة الأقدم لو تجاوزنا حد الكاش (لا نُبعد أحدث استخدام)
        while len(_models) > _MODEL_CACHE_MAX:
            oldest = min(_last_used, key=_last_used.get)
            _models.pop(oldest, None)
            _last_used.pop(oldest, None)
        return model, it


def list_checkpoints():
    """قائمة checkpoints المتاحة في مجلد checkpoints/ (states_*.pth)."""
    items = []
    for p in glob.glob(os.path.join(infer.CKPT_DIR, 'states_*.pth')):
        m = re.search(r'(\d+)(?=\.pth$)', p)
        it = int(m.group(1)) if m else 0
        items.append({'name': os.path.basename(p), 'iter': it,
                      'size_mb': round(os.path.getsize(p) / 1e6, 1)})
    items.sort(key=lambda x: x['iter'], reverse=True)
    rolling = os.path.join(infer.CKPT_DIR, 'states.pth')
    if os.path.exists(rolling):
        items.append({'name': 'states.pth', 'iter': 0,
                      'size_mb': round(os.path.getsize(rolling) / 1e6, 1)})
    return items


def default_checkpoint():
    pinned = os.path.join(infer.CKPT_DIR, infer.DEFAULT_CHECKPOINT)
    if os.path.exists(pinned):
        return infer.DEFAULT_CHECKPOINT
    snaps = list_checkpoints()
    return snaps[0]['name'] if snaps else None


# ============================================================================
# تقسيم النص الطويل إلى مقاطع (قبل التنظيف — على النص الخام)
# ============================================================================
_SENT_END = re.compile(r'(?<=[.!؟?…])\s+')
_SOFT_SPLIT = re.compile(r'(?<=[,،؛;:])\s+')
# PATCH 12: ربط الوسم المنفصل بكلمته قبل التقسيم — «كلمة {ق}» → «كلمة{ق}»
# حتى لا يفصل تقطيع حدود الكلمات (مجموعات الـ12) الوسم عن كلمته فيطبق
# على كلمة أخرى أو يضيع. لا يمس الوسوم الملتحقة أصلًا ولا غير العربية.
_TAG_BIND_RE = re.compile(
    r'([\u0621-\u063A\u0641-\u064A\u064B-\u0652])\s+(\{[^{}]{1,2}\})')


def bind_markers_to_words(raw_text):
    """PATCH 12 — لصق الوسم المنفصل بحرف عربي سابق له (المسافة → لصق)."""
    return _TAG_BIND_RE.sub(r'\1\2', raw_text)


def _n_tokens_of(text, dialect, tok_fns=None):
    # بعد التطبيع الموحد (أرقام→كلمات يغيّر العدد) — infer.count_tokens
    # هو مصدر الحقيقة الوحيد (نفس مسار CLI)
    return infer.count_tokens(text, dialect)


def split_into_chunks(raw_text, dialect, tok_fns=None,
                      max_tokens=infer.TRAIN_MAX_TOKENS):
    """تقسيم النص الخام إلى مقاطع عند نهايات الجمل.

    سياسة التقسيم:
      1) فواصل الأسطر أولًا، ثم . ! ؟ ؟ …
      2) دمج المقاطع الضئيلة جدًا (< 6 توكن) مع جارتها
      3) المقطع الأطول من السقف يُقسَّم عند الفواصل الناعمة (، ؛ ; :)
         ثم عند حدود الكلمات كخيار أخير
    كل مقطع يمرّ لاحقًا بنفس prepare_text/synthesize المستخدمة في infer.py.

    PATCH 12: يُستدعى بعد bind_markers_to_words — الوسم لا ينفصل عن كلمته.
    """
    raw_text = bind_markers_to_words(raw_text)
    segments = [s.strip() for s in raw_text.split('\n') if s.strip()]
    pieces = []
    for seg in segments:
        for p in _SENT_END.split(seg):
            p = p.strip(' \t\r')          # الترقيم يبقى — توكن مدرب
            if p:
                pieces.append(p)

    # دمج القطع الضئيلة مع ما قبلها
    merged = []
    for p in pieces:
        if merged and _n_tokens_of(p, dialect, tok_fns) < 6:
            merged[-1] = merged[-1] + ' ' + p
        else:
            merged.append(p)

    # تقطيع المقاطع الطويلة
    chunks = []
    for p in merged:
        if _n_tokens_of(p, dialect, tok_fns) <= max_tokens:
            chunks.append(p)
            continue
        # عند الفواصل الناعمة أولًا
        sub = [s.strip() for s in _SOFT_SPLIT.split(p) if s.strip()]
        if len(sub) < 2:
            sub = p.split()
            sub = [' '.join(sub[i:i + 12]) for i in range(0, len(sub), 12)]
        for s in sub:
            if _n_tokens_of(s, dialect, tok_fns) <= max_tokens:
                chunks.append(s)
            else:                      # تقطيع نهائي عند حدود الكلمات
                words = s.split()
                cur = []
                cur_n = 0
                for w in words:
                    wn = _n_tokens_of(w, dialect, tok_fns)
                    if cur and cur_n + wn > max_tokens:
                        chunks.append(' '.join(cur))
                        cur, cur_n = [w], wn
                    else:
                        cur.append(w)
                        cur_n += wn
                if cur:
                    chunks.append(' '.join(cur))

    # PATCH 14 (2026-10-02، تقرير المستخدم الرابع: «نمط المتحدث لم يتم
    # توحيده في الجمل الطويلة»): دمج الجمل القصيرة المتجاورة في مقاطع
    # أطول (حتى 90% من سقف التدريب). كل حد مقطع يُصفِّر إعراب الجملة
    # ويبدأ نغمة جديدة — تقليل عدد المقاطع = تقليل عدد صفريات النبرة =
    # نمط متحدث موحّد أكثر عبر النص الطويل كله. النموذج دُرِّب على وحدات
    # متعددة الجمل حتى السقف نفسه (160 توكن) فالدمج داخل التوزيع تمامًا
    # — الجملة الأخيرة الصغيرة لا تُولَّد وحدها بنبرة معزولة شاذة.
    merge_cap = int(max_tokens * 0.9) if max_tokens >= 20 else max_tokens
    merged14 = []
    for c in chunks:
        if merged14 and _n_tokens_of(
                merged14[-1] + ' ' + c, dialect, tok_fns) <= merge_cap:
            merged14[-1] = merged14[-1] + ' ' + c
        else:
            merged14.append(c)
    chunks = merged14
    return [c for c in chunks if infer._AR_LETTERS.search(
        infer.keep_arabic_only(c))]


def effective_vocalize_mode(raw_text, mode, dialect='egy'):
    # مصدر الحقيقة الوحيد: infer.effective_diacritize_mode — يعيد وضع
    # التشكيل النهائي (egyptian/fusha/manual)
    return infer.effective_diacritize_mode(raw_text, mode, dialect)


# ---------------------------------------------------------------- PATCH 13 --
def _unify_chunks_tone(waves):
    """PATCH 13 (2026-10-02) — توحيد نبرة المتحدث في النصوص الطويلة.

    شكوى المستخدم: «ما سبب اختلاف نبرة المتحدث مع النصوص الطويلة؟».
    التشخيص: (1) كل مقطع كان يُطبَّل ذروته على حدة (0.9*wave/max في
    mel_to_wav) فتقفز الجهارة بين المقاطع؛ (2) النموذج حتمي (لا عشوائية
    في infer) لكن كل مقطع يبدأ نطقًا جديدًا فيُصفَّر إعراب الجملة عند كل
    حد — والفروق الصاخبة بين الحدود هي المدرك السمعي «لنبرة مختلفة».

    الحل (بلا إعادة تدريب):
      1. تعادل جهارة RMS كل مقطع نحو وسيط المقاطع (حد كسب 0.25–4.0)
      2. تلاشي قصير (8ms) عند حافتي كل مقطع — يمنع نقرات الوصل
      3. تطبيع ذروة واحد (0.9) على النص كاملاً بعد الوصل — بدل تطبيع
         كل مقطع على حدة
    يعيد (الموجة النهائية، dict بيانات التعادل للعرض)."""
    import numpy as np
    n = len(waves)
    info = {'n_chunks': n, 'rms_gains': None}
    if n == 0:
        return None, info

    # 1) تعادل الجهارة (RMS) نحو وسيط المقاطع — فقط للتعدد
    # PATCH 14: حد الكسب ضُيّق من [0.25, 4.0] إلى [0.4, 2.5] — تعزيز 4×
    # كان يرفع أرضية ضجيج المُصوِّت في المقاطع الهادئة (الجمل الأخيرة
    # ذات الطاقة المتدنية) فسمعها المستخدم «نبرة سيئة جدًا في نهاية الحديث
    # ومختلفة». كسب ±(4–8)dB لطيف محفوظ للجهارة، وما عدا ذلك يبقى
    # طبيعيًا — والدمج الأطول للمقاطع (split_into_chunks) يعالج جذر
    # اختلاف النبرة نفسه.
    rms = [float(np.sqrt(np.mean(np.square(w)))) if len(w) else 0.0
           for w in waves]
    if n > 1:
        target = float(np.median(rms))              # الوسيط — متين ضد الشواذ
        gains = []
        if target > 1e-6:
            for i in range(n):
                g = 1.0
                if rms[i] > 1e-6:
                    g = min(max(target / rms[i], 0.4), 2.5)
                    waves[i] = (waves[i] * g).astype('float32')
                gains.append(round(g, 2))
        info['rms_gains'] = gains

    # 2) تلاشي قصير عند الحواف (منع النقرات عند الوصل)
    fade = max(1, int(_EDGE_FADE_S * SAMPLE_RATE))
    for i in range(n):
        w = waves[i]
        if len(w) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype='float32')
            w[:fade] *= ramp
            w[-fade:] *= ramp[::-1]

    # 3) الوصل (مع فاصل الجملة) ثم تطبيع ذروة واحد للنص كله
    if n == 1:
        final = waves[0]
    else:
        gap = np.zeros(int(CHUNK_GAP_S * SAMPLE_RATE), dtype='float32')
        final = waves[0]
        for w in waves[1:]:
            final = np.concatenate([final, gap, w])
    peak = float(np.abs(final).max()) if len(final) else 0.0
    if peak > 1e-6:
        final = (0.9 * final / peak).astype('float32')
    return final, info
# ---------------------------------------------------------------------------


# ============================================================================
# البث التدريجي (PATCH 18) — محرك مشترك + WebSocket مبسّط + بروتوكول البث
# ============================================================================
# البروتوكول (نص JSON من العميل؛ ومن الخادم: JSON أو إطار PCM16 ثنائي):
#   C→S {"type":"start", speaker, dialect, diacritize, pace, denoise}
#   S→C {"type":"started", request_id, sample_rate, checkpoint}
#   C→S {"type":"feed","text":"..."}          × N (تدريجي)
#   C→S {"type":"finish"}
#   S→C {"type":"chunk", request_id, seq, n_samples, ...} ثم إطار PCM16 ثنائي
#   C→S {"type":"interrupt"}
#   S→C {"type":"interrupted", request_id, applied}
#   S→C {"type":"end", request_id, kind: end|cancelled|error, stats}
# كل رسالة تحمل request_id — المتصفح يسقط أي مقطع من طلب غير النشط
# (مقاطع قديمة لا يمكن أن تُسمع بعد بدء طلب جديد بالبناء).

_STREAM_ENGINE = None                     # محرك التدفق المشترك (تحميل واحد)
_STREAM_ENGINE_LOCK = threading.Lock()
_WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
_WS_MAX_TEXT_FRAME = 256 * 1024           # حد إطار وارد (قطع التغذية صغيرة)
_NULL_CTX = contextlib.nullcontext()


def get_stream_engine(create=True):
    """محرك التدفق المشترك — يشارك النموذج نفسه مع مسار الدفعات.

    get_model() هو مصدر الأوزان الوحيد (كاش webapp نفسه) ثم يُحقن في
    EiqazStreamingTTS — لا تحميل ثانٍ للأوزان نفسها أبدًا. checkpoint
    البث هو checkpoint الإنتاج R1F دائمًا (states_cont_180516.pth) مهما
    اختار المستخدم للمسار الدفعي — البث واجهة الإنتاج الافتراضية."""
    global _STREAM_ENGINE
    with _STREAM_ENGINE_LOCK:
        if _STREAM_ENGINE is None:
            if not create:
                return None
            from stream_tts import EiqazStreamingTTS
            ckpt_path = os.path.join(infer.CKPT_DIR, infer.DEFAULT_CHECKPOINT)
            if not os.path.exists(ckpt_path):
                raise SystemExit(
                    '[خطأ] checkpoint الإنتاج ' + infer.DEFAULT_CHECKPOINT
                    + ' غير موجود — البث التدريجي يحتاجه. استعده من Git.')
            _STREAM_ENGINE = EiqazStreamingTTS(
                checkpoint=ckpt_path, model=get_model(ckpt_path))
        return _STREAM_ENGINE


class _WSConn:
    """قناة WebSocket خادمية مبسّطة (RFC 6455) فوق مقبس مقبول.

    يدعم: إطارات نص/ثنائي، التقنيع من العميل، تجميع الإطارات المجزأة
    (continuation)، إغلاق/بينغ/بونغ. الكتابة آمنة من عدة خيوط (قفل
    إرسال) — إطار واحد لا يتشظى أبدًا مهما تراكبت خيوط الإرسال."""

    def __init__(self, sock, rfile):
        self._sock = sock
        self._rfile = rfile
        self._wlock = threading.Lock()
        self.closed = False

    # ---- إرسال (خادم → عميل: غير مُقنّع) ----
    def send_text(self, text):
        return self._send(0x1, text.encode('utf-8'))

    def send_binary(self, blob):
        return self._send(0x2, blob)

    def pong(self, payload=b''):
        return self._send(0xA, payload)

    def _send(self, opcode, payload):
        if self.closed:
            return False
        header = bytearray()
        header.append(0x80 | opcode)
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += struct.pack('>H', n)
        else:
            header.append(127)
            header += struct.pack('>Q', n)
        try:
            with self._wlock:
                self._sock.sendall(bytes(header) + payload)
            return True
        except OSError:
            self.closed = True
            return False

    # ---- استقبال (عميل → خادم: مُقنّع دائمًا حسب RFC) ----
    def recv(self):
        """يعيد (opcode, payload) أو None عند الإغلاق/الانقطاع.

        يجمع الإطارات المجزأة داخليًا. الأحجام الحدية تُرفض."""
        hdr = self._read_exact(2)
        if hdr is None:
            return None
        b0, b1 = hdr[0], hdr[1]
        opcode = b0 & 0x0F
        fin = bool(b0 & 0x80)
        masked = bool(b1 & 0x80)
        ln, ok = self._read_len(b1)
        if not ok or ln > _WS_MAX_TEXT_FRAME:
            return None
        mask = self._read_exact(4) if masked else None
        payload = self._read_exact(ln) if ln else b''
        if payload is None:
            return None
        if mask is not None:
            payload = _unmask(payload, mask)
        if not fin:                            # تجميع الإطار المجزّأ
            parts = [payload]
            while True:
                nh = self._read_exact(2)
                if nh is None:
                    return None
                n0, n1 = nh[0], nh[1]
                if (n0 & 0x0F) != 0x0:         # continuation فقط
                    return None
                cln, ok = self._read_len(n1)
                if not ok or cln > _WS_MAX_TEXT_FRAME:
                    return None
                cmask = self._read_exact(4) if (n1 & 0x80) else None
                cp = self._read_exact(cln) if cln else b''
                if cp is None:
                    return None
                if cmask is not None:
                    cp = _unmask(cp, cmask)
                parts.append(cp)
                if sum(map(len, parts)) > _WS_MAX_TEXT_FRAME:
                    return None
                if n0 & 0x80:
                    break
            payload = b''.join(parts)
        return opcode, payload

    def _read_len(self, b1):
        ln = b1 & 0x7F
        if ln == 126:
            ext = self._read_exact(2)
            if ext is None:
                return 0, False
            return struct.unpack('>H', ext)[0], True
        if ln == 127:
            ext = self._read_exact(8)
            if ext is None:
                return 0, False
            return struct.unpack('>Q', ext)[0], True
        return ln, True

    def _read_exact(self, n):
        if n == 0:
            return b''
        try:
            data = self._rfile.read(n)
        except OSError:
            self.closed = True
            return None
        if data is None or len(data) != n:
            self.closed = True
            return None
        return data

    def close(self, code=1000):
        try:
            self._send(0x8, struct.pack('>H', code))
        except Exception:                                   # noqa: BLE001
            pass
        self.closed = True


def _unmask(payload, mask):
    return bytes(b ^ mask[i & 3] for i, b in enumerate(payload))


def _save_stream_wav(pcm_parts, stream):
    """حفظ الصوت المبث كملف WAV في web_outputs — تكافئ المسار الدفعي.

    (وحدة wave المدمجة: PCM16 أحادي 22050 — نفس صيغة المخرجات.)"""
    import wave as _wave
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        spk = getattr(stream, 'speaker', 0)
        path = os.path.join(SAVE_DIR, f'web_{stamp}_spk{spk}_stream.wav')
        with _wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            for part in pcm_parts:
                w.writeframes(part)
        log(f'[ws] حُفظ الصوت المبث: {os.path.basename(path)}')
        return path
    except Exception as e:                                   # noqa: BLE001
        log(f'[ws] تعذر حفظ الصوت المبث: {e}')
        return None


def _ws_session(handler):                                   # noqa: C901
    """جلسة WebSocket كاملة: قراءة أوامر العميل + بث المقاطع الصوتية.

    خيط القارئ (هذا الخيط) يعالج start/feed/finish/interrupt فورًا؛
    خيط مرسل لكل طلب يسحب أحداث EiqazStreamingTTS ويدفع chunk+PCM16.
    طلب جديد (start) يزيح القديم عبر cancel_previous — مرسل القديم يخرج
    بنفسه بعد حدثه النهائي، وقفل الإرسال يمنع تشظّي الإطارات."""
    import numpy as np

    ws = _WSConn(handler.connection, handler.rfile)
    st = {'stream': None, 'sender': None, 'fed_chars': 0, 'req': None}

    def _json(obj):
        try:
            return ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:                                   # noqa: BLE001
            return False

    def _err(msg, req=None):
        _json({'type': 'error', 'request_id': req, 'error': msg})

    def _sender(stream):
        pcm_parts = []          # تكافؤ المسار الدفعي: نسخة محفوظة في web_outputs
        try:
            for ev in stream.events():
                if ws.closed:
                    return
                if ev.is_audio:
                    w = ev.wave
                    if w is None or len(w) == 0:
                        continue
                    pcm = (np.clip(w, -1.0, 1.0) * 32767.0).astype('<i2')
                    pcm_parts.append(pcm.tobytes())
                    meta = {
                        'type': 'chunk', 'request_id': ev.request_id,
                        'seq': ev.seq, 'sample_rate': int(ev.sample_rate),
                        'n_samples': int(len(w)),
                        'n_tokens': int(ev.n_tokens),
                        'duration_s': round(float(ev.duration_s), 3),
                        'synth_time_s': round(float(ev.synth_time_s), 3),
                        'is_final': bool(ev.is_final),
                        'gain_applied': ev.gain_applied,
                        'trimmed_s': ev.trimmed_s,
                        't_ready_ms': int(ev.t_ready * 1000),
                        'text': ev.text or '',
                    }
                    if not _json(meta):
                        return
                    if not ws.send_binary(pcm.tobytes()):
                        return
                else:                      # StreamEnd — نهائي واحد بالضبط
                    stats = ev.stats or stream.stats
                    if ev.kind == 'end' and pcm_parts:
                        _save_stream_wav(pcm_parts, stream)
                    slim = {k: stats.get(k) for k in (
                        'ttfa_s', 'ttfa_from_create_s', 'rtf', 'state',
                        'n_chunks_emitted', 'n_chunks_consumed',
                        'n_discarded_inflight', 'n_inferences',
                        'synth_time_total_s', 'audio_duration_total_s',
                        'feed_chars', 'feed_pieces', 'speaker', 'dialect',
                        'pace', 'diacritize_resolved')}
                    _json({'type': 'end', 'request_id': ev.request_id,
                           'kind': ev.kind, 'error': ev.error,
                           'stats': slim})
                    return
        except Exception as e:                               # noqa: BLE001
            _err(f'خطأ داخلي في بث الصوت: {type(e).__name__}: {e}',
                 stream.request_id)
        finally:
            if st['stream'] is stream:
                st['stream'] = None
                st['sender'] = None

    # ---- حلقة القارئ ----
    while True:
        msg = ws.recv()
        if msg is None or ws.closed:
            break
        opcode, payload = msg
        if opcode == 0x8:                    # إغلاق من العميل
            ws.close()
            break
        if opcode == 0x9:                    # ping → pong (إطار تحكم)
            ws.pong(payload)
            continue
        if opcode != 0x1:
            continue
        try:
            data = json.loads(payload.decode('utf-8'))
            if not isinstance(data, dict):
                raise ValueError('ليست كائن JSON')
        except Exception:                                   # noqa: BLE001
            _err('رسالة JSON غير سليمة.')
            continue
        mtype = data.get('type')

        if mtype == 'start':
            try:
                engine = get_stream_engine(create=True)
            except SystemExit as e:
                _err(str(e.code if e.code is not None else
                         'تعذر تهيئة محرك البث'))
                continue
            except Exception as e:                           # noqa: BLE001
                _err(f'تعذر تهيئة محرك البث: {type(e).__name__}: {e}')
                continue
            speaker = data.get('speaker', 0)
            dialect = 'msa' if data.get('dialect') == 'msa' else 'egy'
            diac = data.get('diacritize') or 'auto'
            diac = {'always': 'auto', 'never': 'manual'}.get(diac, diac)
            if diac not in infer.DIACRITIZE_MODES:
                diac = 'auto'
            try:
                pace = float(data.get('pace', 1.0))
            except (TypeError, ValueError):
                pace = 1.0
            pace = min(max(pace, 0.5), 2.0)
            try:
                denoise = float(data.get('denoise', 0.005))
            except (TypeError, ValueError):
                denoise = 0.005
            denoise = min(max(denoise, 0.0), 0.05)
            try:
                speaker = infer.validate_speaker(speaker)
            except SystemExit as e:
                _err(str(e.code if e.code is not None else 'متحدث غير صالح'))
                continue
            try:
                stream = engine.create_stream(
                    speaker=speaker, dialect=dialect, diacritize=diac,
                    pace=pace, denoise=denoise,
                    cancel_previous=True, name='ws')
            except (ValueError, SystemExit) as e:
                _err(str(getattr(e, 'code', None) or e))
                continue
            st['stream'] = stream
            st['fed_chars'] = 0
            st['req'] = stream.request_id
            th = threading.Thread(
                target=_sender, args=(stream,), daemon=True,
                name=f'eqz-ws-send-{stream.request_id}')
            st['sender'] = th
            th.start()
            _json({'type': 'started', 'request_id': stream.request_id,
                   'sample_rate': 22050,
                   'checkpoint': os.path.basename(engine.checkpoint)})

        elif mtype == 'feed':
            stream = st['stream']
            text = data.get('text')
            if stream is None or not isinstance(text, str):
                _err('لا يوجد طلب بث نشط ليتغذى.', st['req'])
                continue
            if st['fed_chars'] + len(text) > MAX_TEXT_CHARS:
                stream.cancel()
                _err(f'النص أطول من الحد {MAX_TEXT_CHARS} حرفًا — '
                     'أُلغي طلب البث.', stream.request_id)
                continue
            try:
                stream.feed(text)
                st['fed_chars'] += len(text)
            except RuntimeError as e:
                _err(f'التغذية مرفوضة: {e}', stream.request_id)

        elif mtype == 'finish':
            stream = st['stream']
            if stream is not None:
                stream.finish()
            else:
                _err('لا يوجد طلب نشط لإعلان انتهائه.', st['req'])

        elif mtype == 'interrupt':
            stream = st['stream']
            if stream is not None:
                applied = stream.cancel()
                _json({'type': 'interrupted',
                       'request_id': stream.request_id,
                       'applied': bool(applied)})
            else:
                _json({'type': 'interrupted', 'request_id': st['req'],
                       'applied': False})

        elif mtype == 'ping':
            _json({'type': 'pong', 't_ms': int(time.time() * 1000)})

        else:
            _err(f'نوع رسالة غير معروف: {mtype!r}', st['req'])

    # انقطع الاتصال — ألغِ أي طلب حي على هذا الاتصال (لا توليد يتيم)
    stream = st.get('stream')
    if stream is not None:
        try:
            stream.cancel()
        except Exception:                                   # noqa: BLE001
            pass
    th = st.get('sender')
    if th is not None:
        th.join(2.0)


# ============================================================================
# تنفيذ مهمة التوليد (في خيط منفصل)
# ============================================================================
def run_job(job_id, params):
    import soundfile as sf
    import numpy as np

    job = JOBS[job_id]
    t0 = time.time()
    tmp_dir = None
    diac_mode_req = params.get('diacritize') or 'auto'
    try:
        with GEN_LOCK:
            job['status'] = 'running'
            tok_fns = infer.get_tokenizer()

            # ----- checkpoint -----
            ckpt_name = params['checkpoint']
            if ckpt_name in (None, '', 'auto'):
                ckpt_name = default_checkpoint()
            if not ckpt_name:
                raise SystemExit(
                    '[خطأ] لا يوجد checkpoint في مجلد checkpoints/ — '
                    'checkpoint الإنتاج المعتمد (يأتي مع المستودع): '
                    + infer.DEFAULT_CHECKPOINT)
            ckpt_path = os.path.join(infer.CKPT_DIR, os.path.basename(ckpt_name))
            if not os.path.exists(ckpt_path):
                raise SystemExit(f'[خطأ] checkpoint غير موجود: {ckpt_name}')
            job['checkpoint'] = os.path.basename(ckpt_path)

            model, it = get_model(ckpt_path)
            job['iter'] = it

            # ----- تقسيم -----
            raw = params['text']
            if params['split']:
                chunks = split_into_chunks(raw, params['dialect'], tok_fns)
            else:
                chunks = [raw]
            if not chunks:
                raise SystemExit('[خطأ] النص لا يحتوي حروفًا عربية.')
            if len(chunks) > MAX_CHUNKS:
                raise SystemExit(
                    f'[خطأ] النص طويل جدًا ({len(chunks)} مقطعًا — الحد '
                    f'{MAX_CHUNKS}). قسّمه على دفعات أصغر.')
            job['total'] = len(chunks)

            diac_mode = effective_vocalize_mode(
                raw, diac_mode_req, params['dialect'])
            job['vocalized'] = diac_mode in ('egyptian', 'fusha')
            job['diacritize'] = diac_mode

            # ----- توليد كل مقطع بنفس دالتي infer.py -----
            # PATCH 13: peak_normalize=False — الموجات الخام؛ توحيد الجهارة
            # والتطبيع الواحد في _unify_chunks_tone بعد اكتمال المقاطع
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix='eiqaz_web_')
            waves = []
            n_tokens_total = 0
            had_numbers = False
            had_translit = False
            had_markers = False
            chunks_detail = []      # تتبع المتحدث لكل مقطع (اختبار الثبات)
            speaker = infer.validate_speaker(params['speaker'])
            for i, chunk_raw in enumerate(chunks):
                job['chunk'] = i + 1
                job['msg'] = f'جاري توليد المقطع {i + 1} من {len(chunks)}'
                res = infer.prepare_text_rich(chunk_raw, diac_mode,
                                              params['dialect'])
                text = res['text']
                had_numbers = had_numbers or res['numbers']
                had_translit = had_translit or res['translit']
                had_markers = had_markers or res['markers']
                tmp_wav = os.path.join(tmp_dir, f'chunk_{i:03d}.wav')
                # PATCH 18: تسلسل وصول النموذج مع مسار البث — قفل المحرك
                # نفسه الذي يمسكه منتج البث لكل مقطع (لو حُمّل المحرك):
                # توليد دفعي ومتدفق لا يتراكبان على نفس الأوزان CPU.
                _eng = get_stream_engine(create=False)
                with (_eng._infer_lock if _eng is not None else _NULL_CTX):
                    n_tok, _ = infer.synthesize(
                        model, text, params['dialect'], speaker,
                        params['pace'], tmp_wav, params['denoise'],
                        peak_normalize=False)
                n_tokens_total += n_tok
                w, _ = sf.read(tmp_wav, dtype='float32')
                waves.append(w)
                chunks_detail.append({
                    'i': i + 1, 'speaker': speaker, 'n_tokens': n_tok,
                    'duration_s': round(len(w) / SAMPLE_RATE, 2),
                    'rms': round(float(np.sqrt(np.mean(np.square(w)))), 4),
                    'sample_rate': SAMPLE_RATE, 'channels': 1,
                })
                job['elapsed'] = round(time.time() - t0, 1)

            # ----- PATCH 13: تجميع الموجات بتوحيد النبرة -----
            final, tone_info = _unify_chunks_tone(waves)

            buf = io.BytesIO()
            sf.write(buf, final, SAMPLE_RATE, subtype='PCM_16',
                     format='WAV')
            wav_bytes = buf.getvalue()

            # نسخة محفوظة في مجلد web_outputs
            os.makedirs(SAVE_DIR, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            saved = os.path.join(
                SAVE_DIR, f'web_{stamp}_spk{params["speaker"]}.wav')
            with open(saved, 'wb') as f:
                f.write(wav_bytes)

            job.update({
                'status': 'done',
                'duration': round(len(final) / SAMPLE_RATE, 1),
                'n_tokens': n_tokens_total,
                'n_chunks': len(chunks),
                'wav_size': len(wav_bytes),
                'saved_to': os.path.basename(saved),
                'elapsed': round(time.time() - t0, 1),
                'wav': wav_bytes,
                'diacritize': diac_mode,
                'numbers_spoken': had_numbers,
                'translit_applied': had_translit,
                'qaf_markers': had_markers,
                'chunks_detail': chunks_detail,
                'tone_unified': bool(tone_info['n_chunks'] > 1),
                'tone_rms_gains': tone_info['rms_gains'],
                'msg': 'تم التوليد بنجاح',
            })
    except SystemExit as e:
        job['status'] = 'error'
        job['error'] = str(e.code if e.code is not None else 'خطأ غير معروف')
    except Exception as e:                                   # noqa: BLE001
        job['status'] = 'error'
        msg = f'{type(e).__name__}: {e}'
        # رسالة ودية لحالة معروفة: سرعة عالية جدًا على نص قصير
        if 'Kernel size' in msg and 'greater' in msg:
            msg += ('\n[تلميح] السرعة المختارة عالية جدًا لنص قصير — '
                    'قلّل "سرعة الكلام" أو أطل النص.')
        job['error'] = msg
    finally:
        if tmp_dir:
            try:
                for f in glob.glob(os.path.join(tmp_dir, '*.wav')):
                    os.remove(f)
                os.rmdir(tmp_dir)
            except OSError:
                pass


def gc_jobs():
    """حذف المهام المنتهية القديمة لتحرير الذاكرة."""
    now = time.time()
    with JOBS_LOCK:
        stale = [k for k, v in JOBS.items()
                 if v['status'] in ('done', 'error')
                 and now - v.get('finished_at', now) > JOB_TTL_S]
        for k in stale:
            JOBS.pop(k, None)
        # حد أقصى بعدد المهام المحفوظة
        done_ids = [k for k, v in JOBS.items() if v['status'] == 'done']
        while len(done_ids) > MAX_JOBS_KEPT:
            JOBS.pop(done_ids.pop(0), None)


# ============================================================================
# خادم HTTP
# ============================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = 'NileWeb/1.0'

    # ---------- أدوات الرد ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg, code=400):
        self._json({'error': msg}, code)

    def _body_json(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 200 * 1024:
            raise ValueError('حجم الطلب كبير جدًا.')
        if n == 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError('جسم الطلب ليس JSON سليمًا (UTF-8).')

    def log_message(self, fmt, *args):                       # noqa: A003
        pass  # نسكت عن سجلات الوصول — الكونسول للمستخدم النهائي

    # ---------- PATCH 18: ترقية WebSocket + جلسة البث ----------
    def _handle_ws(self, key):
        accept = base64.b64encode(hashlib.sha1(
            (key + _WS_GUID).encode('ascii')).digest()).decode('ascii')
        self.connection.sendall(
            ('HTTP/1.1 101 Switching Protocols\r\n'
             'Upgrade: websocket\r\n'
             'Connection: Upgrade\r\n'
             f'Sec-WebSocket-Accept: {accept}\r\n\r\n').encode('ascii'))
        self.close_connection = True
        log('[ws] اتصال البث مفتوح')
        try:
            _ws_session(self)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:                               # noqa: BLE001
            log(f'[ws] خطأ جلسة: {type(e).__name__}: {e}')
        finally:
            log('[ws] أُغلق اتصال البث')

    # ---------- GET ----------
    def do_GET(self):                                        # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == '/ws':               # PATCH 18: بث WebSocket
                upgrade = (self.headers.get('Upgrade') or '').strip().lower()
                key = self.headers.get('Sec-WebSocket-Key')
                if upgrade != 'websocket' or not key:
                    return self._error('تصادق WebSocket ناقص.', 400)
                self._handle_ws(key)
                return
            if u.path in ('/', '/index.html'):
                with open(INDEX_HTML, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type',
                                 'text/html; charset=utf-8')
                # PATCH 19: لا كاش للصفحة إطلاقًا — بعد أي تحديث للكود
                # يجلب المتصفح الواجهة الجديدة دومًا (وإلا عرض نسخة قديمة
                # مخبأة بلا بث — جذر شكوى «الصوت بعد اكتمال التوليد»)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == '/api/build':
                # PATCH 19: بطاقة هوية الخادم — تفحصها الواجهة عند التحميل
                # فتكشف خادمًا قديمًا (بلا نقطة /api/build ولا /ws) وتوجّه
                # المستخدم لإعادة التشغيل بدل التراجع الصامت للمسار الدفعي
                self._json({'app': 'webapp.py', 'build': server_build(),
                            'ws': True, 'streaming': True,
                            'sample_rate': SAMPLE_RATE,
                            'checkpoint_default': infer.DEFAULT_CHECKPOINT})
            elif u.path == '/api/checkpoints':
                self._json({'items': list_checkpoints(),
                            'selected': default_checkpoint()})
            elif u.path == '/api/status':
                job_id = (q.get('job') or [''])[0]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if not job:
                    return self._error('المهمة غير موجودة (انتهت صلاحيتها '
                                       'أو رقمها خطأ).', 404)
                out = {k: v for k, v in job.items() if k != 'wav'}
                self._json(out)
            elif u.path == '/api/audio':
                job_id = (q.get('job') or [''])[0]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if not job or job.get('status') != 'done':
                    return self._error('لا يوجد صوت لهذه المهمة.', 404)
                wav = job['wav']
                name = job.get('saved_to') or 'nile_tts.wav'
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(len(wav)))
                if q.get('dl', ['0'])[0] == '1':
                    self.send_header('Content-Disposition',
                                     f'attachment; filename="{name}"')
                self.end_headers()
                self.wfile.write(wav)
            elif u.path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
            else:
                self._error('المسار غير معروف.', 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                               # noqa: BLE001
            try:
                self._error(f'خطأ داخلي: {e}', 500)
            except Exception:
                pass

    # ---------- POST ----------
    def do_POST(self):                                       # noqa: N802
        u = urlparse(self.path)
        try:
            if u.path == '/api/preview':
                data = self._body_json()
                text = str(data.get('text') or '')[:MAX_TEXT_CHARS]
                dialect = 'msa' if data.get('dialect') == 'msa' else 'egy'
                split = bool(data.get('split', True))
                voc = data.get('diacritize') or data.get('vocalize') or 'auto'
                if not text.strip():
                    return self._error('لا يوجد نص للمعاينة.')
                if not infer._AR_LETTERS.search(
                        infer.keep_arabic_only(text)):
                    return self._error('النص لا يحتوي حروفًا عربية.')
                tok_fns = infer.get_tokenizer()
                dialect_pv = dialect
                chunks = (split_into_chunks(text, dialect, tok_fns)
                          if split else [text])
                infos = [{'text': c[:80],
                          'n_tokens': _n_tokens_of(c, dialect, tok_fns)}
                         for c in chunks[:MAX_CHUNKS]]
                warnings = []
                over = [c for c in infos
                        if c['n_tokens'] > infer.TRAIN_MAX_TOKENS]
                if over:
                    warnings.append(
                        f'{len(over)} مقطعًا يتجاوز سقف '
                        f'{infer.TRAIN_MAX_TOKENS} توكن — قسّم النص '
                        'أو فعّل التقسيم التلقائي.')
                if len(chunks) > MAX_CHUNKS:
                    warnings.append(f'عدد المقاطع {len(chunks)} يتجاوز الحد '
                                    f'{MAX_CHUNKS}.')
                will_voc = effective_vocalize_mode(text, voc, dialect_pv) \
                    in ('egyptian', 'fusha')
                arabic_n = len(infer._AR_LETTERS.findall(text))
                self._json({
                    'chars': len(text),
                    'arabic_chars': arabic_n,
                    'n_chunks': min(len(chunks), MAX_CHUNKS),
                    'total_tokens': sum(i['n_tokens'] for i in infos),
                    'chunks': infos,
                    'warnings': warnings,
                    'will_vocalize': will_voc,
                })
            elif u.path == '/api/generate':
                data = self._body_json()
                # ----- تحقق وتقييد المعاملات (نفس حدود infer.py) -----
                text = str(data.get('text') or '').strip()
                if not text:
                    return self._error('اكتب النص أولًا.')
                if len(text) > MAX_TEXT_CHARS:
                    return self._error(
                        f'النص أطول من الحد {MAX_TEXT_CHARS} حرفًا — '
                        'قسّمه على دفعات.')
                speaker = int(data.get('speaker', 0))
                if speaker not in (0, 1):
                    speaker = 0
                dialect = 'msa' if data.get('dialect') == 'msa' else 'egy'
                try:
                    pace = float(data.get('pace', 1.0))
                except (TypeError, ValueError):
                    pace = 1.0
                pace = min(max(pace, 0.5), 2.0)
                diacritize = (data.get('diacritize')
                              or data.get('vocalize') or 'auto')
                if diacritize not in ('auto', 'egyptian', 'fusha', 'manual',
                                      'always', 'never'):
                    diacritize = 'auto'
                diacritize = {'always': 'auto', 'never': 'manual'}.get(
                    diacritize, diacritize)
                try:
                    denoise = float(data.get('denoise', 0.005))
                except (TypeError, ValueError):
                    denoise = 0.005
                denoise = min(max(denoise, 0.0), 0.05)
                split = bool(data.get('split', True))
                ckpt = str(data.get('checkpoint') or 'auto')
                if ckpt != 'auto':
                    ckpt = os.path.basename(ckpt)     # لا مسارات خارجية
                if ckpt != 'auto':
                    valid = {c['name'] for c in list_checkpoints()}
                    if ckpt not in valid:
                        return self._error(
                            f'checkpoint غير معروف: {ckpt}')

                # ----- إنشاء المهمة -----
                gc_jobs()
                job_id = f'j{int(time.time() * 1000):x}'
                job = {
                    'status': 'queued', 'chunk': 0, 'total': 0,
                    'msg': 'في قائمة الانتظار (مهمة أخرى قيد التوليد)'
                           if GEN_LOCK.locked() else 'بدء التوليد...',
                    'created_at': time.time(), 'finished_at': 0,
                    'params': {
                        'text': text, 'speaker': speaker,
                        'dialect': dialect, 'pace': pace,
                        'diacritize': diacritize, 'denoise': denoise,
                        'split': split, 'checkpoint': ckpt,
                    },
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job

                def _mark_then_run():
                    run_job(job_id, job['params'])
                    with JOBS_LOCK:
                        if job_id in JOBS:
                            JOBS[job_id]['finished_at'] = time.time()

                threading.Thread(target=_mark_then_run,
                                 daemon=True).start()
                self._json({'job_id': job_id})
            else:
                self._error('المسار غير معروف.', 404)
        except ValueError as e:
            self._error(str(e), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                               # noqa: BLE001
            try:
                self._error(f'خطأ داخلي: {e}', 500)
            except Exception:
                pass


# ============================================================================
# التشغيل
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='NileTTS 4h — واجهة ويب محلية للتوليد (CPU فقط)')
    ap.add_argument('--port', type=int, default=8500,
                    help='منفذ الخادم (افتراضي 8500)')
    ap.add_argument('--host', default='127.0.0.1',
                    help='العنوان (افتراضي 127.0.0.1 — جهازك فقط)')
    ap.add_argument('--checkpoint', default=None,
                    help='checkpoint محدد (افتراضي: أحدث states_*.pth)')
    ap.add_argument('--threads', type=int, default=None,
                    help='عدد خيوط CPU لـ torch')
    ap.add_argument('--no-browser', action='store_true',
                    help='عدم فتح المتصفح تلقائيًا')
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    if not os.path.exists(INDEX_HTML):
        raise SystemExit(f'[خطأ] ملف الواجهة غير موجود: {INDEX_HTML}')

    # ----- تحميل النماذج مرة واحدة قبل فتح الصفحة -----
    log('=' * 62)
    log('  NileTTS 4h — استوديو النطق (واجهة ويب محلية — CPU)')
    log(f'  بناء الخادم: {server_build()} — البث الفوري عبر /ws (PATCH 19)')
    log('=' * 62)

    ckpt_arg = args.checkpoint
    if ckpt_arg:
        cand = [ckpt_arg,
                os.path.join(infer.CKPT_DIR, ckpt_arg),
                os.path.join(infer.CKPT_DIR,
                             os.path.basename(ckpt_arg))]
        ckpt_path = next((c for c in cand if os.path.exists(c)), None)
        if not ckpt_path:
            raise SystemExit(f'[خطأ] checkpoint غير موجود: {ckpt_arg}')
    else:
        chosen = default_checkpoint()
        if not chosen:
            raise SystemExit(
                '[خطأ] لا يوجد أي checkpoint في مجلد checkpoints/\n'
                '  checkpoint الإنتاج المعتمد (يأتي مع المستودع): '
                + infer.DEFAULT_CHECKPOINT
                + ' — استعده من Git (clean clone) ثم أعد التشغيل.')
        ckpt_path = os.path.join(infer.CKPT_DIR, chosen)

    log(f'[1/3] تحميل المُشكِّل والمُصوِّت (onnx) ...')
    infer.catt_vocalize('اختبار')        # إحماء catt_eo
    infer.vocos_session()                # إحماء vocos22
    log('      المُشكِّل والمُصوِّت جاهزان (CPU)')

    log(f'[2/3] تحميل نموذج MixerTTS من: '
        f'{os.path.basename(ckpt_path)} ...')
    model, it = get_model(ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'      النموذج جاهز (iter {it}) — {n_params:,} بارامتر')

    # PATCH 18: إحماء محرك البث — يشارك النموذج المحمّل نفسه (لا تحميل
    # ثانٍ للأوزان) عندما يكون checkpoint التشغيل هو الإنتاج R1F.
    if os.path.basename(ckpt_path) == infer.DEFAULT_CHECKPOINT:
        get_stream_engine(create=True)
        log('      محرك البث جاهز — التدفق من المتصفح عبر /ws متاح فورًا')
    else:
        log('      [ملاحظة] البث المتدفق يستخدم checkpoint الإنتاج '
            + infer.DEFAULT_CHECKPOINT + ' (سيُحمّل عند أول طلب بث)')

    # ----- الخادم -----
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        raise SystemExit(
            f'[خطأ] تعذر فتح المنفذ {args.port} ({e}) — '
            f'جرّب: python webapp.py --port {args.port + 1}')
    server.daemon_threads = True

    url = f'http://{args.host}:{args.port}'
    log(f'[3/3] الخادم يعمل الآن ← {url}')
    log(f'      كل توليد يُحفظ أيضًا في مجلد web_outputs/')
    log('      لإيقاف الخادم: أغلق النافذة أو اضغط Ctrl+C')
    log('=' * 62)

    if not args.no_browser:
        threading.Timer(0.8,
                        lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log('\nتم إيقاف الخادم. إلى اللقاء!')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
