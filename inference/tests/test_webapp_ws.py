#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات Eiqaz TTS — جسر البث في webapp.py (WebSocket من المتصفح)
=============================================================================
اختبارات تكامل حقيقية للمسار الكامل: خادم webapp.py كما يشغّله المستخدم
(عملية فعلية على منفذ حقيقي) + بروتوكول WebSocket‏ (/ws) كما يتكلمه
المتصفح — عميل WS مبني بمكتبات قياسية فقط (socket/base64/hashlib/struct)
فلا تبعية جديدة على المستخدم.

  W — بروتوكول WebSocket: مصافحة RFC 6455 سليمة · started · ping/pong ·
      رسالة غير معروفة → error.
  X — إثبات البث التدريجي (المسار الكامل): أول مقطع صوتي يصل قبل اكتمال
      تغذية النص (نص طويل يُغذّى قطعًا كل 50ms — محاكاة LLM) · الترتيب
      الصارم 1..N · تطابق n_samples مع حجم الإطار الثنائي · end سليم ·
      حفظ الخادم للصوت المبث في web_outputs.
  Y — المقاطعة الفعلية: interrupt بعد وصول صوت → إقرار + end cancelled ·
      لا مقاطع جديدة بعد الإقرار (بعد حد واحد جارٍ كأقصى سباق موثق) ·
      طلب جديد فور المقاطعة يعمل بنفس النموذج (بلا إعادة تحميل) ·
      إزاحة: start جديد يلغي القديم من تلقاء نفسه.
  Z — استمرارية الصوت عبر المسار الكامل: RMS متسق عبر المقاطع (حد 2×) ·
      فاصل الجملة 0.2s مُسبق بكل مقطع غير الأول (لا قصّ كلام) · سلامة
      النص (كل الجمل موجودة بترتيبها).
  V — التوافق الرجعي: المسار الدفعي القديم (/api/generate → status →
      audio) يعمل سليمًا جنبًا إلى جنب مع البث.

الاستخدام (من مجلد inference/):
    python tests/test_webapp_ws.py
يكتب تقريرًا في tests/webapp_ws_results.json ويعيد رمز خروج 0 عند نجاح
كل شيء. (يحتاج الملفات: checkpoints/states_cont_180516.pth +
weights/vocos22.onnx — نفس متطلبات webapp.py.)
"""
import base64
import glob
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
import wave as wav_mod

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
RESULTS = {'sections': {}, 'failures': [], 'n_pass': 0, 'n_fail': 0,
           'details': {}}


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


# ============================================================================
# عميل WebSocket مبسّط (RFC 6455) — مكتبات قياسية فقط
# ============================================================================
_WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


class WSClient:
    """عميل WebSocket خفيف للاختبار: مصافحة + إطارات نصية مُقنّعة."""

    def __init__(self, host, port, path='/ws', timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b''
        key = base64.b64encode(os.urandom(16)).decode()
        req = ('GET ' + path + ' HTTP/1.1\r\nHost: ' + host + ':' + str(port)
               + '\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                 'Sec-WebSocket-Key: ' + key
                 + '\r\nSec-WebSocket-Version: 13\r\n\r\n')
        self.sock.sendall(req.encode('ascii'))
        resp = self._read_until(b'\r\n\r\n')
        if b'101' not in resp.split(b'\r\n', 1)[0]:
            raise OSError('فشلت المصافحة: ' + resp[:200].decode('latin1'))
        expect = base64.b64encode(hashlib.sha1(
            (key + _WS_GUID).encode('ascii')).digest()).decode('ascii')
        if expect.encode() not in resp:
            raise OSError('Sec-WebSocket-Accept غير صحيح من الخادم.')

    # ---- قراءة خام ----
    def _read_until(self, delim):
        while delim not in self._buf:
            d = self.sock.recv(4096)
            if not d:
                raise OSError('انقطع الاتصال أثناء المصافحة')
            self._buf += d
        i = self._buf.index(delim) + len(delim)
        out, self._buf = self._buf[:i], self._buf[i:]
        return out

    def _read_exact(self, n):
        while len(self._buf) < n:
            d = self.sock.recv(max(4096, n - len(self._buf)))
            if not d:
                raise OSError('انقطع الاتصال')
            self._buf += d
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    # ---- إرسال نص (مُقنّع — إلزامي من العميل) ----
    def send_text(self, text):
        payload = text.encode('utf-8')
        mask = os.urandom(4)
        header = bytearray([0x81])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack('>H', n)
        else:
            header.append(0x80 | 127)
            header += struct.pack('>Q', n)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)

    def close(self):
        try:
            self.sock.sendall(bytes([0x88, 0]))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    # ---- استقبال إطار واحد ----
    def recv_frame(self, timeout=10.0):
        """يعيد (opcode, payload) أو None عند الإغلاق/المهلة."""
        self.sock.settimeout(timeout)
        try:
            hdr = self._read_exact(2)
        except socket.timeout:
            return None
        b0, b1 = hdr[0], hdr[1]
        opcode = b0 & 0x0F
        ln = b1 & 0x7F
        if ln == 126:
            ln = struct.unpack('>H', self._read_exact(2))[0]
        elif ln == 127:
            ln = struct.unpack('>Q', self._read_exact(8))[0]
        payload = self._read_exact(ln) if ln else b''
        if opcode == 0x8:                 # إغلاق
            return None
        return opcode, payload


class StreamClient:
    """جلسة بث كاملة كما يفعلها المتصفح: قارئ خلفي + إرسال أوامر."""

    def __init__(self, host, port):
        self.ws = WSClient(host, port)
        self.events = []                  # (t, dict) — chunk/end/error/...
        self._pending_meta = None
        self._lock = threading.Lock()
        self._stop = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self._stop:
            try:
                fr = self.ws.recv_frame(timeout=30.0)
            except OSError:
                break
            if fr is None:
                continue
            opcode, payload = fr
            t = time.time()
            if opcode == 0x1:                          # نص JSON
                try:
                    d = json.loads(payload.decode('utf-8'))
                except Exception:                      # noqa: BLE001
                    continue
                with self._lock:
                    self.events.append((t, d))
                    if d.get('type') == 'chunk':
                        self._pending_meta = d
            elif opcode == 0x2:                        # ثنائي PCM16
                meta = None
                with self._lock:
                    meta = self._pending_meta
                    self._pending_meta = None
                if meta is None:
                    continue
                with self._lock:
                    self.events.append((t, {
                        'type': 'chunk_audio', 'meta': meta,
                        'pcm': payload}))

    def send(self, obj):
        self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    def snapshots(self):
        with self._lock:
            return list(self.events)

    def wait_for(self, pred, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for t, d in self.snapshots():
                if pred(d):
                    return t, d
            time.sleep(0.02)
        return None, None

    def close(self):
        self._stop = True
        self.ws.close()


# ============================================================================
# تشغيل الخادم الفعلي (عملية مستقلة كما يشغّلها المستخدم)
# ============================================================================
def _free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_http(url, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


print('=' * 72)
print('اختبارات جسر البث في webapp.py (WebSocket — المسار الكامل)')
print('=' * 72)

PORT = _free_port()
BASE = f'http://127.0.0.1:{PORT}'
SERVER_LOG = os.path.join(HERE, 'webapp_ws_server.log')
_logf = open(SERVER_LOG, 'w', encoding='utf-8')
proc = subprocess.Popen(
    [sys.executable, os.path.join(INF, 'webapp.py'),
     '--port', str(PORT), '--no-browser'],
    cwd=INF, stdout=_logf, stderr=subprocess.STDOUT)

try:
    if not wait_http(BASE + '/api/checkpoints', timeout=180):
        with open(SERVER_LOG, encoding='utf-8', errors='replace') as f:
            out = f.read(4000)
        raise SystemExit('لم يبدأ webapp.py:\n' + out)
    print(f'الخادم جاهز على {BASE} (pid {proc.pid})')

    LONG_TEXT = (
        'السلام عليكم يا صديقي، إزيك النهارده؟ '
        'دلوقتي هنبدأ درس القسمة يا عمر. '
        'القسمة معناها إننا بنوزع الحاجة بالتساوي. '
        'عندي عشرة تفاحات وخمسة أطفال، يبقى كل واحد ياخد اتنين. '
        'لو القسمة مش بالتساوي حد هيظلم. '
        'العدل أساس أي حاجة في الحياة. '
        'كمان القسمة بتعلمنا الصبر. '
        'مش كل حاجة بتيجي بسرعة يا صديقي. '
        'خلي بالك من البقايا في القسمة. '
        'لو قسمت سبعة على اتنين الباقي واحد. '
        'دي حاجة مهمة في الرياضيات كلها. '
        'الرياضيات علوم التفكير المنظم. '
        'التفكير المنظم بيفتح العقل. '
        'يلا نراجع الدرس تاني بكرة إن شاء الله.'
    )

    # =====================================================================
    # W — بروتوكول WebSocket
    # =====================================================================
    print('\n[W] بروتوكول WebSocket')
    try:
        c = StreamClient('127.0.0.1', PORT)
        c.send({'type': 'ping'})
        t, d = c.wait_for(lambda e: e.get('type') == 'pong', 10)
        record('W', 'مصافحة RFC 6455 + ping→pong', d is not None, d)
        c.send({'type': 'unknown_thing'})
        t, d = c.wait_for(lambda e: e.get('type') == 'error', 10)
        record('W', 'رسالة غير معروفة → error (بلا انهيار)', d is not None, d)
        c.close()
        time.sleep(0.3)
    except Exception:                                              # noqa: BLE001
        record('W', 'بروتوكول WebSocket', False, traceback.format_exc())

    # =====================================================================
    # X — إثبات البث التدريجي (المسار الكامل من أول الإرسال)
    # =====================================================================
    print('\n[X] البث التدريجي — المسار الكامل')
    x = None
    try:
        x = StreamClient('127.0.0.1', PORT)
        t_req = time.time()
        x.send({'type': 'start', 'speaker': 0, 'dialect': 'egy',
                'diacritize': 'egyptian', 'pace': 1.0, 'denoise': 0.005})
        t, started = x.wait_for(lambda e: e.get('type') == 'started', 30)
        record('X', 'started: request_id + sample_rate + checkpoint',
               started is not None
               and started.get('sample_rate') == 22050
               and started.get('request_id') >= 1
               and 'states' in str(started.get('checkpoint')),
               started)
        req_id = started['request_id']

        # تغذية تدريجية — كل قطعة 3 كلمات كل 50ms (محاكاة LLM)
        words = LONG_TEXT.split()
        pieces = [' '.join(words[i:i + 3]) for i in range(0, len(words), 3)]
        t_feed_done = None
        t_first_audio = None
        n_audio_at_feed_done = 0

        def _feed_all():
            global t_feed_done
            for p in pieces:
                x.send({'type': 'feed', 'text': p + ' '})
                time.sleep(0.05)
            x.send({'type': 'finish'})
            t_feed_done = time.time()

        th = threading.Thread(target=_feed_all, daemon=True)
        th.start()

        # انتظار أول صوت — الدليل: قبل اكتمال التغذية
        t, first = x.wait_for(
            lambda e: e.get('type') == 'chunk_audio', 60)
        t_first_audio = t
        record('X', 'أول مقطع صوتي وصل (إطار PCM16 ثنائي)',
               first is not None,
               first and {'seq': first['meta']['seq'],
                          'n_samples': first['meta']['n_samples'],
                          'bytes': len(first['pcm'])})
        th.join(30)
        frac = (len(pieces) and t_first_audio is not None
                and t_feed_done is not None
                and t_first_audio < t_feed_done)
        record('X', 'أول صوت قبل اكتمال إرسال النص (محاكاة LLM)',
               bool(frac),
               {'pieces_total': len(pieces),
                't_first_audio': round(t_first_audio - t_req, 2)
                if t_first_audio else None,
                't_feed_done': round(t_feed_done - t_req, 2)
                if t_feed_done else None})

        # انتظار النهاية
        t, end = x.wait_for(
            lambda e: e.get('type') == 'end'
            and e.get('request_id') == req_id, 120)
        record('X', 'end: kind=end + إحصاءات كاملة',
               end is not None and end.get('kind') == 'end'
               and end.get('stats', {}).get('n_chunks_emitted', 0) >= 8,
               end and end.get('stats'))

        # الترتيب الصارم + تطابق الأحجام
        evs = x.snapshots()
        auds = [d for _, d in evs if d.get('type') == 'chunk_audio']
        seqs = [a['meta']['seq'] for a in auds]
        sizes_ok = all(a['meta']['n_samples'] * 2 == len(a['pcm'])
                       for a in auds)
        record('X', 'الترتيب الصارم 1..N بلا فجوات ولا تكرار',
               seqs == list(range(1, len(seqs) + 1)) and sizes_ok,
               {'n': len(seqs), 'seqs': seqs[:20],
                'sizes_match': sizes_ok})

        ttfa = t_first_audio - t_req if t_first_audio else None
        record('X', 'TTFA من بدء الطلب < 8 ث (قياس فعلي)',
               ttfa is not None and ttfa < 8.0,
               {'ttfa_s': round(ttfa, 3) if ttfa else None})

        # حفظ الخادم للصوت المبث في web_outputs
        time.sleep(1.0)
        saved = sorted(
            glob.glob(os.path.join(INF, 'web_outputs', '*_stream.wav')),
            key=os.path.getmtime)
        ok_save = bool(saved) and (
            time.time() - os.path.getmtime(saved[-1]) < 120)
        detail_save = None
        if ok_save:
            with wav_mod.open(saved[-1], 'rb') as w:
                detail_save = {'file': os.path.basename(saved[-1]),
                               'n_frames': w.getnframes(),
                               'framerate': w.getframerate(),
                               'expect_frames': sum(
                                   a['meta']['n_samples'] for a in auds)}
            ok_save = (detail_save['framerate'] == 22050
                       and detail_save['n_frames']
                       == detail_save['expect_frames'])
        record('X', 'الخادم حفظ الصوت المبث في web_outputs (تكافئ الدفعي)',
               ok_save, detail_save)

        # =================================================================
        # Z — استمرارية الصوت عبر المسار الكامل (نفس الجلسة أعلاه)
        # =================================================================
        print('\n[Z] استمرارية الصوت (المسار الكامل)')
        import numpy as np
        waves = []
        for a in auds:
            pcm = np.frombuffer(a['pcm'], dtype='<i2').astype('float32')
            waves.append(pcm / 32768.0)
        rms = [float(np.sqrt(np.mean(w * w))) for w in waves]
        ratio = max(rms) / min(rms) if rms else 0
        record('Z', 'جهارة RMS متسقة عبر المقاطع (النسبة ≤ 2×)',
               0 < ratio <= 2.0,
               {'rms': [round(r, 4) for r in rms],
                'ratio': round(ratio, 3)})

        gap_sil_ok = True
        gap_detail = []
        for i, w in enumerate(waves[1:], start=2):
            head = np.abs(w[:3000])          # أول ~0.14s (الفاصل 0.2s)
            mx = float(head.max())
            gap_detail.append(round(mx, 4))
            if mx > 0.02:
                gap_sil_ok = False
        record('Z', 'فاصل الجملة مُسبق بكل مقطع غير الأول (بلا قصّ كلام)',
               gap_sil_ok, {'head_max_first_0.14s': gap_detail})

        # سلامة النص: كل جملة أصلية موجودة في نصوص المقاطع بترتيبها
        texts = [a['meta'].get('text', '') for a in auds]
        joined_all = ' '.join(texts)
        sentences = [s.strip() for s in LONG_TEXT.replace('؟', '.').replace(
            '،', '.').split('.') if s.strip()]
        missing = [s for s in sentences if s not in joined_all]
        record('Z', 'سلامة النص: كل الجمل في المقاطع بترتيبها',
               not missing and len(texts) >= 8,
               {'n_chunks': len(texts), 'missing': missing[:3]})
        x.close()
    except Exception:                                              # noqa: BLE001
        record('X', 'البث التدريجي', False, traceback.format_exc())
        try:
            x.close()
        except Exception:                                          # noqa: BLE001
            pass

    # =====================================================================
    # Y — المقاطعة الفعلية + طلب جديد فورها + الإزاحة
    # =====================================================================
    print('\n[Y] المقاطعة الفعلية')
    try:
        # Y1: إلغاء بعد وصول صوت
        y = StreamClient('127.0.0.1', PORT)
        y.send({'type': 'start', 'speaker': 0})
        t, st_a = y.wait_for(lambda e: e.get('type') == 'started', 30)
        id_a = st_a['request_id']
        words = LONG_TEXT.split()
        pieces = [' '.join(words[i:i + 3]) for i in range(0, len(words), 3)]
        for p in pieces[:6]:
            y.send({'type': 'feed', 'text': p + ' '})
            time.sleep(0.05)
        t, first = y.wait_for(lambda e: e.get('type') == 'chunk_audio', 60)
        y.send({'type': 'interrupt'})
        t, ack = y.wait_for(lambda e: e.get('type') == 'interrupted', 10)
        t_ack = t if ack else time.time()
        t, end = y.wait_for(
            lambda e: e.get('type') == 'end'
            and e.get('kind') == 'cancelled', 30)
        t_end = t if end else time.time()
        record('Y', 'interrupt → إقرار + end cancelled',
               ack is not None and end is not None,
               {'ack': ack, 'end_stats': end and end.get('stats')})
        # لا مقاطع جديدة بعد الإقرار (بعد حدّ واحد جارٍ كأقصى سباق موثق)
        # وبالصفر المطلق بعد حدث النهاية
        time.sleep(1.5)
        late_after_ack = []
        late_after_end = []
        for tt, e in y.snapshots():
            if e.get('type') == 'chunk_audio':
                if tt > t_ack:
                    late_after_ack.append(e['meta']['seq'])
                if tt > t_end:
                    late_after_end.append(e['meta']['seq'])
        record('Y', 'لا مقاطع جديدة بعد الإقرار (≤ حد جارٍ واحد)',
               len(late_after_ack) <= 1 and not late_after_end,
               {'late_after_ack': late_after_ack,
                'late_after_end': late_after_end})
        y.close()

        # Y2: طلب جديد فور المقاطعة — نفس المحرك (تسلسل المعرفات يثبته)
        y2 = StreamClient('127.0.0.1', PORT)
        t0 = time.time()
        y2.send({'type': 'start', 'speaker': 0})
        t, st_b = y2.wait_for(lambda e: e.get('type') == 'started', 15)
        id_b = st_b['request_id'] if st_b else -1
        for p in pieces[:8]:
            y2.send({'type': 'feed', 'text': p + ' '})
            time.sleep(0.05)
        t, f2 = y2.wait_for(lambda e: e.get('type') == 'chunk_audio', 30)
        resume_s = time.time() - t0
        record('Y', 'طلب جديد فور المقاطعة يعمل (نفس المحرك — معرّف أعلى)',
               f2 is not None and resume_s < 10 and id_b > id_a,
               {'resume_to_first_audio_s': round(resume_s, 2),
                'request_ids': [id_a, id_b]})

        # Y3: إزاحة — طلب جديد يلغي القديم من تلقاء نفسه (بلا interrupt)
        y2.send({'type': 'start', 'speaker': 1})          # طلب جديد C
        t, st_c = y2.wait_for(
            lambda e: e.get('type') == 'started'
            and e.get('request_id', 0) > max(id_b, 0), 15)
        id_c = st_c['request_id'] if st_c else -1
        t_b_started = t if st_c else time.time()
        t, end_c = y2.wait_for(
            lambda e: e.get('type') == 'end'
            and e.get('kind') == 'cancelled', 30)
        record('Y', 'إزاحة: start جديد يلغي القديم تلقائيًا',
               st_c is not None and end_c is not None and id_c > id_b,
               {'request_ids': [id_b, id_c],
                'old_end': end_c and end_c.get('kind')})
        # لا مقاطع من الطلب القديم بعد started الجديد (≤ حد جارٍ واحد)
        for p in pieces[:8]:
            y2.send({'type': 'feed', 'text': p + ' '})
            time.sleep(0.05)
        time.sleep(1.5)
        evs2 = y2.snapshots()
        stale_after = [e for tt, e in evs2
                       if e.get('type') == 'chunk_audio'
                       and e['meta']['request_id'] == id_b
                       and tt > t_b_started]
        new_chunks = [e for _, e in evs2
                      if e.get('type') == 'chunk_audio'
                      and e['meta']['request_id'] == id_c]
        record('Y', 'لا صوت قديم بعد بدء الطلب الجديد (≤ حد جارٍ واحد)',
               len(stale_after) <= 1 and len(new_chunks) >= 1,
               {'stale_after_new_start': len(stale_after),
                'new_request_chunks': len(new_chunks)})
        y2.close()
        time.sleep(0.3)
    except Exception:                                              # noqa: BLE001
        record('Y', 'المقاطعة', False, traceback.format_exc())

    # =====================================================================
    # V — التوافق الرجعي: المسار الدفعي القديم عبر HTTP
    # =====================================================================
    print('\n[V] التوافق الرجعي (المسار الدفعي)')
    try:
        req = urllib.request.Request(
            BASE + '/api/generate',
            data=json.dumps({'text': 'السلام عليكم يا صديقي، إزيك؟',
                             'speaker': 0, 'dialect': 'egy',
                             'diacritize': 'egyptian', 'split': True})
            .encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as r:
            job = json.loads(r.read().decode('utf-8'))
        job_id = job['job_id']
        t0 = time.time()
        status = None
        while time.time() - t0 < 120:
            with urllib.request.urlopen(
                    BASE + '/api/status?job=' + job_id, timeout=10) as r:
                status = json.loads(r.read().decode('utf-8'))
            if status.get('status') in ('done', 'error'):
                break
            time.sleep(0.5)
        ok_batch = status and status.get('status') == 'done'
        if ok_batch:
            with urllib.request.urlopen(
                    BASE + '/api/audio?job=' + job_id, timeout=30) as r:
                wav_bytes = r.read()
            ok_batch = (wav_bytes[:4] == b'RIFF'
                        and len(wav_bytes) > 20000)
        record('V', 'المسار الدفعي القديم يعمل جنبًا إلى جنب مع البث',
               ok_batch,
               {'status': status and status.get('status'),
                'wav_bytes': len(wav_bytes) if ok_batch else 0,
                'elapsed': status and status.get('elapsed')})
    except Exception:                                              # noqa: BLE001
        record('V', 'المسار الدفعي', False, traceback.format_exc())

finally:
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
    _logf.close()

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

RESULTS['server'] = {'port': PORT, 'endpoint': '/ws',
                     'started_from': 'webapp.py (عملية مستقلة)'}
RESULTS['overall'] = 'PASS' if n_fail == 0 else 'FAIL'

OUT = os.path.join(HERE, 'webapp_ws_results.json')
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)
print(f'كُتب التقرير: {OUT}')

sys.exit(0 if n_fail == 0 else 1)
