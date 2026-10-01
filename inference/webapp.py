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
"""

import argparse
import glob
import io
import json
import os
import re
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
        m = re.search(r'states_(\d+)\.pth$', p)
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
    snaps = list_checkpoints()
    return snaps[0]['name'] if snaps else None


# ============================================================================
# تقسيم النص الطويل إلى مقاطع (قبل التنظيف — على النص الخام)
# ============================================================================
_SENT_END = re.compile(r'(?<=[.!؟?…])\s+')
_SOFT_SPLIT = re.compile(r'\s*[,،؛;:]+\s*')


def _n_tokens_of(text, dialect, tok_fns):
    toks_ms, toks_egy, ids_of = tok_fns
    cleaned = infer.keep_arabic_only(text)
    if not infer._AR_LETTERS.search(cleaned):
        return 0
    toks = toks_ms(cleaned) if dialect == 'msa' else toks_egy(cleaned)
    return len(ids_of(toks))


def split_into_chunks(raw_text, dialect, tok_fns,
                      max_tokens=infer.TRAIN_MAX_TOKENS):
    """تقسيم النص الخام إلى مقاطع عند نهايات الجمل.

    سياسة التقسيم:
      1) فواصل الأسطر أولًا، ثم . ! ؟ ؟ …
      2) دمج المقاطع الضئيلة جدًا (< 6 توكن) مع جارتها
      3) المقطع الأطول من السقف يُقسَّم عند الفواصل الناعمة (، ؛ ; :)
         ثم عند حدود الكلمات كخيار أخير
    كل مقطع يمرّ لاحقًا بنفس prepare_text/synthesize المستخدمة في infer.py.
    """
    segments = [s.strip() for s in raw_text.split('\n') if s.strip()]
    pieces = []
    for seg in segments:
        for p in _SENT_END.split(seg):
            p = p.strip(' \t\r,.،؛;:…!؟?')
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
    return [c for c in chunks if infer._AR_LETTERS.search(
        infer.keep_arabic_only(c))]


def effective_vocalize_mode(raw_text, mode):
    """توحيد قرار التشكيل على كل المقاطع (كما يحسمه auto في infer.py)."""
    if mode in ('always', 'never'):
        return mode
    density, _ = infer.diacritic_density(infer.keep_arabic_only(raw_text))
    return 'always' if density < 0.30 else 'never'


# ============================================================================
# تنفيذ مهمة التوليد (في خيط منفصل)
# ============================================================================
def run_job(job_id, params):
    import soundfile as sf
    import numpy as np

    job = JOBS[job_id]
    t0 = time.time()
    tmp_dir = None
    qaf_mode = params.get('qaf') or 'auto'
    det_partial = bool(params.get('det_partial'))
    try:
        with GEN_LOCK:
            job['status'] = 'running'
            tok_fns = infer.get_tokenizer(qaf_mode)

            # ----- checkpoint -----
            ckpt_name = params['checkpoint']
            if ckpt_name in (None, '', 'auto'):
                ckpt_name = default_checkpoint()
            if not ckpt_name:
                raise SystemExit(
                    '[خطأ] لا يوجد checkpoint في مجلد checkpoints/ — '
                    'ضع states_79590.pth هناك أولًا.')
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

            voc_mode = effective_vocalize_mode(raw, params['vocalize'])
            job['vocalized'] = voc_mode == 'always'

            # ----- توليد كل مقطع بنفس دالتي infer.py -----
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix='nile_web_')
            waves = []
            n_tokens_total = 0
            for i, chunk_raw in enumerate(chunks):
                job['chunk'] = i + 1
                job['msg'] = f'جاري توليد المقطع {i + 1} من {len(chunks)}'
                text, _ = infer.prepare_text(chunk_raw, voc_mode,
                                             params['dialect'], qaf_mode,
                                             det_partial)
                tmp_wav = os.path.join(tmp_dir, f'chunk_{i:03d}.wav')
                n_tok, _ = infer.synthesize(
                    model, text, params['dialect'], params['speaker'],
                    params['pace'], tmp_wav, params['denoise'],
                    qaf_mode=qaf_mode)
                n_tokens_total += n_tok
                w, _ = sf.read(tmp_wav, dtype='float32')
                waves.append(w)
                job['elapsed'] = round(time.time() - t0, 1)

            # ----- تجميع الموجات مع فاصل صمت (تجميع مخرجات فقط) -----
            gap = np.zeros(int(CHUNK_GAP_S * SAMPLE_RATE), dtype='float32')
            if len(waves) == 1:
                final = waves[0]
            else:
                final = waves[0]
                for w in waves[1:]:
                    final = np.concatenate([final, gap, w])

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
                'qaf': qaf_mode,
                'det_partial': det_partial,
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

    # ---------- GET ----------
    def do_GET(self):                                        # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ('/', '/index.html'):
                with open(INDEX_HTML, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type',
                                 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
                voc = data.get('vocalize') or 'auto'
                if not text.strip():
                    return self._error('لا يوجد نص للمعاينة.')
                if not infer._AR_LETTERS.search(
                        infer.keep_arabic_only(text)):
                    return self._error('النص لا يحتوي حروفًا عربية.')
                tok_fns = infer.get_tokenizer()
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
                will_voc = effective_vocalize_mode(text, voc) == 'always'
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
                vocalize = data.get('vocalize') or 'auto'
                if vocalize not in ('auto', 'always', 'never'):
                    vocalize = 'auto'
                qaf = data.get('qaf') or 'auto'
                if qaf not in getattr(infer, 'QAF_MODES', ('auto',)):
                    qaf = 'auto'
                det_partial = bool(data.get('det_partial', False))
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
                        'vocalize': vocalize, 'denoise': denoise,
                        'split': split, 'checkpoint': ckpt,
                        'qaf': qaf, 'det_partial': det_partial,
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
                '  ضع states_79590.pth (من مخرجات Kaggle) هناك ثم أعد '
                'التشغيل.')
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
