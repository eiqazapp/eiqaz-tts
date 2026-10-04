# ============================================================================
# Eiqaz TTS v1 — PREP kernel (eiqaz-prep-v1)  —  NO TRAINING HERE
# ============================================================================
# مشروع تدريب مستقل تمامًا عن "NileTTS 4h Train" (مرجع هندسي فقط — لم يُمَس).
#
# المدخلات (كلها READ-ONLY):
#   dataset eiqaz-v1-inputs        : egy_tashkeel_canonical.jsonl (21,854 نصًا
#                                    مصريًا مشكولًا باللهجة المصرية — LLM
#                                    tashkeel-ai، لا يُعاد تشكيله أبدًا)
#                                    + extraction.csv (خريطة الوحدات: src_row,
#                                    i0, i1, spk, split — من مخرجات prep
#                                    القديم، قراءة فقط، لا تعديل)
#                                    + eqz_tokens.py (سياسة القاف v1)
#                                    + eiqaz_eval_set.json
#   dataset msa-tts-data-v1        : clartts/wavs + cv_female/wavs + filelist.csv
#   dataset mixer-tts-scratch-data : mixer_repo + tts_arabic_pkg (كود فقط)
#
# المصدر الصوتي المصري: KickItLikeShika/NileTTS-dataset (HF) — تنزيل مجرى
# على دفعات (نفس منهج prep v2 القديم: دفعة ~2GB → معالجة → حذف).
# التقطيع: بخريطة extraction.csv الجاهزة (i0/i1 في مجال 22050Hz mono بعد
# تطبيع ذروة الصف ≤0.95 — نفس معالجة المرجع حرفيًا) — لا إعادة اشتقاق
# للتقطيع، فمحاذاة معرفات الوحدات مع corpus التشكيل مضمونة.
#
#   kernel eiqaz-prep-v1 output : features/ — استئناف عبر الجلسات (self-attach)
#
# v1.1 (2026-10-04) — دروس الجلسة 1 (انهيار عند 3h25m):
#   1) hf_fetch: 5 محاولات بتراجع أسّي + كتابة ذرية .part→rename
#      (ConnectionResetError واحد كان أسقط مرحلة المصرية كلها)
#   2) تنزيل متوازٍ DL_WORKERS=8 (كان متسلسلًا)
#   3) seed_from_prev: استئناف ملامح النسخة السابقة عبر self-attach
#   4) _msa_units.json تراكمي (الكتابة الفوقية كانت ستمحو وحدات الجلسة
#      السابقة عند الاستئناف)
#   5) صف التنزيل الفاشل بعد المحاولات يُتخطى (resumable) بدل الانهيار
#   6) علم complete بحقيقة أرضية في prep_report.json (بوابة نواة التدريب)
#
# v1.2 (2026-10-04) — خطأ بادئة MSA (اكتُشف بتحليل النسخة 2 الجارية):
#   1) filelist utt بلا بادئة بينما الملامح msa_<utt>.pt — كان يفسد
#      فلتر استئناف MSA (يعيد كل الصفوف كل جلسة) وحساب msa_remaining
#      (يبقى = الكل أبدًا → complete=false أبدًا → بوابة التدريب موصدة).
#      الاختبار المحلي فاته لأن محاكاة filelist كانت utt بالبادئة —
#      الاختبار v1.2 يستخدم البيانات الواقعية (بلا بادئة).
#   2) قوائم الحجر _dead_egy.json/_dead_msa.json: الصفوف المستسلمة
#      نهائيًا (skip_tokens/skip_frames/error) تُسجَّل بالسبب وتُستثنى
#      من remaining بشرط سقف DEAD_CAP لكل جهة — تجاوزه مشكلة نظامية
#      → complete=false. تُنسخ عبر seed_from_prev وتُعاد محاولتها كل
#      جلسة (رخيصة: عشرات صفوف) — نجاحها لاحقًا يرفعها من القائمة.
#
# المخرجات (مستقلة — /kaggle/working):
#   features/{utt}.pt   : {ids, mel[80,T], pitch[T]} — نفس صيغة المرجع
#   features/index.json : فهرس الخلط (مصري+فصحى) + pitch stats + خريطة
#                         المتحدثين الجديدة + إحصاءات q/v/همزة
#   prep_report.json    : التقرير الكامل (أعداد/ساعات/نسب/إسقاطات)
#
# الفروق الجوهرية عن niletts-4h-prep (الموثقة):
#   1. النص المصري = corpus التشكيل المصري LLM كما هو (لا catt_eo إطلاقًا)
#   2. الترميز المصري = eqz_tokens v1: q_default→'q' (98 هيكلًا)،
#      g_default→'v' (22 هيكلًا)، الباقي→'<' — q ستُدرَّب (كانت 0/1.59M)
#   3. بيانات MSA تُعالج في نفس النواة بخريطة متحدثين موحدة جديدة
#   4. لا resume من مخرجات قديمة — المشروع له مخرجاته المستقلة فقط
#
# Phases (argv): driver | egy | msa | report
# ============================================================================
import os
import sys
import json
import time
import glob
import subprocess

PHASE = sys.argv[1] if len(sys.argv) > 1 else 'driver'
T0 = time.time()


def el():
    return f"[{time.strftime('%H:%M:%S', time.gmtime(time.time() - T0))}]"


def log(*a):
    print(el(), *a, flush=True)


INPUT = os.environ.get('KAGGLE_INPUT_ROOT', '/kaggle/input')
WORK = os.environ.get('KAGGLE_WORK', '/kaggle/working')
FEAT_DIR = os.path.join(WORK, 'features')
TMP = os.environ.get('KAGGLE_TMP', '/kaggle/tmp')
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# onnxruntime غير مثبت مسبقًا في صورة Kaggle (درس prep v1 القديم)
try:
    import onnxruntime  # noqa: F401
except ImportError:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'onnxruntime'], capture_output=True, text=True)
    log(f'[deps] onnxruntime install rc={r.returncode}')

# ---------------------------------------------------------------------------
# ثوابت المشروع v1
# ---------------------------------------------------------------------------
NILETTS_REPO = 'KickItLikeShika/NileTTS-dataset'
HF_RESOLVE = f'https://huggingface.co/datasets/{NILETTS_REPO}/resolve/main/'
HF_TREE = f'https://huggingface.co/api/datasets/{NILETTS_REPO}/tree/main'
BATCH_BYTES = 2 * 1024 ** 3            # دفعات تنزيل ~2GB (مجرية)
TARGET_SR = 22050
WRITE_TOK_CAP = 180                    # سقف الكتابة (نفس المرجع)
TRAIN_TOK_CAP = 160                    # علم أهلية train-pool (نفس المرجع)
MAX_FRAMES = 950                       # سقف إطارات التدريب (نفس المرجع)
EGY_PROCESS_CAP = 8.0 * 3600           # سقف معالجة المصري في الجلسة
MSA_PROCESS_CAP = 2.5 * 3600           # سقف معالجة MSA
DOWNLOAD_CAP = 3.0 * 3600
DL_WORKERS = 8                          # تنزيل متوازٍ من HF (v1.1)
DEAD_CAP = {'egy': 150, 'msa': 150}     # سقف الحجر لكل جهة (v1.2)
# 21,854 مصري / 13,110 فصحى → 150 ≈ 0.7% / 1.1%: فوقها مشكلة نظامية

# خريطة المتحدثين الجديدة المستقلة v1 (7 متحدثين؛ سعة النموذج n_speakers=16)
SPEAKER_MAP_V1 = {
    'egy_SPEAKER_01': 0,      # ذكر مصري (NileTTS)
    'egy_SPEAKER_02': 1,      # أنثى مصرية (NileTTS)
    'msa_clartts_male': 2,    # ذكر فصيح كلاسيكي (ClArTTS)
    'msa_cvf_5f810213': 3,    # أنثى فصحى (Common Voice)
    'msa_cvf_78c954e3': 4,    # أنثى فصحى (Common Voice)
    'msa_cvf_cf4d8f89': 5,    # أنثى فصحى (Common Voice)
    'msa_cvf_fc3b87e3': 6,    # أنثى فصحى (Common Voice)
}


def find_root(marker, what):
    env = os.environ.get('KAGGLE_DATA')
    if env and os.path.exists(os.path.join(env, marker)):
        return env
    hits = sorted(glob.glob(f'{INPUT}/**/{marker}', recursive=True))
    if hits:
        root = hits[0]
        for _ in range(marker.count('/') + 1):
            root = os.path.dirname(root)
        return root
    raise FileNotFoundError(f'{what} (marker {marker}) not found under '
                            f'{INPUT}; entries: ' + repr(os.listdir(INPUT)))


def setup_paths():
    """يجهز مسارات الأكواد (mixer_repo + tts_arabic) ومدخلات eiqaz."""
    data = find_root('models/mixer128_pytorch.pth', 'mixer-tts-scratch-data')
    eqz = find_root('eqz_tokens.py', 'eiqaz-v1-inputs')
    for p in (os.path.join(data, 'mixer_repo'),
              os.path.join(data, 'tts_arabic_pkg'),
              eqz):
        if p not in sys.path:
            sys.path.insert(0, p)
    return data, eqz


def load_egy_inputs():
    """corpus التشكيل المصري (كما هو — لا أي إعادة تشكيل) + خريطة الوحدات."""
    eqz = find_root('eqz_tokens.py', 'eiqaz-v1-inputs')
    tash = {}
    path = os.path.join(eqz, 'egy_tashkeel_canonical.jsonl')
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                tash[r['utt']] = (r['text'], r.get('split', ''))
    import pandas as pd
    ex = pd.read_csv(os.path.join(eqz, 'extraction.csv'))
    return tash, ex, eqz


# ---------------------------------------------------------------------------
# عامل المعالجة (يعمل في pool spawn — درس prep v1: لا fork بعد librosa)
# ---------------------------------------------------------------------------
_WS = {}


def _worker_init(data_root, eqz_root):
    for p in (os.path.join(data_root, 'mixer_repo'),
              os.path.join(data_root, 'tts_arabic_pkg'), eqz_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    _WS['data'] = data_root
    _WS['eqz'] = eqz_root


def _get_toks():
    if 'toks' not in _WS:
        import eqz_tokens as ET
        _WS['toks'] = ET.get_tokenizers()
    return _WS['toks']


def _get_mel_fn():
    if 'mel' not in _WS:
        from utils.audio import MelSpectrogram
        _WS['mel'] = MelSpectrogram()
    return _WS['mel']


def _wav_to_feat(ids, wav):
    """مقطع صوتي (float32 mono 22050) -> (mel, pitch) أو None."""
    import numpy as np
    import torch
    import librosa
    mel_fn = _get_mel_fn()
    wav_t = torch.from_numpy(np.ascontiguousarray(wav))[None]
    mel = mel_fn(wav_t).clamp_min(1e-5).log().squeeze(0)   # [80, T]
    f0, _, _ = librosa.pyin(wav, sr=TARGET_SR, fmin=60, fmax=600,
                            frame_length=1024, hop_length=256)
    f0 = np.where(np.isnan(f0), 0., f0)
    pitch = torch.from_numpy(f0).float()
    if pitch.size(0) < mel.size(1):
        pitch = torch.nn.functional.pad(
            pitch, (0, mel.size(1) - pitch.size(0)))
    else:
        pitch = pitch[:mel.size(1)]
    return mel, pitch


def egy_process_row(args):
    """صف NileTTS واحد -> وحداته كملامح.

    args: (wav_path, [(utt, text, split, spk, i0, i1), ...])
    يعيد {'status', 'units': [...], 'drops': [...]}.
    الصوت: قراءة -> مونو -> 22050 -> تطبيع ذروة الصف ≤0.95 (نفس المرجع)
    -> تقطيع [i0:i1] -> ترميز eqz_v1 (بلا أي إعادة تشكيل للنص) -> ملامح.
    """
    import numpy as np
    import torch
    import soundfile as sf
    import librosa
    wav_path, units = args
    drops = []
    out = []
    try:
        wav, sr = sf.read(wav_path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        if peak > 0.95:
            wav = wav * (0.95 / peak)
        toks_ms, toks_egy, ids_of, _ = _get_toks()
        for (utt, text, split, spk, i0, i1) in units:
            u_wav = wav[i0:i1]
            if u_wav.size < TARGET_SR:            # < 1s — حماية
                drops.append((utt, 'slice_too_short'))
                continue
            try:
                toks = toks_egy(text)             # النص كما هو — لا catt
                ids = ids_of(toks)
            except Exception as e:                # noqa: BLE001
                drops.append((utt, f'tokenize:{type(e).__name__}'))
                continue
            if len(ids) < 5 or len(ids) > WRITE_TOK_CAP:
                drops.append((utt, f'skip_tokens(n={len(ids)})'))
                continue
            try:
                mel, pitch = _wav_to_feat(ids, u_wav)
            except Exception as e:                # noqa: BLE001
                drops.append((utt, f'feat:{type(e).__name__}'))
                continue
            n_fr = int(mel.size(1))
            if n_fr > MAX_FRAMES:
                drops.append((utt, f'frames>{MAX_FRAMES}'))
                continue
            torch.save({'ids': torch.LongTensor(ids), 'mel': mel,
                        'pitch': pitch},
                       os.path.join(FEAT_DIR, utt + '.pt'))
            out.append({'utt': utt, 'dialect': 'egy', 'spk': int(spk),
                        'n_tokens': len(ids), 'n_frames': n_fr,
                        'dur_s': round(u_wav.size / TARGET_SR, 2),
                        'split': split,
                        'n_q': int(sum(1 for t in toks if t == 'q')),
                        'n_v': int(sum(1 for t in toks if t == 'v'))})
        return {'status': 'ok', 'units': out, 'drops': drops}
    except Exception as e:                        # noqa: BLE001
        return {'status': 'error',
                'err': f'{type(e).__name__}: {e}', 'units': out,
                'drops': drops}


def msa_process_row(args):
    """صف MSA واحد (wav جاهز 22050 PCM16) -> ملمة واحدة.

    args: (wav_path, utt, text, split, spk)
    الترميز: toks_ms (فصحى — q محفوظة خام)."""
    import numpy as np
    import torch
    import soundfile as sf
    wav_path, utt, text, split, spk = args
    try:
        wav, sr = sf.read(wav_path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != TARGET_SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        toks_ms, toks_egy, ids_of, _ = _get_toks()
        ids = ids_of(toks_ms(text))
        if len(ids) < 5 or len(ids) > WRITE_TOK_CAP:
            return {'status': 'skip_tokens', 'n': len(ids), 'utt': utt}
        mel, pitch = _wav_to_feat(ids, wav)
        n_fr = int(mel.size(1))
        if n_fr > MAX_FRAMES:
            return {'status': 'skip_frames', 'n': n_fr, 'utt': utt}
        torch.save({'ids': torch.LongTensor(ids), 'mel': mel,
                    'pitch': pitch},
                   os.path.join(FEAT_DIR, 'msa_' + utt + '.pt'))
        return {'status': 'ok',
                'unit': {'utt': 'msa_' + utt, 'dialect': 'msa',
                         'spk': int(spk), 'n_tokens': len(ids),
                         'n_frames': n_fr,
                         'dur_s': round(wav.size / TARGET_SR, 2),
                         'split': split,
                         'n_q': int(sum(1 for t in toks_ms(text) if t == 'q')),
                         'n_v': 0}}
    except Exception as e:                        # noqa: BLE001
        return {'status': 'error', 'err': f'{type(e).__name__}: {e}',
                'utt': utt}


# ---------------------------------------------------------------------------
# تنزيل NileTTS من HF (مجرى على دفعات — منهج prep v2 القديم)
# ---------------------------------------------------------------------------
def hf_fetch(url, dest, timeout=300, attempts=5):
    """تنزيل مع إعادة محاولة + كتابة ذرية (.part ثم rename).

    درس الجلسة 1: ConnectionResetError واحد أثناء urlopen أسقط مرحلة
    المصرية كاملة عند 3h25m (8,008/21,854). الآن: محاولات بتراجع أسّي،
    والملف الناقص لا يظهر أبدًا باسم الوجهة النهائية."""
    import urllib.request
    last = None
    for a in range(attempts):
        tmp = dest + '.part'
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'eiqaz-prep/1.1'})
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                    open(tmp, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception as e:                     # noqa: BLE001
            last = e
            try:
                os.remove(tmp)
            except OSError:
                pass
            if a < attempts - 1:
                time.sleep(min(30.0, 2.0 ** a) + (a % 2))
    raise last


def hf_row_size():
    """أحجام ملفات wavs من HF (ترقيم صفحات Link header)."""
    import urllib.request
    all_files, url = [], f'{HF_TREE}?recursive=true'
    while url:
        req = urllib.request.Request(url, headers={'User-Agent':
                                                    'eiqaz-prep/1.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
            link = r.headers.get('Link', '')
        if not data:
            break
        all_files.extend(data)
        if 'rel="next"' in link:
            url = link.split('<')[1].split('>')[0]
        else:
            url = None
    return {os.path.basename(f['path']): f.get('size', 0)
            for f in all_files
            if f['type'] == 'file' and f['path'].endswith('.wav')}


# ---------------------------------------------------------------------------
# PHASE: EGY
# ---------------------------------------------------------------------------
def phase_egy():
    import pandas as pd
    import concurrent.futures as cf
    seed_from_prev()
    data, eqz = setup_paths()
    tash, ex, eqz_root = load_egy_inputs()
    log(f'[egy] tashkeel corpus: {len(tash)} | unit map: {len(ex)} rows')

    # الوحدات ذات التشكيل السليم فقط (قاعدة الإدراج §م — status==ok)
    ex = ex[ex.utt.isin(set(tash.keys()))].copy()
    ex['text'] = ex.utt.map(lambda u: tash[u][0])
    spk_of = dict(zip(ex.hf_speaker.unique(),
                      [SPEAKER_MAP_V1['egy_' + s]
                       for s in ex.hf_speaker.unique()]))
    ex['spk'] = ex.hf_speaker.map(spk_of)
    log(f'[egy] units with ok tashkeel: {len(ex)} '
        f'({ex.split.value_counts().to_dict()})')

    done = {os.path.basename(f)[:-3] for f in
            glob.glob(os.path.join(FEAT_DIR, '*.pt'))}
    done_rows = set()
    # استئناف داخل الجلسة: صف يعتبر منتهيًا إذا كانت كل وحداته موجودة
    for src, grp in ex.groupby('src_row'):
        if all(u in done for u in grp.utt):
            done_rows.add(src)
    todo = ex[~ex.src_row.isin(done_rows)]
    log(f'[egy] rows todo: {todo.src_row.nunique()} '
        f'(done rows: {len(done_rows)})')

    sizes = {}
    for attempt in range(3):
        try:
            sizes = hf_row_size()
            break
        except Exception as e:                     # noqa: BLE001
            log(f'[egy] hf_row_size attempt {attempt + 1} failed: {e}')
            time.sleep(10)
    log(f'[egy] HF tree: {len(sizes)} wavs')

    # تجميع مسبق حسب الصف (كفاءة: لا مسح عمود كامل داخل كل دفعة)
    by_src = {}
    for r in todo.itertuples():
        by_src.setdefault(r.src_row, []).append(
            (r.utt, r.text, r.split, int(r.spk), int(r.i0), int(r.i1)))

    # ترتيب الدفعات: أكبر أولًا لملء الدفعات بكفاءة
    rows = (todo.groupby('src_row')
            .agg(n_units=('utt', 'count'))
            .reset_index())
    rows['bytes'] = rows.src_row.map(lambda s: sizes.get(s + '.wav', 0))
    rows = rows.sort_values('bytes', ascending=False)

    import multiprocessing as mp
    n_proc = max(1, (os.cpu_count() or 4) - 1)
    log(f'[egy] pool workers: {n_proc} (spawn)')

    t_dl = t_proc = 0.0
    stats = {'rows_ok': 0, 'rows_err': 0, 'units': 0, 'drops': {},
             'rows_dl_fail': 0}
    all_units = []
    dead_egy = _load_dead('egy')               # حجر مستأنف عبر الجلسات
    batch, batch_bytes = [], 0
    t_start = time.time()

    def _dl_one(src):
        dest = os.path.join(TMP, 'nile', src + '.wav')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            return dest
        try:
            hf_fetch(HF_RESOLVE + f'wavs/{src}.wav', dest)
            return dest
        except Exception as e:                     # noqa: BLE001
            log(f'[egy] download failed (row skipped — resumable): {src}'
                f' — {type(e).__name__}: {e}')
            return None

    def flush(batch):
        nonlocal t_dl, t_proc, stats, all_units
        if not batch:
            return
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=DL_WORKERS) as ex_:
            paths = list(ex_.map(_dl_one, batch))
        t_dl += time.time() - t0
        stats['rows_dl_fail'] += sum(1 for p in paths if p is None)
        pairs = [(p, src) for p, src in zip(paths, batch)
                 if p is not None]
        t1 = time.time()
        tasks = [(dest, by_src[src]) for dest, src in pairs]
        with mp.get_context('spawn').Pool(
                n_proc, initializer=_worker_init,
                initargs=(data, eqz_root)) as pool:
            for res in pool.imap_unordered(egy_process_row, tasks):
                if res['status'] == 'ok':
                    stats['rows_ok'] += 1
                    for u in res['units']:
                        dead_egy.pop(u['utt'], None)   # نجحت → تُرفع من الحجر
                else:
                    stats['rows_err'] += 1
                    log(f"[egy] row error: {res.get('err')}")
                stats['units'] += len(res['units'])
                all_units.extend(res['units'])
                for utt_d, why in res['drops']:
                    stats['drops'][why] = stats['drops'].get(why, 0) + 1
                    dead_egy[utt_d] = why             # حجر بالسبب (v1.2)
        t_proc += time.time() - t1
        for dest, _src in pairs:
            try:
                os.remove(dest)
            except OSError:
                pass
        el_s = time.time() - t_start
        log(f"[egy] progress: rows_ok={stats['rows_ok']} "
            f"units={stats['units']} drops={stats['drops']} "
            f"dl={t_dl/60:.1f}m proc={t_proc/60:.1f}m "
            f"elapsed={el_s/60:.1f}m")
        # حفظ تراكمي للفهرس بعد كل دفعة (استئناف آمن)
        _save_egy_units(all_units)
        _save_dead('egy', dead_egy)
        if (t_dl + t_proc) > EGY_PROCESS_CAP:
            log('[egy] TIME CAP — stopping (resumable)')
            return False
        if t_dl > DOWNLOAD_CAP:
            log('[egy] DOWNLOAD CAP — stopping (resumable)')
            return False

    for r in rows.itertuples():
        if (t_dl + t_proc) > EGY_PROCESS_CAP or t_dl > DOWNLOAD_CAP:
            log('[egy] cap reached before scheduling more rows')
            break
        sz = r.bytes
        if batch and batch_bytes + sz > BATCH_BYTES:
            flush(batch)
            batch, batch_bytes = [], 0
        batch.append(r.src_row)
        batch_bytes += sz
    flush(batch)
    _save_egy_units(all_units)
    _save_dead('egy', dead_egy)
    log(f"[egy] DONE rows_ok={stats['rows_ok']} err={stats['rows_err']} "
        f"units={stats['units']} drops={stats['drops']} "
        f"dead={len(dead_egy)}")
    return True


def _save_egy_units(units):
    """دمج تراكمي — لا كتابة فوق وحدات الجلسات السابقة (استئناف آمن)."""
    p = os.path.join(FEAT_DIR, '_egy_units.json')
    prev = {}
    if os.path.exists(p):
        try:
            for u in json.load(open(p, encoding='utf-8')):
                prev[u['utt']] = u
        except Exception:                      # noqa: BLE001
            prev = {}
    for u in units:
        prev[u['utt']] = u
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(list(prev.values()), f, ensure_ascii=False)
    os.replace(tmp, p)


def _load_egy_units():
    p = os.path.join(FEAT_DIR, '_egy_units.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    return []


# ---------------------------------------------------------------------------
# قوائم الحجر (v1.2): صفوف مستسلمة نهائيًا — تُستثنى من remaining بسقف
# ---------------------------------------------------------------------------
def _dead_path(which):
    return os.path.join(FEAT_DIR, f'_dead_{which}.json')


def _load_dead(which):
    p = _dead_path(which)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding='utf-8'))
        except Exception:                      # noqa: BLE001
            return {}
    return {}


def _save_dead(which, dead):
    p = _dead_path(which)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(dead, f, ensure_ascii=False)
    os.replace(tmp, p)


def _reason_counts(dead):
    """عد أسبائ الحجر لكل سبب (تقرير شفاف)."""
    c = {}
    for why in dead.values():
        c[why] = c.get(why, 0) + 1
    return c


def _msa_done_map():
    """أسماء وحدات MSA الجاهزة على القرص **بلا بادئة msa_** — تطابق
    عمود utt في filelist.csv (v1.2: القديم كان يقارن بالبادئة فلا
    تطابق أبدًا)."""
    pre = len('msa_')
    return {os.path.basename(f)[:-3][pre:]
            for f in glob.glob(os.path.join(FEAT_DIR, 'msa_*.pt'))}


def seed_from_prev():
    """استئناف عبر الجلسات (v1.1): نسخ ملامح مخرجات النسخة السابقة
    للنواة نفسها (self-attach في kernel_sources) إلى WORK/features —
    idempotent: الموجود لا يُنسخ مجددًا."""
    import shutil
    cands = sorted(glob.glob(f'{INPUT}/**/features/_egy_units.json',
                             recursive=True))
    if not cands:
        log('[seed] no previous kernel output — fresh start')
        return
    src = os.path.dirname(os.path.abspath(cands[0]))
    if src == os.path.abspath(FEAT_DIR):
        return
    files = glob.glob(os.path.join(src, '*.pt')) + [
        os.path.join(src, '_egy_units.json'),
        os.path.join(src, '_msa_units.json')]
    # قوائم الحجر عبر الجلسات (v1.2)
    files += [os.path.join(src, f'_dead_{w}.json') for w in ('egy', 'msa')
              if os.path.exists(os.path.join(src, f'_dead_{w}.json'))]
    n = 0
    for f in files:
        dest = os.path.join(FEAT_DIR, os.path.basename(f))
        if os.path.exists(dest):
            continue
        shutil.copy2(f, dest)
        n += 1
    log(f'[seed] resumed {n} files from previous output: {src}')


def _save_msa_units(units):
    """دمج تراكمي (v1.1 — نفس منهج _egy_units): الكتابة الفوقية كانت
    ستحذف وحدات الجلسات السابقة من الفهرس عند الاستئناف."""
    p = os.path.join(FEAT_DIR, '_msa_units.json')
    prev = {}
    if os.path.exists(p):
        try:
            for u in json.load(open(p, encoding='utf-8')):
                prev[u['utt']] = u
        except Exception:                          # noqa: BLE001
            prev = {}
    for u in units:
        prev[u['utt']] = u
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(list(prev.values()), f, ensure_ascii=False)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# PHASE: MSA
# ---------------------------------------------------------------------------
def phase_msa():
    import pandas as pd
    seed_from_prev()
    data, eqz = setup_paths()
    msa_root = find_root('filelist.csv', 'msa-tts-data-v1')
    df = pd.read_csv(os.path.join(msa_root, 'filelist.csv'))
    log(f'[msa] filelist rows: {len(df)}')

    done = _msa_done_map()          # v1.2: بلا بادئة — يطابق df.utt فعلاً
    df = df[~df.utt.isin(done)]
    log(f'[msa] todo: {len(df)} (done: {len(done)})')

    tasks = []
    for r in df.itertuples():
        spk = SPEAKER_MAP_V1.get('msa_' + r.speaker)
        if spk is None:
            continue
        wav_path = os.path.join(msa_root, r.wav)
        tasks.append((wav_path, r.utt, r.text, r.split, spk))

    import multiprocessing as mp
    n_proc = max(1, (os.cpu_count() or 4) - 1)
    t_start = time.time()
    stats = {'ok': 0, 'skip_tokens': 0, 'skip_frames': 0, 'error': 0}
    units = []
    dead_msa = _load_dead('msa')  # v1.2: حجر مستأنف
    with mp.get_context('spawn').Pool(
            n_proc, initializer=_worker_init,
            initargs=(data, eqz)) as pool:
        for i, res in enumerate(pool.imap_unordered(msa_process_row, tasks,
                                                    chunksize=16)):
            stats[res['status']] = stats.get(res['status'], 0) + 1
            if res['status'] == 'ok':
                units.append(res['unit'])
                # v1.2: نجاح بعد فشل سابق -> تُرفع من الحجر
                dead_msa.pop(res['unit']['utt'][len('msa_'):], None)
            else:
                # v1.2: حجر بالسبب (skip_tokens/skip_frames/error)
                dead_msa[res['utt']] = res['status']
            if (i + 1) % 1000 == 0:
                log(f'[msa] {i+1}/{len(tasks)} ({stats}) '
                    f'{(time.time()-t_start)/60:.1f}m')
                _save_dead('msa', dead_msa)
            if time.time() - t_start > MSA_PROCESS_CAP:
                log('[msa] TIME CAP — stopping (resumable)')
                pool.terminate()
                break
    _save_dead('msa', dead_msa)
    _save_msa_units(units)
    log(f'[msa] DONE ({stats}) dead={len(dead_msa)}')


# ---------------------------------------------------------------------------
# PHASE: REPORT (الفهرس الموحد + الإحصاءات)
# ---------------------------------------------------------------------------
def phase_report():
    import torch
    import pandas as pd
    seed_from_prev()
    egy = _load_egy_units()
    msa = []
    p = os.path.join(FEAT_DIR, '_msa_units.json')
    if os.path.exists(p):
        msa = json.load(open(p, encoding='utf-8'))
    # ground truth من القرص
    on_disk = {os.path.basename(f)[:-3]
               for f in glob.glob(os.path.join(FEAT_DIR, '*.pt'))}
    egy = [u for u in egy if u['utt'] in on_disk]
    msa = [u for u in msa if u['utt'] in on_disk]
    index = egy + msa

    # ---- اكتمال بحقيقة أرضية (v1.1): المتبقي فعلاً مقابل المؤهل ----
    tash_r, ex_r, _ = load_egy_inputs()
    egy_elig = ex_r[ex_r.utt.isin(set(tash_r.keys()))]
    dead_egy = _load_dead('egy')
    egy_remaining = int((~egy_elig.utt.isin(on_disk)
                         & ~egy_elig.utt.isin(set(dead_egy))).sum())
    msa_root_r = find_root('filelist.csv', 'msa-tts-data-v1')
    dfm = pd.read_csv(os.path.join(msa_root_r, 'filelist.csv'))
    mapped = dfm.speaker.map(
        lambda s: SPEAKER_MAP_V1.get('msa_' + s) is not None)
    msa_on_disk = _msa_done_map()  # v1.2: بلا بادئة — يطابق dfm.utt
    dead_msa = _load_dead('msa')
    msa_remaining = int((~dfm.utt.isin(msa_on_disk)
                         & ~dfm.utt.isin(set(dead_msa)) & mapped).sum())
    dead_ok = (len(dead_egy) <= DEAD_CAP['egy']
               and len(dead_msa) <= DEAD_CAP['msa'])
    complete = (egy_remaining == 0 and msa_remaining == 0 and dead_ok)

    pitch_sum, pitch_sq, pitch_n = 0.0, 0.0, 0
    for e in index:
        d = torch.load(os.path.join(FEAT_DIR, e['utt'] + '.pt'),
                       weights_only=True)
        v = d['pitch'][d['pitch'] > 0]
        if v.numel():
            pitch_sum += float(v.sum())
            pitch_sq += float((v ** 2).sum())
            pitch_n += int(v.numel())
    if pitch_n == 0:
        raise RuntimeError('no voiced pitch frames found')
    p_mean = pitch_sum / pitch_n
    p_std = max(1e-3, (pitch_sq / pitch_n - p_mean ** 2) ** 0.5)

    per_spk = {}
    for e in index:
        s = e['spk']
        per_spk.setdefault(s, {'clips': 0, 'hours': 0.0})
        per_spk[s]['clips'] += 1
        per_spk[s]['hours'] += e['n_frames'] * 256 / TARGET_SR / 3600

    n_q = sum(e.get('n_q', 0) for e in index)
    n_tok = sum(e['n_tokens'] for e in index)
    egy_h = sum(e['n_frames'] for e in index if e['dialect'] == 'egy') \
        * 256 / TARGET_SR / 3600
    msa_h = sum(e['n_frames'] for e in index if e['dialect'] == 'msa') \
        * 256 / TARGET_SR / 3600

    meta = {
        'project': 'eiqaz-tts-v1',
        'index': index,
        'pitch_mean': p_mean, 'pitch_std': p_std,
        'speaker_map_v1': SPEAKER_MAP_V1,
        'per_speaker': {str(k): v for k, v in sorted(per_spk.items())},
        'n_features': len(index),
        'mix_hours': {'egy': round(egy_h, 2), 'msa': round(msa_h, 2),
                      'file_ratio_egy': round(
                          len(egy) / max(1, len(index)), 4)},
        'q_tokens': {'total': n_q, 'egy': sum(e.get('n_q', 0) for e in egy),
                     'msa': sum(e.get('n_q', 0) for e in msa),
                     'pct_of_tokens': round(100 * n_q / max(1, n_tok), 3)},
        'v_tokens_egy': sum(e.get('n_v', 0) for e in egy),
        'source': 'Eiqaz v1: NileTTS(HF)+tashkeel-ai + msa-tts-data-v1',
        'complete': complete,
        'expected': {'egy_units': int(len(egy_elig)),
                     'msa_units': int(mapped.sum())},
        'remaining': {'egy': egy_remaining, 'msa': msa_remaining},
        'dead': {'egy': {'n': len(dead_egy),
                        'by_reason': _reason_counts(dead_egy)},
                 'msa': {'n': len(dead_msa),
                         'by_reason': _reason_counts(dead_msa)}},
        'dead_cap': dict(DEAD_CAP),
    }
    with open(os.path.join(FEAT_DIR, 'index.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False)
    log(f'[report] index: {len(index)} features '
        f'(egy {len(egy)}/{round(egy_h,1)}h + msa {len(msa)}/'
        f'{round(msa_h,1)}h) | pitch mean={p_mean:.1f} std={p_std:.1f}')
    log(f"[report] q tokens: {meta['q_tokens']} | v(egy): "
        f"{meta['v_tokens_egy']}")
    log(f'[report] per-speaker: ' + json.dumps(meta['per_speaker']))
    log(f"[report] PREP {'COMPLETE' if complete else 'INCOMPLETE'} — "
        f"remaining egy {egy_remaining}/{len(egy_elig)} | "
        f"msa {msa_remaining}/{int(mapped.sum())} | "
        f"dead egy={len(dead_egy)}/{DEAD_CAP['egy']} "
        f"msa={len(dead_msa)}/{DEAD_CAP['msa']}")
    with open(os.path.join(WORK, 'prep_report.json'), 'w') as f:
        json.dump({k: v for k, v in meta.items() if k != 'index'},
                  f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# PHASE: DRIVER (sanity ثم egy ثم msa ثم report)
# ---------------------------------------------------------------------------
def driver():
    log('=== Eiqaz TTS v1 PREP (independent project) ===')
    import torch
    log(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')

    # ---- sanity: صف مصري واحد كامل بمسار كامل + توكنات القاف ----
    try:
        data, eqz = setup_paths()
        tash, ex, _ = load_egy_inputs()
        import soundfile as sf
        import librosa
        import numpy as np
        row = ex.iloc[0]
        dest = os.path.join(TMP, 'sanity.wav')
        os.makedirs(TMP, exist_ok=True)
        hf_fetch(HF_RESOLVE + f"wavs/{row.src_row}.wav", dest)
        wav, sr = sf.read(dest, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        peak = float(np.abs(wav).max())
        if peak > 0.95:
            wav = wav * (0.95 / peak)
        u_wav = wav[int(row.i0):int(row.i1)]
        text = tash[row.utt][0]
        _worker_init(data, eqz)
        toks_ms, toks_egy, ids_of, _ = _get_toks()
        ids = ids_of(toks_egy(text))
        mel, pitch = _wav_to_feat(ids, u_wav)
        log(f"[sanity] utt={row.utt} sr={sr} unit={u_wav.size/TARGET_SR:.2f}s")
        log(f'[sanity] text: {text[:80]}')
        log(f'[sanity] n_ids={len(ids)} mel={tuple(mel.shape)} '
            f'pitch={tuple(pitch.shape)}')
        import eqz_tokens as ET
        n_q_corpus = 0
        for t in list(tash.values())[:500]:
            if 'q' in toks_egy(t[0]):
                n_q_corpus += 1
        log(f'[sanity] texts with q token (first 500): {n_q_corpus} '
            f'(قاف ستُدرَّب — كانت 0 في المشروع القديم)')
        os.remove(dest)
    except Exception as e:                        # noqa: BLE001
        import traceback
        log(f'SANITY FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        sys.exit(1)

    env = dict(os.environ)
    env['KAGGLE_WORK'] = WORK
    for ph in ('egy', 'msa', 'report'):
        log(f'--- phase {ph} ---')
        r = subprocess.run([sys.executable, os.path.abspath(__file__), ph],
                           env=env)
        log(f'phase {ph} exit code: {r.returncode}')
        if r.returncode != 0 and ph != 'egy':
            sys.exit(r.returncode)
    rep = os.path.join(WORK, 'prep_report.json')
    if os.path.exists(rep):
        try:
            complete = json.load(open(rep)).get('complete')
        except Exception:                          # noqa: BLE001
            complete = None
        log(f'PREP_COMPLETE={complete}')
        if not complete:
            log('!! PREP INCOMPLETE — أعد تشغيل هذه النواة لاستئناف '
                'الوحدات المتبقية قبل أي تدريب')
    log('ALL_DONE')


if __name__ == '__main__':
    if PHASE == 'egy':
        phase_egy()
    elif PHASE == 'msa':
        phase_msa()
    elif PHASE == 'report':
        phase_report()
    else:
        driver()
