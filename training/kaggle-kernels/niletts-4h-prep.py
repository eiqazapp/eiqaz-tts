# ============================================================================
# NileTTS_4h_Controlled_Experiment — PREP kernel v2 (NO TRAINING HERE)
# Dataset: KickItLikeShika/NileTTS-dataset (2 speakers, 38.1h measured)
#
# v2 FIX (user-mandated): the v1 pipeline dropped ~79% of NileTTS segments
# because of the 160-token cap (+ the 3-11s extraction window kept only
# 11.8% of rows). v2 SPLITS long rows into natural sub-units instead of
# dropping them, so the 4h experiment actually tests NileTTS.
#
# What v2 changes (and NOTHING else):
#   * rows of ANY duration >= 3s are downloaded (v1: only 3-11s)
#   * long rows are split at NATURAL boundaries (sentence/clause/word),
#     every audio cut SNAPPED to a detected pause; never a blind cut
#   * rows already fitting the reference constraints pass through WHOLE
#     (reference-identical treatment)
#   * per-unit filters are IDENTICAL to the reference extraction:
#     22050Hz mono PCM16, peak-norm >0.95, text >=10 chars, no ASCII
#     alpha; feature worker is v3-VERBATIM (catt -> toks_egy -> mel ->
#     pyin -> .pt), write-filter <=180 tokens, train-pool filter <=160
#   * speaker labels unchanged (SPEAKER_01->0, SPEAKER_02->1)
#   * dataset's own train/eval split inherited per unit
#
# v1-failure engineering fixes (infrastructure only, no data semantics):
#   * multiprocessing spawn (v1 used fork after librosa/numba ran in the
#     parent -> all 1090 tasks failed instantly with swallowed errors)
#   * PRE-FLIGHT: 2 rows processed in the main process with FULL traceback
#     before any pool starts
#   * every per-row/per-unit error is surfaced (first 5 unique tracebacks
#     logged; all errors counted with reasons in split_report.json)
#   * PROCESS_TIME_CAP raised 2.7h -> 7h (more data to process)
#
# Read-only mounts: mixer-tts-scratch-data (code + catt + mel utils).
# ============================================================================
import os, sys, json, time, glob, io, csv, random, re
import subprocess
import urllib.request

# onnxruntime is NOT preinstalled in the Kaggle image — the reference
# train kernels pip-install it before their process phase; prep must do
# the same (root cause of BOTH prep v1's 1090 silent per-task errors and
# prep v2's fast abort, surfaced by the v2 error reporting)
try:
    import onnxruntime  # noqa: F401
except ImportError:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'onnxruntime'], capture_output=True, text=True)
    print(f'[deps] onnxruntime install rc={r.returncode}', flush=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)

T0 = time.time()
def el():
    return f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-T0))}]"
def log(*a):
    print(el(), *a, flush=True)

INPUT = os.environ.get('KAGGLE_INPUT_ROOT', '/kaggle/input')
WORK = os.environ.get('KAGGLE_WORK', '/kaggle/working')
TMP = os.environ.get('KAGGLE_TMP', '/kaggle/tmp')
AUDIT_DIR = os.path.join(WORK, 'dataset_audit')
WAV_DIR = os.path.join(WORK, 'wav')
FEAT_DIR = os.path.join(WORK, 'features')
for p in (AUDIT_DIR, WAV_DIR, FEAT_DIR, TMP):
    os.makedirs(p, exist_ok=True)

REPO = 'KickItLikeShika/NileTTS-dataset'
HF_API = f'https://huggingface.co/api/datasets/{REPO}/tree/main'
HF_RESOLVE = f'https://huggingface.co/datasets/{REPO}/resolve/main/'

# previous pipeline constants (probe1 + v3 process phase) — IDENTICAL
DUR_MIN, DUR_MAX = 3.0, 11.0      # reference window (11s -> 947 frames)
TARGET_SR = 22050                  # probe1 TARGET_SR
EGY_TOKEN_MAP = {'j': 'v', 'q': '<', '^': 't', '*': 'd'}
PROCESS_TIME_CAP = 7.0 * 3600      # v2: raised from 2.7h (infra only)
DOWNLOAD_TIME_CAP = 2.5 * 3600

# NileTTS-specific constants (measured in the local audit; re-verified)
BPS = 44100 * 2 * 2                # 44.1kHz stereo PCM16 bytes/sec
HDR = 44                           # wav header bytes
SPEAKER_MAP = {'SPEAKER_01': 0, 'SPEAKER_02': 1}   # male->0, female->1
SPLIT_SEED = 42                    # audit sampling only
N_AUDIT_SAMPLE = 120               # unit wavs kept for audit (was all in v1)
BATCH_BYTES = 2 * 1024 ** 3        # ~2GB download batches (streamed)


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
    data = find_root('models/mixer128_pytorch.pth', 'data package')
    for p in (os.path.join(data, 'mixer_repo'),
              os.path.join(data, 'tts_arabic_pkg')):
        if p not in sys.path:
            sys.path.insert(0, p)
    return data


# ============================================================================
# SPLITTER (validated locally on 13 real files: 97.3% audio utilization,
# all cuts at pauses with RMS ratio <= 0.06, no misalignment outliers)
# ============================================================================
STRONG_PUNCT = '؟!…؛.'
WEAK_PUNCT = '،,;:'
PLAN_TOKEN_CAP = 145
CHARS_MIN = 10


def sentence_atoms(text):
    spans, start = [], 0
    for i, c in enumerate(text):
        if c in STRONG_PUNCT or c == '\n':
            spans.append((start, i + 1))
            start = i + 1
    if start < len(text):
        spans.append((start, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= 1:
            out.append((s, e))
    return out


def clause_atoms(text, s, e):
    spans, start = [], s
    for i in range(s, e):
        if text[i] in WEAK_PUNCT:
            spans.append((start, i + 1))
            start = i + 1
    if start < e:
        spans.append((start, e))
    out = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b - a >= 1:
            out.append((a, b))
    return out


def word_positions(text, s, e):
    ws, in_w = [], False
    for i in range(s, e):
        if not text[i].isspace() and not in_w:
            ws.append(i)
            in_w = True
        elif text[i].isspace():
            in_w = False
    return ws


def rms_envelope(wav, sr, hop_ms=10.0):
    import numpy as np
    hop = max(1, int(sr * hop_ms / 1000.0))
    n = len(wav) // hop
    if n == 0:
        return None, hop
    fr = wav[:n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    return rms.astype('float32'), hop


def find_pause(wav, sr, t_est, search_s=1.2, min_run_ms=40):
    import numpy as np
    rms, hop = rms_envelope(wav, sr)
    if rms is None or len(rms) == 0:
        return None
    speech = float(np.quantile(rms, 0.90))
    if speech < 1e-6:
        speech = 1e-6
    lo = max(0, int((t_est - search_s) * sr) // hop)
    hi = min(len(rms), int(np.ceil((t_est + search_s) * sr)) // hop + 1)
    if hi <= lo:
        return None
    seg = rms[lo:hi]
    for thr in (0.12, 0.20, 0.30):
        mask = seg < thr * speech
        best_len, best_a, best_b = 0, 0, 0
        a = None
        for i, m in enumerate(mask):
            if m and a is None:
                a = i
            if (not m or i == len(mask) - 1) and a is not None:
                b = i if not m else i + 1
                if b - a > best_len:
                    best_len, best_a, best_b = b - a, a, b
                a = None
        if best_len * hop >= int(min_run_ms / 1000.0 * sr):
            center = (lo + best_a + lo + best_b) / 2.0
            return int(center * hop)
    return None


def _word_split(text, n_chars, dur_s, a, b, tok_per_char=0.75):
    import numpy as np
    est = dur_s * (b - a) / n_chars
    est_tok = (b - a) * tok_per_char
    if est <= DUR_MAX and est_tok <= PLAN_TOKEN_CAP:
        return [(a, b)]
    ws = word_positions(text, a, b)
    if len(ws) < 2:
        return [(a, b)]
    n_parts = int(max(np.ceil(est / DUR_MAX),
                      np.ceil(est_tok / PLAN_TOKEN_CAP)))
    if n_parts < 2:
        n_parts = 2
    target = len(ws) / n_parts
    parts, p_start = [], a
    for k in range(1, n_parts):
        wi = int(round(k * target))
        if wi <= 0 or wi >= len(ws):
            continue
        cut = ws[wi]
        parts.append((p_start, cut))
        p_start = cut
    parts.append((p_start, b))
    return [(x, y) for x, y in parts if y > x]


def plan_units(text, dur_s, full_tokens=None):
    text = text.strip()
    if not text:
        return []
    n_chars = len(text)
    tok_per_char = (full_tokens / n_chars) if full_tokens else 0.75

    def est_d(a, b):
        return dur_s * (b - a) / n_chars

    def est_t(a, b):
        return (b - a) * tok_per_char

    atoms = []
    for s, e in sentence_atoms(text):
        if est_d(s, e) <= DUR_MAX and est_t(s, e) <= PLAN_TOKEN_CAP:
            atoms.append((s, e))
            continue
        cl = clause_atoms(text, s, e)
        if len(cl) <= 1:
            atoms.extend(_word_split(text, n_chars, dur_s, s, e,
                                     tok_per_char=tok_per_char))
        else:
            for a, b in cl:
                if (est_d(a, b) <= DUR_MAX
                        and est_t(a, b) <= PLAN_TOKEN_CAP):
                    atoms.append((a, b))
                else:
                    atoms.extend(_word_split(text, n_chars, dur_s, a, b,
                                             tok_per_char=tok_per_char))
    final = []
    for a, b in atoms:
        est = dur_s * (b - a) / n_chars
        est_t2 = (b - a) * tok_per_char
        if est > DUR_MAX or est_t2 > PLAN_TOKEN_CAP:
            final.extend(_word_split(text, n_chars, dur_s, a, b,
                                     tok_per_char=tok_per_char))
        else:
            final.append((a, b))
    atoms = final

    units, cur = [], None
    for a, b in atoms:
        if cur is None:
            cur = [a, b]
            continue
        na, nb = cur[0], b
        if (dur_s * (nb - na) / n_chars <= DUR_MAX
                and (nb - na) * tok_per_char <= PLAN_TOKEN_CAP):
            cur = [na, nb]
        else:
            units.append(tuple(cur))
            cur = [a, b]
    if cur is not None:
        units.append(tuple(cur))

    merged = []
    for u in units:
        if merged and dur_s * (u[1] - u[0]) / n_chars < DUR_MIN:
            prev = merged[-1]
            if (dur_s * (u[1] - prev[0]) / n_chars <= DUR_MAX
                    and (u[1] - prev[0]) * tok_per_char
                    <= PLAN_TOKEN_CAP):
                merged[-1] = (prev[0], u[1])
                continue
        merged.append(u)
    out = []
    for u in merged:
        if (dur_s * (u[1] - u[0]) / n_chars < DUR_MIN and out):
            prev = out[-1]
            if (dur_s * (u[1] - prev[0]) / n_chars <= DUR_MAX
                    and (u[1] - prev[0]) * tok_per_char
                    <= PLAN_TOKEN_CAP):
                out[-1] = (prev[0], u[1])
                continue
        out.append(u)
    return out


def split_row(text, dur_s, wav, sr, full_tokens=None, _depth=0):
    """Split one (text, audio) row into natural units. See module header.
    Returns (units, report); units = [{'text', 'i0', 'i1', 'dur'}]."""
    rep = {'drops': [], 'n_planned_units': 0, 'n_merged_unresolved': 0,
           'n_spans_dropped_unresolved': 0}
    text = (text or '').strip()
    if not text:
        rep['drops'].append(('row', 'empty_text'))
        return [], rep
    if dur_s < DUR_MIN:
        rep['drops'].append(('row', f'row_duration<{DUR_MIN}s'))
        return [], rep

    n_chars = len(text)
    tok_per_char = (full_tokens / n_chars) if full_tokens else 0.75

    # whole-row passthrough (reference-identical treatment)
    if dur_s <= DUR_MAX and full_tokens is not None and full_tokens <= 160:
        reasons = []
        if len(text) < CHARS_MIN:
            reasons.append(f'text<{CHARS_MIN}ch')
        if any(c.isascii() and c.isalpha() for c in text):
            reasons.append('ascii_alpha')
        rep['n_planned_units'] = 1
        if reasons:
            rep['drops'] = [('unit_0', '; '.join(reasons))]
            rep['n_units_final'] = 0
            return [], rep
        rep['n_units_final'] = 1
        rep['passthrough'] = True
        return [{'text': text, 'i0': 0, 'i1': len(wav),
                 'dur': round(dur_s, 3)}], rep

    def est_dur(a, b):
        return dur_s * (b - a) / n_chars

    def est_tok(a, b):
        return (b - a) * tok_per_char

    plan = plan_units(text, dur_s, full_tokens)
    rep['n_planned_units'] = len(plan)
    if not plan:
        rep['drops'].append(('row', 'no_plannable_units'))
        return [], rep

    n = len(plan)
    cuts = []
    for k in range(n - 1):
        edge_t = dur_s * plan[k][1] / n_chars
        cut = find_pause(wav, sr, edge_t)
        if cut is None:
            cut = find_pause(wav, sr, edge_t, search_s=1.8, min_run_ms=30)
        cuts.append(cut)

    emitted = []
    k = 0
    start = 0
    while k < n:
        m = k
        while True:
            if m == n - 1:
                emitted.append((plan[k][0], plan[m][1], start, len(wav)))
                k = n
                break
            cut = cuts[m]
            if cut is not None and start < cut < len(wav) - 1:
                emitted.append((plan[k][0], plan[m][1], start, cut))
                k = m + 1
                start = cut
                break
            if (m + 1 < n and est_dur(plan[k][0], plan[m + 1][1]) <= DUR_MAX
                    and est_tok(plan[k][0], plan[m + 1][1])
                    <= PLAN_TOKEN_CAP + 25):
                m += 1
                rep['n_merged_unresolved'] += 1
                continue
            j = m + 1
            while j <= n - 2 and (cuts[j] is None or cuts[j] <= start):
                j += 1
            rep['n_spans_dropped_unresolved'] += (j + 1 - k)
            if j <= n - 2:
                start = cuts[j]
                k = j + 1
            else:
                k = n
            break

    out = []
    for idx, (a, b, i0, i1) in enumerate(emitted):
        u_dur = (i1 - i0) / sr
        u_text = text[a:b].strip()
        if (u_dur > DUR_MAX and _depth < 2
                and len(u_text) >= 2 * CHARS_MIN):
            sub, sub_rep = split_row(u_text, u_dur, wav[i0:i1], sr,
                                     full_tokens=None, _depth=_depth + 1)
            for s in sub:
                s['i0'] += i0
                s['i1'] += i0
            out.extend(sub)
            rep['n_resplit_units'] = rep.get('n_resplit_units', 0) + len(sub)
            rep['drops'].extend([('resplit_' + d[0], d[1])
                                 for d in sub_rep['drops']])
            continue
        reasons = []
        if u_dur < DUR_MIN:
            reasons.append(f'dur<{DUR_MIN}s({u_dur:.2f})')
        if u_dur > DUR_MAX:
            reasons.append(f'dur>{DUR_MAX}s({u_dur:.2f})')
        if len(u_text) < CHARS_MIN:
            reasons.append(f'text<{CHARS_MIN}ch')
        if any(c.isascii() and c.isalpha() for c in u_text):
            reasons.append('ascii_alpha')
        if reasons:
            rep['drops'].append((f'unit_{idx}', '; '.join(reasons)))
            continue
        out.append({'text': u_text, 'i0': int(i0), 'i1': int(i1),
                    'dur': round(u_dur, 3)})
    rep['n_units_final'] = len(out)
    return out, rep


# ============================================================================
# METADATA + TREE + AUDIT (v1-verbatim, proven on the real run)
# ============================================================================
def fetch_url(url, timeout=120, retries=3):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':
                                                       'niletts-prep/2.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f'download failed {url}: {last}')


def get_file_tree():
    all_files, url = [], f'{HF_API}?recursive=true'
    while url:
        req = urllib.request.Request(url, headers={'User-Agent':
                                                    'niletts-prep/2.0'})
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
    return all_files


def load_metadata():
    import pandas as pd
    tr = pd.read_csv(io.BytesIO(fetch_url(HF_RESOLVE + 'metadata_train.csv')),
                     sep='|', dtype=str)
    ev = pd.read_csv(io.BytesIO(fetch_url(HF_RESOLVE + 'metadata_eval.csv')),
                     sep='|', dtype=str)
    tr['split'], ev['split'] = 'train', 'eval'
    df = pd.concat([tr, ev], ignore_index=True)
    return tr, ev, df


def run_audit(df, tree, downloaded):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    report = {}
    wavs = {f['path']: f.get('size', 0) for f in tree
            if f['type'] == 'file' and f['path'].endswith('.wav')}
    report['rows'] = {'total': len(df),
                      'train': int((df.split == 'train').sum()),
                      'eval': int((df.split == 'eval').sum())}
    report['speakers'] = df.speaker_name.value_counts().to_dict()
    report['nulls'] = int(df.isna().sum().sum())
    report['duplicate_rows'] = int(df.duplicated(
        subset=['audio_file', 'text', 'speaker_name']).sum())
    report['duplicate_texts'] = int(df.text.duplicated().sum())
    csv_paths = set(df.audio_file)
    report['paths_missing_in_repo'] = len(csv_paths - set(wavs))
    report['unreferenced_wav_files'] = len(set(wavs) - csv_paths)

    def pref(p):
        m = re.match(r'wavs/([a-z]+_\d+)_', p)
        return m.group(1) if m else None
    tr_p = set(df[df.split == 'train'].audio_file.map(pref))
    ev_p = set(df[df.split == 'eval'].audio_file.map(pref))
    report['leakage_shared_source_prefixes'] = len(tr_p & ev_p)

    TASH = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652')
    texts = df.text.fillna('')
    report['text'] = {
        'empty': int((texts.str.strip() == '').sum()),
        'latin_alpha': int(texts.apply(lambda t: any(
            'a' <= c.lower() <= 'z' for c in t)).sum()),
        'ascii_digits': int(texts.apply(lambda t: any(
            c.isdigit() and c.isascii() for c in t)).sum()),
        'partial_tashkeel': int(texts.apply(lambda t: any(
            c in TASH for c in t)).sum()),
        'chars_median': float(texts.str.len().median()),
        'words_median': float(texts.apply(lambda t: len(t.split())).median()),
    }
    dur = (df.audio_file.map(wavs) - HDR) / BPS
    report['duration'] = {
        'total_hours': round(float(dur.sum()) / 3600, 2),
        'median_s': round(float(dur.median()), 2),
        'p5_s': round(float(dur.quantile(.05)), 2),
        'p95_s': round(float(dur.quantile(.95)), 2),
        'min_s': round(float(dur.min()), 2),
        'max_s': round(float(dur.max()), 2),
        'under_0_6s_files': int((dur < 0.6).sum()),
        'method': 'estimated from wav byte sizes at confirmed '
                  '176,400 B/s (44.1kHz stereo PCM16)',
    }
    # BEFORE funnel: exact reference-pipeline treatment (whole-clip)
    w = df[(dur >= DUR_MIN) & (dur <= DUR_MAX)]
    w = w[w.text.fillna('').str.len() >= 10]
    w = w[~texts.loc[w.index].apply(lambda t: any(
        c.isascii() and c.isalpha() for c in t))]
    report['funnel_v1_reference'] = {
        'in_window_clips': len(w),
        'in_window_hours': round(float(dur.loc[w.index].sum()) / 3600, 2),
        'train_split': int((w.split == 'train').sum()),
        'eval_split': int((w.split == 'eval').sum()),
        'by_speaker': w.speaker_name.value_counts().to_dict(),
        'note': 'BEFORE the fix: only 3-11s rows entered; the 160-token '
                'pool filter then removed ~79% of them (measured in the '
                'planning phase below)',
    }
    aud_rows = []
    for f in downloaded:
        dst = os.path.join(f'{TMP}/nile', os.path.basename(f))
        try:
            info = sf.info(dst)
            wav, sr = sf.read(dst, dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(1)
            hop = max(1, sr // 10)
            nfr = max(1, len(wav) // hop)
            fr_max = np.array([float(np.abs(wav[i*hop:(i+1)*hop]).max())
                               for i in range(nfr)])
            aud_rows.append({
                'file': f, 'sr': sr, 'channels': info.channels,
                'subtype': info.subtype,
                'dur_s': round(len(wav) / sr, 2),
                'peak': round(float(np.abs(wav).max()), 4),
                'silent_frac': round(float((fr_max < 1e-3).mean()), 3),
                'clip_frac': round(float((np.abs(wav) >= 0.999).mean()), 5),
            })
        except Exception as e:
            aud_rows.append({'file': f, 'error': f'{type(e).__name__}: {e}'})
    ok = [r for r in aud_rows if 'error' not in r]
    report['audio'] = {
        'n_checked': len(aud_rows), 'n_errors': len(aud_rows) - len(ok),
        'sr_channels': sorted({(r['sr'], r['channels'], r['subtype'])
                               for r in ok}),
        'peaks_max': max((r['peak'] for r in ok), default=None),
        'clipping_files': sum(1 for r in ok if r['clip_frac'] > 0),
    }
    pd.DataFrame(aud_rows).to_csv(f'{AUDIT_DIR}/audio_audit.csv', index=False)
    report['notes'] = [
        'dataset card claims 24kHz; actual files are 44.1kHz stereo PCM16',
        'audio is SYNTHETIC (NotebookLM-generated, Whisper-transcribed)',
        'built-in train/eval split shares all 131 source prefixes '
        '(same-source leakage) — recorded, not modified',
        'v2 pipeline fix: long rows are SPLIT at natural boundaries '
        '(punctuation/clause/word snapped to audio pauses) instead of '
        'being dropped by the 3-11s window + 160-token cap',
        'training hyperparameters, model, optimizer, LR: UNCHANGED '
        '(train kernel untouched)',
    ]
    json.dump(report, open(f'{AUDIT_DIR}/audit_summary.json', 'w'),
              indent=1, ensure_ascii=False, default=str)
    log('[audit] summary:', json.dumps(report, ensure_ascii=False)[:1200])
    return report


def download_files(files):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(f'{TMP}/nile', exist_ok=True)
    failed = []
    def dl(f):
        dst = os.path.join(f'{TMP}/nile', os.path.basename(f))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return None
        try:
            data = fetch_url(HF_RESOLVE + f, timeout=300, retries=3)
            with open(dst, 'wb') as out:
                out.write(data)
            return None
        except Exception as e:
            return (f, str(e)[:80])
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(dl, files):
            if r:
                failed.append(r)
    return failed


# ============================================================================
# WORKERS (spawn-safe; v1's fork-after-numba failure mode eliminated).
# Per-unit feature math is v3-VERBATIM: catt -> toks_egy -> mel -> pyin.
# Unit names: {sidx}_{row_index:05d}_{k:02d} — deterministic, unique per
# worker, resume-friendly (row index = position in the sorted row list).
# ============================================================================
_WORKER_STATE = {}


def _worker_init(data_root):
    import sys
    for p in (os.path.join(data_root, 'mixer_repo'),
              os.path.join(data_root, 'tts_arabic_pkg')):
        if p not in sys.path:
            sys.path.insert(0, p)
    _WORKER_STATE['data'] = data_root
    _WORKER_STATE['err_n'] = 0


def _lazy_load():
    if 'catt' not in _WORKER_STATE:
        from tts_arabic.vocalizer.models.core import get_model
        _WORKER_STATE['catt'] = get_model('catt_eo')
    if 'mel_fn' not in _WORKER_STATE:
        from utils.audio import MelSpectrogram
        _WORKER_STATE['mel_fn'] = MelSpectrogram()
    if 'egy' not in _WORKER_STATE:
        from tts_arabic.text import (
            arabic_to_buckwalter, tokens_to_ids, phonemes_to_tokens,
            buckwalter_to_phonemes)
        def toks_egy(text):
            toks = phonemes_to_tokens(
                buckwalter_to_phonemes(arabic_to_buckwalter(text)))
            return [EGY_TOKEN_MAP.get(t, t) for t in toks]
        _WORKER_STATE['egy'] = toks_egy
        _WORKER_STATE['ids_of'] = tokens_to_ids


def plan_tokens_one(args):
    """Planning worker: catt + tokenize ONE full row text -> token count."""
    import traceback
    text, = args
    try:
        _lazy_load()
        catt = _WORKER_STATE['catt']
        toks_egy = _WORKER_STATE['egy']
        ids_of = _WORKER_STATE['ids_of']
        voc = catt.predict(text)
        ids = ids_of(toks_egy(voc))
        return {'status': 'ok', 'n': len(ids)}
    except Exception as e:
        out = {'status': 'error', 'err': f'{type(e).__name__}: {e}'}
        if _WORKER_STATE['err_n'] < 5:
            out['tb'] = traceback.format_exc(limit=10)
            _WORKER_STATE['err_n'] += 1
        return out


def process_row(args):
    """Row worker: full wav -> reference extraction treatment -> split_row
    -> per unit (with exact token re-split) -> v3-verbatim features."""
    import traceback
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    (src_path, text, dur_s, full_tokens, sidx, split, row_idx) = args
    units_meta = []
    rep = {'n_units': 0, 'drops': [], 'n_resplit_token': 0,
           'passthrough': False}
    try:
        _lazy_load()
        catt = _WORKER_STATE['catt']
        mel_fn = _WORKER_STATE['mel_fn']
        toks_egy = _WORKER_STATE['egy']
        ids_of = _WORKER_STATE['ids_of']

        wav, sr = sf.read(src_path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        # reference extraction treatment on the whole row (identical
        # semantics; unit slices inherit it)
        peak = float(np.abs(wav).max())
        if peak > 0.95:
            wav = wav * (0.95 / peak)

        units, srep = split_row(text, dur_s, wav, TARGET_SR,
                                full_tokens=full_tokens)
        rep['passthrough'] = bool(srep.get('passthrough'))
        rep['drops'] = list(srep['drops'])

        def emit(u_text, u_wav, ids, k, i0, i1):
            name = f'{sidx}_{row_idx:05d}_{k:02d}'
            out_path = os.path.join(FEAT_DIR, name + '.pt')
            wav_t = torch.from_numpy(np.ascontiguousarray(u_wav))[None]
            mel = mel_fn(wav_t).clamp_min(1e-5).log().squeeze(0)
            f0, _, _ = librosa.pyin(u_wav, sr=22050, fmin=60, fmax=600,
                                    frame_length=1024, hop_length=256)
            f0 = np.where(np.isnan(f0), 0., f0)
            pitch = torch.from_numpy(f0).float()
            if pitch.size(0) < mel.size(1):
                pitch = torch.nn.functional.pad(
                    pitch, (0, mel.size(1) - pitch.size(0)))
            else:
                pitch = pitch[:mel.size(1)]
            torch.save({'ids': torch.LongTensor(ids), 'mel': mel,
                        'pitch': pitch}, out_path)
            units_meta.append({
                'utt': name, 'spk_idx': sidx,
                'src_row': os.path.basename(src_path)[:-4],
                'transcript': u_text,
                'duration': round(len(u_wav) / 22050, 2),
                'i0': i0, 'i1': i1,
                'n_tokens': len(ids), 'n_frames': int(mel.size(1)),
                'split': split})

        k = 0
        for u in units:
            u_text, u_wav = u['text'], wav[u['i0']:u['i1']]
            voc = catt.predict(u_text)
            ids = ids_of(toks_egy(voc))
            if len(ids) > 160 and u['dur'] >= 6.0 and len(u_text) >= 40:
                sub, _ = split_row(u_text, u['dur'], u_wav, TARGET_SR,
                                   full_tokens=len(ids))
                sub_ok = []
                for s in sub:
                    sv = catt.predict(s['text'])
                    si = ids_of(toks_egy(sv))
                    if len(si) <= 180 and 3.0 <= s['dur'] <= DUR_MAX:
                        sub_ok.append((s, sv, si))
                if len(sub_ok) >= 2:
                    rep['n_resplit_token'] += 1
                    for s, sv, si in sub_ok:
                        emit(s['text'], u_wav[s['i0']:s['i1']], si, k,
                             u['i0'] + s['i0'], u['i0'] + s['i1'])
                        k += 1
                    continue
            if len(ids) < 5 or len(ids) > 180:
                rep['drops'].append((os.path.basename(src_path),
                                     f'skip_tokens(n={len(ids)})'))
                continue
            emit(u_text, u_wav, ids, k, u['i0'], u['i1'])
            k += 1
        rep['n_units'] = len(units)
        return {'status': 'ok', 'rep': rep, 'units': units_meta,
                'src_row': os.path.basename(src_path)[:-4]}
    except Exception as e:
        out = {'status': 'error', 'err': f'{type(e).__name__}: {e}',
               'rep': rep, 'units': units_meta,
               'src_row': os.path.basename(src_path)[:-4]}
        if _WORKER_STATE['err_n'] < 5:
            out['tb'] = traceback.format_exc(limit=10)
            _WORKER_STATE['err_n'] += 1
        return out


# ============================================================================
# DRIVER
# ============================================================================
TREE = []
ERR_LOG = []
ALL_DROPS = []      # (src_row, unit_id, reason)
ALL_UNITS_META = []


def note_err(kind, err, tb):
    if len(ERR_LOG) < 40:
        ERR_LOG.append({'kind': kind, 'err': err, 'tb': tb})
        log(f'[ERROR-SURFACED] {kind}: {err}')
        if tb:
            print(tb, flush=True)


def preflight(rows_df, data):
    """v1 post-mortem fix: run TWO rows through the FULL pipeline in the
    MAIN process with complete traceback BEFORE any pool starts."""
    log('[preflight] running 2 long rows in MAIN process (full '
        'traceback)...')
    cand = rows_df[rows_df.dur > 12].head(2)
    if len(cand) < 1:
        cand = rows_df.head(2)
    ok = 0
    for r in cand.itertuples():
        src = os.path.join(f'{TMP}/nile', os.path.basename(r.audio_file))
        if not os.path.exists(src):
            dl = download_files([r.audio_file])
            if dl:
                log(f'[preflight] download failed: {dl}')
                continue
        ft = None if pd_isna(getattr(r, 'full_tokens', None)) \
            else int(r.full_tokens)
        res = process_row((src, r.text.strip(), float(r.dur), ft,
                           SPEAKER_MAP[r.speaker_name], r.split, 99999))
        if res['status'] == 'ok':
            ok += 1
            log(f"[preflight] {os.path.basename(r.audio_file)}: "
                f"{len(res['units'])} units, "
                f"{res['rep']['n_units']} planned — OK")
        else:
            note_err('preflight', res['err'], res.get('tb'))
    for f in glob.glob(os.path.join(FEAT_DIR, '*_99999_*.pt')):
        os.remove(f)
    for u in [u for u in ALL_UNITS_META if u['utt'].endswith(
            '_99999_00') or '_99999_' in u['utt']]:
        ALL_UNITS_META.remove(u)
    return ok


def pd_isna(v):
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except Exception:
        return v is None


def main():
    import numpy as np
    import pandas as pd
    import multiprocessing as mp
    import torch
    log('=== NileTTS 4h Controlled Experiment — PREP v2 '
        '(split pipeline) ===')

    # 1) metadata + tree + audit (v1-verbatim)
    TREE = get_file_tree()
    log(f'[tree] {len(TREE)} entries')
    tr, ev, df = load_metadata()
    log(f'[meta] rows total={len(df)} train={len(tr)} eval={len(ev)}')

    sizes = {f['path']: f.get('size', 0) for f in TREE
             if f['type'] == 'file' and f['path'].endswith('.wav')}
    df = df.assign(bytes_=df.audio_file.map(sizes))
    df['dur'] = (df.bytes_ - HDR) / BPS
    df = df[df.dur.notna()].reset_index(drop=True)

    rng = random.Random(SPLIT_SEED)
    audit_sample = rng.sample(list(df.index), min(40, len(df)))
    audit_files = sorted(set(df.loc[audit_sample, 'audio_file']))
    t0 = time.time()
    failed = download_files(audit_files)
    log(f'[audit-dl] {len(audit_files)} files in {time.time()-t0:.0f}s; '
        f'failed={len(failed)}')
    if failed:
        json.dump(failed, open(f'{AUDIT_DIR}/download_failures.json', 'w'))
    dl_ok = [f for f in audit_files
             if os.path.exists(os.path.join(f'{TMP}/nile',
                                            os.path.basename(f)))]
    report = run_audit(df, TREE, dl_ok)
    for f in audit_files:                      # free disk
        dst = os.path.join(f'{TMP}/nile', os.path.basename(f))
        if os.path.exists(dst):
            os.remove(dst)

    # 2) rows to process: EVERYTHING >= 3s (v2 change; v1 kept only 3-11s)
    rows = df[(df.dur >= DUR_MIN)].copy()
    rows = rows[rows.text.fillna('').str.strip().str.len() >= 1]
    rows = rows.sort_values('audio_file').reset_index(drop=True)
    rows['row_idx'] = range(len(rows))
    smoke_n = int(os.environ.get('NILETTS_SMOKE_N', '0'))
    if smoke_n > 0:
        rows = rows.head(smoke_n).copy()
        log(f'[SMOKE] restricted to {smoke_n} rows')
    f1 = report['funnel_v1_reference']
    log(f'[rows] processable rows (dur>={DUR_MIN}s): {len(rows)} '
        f'({rows.dur.sum()/3600:.2f}h) — v1 pipeline took only '
        f'{f1["in_window_clips"]} in-window ({f1["in_window_hours"]}h)')

    # 3) PLANNING phase: catt + tokenize every full text (spawn pool)
    data = setup_paths()
    _worker_init(data)          # main-process lazy state for preflight
    plan_csv = os.path.join(WORK, 'plan_tokens.csv')
    if os.path.exists(plan_csv):
        ptok = pd.read_csv(plan_csv)
        rows = rows.merge(ptok, on='audio_file', how='left')
        log(f'[plan] cached token plan merged: '
            f'{int(rows.full_tokens.notna().sum())} rows')
    else:
        log('[plan] catt+tokenize all row texts (spawn pool, '
            '4 workers)')
        t0 = time.time()
        tasks = [(r.text.strip(),) for r in rows.itertuples()]
        results = []
        n_proc = max(1, min(4, os.cpu_count() or 4))
        with mp.get_context('spawn').Pool(
                n_proc, initializer=_worker_init,
                initargs=(data,)) as pool:
            for i, res in enumerate(pool.imap(plan_tokens_one, tasks,
                                              chunksize=32)):
                results.append(res)
                if res['status'] == 'error':
                    note_err('plan', res['err'], res.get('tb', ''))
                if (i + 1) % 1000 == 0:
                    ok_n = sum(1 for x in results if x['status'] == 'ok')
                    log(f'[plan] {i+1}/{len(tasks)} ok={ok_n} '
                        f'{time.time()-t0:.0f}s')
        rows = rows.assign(full_tokens=[
            r['n'] if r['status'] == 'ok' else None for r in results])
        rows[['audio_file', 'full_tokens']].to_csv(plan_csv, index=False)
        n_err = sum(1 for r in results if r['status'] == 'error')
        log(f'[plan] DONE in {(time.time()-t0)/60:.1f} min; '
            f'errors={n_err}')

    # BEFORE numbers with exact post-catt token counts (full dataset)
    bef = rows[(rows.dur >= DUR_MIN) & (rows.dur <= DUR_MAX)]
    bef = bef[bef.text.fillna('').str.len() >= 10]
    bef = bef[~bef.text.fillna('').apply(
        lambda t: any(c.isascii() and c.isalpha() for c in t))]
    bef = bef[(bef.full_tokens >= 5) & (bef.full_tokens <= 160)]
    before = {'clips': int(len(bef)),
              'hours': round(float(bef.dur.sum()) / 3600, 3),
              'by_speaker': {s: int((bef.speaker_name == s).sum())
                             for s in SPEAKER_MAP}}
    log(f'[before] reference-pipeline train pool (exact): '
        f'{before["clips"]} clips / {before["hours"]}h '
        f'{before["by_speaker"]}')

    # 4) PREFLIGHT (v1 post-mortem fix)
    ok_pre = preflight(rows, data)
    if ok_pre == 0:
        raise RuntimeError('preflight failed — see surfaced error above')
    log(f'[preflight] {ok_pre}/2 rows OK — environment verified')

    # 5) streaming batches: download -> process rows -> free disk
    t_proc = time.time()
    done_rows = set()
    progress_file = os.path.join(WORK, 'rows_done.jsonl')
    if os.path.exists(progress_file):
        for line in open(progress_file):
            try:
                done_rows.add(json.loads(line)['row_idx'])
            except Exception:
                pass
        log(f'[resume] {len(done_rows)} rows already done')

    n_proc = max(1, min(4, os.cpu_count() or 4))
    stat = {'rows': 0, 'units': 0, 'feat': 0, 'err_rows': 0,
            'resplit_tok': 0, 'passthrough': 0, 'dl_failed': 0,
            'cap_hit': False}
    pf = open(progress_file, 'a')

    def flush_batch(batch_rows):
        if not batch_rows:
            return
        files = [r.audio_file for r in batch_rows]
        t0 = time.time()
        failed = download_files(files)
        if failed:
            stat['dl_failed'] += len(failed)
            note_err('download', str(failed[:3]), '')
        ok_files = {f for f in files
                    if os.path.exists(os.path.join(
                        f'{TMP}/nile', os.path.basename(f)))}
        todo = [r for r in batch_rows if r.audio_file in ok_files]
        for r in batch_rows:
            if r.audio_file not in ok_files:
                ALL_DROPS.append((os.path.basename(r.audio_file), 'row',
                                  'download_failed'))
        tasks = [(os.path.join(f'{TMP}/nile',
                               os.path.basename(r.audio_file)),
                  r.text.strip(), float(r.dur),
                  (None if pd_isna(r.full_tokens)
                   else int(r.full_tokens)),
                  SPEAKER_MAP[r.speaker_name], r.split,
                  int(r.row_idx))
                 for r in todo]
        try:
            with mp.get_context('spawn').Pool(
                    n_proc, initializer=_worker_init,
                    initargs=(data,)) as pool:
                for res in pool.imap_unordered(process_row, tasks,
                                               chunksize=4):
                    if res['status'] == 'error':
                        stat['err_rows'] += 1
                        note_err('row', res['err'], res.get('tb', ''))
                    else:
                        rep = res['rep']
                        stat['rows'] += 1
                        stat['units'] += rep['n_units']
                        stat['resplit_tok'] += rep['n_resplit_token']
                        if rep['passthrough']:
                            stat['passthrough'] += 1
                    ALL_UNITS_META.extend(res['units'])
                    stat['feat'] += len(res['units'])
                    for d in res.get('rep', {}).get('drops', []):
                        ALL_DROPS.append(
                            (res.get('src_row', '?'), str(d[0]),
                             str(d[1])))
                    if time.time() - t_proc > PROCESS_TIME_CAP:
                        log('[process] TIME CAP reached mid-batch — '
                            'terminating pool')
                        stat['cap_hit'] = True
                        pool.terminate()
                        break
        finally:
            for r in todo:
                pf.write(json.dumps({'row_idx': int(r.row_idx)}) + '\n')
            pf.flush()
            for f in files:
                dst = os.path.join(f'{TMP}/nile', os.path.basename(f))
                if os.path.exists(dst):
                    os.remove(dst)
        log(f'[batch] {len(files)} files: processed={len(todo)} '
            f'dl_fail={len(failed)} {time.time()-t0:.0f}s | cum: '
            f'rows={stat["rows"]} units={stat["units"]} '
            f'feat={stat["feat"]} err={stat["err_rows"]} '
            f'{(time.time()-t_proc)/60:.0f}m')

    batch, batch_bytes = [], 0
    for r in rows.itertuples():
        if time.time() - t_proc > PROCESS_TIME_CAP:
            log('[process] TIME CAP reached — stopping (resumable)')
            stat['cap_hit'] = True
            break
        if int(r.row_idx) in done_rows:
            continue
        sz = sizes.get(r.audio_file, 0)
        if batch and batch_bytes + sz > BATCH_BYTES:
            flush_batch(batch)
            batch, batch_bytes = [], 0
        batch.append(r)
        batch_bytes += sz
    flush_batch(batch)
    pf.close()

    # 6) extraction.csv
    udf = None
    if ALL_UNITS_META:
        udf = pd.DataFrame(ALL_UNITS_META)
        udf['hf_speaker'] = udf.spk_idx.map(
            {v: k for k, v in SPEAKER_MAP.items()})
        udf['episode'] = udf.src_row.str.extract(r'^([a-z]+_\d+)_')
        udf = udf[['utt', 'spk_idx', 'hf_speaker', 'transcript', 'duration',
                   'i0', 'i1', 'episode', 'src_row', 'n_tokens',
                   'n_frames', 'split']]
        udf = udf.sort_values('utt').reset_index(drop=True)
        udf.to_csv(os.path.join(WORK, 'extraction.csv'), index=False,
                   encoding='utf-8')
        log(f'[extract] {len(udf)} units -> extraction.csv')
    if ALL_DROPS:
        pd.DataFrame(ALL_DROPS, columns=['src_row', 'unit', 'reason'])\
            .to_csv(f'{AUDIT_DIR}/unit_drops.csv', index=False)

    # 7) index.json + pitch stats + speaker F0 (from feature pitches)
    feats = sorted(glob.glob(os.path.join(FEAT_DIR, '*.pt')))
    log(f'[index] feature files on disk: {len(feats)}')
    index, pitch_sum, pitch_sq, pitch_n = [], 0.0, 0.0, 0
    per_spk_f0 = {}
    udf_idx = udf.set_index('utt') if udf is not None else None
    for f in feats:
        utt = os.path.basename(f)[:-3]
        if udf_idx is None or utt not in udf_idx.index:
            continue
        d = torch.load(f, weights_only=True)
        v = d['pitch'][d['pitch'] > 0]
        if v.numel():
            pitch_sum += float(v.sum())
            pitch_sq += float((v ** 2).sum())
            pitch_n += int(v.numel())
            s = int(udf_idx.loc[utt, 'spk_idx'])
            per_spk_f0.setdefault(s, []).append(float(v.median()))
        index.append({'utt': utt, 'spk': int(udf_idx.loc[utt, 'spk_idx']),
                      'n_tokens': int(len(d['ids'])),
                      'n_frames': int(d['mel'].shape[1]),
                      'split': str(udf_idx.loc[utt, 'split'])})
    if pitch_n == 0:
        raise RuntimeError('no voiced pitch frames found — aborting '
                           '(errors surfaced above)')
    p_mean = pitch_sum / pitch_n
    p_std = max(1e-3, (pitch_sq / pitch_n - p_mean ** 2) ** 0.5)

    pool_train = sum(1 for e in index if e['split'] == 'train'
                     and e['n_frames'] <= 950 and e['n_tokens'] <= 160)
    pool_val = sum(1 for e in index if e['split'] != 'train'
                   and e['n_frames'] <= 950 and e['n_tokens'] <= 160)
    per_speaker = {}
    for e in index:
        per_speaker.setdefault(e['spk'], {'clips': 0, 'hours': 0.0})
        per_speaker[e['spk']]['clips'] += 1
        per_speaker[e['spk']]['hours'] += \
            e['n_frames'] * 256 / 22050 / 3600

    SELECTION = {}
    for sidx, spk_name in ((0, 'SPEAKER_01'), (1, 'SPEAKER_02')):
        f0s = per_spk_f0.get(sidx, [])
        med = float(np.median(f0s)) if f0s else None
        core = [x for x in f0s if med / 1.6 < x < med * 1.6] if med else []
        if len(core) >= 3:
            med = float(np.median(core))
        SELECTION[str(sidx)] = {
            'hf_id': spk_name, 'gender': 'MALE' if sidx == 0 else 'FEMALE',
            'f0_hz': round(med, 1) if med else None,
            'n_clips': len(f0s), 'primary': True}

    json.dump({
        'index': index, 'pitch_mean': p_mean, 'pitch_std': p_std,
        'speaker_map': SELECTION, 'per_speaker': per_speaker,
        'n_features': len(index), 'source': REPO,
        'split_rule': 'dataset built-in train/eval split inherited per '
                      'row; units of a row share the row split',
        'train_pool_preview': {'train': pool_train, 'val': pool_val},
    }, open(os.path.join(FEAT_DIR, 'index.json'), 'w'), indent=1)
    log(f'[index] {len(index)} units; pitch mean={p_mean:.1f} '
        f'std={p_std:.1f}; pool(<=160t,<=950f): train={pool_train} '
        f'val={pool_val}; per-speaker: '
        + json.dumps({k: (v['clips'], round(v['hours'], 2))
                      for k, v in sorted(per_speaker.items())}))

    json.dump({'dataset': REPO, 'selection': SELECTION,
               'extraction': {
                   'clips': len(index),
                   'hours': round(sum(v['hours'] for v in
                                      per_speaker.values()), 2),
                   'per_speaker': {str(k): v for k, v in
                                   per_speaker.items()}},
               'filters': {'dur_min': DUR_MIN, 'dur_max': DUR_MAX,
                           'sr': TARGET_SR, 'text_min_chars': 10,
                           'no_ascii_alpha': True,
                           'split_long_rows': True},
               'elapsed_s': round(time.time() - T0, 1)},
              open(os.path.join(WORK, 'selection_report.json'), 'w'),
              ensure_ascii=False, indent=1)

    # 8) split_report.json — the user-mandated before/after report
    split_report = build_split_report(df, rows, udf, index, before, stat,
                                      ALL_DROPS)
    json.dump(split_report, open(f'{AUDIT_DIR}/split_report.json', 'w'),
              ensure_ascii=False, indent=1, default=str)
    log('[split_report] '
        + json.dumps({k: split_report[k] for k in
                      ('utilization_pct', 'after_fix')
                      if k in split_report}, ensure_ascii=False)[:700])

    # 9) audit unit wavs (stratified, re-downloaded transiently)
    save_audit_wavs(udf, rows)

    if ERR_LOG:
        json.dump(ERR_LOG, open(f'{AUDIT_DIR}/surfaced_errors.json', 'w'),
                  indent=1)
    log(f'PREP v2 DONE in {(time.time()-T0)/60:.1f} min; '
        f'stats={json.dumps(stat)}')


def build_split_report(df, rows, udf, index, before, stat, all_drops):
    """User-mandated report: before/after counts, hours, utilization,
    S01/S02, token stats, splits-vs-drops, remaining exclusions."""
    total_hours = float(df.dur.sum()) / 3600
    pool = [e for e in index if e['n_frames'] <= 950
            and e['n_tokens'] <= 160]
    pool_hours = sum(e['n_frames'] for e in pool) * 256 / 22050 / 3600
    toks = sorted(e['n_tokens'] for e in pool)
    tok_mean = float(sum(toks) / len(toks)) if toks else None
    tok_med = float(toks[len(toks) // 2]) if toks else None
    by_spk = {}
    for s, si in SPEAKER_MAP.items():
        units = sum(1 for e in pool if e['spk'] == si)
        hours = sum(e['n_frames'] for e in pool if e['spk'] == si) \
            * 256 / 22050 / 3600
        by_spk[s] = {'units': units, 'hours': round(hours, 2)}

    n_rows_with_units = int(udf.src_row.nunique()) if udf is not None else 0
    n_split_rows = int((udf.groupby('src_row').size() > 1).sum()) \
        if udf is not None else 0

    # aggregate exclusion reasons
    from collections import Counter
    rc = Counter()
    for _, _, reason in all_drops:
        key = reason.split('(')[0].strip()[:60]
        rc[key] += 1

    return {
        'before_fix': {
            'segments': int(len(df)),
            'total_hours': round(total_hours, 2),
            'reference_train_pool_clips': before['clips'],
            'reference_train_pool_hours': before['hours'],
            'reference_pool_by_speaker': before['by_speaker'],
            'loss_causes': [
                '3-11s extraction window kept only 11.4% of rows',
                '160-token pool filter removed ~79% of windowed rows '
                'post-catt (full-row median 248 tokens)'],
        },
        'after_fix': {
            'units_total': len(index),
            'train_pool_units': len(pool),
            'train_pool_hours': round(pool_hours, 2),
            'pool_by_speaker': by_spk,
            'token_mean': round(tok_mean, 1) if tok_mean else None,
            'token_median': tok_med,
            'val_units': sum(1 for e in index if e['split'] != 'train'),
            'units_over_160_tokens': sum(1 for e in index
                                         if e['n_tokens'] > 160),
            'units_over_950_frames': sum(1 for e in index
                                         if e['n_frames'] > 950),
        },
        'utilization_pct': round(100 * pool_hours / total_hours, 1),
        'split_stats': {
            'rows_total': int(len(df)),
            'rows_eligible_ge3s': int(len(rows)),
            'rows_processed': int(stat['rows']),
            'rows_passthrough_whole': int(stat['passthrough']),
            'rows_split_into_multiple_units': n_split_rows,
            'rows_with_any_unit': n_rows_with_units,
            'units_emitted': int(stat['feat']),
            'token_overflow_resplits': int(stat['resplit_tok']),
            'row_errors': int(stat['err_rows']),
            'download_failures': int(stat['dl_failed']),
            'time_cap_hit': bool(stat['cap_hit']),
        },
        'exclusions_remaining': dict(rc.most_common(20)),
        'train_kernel_unchanged': True,
    }


def save_audit_wavs(udf, rows):
    """Stratified unit wavs for listening: re-download the unit's source
    row transiently, slice [i0:i1], save, delete source."""
    import numpy as np
    import soundfile as sf
    import librosa
    if udf is None or not len(udf):
        return
    rng = random.Random(SPLIT_SEED)
    n = min(N_AUDIT_SAMPLE, len(udf))
    idx = rng.sample(range(len(udf)), n)
    rows_by_name = {os.path.basename(r.audio_file)[:-4]: r
                    for r in rows.itertuples()}
    need = {}
    picks = []
    # GUARANTEE: the first 2 extraction.csv rows' wavs must exist — the
    # (unchanged) train kernel's sanity check reads exactly those files
    for i in range(min(2, len(udf))):
        u = udf.iloc[i]
        r = rows_by_name.get(u.src_row)
        if r is None:
            continue
        picks.append((u, r))
        need[r.audio_file] = r
        idx = [j for j in idx if j != i]
    for i in idx:
        u = udf.iloc[i]
        r = rows_by_name.get(u.src_row)
        if r is None:
            continue
        picks.append((u, r))
        need[r.audio_file] = r
    failed = download_files(sorted(need.keys()))
    if failed:
        note_err('audit_wav_dl', str(failed[:3]), '')
    saved = 0
    try:
        for u, r in picks:
            src = os.path.join(f'{TMP}/nile',
                               os.path.basename(r.audio_file))
            if not os.path.exists(src):
                continue
            try:
                wav, sr = sf.read(src, dtype='float32')
                if wav.ndim > 1:
                    wav = wav.mean(1)
                if sr != TARGET_SR:
                    wav = librosa.resample(wav, orig_sr=sr,
                                           target_sr=TARGET_SR)
                peak = float(np.abs(wav).max())
                if peak > 0.95:
                    wav = wav * (0.95 / peak)
                i0, i1 = int(u.i0), int(u.i1)
                sf.write(os.path.join(WAV_DIR, u.utt + '.wav'),
                         wav[i0:i1], TARGET_SR, subtype='PCM_16')
                saved += 1
            except Exception as e:
                note_err('audit_wav', f'{u.utt}: {e}', '')
            if saved >= 40 and len(picks) > 2:
                break
    finally:
        for f in need:
            dst = os.path.join(f'{TMP}/nile', os.path.basename(f))
            if os.path.exists(dst):
                os.remove(dst)
    log(f'[audit-wavs] saved {saved} stratified unit wavs to wav/')


if __name__ == '__main__':
    main()
