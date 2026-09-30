# ============================================================================
# NileTTS_4h_Controlled_Experiment — TRAIN kernel (4-hour hard cap)
# Same MixerTTS architecture + same training recipe as the previous
# from-scratch experiment (verified v3 kernel), on a NEW dataset:
#   KickItLikeShika/NileTTS-dataset — 2 speakers (SPEAKER_01 male,
#   SPEAKER_02 female). Fresh from-scratch start (NO resume — the previous
#   experiment's datasets are deliberately NOT mounted).
# Documented diffs vs v3 (see training_config/param_diff in the report):
#   - SCRATCH_SPEAKERS '0,1' (dataset has 2 speakers; n_speakers=16 model
#     capacity unchanged)
#   - TIME_BUDGET_S default 15300 -> training wall stops at exactly
#     14400s = 4h00m (budget - MARGIN_S 900)
#   - TRAIN/VALIDATION: uses the dataset's OWN built-in split
#     (metadata_train/metadata_eval, recorded per clip in index.json);
#     eval-split clips are held out and monitored (val_log.csv)
#   - resource_log.csv: nvidia-smi + RAM sampled every 60s (monitoring)
#   - extra user evaluation sentences after training (evaluation only)
# Phases (argv): driver | process | train | generate
#   process  : vocalize(catt_eo)+tokenize(egy)+mel+pyin -> features (once)
#   train    : from-scratch GAN training, time-guarded (4h cap)
#   generate : all voices x test sentences + F0 metrics + original-MSA A/B
#     + user_eval (Egyptian / code-switching / difficult words)
# Mounts: dataset mixer-tts-scratch-data (code+models, READ-ONLY),
#         kernel niletts-4h-prep (features + index + audit, READ-ONLY).
# ============================================================================
import os, sys, json, time, random, glob, subprocess

PHASE = sys.argv[1] if len(sys.argv) > 1 else 'driver'

T0 = time.time()
def el():
    return f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-T0))}]"
def log(*a):
    print(el(), *a, flush=True)


INPUT = os.environ.get('KAGGLE_INPUT_ROOT', '/kaggle/input')


def find_root(marker, what):
    """Find the package root containing `marker` (relative path with
    possible subdirs). Root depth is derived from the marker itself."""
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
                            f'{INPUT}; entries: '
                            + repr(os.listdir(INPUT)))


def setup_paths():
    data = find_root('models/mixer128_pytorch.pth', 'data package')
    for p in (os.path.join(data, 'mixer_repo'),
              os.path.join(data, 'tts_arabic_pkg')):
        if p not in sys.path:
            sys.path.insert(0, p)
    return data


WORK = os.environ.get('KAGGLE_WORK', '/kaggle/working')
CKPT_DIR = os.path.join(WORK, 'checkpoints')
FEAT_DIR = os.path.join(WORK, 'features')
for p in (WORK, CKPT_DIR, FEAT_DIR):
    os.makedirs(p, exist_ok=True)

NET_CONFIG_OVERRIDES = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}
EGY_TOKEN_MAP = {'j': 'v', 'q': '<', '^': 't', '*': 'd'}
NILETTS_REPO = 'KickItLikeShika/NileTTS-dataset'
SPEAKER_KEYS = [int(x) for x in os.environ.get(
    'SCRATCH_SPEAKERS', '0,1').split(',')]
PROCESS_TIME_CAP = 2.7 * 3600          # feature extraction cap in this session
# GPU commit runs are capped at 9h (safe assumption): reserve ~45min for
# generate + margin.
TOTAL_TIME_CAP = 8.3 * 3600
GENERATE_RESERVE_S = 45 * 60
SESSION_T0 = float(os.environ.get('SCRATCH_SESSION_T0', time.time()))
# 4h controlled-experiment cap: budget - MARGIN_S(=900) = 14400s = 4h00m
# of training wall time, then save + stop. (Previous default was 28800.)
TRAIN_TIME_BUDGET = float(os.environ.get('TIME_BUDGET_S', 15300))

# User-mandated evaluation sentences (evaluation ONLY, after training)
USER_EVAL_SENTENCES = {
    'egyptian': [
        'دلوقتي هنبدأ الدرس يا عمر.',
        'أنا عايز أعرف إنت عملت إيه النهارده.',
        'بص معايا هنا وخلينا نجرب مرة تانية.',
        'هو إنت ليه ما قلتليش من الأول؟',
    ],
    'code_switching': [
        'أنا هعمل meeting بكرة الصبح.',
        'ممكن تبعتلي الـreport بعد شوية؟',
        'إحنا محتاجين نراجع الـdeadline.',
    ],
    'difficult_words': [
        'الجمل شايل الحمل فوق الجبل في عز الشمس.',
        'القرش غالي والقلب بيوجع يا قمر.',
        'ظرف الثلاجة فيه تلات كوبايين شاي بلح.',
        'عم حسين غطى الغلاية بغلاف فضي غليظ.',
    ],
}


def get_tokenizer():
    setup_paths()
    from tts_arabic.text import (
        arabic_to_buckwalter, tokens_to_ids, phonemes_to_tokens,
        buckwalter_to_phonemes)

    def toks_ms(text):
        return phonemes_to_tokens(buckwalter_to_phonemes(arabic_to_buckwalter(text)))

    def toks_egy(text):
        return [EGY_TOKEN_MAP.get(t, t) for t in toks_ms(text)]

    return toks_ms, toks_egy, tokens_to_ids


def load_test_sentences():
    data = setup_paths()
    sections, cur = {}, None
    for line in open(os.path.join(data, 'evaluation', 'test_sentences.txt'),
                     encoding='utf-8'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            cur = line[1:-1]
            sections[cur] = []
        elif line and cur:
            sections[cur].append(line)
    return sections


_VOCOS = None
def vocos_session():
    global _VOCOS
    if _VOCOS is None:
        data = setup_paths()
        import onnxruntime as ort
        _VOCOS = ort.InferenceSession(
            os.path.join(data, 'models', 'vocos22.onnx'),
            providers=['CPUExecutionProvider'])
    return _VOCOS


def mel_to_wav(mel_80_T):
    import numpy as np
    sess = vocos_session()
    wave = sess.run(None, {
        'mel_spec': mel_80_T[None].astype('float32'),
        'denoise': np.array([0.005], dtype='float32'),
    })[0].astype('float32')[0]
    return 0.9 * wave / (np.abs(wave).max() + 1e-5)


def build_scratch_model(device):
    setup_paths()
    import torch
    from models.mixer_tts.mixer_tts import MixerTTSModel
    from models.mixer_tts import net_config
    net_config.update(NET_CONFIG_OVERRIDES)
    model = MixerTTSModel(**net_config)
    model.add_bin_loss = True
    model.bin_loss_scale = 1.0
    return model.to(device)


def build_original_model(device):
    """Original pretrained MSA mixer128 (for A/B reference samples)."""
    setup_paths()
    import torch
    from models.mixer_tts.mixer_tts import MixerTTSModel
    from models.mixer_tts import net_config
    net_config.update(NET_CONFIG_OVERRIDES)
    model = MixerTTSModel(**net_config)
    data = setup_paths()
    ckpt = torch.load(os.path.join(data, 'models', 'mixer128_pytorch.pth'),
                      map_location='cpu', weights_only=False)
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


# ============================================================================
# PHASE: PROCESS  (vocalize + tokenize + mel + pitch -> features; once)
# ============================================================================
_WORKER_STATE = {}


def _worker_init(data_root):
    import sys
    for p in (os.path.join(data_root, 'mixer_repo'),
              os.path.join(data_root, 'tts_arabic_pkg')):
        if p not in sys.path:
            sys.path.insert(0, p)
    _WORKER_STATE['data'] = data_root


def _worker_process(args):
    """(wav_path, transcript, out_path) -> dict | None. Runs in worker proc."""
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    try:
        if 'catt' not in _WORKER_STATE:
            from tts_arabic.vocalizer.models.core import get_model
            _WORKER_STATE['catt'] = get_model('catt_eo')
        catt = _WORKER_STATE['catt']
        if 'mel_fn' not in _WORKER_STATE:
            from utils.audio import MelSpectrogram
            _WORKER_STATE['mel_fn'] = MelSpectrogram()
        mel_fn = _WORKER_STATE['mel_fn']
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
        toks_egy = _WORKER_STATE['egy']; ids_of = _WORKER_STATE['ids_of']

        wav_path, transcript, out_path = args
        voc = catt.predict(transcript)
        ids = ids_of(toks_egy(voc))
        if len(ids) < 5 or len(ids) > 180:
            return {'status': 'skip_tokens', 'n': len(ids)}
        wav, sr = sf.read(wav_path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 22050, f'unexpected sr {sr}'
        wav_t = torch.from_numpy(wav)[None]
        mel = mel_fn(wav_t).clamp_min(1e-5).log().squeeze(0)   # [80, T]
        f0, _, _ = librosa.pyin(wav, sr=22050, fmin=60, fmax=600,
                                frame_length=1024, hop_length=256)
        f0 = np.where(np.isnan(f0), 0., f0)
        pitch = torch.from_numpy(f0).float()
        if pitch.size(0) < mel.size(1):
            pitch = torch.nn.functional.pad(pitch, (0, mel.size(1) - pitch.size(0)))
        else:
            pitch = pitch[:mel.size(1)]
        torch.save({'ids': torch.LongTensor(ids), 'mel': mel, 'pitch': pitch},
                   out_path)
        return {'status': 'ok', 'n_tokens': len(ids), 'n_frames': mel.size(1)}
    except Exception as e:
        return {'status': 'error', 'err': f'{type(e).__name__}: {e}'}


def phase_process():
    import pandas as pd
    data = setup_paths()
    probe = find_root('selection_report.json', 'probe1 output')
    report = json.load(open(os.path.join(probe, 'selection_report.json')))
    csv_path = os.path.join(probe, 'extraction.csv')
    wav_dir = os.path.join(probe, 'wav')
    df = pd.read_csv(csv_path)
    log(f'[process] {len(df)} extracted clips from probe1; '
        f'{report["extraction"]["hours"]}h')

    # order: primary male (0) then primary female (4) then the rest
    priority = {0: 0, 4: 1}
    df['prio'] = df.spk_idx.map(lambda s: priority.get(s, 2 + s))
    df = df.sort_values(['prio']).reset_index(drop=True)

    done = set(os.path.basename(f)[:-3] for f in
               glob.glob(os.path.join(FEAT_DIR, '*.pt')))
    tasks = []
    for _, r in df.iterrows():
        if r.utt in done:
            continue
        tasks.append((os.path.join(wav_dir, r.utt + '.wav'), r.transcript,
                      os.path.join(FEAT_DIR, r.utt + '.pt')))
    log(f'[process] to process: {len(tasks)} (already done: {len(done)})')

    import multiprocessing as mp
    n_proc = max(1, os.cpu_count() or 4)
    log(f'[process] pool workers: {n_proc}')
    t_start = time.time()
    stats = {'ok': 0, 'skip_tokens': 0, 'error': 0}
    with mp.get_context('fork').Pool(n_proc, initializer=_worker_init,
                                     initargs=(data,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_process, tasks,
                                                    chunksize=8)):
            stats[res['status']] = stats.get(res['status'], 0) + 1
            if (i + 1) % 200 == 0:
                log(f'[process] {i+1}/{len(tasks)} '
                    f'({stats}) {time.time()-t_start:.0f}s '
                    f'cap={PROCESS_TIME_CAP:.0f}s')
            if time.time() - t_start > PROCESS_TIME_CAP:
                log('[process] TIME CAP reached — stopping early')
                pool.terminate()
                break

    # rebuild index from saved files (ground truth) joined with csv rows
    feats = glob.glob(os.path.join(FEAT_DIR, '*.pt'))
    log(f'[process] feature files on disk: {len(feats)} | {stats}')
    df_index = df.set_index('utt')
    index = []
    import torch
    pitch_sum, pitch_sq, pitch_n = 0.0, 0.0, 0
    for f in feats:
        utt = os.path.basename(f)[:-3]
        if utt not in df_index.index:
            continue
        d = torch.load(f, weights_only=True)
        v = d['pitch'][d['pitch'] > 0]
        if v.numel():
            pitch_sum += float(v.sum()); pitch_sq += float((v ** 2).sum())
            pitch_n += int(v.numel())
        index.append({'utt': utt, 'spk': int(df_index.loc[utt, 'spk_idx']),
                      'n_tokens': int(len(d['ids'])),
                      'n_frames': int(d['mel'].shape[1])})
    if pitch_n == 0:
        raise RuntimeError('no voiced pitch frames found — aborting')
    p_mean = pitch_sum / pitch_n
    p_std = max(1e-3, (pitch_sq / pitch_n - p_mean ** 2) ** 0.5)
    speaker_map = report['selection']
    per_spk = {}
    for e in index:
        s = e['spk']
        per_spk.setdefault(s, {'clips': 0, 'hours': 0.0})
        per_spk[s]['clips'] += 1
        per_spk[s]['hours'] += e['n_frames'] * 256 / 22050 / 3600
    json.dump({
        'index': index, 'pitch_mean': p_mean, 'pitch_std': p_std,
        'speaker_map': speaker_map, 'per_speaker': per_spk,
        'n_features': len(index), 'source': 'masri-podcast-300h (egyptian)',
    }, open(os.path.join(FEAT_DIR, 'index.json'), 'w'), indent=1)
    log(f'[process] index: {len(index)} utts, pitch mean={p_mean:.1f} '
        f'std={p_std:.1f}; per-speaker: '
        + json.dumps({k: (v["clips"], round(v["hours"], 2))
                      for k, v in sorted(per_spk.items())}))


# ============================================================================
# PHASE: TRAIN  (from-scratch GAN, author recipe, resumable)
# ============================================================================
def phase_train():
    import numpy as np
    import torch
    import torch.nn.functional as F
    setup_paths()
    from models.mixer_tts.mixer_tts import MixerTTSModel
    from models.mixer_tts import net_config
    from models.common.loss import (
        PatchDiscriminatorCond, extract_chunks, calc_feature_match_loss)
    from models.mixer_tts.modules.data_function import BetaBinomialInterpolator

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'[train] device={device}')
    if device == 'cpu':
        log('[train] WARNING: no GPU detected!')

    BATCH_SIZE = int(os.environ.get('SCRATCH_BATCH_SIZE', 16))
    GAN_WARMUP = int(os.environ.get('GAN_WARMUP', 2000))
    G_LR, D_LR = 1e-4, 1e-4
    LR_MIN = 1e-5
    LR_DECAY_ITERS = 150000   # iterations until LR reaches its minimum
    # FIXED anchor (user-requested fix): decay progress is ALWAYS measured
    # from LR_ANCHOR_IT, never from start_iter. If a session dies at iter
    # 250000 and resumes there, progress must be (250000 - 196142) and NOT
    # (250000 - 250000) = 0, which would wrongly reset LR to full 1e-4.
    LR_ANCHOR_IT = 196142

    def lr_at(current_it):
        # current_it: global current iteration number
        # anchor: the FIXED constant LR_ANCHOR_IT above (NOT start_iter),
        # so resuming from any later checkpoint continues the same
        # schedule with no LR reset/jump at the resume point.
        progress = min(1.0, max(0.0,
                                (current_it - LR_ANCHOR_IT) / LR_DECAY_ITERS))
        import math
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return LR_MIN + (G_LR - LR_MIN) * cos_factor
    FMATCH_W, SCORE_W = 1.0, 4.0
    SAVE_EVERY = int(os.environ.get('SCRATCH_SAVE_EVERY', 500))
    SNAPSHOT_EVERY = int(os.environ.get('SCRATCH_SNAPSHOT_EVERY', 5000))
    PROBE_EVERY = int(os.environ.get('SCRATCH_PROBE_EVERY', 2000))
    LOG_EVERY = 25
    TAR_LEN, MAX_FRAMES, MAX_TOKENS = 128, 950, 160
    MAX_ITERS = int(os.environ.get('SCRATCH_MAX_ITERS', 400000))

    session_elapsed = time.time() - SESSION_T0
    budget = min(TRAIN_TIME_BUDGET,
                 TOTAL_TIME_CAP - session_elapsed - GENERATE_RESERVE_S)
    if budget <= 300:
        log(f'[train] almost no time left (elapsed {session_elapsed:.0f}s) '
            f'— saving what exists and skipping training')
        budget = 300
    MARGIN_S = min(900.0, budget * 0.1)
    log(f'[train] session elapsed {session_elapsed/60:.0f}m -> train '
        f'budget {budget/60:.0f}m')

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = True

    # ---------------- features ----------------
    index_cands = [os.path.join(FEAT_DIR, 'index.json')] + sorted(
        glob.glob(f'{INPUT}/**/features/index.json', recursive=True))
    feat_root = next((c for c in index_cands if os.path.exists(c)), None)
    if feat_root is None:
        raise FileNotFoundError('no features/index.json found (run process '
                                'phase or mount previous output)')
    if os.path.dirname(feat_root) != FEAT_DIR:
        log(f'[train] using mounted features at {os.path.dirname(feat_root)}')
    fdir = os.path.dirname(feat_root)
    meta = json.load(open(feat_root))
    ps = {'mean': meta['pitch_mean'], 'std': meta['pitch_std']}
    # TRAIN/VALIDATION: dataset's own built-in split, recorded per clip in
    # index.json ('split': 'train' | 'eval'). Eval-split clips are HELD OUT
    # of training and used for validation monitoring only. (Previous
    # experiment had no val split — documented difference, user-mandated.)
    items = [(e['utt'], e['spk']) for e in meta['index']
             if e.get('split', 'train') == 'train'
             and e['n_frames'] <= MAX_FRAMES and e['n_tokens'] <= MAX_TOKENS]
    val_items = [(e['utt'], e['spk']) for e in meta['index']
                 if e.get('split', 'train') != 'train'
                 and e['n_frames'] <= MAX_FRAMES
                 and e['n_tokens'] <= MAX_TOKENS]
    log(f'[train] pool: {len(items)} train / {len(val_items)} val (held out)'
        f' / {len(meta["index"])} features '
        f'(pitch mean={ps["mean"]:.1f} std={ps["std"]:.1f})')

    cache = {}
    def get_feat(utt):
        if utt not in cache:
            cache[utt] = torch.load(os.path.join(fdir, utt + '.pt'),
                                    weights_only=True)
        return cache[utt]

    betabin = BetaBinomialInterpolator(round_mel_len_to=200,
                                       round_text_len_to=40)

    def make_batch(item_list):
        data = []
        for utt, spk in item_list:
            d = get_feat(utt)
            pitch = d['pitch']
            mask = pitch > 0
            pitch = torch.where(mask, (pitch - ps['mean']) / ps['std'],
                                torch.zeros_like(pitch))
            mel = d['mel']; ids = d['ids']
            energy = torch.norm(mel.float(), dim=0, p=2)
            prior = torch.from_numpy(betabin(mel.size(1), len(ids))).float()
            data.append((ids, mel, pitch, energy, prior, spk))
        data.sort(key=lambda x: -x[1].size(1))
        B = len(data)
        max_tok = max(len(x[0]) for x in data)
        max_fr = max(x[1].size(1) for x in data)
        text_p = torch.zeros(B, max_tok, dtype=torch.long)
        in_len = torch.zeros(B, dtype=torch.long)
        mel_p = torch.zeros(B, 80, max_fr)
        out_len = torch.zeros(B, dtype=torch.long)
        pit_p = torch.zeros(B, max_fr)
        ene_p = torch.zeros(B, max_fr)
        pri_p = torch.zeros(B, max_fr, max_tok)
        spk_t = torch.zeros(B, dtype=torch.long)
        for i, (ids, mel, pitch, energy, prior, spk) in enumerate(data):
            L = len(ids); T = mel.size(1)
            text_p[i, :L] = ids; in_len[i] = L
            mel_p[i, :, :T] = mel; out_len[i] = T
            pit_p[i, :T] = pitch; ene_p[i, :T] = energy
            pri_p[i, :T, :L] = prior
            spk_t[i] = spk
        return (text_p.to(device), in_len.to(device), mel_p.to(device),
                out_len.to(device), pit_p.to(device), ene_p.to(device),
                pri_p.to(device), spk_t.to(device))

    # ---------------- model + optimizers (author recipe) ----------------
    net_config.update(NET_CONFIG_OVERRIDES)
    model = MixerTTSModel(**net_config)
    model.add_bin_loss = True
    model.bin_loss_scale = 1.0
    n_params = sum(p.numel() for p in model.parameters())
    log(f'[train] fresh model: {n_params/1e6:.2f}M params')
    model = model.to(device)

    critic = PatchDiscriminatorCond(
        2, 32, d_emb=model.speaker_emb.embedding_dim).to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=G_LR, betas=(0.0, 0.99),
                              weight_decay=1e-6)
    opt_d = torch.optim.AdamW(critic.parameters(), lr=D_LR, betas=(0.0, 0.99),
                              weight_decay=1e-6)

    # ---------------- resume (own previous version output) ----------------
    resume_cands = sorted(
        glob.glob(f'{INPUT}/**/checkpoints/states.pth', recursive=True))
    own = os.path.join(CKPT_DIR, 'states.pth')
    if os.path.exists(own):
        resume_cands.append(own)
    start_iter = 0
    if resume_cands:
        best, best_it = None, -1
        for c in resume_cands:
            try:
                it = torch.load(c, map_location='cpu', weights_only=False)\
                    .get('iter', 0)
                if it > best_it:
                    best, best_it = c, it
            except Exception:
                continue
        if best is not None:
            st = torch.load(best, map_location='cpu', weights_only=False)
            model.load_state_dict(st['model'], strict=True)
            critic.load_state_dict(st['critic'], strict=True)
            if 'opt_g' in st:
                opt_g.load_state_dict(st['opt_g'])
                opt_d.load_state_dict(st['opt_d'])
            start_iter = st.get('iter', 0)
            for opt in (opt_g, opt_d):
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            log(f'[train] RESUMED from {best} (iter {start_iter})')
    if start_iter == 0:
        log('[train] NO resume checkpoint found — FRESH FROM-SCRATCH start '
            '(iter 0). Controlled experiment: previous experiment\'s '
            'checkpoints deliberately not mounted.')

    log_path = os.path.join(WORK, 'train_log.csv')
    val_log_path = os.path.join(WORK, 'val_log.csv')
    if not os.path.exists(log_path) or start_iter == 0:
        with open(log_path, 'w') as f:
            f.write('iter,phase,loss,mel,durs,pitch,ctc,bin,score,fmatch,'
                    'loss_d,secs,it_per_s,skipped,lr\n')
        with open(val_log_path, 'w') as f:
            f.write('iter,loss,mel,durs,pitch,ctc,bin,secs\n')
        json.dump({
                'experiment': 'NileTTS_4h_Controlled_Experiment',
                'dataset': NILETTS_REPO,
                'speakers': SPEAKER_KEYS,
                'n_train_pool': len(items), 'n_val_pool': len(val_items),
                'time_budget_s': TRAIN_TIME_BUDGET,
                'batch_size': BATCH_SIZE, 'gan_warmup': GAN_WARMUP,
                'g_lr': G_LR, 'd_lr': D_LR, 'fmatch_w': FMATCH_W,
                'score_w': SCORE_W, 'tar_len': TAR_LEN,
                'lr_min': LR_MIN, 'lr_decay_iters': LR_DECAY_ITERS,
                'lr_anchor_it': LR_ANCHOR_IT,
                'max_frames': MAX_FRAMES, 'max_tokens': MAX_TOKENS,
                'max_iters': MAX_ITERS, 'n_params': n_params,
                'pitch_mean': ps['mean'], 'pitch_std': ps['std'],
                'device': device,
                'design': ('from-scratch MixerTTS (author train_ms recipe): '
                           'AdamW(1e-4, betas 0/0.99, wd 1e-6), LSGAN '
                           'PatchDiscriminatorCond from iter GAN_WARMUP, '
                           'bin loss scale 1.0 from start, 2 voices '
                           '(SPEAKER_01 male + SPEAKER_02 female) from '
                           'KickItLikeShika/NileTTS-dataset; same recipe as '
                           'the previous masri-podcast-300h experiment'),
            }, open(os.path.join(WORK, 'run_config.json'), 'w'), indent=1)

    def save_states(it, snapshot=False):
        torch.save({'model': model.state_dict(), 'critic': critic.state_dict(),
                    'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                    'iter': it, 'net_config': net_config,
                    'pitch_mean': ps['mean'], 'pitch_std': ps['std']},
                   os.path.join(CKPT_DIR, 'states.pth'))
        if snapshot:
            torch.save({'model': model.state_dict(), 'iter': it,
                        'net_config': net_config,
                        'pitch_mean': ps['mean'], 'pitch_std': ps['std']},
                       os.path.join(CKPT_DIR, f'states_{it}.pth'))

    # ---------------- probes ----------------
    toks_ms, toks_egy, ids_of = get_tokenizer()
    sections = load_test_sentences()
    probe_texts = [sections['egyptian'][0], sections['egyptian'][4]]

    def gen_probe(it):
        try:
            import soundfile as sf
            model.eval()
            pdir = os.path.join(WORK, 'probes')
            os.makedirs(pdir, exist_ok=True)
            for si, text in enumerate(probe_texts):
                ids = ids_of(toks_egy(text))
                x = torch.LongTensor([ids]).to(device)
                for spk in SPEAKER_KEYS:
                    mel = model.infer(x, pace=1.0, speaker=spk, emotion=0)
                    m = mel.transpose(1, 2)[0].cpu().numpy()
                    sf.write(os.path.join(pdir, f'it{it:06d}_s{si}_spk{spk}.wav'),
                             mel_to_wav(m), 22050, subtype='PCM_16')
            model.train()
            log(f'[train] probes saved at iter {it}')
        except Exception as e:
            model.train()
            log(f'[train] probe FAILED (non-fatal): {type(e).__name__}: {e}')

    # ---------------- validation monitoring (user-mandated; monitoring
    # ONLY — does not touch training math). Uses the dataset's own held-out
    # eval-split clips with the same pool criteria. ----------------
    def eval_val(it):
        if not val_items:
            return
        try:
            model.eval()
            t0 = time.time()
            agg = {'loss': 0., 'mel': 0., 'durs': 0., 'pitch': 0.,
                   'ctc': 0., 'bin': 0.}
            n_b = 0
            with torch.no_grad():
                for bstart in range(0, len(val_items), BATCH_SIZE):
                    bidx = list(range(bstart,
                                      min(bstart + BATCH_SIZE, len(val_items))))
                    (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p,
                     spk_t) = make_batch([val_items[i] for i in bidx])
                    (pred_spect, _, pred_log_durs, pred_pitch, _, attn_soft,
                     attn_logprob, attn_hard, attn_hard_dur) = model(
                        text=text_p, text_len=in_len, pitch=pit_p,
                        energy=ene_p, spect=mel_p, spect_len=out_len,
                        attn_prior=pri_p, lm_tokens=None, speaker=spk_t,
                        emotion=torch.zeros_like(spk_t))
                    (vloss, vdurs, _acc, _, _, vpitch, _, vmel,
                     vctc, vbin) = model._metrics(
                        pred_durs=pred_log_durs, pred_pitch=pred_pitch,
                        pred_energy=None, true_durs=attn_hard_dur,
                        true_text_len=in_len, true_pitch=pit_p,
                        true_energy=ene_p, true_spect=mel_p,
                        pred_spect=pred_spect, true_spect_len=out_len,
                        attn_logprob=attn_logprob, attn_soft=attn_soft,
                        attn_hard=attn_hard, attn_hard_dur=attn_hard_dur)
                    agg['loss'] += float(vloss); agg['mel'] += float(vmel)
                    agg['durs'] += float(vdurs); agg['pitch'] += float(vpitch)
                    agg['ctc'] += float(vctc)
                    agg['bin'] += float(vbin) if vbin is not None else 0.
                    n_b += 1
            model.train()
            if n_b:
                row = {k: v / n_b for k, v in agg.items()}
                with open(val_log_path, 'a') as f:
                    f.write(f"{it},{row['loss']:.5f},{row['mel']:.5f},"
                            f"{row['durs']:.5f},{row['pitch']:.5f},"
                            f"{row['ctc']:.5f},{row['bin']:.5f},"
                            f"{time.time()-t0:.1f}\n")
                log(f"[val] it {it} loss={row['loss']:.3f} "
                    f"mel={row['mel']:.3f} durs={row['durs']:.3f} "
                    f"pitch={row['pitch']:.3f} ctc={row['ctc']:.3f} "
                    f"({time.time()-t0:.1f}s)")
        except Exception as e:
            model.train()
            log(f'[val] FAILED (non-fatal): {type(e).__name__}: {e}')

    # ---------------- resource monitor (monitoring only) ----------------
    def start_resource_monitor():
        import threading
        res_path = os.path.join(WORK, 'resource_log.csv')
        with open(res_path, 'w') as f:
            f.write('t,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,'
                    'ram_used_gb\n')
        def sample():
            try:
                gpu = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,'
                     'memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                parts = [x.strip() for x in gpu.split(',')]
                ram = subprocess.run(['free', '-m'], capture_output=True,
                                     text=True, timeout=10).stdout
                ram_used = ''
                for line in ram.splitlines():
                    if line.startswith('Mem:'):
                        ram_used = f'{int(line.split()[2]) / 1024:.1f}'
                with open(res_path, 'a') as f:
                    f.write(f"{time.time()-t_start:.0f},{parts[0]},"
                            f"{parts[1]},{parts[2]},{ram_used}\n")
            except Exception:
                pass
        class Monitor(threading.Thread):
            def run(self):
                while True:
                    sample()
                    time.sleep(60)
        m = Monitor(daemon=True)
        m.start()
        return m

    # ---------------- loop ----------------
    model.train(); critic.train()
    t_start = time.time()
    it = start_iter
    skipped = 0
    speed_smooth = 0.0
    start_resource_monitor()   # monitoring only (resource_log.csv)
    log(f'[train] start iter={it} budget={budget:.0f}s '
        f'pool={len(items)} val={len(val_items)} '
        f'gan_warmup={GAN_WARMUP}')

    stop = False
    while not stop and it < MAX_ITERS:
        order = list(range(len(items)))
        random.shuffle(order)
        for bstart in range(0, len(order), BATCH_SIZE):
            if time.time() - t_start >= budget - MARGIN_S:
                stop = True; break
            if it >= MAX_ITERS:
                stop = True; break
            if time.time() - SESSION_T0 >= TOTAL_TIME_CAP - GENERATE_RESERVE_S:
                log('[train] TOTAL session cap reached — saving + stopping')
                stop = True; break
            t_it = time.time()
            try:
                bidx = order[bstart:bstart + BATCH_SIZE]
                (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p,
                 spk_t) = make_batch([items[i] for i in bidx])

                (pred_spect, _, pred_log_durs, pred_pitch, _, attn_soft,
                 attn_logprob, attn_hard, attn_hard_dur) = model(
                    text=text_p, text_len=in_len, pitch=pit_p, energy=ene_p,
                    spect=mel_p, spect_len=out_len, attn_prior=pri_p,
                    lm_tokens=None, speaker=spk_t,
                    emotion=torch.zeros_like(spk_t))

                (loss, durs_loss, acc, _, _, pitch_loss, _, mel_loss,
                 ctc_loss, bin_loss) = model._metrics(
                    pred_durs=pred_log_durs, pred_pitch=pred_pitch,
                    pred_energy=None, true_durs=attn_hard_dur,
                    true_text_len=in_len, true_pitch=pit_p, true_energy=ene_p,
                    true_spect=mel_p, pred_spect=pred_spect,
                    true_spect_len=out_len, attn_logprob=attn_logprob,
                    attn_soft=attn_soft, attn_hard=attn_hard,
                    attn_hard_dur=attn_hard_dur)

                score_l = torch.tensor(0., device=device)
                fmatch_l = torch.tensor(0., device=device)
                loss_d_val = 0.
                warmup = it < GAN_WARMUP

                if not warmup:
                    tar_len_ = min(out_len.min().item(), TAR_LEN)
                    mel_ids = torch.arange(mel_p.size(0), device=mel_p.device)
                    ofx_perc = torch.rand(out_len.size(), device=out_len.device)
                    ofx = (ofx_perc * (out_len + tar_len_) - tar_len_ / 2) \
                        .clamp(out_len * 0, out_len - tar_len_).long()
                    chunks_org = extract_chunks(mel_p, ofx, mel_ids=mel_ids,
                                                chunk_len=tar_len_)
                    chunks_gen = extract_chunks(
                        pred_spect.transpose(1, 2), ofx, mel_ids=mel_ids,
                        chunk_len=tar_len_)
                    chunks_org_ = (chunks_org.unsqueeze(1) + 4.5) / 2.5
                    chunks_gen_ = (chunks_gen.unsqueeze(1) + 4.5) / 2.5
                    with torch.no_grad():
                        spk_vecs = model.speaker_emb.weight[spk_t]
                        spk_vecs = F.normalize(spk_vecs, p=2, dim=1)

                    d_org, fmaps_org = critic(
                        chunks_org_.requires_grad_(True), spk_vecs)
                    d_gen, _ = critic(chunks_gen_.detach(), spk_vecs)
                    loss_d = 0.5 * (d_org - 1).square().mean() \
                        + 0.5 * (d_gen).square().mean()
                    critic.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1000.)
                    opt_d.step()
                    loss_d_val = loss_d.item()

                    d_gen2, fmaps_gen = critic(chunks_gen_, spk_vecs)
                    score_l = (d_gen2 - 1).square().mean()
                    fmatch_l = calc_feature_match_loss(fmaps_gen, fmaps_org)
                    loss = loss + SCORE_W * score_l + FMATCH_W * fmatch_l

                opt_g.zero_grad()
                loss.backward()
                current_lr = lr_at(it)
                for g in opt_g.param_groups:
                    g['lr'] = current_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20.)
                opt_g.step()
                it += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt_g.zero_grad(); opt_d.zero_grad()
                skipped += 1
                continue

            dt = time.time() - t_it
            inst = 1.0 / max(dt, 1e-6)
            speed_smooth = 0.98 * speed_smooth + 0.02 * inst if speed_smooth else inst

            if it <= 6 or it % LOG_EVERY == 0:
                el_s = time.time() - t_start
                log(f"it {it} loss={loss.item():.3f} mel={mel_loss.item():.3f} "
                    f"durs={durs_loss.item():.3f} "
                    f"pitch={pitch_loss.item():.3f} "
                    f"ctc={ctc_loss.item():.3f} "
                    f"bin={bin_loss.item() if bin_loss is not None else 0:.3f} "
                    f"score={score_l.item():.3f} fmatch={fmatch_l.item():.3f} "
                    f"d={loss_d_val:.3f} lr={current_lr:.3e} | "
                    f"{speed_smooth:.2f}it/s "
                    f"skip={skipped} ({el_s/60:.1f}m)")
                with open(log_path, 'a') as f:
                    f.write(f"{it},{'warmup' if warmup else 'gan'},"
                            f"{loss.item():.5f},{mel_loss.item():.5f},"
                            f"{durs_loss.item():.5f},{pitch_loss.item():.5f},"
                            f"{ctc_loss.item():.5f},"
                            f"{bin_loss.item() if bin_loss is not None else 0:.5f},"
                            f"{score_l.item():.5f},{fmatch_l.item():.5f},"
                            f"{loss_d_val:.5f},{time.time()-t_start:.1f},"
                            f"{speed_smooth:.4f},{skipped},{current_lr:.6e}\n")

            if it % SAVE_EVERY == 0:
                save_states(it)
            if it % SNAPSHOT_EVERY == 0:
                save_states(it, snapshot=True)
            if it % PROBE_EVERY == 0:
                gen_probe(it)
                eval_val(it)
        # end epoch loop

    save_states(it, snapshot=(it % SNAPSHOT_EVERY != 0 and it > start_iter))
    try:
        log(f'[train] max GPU mem: '
            f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    except Exception:
        pass
    log(f'[train] FINISHED at iter {it} '
        f'wall={time.strftime("%H:%M:%S", time.gmtime(time.time()-t_start))}')


# ============================================================================
# PHASE: GENERATE  (samples + F0 identity metrics + original-MSA A/B
#                   + user-mandated evaluation set)
# ============================================================================
def synth_set(model, out_base, speakers, tag_manifest):
    import torch
    import soundfile as sf
    toks_ms, toks_egy, ids_of = get_tokenizer()
    sections = load_test_sentences()
    device = next(model.parameters()).device
    os.makedirs(out_base, exist_ok=True)
    manifest = []
    for spk in speakers:
        spk_dir = os.path.join(out_base, f'speaker_{spk}')
        os.makedirs(spk_dir, exist_ok=True)
        for mode, prefix in (('egyptian', 'egy'), ('msa', 'msa')):
            for i, text in enumerate(sections[mode]):
                try:
                    toks = toks_ms(text) if mode == 'msa' else toks_egy(text)
                    ids = ids_of(toks)
                    x = torch.LongTensor([ids]).to(device)
                    with torch.inference_mode():
                        mel = model.infer(x, pace=1.0, speaker=spk, emotion=0)
                    m = mel.transpose(1, 2)[0].cpu().numpy()
                    stem = f'{prefix}_{i:02d}'
                    sf.write(os.path.join(spk_dir, stem + '.wav'),
                             mel_to_wav(m), 22050, subtype='PCM_16')
                    manifest.append({'speaker': spk, 'mode': mode, 'index': i,
                                     'text': text, 'n_tokens': len(ids)})
                except Exception as e:
                    log(f'[generate{tag_manifest}] speaker_{spk} {prefix}_{i:02d}'
                        f' FAILED (skipped): {type(e).__name__}: {e}')
        log(f'[generate{tag_manifest}] speaker_{spk} done')
    with open(os.path.join(out_base, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_base


def synth_user_eval(model, out_base, speakers):
    """User-mandated evaluation sentences (Egyptian, code-switching,
    difficult words) for each speaker. Evaluation ONLY — runs after the
    4h training budget is exhausted, no training influence."""
    import torch
    import soundfile as sf
    toks_ms, toks_egy, ids_of = get_tokenizer()
    device = next(model.parameters()).device
    os.makedirs(out_base, exist_ok=True)
    manifest = []
    for spk in speakers:
        spk_dir = os.path.join(out_base, f'speaker_{spk}')
        os.makedirs(spk_dir, exist_ok=True)
        for cat, sentences in USER_EVAL_SENTENCES.items():
            for i, text in enumerate(sentences):
                try:
                    toks = toks_egy(text)
                    ids = ids_of(toks)
                    x = torch.LongTensor([ids]).to(device)
                    with torch.inference_mode():
                        mel = model.infer(x, pace=1.0, speaker=spk, emotion=0)
                    m = mel.transpose(1, 2)[0].cpu().numpy()
                    sf.write(os.path.join(spk_dir, f'{cat}_{i:02d}.wav'),
                             mel_to_wav(m), 22050, subtype='PCM_16')
                    manifest.append({'speaker': spk, 'category': cat,
                                     'index': i, 'text': text,
                                     'n_tokens': len(ids)})
                except Exception as e:
                    log(f'[generate:user_eval] speaker_{spk} {cat}_{i:02d}'
                        f' FAILED (skipped): {type(e).__name__}: {e}')
        log(f'[generate:user_eval] speaker_{spk} done')
    with open(os.path.join(out_base, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_base


def f0_report(sample_dir, out_json, source_f0=None):
    import numpy as np
    import soundfile as sf
    import librosa
    out = {}
    for spk_dir in sorted(glob.glob(os.path.join(sample_dir, 'speaker_*'))):
        spk = os.path.basename(spk_dir)
        f0s = []
        for f in sorted(glob.glob(os.path.join(spk_dir, '*.wav'))):
            wav, sr = sf.read(f, dtype='float32')
            f0, _, _ = librosa.pyin(wav, sr=sr, fmin=60, fmax=400,
                                    frame_length=1024, hop_length=256)
            f0 = f0[~np.isnan(f0)]
            if f0.size > 30:
                f0s.append(float(np.median(f0)))
        med = float(np.median(f0s)) if f0s else 0.0
        src = (source_f0 or {}).get(spk, {})
        out[spk] = {'synth_median_f0': round(med, 1),
                    'source_median_f0': src.get('f0_hz'),
                    'source_gender': src.get('gender'),
                    'dev_hz': (round(med - src['f0_hz'], 1)
                               if src.get('f0_hz') else None)}
        log(f'[f0] {spk}: synth={med:.1f}Hz source={src.get("f0_hz")}Hz')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, 'w'), indent=1)


def phase_generate():
    import torch
    import re
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'[generate] device={device}')

    # latest checkpoint (own or mounted)
    cands = sorted(glob.glob(f'{INPUT}/**/checkpoints/states.pth',
                             recursive=True))
    own = os.path.join(CKPT_DIR, 'states.pth')
    if os.path.exists(own):
        cands.append(own)
    if not cands:
        log('[generate] NO checkpoint — nothing to generate from!')
        return
    best, best_it = None, -1
    for c in cands:
        try:
            it = torch.load(c, map_location='cpu', weights_only=False)\
                .get('iter', 0)
            if it > best_it:
                best, best_it = c, it
        except Exception:
            continue
    log(f'[generate] using checkpoint {best} (iter {best_it})')

    # speaker map for source F0 reference
    source_f0 = {}
    try:
        probe = find_root('selection_report.json', 'probe1 output')
        rep = json.load(open(os.path.join(probe, 'selection_report.json')))
        for k, v in rep['selection'].items():
            source_f0[f'speaker_{k}'] = v
    except Exception as e:
        log(f'[generate] no source F0 map: {e}')

    st = torch.load(best, map_location='cpu', weights_only=False)
    model = build_scratch_model(device)
    model.load_state_dict(st['model'], strict=True)
    model.eval()
    out = synth_set(model, os.path.join(WORK, 'samples', 'egyptian_scratch'),
                    SPEAKER_KEYS, ':new')
    f0_report(out, os.path.join(WORK, 'metrics', 'f0_new_model.json'),
              source_f0)
    # ---- user-mandated evaluation set (Egyptian / code-switching /
    #      difficult words), both speakers, evaluation ONLY ----
    try:
        synth_user_eval(model, os.path.join(WORK, 'samples', 'user_eval'),
                        SPEAKER_KEYS)
    except Exception as e:
        log(f'[generate] user_eval failed: {e}')
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ---- A/B: original pretrained MSA model, speakers 0-3 ----
    orig_spks = [int(x) for x in os.environ.get('SCRATCH_ORIG_SPEAKERS',
                                                '0,1,2,3').split(',')]
    try:
        om = build_original_model(device)
        out2 = synth_set(om, os.path.join(WORK, 'samples',
                                          'original_msa_reference'),
                         orig_spks, ':orig')
        f0_report(out2, os.path.join(WORK, 'metrics', 'f0_original.json'))
        del om
    except Exception as e:
        log(f'[generate] original-model A/B failed: {e}')

    # ---- training curve ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        log_path = os.path.join(WORK, 'train_log.csv')
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if len(df) > 2:
                fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                                         constrained_layout=True)
                for ax, col in zip(axes.flat, ['mel', 'durs', 'ctc', 'loss_d']):
                    if col in df:
                        ax.plot(df.iter, df[col], lw=0.6)
                        ax.set_title(col)
                        ax.set_xlabel('iter')
                fig.savefig(os.path.join(WORK, 'training_curve.png'), dpi=120)
                log('[generate] training_curve.png saved')
    except Exception as e:
        log(f'[generate] curve plot failed: {e}')
    log('[generate] DONE')


# ============================================================================
# PHASE: DRIVER
# ============================================================================
def driver():
    log('=== MixerTTS Egyptian FROM-SCRATCH (GPU) ===')
    import torch
    log(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log(f'  gpu {i}: {torch.cuda.get_device_name(i)}')

    try:
        import onnxruntime  # noqa
        log('onnxruntime available')
    except ImportError:
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                            'onnxruntime'], capture_output=True, text=True)
        log(f'onnxruntime install rc={r.returncode}')

    # ---- sanity: pipeline + fresh model forward + infer + vocos ----
    torch.manual_seed(0)   # deterministic sanity (user-requested)
    try:
        import soundfile as sf
        import torch
        data = setup_paths()
        probe = find_root('selection_report.json', 'probe1 output')
        import pandas as pd
        df = pd.read_csv(os.path.join(probe, 'extraction.csv')).head(2)
        _worker_init(data)
        tmp_dir = os.environ.get('KAGGLE_TMP', '/kaggle/tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        sanity_pt = os.path.join(tmp_dir, 'sanity.pt')
        for _, r in df.iterrows():
            res = _worker_process((os.path.join(probe, 'wav', r.utt + '.wav'),
                                   r.transcript, sanity_pt))
            log(f'sanity feature: {res}')
        d = torch.load(sanity_pt, weights_only=True)
        log(f"sanity feature shape: ids={len(d['ids'])} "
            f"mel={tuple(d['mel'].shape)} pitch={tuple(d['pitch'].shape)}")
        # fresh model forward + one step
        m = build_scratch_model('cuda' if torch.cuda.is_available() else 'cpu')
        dev = next(m.parameters()).device
        ids = d['ids'][None].to(dev)
        mel = d['mel'][None].to(dev)
        pitch = d['pitch'][None].to(dev)
        out = m(text=ids, text_len=torch.LongTensor([len(d['ids'])]).to(dev),
                pitch=pitch, energy=None, spect=mel,
                spect_len=torch.LongTensor([mel.shape[2]]).to(dev),
                attn_prior=None, lm_tokens=None,
                speaker=torch.LongTensor([0]).to(dev),
                emotion=torch.LongTensor([0]).to(dev))
        log(f'sanity forward ok: spect={tuple(out[0].shape)}')
        m.eval()
        mel_o = m.infer(ids, pace=1.0, speaker=0, emotion=0)
        mm = mel_o.transpose(1, 2)[0].cpu().numpy()
        sf.write(os.path.join(WORK, 'sanity_random_init.wav'), mel_to_wav(mm),
                 22050, subtype='PCM_16')
        log('sanity random-init audio saved (expected: noise — untrained)')
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        log(f'SANITY FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        log('Aborting before wasting GPU hours.')
        sys.exit(1)

    env = dict(os.environ)
    env['KAGGLE_WORK'] = WORK
    env['SCRATCH_SESSION_T0'] = str(T0)

    # ---- process features if no mounted features exist ----
    have_features = bool(glob.glob(f'{INPUT}/**/features/index.json',
                                    recursive=True)) \
        or os.path.exists(os.path.join(FEAT_DIR, 'index.json'))
    if not have_features:
        log('no mounted features — running PROCESS phase')
        r0 = subprocess.run([sys.executable, os.path.abspath(__file__),
                             'process'], env=env)
        log(f'process exit code: {r0.returncode}')
        if r0.returncode != 0:
            log('PROCESS FAILED — trying generate from any existing ckpt, '
                'then aborting')
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            'generate'], env=env)
            sys.exit(1)
    else:
        log('features found (mounted or local) — skipping process phase')

    # ---- train ----
    r1 = subprocess.run([sys.executable, os.path.abspath(__file__), 'train'],
                        env=env)
    log(f'train exit code: {r1.returncode}')

    # ---- generate (ALWAYS) ----
    r2 = subprocess.run([sys.executable, os.path.abspath(__file__), 'generate'],
                        env=env)
    log(f'generate exit code: {r2.returncode}')

    files = sorted(glob.glob(os.path.join(WORK, '**', '*.*'), recursive=True))
    n_wav = sum(1 for f in files if f.endswith('.wav'))
    log(f'SUMMARY: files={len(files)} wavs={n_wav} '
        f'train_rc={r1.returncode} gen_rc={r2.returncode}')
    it = 0
    lp = os.path.join(WORK, 'train_log.csv')
    if os.path.exists(lp):
        lines = open(lp).read().strip().splitlines()
        if len(lines) > 1:
            it = lines[-1].split(',')[0]
    log(f'FINAL_ITER={it}')
    log('ALL_DONE')


if __name__ == '__main__':
    if PHASE == 'process':
        phase_process()
    elif PHASE == 'train':
        phase_train()
    elif PHASE == 'generate':
        phase_generate()
    else:
        driver()
