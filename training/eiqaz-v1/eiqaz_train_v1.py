# ============================================================================
# Eiqaz TTS v1 — TRAIN kernel (eiqaz-train-v1)  —  4h PILOT, FROM SCRATCH
# ============================================================================
# مشروع مستقل تمامًا عن "NileTTS 4h Train" (مرجع هندسي فقط).
#
# *** Scratch / Random Initialization ***
#   - لا Resume من أي checkpoint قديم إطلاقًا (لا states_79590.pth ولا غيره)
#   - لا تحميل أوزان سابقة في أي مرحلة — نماذج G وD تُنشأ عشوائية (seed 1234)
#   - الوحيدة المسموح تحميلها: مخرجات THIS kernel نفسها عند إعادة تشغيل
#     جلسة انهارت MID-RUN؟ — لا: حتى هذه ممنوعة في v1 (قرار المستخدم:
#     تجربة مستقلة نقية). أي انهيار = إعادة الجولة من الصفر.
#
# Mounts (READ-ONLY):
#   kernel eiqaz-prep-v1 output : features/ (ملامح مصري+فصحى + index.json)
#   dataset eiqaz-v1-inputs     : eqz_tokens.py + eiqaz_eval_set.json
#   dataset mixer-tts-scratch-data: mixer_repo + models/vocos22.onnx
#   (لا يُركَّب أي مخرج لمشروع قديم — لا states_79590.pth ولا ملامح قديمة)
#
# المزج (طلب المستخدم §5): 65% مصري / 35% فصحى **بالمدة** لا بعدد الملفات —
# حصة مدة لكل epoch: تدوير حلقي على كل جانب (تغطية كاملة للمصري عبر
# الـepochs + تكرار MSA بنسبة تساوي حصتها).
#
# الطب الأساسي (وصف المؤلف من المرجع، دون تغيير):
#   AdamW(1e-4, betas 0/0.99, wd 1e-6) · LSGAN PatchDiscriminatorCond من
#   iter 2000 · feature matching 1.0 / score 4.0 · bin loss 1.0 ·
#   batch 16 · max_frames 950 · max_tokens 160 · clip G=20 / D=1000
#   LR ثابت 1e-4 خلال نافذة الـ4h (نفس السلوك الفعلي لجولة المرجع:
#   المرساة 196k كانت فوق عدد iters الجولة فلم يحدث أي decay).
#
# Checkpoints (طلب المستخدم §14): بداية + منتصف + نهاية:
#   states_start.pth (أول حفظ) · states_mid.pth (منتصف الميزانية) ·
#   states.pth (متداول كل 500) · states_{it}.pth (لقطة نهاية)
#
# Phases (argv): driver | train | generate
# ============================================================================
import os
import sys
import json
import time
import random
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
CKPT_DIR = os.path.join(WORK, 'checkpoints')
for p in (WORK, CKPT_DIR):
    os.makedirs(p, exist_ok=True)

NET_CONFIG_OVERRIDES = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}
# --- إعدادات الجولة v1 ---
SPEAKERS_V1 = [int(x) for x in os.environ.get(
    'EQZ_SPEAKERS', '0,1,2,3,4,5,6').split(',')]   # 0/1 مصري + 2 ClArTTS + 3-6 CV
EGY_FRAC = float(os.environ.get('EQZ_EGY_FRAC', 0.65))   # حصة المدة
TRAIN_TIME_BUDGET = float(os.environ.get('TIME_BUDGET_S', 15300))  # 4h+هامش
TOTAL_TIME_CAP = 8.3 * 3600
GENERATE_RESERVE_S = 45 * 60
BATCH_SIZE = int(os.environ.get('SCRATCH_BATCH_SIZE', 16))
GAN_WARMUP = int(os.environ.get('GAN_WARMUP', 2000))
G_LR, D_LR = 1e-4, 1e-4
LR_MIN = 1e-5
LR_DECAY_ITERS = 150000
LR_ANCHOR_IT = 200000        # فوق أي iter في نافذة 4h → LR ثابت 1e-4 (مطابق
                             # السلوك الفعلي لجولة المرجع — موثق أعلاه)
FMATCH_W, SCORE_W = 1.0, 4.0
SAVE_EVERY = int(os.environ.get('SCRATCH_SAVE_EVERY', 500))
PROBE_EVERY = int(os.environ.get('SCRATCH_PROBE_EVERY', 2000))
LOG_EVERY = 25
TAR_LEN, MAX_FRAMES, MAX_TOKENS = 128, 950, 160
MAX_ITERS = int(os.environ.get('SCRATCH_MAX_ITERS', 400000))
SESSION_T0 = float(os.environ.get('SCRATCH_SESSION_T0', time.time()))


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
                            f'{INPUT}; entries: '
                            + repr(os.listdir(INPUT)))


def setup_paths():
    data = find_root('models/mixer128_pytorch.pth', 'mixer-tts-scratch-data')
    eqz = find_root('eqz_tokens.py', 'eiqaz-v1-inputs')
    for p in (os.path.join(data, 'mixer_repo'),
              os.path.join(data, 'tts_arabic_pkg'), eqz):
        if p not in sys.path:
            sys.path.insert(0, p)
    return data, eqz


def get_tokenizer():
    import eqz_tokens as ET
    return ET.get_tokenizers()


def load_eval_set():
    eqz = find_root('eqz_tokens.py', 'eiqaz-v1-inputs')
    return json.load(open(os.path.join(eqz, 'eiqaz_eval_set.json'),
                          encoding='utf-8'))


_VOCOS = None


def vocos_session():
    global _VOCOS
    if _VOCOS is None:
        data, _ = setup_paths()
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
    """نموذج جديد بالكامل — Random Initialization (لا أي تحميل أوزان)."""
    setup_paths()
    import torch
    from models.mixer_tts.mixer_tts import MixerTTSModel
    from models.mixer_tts import net_config
    net_config.update(NET_CONFIG_OVERRIDES)
    model = MixerTTSModel(**net_config)
    model.add_bin_loss = True
    model.bin_loss_scale = 1.0
    return model.to(device)


# ============================================================================
# PHASE: TRAIN — من الصفر، مزج 65/35 بالمدة
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

    # بوابة GPU (مضاعفة داخل الطور — حماية من تشغيل الطور مباشرة)
    if not torch.cuda.is_available():
        log('FATAL [train]: no GPU visible — refusing to train on CPU')
        sys.exit(2)
    device = 'cuda'
    log(f'[train] device={device} gpu={torch.cuda.get_device_name(0)}')

    def lr_at(current_it):
        progress = min(1.0, max(0.0,
                                (current_it - LR_ANCHOR_IT) / LR_DECAY_ITERS))
        import math
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return LR_MIN + (G_LR - LR_MIN) * cos_factor

    session_elapsed = time.time() - SESSION_T0
    budget = min(TRAIN_TIME_BUDGET,
                 TOTAL_TIME_CAP - session_elapsed - GENERATE_RESERVE_S)
    if budget <= 300:
        budget = 300
    MARGIN_S = min(900.0, budget * 0.1)
    log(f'[train] session elapsed {session_elapsed/60:.0f}m -> '
        f'train budget {budget/60:.0f}m')

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = True

    # ---------------- الملامح (مخرجات prep الخاصة بالمشروع) ----------------
    feat_root = os.path.join(WORK, 'features', 'index.json')
    if not os.path.exists(feat_root):
        cands = sorted(glob.glob(f'{INPUT}/**/features/index.json',
                                 recursive=True))
        if not cands:
            raise FileNotFoundError('no features/index.json — run '
                                    'eiqaz-prep-v1 first')
        feat_root = cands[0]
    fdir = os.path.dirname(feat_root)
    meta = json.load(open(feat_root))
    ps = {'mean': meta['pitch_mean'], 'std': meta['pitch_std']}

    dur_of = {}
    pool = {'egy': [], 'msa': []}
    val_pool = {'egy': [], 'msa': []}
    for e in meta['index']:
        dur_of[e['utt']] = e['n_frames'] * 256 / 22050
        side = 'egy' if e['dialect'] == 'egy' else 'msa'
        if (e.get('split', 'train') == 'train'
                and e['n_frames'] <= MAX_FRAMES
                and e['n_tokens'] <= MAX_TOKENS):
            pool[side].append((e['utt'], e['spk']))
        elif (e.get('split', 'train') != 'train'
              and e['n_frames'] <= MAX_FRAMES
              and e['n_tokens'] <= MAX_TOKENS):
            val_pool[side].append((e['utt'], e['spk']))
    h_egy = sum(dur_of[u] for u, _ in pool['egy']) / 3600
    h_msa = sum(dur_of[u] for u, _ in pool['msa']) / 3600
    items = pool['egy'] + pool['msa']
    val_items = val_pool['egy'] + val_pool['msa']
    log(f'[train] pool: {len(items)} train ({h_egy:.1f}h egy + '
        f'{h_msa:.1f}h msa = {100*h_egy/(h_egy+h_msa):.1f}% egy) / '
        f'{len(val_items)} val / {len(meta["index"])} features')
    log(f'[train] pitch mean={ps["mean"]:.1f} std={ps["std"]:.1f}')

    # ---- خطة المزج 65/35 بالمدة (حصة كل epoch بتدوير حلقي) ----
    E = min(h_egy / EGY_FRAC, h_msa / (1.0 - EGY_FRAC)) * 3600   # ثواني
    quota = {'egy': EGY_FRAC * E, 'msa': (1.0 - EGY_FRAC) * E}
    rings = {}
    for side in ('egy', 'msa'):
        random.shuffle(pool[side])
        rings[side] = {'list': [u for u, _ in pool[side]], 'pos': 0}
    log(f'[train] mixing plan: epoch={E/3600:.1f}h '
        f'(egy quota {quota["egy"]/3600:.1f}h + msa quota '
        f'{quota["msa"]/3600:.1f}h = {EGY_FRAC:.0%}/{1-EGY_FRAC:.0%})')

    spk_of = dict((u, s) for u, s in items)

    def build_epoch():
        epoch = []
        for side in ('egy', 'msa'):
            need = quota[side]
            got = 0.0
            r = rings[side]
            L = r['list']
            while got < need:
                u = L[r['pos']]
                epoch.append(u)
                got += dur_of[u]
                r['pos'] = (r['pos'] + 1) % len(L)
                if r['pos'] == 0:
                    random.shuffle(L)      # إعادة خلط عند الالتفاف الكامل
        random.shuffle(epoch)
        return epoch

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
        for utt in item_list:
            d = get_feat(utt)
            pitch = d['pitch']
            mask = pitch > 0
            pitch = torch.where(mask, (pitch - ps['mean']) / ps['std'],
                                torch.zeros_like(pitch))
            mel = d['mel']
            ids = d['ids']
            energy = torch.norm(mel.float(), dim=0, p=2)
            prior = torch.from_numpy(
                betabin(mel.size(1), len(ids))).float()
            data.append((ids, mel, pitch, energy, prior, spk_of[utt]))
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
            L = len(ids)
            T = mel.size(1)
            text_p[i, :L] = ids
            in_len[i] = L
            mel_p[i, :, :T] = mel
            out_len[i] = T
            pit_p[i, :T] = pitch
            ene_p[i, :T] = energy
            pri_p[i, :T, :L] = prior
            spk_t[i] = spk
        return (text_p.to(device), in_len.to(device), mel_p.to(device),
                out_len.to(device), pit_p.to(device), ene_p.to(device),
                pri_p.to(device), spk_t.to(device))

    # ---------------- النماذج — عشوائية بالكامل ----------------
    net_config.update(NET_CONFIG_OVERRIDES)
    model = MixerTTSModel(**net_config)
    model.add_bin_loss = True
    model.bin_loss_scale = 1.0
    n_params = sum(p.numel() for p in model.parameters())
    log(f'[train] FRESH model (random init): {n_params/1e6:.2f}M params')
    model = model.to(device)
    critic = PatchDiscriminatorCond(
        2, 32, d_emb=model.speaker_emb.embedding_dim).to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=G_LR, betas=(0.0, 0.99),
                              weight_decay=1e-6)
    opt_d = torch.optim.AdamW(critic.parameters(), lr=D_LR, betas=(0.0, 0.99),
                              weight_decay=1e-6)
    # تأكيد صريح: لا يوجد أي مسار resume — states_79590.pth لا يُركب أصلًا
    log('[train] RANDOM INITIALIZATION confirmed — no resume path exists '
        'in this kernel; old project checkpoints are NOT mounted')

    log_path = os.path.join(WORK, 'train_log.csv')
    val_log_path = os.path.join(WORK, 'val_log.csv')
    with open(log_path, 'w') as f:
        f.write('iter,phase,loss,mel,durs,pitch,ctc,bin,score,fmatch,'
                'loss_d,n_q,n_v,secs,it_per_s,skipped,lr\n')
    with open(val_log_path, 'w') as f:
        f.write('iter,loss,mel,durs,pitch,ctc,bin,secs\n')
    json.dump({
        'experiment': 'Eiqaz_TTS_v1_4h_Pilot',
        'initialization': 'RANDOM (scratch) — states_79590.pth NOT used, '
                          'NOT mounted, NOT loaded',
        'dataset': 'eiqaz-prep-v1 features (NileTTS+tashkeel-ai 65% + '
                   'msa-tts-data-v1 35% by duration)',
        'mix': {'egy_frac': EGY_FRAC, 'epoch_hours': round(E / 3600, 2),
                'egy_quota_h': round(quota['egy'] / 3600, 2),
                'msa_quota_h': round(quota['msa'] / 3600, 2)},
        'speakers': SPEAKERS_V1,
        'n_train_pool': len(items), 'n_val_pool': len(val_items),
        'hours': {'egy': round(h_egy, 2), 'msa': round(h_msa, 2)},
        'time_budget_s': TRAIN_TIME_BUDGET, 'batch_size': BATCH_SIZE,
        'gan_warmup': GAN_WARMUP, 'g_lr': G_LR, 'd_lr': D_LR,
        'fmatch_w': FMATCH_W, 'score_w': SCORE_W, 'tar_len': TAR_LEN,
        'lr_min': LR_MIN, 'lr_decay_iters': LR_DECAY_ITERS,
        'lr_anchor_it': LR_ANCHOR_IT, 'max_frames': MAX_FRAMES,
        'max_tokens': MAX_TOKENS, 'max_iters': MAX_ITERS,
        'n_params': n_params, 'pitch_mean': ps['mean'],
        'pitch_std': ps['std'], 'device': device,
        'tokenizer': 'eqz_tokens v1 (q_default 98 skel / g_default 22 skel '
                     '/ else hamza; markers {ق}/{ج}/{ء} override)',
    }, open(os.path.join(WORK, 'run_config.json'), 'w'), indent=1)

    state = {'saved_start': False, 'saved_mid': False}

    def save_states(it, name=None):
        torch.save({'model': model.state_dict(),
                    'critic': critic.state_dict(),
                    'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                    'iter': it, 'net_config': net_config,
                    'pitch_mean': ps['mean'], 'pitch_std': ps['std']},
                   os.path.join(CKPT_DIR, name or 'states.pth'))

    # ---------------- مسابر أثناء التدريب ----------------
    toks_ms, toks_egy, ids_of, _ = get_tokenizer()
    probe_texts = [
        'اِلْقَانُون اِلْمِصْرِي بِيِحْمِي اِلْحُقُوق.',        # قاف q_default
        'قَال لِي اِنْ اِلْوَلَد جَاي.',                      # قال همزة + جيم
    ]

    def gen_probe(it):
        try:
            import soundfile as sf
            model.eval()
            pdir = os.path.join(WORK, 'probes')
            os.makedirs(pdir, exist_ok=True)
            for si, text in enumerate(probe_texts):
                ids = ids_of(toks_egy(text))
                x = torch.LongTensor([ids]).to(device)
                for spk in SPEAKERS_V1[:3]:        # 0,1,2 (مصري+ClArTTS)
                    mel = model.infer(x, pace=1.0, speaker=spk, emotion=0)
                    m = mel.transpose(1, 2)[0].cpu().numpy()
                    sf.write(os.path.join(
                        pdir, f'it{it:06d}_s{si}_spk{spk}.wav'),
                        mel_to_wav(m), 22050, subtype='PCM_16')
            model.train()
            log(f'[train] probes saved at iter {it}')
        except Exception as e:                    # noqa: BLE001
            model.train()
            log(f'[train] probe FAILED (non-fatal): {type(e).__name__}: {e}')

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
                                      min(bstart + BATCH_SIZE,
                                          len(val_items))))
                    (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p,
                     spk_t) = make_batch([val_items[i][0] for i in bidx])
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
                    agg['loss'] += float(vloss)
                    agg['mel'] += float(vmel)
                    agg['durs'] += float(vdurs)
                    agg['pitch'] += float(vpitch)
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
                    f"pitch={row['pitch']:.3f} ctc={row['ctc']:.3f}")
        except Exception as e:                    # noqa: BLE001
            model.train()
            log(f'[val] FAILED (non-fatal): {type(e).__name__}: {e}')

    # ---------------- مراقبة الموارد ----------------
    def start_resource_monitor():
        import threading
        res_path = os.path.join(WORK, 'resource_log.csv')
        with open(res_path, 'w') as f:
            f.write('t,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,'
                    'ram_used_gb\n')

        def sample():
            try:
                gpu = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,'
                     'memory.used,memory.total',
                     '--format=csv,noheader,nounits'],
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
            except Exception:                     # noqa: BLE001
                pass

        class Monitor(threading.Thread):
            def run(self):
                while True:
                    sample()
                    time.sleep(60)
        Monitor(daemon=True).start()

    # ---------------- الحلقة ----------------
    model.train()
    critic.train()
    t_start = time.time()
    it = 0
    skipped = 0
    speed_smooth = 0.0
    q_seen = v_seen = 0
    start_resource_monitor()
    log(f'[train] start iter=0 (RANDOM INIT) budget={budget:.0f}s '
        f'pool={len(items)} gan_warmup={GAN_WARMUP}')

    stop = False
    while not stop and it < MAX_ITERS:
        epoch_utts = build_epoch()
        for bstart in range(0, len(epoch_utts), BATCH_SIZE):
            if time.time() - t_start >= budget - MARGIN_S:
                stop = True
                break
            if it >= MAX_ITERS:
                stop = True
                break
            if time.time() - SESSION_T0 >= TOTAL_TIME_CAP - \
                    GENERATE_RESERVE_S:
                log('[train] TOTAL session cap — saving + stopping')
                stop = True
                break
            t_it = time.time()
            try:
                batch_utts = epoch_utts[bstart:bstart + BATCH_SIZE]
                (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p,
                 spk_t) = make_batch(batch_utts)
                # مراقبة رموز القاف في الدفعات (تشخيص فقط)
                q_seen += int((text_p == 29).sum())
                v_seen += int((text_p == 37).sum())

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
                    true_text_len=in_len, true_pitch=pit_p,
                    true_energy=ene_p, true_spect=mel_p,
                    pred_spect=pred_spect, true_spect_len=out_len,
                    attn_logprob=attn_logprob, attn_soft=attn_soft,
                    attn_hard=attn_hard, attn_hard_dur=attn_hard_dur)

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
                opt_g.zero_grad()
                opt_d.zero_grad()
                skipped += 1
                continue

            dt = time.time() - t_it
            inst = 1.0 / max(dt, 1e-6)
            speed_smooth = (0.98 * speed_smooth + 0.02 * inst
                            if speed_smooth else inst)

            if it <= 6 or it % LOG_EVERY == 0:
                log(f"it {it} loss={loss.item():.3f} "
                    f"mel={mel_loss.item():.3f} "
                    f"durs={durs_loss.item():.3f} "
                    f"pitch={pitch_loss.item():.3f} "
                    f"ctc={ctc_loss.item():.3f} "
                    f"bin={bin_loss.item() if bin_loss is not None else 0:.3f}"
                    f" score={score_l.item():.3f} "
                    f"fmatch={fmatch_l.item():.3f} d={loss_d_val:.3f} "
                    f"lr={current_lr:.3e} | {speed_smooth:.2f}it/s "
                    f"skip={skipped} q_seen={q_seen} v_seen={v_seen} "
                    f"({(time.time()-t_start)/60:.1f}m)")
                with open(log_path, 'a') as f:
                    f.write(f"{it},{'warmup' if warmup else 'gan'},"
                            f"{loss.item():.5f},{mel_loss.item():.5f},"
                            f"{durs_loss.item():.5f},"
                            f"{pitch_loss.item():.5f},"
                            f"{ctc_loss.item():.5f},"
                            f"{bin_loss.item() if bin_loss is not None else 0:.5f},"
                            f"{score_l.item():.5f},{fmatch_l.item():.5f},"
                            f"{loss_d_val:.5f},{q_seen},{v_seen},"
                            f"{time.time()-t_start:.1f},"
                            f"{speed_smooth:.4f},{skipped},"
                            f"{current_lr:.6e}\n")

            if it % SAVE_EVERY == 0:
                save_states(it)
                if not state['saved_start']:
                    save_states(it, 'states_start.pth')
                    state['saved_start'] = True
                    log(f'[train] START checkpoint saved (iter {it})')
            if (not state['saved_mid']
                    and time.time() - t_start >= budget / 2):
                save_states(it, 'states_mid.pth')
                state['saved_mid'] = True
                log(f'[train] MID checkpoint saved (iter {it}, '
                    f'{(time.time()-t_start)/60:.0f}m)')
            if it % PROBE_EVERY == 0:
                gen_probe(it)
                eval_val(it)

    save_states(it, f'states_{it}.pth')
    if not state['saved_start']:
        save_states(it, 'states_start.pth')
    if not state['saved_mid']:
        save_states(it, 'states_mid.pth')
    try:
        log(f'[train] max GPU mem: '
            f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    except Exception:                             # noqa: BLE001
        pass
    log(f'[train] FINISHED at iter {it} '
        f'wall={time.strftime("%H:%M:%S", time.gmtime(time.time()-t_start))}'
        f' | q tokens seen={q_seen} v tokens seen={v_seen}')


# ============================================================================
# PHASE: GENERATE — مجموعة التقييم الثابتة لكل المتحدثين + F0 + منحنى
# ============================================================================
def critical_test(model, out_base):
    """الاختبار الحاسم (مواصفة المستخدم §8): نفس الكلمة مع العلامات
    الثلاث {ق}/{ج}/{ء} لكل متحدث — والتركيز على المتحدثين المصريين
    (0=SPEAKER_01 ذكر، 1=SPEAKER_02 أنثى): إذا أنتجا /q/ عند {ق} مع
    بقاء هويتهما المصرية، فالقاف الفصحى أصبحت خاصية قابلة للتحكم
    لا خاصية مرتبطة بمتحدثي MSA."""
    import torch
    import soundfile as sf
    toks_ms, toks_egy, ids_of, _ = get_tokenizer()
    device = next(model.parameters()).device
    cdir = os.path.join(out_base, 'critical_haqiqa')
    os.makedirs(cdir, exist_ok=True)
    forms = [
        # (اسم، نص، وضع الترميز) — عزلة بلا تشكيل + مشكول + جملة كاملة
        ('bare', 'حقيقة', 'egy'),
        ('q', 'حَقِيقَة{ق}', 'egy'),
        ('g', 'حَقِيقَة{ج}', 'egy'),
        ('h', 'حَقِيقَة{ء}', 'egy'),
        ('q_sent', 'دِي هِيَ اِلْحَقِيقَة{ق}.', 'egy'),
        ('g_sent', 'دِي هِيَ اِلْحَقِيقَة{ج}.', 'egy'),
        ('h_sent', 'دِي هِيَ اِلْحَقِيقَة{ء}.', 'egy'),
    ]
    manifest = []
    for spk in SPEAKERS_V1:
        for name, text, mode in forms:
            try:
                toks = toks_egy(text) if mode == 'egy' else toks_ms(text)
                ids = ids_of(toks)
                x = torch.LongTensor([ids]).to(device)
                with torch.inference_mode():
                    mel = model.infer(x, pace=1.0, speaker=spk, emotion=0)
                m = mel.transpose(1, 2)[0].cpu().numpy()
                fn = os.path.join(cdir, f'spk{spk}_haqiqa_{name}.wav')
                sf.write(fn, mel_to_wav(m), 22050, subtype='PCM_16')
                manifest.append({
                    'speaker': spk, 'form': name, 'text': text,
                    'q': int('q' in toks), 'v': int('v' in toks),
                    'hamza': int('<' in toks), 'n_tokens': len(ids),
                })
            except Exception as e:                # noqa: BLE001
                log(f'[critical] FAILED spk{spk} {name}: '
                    f'{type(e).__name__}: {e}')
    # تقرير التمييز الرمزي: العلامات الثلاث تعطي توكنات مختلفة فعلًا؟
    ok = {}
    for form, want in (('q', 'q'), ('g', 'v'), ('h', 'hamza')):
        rows = [m for m in manifest if m['form'] == form]
        ok[form] = all(m[want] == 1 for m in rows) if rows else False
    json.dump({'manifest': manifest, 'token_distinct': ok,
               'egyptian_speakers': [0, 1],
               'note': 'same word haqiqa with 3 markers — Egyptian '
                       'identity must persist across q/g/h'},
              open(os.path.join(cdir, 'critical_test.json'), 'w'),
              ensure_ascii=False, indent=1)
    log(f'[critical] haqiqa triple-test: token distinct '
        f'q={ok["q"]} g={ok["g"]} h={ok["h"]} '
        f'({len(manifest)} files for {len(SPEAKERS_V1)} speakers)')
    return cdir


def synth_eval(model, out_base, tag):
    """توليد مجموعة التقييم الثابتة كاملة لكل متحدثي v1."""
    import torch
    import soundfile as sf
    toks_ms, toks_egy, ids_of, _ = get_tokenizer()
    ev = load_eval_set()
    device = next(model.parameters()).device
    os.makedirs(out_base, exist_ok=True)
    manifest = []
    s = ev['sections']

    def synth(text, mode, out_path):
        try:
            toks = toks_egy(text) if mode == 'egy' else toks_ms(text)
            ids = ids_of(toks)
            if len(ids) > MAX_TOKENS:
                return None
            x = torch.LongTensor([ids]).to(device)
            with torch.inference_mode():
                mel = model.infer(x, pace=1.0, speaker=spk, emotion=0)
            m = mel.transpose(1, 2)[0].cpu().numpy()
            sf.write(out_path, mel_to_wav(m), 22050, subtype='PCM_16')
            return {'n_tokens': len(ids),
                    'q': int('q' in toks), 'v': int('v' in toks)}
        except Exception as e:                    # noqa: BLE001
            log(f'[gen{tag}] FAILED {out_path}: {type(e).__name__}: {e}')
            return None

    for spk in SPEAKERS_V1:
        spk_dir = os.path.join(out_base, f'speaker_{spk}')
        os.makedirs(spk_dir, exist_ok=True)
        n = 0
        for cat in ('egy_baseline', 'q_candidates', 'context_watch'):
            for item in s[cat]:
                for j, text in enumerate(item['sentences']):
                    r = synth(text, 'egy',
                              os.path.join(spk_dir, f'{cat}_{item["word"]}_{j}.wav'))
                    if r:
                        manifest.append({'speaker': spk, 'cat': cat,
                                         'word': item['word'], 'idx': j,
                                         'text': text, **r})
                        n += 1
        for item in s['explicit']:
            r = synth(item['text'], 'egy',
                      os.path.join(spk_dir,
                                   f'explicit_{item["tag"]}.wav'))
            if r:
                manifest.append({'speaker': spk, 'cat': 'explicit',
                                 'tag': item['tag'], 'text': item['text'],
                                 **r})
                n += 1
        for j, text in enumerate(s['msa']):
            r = synth(text, 'msa', os.path.join(spk_dir, f'msa_{j:02d}.wav'))
            if r:
                manifest.append({'speaker': spk, 'cat': 'msa', 'idx': j,
                                 'text': text, **r})
                n += 1
        for grp, words in s['words'].items():
            for j, w in enumerate(words):
                r = synth(w, 'egy', os.path.join(spk_dir,
                                                 f'word_{grp}_{j}.wav'))
                if r:
                    manifest.append({'speaker': spk, 'cat': 'word',
                                     'grp': grp, 'text': w, **r})
                    n += 1
        log(f'[gen{tag}] speaker_{spk}: {n} files')
    with open(os.path.join(out_base, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_base


def f0_report(sample_dir, out_json):
    import numpy as np
    import soundfile as sf
    import librosa
    out = {}
    for spk_dir in sorted(glob.glob(os.path.join(sample_dir, 'speaker_*'))):
        spk = os.path.basename(spk_dir)
        f0s = []
        for f in sorted(glob.glob(os.path.join(spk_dir, '*.wav')))[:40]:
            wav, sr = sf.read(f, dtype='float32')
            f0, _, _ = librosa.pyin(wav, sr=sr, fmin=60, fmax=400,
                                    frame_length=1024, hop_length=256)
            f0 = f0[~np.isnan(f0)]
            if f0.size > 30:
                f0s.append(float(np.median(f0)))
        med = float(np.median(f0s)) if f0s else 0.0
        out[spk] = {'synth_median_f0': round(med, 1), 'n_samples':
                    len(f0s)}
        log(f'[f0] {spk}: synth median={med:.1f}Hz ({len(f0s)} samples)')
    json.dump(out, open(out_json, 'w'), indent=1)


def phase_generate():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'[generate] device={device}')
    own = os.path.join(CKPT_DIR, 'states.pth')
    snaps = sorted(glob.glob(os.path.join(CKPT_DIR, 'states_*.pth')))
    cands = [p for p in ([own] + snaps) if os.path.exists(p)]
    if not cands:
        cands = sorted(glob.glob(f'{INPUT}/**/checkpoints/states.pth',
                                 recursive=True))
    if not cands:
        log('[generate] NO checkpoint — nothing to generate!')
        return

    def it_of(p):
        try:
            return torch.load(p, map_location='cpu',
                              weights_only=False).get('iter', 0)
        except Exception:                         # noqa: BLE001
            return 0
    best = max(cands, key=it_of)
    log(f'[generate] using checkpoint {best} (iter {it_of(best)})')
    st = torch.load(best, map_location='cpu', weights_only=False)
    model = build_scratch_model(device)
    model.load_state_dict(st['model'], strict=True)
    model.eval()
    critical_test(model, os.path.join(WORK, 'samples'))
    out = synth_eval(model, os.path.join(WORK, 'samples', 'eiqaz_v1_eval'),
                     ':new')
    f0_report(out, os.path.join(WORK, 'metrics', 'f0_eiqaz_v1.json'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        lp = os.path.join(WORK, 'train_log.csv')
        if os.path.exists(lp):
            df = pd.read_csv(lp)
            if len(df) > 2:
                fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                                         constrained_layout=True)
                for ax, col in zip(axes.flat, ['mel', 'durs', 'ctc',
                                               'loss_d']):
                    if col in df:
                        ax.plot(df.iter, df[col], lw=0.6)
                        ax.set_title(col)
                        ax.set_xlabel('iter')
                fig.savefig(os.path.join(WORK, 'training_curve.png'), dpi=120)
                log('[generate] training_curve.png saved')
    except Exception as e:                        # noqa: BLE001
        log(f'[generate] curve plot failed: {e}')
    log('[generate] DONE')


# ============================================================================
# PHASE: DRIVER — sanity ثم train ثم generate
# ============================================================================
def driver():
    log('=== Eiqaz TTS v1 TRAIN (from scratch, 4h pilot) ===')
    import torch
    log(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')
    # ---- بوابة GPU الصارمة (مواصفة المستخدم §1) ----
    # لا تدريب على CPU إطلاقًا: إذا لم يرَ PyTorch الـGPU نتوقف فورًا
    # قبل حرق أي وقت — الإصلاح يكون في إعدادات النواة (enable_gpu/T4).
    if not torch.cuda.is_available():
        log('FATAL: torch.cuda.is_available() == False — GPU NOT VISIBLE.')
        log('Aborting BEFORE training (user spec: no CPU fallback, '
            'no local session). Fix kernel GPU settings and re-push.')
        sys.exit(2)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        log(f'  gpu {i}: {name} (compute {cap[0]}.{cap[1]}) '
            f'cuda={torch.version.cuda}')
    gpu_name = torch.cuda.get_device_name(0)
    if 'T4' not in gpu_name:
        log(f'NOTE: requested NVIDIA T4 if available; got {gpu_name} '
            '(still a real GPU — proceeding per spec "T4 if available")')
    else:
        log(f'GPU CONFIRMED: NVIDIA T4 ({gpu_name})')
    json.dump({'gpu': gpu_name, 'cuda_available': True,
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'n_gpu': torch.cuda.device_count()},
              open(os.path.join(WORK, 'gpu_check.json'), 'w'), indent=1)
    try:
        import onnxruntime  # noqa
        log('onnxruntime available')
    except ImportError:
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                            'onnxruntime'], capture_output=True, text=True)
        log(f'onnxruntime install rc={r.returncode}')

    # ---- sanity (دخان Kaggle-side): ملامح + توكنات القاف + نموذج عشوائي ----
    torch.manual_seed(0)
    try:
        import soundfile as sf
        data, eqz = setup_paths()
        feat_root = None
        for c in ([os.path.join(WORK, 'features', 'index.json')] +
                  sorted(glob.glob(f'{INPUT}/**/features/index.json',
                                   recursive=True))):
            if os.path.exists(c):
                feat_root = c
                break
        assert feat_root, 'no features/index.json — run eiqaz-prep-v1 first'
        meta = json.load(open(feat_root))
        # بوابة الاكتمال (v1.1): لا تدريب على بيانات جزئية — المزج 65/35
        # بالمدة يفقد معناه إذا لم يكن الجانب المصري مكتملاً
        assert meta.get('complete') is True, (
            "prep INCOMPLETE (remaining=%s, expected=%s) — استأنف "
            'eiqaz-prep-v1 حتى complete=true قبل التدريب'
            % (meta.get('remaining'), meta.get('expected')))
        fdir = os.path.dirname(feat_root)
        e0 = next(e for e in meta['index'] if e['dialect'] == 'egy')
        e1 = next(e for e in meta['index'] if e['dialect'] == 'msa')
        for e in (e0, e1):
            d = torch.load(os.path.join(fdir, e['utt'] + '.pt'),
                           weights_only=True)
            log(f"sanity {e['dialect']} {e['utt']}: ids={len(d['ids'])} "
                f"mel={tuple(d['mel'].shape)} pitch={tuple(d['pitch'].shape)}"
                f" n_q={e.get('n_q')} spk={e['spk']}")
        toks_ms, toks_egy, ids_of, _ = get_tokenizer()
        tq = ids_of(toks_egy('اَلْقَانُون وَ اَلْقُرْآن قَطْعَة{ق} رَقَم{ج} قَال{ء}'))
        assert 29 in tq, 'q token (id 29) missing — tokenizer policy broken!'
        log(f'sanity qaf tokens: ids contain q(29)={29 in tq} '
            f'v(37)={37 in tq} hamza(9)={9 in tq}')
        m = build_scratch_model('cuda' if torch.cuda.is_available() else 'cpu')
        dev = next(m.parameters()).device
        d = torch.load(os.path.join(fdir, e0['utt'] + '.pt'),
                       weights_only=True)
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
        sf.write(os.path.join(WORK, 'sanity_random_init.wav'),
                 mel_to_wav(mm), 22050, subtype='PCM_16')
        log('sanity random-init audio saved (expected: noise — untrained)')
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:                        # noqa: BLE001
        import traceback
        log(f'SANITY FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        log('Aborting before wasting GPU hours.')
        sys.exit(1)

    env = dict(os.environ)
    env['KAGGLE_WORK'] = WORK
    env['SCRATCH_SESSION_T0'] = str(T0)
    r1 = subprocess.run([sys.executable, os.path.abspath(__file__), 'train'],
                        env=env)
    log(f'train exit code: {r1.returncode}')
    r2 = subprocess.run([sys.executable, os.path.abspath(__file__), 'generate'],
                        env=env)
    log(f'generate exit code: {r2.returncode}')
    lp = os.path.join(WORK, 'train_log.csv')
    it = 0
    if os.path.exists(lp):
        lines = open(lp).read().strip().splitlines()
        if len(lines) > 1:
            it = lines[-1].split(',')[0]
    log(f'FINAL_ITER={it}')
    log('ALL_DONE')


if __name__ == '__main__':
    if PHASE == 'train':
        phase_train()
    elif PHASE == 'generate':
        phase_generate()
    else:
        driver()
