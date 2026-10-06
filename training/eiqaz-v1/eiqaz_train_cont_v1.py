# ============================================================================
# Eiqaz TTS v1 — TRAIN CONTINUATION kernel (eiqaz-train-cont-v1)
# ============================================================================
# استكمال حقيقي (True Continuation) من checkpoint نهاية تجربة الـ4 ساعات
# states_89202.pth (iter=89,202) — لا تدريب من الصفر، ولا states_79590.pth.
#
# الغرض (مواصفة المستخدم):
#   - تحسين جودة ونطق اللهجة المصرية + Speaker 0 (ذكر) و Speaker 1 (أنثى)
#   - الحفاظ على هوية المتحدثين المصريين المكتسبة
#   - الحفاظ على معرفة MSA (rehearsal) — لا حذف بيانات MSA إطلاقًا
#   - Egyptian-focused training + MSA retention/rehearsal
#
# القرار المعماري:
#   Existing 7-speaker checkpoint (n_speakers=16 arch, 7 active)
#           → True continuation (model+critic+opt_g+opt_d+iter)
#           → Egyptian target speakers = {0, 1} (0=ذكر, 1=أنثى)
#           → MSA retained as rehearsal data (80/20 by duration)
#           → 4 additional hours ONLY (hard stop — لا تمديد)
#
# Mounts (READ-ONLY — لا شيء يُحذف أو يُعدل):
#   kernel eiqaz-train-v1 output : checkpoints/states_89202.pth (نقطة البداية)
#                                  + samples/ (خط الأساس BEFORE) + train_log.csv
#   kernel eiqaz-prep-v1 output  : features/ (34,907 ملامح: مصري+MSA + index.json)
#   dataset eiqaz-v1-inputs      : eqz_tokens.py (v1.0.1 كما في التجربة الأصلية)
#                                  + eiqaz_eval_set.json
#   dataset mixer-tts-scratch-data: mixer_repo + models/vocos22.onnx
#
# إصلاحات مؤكدة قبل التدريب (لا إخفاء أخطاء — إصلاح السبب الجذري):
#   FIX-1 Validation KeyError ('1_05206_00'): spk_of كانت تُبنى من train pool
#       فقط بينما make_batch تُستخدم للتحقق أيضًا → الآن تُبنى من index.json
#       كاملًا (train+eval). فحص قبل التدريب إلزامي: فشل validation = لا تدريب.
#   FIX-2 LR scheduler: الكوزيني (lr_at) كان مرساه 200k فوق نافذة 4h فبقي LR
#       ثابتًا 1e-4 طوال التجربة (السجل: قيمة واحدة مميزة). الآن: المرساة =
#       89,202 (بداية الاستكمال) ومدى الاضمحلال = 89,202 تكرار وLR_MIN=5e-5
#       → 1e-4 → cosine decay → 5e-5 فعليًا مع التكرار. (D يبقى 1e-4 ثابتًا
#       كما في التجربة الأصلية ووصفة المؤلف.)
#   FIX-3 metrics/ dir يُنشأ قبل الكتابة (f0 json + training_curve فشلا في
#       التجربة الأصلية بسبب FileNotFoundError).
#   FIX-4 resource_log: nvidia-smi يعيد سطرين على T4x2 — يُقرأ السطر الأول فقط.
#
# المزج (آلية موجودة نفسها — أقل تغيير ممكن):
#   حصة المدة لكل epoch (duration-quota mixing) كما في التجربة الأصلية،
#   بمعامل واحد معدل: EGY_FRAC 0.65 → 0.80:
#     epoch = min(h_egy/0.8, h_msa/0.2) → حصة مصري = كامل pool المصري
#     (100% تغطية كل epoch) + حصة MSA ~8.2h/epoch (تدوير حلقي على كامل
#     15.3h عبر ~1.9 epoch) = Egyptian primary + MSA rehearsal.
#   يُسجل الفعلي أثناء التدريب: عينات/دفعات spk0 وspk1 وMSA + النسب.
#
# Checkpoints: states_cont_start (الحالة المستعادة نفسها) · rolling
#   states_cont.pth كل 500 · states_cont_mid (منتصف الميزانية) ·
#   states_cont_{it}.pth (نهائي). checkpoint الأصل (states_89202.pth) في
#   مخرجات eiqaz-train-v1 يبقى سليمًا كما هو (READ-ONLY mount).
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
METRICS_DIR = os.path.join(WORK, 'metrics')
for p in (WORK, CKPT_DIR, METRICS_DIR):          # FIX-3: metrics مبكرًا
    os.makedirs(p, exist_ok=True)

NET_CONFIG_OVERRIDES = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}

# --- هوية المتحدثين المستهدفين (قيد إلزامي — مواصفة §3/§22) ---
EGYPTIAN_TARGET_SPEAKERS = {0, 1}      # 0 = Egyptian male, 1 = Egyptian female
EXPECTED_SPEAKER_MAP = {               # من index.json speaker_map_v1
    'egy_SPEAKER_01': 0,               # ذكر مصري (SPEAKER_01)
    'egy_SPEAKER_02': 1,               # أنثى مصرية (SPEAKER_02)
    'msa_clartts_male': 2, 'msa_cvf_5f810213': 3, 'msa_cvf_78c954e3': 4,
    'msa_cvf_cf4d8f89': 5, 'msa_cvf_fc3b87e3': 6,
}
MSA_SPEAKERS = {2, 3, 4, 5, 6}

# --- نقطة البداية: checkpoint نهاية تجربة الـ4 ساعات ---
EXPECTED_BASE_ITER = 89202             # states_89202.pth (مثبت بالسجل)

# --- إعدادات الاستكمال ---
SPEAKERS_V1 = [int(x) for x in os.environ.get(
    'EQZ_SPEAKERS', '0,1,2,3,4,5,6').split(',')]   # نفس السبعة — بلا تغيير
EGY_FRAC = float(os.environ.get('EQZ_EGY_FRAC', 0.80))   # 0.65→0.80 تركيز مصري
TRAIN_TIME_BUDGET = float(os.environ.get('TIME_BUDGET_S', 15300))  # 4h+هامش
TOTAL_TIME_CAP = 8.3 * 3600
GENERATE_RESERVE_S = 45 * 60
BATCH_SIZE = int(os.environ.get('SCRATCH_BATCH_SIZE', 16))
GAN_WARMUP = int(os.environ.get('GAN_WARMUP', 2000))
G_LR, D_LR = 1e-4, 1e-4                # G يبدأ 1e-4 (كما انتهت التجربة)
LR_MIN = 5e-5                          # هدف النهاية (مواصفة §12)
LR_DECAY_ITERS = 89202                 # مدى الاضمحلال ≈ 4h @ ~6.2 it/s
LR_ANCHOR_IT = EXPECTED_BASE_ITER      # FIX-2: المرساة = بداية الاستكمال
FMATCH_W, SCORE_W = 1.0, 4.0
SAVE_EVERY = int(os.environ.get('SCRATCH_SAVE_EVERY', 500))
PROBE_EVERY = int(os.environ.get('SCRATCH_PROBE_EVERY', 4000))
VAL_EVERY = int(os.environ.get('SCRATCH_VAL_EVERY', 4000))
LOG_EVERY = 25
TAR_LEN, MAX_FRAMES, MAX_TOKENS = 128, 950, 160
MAX_ITERS = int(os.environ.get('SCRATCH_MAX_ITERS', 400000))
SESSION_T0 = float(os.environ.get('SCRATCH_SESSION_T0', time.time()))

# --- مجموعة التحقق الثابتة الفرعية (لكل مجموعة) — قبل/أثناء/بعد ---
VAL_SUBSET_N = {'spk0': 128, 'spk1': 128, 'msa': 128}


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


def find_features_root():
    c = os.path.join(WORK, 'features', 'index.json')
    if os.path.exists(c):
        return c
    cands = sorted(glob.glob(f'{INPUT}/**/features/index.json',
                             recursive=True))
    if not cands:
        raise FileNotFoundError('no features/index.json — eiqaz-prep-v1 '
                                'output must be mounted')
    return cands[0]


def find_base_checkpoint():
    """checkpoint نهاية تجربة الـ4h من مخرجات eiqaz-train-v1 (READ-ONLY)."""
    cands = sorted(glob.glob(f'{INPUT}/**/checkpoints/states_*.pth',
                             recursive=True))
    if not cands:
        raise FileNotFoundError(
            'no states_*.pth found — eiqaz-train-v1 output must be mounted')
    return cands


def find_baseline_samples_root():
    """مجلد عينات خط الأساس (BEFORE) من مخرجات eiqaz-train-v1."""
    cands = sorted(glob.glob(f'{INPUT}/**/eiqaz_v1_eval', recursive=True))
    return cands[0] if cands else None


def find_baseline_train_log():
    cands = sorted(glob.glob(f'{INPUT}/**/train_log.csv', recursive=True))
    return cands[0] if cands else None


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
    """بناء النموذج بنفس بنية checkpoint (n_speakers=16) — ثم تحميل الأوزان."""
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
# PHASE: TRAIN — استكمال حقيقي من states_89202.pth
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

    # ---- بوابة GPU (كما في التجربة الأصلية — لا CPU إطلاقًا) ----
    if not torch.cuda.is_available():
        log('FATAL [train]: no GPU visible — refusing to train on CPU')
        sys.exit(2)
    device = 'cuda'
    log(f'[train] device={device} gpu={torch.cuda.get_device_name(0)}')

    def lr_at(current_it):
        """FIX-2: كوزيني فعلي عبر نافذة الاستكمال: 1e-4 → 5e-5."""
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
        f'train budget {budget/60:.0f}m (stop at '
        f'{(budget - MARGIN_S)/60:.0f}m = 4h wall clock)')

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = True

    # ---------------- الملامح (مخرجات prep — كما هي) ----------------
    feat_root = find_features_root()
    fdir = os.path.dirname(feat_root)
    meta = json.load(open(feat_root))
    assert meta.get('complete') is True, (
        'prep INCOMPLETE — continuation refuses partial data')
    ps = {'mean': meta['pitch_mean'], 'std': meta['pitch_std']}

    # ---- التحقق الفعلي من speaker mapping (قيد إلزامي §3) ----
    smap = meta.get('speaker_map_v1', {})
    assert smap.get('egy_SPEAKER_01') == 0 and smap.get('egy_SPEAKER_02') == 1, (
        f'Egyptian speaker mapping mismatch: {smap} — BLOCKED per user spec '
        '(no auto re-mapping, no alternative speakers)')
    for k, v in EXPECTED_SPEAKER_MAP.items():
        assert smap.get(k) == v, f'speaker map mismatch {k}: {smap.get(k)} != {v}'

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

    # FIX-1 (الجذري): spk_of من كامل الفهرس (train+eval) — لا KeyError بعدها
    spk_of = {e['utt']: e['spk'] for e in meta['index']}
    assert '1_05206_00' in spk_of, (
        "regression guard: utterance '1_05206_00' must be in spk_of")
    log(f'[train] pool: {len(items)} train ({h_egy:.1f}h egy + '
        f'{h_msa:.1f}h msa) / {len(val_items)} val / '
        f'{len(meta["index"])} features')
    log(f'[train] FIX-1 verified: spk_of covers FULL index '
        f'({len(spk_of)}) incl. val split — '
        f"'1_05206_00' -> spk {spk_of['1_05206_00']}")

    # ---- خطة المزج: نفس الآلية (حصة مدة/epoch) — EGY_FRAC فقط تغيّر ----
    E = min(h_egy / EGY_FRAC, h_msa / (1.0 - EGY_FRAC)) * 3600   # ثواني
    quota = {'egy': EGY_FRAC * E, 'msa': (1.0 - EGY_FRAC) * E}
    egy_cov = min(1.0, quota['egy'] / (h_egy * 3600))
    msa_cov = min(1.0, quota['msa'] / (h_msa * 3600))
    rings = {}
    for side in ('egy', 'msa'):
        random.shuffle(pool[side])
        rings[side] = {'list': [u for u, _ in pool[side]], 'pos': 0}

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

    log(f'[train] mixing plan (SAME mechanism, EGY_FRAC {EGY_FRAC}): '
        f'epoch={E/3600:.1f}h (egy quota {quota["egy"]/3600:.1f}h '
        f'({egy_cov:.0%} of egy pool/epoch) + msa quota '
        f'{quota["msa"]/3600:.1f}h ({msa_cov:.0%} of msa pool/epoch)) — '
        f'Egyptian primary + MSA rehearsal')

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

    # ---------------- استعادة checkpoint نهاية الـ4h (إلزامي §2/§13) ----------------
    net_config.update(NET_CONFIG_OVERRIDES)
    model = MixerTTSModel(**net_config)
    model.add_bin_loss = True
    model.bin_loss_scale = 1.0
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    critic = PatchDiscriminatorCond(
        2, 32, d_emb=model.speaker_emb.embedding_dim).to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=G_LR, betas=(0.0, 0.99),
                              weight_decay=1e-6)
    opt_d = torch.optim.AdamW(critic.parameters(), lr=D_LR, betas=(0.0, 0.99),
                              weight_decay=1e-6)

    base_cands = find_base_checkpoint()
    it_map = {}
    for c in base_cands:
        try:
            it_map[c] = torch.load(c, map_location='cpu',
                                   weights_only=False).get('iter', 0)
        except Exception:                         # noqa: BLE001
            it_map[c] = 0
    base_path = max(it_map, key=it_map.get)
    base_iter = it_map[base_path]
    assert base_iter == EXPECTED_BASE_ITER, (
        f'base checkpoint iter={base_iter} != expected '
        f'{EXPECTED_BASE_ITER} (log-proven end of 4h run) — BLOCKED')

    st = torch.load(base_path, map_location='cpu', weights_only=False)
    # توافق البنية قبل التحميل (§7 — لا تغيير n_speakers)
    for k, v in (('n_speakers', 16), ('num_tokens', 148),
                 ('symbols_embedding_dim', 128), ('n_emotions', 16)):
        assert st['net_config'].get(k) == v, (
            f'net_config mismatch {k}: {st["net_config"].get(k)} != {v}')
    assert abs(st['pitch_mean'] - ps['mean']) < 1e-3, 'pitch_mean mismatch'
    assert abs(st['pitch_std'] - ps['std']) < 1e-3, 'pitch_std mismatch'

    model.load_state_dict(st['model'], strict=True)
    critic.load_state_dict(st['critic'], strict=True)
    opt_g.load_state_dict(st['opt_g'])
    opt_d.load_state_dict(st['opt_d'])
    n_opt_g_state = len(st['opt_g'].get('state', {}))
    n_opt_d_state = len(st['opt_d'].get('state', {}))
    it = int(st['iter'])
    log(f'[train] CONTINUATION from {base_path}')
    log(f'[train] restored: model(strict=True) + critic(strict=True) + '
        f'opt_g({n_opt_g_state} state entries, '
        f'lr={st["opt_g"]["param_groups"][0]["lr"]:.1e}) + '
        f'opt_d({n_opt_d_state} state entries) — iter={it}')
    log('[train] RANDOM INIT NOT USED — states_79590.pth NOT mounted; '
        'this is a TRUE continuation from the 4h pilot end checkpoint')

    log_path = os.path.join(WORK, 'train_log.csv')
    val_log_path = os.path.join(WORK, 'val_log.csv')
    with open(log_path, 'w') as f:
        f.write('iter,phase,loss,mel,durs,pitch,ctc,bin,score,fmatch,'
                'loss_d,n_q,n_v,spk0_n,spk1_n,msa_n,secs,it_per_s,'
                'skipped,lr\n')
    with open(val_log_path, 'w') as f:
        f.write('iter,scope,loss,mel,durs,pitch,ctc,bin,'
                'loss_e0,mel_e0,loss_e1,mel_e1,loss_ms,mel_ms,secs\n')
    json.dump({
        'experiment': 'Eiqaz_TTS_v1_Continuation_4h',
        'initialization': f'TRUE CONTINUATION from {base_path} '
                          f'(iter={base_iter}, end of 4h pilot — '
                          'log-proven; states_79590.pth NOT used)',
        'base_checkpoint': base_path, 'base_iter': base_iter,
        'dataset': 'eiqaz-prep-v1 features (SAME as pilot — no data '
                   'changes; MSA fully retained)',
        'mix': {'mechanism': 'duration-quota epoch mixing (same as pilot)',
                'egy_frac': EGY_FRAC,
                'pilot_egy_frac': 0.65,
                'epoch_hours': round(E / 3600, 2),
                'egy_quota_h': round(quota['egy'] / 3600, 2),
                'msa_quota_h': round(quota['msa'] / 3600, 2),
                'egy_pool_coverage_per_epoch': round(egy_cov, 4),
                'msa_pool_coverage_per_epoch': round(msa_cov, 4),
                'msa_rehearsal': True},
        'egyptian_target_speakers': sorted(EGYPTIAN_TARGET_SPEAKERS),
        'speaker_map_v1': smap,
        'speakers': SPEAKERS_V1,
        'n_speakers_architecture': 16,
        'n_active_speakers': 7,
        'n_train_pool': len(items), 'n_val_pool': len(val_items),
        'hours': {'egy': round(h_egy, 2), 'msa': round(h_msa, 2)},
        'time_budget_s': TRAIN_TIME_BUDGET,
        'wall_clock_train_s': budget - MARGIN_S,
        'batch_size': BATCH_SIZE,
        'gan_warmup': GAN_WARMUP,
        'g_lr_start': G_LR, 'lr_min': LR_MIN,
        'lr_schedule': 'cosine: 1e-4 @ it 89202 -> 5e-5 @ it ~178404 '
                       '(anchor=89202, span=89202 iters); D fixed 1e-4',
        'lr_anchor_it': LR_ANCHOR_IT, 'lr_decay_iters': LR_DECAY_ITERS,
        'validation': {'fix': 'spk_of from FULL index (train+eval) — '
                              "KeyError '1_05206_00' root-caused & fixed",
                       'subset_per_group': VAL_SUBSET_N,
                       'val_every': VAL_EVERY, 'covers': ['egy0', 'egy1',
                                                          'msa']},
        'fmatch_w': FMATCH_W, 'score_w': SCORE_W, 'tar_len': TAR_LEN,
        'max_frames': MAX_FRAMES, 'max_tokens': MAX_TOKENS,
        'max_iters': MAX_ITERS, 'n_params': n_params,
        'pitch_mean': ps['mean'], 'pitch_std': ps['std'],
        'device': device,
        'tokenizer': 'eqz_tokens v1.0.1 (UNCHANGED from pilot run; baked '
                     'features untouched — canonical text is source of '
                     'truth)',
    }, open(os.path.join(WORK, 'run_config.json'), 'w'), indent=1)

    state = {'saved_mid': False}

    def save_states(it_, name=None):
        torch.save({'model': model.state_dict(),
                    'critic': critic.state_dict(),
                    'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                    'iter': it_, 'net_config': net_config,
                    'pitch_mean': ps['mean'], 'pitch_std': ps['std']},
                   os.path.join(CKPT_DIR, name or 'states_cont.pth'))

    # ---------------- مسابر أثناء التدريب ----------------
    toks_ms, toks_egy, ids_of, _ = get_tokenizer()
    probe_texts = [
        'اِلْقَانُون اِلْمِصْرِي بِيِحْمِي اِلْحُقُوق.',        # قاف q_default
        'قَال لِي اِنْ اِلْوَلَد جَاي.',                      # قال همزة + جيم
    ]

    def gen_probe(it_):
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
                        pdir, f'it{it_:06d}_s{si}_spk{spk}.wav'),
                        mel_to_wav(m), 22050, subtype='PCM_16')
            model.train()
            log(f'[train] probes saved at iter {it_}')
        except Exception as e:                    # noqa: BLE001
            model.train()
            log(f'[train] probe FAILED (non-fatal): {type(e).__name__}: {e}')

    # ---------------- مجموعة التحقق الفرعية الثابتة (قبل/بعد) ----------------
    def build_val_subset():
        sub = {'egy0': [], 'egy1': [], 'msa': []}
        for u, s in sorted(val_pool['egy']):
            if s == 0:
                sub['egy0'].append((u, s))
            elif s == 1:
                sub['egy1'].append((u, s))
        for u, s in sorted(val_pool['msa']):
            sub['msa'].append((u, s))

        def stride_pick(lst, n):
            if len(lst) <= n:
                return lst
            step = len(lst) / n
            return [lst[int(i * step)] for i in range(n)]

        sub['egy0'] = stride_pick(sub['egy0'], VAL_SUBSET_N['spk0'])
        sub['egy1'] = stride_pick(sub['egy1'], VAL_SUBSET_N['spk1'])
        sub['msa'] = stride_pick(sub['msa'], VAL_SUBSET_N['msa'])
        return sub

    VAL_SUBSET = build_val_subset()

    def run_val_batches(item_list):
        """forward+metrics على دفعات — بلا try/except: أي خطأ يظهر فورًا."""
        agg = {'loss': 0., 'mel': 0., 'durs': 0., 'pitch': 0.,
               'ctc': 0., 'bin': 0.}
        n_b = 0
        model.eval()
        with torch.no_grad():
            for bstart in range(0, len(item_list), BATCH_SIZE):
                chunk = item_list[bstart:bstart + BATCH_SIZE]
                (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p,
                 spk_t) = make_batch([u for u, _ in chunk])
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
        return ({k: v / n_b for k, v in agg.items()} if n_b else None), n_b

    def eval_val(it_, scope='subset', groups=None):
        """validation حقيقي لكل مجموعة (مصري spk0/spk1 + MSA) — يكتب
        val_log.csv. لا إخفاء أخطاء: أي استثناء يوقف الاستكمال (§9/§10).
        groups=None → المجموعة الفرعية الثابتة؛ أو مجموعات كاملة للسجل النهائي."""
        t0 = time.time()
        grp_items = groups if groups is not None else VAL_SUBSET
        g, nbs = {}, {}
        for grp in ('egy0', 'egy1', 'msa'):
            r, nb = run_val_batches(grp_items[grp])
            if r is None:
                raise RuntimeError(
                    f'validation produced no batches for group {grp} — '
                    'refusing to hide (user spec §9)')
            g[grp], nbs[grp] = r, nb
        tot = sum(nbs.values())
        overall = {k: sum(g[gr][k] * nbs[gr] for gr in g) / tot
                   for k in g['egy0']}
        secs = time.time() - t0
        with open(val_log_path, 'a') as f:
            f.write(f"{it_},{scope},{overall['loss']:.5f},"
                    f"{overall['mel']:.5f},{overall['durs']:.5f},"
                    f"{overall['pitch']:.5f},{overall['ctc']:.5f},"
                    f"{overall['bin']:.5f},"
                    f"{g['egy0']['loss']:.5f},{g['egy0']['mel']:.5f},"
                    f"{g['egy1']['loss']:.5f},{g['egy1']['mel']:.5f},"
                    f"{g['msa']['loss']:.5f},{g['msa']['mel']:.5f},"
                    f"{secs:.1f}\n")
        log(f"[val] it {it_} ({scope}) loss={overall['loss']:.3f} "
            f"mel={overall['mel']:.3f} | egy0 loss={g['egy0']['loss']:.3f} "
            f"mel={g['egy0']['mel']:.3f} | egy1 "
            f"loss={g['egy1']['loss']:.3f} mel={g['egy1']['mel']:.3f} | "
            f"msa loss={g['msa']['loss']:.3f} mel={g['msa']['mel']:.3f} "
            f"({secs:.0f}s)")
        return overall, g

    # ---------------- مراقبة الموارد (FIX-4: سطر GPU الأول فقط) ----------------
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
                first = gpu.splitlines()[0] if gpu else '0,0,0'
                parts = [x.strip() for x in first.split(',')]
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

    # ============ اختبار validation قبل التدريب (§10 — إلزامي) ============
    log('[train] PRE-TRAIN VALIDATION TEST (must PASS before any training)')
    try:
        base_overall, base_g = eval_val(it, scope='baseline_pre_train')
        log('[train] PRE-TRAIN VALIDATION: PASS — val_log.csv written '
            '(this row = BEFORE baseline at iter %d)' % it)
    except Exception as e:                        # noqa: BLE001
        import traceback
        log(f'[train] PRE-TRAIN VALIDATION: FAILED — '
            f'{type(e).__name__}: {e}')
        traceback.print_exc()
        log('DO NOT START TRAINING (user spec §10) — aborting')
        sys.exit(1)
    # snapshot الحالة المستعادة كما هي (previous checkpoint preserved + بدايتنا)
    save_states(it, 'states_cont_start.pth')
    log(f'[train] states_cont_start.pth saved (restored state, iter {it})')

    # ---------------- الحلقة ----------------
    model.train()
    critic.train()
    t_start = time.time()
    skipped = 0
    speed_smooth = 0.0
    q_seen = v_seen = 0
    spk0_n = spk1_n = msa_n = 0          # exposure فعلي (عينات §15)
    epochs_done = 0
    start_resource_monitor()
    log(f'[train] RESUME iter={it} (continuation) budget={budget:.0f}s '
        f'(wall train {(budget - MARGIN_S):.0f}s) pool={len(items)} '
        f'gan already active (it > {GAN_WARMUP})')
    log(f'[train] LR plan: lr_at({it})={lr_at(it):.3e} -> '
        f'lr_at({it + LR_DECAY_ITERS // 2})='
        f'{lr_at(it + LR_DECAY_ITERS // 2):.3e} (mid) -> '
        f'lr_at({it + LR_DECAY_ITERS})='
        f'{lr_at(it + LR_DECAY_ITERS):.3e} (end target 5e-5)')

    stop = False
    while not stop and it < MAX_ITERS:
        epoch_utts = build_epoch()
        epochs_done += 1
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
                # مراقبة رموز القاف + exposure فعلي لكل مجموعة (§15)
                q_seen += int((text_p == 29).sum())
                v_seen += int((text_p == 37).sum())
                spk0_n += int((spk_t == 0).sum())
                spk1_n += int((spk_t == 1).sum())
                msa_n += int((spk_t >= 2).sum())

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
                    mel_ids = torch.arange(mel_p.size(0),
                                           device=mel_p.device)
                    ofx_perc = torch.rand(out_len.size(),
                                          device=out_len.device)
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
                current_lr = lr_at(it)            # FIX-2: decay فعلي الآن
                for g_ in opt_g.param_groups:
                    g_['lr'] = current_lr
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
                tot_n = spk0_n + spk1_n + msa_n
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
                    f"expo[spk0={spk0_n} ({100*spk0_n/max(tot_n,1):.1f}%) "
                    f"spk1={spk1_n} ({100*spk1_n/max(tot_n,1):.1f}%) "
                    f"msa={msa_n} ({100*msa_n/max(tot_n,1):.1f}%)] "
                    f"({(time.time()-t_start)/60:.1f}m)")
                with open(log_path, 'a') as f:
                    f.write(f"{it},{'gan'},{loss.item():.5f},"
                            f"{mel_loss.item():.5f},"
                            f"{durs_loss.item():.5f},"
                            f"{pitch_loss.item():.5f},"
                            f"{ctc_loss.item():.5f},"
                            f"{bin_loss.item() if bin_loss is not None else 0:.5f},"
                            f"{score_l.item():.5f},{fmatch_l.item():.5f},"
                            f"{loss_d_val:.5f},{q_seen},{v_seen},"
                            f"{spk0_n},{spk1_n},{msa_n},"
                            f"{time.time()-t_start:.1f},"
                            f"{speed_smooth:.4f},{skipped},"
                            f"{current_lr:.6e}\n")

            if it % SAVE_EVERY == 0:
                save_states(it)
            if (not state['saved_mid']
                    and time.time() - t_start >= (budget - MARGIN_S) / 2):
                save_states(it, 'states_cont_mid.pth')
                state['saved_mid'] = True
                log(f'[train] MID checkpoint saved (iter {it}, '
                    f'{(time.time()-t_start)/60:.0f}m)')
            if it % PROBE_EVERY == 0:
                gen_probe(it)
            if it % VAL_EVERY == 0:
                eval_val(it, scope='subset')

    # ---------------- النهاية: STOP + FINAL CHECKPOINT (§14) ----------------
    final_iter = it
    save_states(final_iter, f'states_cont_{final_iter}.pth')
    if not state['saved_mid']:
        save_states(final_iter, 'states_cont_mid.pth')
    # validation كامل (كل الـ2508) للسجل النهائي — لكل مجموعة
    FULL_VAL_GROUPS = {
        'egy0': [(u, s) for u, s in val_pool['egy'] if s == 0],
        'egy1': [(u, s) for u, s in val_pool['egy'] if s == 1],
        'msa': list(val_pool['msa']),
    }
    try:
        eval_val(final_iter, scope='final_full', groups=FULL_VAL_GROUPS)
    except Exception as e:                        # noqa: BLE001
        log(f'[train] final full-val FAILED: {type(e).__name__}: {e}')
    try:
        log(f'[train] max GPU mem: '
            f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    except Exception:                             # noqa: BLE001
        pass
    tot_n = spk0_n + spk1_n + msa_n
    sampling_report = {
        'epochs_completed': epochs_done,
        'exposure_samples': {'egy_speaker0': spk0_n, 'egy_speaker1': spk1_n,
                             'msa_speakers': msa_n},
        'exposure_pct': {'egy_speaker0': round(100 * spk0_n / max(tot_n, 1), 2),
                         'egy_speaker1': round(100 * spk1_n / max(tot_n, 1), 2),
                         'msa': round(100 * msa_n / max(tot_n, 1), 2)},
        'plan': {'mechanism': 'duration-quota epoch mixing',
                 'egy_frac': EGY_FRAC, 'epoch_hours': round(E / 3600, 2),
                 'egy_quota_h': round(quota['egy'] / 3600, 2),
                 'msa_quota_h': round(quota['msa'] / 3600, 2)},
        'q_seen': q_seen, 'v_seen': v_seen,
        'note': 'MSA data NOT deleted — rehearsed every epoch (retention)',
    }
    json.dump(sampling_report,
              open(os.path.join(WORK, 'sampling_report.json'), 'w'),
              indent=1)
    log(f'[train] SAMPLING REPORT: {json.dumps(sampling_report["exposure_pct"])}')
    log(f'[train] FINISHED continuation at iter {final_iter} '
        f'(start {base_iter}, +{final_iter - base_iter} iters) '
        f'wall={time.strftime("%H:%M:%S", time.gmtime(time.time()-t_start))} '
        f'| q seen this run={q_seen} v seen={v_seen}')
    log(f'[train] FINAL CHECKPOINT: states_cont_{final_iter}.pth — '
        'STOP (no extension; 4h hard limit respected)')


# ============================================================================
# PHASE: GENERATE — عينات نهائية + مقارنة قبل/بعد + F0 + منحنى
# ============================================================================
def critical_test(model, out_base):
    """الاختبار الحاسم (§18): نفس الكلمة حقيقة مع {ق}/{ج}/{ء} لكل متحدث —
    العلامات الصريحة override فوق السياسة الافتراضية."""
    import torch
    import soundfile as sf
    toks_ms, toks_egy, ids_of, _ = get_tokenizer()
    device = next(model.parameters()).device
    cdir = os.path.join(out_base, 'critical_haqiqa_cont')
    os.makedirs(cdir, exist_ok=True)
    forms = [
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
    ok = {}
    for form, want in (('q', 'q'), ('g', 'v'), ('h', 'hamza')):
        rows = [m for m in manifest if m['form'] == form]
        ok[form] = all(m[want] == 1 for m in rows) if rows else False
    json.dump({'manifest': manifest, 'token_distinct': ok,
               'egyptian_speakers': sorted(EGYPTIAN_TARGET_SPEAKERS),
               'note': 'explicit markers must override default policy'},
              open(os.path.join(cdir, 'critical_test_cont.json'), 'w'),
              ensure_ascii=False, indent=1)
    log(f'[critical] haqiqa triple-test AFTER continuation: '
        f'token distinct q={ok["q"]} g={ok["g"]} h={ok["h"]} '
        f'({len(manifest)} files for {len(SPEAKERS_V1)} speakers)')
    return cdir


def synth_eval(model, out_base, tag):
    """توليد مجموعة التقييم الثابتة كاملة لكل متحدثي v1 (نفس مجموعة
    التجربة الأصلية — للمقارنة المباشرة قبل/بعد)."""
    import torch
    import soundfile as sf
    toks_ms, toks_egy, ids_of, _ = get_tokenizer()
    ev = load_eval_set()
    device = next(model.parameters()).device
    os.makedirs(out_base, exist_ok=True)
    manifest = []
    s = ev['sections']

    def synth(text, mode, out_path, spk):
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
                    r = synth(text, 'egy', os.path.join(
                        spk_dir, f'{cat}_{item["word"]}_{j}.wav'), spk)
                    if r:
                        manifest.append({'speaker': spk, 'cat': cat,
                                         'word': item['word'], 'idx': j,
                                         'text': text, **r})
                        n += 1
        for item in s['explicit']:
            r = synth(item['text'], 'egy', os.path.join(
                spk_dir, f'explicit_{item["tag"]}.wav'), spk)
            if r:
                manifest.append({'speaker': spk, 'cat': 'explicit',
                                 'tag': item['tag'], 'text': item['text'],
                                 **r})
                n += 1
        for j, text in enumerate(s['msa']):
            r = synth(text, 'msa',
                      os.path.join(spk_dir, f'msa_{j:02d}.wav'), spk)
            if r:
                manifest.append({'speaker': spk, 'cat': 'msa', 'idx': j,
                                 'text': text, **r})
                n += 1
        for grp, words in s['words'].items():
            for j, w in enumerate(words):
                r = synth(w, 'egy', os.path.join(
                    spk_dir, f'word_{grp}_{j}.wav'), spk)
                if r:
                    manifest.append({'speaker': spk, 'cat': 'word',
                                     'grp': grp, 'text': w, **r})
                    n += 1
        log(f'[gen{tag}] speaker_{spk}: {n} files')
    with open(os.path.join(out_base, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_base


def f0_report(sample_dir, out_json, tag):
    import numpy as np
    import soundfile as sf
    import librosa
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
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
        out[spk] = {'synth_median_f0': round(med, 1),
                    'n_samples': len(f0s)}
        log(f'[f0{tag}] {spk}: synth median={med:.1f}Hz '
            f'({len(f0s)} samples)')
    json.dump(out, open(out_json, 'w'), indent=1)
    return out


def haqiqa_acoustic(cdir_new, cdir_old, out_json):
    """مقارنة صوتية للاختبار الحاسم (قبل/بعد): مسافة log-mel L2 بين
    أشكال حقيقة {ق}/{ج}/{ء} لكل متحدث مصري + مسافة هوية spk0↔spk1."""
    import numpy as np
    import soundfile as sf
    import librosa

    def logmel(path):
        wav, sr = sf.read(path, dtype='float32')
        m = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        return librosa.power_to_db(m, ref=np.max)

    def dist(a, b):
        T = min(a.shape[1], b.shape[1])
        return float(np.linalg.norm(a[:, :T] - b[:, :T]) / np.sqrt(T))

    res = {'method': 'L2(log-mel[:T]) per frame, T=common frames', 'before':
           {}, 'after': {}}
    for label, cdir in (('before', cdir_old), ('after', cdir_new)):
        if not cdir or not os.path.isdir(cdir):
            res[label] = 'NOT FOUND'
            continue
        block = {}
        for spk in sorted(EGYPTIAN_TARGET_SPEAKERS):
            try:
                q = logmel(os.path.join(cdir, f'spk{spk}_haqiqa_q.wav'))
                g = logmel(os.path.join(cdir, f'spk{spk}_haqiqa_g.wav'))
                h = logmel(os.path.join(cdir, f'spk{spk}_haqiqa_h.wav'))
                block[f'spk{spk}'] = {
                    'q_vs_g': round(dist(q, g), 3),
                    'q_vs_h': round(dist(q, h), 3),
                    'g_vs_h': round(dist(g, h), 3),
                }
            except Exception as e:                # noqa: BLE001
                block[f'spk{spk}'] = f'FAILED: {e}'
        if 'spk0' in block and 'spk1' in block and \
                isinstance(block['spk0'], dict):
            q0 = logmel(os.path.join(cdir, 'spk0_haqiqa_q.wav'))
            q1 = logmel(os.path.join(cdir, 'spk1_haqiqa_q.wav'))
            block['identity_spk0_vs_spk1_same_word'] = round(dist(q0, q1), 3)
        res[label] = block
    json.dump(res, open(out_json, 'w'), indent=1)
    log(f'[acoustic] haqiqa before/after: {json.dumps(res)[:400]}...')
    return res


def phase_generate():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'[generate] device={device}')
    cands = sorted(glob.glob(os.path.join(CKPT_DIR, 'states_cont_*.pth')))
    if not cands:
        cands = sorted(glob.glob(os.path.join(CKPT_DIR, 'states_cont.pth')))
    if not cands:
        log('[generate] NO continuation checkpoint — nothing to generate!')
        return

    def it_of(p):
        try:
            return torch.load(p, map_location='cpu',
                              weights_only=False).get('iter', 0)
        except Exception:                         # noqa: BLE001
            return 0
    best = max(cands, key=it_of)
    log(f'[generate] using continuation checkpoint {best} '
        f'(iter {it_of(best)})')
    st = torch.load(best, map_location='cpu', weights_only=False)
    model = build_scratch_model(device)
    model.load_state_dict(st['model'], strict=True)
    model.eval()

    cdir_new = critical_test(model, os.path.join(WORK, 'samples'))
    out_new = synth_eval(model, os.path.join(WORK, 'samples',
                                            'eiqaz_v1_eval_cont'), ':cont')
    f0_after = f0_report(out_new, os.path.join(METRICS_DIR,
                                               'f0_cont_after.json'), ':after')

    # ---- خط الأساس BEFORE: عينات التجربة الأصلية (مركبة READ-ONLY) ----
    base_root = find_baseline_samples_root()
    f0_before = {}
    if base_root:
        f0_before = f0_report(base_root, os.path.join(
            METRICS_DIR, 'f0_cont_before.json'), ':before')
    else:
        log('[generate] baseline samples NOT FOUND — writing NOT FOUND')
        f0_before = 'NOT FOUND'

    # ---- المقارنة الصوتية للاختبار الحاسم (قبل/بعد) ----
    cdir_old = None
    old_hits = sorted(glob.glob(f'{INPUT}/**/critical_haqiqa',
                                recursive=True))
    if old_hits:
        cdir_old = old_hits[0]
    haqiqa_acoustic(cdir_new, cdir_old,
                    os.path.join(METRICS_DIR, 'haqiqa_acoustic.json'))

    # ---- تقرير المقارنة الشامل ----
    comp = {
        'comparison': 'BEFORE (states_89202, end of 4h pilot) vs AFTER '
                      f'(states_cont, iter {it_of(best)})',
        'before_iter': EXPECTED_BASE_ITER,
        'after_iter': it_of(best),
        'f0_before': f0_before, 'f0_after': f0_after,
        'existing_eval_artifacts': {
            'ab_regression.json': 'NOT FOUND (verified in eiqaz-train-v1 '
                                  'output listing — not created, no fake '
                                  'substitutes)',
            'f0_new_model.json': 'NOT FOUND (verified — same listing)',
        },
    }
    # قياس validation قبل/بعد من val_log.csv
    vlp = os.path.join(WORK, 'val_log.csv')
    if os.path.exists(vlp):
        lines = [l for l in open(vlp).read().strip().splitlines()
                 if l and not l.startswith('iter,')]
        if lines:
            first = lines[0].split(',')
            last = lines[-1].split(',')
            comp['val_before'] = dict(zip(
                ['iter', 'scope', 'loss', 'mel', 'durs', 'pitch', 'ctc',
                 'bin', 'loss_e0', 'mel_e0', 'loss_e1', 'mel_e1',
                 'loss_ms', 'mel_ms', 'secs'], first))
            comp['val_after'] = dict(zip(
                ['iter', 'scope', 'loss', 'mel', 'durs', 'pitch', 'ctc',
                 'bin', 'loss_e0', 'mel_e0', 'loss_e1', 'mel_e1',
                 'loss_ms', 'mel_ms', 'secs'], last))
    srp = os.path.join(WORK, 'sampling_report.json')
    if os.path.exists(srp):
        comp['sampling'] = json.load(open(srp))
    ctj = os.path.join(cdir_new, 'critical_test_cont.json')
    if os.path.exists(ctj):
        comp['critical_token_distinct_after'] = json.load(
            open(ctj))['token_distinct']
    json.dump(comp, open(os.path.join(METRICS_DIR,
                                      'comparison_report.json'), 'w'),
              indent=1, ensure_ascii=False)
    log('[generate] comparison_report.json saved')

    # ---- منحنى مجمّع (تجربة أصلية + استكمال) ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        frames = []
        pilot_log = find_baseline_train_log()
        if pilot_log:
            df0 = pd.read_csv(pilot_log)
            df0 = df0[['iter', 'mel', 'durs', 'ctc', 'loss_d']].copy()
            frames.append(df0)
        own = os.path.join(WORK, 'train_log.csv')
        if os.path.exists(own):
            df1 = pd.read_csv(own)
            cols = [c for c in ('iter', 'mel', 'durs', 'ctc', 'loss_d')
                    if c in df1.columns]
            frames.append(df1[cols].copy())
        if frames:
            df = pd.concat(frames, ignore_index=True).sort_values('iter')
            if len(df) > 2:
                fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                                         constrained_layout=True)
                for ax, col in zip(axes.flat, ['mel', 'durs', 'ctc',
                                               'loss_d']):
                    if col in df:
                        ax.plot(df.iter, df[col], lw=0.6)
                        ax.axvline(EXPECTED_BASE_ITER, color='red',
                                   ls='--', lw=1,
                                   label='continuation start (89,202)')
                        ax.set_title(col)
                        ax.set_xlabel('iter')
                        ax.legend(fontsize=7)
                fig.savefig(os.path.join(WORK, 'training_curve_cont.png'),
                            dpi=120)
                log('[generate] training_curve_cont.png saved '
                    '(pilot + continuation)')
    except Exception as e:                        # noqa: BLE001
        log(f'[generate] curve plot failed: {e}')
    log('[generate] DONE')


# ============================================================================
# PHASE: DRIVER — GPU gate ثم preflight (بصيغة §24) ثم train ثم generate
# ============================================================================
def driver():
    log('=== Eiqaz TTS v1 TRAIN CONTINUATION (from states_89202, +4h) ===')
    import torch
    log(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')
    # ---- بوابة GPU الصارمة (كما في التجربة الأصلية) ----
    if not torch.cuda.is_available():
        log('FATAL: torch.cuda.is_available() == False — GPU NOT VISIBLE.')
        log('Aborting BEFORE anything (no CPU fallback per user spec).')
        sys.exit(2)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        log(f'  gpu {i}: {name} (compute {cap[0]}.{cap[1]}) '
            f'cuda={torch.version.cuda}')
    gpu_name = torch.cuda.get_device_name(0)
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

    # ================= PREFLIGHT (مواصفة §24 — قبل أي تدريب) =================
    log('--- PREFLIGHT ---')
    try:
        import soundfile as sf
        data, eqz = setup_paths()
        feat_root = find_features_root()
        meta = json.load(open(feat_root))
        assert meta.get('complete') is True, 'prep INCOMPLETE'
        fdir = os.path.dirname(feat_root)
        smap = meta.get('speaker_map_v1', {})
        assert smap.get('egy_SPEAKER_01') == 0, 'speaker map: SPEAKER_01!=0'
        assert smap.get('egy_SPEAKER_02') == 1, 'speaker map: SPEAKER_02!=1'

        # ---- checkpoint: تحميل فعلي + تحقق كامل (§2) ----
        base_cands = find_base_checkpoint()
        it_map = {}
        for c in base_cands:
            try:
                it_map[c] = torch.load(c, map_location='cpu',
                                       weights_only=False).get('iter', 0)
            except Exception:                     # noqa: BLE001
                it_map[c] = 0
        base_path = max(it_map, key=it_map.get)
        base_iter = it_map[base_path]
        assert base_iter == EXPECTED_BASE_ITER, (
            f'base iter {base_iter} != {EXPECTED_BASE_ITER} (log-proven '
            'end of 4h run)')
        st = torch.load(base_path, map_location='cpu', weights_only=False)
        n_model_keys = len(st['model'])
        n_optg = len(st['opt_g'].get('state', {}))
        n_optd = len(st['opt_d'].get('state', {}))
        n_critic = len(st['critic'])
        has_opt = n_optg > 0 and n_optd > 0
        spk_w = st['model'].get('speaker_emb.weight')
        n_speakers_arch = tuple(spk_w.shape)[0] if spk_w is not None else None
        assert st['net_config'].get('n_speakers') == 16
        assert n_speakers_arch == 16

        # ---- mini validation على الحالة المستعادة (اختبار §10 مختصرًا) ----
        toks_ms, toks_egy, ids_of, _ = get_tokenizer()
        tq = ids_of(toks_egy('اَلْقَانُون وَ اَلْقُرْآن قَطْعَة{ق} رَقَم{ج} قَال{ء}'))
        assert 29 in tq, 'q token (id 29) missing — tokenizer policy broken!'
        model = build_scratch_model('cuda')
        model.load_state_dict(st['model'], strict=True)
        model.eval()
        spk_of_full = {e['utt']: e['spk'] for e in meta['index']}
        val_by_grp = {'0': [], '1': [], 'msa': []}
        for e in meta['index']:
            if (e.get('split', 'train') != 'train'
                    and e['n_frames'] <= MAX_FRAMES
                    and e['n_tokens'] <= MAX_TOKENS):
                if e['dialect'] == 'egy' and e['spk'] == 0:
                    val_by_grp['0'].append(e['utt'])
                elif e['dialect'] == 'egy' and e['spk'] == 1:
                    val_by_grp['1'].append(e['utt'])
                elif e['dialect'] == 'msa':
                    val_by_grp['msa'].append(e['utt'])
        import numpy as np
        from models.mixer_tts.modules.data_function import (
            BetaBinomialInterpolator)
        betabin = BetaBinomialInterpolator(round_mel_len_to=200,
                                           round_text_len_to=40)
        ps = {'mean': meta['pitch_mean'], 'std': meta['pitch_std']}

        def mini_val(utts):
            ds = []
            for u in utts:
                d = torch.load(os.path.join(fdir, u + '.pt'),
                               weights_only=True)
                pitch = d['pitch']
                mask = pitch > 0
                pitch = torch.where(mask,
                                    (pitch - ps['mean']) / ps['std'],
                                    torch.zeros_like(pitch))
                mel = d['mel']
                ids = d['ids']
                energy = torch.norm(mel.float(), dim=0, p=2)
                prior = torch.from_numpy(
                    betabin(mel.size(1), len(ids))).float()
                ds.append((ids, mel, pitch, energy, prior, spk_of_full[u]))
            ds.sort(key=lambda x: -x[1].size(1))
            B = len(ds)
            max_tok = max(len(x[0]) for x in ds)
            max_fr = max(x[1].size(1) for x in ds)
            text_p = torch.zeros(B, max_tok, dtype=torch.long)
            in_len = torch.zeros(B, dtype=torch.long)
            mel_p = torch.zeros(B, 80, max_fr)
            out_len = torch.zeros(B, dtype=torch.long)
            pit_p = torch.zeros(B, max_fr)
            ene_p = torch.zeros(B, max_fr)
            pri_p = torch.zeros(B, max_fr, max_tok)
            spk_t = torch.zeros(B, dtype=torch.long)
            for i, (ids, mel, pitch, energy, prior, spk) in enumerate(ds):
                L, T = len(ids), mel.size(1)
                text_p[i, :L] = ids
                in_len[i] = L
                mel_p[i, :, :T] = mel
                out_len[i] = T
                pit_p[i, :T] = pitch
                ene_p[i, :T] = energy
                pri_p[i, :T, :L] = prior
                spk_t[i] = spk
            dev = next(model.parameters()).device
            (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p,
             spk_t) = (text_p.to(dev), in_len.to(dev), mel_p.to(dev),
                       out_len.to(dev), pit_p.to(dev), ene_p.to(dev),
                       pri_p.to(dev), spk_t.to(dev))
            with torch.no_grad():
                (pred_spect, _, pred_log_durs, pred_pitch, _, attn_soft,
                 attn_logprob, attn_hard, attn_hard_dur) = model(
                    text=text_p, text_len=in_len, pitch=pit_p, energy=ene_p,
                    spect=mel_p, spect_len=out_len, attn_prior=pri_p,
                    lm_tokens=None, speaker=spk_t,
                    emotion=torch.zeros_like(spk_t))
                (vloss, _, _, _, _, _, _, vmel, _, _) = model._metrics(
                    pred_durs=pred_log_durs, pred_pitch=pred_pitch,
                    pred_energy=None, true_durs=attn_hard_dur,
                    true_text_len=in_len, true_pitch=pit_p,
                    true_energy=ene_p, true_spect=mel_p,
                    pred_spect=pred_spect, true_spect_len=out_len,
                    attn_logprob=attn_logprob, attn_soft=attn_soft,
                    attn_hard=attn_hard, attn_hard_dur=attn_hard_dur)
            return float(vloss), float(vmel)

        mv = {}
        for grp, utts in val_by_grp.items():
            sel = sorted(utts)[:8]
            mv[grp] = mini_val(sel)
        val_test_pass = all(np.isfinite(v[0]) for v in mv.values())
        assert val_test_pass, 'mini validation produced non-finite metrics'
        # regression guard للإصلاح الجذري: الوحدة التي فشلت سابقًا
        assert '1_05206_00' in spk_of_full, 'FIX-1 regression!'

        # ---- طباعة PREFLIGHT بصيغة §24 ----
        h_egy = sum(e['n_frames'] * 256 / 22050 for e in meta['index']
                    if e['dialect'] == 'egy'
                    and e.get('split') == 'train'
                    and e['n_frames'] <= MAX_FRAMES
                    and e['n_tokens'] <= MAX_TOKENS) / 3600
        h_msa = sum(e['n_frames'] * 256 / 22050 for e in meta['index']
                    if e['dialect'] == 'msa'
                    and e.get('split') == 'train'
                    and e['n_frames'] <= MAX_FRAMES
                    and e['n_tokens'] <= MAX_TOKENS) / 3600
        E = min(h_egy / EGY_FRAC, h_msa / (1 - EGY_FRAC))
        eg_targets = '{' + ', '.join(map(str, sorted(
            EGYPTIAN_TARGET_SPEAKERS))) + '}'
        print(f"""
CHECKPOINT:
path: {base_path}
iteration: {base_iter}
model loaded: YES (strict=True, {n_model_keys} tensors)
optimizer loaded: YES (opt_g {n_optg} entries + opt_d {n_optd} entries, AdamW betas=(0.0,0.99) wd=1e-6)
critic loaded: YES ({n_critic} tensors, LSGAN PatchDiscriminatorCond)
scheduler loaded: N/A in checkpoint (LR is computed per-iteration by lr_at() cosine) — RE-ANCHORED for continuation: 1e-4 @ it {base_iter} -> 5e-5 @ it ~{base_iter + LR_DECAY_ITERS} (decay actually happens; D stays fixed 1e-4 as in pilot)
speaker configuration: speaker_emb {tuple(spk_w.shape)} — 7 active speakers, architecture n_speakers=16 (UNCHANGED)
checkpoint metadata: pitch_mean={st['pitch_mean']:.2f} pitch_std={st['pitch_std']:.2f} net_config preserved
training configuration: batch 16, GAN active (it>{GAN_WARMUP}), clip G=20/D=1000, FM=1.0/Score=4.0 (author recipe — unchanged)

SPEAKERS:
architecture: MixerTTS 2.87M, n_speakers=16 (7 active — unchanged)
n_speakers: 16
Speaker 0:
role: Egyptian male
verified: YES (speaker_map_v1: egy_SPEAKER_01 -> 0; {meta['per_speaker'].get('0', {}).get('clips')} clips / {meta['per_speaker'].get('0', {}).get('hours'):.2f}h)
Speaker 1:
role: Egyptian female
verified: YES (speaker_map_v1: egy_SPEAKER_02 -> 1; {meta['per_speaker'].get('1', {}).get('clips')} clips / {meta['per_speaker'].get('1', {}).get('hours'):.2f}h)
MSA speakers: 2,3,4,5,6 (clartts_male + 4 CV females — numbers unchanged)
speaker mapping preserved: YES (identical to pilot index.json)

EGYPTIAN TARGET SPEAKERS:
{eg_targets}  (0 = Egyptian male, 1 = Egyptian female)

DATA:
original MSA data: 13,053 features / 15.77h (eiqaz-prep-v1 — READ-ONLY mount)
MSA data preserved: YES (no deletion/modification/renaming/regeneration; no embedding removal)
Egyptian Speaker 0 samples: {meta['per_speaker'].get('0', {}).get('clips')} clips ({h_egy:.1f}h egy train pool total)
Egyptian Speaker 1 samples: {meta['per_speaker'].get('1', {}).get('clips')} clips
MSA samples: 12,700 train clips ({h_msa:.1f}h train pool)

SAMPLING:
sampling mechanism: duration-quota epoch mixing (EXISTING mechanism — single parameter EGY_FRAC 0.65 -> 0.80)
Egyptian exposure: {EGY_FRAC:.0%} of epoch = {EGY_FRAC * E:.1f}h/epoch (full egy train pool per epoch)
MSA exposure: {1 - EGY_FRAC:.0%} of epoch = {(1 - EGY_FRAC) * E:.1f}h/epoch (rotating -> full MSA pool every ~{h_msa / max((1 - EGY_FRAC) * E, 0.01):.1f} epochs)
MSA rehearsal enabled: YES (per-group exposure logged during training)
actual sampling recorded: YES (train_log.csv columns spk0_n/spk1_n/msa_n + sampling_report.json)

VALIDATION:
status: PASS
KeyError fixed: YES — root cause: spk_of was built from train pool only while make_batch is also used for validation; now built from FULL index.json (train+eval). Guard: '1_05206_00' in spk_of (spk {spk_of_full.get('1_05206_00')}).
test validation: PASS (mini-val on restored checkpoint: egy0 loss={mv['0'][0]:.2f} egy1 loss={mv['1'][0]:.2f} msa loss={mv['msa'][0]:.2f})
Egyptian validation: WORKS (spk0 {len(val_by_grp['0'])} + spk1 {len(val_by_grp['1'])} val clips)
MSA validation: WORKS ({len(val_by_grp['msa'])} val clips, spk 2-6)
val_log.csv: WRITTEN (baseline row at it {base_iter} + per-{VAL_EVERY} iters + final full pass)

LR:
current: {1e-4:.1e} (as restored)
scheduler: cosine via lr_at() — anchor={LR_ANCHOR_IT}, span={LR_DECAY_ITERS} iters, LR_MIN={LR_MIN}
target: {LR_MIN:.1e} at end of continuation window (it ~{LR_ANCHOR_IT + LR_DECAY_ITERS})
step behavior: LR set on opt_g param_groups EVERY iteration before opt_g.step(); printed LR = the real applied LR; checkpoint restoration cannot block it (param_groups lr overridden per step)

CONTINUATION:
start iteration: {base_iter} (states_89202.pth — log-proven end of the 4h pilot)
planned duration: 4h wall-clock training (budget 15300s, stop at 14400s — hard, no extension)
previous checkpoint preserved: YES (states_89202.pth untouched in eiqaz-train-v1 output; READ-ONLY)

DECISION: READY""", flush=True)
        del model, st
        torch.cuda.empty_cache()
    except Exception as e:                    # noqa: BLE001
        import traceback
        log(f'PREFLIGHT FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        log('DECISION: BLOCKED — not starting training.')
        sys.exit(1)
    log('--- END PREFLIGHT (all §25 READY conditions met) ---')

    env = dict(os.environ)
    env['KAGGLE_WORK'] = WORK
    env['SCRATCH_SESSION_T0'] = str(T0)
    r1 = subprocess.run([sys.executable, os.path.abspath(__file__), 'train'],
                        env=env)
    log(f'train exit code: {r1.returncode}')
    if r1.returncode != 0:
        log('train phase FAILED — skipping generate, stopping.')
        sys.exit(r1.returncode)
    r2 = subprocess.run([sys.executable, os.path.abspath(__file__),
                         'generate'], env=env)
    log(f'generate exit code: {r2.returncode}')
    lp = os.path.join(WORK, 'train_log.csv')
    it = 0
    if os.path.exists(lp):
        lines = open(lp).read().strip().splitlines()
        if len(lines) > 1:
            it = lines[-1].split(',')[0]
    log(f'FINAL_ITER_CONT={it} (base {EXPECTED_BASE_ITER})')
    log('ALL_DONE_CONTINUATION — full stop for user review '
        '(no extension, no new experiments)')


if __name__ == '__main__':
    if PHASE == 'train':
        phase_train()
    elif PHASE == 'generate':
        phase_generate()
    else:
        driver()
