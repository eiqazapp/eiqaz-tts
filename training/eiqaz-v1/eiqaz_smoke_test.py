#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eiqaz_smoke_test.py — اختبار الدخان الإلزامي قبل أي تدريب (طلب المستخدم §18).

13 بندًا:
 1. تحميل البيانات المصرية (corpus التشكيل المصري من مستودع Eiqaz — GitHub)
 2. تحميل بيانات MSA (filelist msa-tts-data-v1)
 3. تحميل الصوت (NileTTS من HF + wavs الـMSA)
 4. قراءة النص المصري المشكول كما هو (لا أي إعادة تشكيل — مطابقة حرفية)
 5. قراءة نص MSA
 6. تحويل النص إلى tokens (المساران)
 7. ظهور q/hamza/g بشكل صحيح في التوكنات (والعلامات {ق}/{ج}/{ء})
 8. speaker IDs (الخريطة v1 — 7 متحدثين)
 9. batch creation
10. forward pass
11. backward pass
12. optimizer step
13. checkpoint save/load (حفظ + إعادة بناء + تحميل + تطابق مخرجات)

يعمل محليًا على CPU بنفس أكواد النوى (استيراد مباشر لدوال النواة).
"""
import json
import os
import sys
import glob
import tempfile
import time

BASE = '/home/z/my-project/work/eqz_v1'
PROJ = f'{BASE}/proj'
MIXER = f'{BASE}/kaggle_dl/mixer_data'
TTS_PKG = f'{MIXER}/tts_arabic_pkg'
REPO = '/home/z/my-project/work/github_repo/eiqaz-tts'
PH3 = f'{REPO}/tashkeel-pipeline/state/ph3_full_output.jsonl'
INP = f'{BASE}/inputs'

sys.path.insert(0, PROJ)
sys.path.insert(0, f'{MIXER}/mixer_repo')
sys.path.insert(0, TTS_PKG)

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ''), flush=True)
    return ok


print('=== Eiqaz v1 — SMOKE TEST (13 items) ===', flush=True)

# ---------------------------------------------------------------------------
# 1) البيانات المصرية (من مستودع Eiqaz — حالة GitHub المحلية)
# ---------------------------------------------------------------------------
print('[1] تحميل البيانات المصرية من مستودع Eiqaz')
tash = {}
for line in open(PH3, encoding='utf-8'):
    line = line.strip()
    if line:
        r = json.loads(line)
        if r.get('status') == 'ok':
            tash[r['id']] = (r['out'], None)
# التقسيم من extraction.csv
import pandas as pd
ex = pd.read_csv(f'{INP}/extraction.csv')
for _, row in ex.iterrows():
    if row.utt in tash:
        tash[row.utt] = (tash[row.utt][0], row.split)
n_train = sum(1 for v in tash.values() if v[1] == 'train')
check('البيانات المصرية (corpus التشكيل)', len(tash) == 21854,
      f'{len(tash)} نصًا مشكولًا (train {n_train} / eval '
      f'{len(tash)-n_train})')

# ---------------------------------------------------------------------------
# 2) بيانات MSA
# ---------------------------------------------------------------------------
print('[2] تحميل بيانات MSA')
msa = pd.read_csv(f'{INP}/filelist.csv')
check('بيانات MSA (filelist)', len(msa) == 13110,
      f'{len(msa)} مقطعًا / {msa.duration_s.sum()/3600:.2f} ساعة / '
      f'{msa.speaker.nunique()} متحدثين / n_q={int(msa.n_q.sum())}')

# ---------------------------------------------------------------------------
# 3) تحميل الصوت
# ---------------------------------------------------------------------------
print('[3] تحميل الصوت (عينات NileTTS من HF + MSA wavs)')
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 22050
audio_ok = True
details = []

# مصري: صفان من NileTTS (نفس معالجة نواة prep: مونو → 22050 → ذروة ≤0.95)
egy_slices = {}
for wav_name in ('general_10_00000', 'general_10_00001'):
    wav, sr = sf.read(f'{INP}/nile/{wav_name}.wav', dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.abs(wav).max())
    if peak > 0.95:
        wav = wav * (0.95 / peak)
    rows = ex[ex.src_row == wav_name]
    for _, r in rows.iterrows():
        if r.utt in tash:
            egy_slices[r.utt] = (wav[int(r.i0):int(r.i1)], int(r.spk_idx),
                                 r.hf_speaker)
    details.append(f'{wav_name}: {sr}Hz→22050, {len(rows)} وحدة')
audio_ok = len(egy_slices) >= 2
check('تحميل الصوت المصري (NileTTS)', audio_ok,
      f'{len(egy_slices)} وحدة مقطعة — ' + '; '.join(details))

msa_audio = {}
for utt, wav_rel in (('clt_00000', 'msa_wavs/clt_00000.wav'),
                     ('cvf_00000', 'msa_wavs/cvf_00000.wav')):
    w, sr = sf.read(f'{INP}/{wav_rel}', dtype='float32')
    msa_audio[utt] = (w, sr)
    assert sr == TARGET_SR
check('تحميل صوت MSA', len(msa_audio) == 2,
      'ClArTTS + CV_female عند 22050Hz')

# ---------------------------------------------------------------------------
# 4) النص المصري المشكول كما هو
# ---------------------------------------------------------------------------
print('[4] قراءة النص المصري المشكول كما هو (لا إعادة تشكيل)')
import re
TASHKEEL = re.compile(r'[\u064B-\u0652]')
sample_utt = next(iter(egy_slices))
egy_text = tash[sample_utt][0]
n_diac = len(TASHKEEL.findall(egy_text))
raw_row = ex[ex.utt == sample_utt].iloc[0].transcript


def _skel(s):
    """هيكل مقارنة: نزع الحركات + توحيد ترقيم ph2 (،→, و ؟→? — التحويل """
    """الموثق قبل التشكيل، لا تغيير حروف)"""
    s = re.sub(r'[\u064B-\u0652\u0670]', '', s)
    return s.replace('،', ',').replace('؟', '?').replace(
        '؛', ';').strip()


skeleton_unchanged = _skel(egy_text) == _skel(raw_row)
check('النص المصري مشكول بالكامل كما هو',
      n_diac > 5 and skeleton_unchanged,
      f'uttd={sample_utt} | حركات={n_diac} | الهيكل مطابق للخام (تشكيل '
      f'+ تحويل ترقيم ph2 الموثق فقط): {skeleton_unchanged}')
print(f'      مثال: {egy_text[:90]}')

# ---------------------------------------------------------------------------
# 5) قراءة نص MSA
# ---------------------------------------------------------------------------
print('[5] قراءة نص MSA')
msa_row = msa[msa.utt == 'clt_00000'].iloc[0]
check('نص MSA مشكول فصيح', len(TASHKEEL.findall(msa_row.text)) > 5,
      f'"{msa_row.text}"')

# ---------------------------------------------------------------------------
# 6) تحويل النص إلى tokens
# ---------------------------------------------------------------------------
print('[6] تحويل النص إلى tokens (المساران)')
import eqz_tokens as ET
toks_ms, toks_egy, ids_of, parse = ET.get_tokenizers([TTS_PKG])
ids_egy = ids_of(toks_egy(egy_text))
ids_msa = ids_of(toks_ms(msa_row.text))
check('ترميز مصري', 5 <= len(ids_egy) <= 180, f'n={len(ids_egy)}')
check('ترميز MSA', 5 <= len(ids_msa) <= 180, f'n={len(ids_msa)}')

# ---------------------------------------------------------------------------
# 7) ظهور q/hamza/g بشكل صحيح
# ---------------------------------------------------------------------------
print('[7] ظهور q/hamza/g بشكل صحيح + العلامات')
t_q = toks_egy('اَلْقَانُون وَ اَلْقُرْآن قَطْعَة')
t_h = toks_egy('قَالَ رَاجِل قِيمَة قُوَّة')
t_g = toks_egy('رَقَم قِيرَاط مَقَام')
t_m = toks_ms('اَلْقَانُونُ اَلْعَرَبِيُّ')
t_mk = toks_egy('قَطْعَة{ق} رَقَم{ج} قَال{ء} نَقْرَأ{ق}')
hamza_ok = '<' in t_h and 'q' not in t_h
ok7 = ('q' in t_q and hamza_ok and 'v' in t_g and 'q' in t_m
       and t_mk.count('q') == 2 and 'v' in t_mk)
ids7 = ids_of(t_mk)
check('q (29) / hamza (9) / g-v (37) صحيحة', ok7 and 29 in ids7
      and 37 in ids7,
      f'قانون/قرآن/قطعة→q: {"q" in t_q} | قال/قيمة/قوة→همزة: '
      f'{"q" not in t_h} | رقم/قيراط/مقام→v: {"v" in t_g} | '
      f'فصحى q خام: {"q" in t_m} | علامات مختلطة: q×{t_mk.count("q")}, '
      f'v={"v" in t_mk}')

# ---------------------------------------------------------------------------
# 8) speaker IDs
# ---------------------------------------------------------------------------
print('[8] speaker IDs (خريطة v1)')
SPEAKER_MAP_V1 = {
    'egy_SPEAKER_01': 0, 'egy_SPEAKER_02': 1, 'msa_clartts_male': 2,
    'msa_cvf_5f810213': 3, 'msa_cvf_78c954e3': 4, 'msa_cvf_cf4d8f89': 5,
    'msa_cvf_fc3b87e3': 6}
spk_egy = SPEAKER_MAP_V1['egy_' + list(egy_slices.values())[0][2]]
spk_msas = [SPEAKER_MAP_V1['msa_' + s] for s in
            ('clartts_male', 'cvf_5f810213')]
check('خريطة المتحدثين v1 (7)', len(SPEAKER_MAP_V1) == 7
      and spk_egy in (0, 1) and spk_msas == [2, 3],
      f'مصري→{spk_egy} | clartts→2 | cvf→3 (سعة النموذج n_speakers=16)')

# ---------------------------------------------------------------------------
# بناء ملامح العينة (نفس دوال نواة prep)
# ---------------------------------------------------------------------------
print('[*] بناء ملامح العينة (نفس مسار prep: mel + pyin)')
FEAT_DIR = tempfile.mkdtemp(prefix='eqz_smoke_')
from utils.audio import MelSpectrogram
mel_fn = MelSpectrogram()


def wav_to_feat(ids, wav):
    import torch
    wav_t = torch.from_numpy(np.ascontiguousarray(wav))[None]
    mel = mel_fn(wav_t).clamp_min(1e-5).log().squeeze(0)
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


import torch
feats = {}
index = []
for utt, (wav, spk_idx, _) in egy_slices.items():
    ids = ids_of(toks_egy(tash[utt][0]))
    mel, pitch = wav_to_feat(ids, wav)
    torch.save({'ids': torch.LongTensor(ids), 'mel': mel, 'pitch': pitch},
               f'{FEAT_DIR}/{utt}.pt')
    feats[utt] = (torch.LongTensor(ids), mel, pitch)
    index.append({'utt': utt, 'dialect': 'egy', 'spk': spk_idx,
                  'n_tokens': len(ids), 'n_frames': int(mel.size(1)),
                  'split': 'train'})
for utt, (wav, sr) in msa_audio.items():
    row = msa[msa.utt == utt].iloc[0]
    ids = ids_of(toks_ms(row.text))
    mel, pitch = wav_to_feat(ids, wav)
    name = 'msa_' + utt
    torch.save({'ids': torch.LongTensor(ids), 'mel': mel, 'pitch': pitch},
               f'{FEAT_DIR}/{name}.pt')
    feats[name] = (torch.LongTensor(ids), mel, pitch)
    index.append({'utt': name, 'dialect': 'msa',
                  'spk': SPEAKER_MAP_V1['msa_' + row.speaker],
                  'n_tokens': len(ids), 'n_frames': int(mel.size(1)),
                  'split': 'train'})
print(f'      {len(index)} ملامح: '
      + ', '.join(f"{e['utt']}({e['dialect']},spk{e['spk']},"
                  f"{e['n_frames']}fr)" for e in index))

# ---------------------------------------------------------------------------
# 9) batch creation
# ---------------------------------------------------------------------------
print('[9] batch creation (نفس make_batch في نواة التدريب)')
from models.mixer_tts.modules.data_function import BetaBinomialInterpolator
betabin = BetaBinomialInterpolator(round_mel_len_to=200,
                                   round_text_len_to=40)
ps = {'mean': 150.0, 'std': 30.0}          # قيم اختبار فقط


def make_batch(items):
    data = []
    for e in items:
        ids, mel, pitch = feats[e['utt']]
        mask = pitch > 0
        pitch_n = torch.where(mask, (pitch - ps['mean']) / ps['std'],
                              torch.zeros_like(pitch))
        energy = torch.norm(mel.float(), dim=0, p=2)
        prior = torch.from_numpy(betabin(mel.size(1), len(ids))).float()
        data.append((ids, mel, pitch_n, energy, prior, e['spk']))
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
        L, T = len(ids), mel.size(1)
        text_p[i, :L] = ids
        in_len[i] = L
        mel_p[i, :, :T] = mel
        out_len[i] = T
        pit_p[i, :T] = pitch
        ene_p[i, :T] = energy
        pri_p[i, :T, :L] = prior
        spk_t[i] = spk
    return (text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p, spk_t)


batch = make_batch(index)
check('batch creation', batch[0].shape[0] == len(index)
      and batch[7].tolist() == [e['spk'] for e in sorted(
          index, key=lambda e: -feats[e['utt']][1].size(1))],
      f'B={batch[0].shape[0]} max_tok={batch[0].shape[1]} '
      f'max_fr={batch[0].shape[1] and batch[2].shape[2]} '
      f'speakers={batch[7].tolist()}')

# ---------------------------------------------------------------------------
# 10) forward pass — 11) backward — 12) optimizer step — 13) checkpoint
# ---------------------------------------------------------------------------
print('[10] forward pass (نموذج عشوائي جديد — Scratch)')
from models.mixer_tts.mixer_tts import MixerTTSModel
from models.mixer_tts import net_config
net_config.update({'num_tokens': 148, 'padding_idx': 0,
                   'symbols_embedding_dim': 128, 'n_speakers': 16,
                   'n_emotions': 16, 'energy_conditioning': False})
model = MixerTTSModel(**net_config)
model.add_bin_loss = True
model.bin_loss_scale = 1.0
torch.manual_seed(1234)
n_params = sum(p.numel() for p in model.parameters())
text_p, in_len, mel_p, out_len, pit_p, ene_p, pri_p, spk_t = batch
out = model(text=text_p, text_len=in_len, pitch=pit_p, energy=ene_p,
            spect=mel_p, spect_len=out_len, attn_prior=pri_p,
            lm_tokens=None, speaker=spk_t, emotion=torch.zeros_like(spk_t))
(pred_spect, _, pred_log_durs, pred_pitch, _, attn_soft, attn_logprob,
 attn_hard, attn_hard_dur) = out
check('forward pass', pred_spect.shape[0] == len(index),
      f'spect={tuple(pred_spect.shape)} | params={n_params/1e6:.2f}M')

print('[11] backward pass')
(loss, durs_loss, acc, _, _, pitch_loss, _, mel_loss, ctc_loss,
 bin_loss) = model._metrics(
    pred_durs=pred_log_durs, pred_pitch=pred_pitch, pred_energy=None,
    true_durs=attn_hard_dur, true_text_len=in_len, true_pitch=pit_p,
    true_energy=ene_p, true_spect=mel_p, pred_spect=pred_spect,
    true_spect_len=out_len, attn_logprob=attn_logprob, attn_soft=attn_soft,
    attn_hard=attn_hard, attn_hard_dur=attn_hard_dur)
loss.backward()
grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
              for p in model.parameters() if p.requires_grad)
check('backward pass', grad_ok and float(loss) > 0,
      f'loss={float(loss):.3f} mel={float(mel_loss):.3f} '
      f'ctc={float(ctc_loss):.3f} تدرجات nonzero={grad_ok}')

print('[12] optimizer step')
opt_g = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.0, 0.99),
                          weight_decay=1e-6)
w_before = model.parameters().__next__().detach().clone()
opt_g.step()
opt_g.zero_grad()
w_after = model.parameters().__next__().detach()
check('optimizer step', not torch.equal(w_before, w_after),
      'الأوزان تحركت (AdamW خطوة واحدة)')

print('[13] checkpoint save/load')
ck = f'{FEAT_DIR}/states_smoke.pth'
torch.save({'model': model.state_dict(), 'iter': 1,
            'net_config': net_config, 'pitch_mean': ps['mean'],
            'pitch_std': ps['std']}, ck)
model2 = MixerTTSModel(**net_config)
model2.add_bin_loss = True
model2.bin_loss_scale = 1.0
st = torch.load(ck, map_location='cpu', weights_only=False)
model2.load_state_dict(st['model'], strict=True)
model2.eval()
model.eval()
with torch.no_grad():
    o1 = model.infer(text_p[:1], pace=1.0, speaker=int(spk_t[0]), emotion=0)
    o2 = model2.infer(text_p[:1], pace=1.0, speaker=int(spk_t[0]), emotion=0)
same = torch.allclose(o1, o2, atol=1e-6)
check('checkpoint save/load', same and st['iter'] == 1,
      f'حفظ + إعادة بناء + تحميل strict=True + تطابق المخرجات={same}')

# ---------------------------------------------------------------------------
print()
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f'=== SMOKE TEST: {n_pass}/{len(RESULTS)} PASS ===')
for name, ok, det in RESULTS:
    if not ok:
        print(f'  FAILED: {name} — {det}')
json.dump({'n_pass': n_pass, 'n_total': len(RESULTS),
           'items': [{'name': n, 'ok': ok, 'detail': d}
                     for n, ok, d in RESULTS]},
          open(f'{BASE}/smoke_test_results.json', 'w'),
          ensure_ascii=False, indent=1)
sys.exit(0 if n_pass == len(RESULTS) else 1)
