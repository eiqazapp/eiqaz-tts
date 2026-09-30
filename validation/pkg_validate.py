#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبار صحة الاستخراج: مقارنة مخرجات الحزمة المستقلة ضد مسار كود
النواة الأصلي (build_scratch_model + model.infer + mel_to_wav) على نفس
الـcheckpoint ونفس الجهاز (CPU). المتوقع: تطابق تام في الميل، وتطابق
تام/شبه تام في WAV النهائي."""
import os, sys, time
import numpy as np

MY = '/home/z/my-project/work/inference_package/NileTTS-4h-Inference'
ORIG_DATA = '/home/z/my-project/work/scratch_data'          # mixer_repo الأصلي
ORIG_TTS = '/home/z/my-project/work/tts_arabic_pkg'          # tts_arabic الأصلي
CKPT = '/home/z/my-project/work/train_output/checkpoints/states_79590.pth'
TEXT = 'إِزَّيْك يَا صَاحِبِي، عَامِل إِيهْ دِلْوَقْتِي؟'
PROBE = '/home/z/my-project/work/train_output/probes/it078000_s0_spk0.wav'

import torch

# ---------- المسار الأصلي (كود النواة كما هو) ----------
sys.path.insert(0, os.path.join(ORIG_DATA, 'mixer_repo'))
sys.path.insert(0, ORIG_TTS)
NET_CONFIG_OVERRIDES = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}
EGY_TOKEN_MAP = {'j': 'v', 'q': '<', '^': 't', '*': 'd'}

def toks_egy_orig(text):
    from tts_arabic.text import (
        arabic_to_buckwalter, phonemes_to_tokens, buckwalter_to_phonemes)
    return [EGY_TOKEN_MAP.get(t, t) for t in phonemes_to_tokens(
        buckwalter_to_phonemes(arabic_to_buckwalter(text)))]

from tts_arabic.text import tokens_to_ids as ids_of_orig
from models.mixer_tts.mixer_tts import MixerTTSModel
from models.mixer_tts import net_config as nc_orig
nc_orig.update(NET_CONFIG_OVERRIDES)
model_o = MixerTTSModel(**nc_orig)
model_o.add_bin_loss = True
model_o.bin_loss_scale = 1.0
st = torch.load(CKPT, map_location='cpu', weights_only=False)
model_o.load_state_dict(st['model'], strict=True)
model_o.eval().to('cpu')

ids_o = ids_of_orig(toks_egy_orig(TEXT))
x = torch.LongTensor([ids_o])
with torch.inference_mode():
    mel_o = model_o.infer(x, pace=1.0, speaker=0, emotion=0)
m_o = mel_o.transpose(1, 2)[0].cpu().numpy()

import onnxruntime as ort
sess_o = ort.InferenceSession(
    os.path.join(ORIG_DATA, 'extracted', 'models', 'vocos22.onnx'),
    providers=['CPUExecutionProvider'])
def mel_to_wav_orig(mel_80_T):
    wave = sess_o.run(None, {
        'mel_spec': mel_80_T[None].astype('float32'),
        'denoise': np.array([0.005], dtype='float32'),
    })[0].astype('float32')[0]
    return 0.9 * wave / (np.abs(wave).max() + 1e-5)
w_o = mel_to_wav_orig(m_o)

# تفريغ الوحدات لاستيراد نسخ الحزمة بنفس الأسماء (models.*, tts_arabic)
for m in list(sys.modules):
    if m == 'models' or m.startswith('models.') or m == 'tts_arabic' or m.startswith('tts_arabic.'):
        del sys.modules[m]
sys.path = [p for p in sys.path if p not in (
    os.path.join(ORIG_DATA, 'mixer_repo'), ORIG_TTS)]

# ---------- مسار الحزمة المستقلة (infer.py كما هو) ----------
sys.path.insert(0, os.path.join(MY, 'lib'))
sys.path.insert(0, os.path.join(MY, 'lib', 'mixer_repo'))
sys.path.insert(0, MY)
import importlib.util as ilu
spec = ilu.spec_from_file_location('infer_mod', os.path.join(MY, 'infer.py'))
infer_mod = ilu.module_from_spec(spec)
# منع تنفيذ main() — infer.py محمي بـ __main__
spec.loader.exec_module(infer_mod)

model_m, it_m = infer_mod.load_model(CKPT)
text_prep, did_voc = infer_mod.prepare_text(TEXT, 'never', 'egy')
with torch.inference_mode():
    mel_m = model_m.infer(torch.LongTensor([infer_mod.get_tokenizer()[1](text_prep) and
                                            infer_mod.get_tokenizer()[2](
        infer_mod.get_tokenizer()[1](text_prep))]), pace=1.0, speaker=0, emotion=0)
m_m = mel_m.transpose(1, 2)[0].cpu().numpy()
w_m = infer_mod.mel_to_wav(m_m)

# ---------- المقارنات ----------
print('=== TEXT PIPELINE ===')
print('orig ids :', len(ids_o), ids_o[:12], '...')
ids_m = infer_mod.get_tokenizer()[2](infer_mod.get_tokenizer()[1](text_prep))
print('mine ids :', len(ids_m), ids_m[:12], '...')
print('TOKENS IDENTICAL:', ids_o == ids_m)
print('prepared text:', text_prep)

print('=== MEL ===')
print('orig shape:', m_o.shape, 'mine shape:', m_m.shape)
if m_o.shape == m_m.shape:
    d = np.abs(m_o - m_m)
    print(f'max abs diff: {d.max():.3e} | mean: {d.mean():.3e}')
    print('MEL BITWISE IDENTICAL:', np.array_equal(m_o, m_m))

print('=== WAV ===')
print('orig len:', len(w_o), 'mine len:', len(w_m))
if len(w_o) == len(w_m):
    dw = np.abs(w_o - w_m)
    print(f'wav max abs diff: {dw.max():.3e} | mean: {dw.mean():.3e}')
    print('WAV BITWISE IDENTICAL:', np.array_equal(w_o, w_m))
    corr = np.corrcoef(w_o, w_m)[0, 1]
    print(f'correlation: {corr:.6f}')

# مقابل probe النواة (GPU، iter 78000 ≠ 79590) — تشابه نوعي فقط
import soundfile as sf
if os.path.exists(PROBE):
    pw, sr = sf.read(PROBE, dtype='float32')
    n = min(len(pw), len(w_m))
    c = np.corrcoef(pw[:n], w_m[:n])[0, 1]
    print(f'=== vs kernel GPU probe (it78000) === len={len(pw)} corr={c:.4f}')
    sf.write('/home/z/my-project/work/inference_package/pkg_vs_probe_mine.wav', w_m, 22050, subtype='PCM_16')
