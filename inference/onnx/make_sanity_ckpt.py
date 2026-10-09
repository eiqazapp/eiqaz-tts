#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""إنشاء checkpoint عشوائي (sanity) بنفس بنية النموذج — أداة تطوير.

الغرض: التحقق الميكانيكي من سلسلة التصدير/التكميم/المتصفح في بيئة بلا
checkpoint الإنتاج (states_79590.pth يعيش على جهاز Windows لدى المستخدم
ومستثنى من git). الأوزان العشوائية تنتج ضجيجًا لا كلامًا — وهذا معلن في
كل التقارير (نفس مفهوم training/run-artifacts/sanity_random_init.wav).

الاستخدام (من inference/onnx/):
    python make_sanity_ckpt.py     ← يكتب ../checkpoints/states_sanity_random.pth
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(INF_DIR, 'lib', 'mixer_repo'))

import torch  # noqa: E402
from models.mixer_tts.mixer_tts import MixerTTSModel  # noqa: E402

NET_CONFIG = {          # نفس NET_CONFIG_FALLBACK في infer.py
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}

torch.manual_seed(42)
model = MixerTTSModel(**NET_CONFIG)
model.eval()

out = os.path.join(INF_DIR, 'checkpoints', 'states_sanity_random.pth')
torch.save({
    'model': model.state_dict(),
    'net_config': NET_CONFIG,
    'iter': 0,
    'pitch_mean': 212.35853576660156,
    'pitch_std': 67.24,
}, out)
print('saved:', out)

x = torch.LongTensor([[5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                       55, 60, 65, 70, 75, 80, 85, 90, 95, 100]])
with torch.inference_mode():
    mel = model.infer(x, pace=1.0, speaker=0, emotion=0)
print('smoke infer mel:', tuple(mel.shape), '— OK')
