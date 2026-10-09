#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تصدير MixerTTS إلى ONNX (FP32 أولًا — دون تكميم)
====================================================
يصدّر مسار الاستدلال الصوتي فقط (text → mel) — نفس نداء الإنتاج
`model.infer(x, pace, speaker, emotion=0)` حرفيًا عبر غلاف رقيق، ثم:

  1) فحص تكافؤ (parity) مخرجات mel مقابل PyTorch على توكنز النصوص الذهبية
     الحقيقية (المسار النصي الكامل عبر infer.prepare_text_rich + toks_egy).
  2) حفظ net_config + قائمة الرموز (symbols) بجانب النموذج لتشغيل المتصفح.

لا يمس هذا السكريبت: الأوزان الأصلية، مسار infer.py، الـtokenizer،
أو أي ملف إنتاجي — كل المخرجات في inference/onnx/ و web-exp/models/.

الاستخدام:
    python export_onnx.py --checkpoint states_79590.pth        (على Windows)
    python export_onnx.py --checkpoint states_sanity_random.pth --tag sanity
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)                 # inference/
REPO = os.path.dirname(INF_DIR)
MIXER_LIB = os.path.join(INF_DIR, 'lib', 'mixer_repo')
WEBEXP_MODELS = os.path.join(REPO, 'web-exp', 'models')

sys.path.insert(0, INF_DIR)
sys.path.insert(0, MIXER_LIB)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import infer  # noqa: E402


# ============================================================================
# غلاف التصدير — توقيع نظيف للمتصفح: (text, pace, speaker) → mel [1,80,T']
# ============================================================================
class InferExportWrapper(nn.Module):
    """غلاف رقيق حول model.infer بلا أي تغيير حسابي.

    المدخلات:
        text    : int64  [1, T]   — توكنز الرموز (ids)
        pace    : float32 scalar  — سرعة الكلام (1.0 = طبيعي)
        speaker : int64  scalar   — 0 ذكر / 1 أنثى
    المخرجات:
        mel     : float32 [1, 80, T'] — سبكتروجرام ميل (جاهز لدخول vocos)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, pace, speaker):
        mel = self.model.infer(text, pace=pace, speaker=speaker, emotion=0)
        # infer يعيد [B, T', 80] — نطبق [B, 80, T'] كمدخل vocos مباشرة
        return mel.transpose(1, 2).contiguous()


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def golden_token_sets(gold_path):
    """توكنز حقيقية من النصوص الذهبية عبر مسار النص الإنتاجي نفسه."""
    with open(gold_path, encoding='utf-8') as f:
        gold = json.load(f)
    toks_ms, toks_egy, ids_of = infer.get_tokenizer('auto')
    out = []
    for item in gold['items']:
        try:
            res = infer.prepare_text_rich(item['text'], 'auto',
                                          item['dialect'], 'auto')
            toks = (infer.get_msa_synthesis_tokens(res['text'], 'auto')
                    if item['dialect'] == 'msa' else toks_egy(res['text']))
            ids = ids_of(toks)
            if 2 <= len(ids) <= 160:          # داخل سقف التدريب
                out.append((item['id'], ids))
        except Exception:                      # noqa: BLE001
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description='تصدير MixerTTS إلى ONNX FP32')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--gold', default=os.path.join(HERE, 'golden_texts.json'))
    ap.add_argument('--out-dir', default=WEBEXP_MODELS)
    ap.add_argument('--tag', default=None,
                    help='وسم في manifest (sanity/production)')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    import onnx  # noqa: E402
    import onnxruntime as ort  # noqa: E402

    ckpt = infer.resolve_checkpoint(args.checkpoint)
    ckpt_name = os.path.basename(ckpt)
    print(f'[export] checkpoint: {ckpt_name}')

    model, it = infer.load_model(ckpt)
    wrapper = InferExportWrapper(model).eval()

    # ---- مثال تتبّع بطول واقعي (40 توكن) ---------------------------------
    torch.manual_seed(0)
    ex_text = torch.randint(2, 148, (1, 40), dtype=torch.long)
    ex_pace = torch.tensor(1.0, dtype=torch.float32)
    ex_speaker = torch.tensor(0, dtype=torch.long)

    out_path = os.path.join(args.out_dir, 'mixertts_fp32.onnx')
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (ex_text, ex_pace, ex_speaker),
            out_path,
            input_names=['text', 'pace', 'speaker'],
            output_names=['mel'],
            dynamic_axes={
                'text': {0: 'batch', 1: 'T'},
                'mel': {0: 'batch', 2: 'T_out'},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )
    export_s = time.perf_counter() - t0
    print(f'[export] saved {out_path} in {export_s:.1f}s '
          f'({os.path.getsize(out_path) / 1e6:.1f} MB)')

    # ---- فحص صحة البنية ---------------------------------------------------
    m = onnx.load(out_path)
    onnx.checker.check_model(m, full_check=False)
    opset_used = m.opset_import[0].version
    print(f'[export] onnx check OK (opset {opset_used})')

    # ---- فحص التكافؤ على توكنز ذهبية حقيقية ------------------------------
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    sets = golden_token_sets(args.gold)
    print(f'[parity] testing on {len(sets)} golden token sets ...')

    rows = []
    for gid, ids in sets:
        x = torch.LongTensor([ids])
        with torch.no_grad():
            ref = wrapper(x, torch.tensor(1.0), torch.tensor(0))
        got = sess.run(None, {
            'text': x.numpy(),
            'pace': np.array(1.0, dtype=np.float32),
            'speaker': np.array(0, dtype=np.int64),
        })[0]
        ref_np = ref.numpy()
        same_shape = ref_np.shape == got.shape
        if same_shape:
            max_diff = float(np.abs(ref_np - got).max())
            rel = max_diff / (float(np.abs(ref_np).max()) + 1e-9)
        else:
            max_diff, rel = None, None
        rows.append({'id': gid, 'n_tokens': len(ids),
                     'mel_frames': int(got.shape[-1]),
                     'shape_match': same_shape,
                     'max_abs_diff': max_diff,
                     'rel_diff': rel})
        print(f"  {gid:<20} tokens={len(ids):<4} frames={got.shape[-1]:<4} "
              f"maxdiff={max_diff if max_diff is not None else 'SHAPE!'}")

    n_bad = sum(1 for r in rows if not r['shape_match'] or
                (r['max_abs_diff'] or 1) > 1e-3)
    verdict = 'PASS' if n_bad == 0 else 'FAIL'
    print(f'[parity] {verdict} — {len(rows) - n_bad}/{len(rows)} '
          f'ضمن 1e-3')

    # ---- حفظ net_config + الرموز للمتصفح ----------------------------------
    net_config = dict(getattr(model, '_net_config', None) or
                      infer.NET_CONFIG_FALLBACK)
    # المصدر الحقيقي: أعد قراءته من الـcheckpoint
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    net_config = dict(st.get('net_config') or infer.NET_CONFIG_FALLBACK)
    from tts_arabic.text.symbols import symbols as SYMBOLS  # noqa: E402
    meta = {
        'name': 'mixertts_fp32.onnx',
        'file': 'mixertts_fp32.onnx',
        'quant': 'fp32',
        'precision': 'fp32',
        'checkpoint': ckpt_name,
        'checkpoint_iter': str(it),
        'tag': args.tag or ('sanity' if 'sanity' in ckpt_name
                            else 'production'),
        'opset': int(opset_used),
        'size_bytes': os.path.getsize(out_path),
        'sha256': sha256_of(out_path),
        'net_config': net_config,
        'pitch_mean': float(getattr(model, 'pitch_mean', 212.35853576660156)),
        'pitch_std': float(getattr(model, 'pitch_std', 67.24)),
        'symbols': SYMBOLS,
        'egy_token_map': infer.EGY_TOKEN_MAP,
        'inputs': {'text': 'int64[1,T]', 'pace': 'float32',
                   'speaker': 'int64'},
        'outputs': {'mel': 'float32[1,80,T_out]'},
        'parity': {'verdict': verdict,
                   'n_cases': len(rows),
                   'n_pass': len(rows) - n_bad,
                   'tolerance': 1e-3,
                   'rows': rows},
    }
    meta_path = os.path.join(args.out_dir, 'mixertts_fp32.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'[export] meta: {meta_path}')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
