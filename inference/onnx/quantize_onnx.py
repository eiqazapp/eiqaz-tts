#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تكميم النسخ — FP16 / INT8 ديناميكي / INT8 ثابت + قياس التكافؤ
================================================================
يبدأ من mixertts_fp32.onnx (ناتج export_onnx.py) وينشئ:

  1) mixertts_fp16.onnx       — FP16 كامل (مع حماية عقد Cast/الشكل والمتنبئات)
  2) mixertts_int8dyn.onnx    — INT8 ديناميكي (أوزان QInt8، المتنبئات محمية)
  3) mixertts_int8static.onnx — INT8 ثابت QDQ بمعايرة توكنز النصوص الذهبية

واختياريًا نسختين من المُصوِّت: vocos22_fp16 / vocos22_int8dyn.

مقاييس التكافؤ لكل نسخة (مقارنة بـFP32 على النصوص الذهبية):
  - frames_diff: أقصى فرق في عدد إطارات mel (يؤثر على طول الصوت)
  - mel_cos: أدنى تشابه جيب التماس على المنطقة المتقاطعة
  - wav_cos: أدنى تشابه جيب التماس للموجة بعد المُصوِّت vocos (المقياس
    الأقرب لما يسمعه المستخدم)
  - max_abs_diff: أقصى فرق مطلق في mel

ملاحظات صادقة:
  - INT4 (MatMulNBits) غير متاح في أدوات onnxruntime بهذه البيئة — مسجل
    "غير مدعوم" في manifest.
  - القياسات هنا بأوزان عشوائية (sanity) تكون متشائمة: توزيعات التنشيط
    غير مدرَّبة تجعل ضجيج التكميم يبدو أكبر. أعِد التشغيل على Windows
    بالأوزان الحقيقية للأرقام الرسمية.

الاستخدام:
    python quantize_onnx.py                (بعد export_onnx.py)
    python quantize_onnx.py --skip-vocos
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(INF_DIR)
MODELS_DIR = os.path.join(REPO, 'web-exp', 'models')

sys.path.insert(0, INF_DIR)
sys.path.insert(0, os.path.join(INF_DIR, 'lib', 'mixer_repo'))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper, TensorProto  # noqa: E402
import onnxruntime as ort  # noqa: E402

import infer  # noqa: E402

SHAPE_OPS = {'Range', 'CumSum', 'Equal', 'Where', 'NonZero', 'Expand'}


# ---------------------------------------------------------------------------
# أدوات النموذج
# ---------------------------------------------------------------------------
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def strip_identity(model):
    """إزالة عقد Identity التي تمرر initializer — تُفسد تحويل fp16."""
    graph = model.graph
    inits = {i.name for i in graph.initializer}
    inputs = {i.name for i in graph.input}
    mapping = {}
    keep = []
    for n in graph.node:
        if n.op_type == 'Identity' and n.input[0] in (inits | inputs):
            mapping[n.output[0]] = n.input[0]
        else:
            keep.append(n)
    if mapping:
        for n in keep:
            for k, v in enumerate(n.input):
                if v in mapping:
                    n.input[k] = mapping[v]
        new_outs = [o for o in graph.output if o.name not in mapping]
        del graph.output[:]
        graph.output.extend(new_outs)
        del graph.node[:]
        graph.node.extend(keep)
    return len(mapping)


def node_names_containing(model, *subs):
    return [n.name for n in model.graph.node
            if any(s in (n.name or '') for s in subs)]


def fp16_blocklist(model):
    """العقد المحمية من fp16: Cast + المتنبئات + عقد الشكل الديناميكي.

    الحماية مبنية على تجربة موثقة: عقد Cast المخترعة بالتتبع تكسر أنواع
    الحواف عند التحويل، والمتنبئات (المدد/النبرة) تُحمى لضمان ثبات
    الإيقاع، وعقد الشكل (Range/CumSum...) بُنيت على int64 أصلاً."""
    return [n.name for n in model.graph.node
            if n.op_type == 'Cast'
            or n.op_type in SHAPE_OPS
            or 'duration_predictor' in (n.name or '')
            or 'pitch_predictor' in (n.name or '')]


def golden_token_sets(gold_path):
    with open(gold_path, encoding='utf-8') as f:
        gold = json.load(f)
    toks_ms, toks_egy, ids_of = infer.get_tokenizer()
    out = []
    for item in gold['items']:
        try:
            res = infer.prepare_text_rich(item['text'], 'manual',
                                          item['dialect'])
            toks = toks_ms(res['text']) if item['dialect'] == 'msa' \
                else toks_egy(res['text'])
            ids = ids_of(toks)
            if 2 <= len(ids) <= 160:
                out.append((item['id'], ids))
        except Exception:                                      # noqa: BLE001
            continue
    return out


def run_mel(sess, ids, pace=1.0, speaker=0):
    return sess.run(None, {
        'text': np.array([ids], dtype=np.int64),
        'pace': np.array(pace, dtype=np.float32),
        'speaker': np.array(speaker, dtype=np.int64),
    })[0]


# ---------------------------------------------------------------------------
# القياس
# ---------------------------------------------------------------------------
def _stft_mag(w, n=1024, hop=256):
    """طيف القدرة التقريبي (بلا مكتبات خارجية) — مقاوم للانزياح الزمني."""
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    n_frames = 1 + (len(w) - n) // hop
    win = np.hanning(n).astype(w.dtype)
    frames = np.stack([w[i * hop:i * hop + n] * win
                       for i in range(n_frames)])
    return np.abs(np.fft.rfft(frames, axis=1))


def _spec_cos(wa, wb):
    ma, mb = _stft_mag(wa), _stft_mag(wb)
    L = min(len(ma), len(mb))
    if L < 2:
        return None
    a, b = ma[:L].ravel(), mb[:L].ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None


def parity_rows(sess_q, refs, vocos_sess=None):
    rows = []
    for gid, ids, ref in refs:
        got = run_mel(sess_q, ids)
        fd = abs(int(got.shape[-1]) - int(ref.shape[-1]))
        T = min(got.shape[-1], ref.shape[-1])
        a, b = ref[..., :T].ravel(), got[..., :T].ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None
        row = {
            'id': gid, 'frames_ref': int(ref.shape[-1]),
            'frames_q': int(got.shape[-1]), 'frames_diff': fd,
            'mel_cos': round(cos, 6) if cos is not None else None,
            'max_abs_diff': round(float(np.abs(a - b).max()), 6),
        }
        if vocos_sess is not None and T >= 2:
            try:
                wa = vocos_sess.run(None, {
                    'mel_spec': ref[..., :T].astype('float32'),
                    'denoise': np.array([0.005], dtype='float32')})[0][0]
                wb = vocos_sess.run(None, {
                    'mel_spec': got[..., :T].astype('float32'),
                    'denoise': np.array([0.005], dtype='float32')})[0][0]
                L = min(len(wa), len(wb))
                va, vb = wa[:L], wb[:L]
                nva, nvb = np.linalg.norm(va), np.linalg.norm(vb)
                row['wav_cos'] = round(
                    float(np.dot(va, vb) / (nva * nvb)), 6) \
                    if nva > 0 and nvb > 0 else None
                # ارتباط طيف القدرة — مقاوم لانزياح 1-2 إطار (~23-46مل)
                # الذي يفسد ارتباط الموجة الخام بلا أثر إدراكي مكافئ
                sc = _spec_cos(va, vb)
                row['spec_cos'] = round(sc, 6) if sc is not None else None
            except Exception:                                  # noqa: BLE001
                row['wav_cos'] = None
                row['spec_cos'] = None
        rows.append(row)
    return rows


def summarize(rows):
    fd = max(r['frames_diff'] for r in rows)
    cos = min((r['mel_cos'] for r in rows if r['mel_cos'] is not None),
              default=None)
    wcos = min((r['wav_cos'] for r in rows if r['wav_cos'] is not None),
               default=None)
    scos = min((r['spec_cos'] for r in rows if r['spec_cos'] is not None),
               default=None)
    return fd, cos, wcos, scos


def bench_load_and_run(path, token_sets, n=5):
    t0 = time.perf_counter()
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _, ids in token_sets[:n]:
        run_mel(sess, ids)
    run_ms = (time.perf_counter() - t0) / min(n, len(token_sets)) * 1000
    return round(load_s, 3), round(run_ms, 1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='تكميم نسخ ONNX')
    ap.add_argument('--gold', default=os.path.join(HERE, 'golden_texts.json'))
    ap.add_argument('--skip-vocos', action='store_true')
    args = ap.parse_args()

    from onnxconverter_common import float16
    from onnxruntime.quantization import (
        quant_pre_process, quantize_dynamic, quantize_static, QuantType,
        QuantFormat, CalibrationDataReader, CalibrationMethod)

    fp32_path = os.path.join(MODELS_DIR, 'mixertts_fp32.onnx')
    if not os.path.exists(fp32_path):
        raise SystemExit('[خطأ] شغّل export_onnx.py أولًا.')

    calib = golden_token_sets(args.gold)
    print(f'[quant] golden token sets: {len(calib)}')

    sess32 = ort.InferenceSession(fp32_path,
                                  providers=['CPUExecutionProvider'])
    refs = [(gid, ids, run_mel(sess32, ids)) for gid, ids in calib]

    vocos_src = os.path.join(INF_DIR, 'weights', 'vocos22.onnx')
    vocos_sess = None
    if os.path.exists(vocos_src):
        vocos_sess = ort.InferenceSession(
            vocos_src, providers=['CPUExecutionProvider'])

    # pre-process للتكميم (بلا استنتاج شكلي رمزي — الرموز الديناميكية تكسره)
    pp_path = os.path.join(MODELS_DIR, '_pre.onnx')
    print('[quant] pre-processing ...')
    quant_pre_process(fp32_path, pp_path, skip_optimization=False,
                      skip_onnx_shape=False, skip_symbolic_shape=True)
    pp_model = onnx.load(pp_path)
    pp_preds = node_names_containing(pp_model, 'duration_predictor',
                                     'pitch_predictor')

    with open(os.path.join(MODELS_DIR, 'mixertts_fp32.meta.json'),
              encoding='utf-8') as f:
        fp32_meta = json.load(f)
    base = {
        'role': 'acoustic', 'checkpoint': fp32_meta.get('checkpoint', '?'),
        'tag': fp32_meta.get('tag', 'production'),
        'inputs': fp32_meta['inputs'], 'outputs': fp32_meta['outputs'],
        'net_config': fp32_meta['net_config'],
    }

    manifest = []
    load_s, run_ms = bench_load_and_run(fp32_path, calib)
    manifest.append({
        **base, 'id': 'fp32', 'file': 'mixertts_fp32.onnx',
        'precision': 'fp32', 'size_bytes': os.path.getsize(fp32_path),
        'sha256': sha256_of(fp32_path),
        'server_bench': {'load_s': load_s, 'avg_run_ms': run_ms},
        'parity': {'verdict': 'reference',
                   'note_ar': 'المرجع — بقية النسخ تقارن به'},
        'notes_ar': 'المرجع الكامل الدقة (ناتج التصدير مباشرة)',
    })

    # ---- FP16 --------------------------------------------------------------
    print('[quant] FP16 ...')
    try:
        mclean = onnx.load(fp32_path)
        strip_identity(mclean)
        m16 = float16.convert_float_to_float16(
            mclean, keep_io_types=True,
            node_block_list=fp16_blocklist(mclean))
        p16 = os.path.join(MODELS_DIR, 'mixertts_fp16.onnx')
        onnx.save(m16, p16)
        sess16 = ort.InferenceSession(p16,
                                      providers=['CPUExecutionProvider'])
        rows = parity_rows(sess16, refs, vocos_sess)
        fd, cos, wcos, scos = summarize(rows)
        load_s, run_ms = bench_load_and_run(p16, calib)
        verdict = ('PASS' if fd == 0 and (cos or 0) > 0.9999 else 'PARTIAL')
        manifest.append({
            **base, 'id': 'fp16', 'file': 'mixertts_fp16.onnx',
            'precision': 'fp16', 'size_bytes': os.path.getsize(p16),
            'sha256': sha256_of(p16),
            'server_bench': {'load_s': load_s, 'avg_run_ms': run_ms},
            'parity': {'verdict': verdict, 'frames_diff_max': fd,
                       'mel_cos_min': cos, 'wav_cos_min': wcos,
                       'spec_cos_min': scos,
                       'rows': rows},
            'notes_ar': ('نصف الدقة مع حماية عقد Cast والشكل والمتنبئات — '
                         'حجم ~63% من FP32'),
        })
        print(f'  fp16: {os.path.getsize(p16)/1e6:.1f}MB fd={fd} '
              f'cos={cos} wav_cos={wcos} spec_cos={scos}')
    except Exception as e:                                      # noqa: BLE001
        manifest.append({**base, 'id': 'fp16', 'file': 'mixertts_fp16.onnx',
                         'precision': 'fp16', 'status': 'failed',
                         'error': f'{type(e).__name__}: {e}'})

    # ---- INT8 dynamic -------------------------------------------------------
    print('[quant] INT8 dynamic ...')
    try:
        p8d = os.path.join(MODELS_DIR, 'mixertts_int8dyn.onnx')
        quantize_dynamic(pp_path, p8d, weight_type=QuantType.QInt8,
                         nodes_to_exclude=pp_preds)
        s8d = ort.InferenceSession(p8d,
                                   providers=['CPUExecutionProvider'])
        rows = parity_rows(s8d, refs, vocos_sess)
        fd, cos, wcos, scos = summarize(rows)
        load_s, run_ms = bench_load_and_run(p8d, calib)
        verdict = ('PASS' if fd <= 2 and (scos or 0) > 0.97 else 'PARTIAL')
        manifest.append({
            **base, 'id': 'int8dyn', 'file': 'mixertts_int8dyn.onnx',
            'precision': 'int8-dynamic',
            'size_bytes': os.path.getsize(p8d), 'sha256': sha256_of(p8d),
            'server_bench': {'load_s': load_s, 'avg_run_ms': run_ms},
            'parity': {'verdict': verdict, 'frames_diff_max': fd,
                       'mel_cos_min': cos, 'wav_cos_min': wcos,
                       'spec_cos_min': scos,
                       'tolerance_ar': ('frames<=2 و spec_cos>0.97 (ارتباط '
                                        'الطيف — مقاوم للانزياح الزمني؛ '
                                        'wav_cos الخام ينهار عند انزياح '
                                        'إطار واحد ~23مل بلا أثر إدراكي)'),
                       'rows': rows},
            'notes_ar': 'تكميم ديناميكي INT8 (أوزان QInt8، المتنبئات محمية)',
        })
        print(f'  int8dyn: {os.path.getsize(p8d)/1e6:.1f}MB fd={fd} '
              f'cos={cos} wav_cos={wcos} spec_cos={scos}')
    except Exception as e:                                      # noqa: BLE001
        manifest.append({**base, 'id': 'int8dyn',
                         'file': 'mixertts_int8dyn.onnx',
                         'precision': 'int8-dynamic', 'status': 'failed',
                         'error': f'{type(e).__name__}: {e}'})

    # ---- INT8 static --------------------------------------------------------
    print('[quant] INT8 static (QDQ, معايرة ذهبية) ...')

    class Reader(CalibrationDataReader):
        def __init__(self, items):
            self.items = list(items)
            self.i = 0

        def get_next(self):
            if self.i >= len(self.items):
                return None
            _, ids = self.items[self.i]
            self.i += 1
            return {'text': np.array([ids], dtype=np.int64),
                    'pace': np.array(1.0, dtype=np.float32),
                    'speaker': np.array(0, dtype=np.int64)}

    try:
        p8s = os.path.join(MODELS_DIR, 'mixertts_int8static.onnx')
        quantize_static(pp_path, p8s, Reader(calib),
                        calibrate_method=CalibrationMethod.MinMax,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8, per_channel=True,
                        nodes_to_exclude=pp_preds)
        s8s = ort.InferenceSession(p8s,
                                   providers=['CPUExecutionProvider'])
        rows = parity_rows(s8s, refs, vocos_sess)
        fd, cos, wcos, scos = summarize(rows)
        load_s, run_ms = bench_load_and_run(p8s, calib)
        verdict = ('PASS' if fd <= 4 and (scos or 0) > 0.95 else 'PARTIAL')
        manifest.append({
            **base, 'id': 'int8static', 'file': 'mixertts_int8static.onnx',
            'precision': 'int8-static-qdq',
            'size_bytes': os.path.getsize(p8s), 'sha256': sha256_of(p8s),
            'server_bench': {'load_s': load_s, 'avg_run_ms': run_ms},
            'parity': {'verdict': verdict, 'frames_diff_max': fd,
                       'mel_cos_min': cos, 'wav_cos_min': wcos,
                       'spec_cos_min': scos,
                       'tolerance_ar': ('frames<=4 و spec_cos>0.95 (ارتباط '
                                        'الطيف — انظر ملاحظة int8dyn)'),
                       'rows': rows},
            'notes_ar': ('تكميم ثابت QDQ بمعايرة MinMax على توكنز النصوص '
                         'الذهبية (per-channel)'),
        })
        print(f'  int8static: {os.path.getsize(p8s)/1e6:.1f}MB fd={fd} '
              f'cos={cos} wav_cos={wcos} spec_cos={scos}')
    except Exception as e:                                      # noqa: BLE001
        manifest.append({**base, 'id': 'int8static',
                         'file': 'mixertts_int8static.onnx',
                         'precision': 'int8-static-qdq', 'status': 'failed',
                         'error': f'{type(e).__name__}: {e}'})

    # ---- المُصوِّت vocos ------------------------------------------------------
    vocos_entries = []
    if os.path.exists(vocos_src):
        import shutil
        vocos_dst = os.path.join(MODELS_DIR, 'vocos22.onnx')
        if not os.path.exists(vocos_dst):
            shutil.copy2(vocos_src, vocos_dst)
        vocos_entries.append({
            'role': 'vocoder', 'id': 'fp32', 'file': 'vocos22.onnx',
            'size_bytes': os.path.getsize(vocos_dst),
            'sha256': sha256_of(vocos_dst),
            'parity': {'verdict': 'as-is',
                       'note_ar': 'نفس ملف الإنتاج weights/vocos22.onnx'},
            'notes_ar': 'المُصوِّت الأصلي — بلا أي تعديل',
        })
        if not args.skip_vocos:
            mel_ex = refs[0][2]
            feed = {'mel_spec': mel_ex.astype('float32'),
                    'denoise': np.array([0.005], dtype='float32')}
            ref_w = vocos_sess.run(None, feed)[0]
            for vid, fname, note in [
                ('fp16', 'vocos22_fp16.onnx', 'المُصوِّت fp16'),
                ('int8dyn', 'vocos22_int8dyn.onnx',
                 'المُصوِّت INT8 ديناميكي'),
            ]:
                dst = os.path.join(MODELS_DIR, fname)
                try:
                    if vid == 'fp16':
                        vm = onnx.load(vocos_src)
                        strip_identity(vm)
                        vm16 = float16.convert_float_to_float16(
                            vm, keep_io_types=True,
                            node_block_list=fp16_blocklist(vm))
                        onnx.save(vm16, dst)
                    else:
                        quantize_dynamic(vocos_src, dst,
                                         weight_type=QuantType.QInt8)
                    sv = ort.InferenceSession(dst,
                                              providers=[
                                                  'CPUExecutionProvider'])
                    got_w = sv.run(None, feed)[0]
                    if got_w.shape == ref_w.shape:
                        d = float(np.abs(got_w - ref_w).max())
                        na, nb = np.linalg.norm(ref_w), np.linalg.norm(got_w)
                        cos = float(np.dot(ref_w.ravel(), got_w.ravel())
                                    / (na * nb))
                        ok = cos > 0.999
                    else:
                        d, cos, ok = None, None, False
                    vocos_entries.append({
                        'role': 'vocoder', 'id': vid, 'file': fname,
                        'size_bytes': os.path.getsize(dst),
                        'sha256': sha256_of(dst),
                        'parity': {'verdict': 'PASS' if ok else 'FAIL',
                                   'max_abs_diff': round(d, 6)
                                   if d is not None else None,
                                   'wav_cos': round(cos, 6)
                                   if cos is not None else None},
                        'notes_ar': note,
                    })
                    print(f'  vocos {vid}: {os.path.getsize(dst)/1e6:.1f}MB '
                          f'cos={round(cos,6) if cos else "?"}')
                except Exception as e:                          # noqa: BLE001
                    vocos_entries.append({
                        'role': 'vocoder', 'id': vid, 'file': fname,
                        'status': 'failed',
                        'error': f'{type(e).__name__}: {e}',
                        'notes_ar': note,
                    })
                    print(f'  vocos {vid}: FAILED {e}')

    # ---- catt_eo (تشكيل — كما هو) -------------------------------------------
    catt_src = os.path.join(INF_DIR, 'lib', 'tts_arabic', 'data',
                            'catt_eo.onnx')
    if os.path.exists(catt_src):
        import shutil
        catt_dst = os.path.join(MODELS_DIR, 'catt_eo.onnx')
        if not os.path.exists(catt_dst):
            shutil.copy2(catt_src, catt_dst)
        vocos_entries.append({
            'role': 'tashkeel', 'id': 'fp32', 'file': 'catt_eo.onnx',
            'size_bytes': os.path.getsize(catt_dst),
            'sha256': sha256_of(catt_dst),
            'parity': {'verdict': 'as-is',
                       'note_ar': 'نفس ملف الإنتاج — بلا أي تعديل'},
            'notes_ar': 'المُشكِّل catt_eo الأصلي (ONNX من الأساس)',
        })

    # ---- manifest ------------------------------------------------------------
    out = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'acoustic_variants': manifest,
        'other_models': vocos_entries,
        'int4': {
            'status': 'not-supported-here',
            'note_ar': ('INT4 (MatMulNBits) غير متاح في أدوات onnxruntime '
                        'بهذه البيئة — لم يُنشأ ولا يُدَّعى دعمه.'),
        },
    }
    mpath = os.path.join(MODELS_DIR, 'manifest.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'[quant] manifest: {mpath}')
    for e in manifest + vocos_entries:
        if 'file' in e:
            print(f"  {e.get('role','?'):<9} {e['id']:<11} "
                  f"{e.get('size_bytes',0)/1e6:6.1f}MB  "
                  f"{e.get('parity',{}).get('verdict','?')}")

    # تنظيف الملفات الوسيطة
    for tmp in ('_pre.onnx',):
        p = os.path.join(MODELS_DIR, tmp)
        if os.path.exists(p):
            os.remove(p)


if __name__ == '__main__':
    main()
