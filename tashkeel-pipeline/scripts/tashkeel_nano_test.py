# -*- coding: utf-8 -*-
"""tashkeel_nano_test.py — Track B feasibility probe for
ahmadsy/tashkeel-nano-4p5m (4.49M params, 17.2MB fp32 safetensors).

Answers THREE questions (NO actual fine-tuning is performed or persisted):
  1) Does the model load and run locally in this environment? (local files,
     no hub dependency; strict state-dict load; param count check)
  2) Zero-shot quality on OUR domain: run it on the approved smoke-sample
     raw texts and measure per-letter label agreement with our guide-based
     approved outputs (Egyptian conventions), plus tanween/i'rab statistics.
  3) Fine-tuning cost on THIS machine (2 CPU cores): a SPEED BENCHMARK ONLY
     (forward+backward+AdamW on random tensors, in-memory, nothing saved) to
     extrapolate wall-time for 1k/2k/5k/10k-sentence fine-tunes.

Report: work/tashkeel_nano/feasibility_report.json (+ stdout summary).
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path('/home/z/my-project/work/tashkeel_nano')
sys.path.insert(0, str(REPO))

from tashkeel_nano.inference import TashkeelNano, strip_marks  # noqa: E402
from tashkeel_nano.model import build_from_config  # noqa: E402
from tashkeel_nano.tokenizer import VOCAB_SIZE  # noqa: E402
from tashkeel_nano.passthrough import parse, is_arabic_base  # noqa: E402
from tashkeel_nano.labels import CLASS_NAMES  # noqa: E402

torch.set_num_threads(2)

OUT = {}
ok_fail = []


def note(key, val):
    OUT[key] = val


# ---------------------------------------------------------------- 1) loading
cfg = json.loads((REPO / 'config.json').read_text(encoding='utf-8'))
model_cfg = {'model': {
    'hidden': cfg['hidden_size'], 'layers': cfg['num_hidden_layers'],
    'n_heads': cfg['num_attention_heads'], 'n_kv': cfg['num_key_value_heads'],
    'ffn': cfg['intermediate_size'], 'ctx': cfg['max_position_embeddings'],
    'dropout': cfg.get('dropout', 0.0), 'causal': cfg.get('causal', False),
    'kv_pattern': cfg.get('kv_pattern', 'new'),
    'shared_layers': cfg.get('shared_layers', 0),
}}
model = build_from_config(model_cfg, vocab_size=VOCAB_SIZE)

from safetensors.torch import load_file  # noqa: E402
sd = load_file(str(REPO / 'model.safetensors'))
missing, unexpected = model.load_state_dict(sd, strict=False)[:2]
strict_ok = not missing and not unexpected
n_params = sum(p.numel() for p in model.parameters())
note('load', {
    'config': {k: cfg[k] for k in ('vocab_size', 'hidden_size',
                                   'num_hidden_layers', 'num_attention_heads',
                                   'num_key_value_heads', 'intermediate_size',
                                   'max_position_embeddings', 'num_classes')},
    'state_dict_keys': len(sd),
    'strict_load_clean': strict_ok,
    'params': n_params,
    'params_match_card': n_params == 4488160,
    'weights_file_bytes': (REPO / 'model.safetensors').stat().st_size,
})
ok_fail.append(('local load (strict, no hub)', strict_ok and n_params == 4488160))

nano = TashkeelNano(model, ctx=cfg['max_position_embeddings'], device='cpu')

# sanity: the README's own example
ex = nano.diacritize('السلام عليكم ورحمة الله')
note('readme_example', {'in': 'السلام عليكم ورحمة الله', 'out': ex})
ok_fail.append(('runs inference (README example)', 'السَّلَامُ' in ex or len(ex) > 20))

# ------------------------------------------------- 2) zero-shot on OUR domain
smoke = [json.loads(l) for l in open(
    '/home/z/my-project/work/ph3_smoke2_output.jsonl', encoding='utf-8')
    if l.strip()]
refs = {r['id']: r for r in smoke if r['status'] == 'ok' and r.get('out')}

TANWEEN_CLASSES = {5, 6, 7, 12, 13, 14}
VOWEL_CLASSES = {1, 2, 3}  # fatha/damma/kasra


def base_seq(units):
    return ''.join(u.raw for u in units if u.is_target)


def word_final_mask(units):
    """True at the LAST base letter of each word (word = run of target units
    not separated by whitespace passthrough)."""
    seq = []
    prev_target = False
    for i, u in enumerate(units):
        if u.is_target:
            nxt = units[i + 1] if i + 1 < len(units) else None
            is_final = not (nxt and nxt.is_target)
            seq.append(is_final)
            prev_target = True
        else:
            if prev_target and any(c.isspace() for c in u.raw) and seq:
                pass  # word boundary after a target run
            prev_target = False
    # recompute properly: a target unit is word-final if the next unit is not
    # a target (i.e., followed by passthrough/whitespace/end)
    out = []
    for i, u in enumerate(units):
        if not u.is_target:
            continue
        nxt = units[i + 1] if i + 1 < len(units) else None
        out.append(not (nxt and nxt.is_target))
    return out


per_unit = []
agree_all = tanw_pred = tanw_ref = wf_total = wf_agree = 0
wf_ref_sukun_or_bare = wf_nano_vowel = wf_nano_tanween = 0
label_conf = {}
for uid, r in refs.items():
    raw_in = r['converted']  # ASCII-punct converted text (our pipeline input)
    pred = nano.diacritize(raw_in)
    ru = parse(r['out'])
    pu = parse(pred)
    # both must share the same base-letter skeleton
    if base_seq(ru) != base_seq(pu):
        per_unit.append({'id': uid, 'error': 'base-sequence mismatch'})
        continue
    rl = [u.label for u in ru if u.is_target]
    pl = [u.label for u in pu if u.is_target]
    wf = word_final_mask(ru)
    a = sum(1 for x, y in zip(rl, pl) if x == y)
    agree_all += a
    tanw_pred += sum(1 for x in pl if x in TANWEEN_CLASSES)
    tanw_ref += sum(1 for x in rl if x in TANWEEN_CLASSES)
    for x, y in zip(rl, pl):
        label_conf[(x, y)] = label_conf.get((x, y), 0) + 1
    for x, y, w in zip(rl, pl, wf):
        if w:
            wf_total += 1
            wf_agree += int(x == y)
            if x in (0, 4):
                wf_ref_sukun_or_bare += 1
                if y in VOWEL_CLASSES:
                    wf_nano_vowel += 1
                if y in TANWEEN_CLASSES:
                    wf_nano_tanween += 1
    per_unit.append({'id': uid, 'n_letters': len(rl),
                     'agree': a, 'agree_pct': round(100 * a / max(1, len(rl)), 1),
                     'pred_tanween': sum(1 for x in pl if x in TANWEEN_CLASSES),
                     'ref_tanween': sum(1 for x in rl if x in TANWEEN_CLASSES),
                     'pred': pred, 'ref': r['out']})

n_letters = sum(u['n_letters'] for u in per_unit if 'n_letters' in u)
note('zero_shot_vs_approved_smoke', {
    'n_units': len([u for u in per_unit if 'n_letters' in u]),
    'n_letters': n_letters,
    'letter_agreement_pct': round(100 * agree_all / max(1, n_letters), 1),
    'tanween_positions_pred': tanw_pred,
    'tanween_positions_ref': tanw_ref,
    'word_final': {
        'n': wf_total,
        'agreement_pct': round(100 * wf_agree / max(1, wf_total), 1),
        'ref_is_bare_or_sukun': wf_ref_sukun_or_bare,
        'nano_predicts_vowel_where_ref_bare_sukun': wf_nano_vowel,
        'nano_predicts_tanween_where_ref_bare_sukun': wf_nano_tanween,
        'ref_no_i3rab_rate_pct': round(
            100 * wf_ref_sukun_or_bare / max(1, wf_total), 1),
    },
    'per_unit': per_unit,
})

# ------------------------------------------- 3) SPEED benchmark (NOT training)
# In-memory, random tensors, NOTHING saved, model weights left untouched:
# this measures only how long a train STEP takes on this 2-core CPU.
B, T = 32, 128
bench_model = build_from_config(model_cfg, vocab_size=VOCAB_SIZE)
bench_model.load_state_dict(sd, strict=True)
bench_model.train()
opt = torch.optim.AdamW(bench_model.parameters(), lr=2e-4, weight_decay=0.1)
ids = torch.randint(4, VOCAB_SIZE, (B, T))
lab = torch.full((B, T), -1, dtype=torch.long)
mask = torch.rand(B, T) < 0.62  # ~62% of chars are Arabic targets
lab[mask] = torch.randint(0, 15, (int(mask.sum()),))
times = []
for step in range(12):
    t0 = time.perf_counter()
    out = bench_model(ids, labels=lab)
    out['loss'].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times.append((time.perf_counter() - t0) * 1000)
warm, measured = times[:2], times[2:]
ms_step = sorted(measured)[len(measured) // 2]
note('train_step_benchmark', {
    'batch': B, 'ctx': T, 'n_measured_steps': len(measured),
    'median_ms_per_step': round(ms_step, 1),
    'note': 'speed benchmark on random tensors in memory; no weights saved, '
            'no fine-tuning performed; published checkpoint untouched',
    'extrapolation': {
        f'{n}_sentences_{e}ep': {
            'steps': -(-n * e // B),
            'hours': round(-(-n * e // B) * ms_step / 1000 / 3600, 2)}
        for n in (1000, 2000, 5000, 10000) for e in (3,)
    },
})
# inference speed
t0 = time.perf_counter()
for uid, r in list(refs.items())[:5]:
    nano.diacritize(r['converted'])
inf_s = (time.perf_counter() - t0) / 5
note('inference_speed', {
    'median_s_per_sentence': round(inf_s, 3),
    'sentences_per_min_per_core_pair': round(60 / inf_s, 1),
})

note('environment', {
    'torch': torch.__version__, 'device': 'cpu',
    'cores': 2, 'ram_gb': 3.9,
    'model_card_sha256_ignored': None,
})
note('license_note', 'research-only per model card (training mix includes '
     'Tashkeela/GPL-2-derived data) — flagged for the user decision')

Path('/home/z/my-project/work/tashkeel_nano').mkdir(exist_ok=True)
with open('/home/z/my-project/work/tashkeel_nano/feasibility_report.json', 'w',
          encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

print('=== 1) LOCAL LOAD ===')
print(f"  strict load clean: {strict_ok} | params: {n_params:,} "
      f"(card: 4,488,160) | weights: 17.2MB")
print(f"  README example -> {ex}")
print('=== 2) ZERO-SHOT vs APPROVED SMOKE (Egyptian guide) ===')
z = OUT['zero_shot_vs_approved_smoke']
print(f"  units={z['n_units']} letters={z['n_letters']} "
      f"agreement={z['letter_agreement_pct']}%")
print(f"  tanween: nano={z['tanween_positions_pred']} vs ref="
      f"{z['tanween_positions_ref']}")
wf = z['word_final']
print(f"  word-final: agreement={wf['agreement_pct']}% | ref bare/sukun rate="
      f"{wf['ref_no_i3rab_rate_pct']}% | nano puts vowel where ref has "
      f"bare/sukun: {wf['nano_predicts_vowel_where_ref_bare_sukun']}, "
      f"tanween: {wf['nano_predicts_tanween_where_ref_bare_sukun']}")
for u in per_unit[:4]:
    if 'pred' in u:
        print(f"  [{u['id']}] agree={u['agree_pct']}%")
        print(f"    pred: {u['pred'][:90]}")
        print(f"    ref : {u['ref'][:90]}")
print('=== 3) TRAIN-STEP SPEED BENCHMARK (no training performed) ===')
b = OUT['train_step_benchmark']
print(f"  batch 32x128: {b['median_ms_per_step']} ms/step (median of "
      f"{b['n_measured_steps']})")
for k, v in b['extrapolation'].items():
    print(f"  {k}: {v['steps']} steps ≈ {v['hours']} h")
print(f"  inference: {OUT['inference_speed']['median_s_per_sentence']}s/sentence")
print()
for label, good in ok_fail:
    print(f"  [{'OK' if good else 'FAIL'}] {label}")
