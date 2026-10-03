# -*- coding: utf-8 -*-
"""تقرير عينة الدخان الرسمية (الدليل v2.1 / المحرك v5) — عدّ توكنز + كثافة + آليات."""
import json
import re
import sys

sys.path.insert(0, '/home/z/my-project/scripts')
from step0_counter import n_tokens_of, get_catt, catt_n_tokens

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')

inp = {r['id']: r for r in json.load(
    open('/home/z/my-project/work/ph3_smoke2_input.json', encoding='utf-8'))}
notes = json.load(open('/home/z/my-project/work/ph3_smoke2_notes.json',
                       encoding='utf-8'))
outs = [json.loads(l) for l in open(
    '/home/z/my-project/work/ph3_smoke2_output.jsonl', encoding='utf-8')]

catt = get_catt()

rows = []
for r in outs:
    uid = r['id']
    # for the letter-fix unit: token counts are on the CORRECTED text
    raw_for_count = r['raw']  # engine record raw = corrected when letter_fix
    rec = inp[uid]['n_tokens_recorded']
    if r['status'] == 'quarantined':
        rows.append({'id': uid, 'status': 'quarantined', 'note': notes[uid],
                     'raw': r['raw'], 'out': None})
        continue
    catt_n, catt_voc = catt_n_tokens(catt, raw_for_count)
    mine_n = n_tokens_of(r['out']) if r['out'] else None
    words = [w for w in r['out'].split()
             if any('\u0621' <= c <= '\u064A' for c in w)] if r['out'] else []
    density = len(TASH.findall(r['out'])) / max(1, len(words)) if r['out'] else 0
    rows.append({
        'id': uid, 'status': r['status'], 'attempts': r['attempts'],
        'note': notes[uid], 'raw': r['raw'], 'out': r['out'],
        'letter_fix': r.get('letter_fix'),
        'mechanisms': r.get('fixes', {}),
        'reason_failed': r.get('reason'),
        'n_tok_recorded_catt': rec, 'n_tok_catt_recheck': catt_n,
        'catt_eo': catt_voc,
        'n_tok_mine': mine_n,
        'ratio_mine_vs_catt': round(mine_n / catt_n, 3) if mine_n and catt_n else None,
        'density_marks_per_word': round(density, 2),
        'over_160': (mine_n > 160) if mine_n else None,
    })

json.dump(rows, open('/home/z/my-project/work/ph3_smoke2_report.json', 'w',
                     encoding='utf-8'), ensure_ascii=False, indent=1)

ok = [r for r in rows if r['status'] == 'ok']
fail = [r for r in rows if r['status'] == 'failed']
quar = [r for r in rows if r['status'] == 'quarantined']
print(f'ok={len(ok)}/{len(rows)-len(quar)} LLM | failed={len(fail)} | quarantined={len(quar)}')
print(f"{'id':14} {'st':6} {'att':4} {'catt':>5} {'mine':>5} {'ratio':>6} {'dens':>5} {'>160':>5}  mechanisms")
for r in rows:
    if r['status'] == 'quarantined':
        print(f"{r['id']:14} QUARANTINED — skipped, no LLM (حجر §ل-5)")
        continue
    mech = ','.join(k for k in r['mechanisms']) or '-'
    print(f"{r['id']:14} {r['status']:6} {r['attempts']:<4} "
          f"{r['n_tok_recorded_catt']:>5} {str(r['n_tok_mine']):>5} "
          f"{str(r['ratio_mine_vs_catt']):>6} {r['density_marks_per_word']:>5} "
          f"{str(r['over_160']):>5}  {mech}")
if ok:
    rs = [r['ratio_mine_vs_catt'] for r in ok]
    ds = [r['density_marks_per_word'] for r in ok]
    rs_s = sorted(rs)
    print(f'\nratio mine/catt: median={rs_s[len(rs_s)//2]:.3f} range={min(rs):.3f}-{max(rs):.3f}')
    print(f'density: median={sorted(ds)[len(ds)//2]:.2f} range={min(ds):.2f}-{max(ds):.2f}')
    print('over-160:', sum(1 for r in ok if r['over_160']))
    print('attempts total:', sum(r['attempts'] for r in ok), '| single-retry units:',
          sum(1 for r in ok if r['attempts'] > 1))
