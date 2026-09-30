# -*- coding: utf-8 -*-
"""Phase 1 ext — Rule 4: phonetic-safety detector.

Purpose: during the full tashkeel run, EVERY vocalized word in EVERY unit
is checked for accidental pronunciation collision (skeleton-level OR
phoneme-sequence-level) with inappropriate/embarrassing Arabic words.

Tier 1 — unambiguously vulgar words (list kept in this file; NOT printed
          in chat reports). Any hit => unit quarantined for human review.
Tier 2 — context-dependent embarrassing homophones (medical/legal terms
          that could embarrass out of context). Hits => review list with
          explanation; user decides per case.

Phoneme keys use the EXACT prep-kernel G2P replica (step0_counter).
This module is importable (detector functions) and runnable (demo/pre-scan).
"""
import json
import re
import sys

sys.path.insert(0, '/home/z/my-project/scripts')
from step0_counter import toks_egy

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,?!:;"()«»\u060C\u061F\u061B\u201C\u201D\u2026-'

# ---- Tier 1: unambiguous vulgar words (vocalized to fix pronunciation) ----
TIER1 = [
    'كَسّ', 'كُسّ', 'كَسْ', 'خَرَا', 'زَبّ', 'زُبْ', 'زَبْر', 'يَنِيكُ',
    'يَنِيكْ', 'نَاكَ', 'نِيكْ', 'قَحْبَة', 'شَرْمُوطَة', 'خُولْ',
    'مَتْنَاكْ', 'مِتْنَاكَة', 'عَرَّصْ', 'فَشْخْ', 'طَيْزَة', 'اِسْتَ',
    'اِسْتِ', 'إِسْتَ', 'قُوَّادْ', 'كَوَادَة',
]

# ---- Tier 2: context-dependent embarrassing homophones ----
TIER2 = [
    'فَرْجْ',      # legitimate: relief/opening — vulgar: female genital
    'دُبْرْ',      # legitimate: rear/Qur'anic (الليل إذا دبر) — vulgar: butt
    'عِيرْ',       # legitimate classical: caravan/whip — vulgar: penis
    'مَصَّ',       # legitimate verb: to suck (medicine) — crude in isolation
    'الشَّرَجْ',   # medical: rectum (anatomy) — embarrassing out of context
]


def phkey(word):
    """Phoneme-sequence key of a (vocalized) word via the prep G2P."""
    try:
        toks = toks_egy(word)
    except Exception:
        return None
    return tuple(t for t in toks if t not in ('_+_', '_eos_'))


_LEX1 = {w: phkey(w) for w in TIER1}
_LEX2 = {w: phkey(w) for w in TIER2}
_SK1 = {TASH.sub('', w) for w in TIER1}
_SK2 = {TASH.sub('', w) for w in TIER2}
_LEXALL = {**_LEX1, **_LEX2}


def check_vocalized_word(vw):
    """Return list of (tier, trigger-word, match-type) hits for a word."""
    if not vw:
        return []
    s = TASH.sub('', vw).strip(PUNCT)
    hits = []
    if s in _SK1:
        hits.append((1, s, 'spelling'))
    elif s in _SK2:
        hits.append((2, s, 'spelling'))
    k = phkey(vw)
    if k:
        for lw, lk in _LEXALL.items():
            if lk == k and TASH.sub('', lw) != s:
                hits.append((1 if lw in _LEX1 else 2, TASH.sub('', lw),
                             'phoneme'))
                break
    return hits


def check_unit(vocalized_text):
    """Check every word of a vocalized unit. Returns hits list."""
    out = []
    for w in vocalized_text.split():
        for tier, lw, mtype in check_vocalized_word(w):
            out.append({'word': w, 'tier': tier, 'matched': lw,
                        'match_type': mtype})
    return out


# ---------------- demo / pre-scan ----------------
if __name__ == '__main__':
    import importlib.util

    print(f'tier1 entries: {len(TIER1)} | tier2 entries: {len(TIER2)}')

    # (a) pre-scan raw corpus at spelling level (unvocalized skeletons)
    import csv
    rows = list(csv.DictReader(open(
        '/home/z/my-project/work/prep_output/extraction.csv', encoding='utf-8')))
    sk_all = set()
    for r in rows:
        for w in r['transcript'].split():
            sk_all.add(TASH.sub('', w).strip(PUNCT))
    raw_hits1 = sk_all & _SK1
    raw_hits2 = sk_all & _SK2
    print(f'raw corpus spelling pre-scan: tier1={len(raw_hits1)} '
          f'tier2={len(raw_hits2)} '
          f'{sorted(raw_hits2) if raw_hits2 else ""}')

    # (b) demo on the 50-sentence manual reference (MY50)
    spec = importlib.util.spec_from_file_location(
        'm50', '/home/z/my-project/scripts/step0_my_tashkeel.py')
    m50 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m50)
    hits = []
    for utt, voc in m50.MY50.items():
        for h in check_unit(voc):
            hits.append((utt, h))
    print(f'MY50 (50 manual): hits={len(hits)}')
    for utt, h in hits[:10]:
        print('  ', utt, h)

    # (c) demo on the 9 accepted smoke outputs
    hits = []
    for line in open('/home/z/my-project/work/ph3_smoke_output.jsonl',
                     encoding='utf-8'):
        rec = json.loads(line)
        if rec['status'] != 'ok' or not rec['out']:
            continue
        for h in check_unit(rec['out']):
            hits.append((rec['id'], h))
    print(f'smoke-9 (engine outputs): hits={len(hits)}')
    for uid, h in hits[:10]:
        print('  ', uid, h)

    print('\nSelf-test: phoneme-level detection works?')
    demo = ['كَسَّرَ', 'كَسَّ', 'فَرَجَ', 'فَرْجْ', 'شَرَحَ', 'الشَّرَجْ']
    for d in demo:
        print(f'  {d!r}: {check_vocalized_word(d)}')
