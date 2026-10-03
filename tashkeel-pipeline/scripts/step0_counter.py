# -*- coding: utf-8 -*-
"""Step 0 — exact replica of the NileTTS 4h Prep kernel token counter.

Pipeline (verbatim from niletts-4h-prep.py, _lazy_load / plan_tokens_one):
    catt.predict(text)                       -> diacritized text (catt_eo)
    arabic_to_buckwalter(voc)                -> Buckwalter ASCII
    buckwalter_to_phonemes(bw)  = process_utterance(bw)
    phonemes_to_tokens(phonemes)             -> token list (+ separator/EOS)
    EGY_TOKEN_MAP replacement                -> Egyptian phoneme set
    tokens_to_ids(toks)                      -> ids; len(ids) = token count
"""
import sys

TTS_PKG = '/home/z/my-project/work/tts_arabic_pkg'
if TTS_PKG not in sys.path:
    sys.path.insert(0, TTS_PKG)

from tts_arabic.text import (
    arabic_to_buckwalter, tokens_to_ids, phonemes_to_tokens,
    buckwalter_to_phonemes)

EGY_TOKEN_MAP = {'j': 'v', 'q': '<', '^': 't', '*': 'd'}   # prep kernel line 73


def toks_egy(text):
    toks = phonemes_to_tokens(
        buckwalter_to_phonemes(arabic_to_buckwalter(text)))
    return [EGY_TOKEN_MAP.get(t, t) for t in toks]


def n_tokens_of(text):
    """Token count of a (possibly diacritized) Arabic text — the exact
    function the Prep kernel applies AFTER catt.predict()."""
    return len(tokens_to_ids(toks_egy(text)))


def get_catt():
    from tts_arabic.vocalizer.models.core import get_model
    return get_model('catt_eo')


def catt_n_tokens(catt, raw_text):
    voc = catt.predict(raw_text)
    return n_tokens_of(voc), voc


if __name__ == '__main__':
    # self-test on the first 3 units of extraction.csv with recorded counts
    import pandas as pd
    df = pd.read_csv('/home/z/my-project/work/prep_output/extraction.csv')
    catt = get_catt()
    print('catt loaded OK')
    for i in range(3):
        r = df.iloc[i]
        n, voc = catt_n_tokens(catt, r.transcript)
        print(f"utt={r.utt} recorded={r.n_tokens} replica={n} "
              f"match={n == r.n_tokens}")
        print('  catt out:', voc[:90])
