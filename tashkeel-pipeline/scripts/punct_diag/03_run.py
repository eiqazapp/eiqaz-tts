# -*- coding: utf-8 -*-
"""Punctuation-alignment diagnostic — 03: run the alignment mechanism.

For each of the 20 selected NileTTS-4h units (audio UNTOUCHED — read only):

  A) BASELINE (exact training conditions of the "NileTTS 4h Train" kernel):
     transcript -> catt.predict -> EGY tokens -> ids ; mel from wav slice.
     Run model.forward with spect=mel (teacher forcing) exactly like
     phase_train's make_batch, and record attn_hard_dur.

  B) PUNCTUATED (text-only change, audio identical):
     same as A but ASCII ',' / '?' tokens (ids 6/7 in the tts_arabic
     vocab, present but UNTRAINED in the 4h checkpoint) are inserted at
     the word positions where the original transcript had Arabic comma /
     question marks. Same forward -> attn_hard_dur.

  C) ORIGINAL MSA mixer128 (author-pretrained; its pipeline preserved
     punctuation via espeak preserve_punctuation=True): same clips,
     tokens mapped to the mixer_repo 148-char vocabulary (its ',' and '?'
     rows WERE seen in the author's English training). RMS-normalized
     wav (-27 dB, author's lj_dataset treatment) for its mel.

Also saved: per-frame RMS(dB) of the wav (for silence detection later),
and the soft-attention row of every punctuation token.

Everything is saved to work/punct_diag/results/raw_<utt>.npz so the
analysis (04) can re-tune the silence threshold without re-running.
NO training, NO GPU, NO writes to any dataset.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

MIXER = '/home/z/my-project/work/scratch_data/mixer_repo'
TTSPKG = '/home/z/my-project/work/tts_arabic_pkg'
for p in (MIXER, TTSPKG):
    if p not in sys.path:
        sys.path.insert(0, p)

SEL = '/home/z/my-project/work/punct_diag/selection.json'
WAV_DIR = '/home/z/my-project/work/punct_diag/wav'
RES_DIR = '/home/z/my-project/work/punct_diag/results'
CKPT_4H = '/home/z/my-project/work/train_output/checkpoints/states_79590.pth'
CKPT_MSA = '/home/z/my-project/work/scratch_data/models/mixer128_pytorch.pth'

NET_CONFIG_OVERRIDES = {
    'num_tokens': 148, 'padding_idx': 0, 'symbols_embedding_dim': 128,
    'n_speakers': 16, 'n_emotions': 16, 'energy_conditioning': False,
}
EGY_TOKEN_MAP = {'j': 'v', 'q': '<', '^': 't', '*': 'd'}
TARGET_SR = 22050
ARABIC_TO_ASCII = {'\u060c': ',', '\u061f': '?'}   # comma, question

import librosa  # noqa: E402


def toks_egy(text):
    from tts_arabic.text import (arabic_to_buckwalter, phonemes_to_tokens,
                                 buckwalter_to_phonemes)
    toks = phonemes_to_tokens(
        buckwalter_to_phonemes(arabic_to_buckwalter(text)))
    return [EGY_TOKEN_MAP.get(t, t) for t in toks]


def load_audio(unit):
    """Episode wav -> mono -> 22.05 kHz -> peak-norm(>0.95) -> slice.
    Identical to niletts-4h-prep.py lines 695-706."""
    wav, sr = sf.read(os.path.join(WAV_DIR, unit['src_row'] + '.wav'),
                      dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.abs(wav).max())
    if peak > 0.95:
        wav = wav * (0.95 / peak)
    return np.ascontiguousarray(wav[unit['i0']:unit['i1']])


def frame_rms_db(wav, n_frames, hop=256, win=1024, pad=384):
    """RMS(dBFS) per mel frame (MelSpectrogram uses center=False with
    reflect padding of (n_fft-hop)/2=384, so frame f covers
    wav[f*hop-pad : f*hop-pad+win])."""
    out = np.full(n_frames, -120.0)
    for f in range(n_frames):
        a = max(0, f * hop - pad)
        b = min(len(wav), f * hop - pad + win)
        if b > a:
            out[f] = 20 * np.log10(np.sqrt(np.mean(wav[a:b] ** 2)) + 1e-9)
    return out


def build_model(load_ckpt=None, strict=True):
    from models.mixer_tts.mixer_tts import MixerTTSModel
    from models.mixer_tts import net_config
    net_config.update(NET_CONFIG_OVERRIDES)
    m = MixerTTSModel(**net_config)
    m.add_bin_loss = True
    m.bin_loss_scale = 1.0
    if load_ckpt is not None:
        ck = torch.load(load_ckpt, map_location='cpu', weights_only=False)
        sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        res = m.load_state_dict(sd, strict=strict)
        if not strict:
            print(f'  [load {os.path.basename(load_ckpt)}] '
                  f'missing={len(res.missing_keys)} '
                  f'unexpected={len(res.unexpected_keys)}')
            for k in res.missing_keys[:6]:
                print('    missing:', k)
    return m.eval()


def run_aligner(model, ids, mel, spk, betabin):
    """Exactly phase_train's forward call (B=1, eval mode, no_grad)."""
    L, T = len(ids), mel.shape[1]
    text_p = torch.LongTensor([ids])
    in_len = torch.LongTensor([L])
    mel_p = mel[None]
    out_len = torch.LongTensor([T])
    prior = torch.from_numpy(betabin(T, L)).float()[None]
    with torch.no_grad():
        out = model(text=text_p, text_len=in_len, pitch=None, energy=None,
                    spect=mel_p, spect_len=out_len, attn_prior=prior,
                    speaker=torch.LongTensor([spk]),
                    emotion=torch.zeros(1, dtype=torch.long))
    (pred_spect, _, log_durs, pitch_pred, _, attn_soft, attn_logprob,
     attn_hard, attn_hard_dur) = out
    return {
        'dur': attn_hard_dur[0].numpy().astype(np.int64),
        'attn_soft': attn_soft[0, 0].numpy(),  # (T, L)
    }


def mix_token_ids(tok, mix_sym):
    """EGY token -> list of mixer_repo vocabulary char ids
    (author's char-level input format)."""
    if tok == '_+_':
        return [mix_sym[' ']]
    if tok in ('_eos_', '_dbl_'):
        return []
    return [mix_sym[c] for c in tok if c in mix_sym]


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    from tts_arabic.vocalizer.models.core import get_model
    from tts_arabic.text import tokens_to_ids
    from models.mixer_tts.modules.data_function import (
        BetaBinomialInterpolator)
    from models.symbols import symbols_to_id as mix_sym
    from utils.audio import MelSpectrogram

    with open(SEL, encoding='utf-8') as f:
        units = json.load(f)['units']

    catt = get_model('catt_eo')
    mel_fn = MelSpectrogram()
    betabin = BetaBinomialInterpolator(round_mel_len_to=200,
                                       round_text_len_to=40)

    print('building 4h scratch model (iter 79590)...')
    m4h = build_model(CKPT_4H, strict=True)
    print('building original MSA mixer128 model...')
    mmsa = build_model(CKPT_MSA, strict=False)

    punct_ids = {5: '.', 6: ',', 7: '?', 8: '!'}
    report = []
    for ui, unit in enumerate(units):
        utt = unit['utt']
        raw = load_audio(unit)
        wav_t = torch.from_numpy(raw)[None]
        mel = mel_fn(wav_t).clamp_min(1e-5).log().squeeze(0)   # [80, T]
        T = mel.shape[1]
        if T != unit['n_frames']:
            print(f'  [warn] {utt}: mel frames {T} != recorded '
                  f'{unit["n_frames"]}')
        rms_db = frame_rms_db(raw, T)

        # ---- baseline ids (exact training path) ----
        voc = catt.predict(unit['transcript'])
        toks_base = toks_egy(voc)
        ids_base = tokens_to_ids(toks_base)
        ok_tok = (len(ids_base) == unit['n_tokens'])

        # ---- punctuated variant (text-only change) ----
        orig_words = unit['transcript'].split()
        voc_words = voc.split()
        marks = unit['marks']
        inserted = []
        if len(voc_words) == len(orig_words):
            for wi, ch in marks:
                if wi < len(voc_words):
                    voc_words[wi] = voc_words[wi] + ARABIC_TO_ASCII[ch]
            voc_punct = ' '.join(voc_words)
            toks_punct = toks_egy(voc_punct)
            ids_punct = tokens_to_ids(toks_punct)
            round_trip = ([i for i in ids_punct if i not in punct_ids]
                          == ids_base)
            inserted = [k for k, i in enumerate(ids_punct)
                        if i in punct_ids]
        else:
            ids_punct, round_trip = None, False

        # ---- 4h model alignments ----
        a_base = run_aligner(m4h, ids_base, mel, unit['spk_idx'], betabin)
        a_punct = (run_aligner(m4h, ids_punct, mel, unit['spk_idx'],
                               betabin)
                   if ids_punct is not None else None)

        # ---- mixer128 alignments (author's char vocab + rms -27dB) ----
        from utils.audio import rms_normalize
        wav_norm = rms_normalize(torch.from_numpy(raw)[None], -27)
        mel_msa = mel_fn(wav_norm).clamp_min(1e-5).log().squeeze(0)
        ids_mix_base = [i for t in toks_base for i in mix_token_ids(t,
                                                                    mix_sym)]
        ids_mix_punct = ([i for t in toks_punct for i in mix_token_ids(t,
                                                                       mix_sym)]
                         if ids_punct is not None else None)
        # mixer-vocab punctuation ids: ',' and '?'
        mix_punct_pos_base = [k for k, i in enumerate(ids_mix_base)
                              if mix_sym.get(',') == i
                              or mix_sym.get('?') == i]
        a_mix_base = run_aligner(mmsa, ids_mix_base, mel_msa,
                                 unit['spk_idx'], betabin)
        a_mix_punct = (run_aligner(mmsa, ids_mix_punct, mel_msa,
                                   unit['spk_idx'], betabin)
                       if ids_mix_punct is not None else None)

        # ---- soft attention rows of the inserted punctuation (4h) ----
        soft_rows = {}
        if a_punct is not None:
            for k in inserted:
                soft_rows[str(k)] = a_punct['attn_soft'][:, k].astype(
                    np.float32)

        np.savez_compressed(
            os.path.join(RES_DIR, f'raw_{utt}.npz'),
            ids_base=np.array(ids_base), ids_punct=np.array(ids_punct),
            ids_mix_base=np.array(ids_mix_base),
            ids_mix_punct=(np.array(ids_mix_punct)
                           if ids_mix_punct is not None else np.array([])),
            dur_base=a_base['dur'],
            dur_punct=(a_punct['dur'] if a_punct is not None
                       else np.array([])),
            dur_mix_base=a_mix_base['dur'],
            dur_mix_punct=(a_mix_punct['dur'] if a_mix_punct is not None
                           else np.array([])),
            rms_db=rms_db.astype(np.float32),
            inserted=np.array(inserted),
            **{f'soft_{k}': v for k, v in soft_rows.items()},
        )
        report.append({
            'utt': utt, 'spk': unit['spk_idx'], 'T': int(T),
            'L_base': len(ids_base),
            'L_punct': (len(ids_punct) if ids_punct is not None else None),
            'L_mix_base': len(ids_mix_base),
            'L_mix_punct': (len(ids_mix_punct)
                            if ids_mix_punct is not None else None),
            'n_tokens_match': bool(ok_tok), 'round_trip': bool(round_trip),
            'words_match': bool(len(voc_words) == len(orig_words)),
            'marks': marks,
            'inserted': inserted,
            'mix_punct_pos_base': mix_punct_pos_base,
            'rms_db_pct': {p: float(np.percentile(rms_db, p))
                           for p in (5, 25, 50, 75, 95)},
        })
        print(f'[{ui+1}/{len(units)}] {utt} spk{unit["spk_idx"]} '
              f'T={T} L={len(ids_base)}'
              f'+{len(inserted) if inserted else 0} '
              f'tok_match={ok_tok} rt={round_trip}')

    with open(os.path.join(RES_DIR, 'run_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    n_ok = sum(1 for r in report if r['round_trip'])
    print(f'DONE: {len(report)} units, round-trip OK: {n_ok}')


if __name__ == '__main__':
    main()
