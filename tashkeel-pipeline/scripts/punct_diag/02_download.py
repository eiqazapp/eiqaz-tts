# -*- coding: utf-8 -*-
"""Punctuation-alignment diagnostic — 02: download source-row audio.

Downloads ONLY the episode wavs referenced by the selected units from
HuggingFace (KickItLikeShika/NileTTS-dataset). Read-only fetch; nothing
is uploaded or modified anywhere.
"""
import json
import os
import subprocess
import sys

SEL = '/home/z/my-project/work/punct_diag/selection.json'
WAV_DIR = '/home/z/my-project/work/punct_diag/wav'
BASE = ('https://huggingface.co/datasets/'
        'KickItLikeShika/NileTTS-dataset/resolve/main/wavs/')


def main():
    with open(SEL, encoding='utf-8') as f:
        rows = json.load(f)['rows']
    os.makedirs(WAV_DIR, exist_ok=True)
    ok, fail = [], []
    for r in rows:
        dst = os.path.join(WAV_DIR, r + '.wav')
        if os.path.exists(dst) and os.path.getsize(dst) > 44:
            ok.append(r)
            continue
        url = BASE + r + '.wav'
        rc = subprocess.run(['curl', '-sL', '--retry', '3', '-o', dst, url],
                            capture_output=True).returncode
        if rc == 0 and os.path.exists(dst) and os.path.getsize(dst) > 44:
            ok.append(r)
        else:
            fail.append(r)
    print(f'downloaded/present: {len(ok)} | failed: {len(fail)}')
    if fail:
        print('FAILED:', fail)
        sys.exit(1)


if __name__ == '__main__':
    main()
