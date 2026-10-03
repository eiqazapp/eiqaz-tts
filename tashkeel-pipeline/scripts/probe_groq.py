#!/usr/bin/env python3
"""probe_groq.py — probe Groq chat candidates with ENGINE-SHAPED requests.

Groq on_demand (free) tier limits discovered 2026-10-03 for this key:
  qwen/qwen3.8-27b   ITPM 7000   (needed ~11.4k input — impossible)
  openai/gpt-oss-20b TPM  8000   (needed ~12.4k — impossible)
  openai/gpt-oss-120b TPM 8000   (needed ~12.5k — impossible)
  allam-2-7b         HTTP 400 at max_tokens 4096 (context-shaped refusal)
The frozen system prompt alone is ~10.5k tokens, so full-batch calls cannot
fit any of these limits. This probe walks a SHAPE LADDER to find the largest
per-model request shape the tier accepts (system prompt stays byte-exact;
only user-message size and completion reserve shrink):

  BATCH25 mt4096  == engine default first-pass batch (25 sentences)
  BATCH5  mt1024  == engine retry-batch shape (5 sentences)
  SINGLE  mt512   == tashkeelSingle shape (1 sentence), fallback mt256

Output per model (one line per shape attempt, plus a final verdict):
  PROBE <model> <shape> HTTP <code> <notes>
  VERDICT <model> USABLE-BATCH25 | USABLE-BATCH5 | USABLE-SINGLE mt=<n> | UNUSABLE

Usage: probe_groq.py <input.json> <model> [<model> ...]
Env:   GROQ_API_KEY, PH3_WORK_DIR
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request

URL = 'https://api.groq.com/openai/v1/chat/completions'
UA = {'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}  # CF 1010 bans python-urllib


def call(key, model, sysmsg, usermsg, max_tokens):
    body = json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': sysmsg},
            {'role': 'user', 'content': usermsg},
        ],
        'max_tokens': max_tokens,
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={'Authorization': 'Bearer ' + key,
                 'Content-Type': 'application/json', **UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            r.read()
            return 200, ''
    except urllib.error.HTTPError as e:
        txt = e.read().decode()[:400]
        mm = re.search(r'(?:ITPM|TPM)\): Limit (\d+), Requested (\d+)', txt)
        note = f'LIMIT={mm.group(1)} NEED~{mm.group(2)}' if mm else ''
        return e.code, f'{note} {txt[:150]}'.strip()
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def main():
    key = os.environ['GROQ_API_KEY']
    wd = os.environ['PH3_WORK_DIR']
    sysmsg = open(os.path.join(wd, 'tashkeel_guide_v2.md'),
                  encoding='utf-8').read()[:25398]
    with open(sys.argv[1], encoding='utf-8') as f:
        units = json.load(f)

    numbered = lambda n: '\n'.join(f'{i + 1}. {u["text"]}'
                                   for i, u in enumerate(units[:n]))
    shapes = [
        ('BATCH25', numbered(25), 4096),
        ('BATCH5', numbered(5), 1024),
        ('SINGLE', f'1. {units[0]["text"]}', 512),
        ('SINGLE', f'1. {units[0]["text"]}', 256),
    ]

    for model in sys.argv[2:]:
        verdict = 'UNUSABLE'
        seen_single = None
        for name, usermsg, mt in shapes:
            code, note = call(key, model, sysmsg, usermsg, mt)
            label = name if name != 'SINGLE' else f'SINGLE mt={mt}'
            print(f'PROBE {model} {label} HTTP {code} {note}', flush=True)
            if code == 200:
                if name == 'BATCH25':
                    verdict = 'USABLE-BATCH25'
                    break
                if name == 'BATCH5':
                    verdict = 'USABLE-BATCH5'
                    break
                if seen_single is None:
                    seen_single = mt
                    verdict = f'USABLE-SINGLE mt={mt}'
                    break
        print(f'VERDICT {model} {verdict}', flush=True)


if __name__ == '__main__':
    main()
