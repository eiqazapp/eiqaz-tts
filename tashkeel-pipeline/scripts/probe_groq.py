#!/usr/bin/env python3
"""probe_groq.py — probe Groq chat candidates with an ENGINE-SHAPED request.

A tiny "say ok" probe passes models that then hard-fail with 413 (TPM limit
below the frozen system prompt). This probe replicates the real call shape:
  system  = first 25,398 chars of the guide (== SYSTEM_PROMPT size)
  user    = the first 25 input sentences, numbered (== first-pass batch)
  max_tokens = 4096 (== engine Groq client default)
A model is USABLE iff this exact-shaped request returns HTTP 200.

Output per model (stdout, one line each):
  PROBE <model> HTTP 200 USABLE
  PROBE <model> HTTP <code> TPM=<limit> NEEDED~<requested> <body-snippet>

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


def main():
    key = os.environ['GROQ_API_KEY']
    wd = os.environ['PH3_WORK_DIR']
    sysmsg = open(os.path.join(wd, 'tashkeel_guide_v2.md'),
                  encoding='utf-8').read()[:25398]
    with open(sys.argv[1], encoding='utf-8') as f:
        units = json.load(f)
    numbered = '\n'.join(f'{i + 1}. {u["text"]}'
                         for i, u in enumerate(units[:25]))
    usermsg = ('المهمة: إضافة التشكيل. أعد كل جملة مشكولة بنفس الترقيم.\n\n'
               + numbered)

    for model in sys.argv[2:]:
        body = json.dumps({
            'model': model,
            'messages': [
                {'role': 'system', 'content': sysmsg},
                {'role': 'user', 'content': usermsg},
            ],
            'max_tokens': 4096,
        }).encode()
        req = urllib.request.Request(
            URL, data=body,
            headers={'Authorization': 'Bearer ' + key,
                     'Content-Type': 'application/json',
                     # Cloudflare 1010 bans the default Python-urllib signature
                     # on api.groq.com; curl and node-fetch signatures pass.
                     'User-Agent': 'curl/8.5.0',
                     'Accept': '*/*'})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
                print(f'PROBE {model} HTTP 200 USABLE', flush=True)
        except urllib.error.HTTPError as e:
            txt = e.read().decode()[:500]
            mm = re.search(r'Limit (\d+), Requested (\d+)', txt)
            note = f'TPM={mm.group(1)} NEEDED~{mm.group(2)}' if mm else ''
            print(f'PROBE {model} HTTP {e.code} {note} {txt[:200]}', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'PROBE {model} ERROR {e}', flush=True)


if __name__ == '__main__':
    main()
