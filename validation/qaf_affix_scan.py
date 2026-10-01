# -*- coding: utf-8 -*-
"""qaf_affix_scan.py — مسح corpus لأشكال السوابق/التعريف لهياكل المعجم المدروس.

القرار الذي يبنيه: أي هيكل له تعرض موثق في corpus لأشكال مسبوقة (ال/و/ف/ب/ل/ك)
يُزرع مسبوقًا بعلامة {ق} (داخل التوزيع)؛ ومن لا تعرض له → تقريب [k] حتمي
(شكل OOD مثل الْقُرْآن — صفر مواضع §3-أ — يُسمع همزة).
+ أدلة إضافية: qr|nA (قرآنا)، Hqyqp (حقيقة — مرشحة TRUST).
"""
import json
import re
import sys

REPO = '/home/z/my-project/work/github_repo/eiqaz-tts/inference'
sys.path.insert(0, REPO)
import infer  # noqa: E402

CORPUS = ('/home/z/my-project/work/github_repo/eiqaz-tts/tashkeel-pipeline/'
          'state/ph2_input_full.json')
UNITS = json.load(open(CORPUS, encoding='utf-8'))

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,!?;:()"«»\u060C\u061F\u061B\u2026-—'

# هيكل كل كلمة corpus (نص خام غير مشكل → الهيكل مباشرة)
word_skels = {}
for u in UNITS:
    for w in u['text'].split():
        b = TASH.sub('', w).strip(PUNCT)
        if 'ق' not in b:
            continue
        try:
            sk = infer._ar_skel(b)
        except Exception:  # noqa: BLE001
            continue
        if sk:
            word_skels[sk] = word_skels.get(sk, 0) + 1

CLITICS = ('w', 'f', 'b', 'l', 'k')
report = {}
for S in sorted(infer.QAF_Q_STUDY_FORMS):
    prefixed_variants = {'Al' + S}
    for c in CLITICS:
        prefixed_variants.add(c + S)
        prefixed_variants.add(c + 'Al' + S)
    pref = sum(n for v, n in word_skels.items() if v in prefixed_variants)
    bare = word_skels.get(S, 0)
    report[S] = {'bare': bare, 'prefixed': pref,
                 'affix_ok': bool(pref > 0)}

# أدلة إضافية
extra = {}
for S in ('qr|nA', 'qrAn', 'Hqyqp', 'nqr>h', 'nqr>p', 'yqr>', 'qr|nnA',
          'AlHqyqp', 'HqyqAt', 'Alqr|n', 'AlqSS'):
    extra[S] = word_skels.get(S, 0)

print('== تعرض السوابق/التعريف لهياكل المعجم المدروس (corpus 21,880 وحدة):')
for S, r in report.items():
    flag = '✓ يُزرع مسبوقًا' if r['affix_ok'] else '✗ مسبوق → [k]'
    print(f"  {S:12s} bare={r['bare']:4d}  prefixed={r['prefixed']:3d}  {flag}")
print()
print('== أشكال إضافية (أدلة):')
for S, n in extra.items():
    print(f'  {S:12s} {n}')
print()
ok_set = sorted(S for S, r in report.items() if r['affix_ok'])
print('QAF_Q_AFFIX_OK =', ok_set)

out = {'evidence': report, 'extra': extra, 'affix_ok': ok_set,
       'source': 'ph2_input_full.json (21,880 units)',
       'purpose': 'PATCH 12 — قرار زرع المسبوقة بعلامة {ق} بالأدلة'}
path = ('/home/z/my-project/work/github_repo/eiqaz-tts/validation/'
        'qaf_affix_scan.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('حُفظ:', path)
