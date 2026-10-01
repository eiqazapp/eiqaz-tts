# -*- coding: utf-8 -*-
"""qaf_corpus_forms_scan.py — مسح شامل لكلمات القاف في corpus التدريب المُشكَّل.

الهدف (PATCH 13):
  1. كل كلمة فيها ق + شكلها المُشكَّل الفعلي (catt داخل جُمل = توزيع التدريب)
  2. تمييز بيئة القاف: deep (كسرة/ضمة/تنونهما) مقابل plain (فتحة/سكون/بلا)
  3. بناء معجم موسع {هيكل: شكل} لكل كلمة قاف ببيئة deep — لزرع علامة {ق}
  4. فحص عائلات محددة: قيام/القيامة/قيامة، قال، قلب، رقم...
"""
import json
import glob
import re
from collections import Counter, defaultdict

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,!?;:()"«»\u060C\u061F\u061B\u2026-—'

STATE = ('/home/z/my-project/work/github_repo/eiqaz-tts/tashkeel-pipeline/'
         'state/ph3_full_output.jsonl')
ALL_SHARDS = (sorted(glob.glob('/home/z/my-project/work/ph3_full_shard_*.jsonl')) +
              sorted(glob.glob(
                  '/home/z/my-project/work/github_repo/eiqaz-tts/'
                  'tashkeel-pipeline/state/ph3_full_shard_*.jsonl')))
UNITS_RAW = json.load(open(
    '/home/z/my-project/work/github_repo/eiqaz-tts/tashkeel-pipeline/'
    'state/ph2_input_full.json', encoding='utf-8'))


def qaf_env(form):
    """بيئة القاف في الشكل المُشكَّل: deep (قِ/قُ/تنونيهما) أم plain أم bare."""
    i = form.find('ق')
    if i < 0 or i + 1 >= len(form):
        return 'end'
    nxt = form[i + 1]
    if nxt in ('\u0650', '\u064D'):        # كسرة / تنوين كسر
        return 'deep'
    if nxt in ('\u064F', '\u064C'):        # ضمة / تنوين ضم
        return 'deep'
    if nxt in ('\u064E', '\u064B'):        # فتحة / تنوين فتح
        return 'plain'
    if nxt == '\u0652':                    # سكون
        return 'sukun'
    if '\u0621' <= nxt <= '\u064A':        # حرف تالٍ بلا حركة
        return 'bare'
    return 'other'


# ---- 1) المسح على الوحدات المُشكَّلة (توزيع التدريب) --------------------
forms = defaultdict(Counter)     # هيكل -> Counter(شكل مُشكَّل)
envs = defaultdict(Counter)      # هيكل -> Counter(بيئة)
units_ok = 0
seen_ids = set()
for fn in ALL_SHARDS:
    with open(fn, encoding='utf-8') as f:
        for line in f:
            try:
                u = json.loads(line)
            except Exception:
                continue
            if u.get('status') != 'ok' or u.get('id') in seen_ids:
                continue
            seen_ids.add(u.get('id'))
            units_ok += 1
            for w in u['out'].split():
                if 'ق' not in w:
                    continue
                sk = TASH.sub('', w).strip(PUNCT)
                if len(sk) < 2 or 'ق' not in sk:
                    continue
                wf = w.strip(PUNCT)
                forms[sk][wf] += 1
                envs[sk][qaf_env(wf)] += 1

print(f'وحدات ok مُشكَّلة: {units_ok}')
print(f'هياكل قاف فريدة (بأشكالها المُشكَّلة): {len(forms)}')

deep_skels = {sk for sk, c in envs.items() if c.get('deep', 0) > 0}
print(f'هياكل ببيئة deep (قِ/قُ) واحدة على الأقل: {len(deep_skels)}')

# ---- 2) عائلات مطلوبة بعينها -------------------------------------------
print('\n== عائلات: قيام / قيامة / القيامة / قال / قلب / رقم / قصة ==')
for pat in ('قيام', 'القيام', 'قال', 'قلب', 'رقم', 'قصة', 'قرآن'):
    hits = {sk: c for sk, c in forms.items() if pat in sk}
    tot = sum(sum(c.values()) for c in hits.values())
    print(f'  {pat}: هياكل {len(hits)} | مواضع {tot}')
    for sk, c in sorted(hits.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
        top = list(c.most_common(2))
        print(f'    {sk:<16} env={dict(envs[sk])} | {top}')

# ---- 3) الـcorpus الخام كله: هل توجد قيام/القيامة أصلًا؟ -----------------
print('\n== المسح الخام (21,880 وحدة) ==')
raw_counter = Counter()
for u in UNITS_RAW:
    for w in TASH.sub('', u['text']).split():
        w = w.strip(PUNCT)
        if 'ق' in w and len(w) >= 2:
            raw_counter[w] += 1
for pat in ('قيام', 'القيام', 'قيامة'):
    hits = {w: n for w, n in raw_counter.items() if pat in w}
    print(f'  {pat}: {sum(hits.values())} موضعًا | {sorted(hits.items(), key=lambda kv: -kv[1])[:10]}')

# ---- 4) أعلى 60 هيكل قاف (المُشكَّل) ببيئاتها ----------------------------
print('\n== أعلى 60 هيكل (المُشكَّل): هيكل | مواضع | بيئة غالبة | الشكل الأشيع ==')
ranked = sorted(forms.items(), key=lambda kv: -sum(kv[1].values()))
for sk, c in ranked[:60]:
    env = envs[sk].most_common(1)[0][0]
    top_form = c.most_common(1)[0][0]
    print(f'  {sk:<18} {sum(c.values()):>4} | {env:<6} | {top_form}')

# ---- 5) حفظ المعجم الموسع (deep فقط — مرشح زرع علامة {ق}) ----------------
expanded = {}
for sk in sorted(deep_skels):
    top_form = forms[sk].most_common(1)[0][0]
    # الشكل الأشيع يجب أن يكون deep نفسه (لا شكلاً plain نادرًا)
    if qaf_env(top_form) == 'deep':
        expanded[sk] = {
            'form': top_form, 'n': sum(forms[sk].values()),
            'envs': dict(envs[sk]),
            'all_forms': dict(forms[sk].most_common(4)),
        }
out = {
    'units_ok': units_ok,
    'n_skeletons': len(forms),
    'n_deep_skeletons': len(deep_skels),
    'expanded_deep': expanded,
}
with open('/home/z/my-project/work/qaf_results/corpus_forms_expanded.json',
          'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nحُفظ: work/qaf_results/corpus_forms_expanded.json "
      f"({len(expanded)} هيكلًا deep جاهزًا للزرع)")
