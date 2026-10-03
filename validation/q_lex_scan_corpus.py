#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q_lex_scan_corpus.py — المسح الشامل لكلمات القاف في corpus المصري المشكول
(21,854 نصًا من msa-tts-data-v1/niletts_tashkeel_text/transcripts.jsonl).

لكل كلمة تحتوي ق:
  - الهيكل الكامل (باكوالتير بلا حركات، مع السوابق كما وردت)
  - بيئة القاف الصوتية في الشكل المشكول (كسرة/ضمة=deep، فتحة=plain، سكون، ...)
  - العدد، الأشكال المشكولة الشائعة، سياقان مثالان
  - تصنيف baseline الحالي (سلوك auto الفعلي في infer.py):
      g        : في QAF_G_SKELETONS (فئة B → [g])
      q_planted: في VERIFIED|TIER1|TRUST (زرع auto → قاف خام عميقة)
      q_qafonly: في قوائم Q الأخرى (دراسة/corpus/قائمة Q — qaf mode/علامة فقط)
      hamza    : لا شيء مما سبق (q→'<' عبر الخريطة = همزة)

مطابقة الهياكل تحاكي _qaf_skel_variants في infer.py حرفيًا
(نزع ال + السوابق w/f/b/l/k + ه↔ة + نزع ألف تنوين النصب الطرفية).

الخرج: work/q_lex/corpus_q_stats.json
"""
import ast
import json
import re
import sys
from collections import Counter, defaultdict

INFER = ('/home/z/my-project/work/github_repo/eiqaz-tts/inference/'
         'infer.py')
TRANSCRIPTS = '/home/z/my-project/work/q_lex/transcripts.jsonl'
OUT = '/home/z/my-project/work/q_lex/corpus_q_stats.json'

# ---- باكوالتير (نسخة مطابقة من tts_arabic/text/phonetise_buckwalter.py) ----
AR2BW = {
    'ب': 'b', 'ذ': '*', 'ط': 'T', 'م': 'm', 'ت': 't', 'ر': 'r', 'ظ': 'Z',
    'ن': 'n', 'ث': '^', 'ز': 'z', 'ع': 'E', 'ه': 'h', 'ج': 'j', 'س': 's',
    'غ': 'g', 'ح': 'H', 'ق': 'q', 'ف': 'f', 'خ': 'x', 'ص': 'S', 'ش': '$',
    'د': 'd', 'ض': 'D', 'ك': 'k', 'أ': '>', 'ء': "'", 'ئ': '}', 'ؤ': '&',
    'إ': '<', 'آ': '|', 'ا': 'A', 'ى': 'Y', 'ة': 'p', 'ي': 'y', 'ل': 'l',
    'و': 'w', 'ً': 'F', 'ٌ': 'N', 'ٍ': 'K', 'َ': 'a', 'ُ': 'u', 'ِ': 'i',
    'ّ': '~', 'ْ': 'o',
}
VOWELS_BW = set('aiuFNKo~')       # حركات باكوالتير + الشدة
EXTRA_STRIP = {'\u0670'}          # ألف خنجرية علوية

WORD_RE = re.compile(r'[\u0621-\u064A][\u0621-\u064A\u064B-\u0652\u0670]*')
CLITICS = ('w', 'f', 'b', 'l', 'k')


def to_bw(word):
    return ''.join(AR2BW.get(c, c) for c in word)


def skel_of(word):
    """هيكل الكلمة: باكوالتير بلا حركات."""
    return ''.join(c for c in to_bw(word)
                   if c not in VOWELS_BW and c not in EXTRA_STRIP)


def qaf_env(word):
    """بيئة القاف في الشكل المشكول (الأولى فقط — النادر ذو قافين يُفحص يدويًا)."""
    i = word.find('ق')
    if i < 0:
        return None
    if i + 1 >= len(word):
        return 'end'
    nxt = word[i + 1]
    if nxt in ('\u0650', '\u064D'):        # كسرة / تنوين كسر
        return 'deep_i'
    if nxt in ('\u064F', '\u064C'):        # ضمة / تنوين ضم
        return 'deep_u'
    if nxt in ('\u064E', '\u064B'):        # فتحة / تنوين فتح
        return 'plain_a'
    if nxt == '\u0652':                    # سكون
        return 'sukun'
    if nxt == '\u0651':                    # شدة (انظر ما بعدها)
        return qaf_env(word[:i + 1] + word[i + 2:]) if len(word) > i + 2 else 'shadda_end'
    if '\u0621' <= nxt <= '\u064A':        # حرف تالٍ بلا حركة
        return 'bare'
    return 'other'


def skel_variants(s):
    """محاكاة _qaf_skel_variants في infer.py."""
    out = set()
    def add(x):
        if x:
            out.add(x)
            if x.endswith('h'):
                add_h = x[:-1] + 'p'
                out.add(add_h)
            if x.endswith('A') and len(x) > 2:
                out.add(x[:-1])            # نزع ألف تنوين النصب الطرفية
    base = list({s})
    for b in list(base):
        add(b)
        if b.startswith('Al') and len(b) > 3:
            add(b[2:])
        for c in CLITICS:
            if b.startswith(c) and len(b) > 2:
                add(b[1:])
                if b[1:3] == 'Al' and len(b) > 4:
                    add(b[3:])
    # ه↔ة على كل الصيغ المولدة
    for v in list(out):
        if v.endswith('h'):
            out.add(v[:-1] + 'p')
        elif v.endswith('p'):
            out.add(v[:-1] + 'h')
    return out


def parse_baseline():
    """استخراج مجموعات infer.py الحرفية عبر ast."""
    src = open(INFER, encoding='utf-8').read()
    tree = ast.parse(src)
    sets, dicts = {}, {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign,)):
            continue
        targets = [t.id for t in node.targets
                   if isinstance(t, ast.Name)]
        if not targets:
            continue
        name = targets[0]
        v = node.value
        if isinstance(v, (ast.Set, ast.SetComp)) and isinstance(v, ast.Set):
            try:
                sets[name] = {ast.literal_eval(e) for e in v.elts}
            except Exception:
                pass
        elif isinstance(v, ast.Dict):
            try:
                dicts[name] = {ast.literal_eval(k): ast.literal_eval(val)
                               for k, val in zip(v.keys, v.values)}
            except Exception:
                pass
        elif isinstance(v, ast.Call) and getattr(v.func, 'id', '') == 'frozenset':
            arg = v.args[0]
            if isinstance(arg, ast.Set):
                try:
                    sets[name] = frozenset(ast.literal_eval(e)
                                           for e in arg.elts)
                except Exception:
                    pass
    # DEEP_OK = VERIFIED | TIER1 | TRUST | {قرآن family} — يُحسب
    deep_ok = set(sets.get('QAF_Q_VERIFIED', set())) | set(
        sets.get('QAF_Q_TIER1', set())) | set(
        sets.get('QAF_Q_TRUST', set())) | {'qr|n', 'qrAn', 'qr|nA', 'qSS'}
    return sets, dicts, deep_ok


def baseline_auto(variants, g_set, deep_ok, q_all):
    """سلوك الوضع auto الفعلي للهيكل عبر صيغه."""
    vs = variants
    if any(v in g_set for v in vs):
        return 'g'
    if any(v in deep_ok for v in vs):
        return 'q_planted'
    if any(v in q_all for v in vs):
        return 'q_qafonly'
    return 'hamza'


def main():
    g_set, deep_ok = None, None
    sets, dicts, deep_ok = parse_baseline()
    g_set = set(sets.get('QAF_G_SKELETONS', set()))
    q_list = set(sets.get('QAF_Q_SKELETONS', set()))
    study = set(dicts.get('QAF_Q_STUDY_FORMS', {}).keys())
    corpus_forms = set(dicts.get('QAF_Q_CORPUS_FORMS', {}).keys())
    q_all = q_list | study | corpus_forms
    print(f'baseline: G={len(g_set)} Q_list={len(q_list)} '
          f'study={len(study)} corpus_forms={len(corpus_forms)} '
          f'DEEP_OK={len(deep_ok)}')

    words = {}          # skel -> {count, forms, envs, examples, baseline}
    n_units = 0
    n_positions = 0
    for line in open(TRANSCRIPTS, encoding='utf-8'):
        try:
            u = json.loads(line)
        except Exception:
            continue
        n_units += 1
        text = u.get('text', '')
        for w in WORD_RE.findall(text):
            if 'ق' not in w:
                continue
            n_positions += 1
            sk = skel_of(w)
            if len(sk) < 2 or 'q' not in sk:
                continue
            rec = words.setdefault(sk, {
                'count': 0, 'forms': Counter(), 'envs': Counter(),
                'examples': [], 'word_ar_best': ''})
            rec['count'] += 1
            rec['forms'][w] += 1
            rec['envs'][qaf_env(w)] += 1
            if len(rec['examples']) < 2:
                rec['examples'].append(
                    {'utt': u.get('utt', ''), 'text': text[:90]})

    # baseline عبر الصيغ
    for sk, rec in words.items():
        variants = skel_variants(sk)
        rec['baseline_auto'] = baseline_auto(
            variants, g_set, deep_ok, q_all)
        rec['baseline_lists'] = {
            'G': sorted(v for v in variants if v in g_set),
            'Q_list': sorted(v for v in variants if v in q_list),
            'study_forms': sorted(v for v in variants if v in study),
            'corpus_forms': sorted(v for v in variants if v in corpus_forms),
            'deep_ok': sorted(v for v in variants if v in deep_ok),
        }
        rec['forms'] = rec['forms'].most_common(5)
        rec['envs'] = dict(rec['envs'])
        rec['word_ar_best'] = rec['forms'][0][0] if rec['forms'] else ''

    total_by_baseline = Counter(r['baseline_auto'] for r in words.values())
    positions_by_baseline = Counter()
    for r in words.values():
        positions_by_baseline[r['baseline_auto']] += r['count']

    stats = {
        'meta': {
            'n_units': n_units,
            'n_q_positions': n_positions,
            'n_unique_skeletons': len(words),
            'baseline_sets': {
                'G_size': len(g_set), 'Q_list_size': len(q_list),
                'study_size': len(study),
                'corpus_forms_size': len(corpus_forms),
                'deep_ok_size': len(deep_ok)},
            'by_baseline_skeletons': dict(total_by_baseline),
            'by_baseline_positions': dict(positions_by_baseline),
        },
        'words': dict(sorted(words.items(),
                             key=lambda kv: -kv[1]['count'])),
    }
    json.dump(stats, open(OUT, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"units={n_units} positions={n_positions} "
          f"unique={len(words)}")
    print('by baseline (skeletons):', dict(total_by_baseline))
    print('by baseline (positions):', dict(positions_by_baseline))


if __name__ == '__main__':
    main()
