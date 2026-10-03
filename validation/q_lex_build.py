#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q_lex_build.py — بناء egyptian_q_default_lexicon_v1.json +
egyptian_q_lexicon_audit.csv من: مسح corpus + قاعدة المعرفة المنسّقة +
قوائم baseline (infer.py).

التصنيفات: q_default / g_default / hamza_default / context_dependent /
uncertain. الذيل غير المغطى -> hamza_default (قاعدة الانعكاس القاهري)
بثقة منخفضة + علم مراجعة.
"""
import csv
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/home/z/my-project/scripts')
from q_lex_kb import ENTRIES                      # noqa: E402
from q_lex_scan_corpus import (                    # noqa: E402
    skel_variants, parse_baseline)

STATS = '/home/z/my-project/work/q_lex/corpus_q_stats.json'
OUT_DIR = ('/home/z/my-project/work/github_repo/eiqaz-tts/validation')
OUT_JSON = OUT_DIR + '/egyptian_q_default_lexicon_v1.json'
OUT_CSV = OUT_DIR + '/egyptian_q_lexicon_audit.csv'
OUT_SUM = '/home/z/my-project/work/q_lex/build_summary.json'

CAT_LABEL = {
    'q': 'q_default', 'g': 'g_default', 'hamza': 'hamza_default',
    'context': 'context_dependent', 'uncertain': 'uncertain',
}


def match_entry(csk, skel2entry):
    """مطابقة هيكل corpus بإدخال KB: مباشرة، ثم نزع سوابق تدريجي
    (و/ف/ب/ل/ك + ال)، ثم لواحق ضمائر (ه/ها/هم/هن/ك/كم/نا/وا)."""
    seen = [csk]
    # 1) مطابقة مباشرة عبر صيغ الهيكل
    for var in skel_variants(csk):
        if var in skel2entry:
            return skel2entry[var]
    # 2) نزع لواحق الضمائر (مرة واحدة)
    suffixes = ['hmA', 'hm', 'hA', 'hn', 'km', 'ki', 'kum', 'nA', 'wA',
                'h', 'k', 'y']
    bases = list(seen)
    for suf in suffixes:
        if csk.endswith(suf) and len(csk) - len(suf) >= 2:
            bases.append(csk[:-len(suf)])
    # 3) نزع سوابق متكرر (حتى 3) + ال
    expanded = list(bases)
    for b in list(bases):
        cur = b
        for _ in range(3):
            changed = False
            if cur.startswith('Al') and len(cur) > 3:
                cur2 = cur[2:]
                expanded.append(cur2)
                cur = cur2
                changed = True
            for c in ('w', 'f', 'b', 'l', 'k'):
                if cur.startswith(c) and len(cur) > 2:
                    expanded.append(cur[1:])
                    cur = cur[1:]
                    changed = True
                    break
            if not changed:
                break
    for cand in expanded:
        if cand in skel2entry:
            return skel2entry[cand]
        for var in skel_variants(cand):
            if var in skel2entry:
                return skel2entry[var]
    return None


def main():
    # ---- 1) المدخلات ----
    stats = json.load(open(STATS, encoding='utf-8'))
    words = stats['words']
    sets, dicts, deep_ok = parse_baseline()
    g_set = set(sets.get('QAF_G_SKELETONS', set()))
    q_list = set(sets.get('QAF_Q_SKELETONS', set()))
    study = set(dicts.get('QAF_Q_STUDY_FORMS', {}).keys())
    corpus_forms = set(dicts.get('QAF_Q_CORPUS_FORMS', {}).keys())

    # ---- 2) توسيع KB على corpus ----
    # skeleton -> entry index (آخر إدخال يفوز = override)
    skel2entry = {}
    for idx, e in enumerate(ENTRIES):
        if e['pron'] == 'hamza' and not e['skels']:
            continue          # مدخل قاعدة الذيل
        for sk in e['skels']:
            for var in skel_variants(sk):
                skel2entry.setdefault(var, idx)

    matched = defaultdict(set)      # entry idx -> corpus skeletons
    covered = set()
    for csk, rec in words.items():
        idx = match_entry(csk, skel2entry)
        if idx is not None:
            matched[idx].add(csk)
            covered.add(csk)

    # ---- 3) بناء الإدخالات ----
    lex = defaultdict(list)
    rows = []
    conflicts = []
    for idx, e in enumerate(ENTRIES):
        if e['pron'] == 'hamza' and not e['skels']:
            continue
        corpus_sks = sorted(matched.get(idx, set()),
                            key=lambda s: -words[s]['count'] if s in words
                            else 0)
        cnt = sum(words[s]['count'] for s in corpus_sks if s in words)
        # baseline الفعلي لأشكال العائلة في corpus (مرجّح بالمواضع)
        bl = Counter()
        for s in corpus_sks:
            if s in words:
                bl[words[s]['baseline_auto']] += words[s]['count']
        if not bl:
            bl = Counter({'absent': 0})
        entry = {
            'word': e.get('word', e['fam']),
            'family': e['fam'],
            'category': e['cat'],
            'pronunciation': CAT_LABEL[e['pron']],
            'confidence': e['conf'],
            'evidence': e['ev'],
            'source': 'curated_kb+corpus_scan',
            'notes': e['notes'],
            'corpus_audio_evidence': e['audio'],
            'corpus': {
                'positions': cnt,
                'skeletons': corpus_sks[:12],
                'n_skeleton_forms': len(corpus_sks),
                'baseline_auto_mix': dict(bl),
            },
        }
        lex[CAT_LABEL[e['pron']]].append(entry)
        # تعارضات مع baseline
        new_pron = CAT_LABEL[e['pron']]
        for bk, n in bl.items():
            if bk == 'absent':
                continue
            if bk == 'g' and new_pron != 'g_default':
                conflicts.append(
                    dict(word=e['word'], family=e['fam'], skel_note=corpus_sks[:3],
                         baseline='g (B list)', new=new_pron, positions=n,
                         kind='B_g_vs_new'))
            if bk == 'q_planted' and new_pron not in (
                    'q_default',):
                conflicts.append(
                    dict(word=e['word'], family=e['fam'], skel_note=corpus_sks[:3],
                         baseline='q_planted (auto)', new=new_pron, positions=n,
                         kind='planted_q_vs_new'))
            if bk == 'q_qafonly' and new_pron in (
                    'hamza_default', 'context_dependent'):
                conflicts.append(
                    dict(word=e['word'], family=e['fam'], skel_note=corpus_sks[:3],
                         baseline='q_qafonly (Q list — علامة/وضع qaf)',
                         new=new_pron, positions=n,
                         kind='Q_list_vs_new'))
        rows.extend(_rows_for(entry, corpus_sks, words, e, sets, dicts))

    # ---- 4) الذيل غير المغطى ----
    tail = [(s, r) for s, r in words.items() if s not in covered]
    tail_review = []
    for s, r in sorted(tail, key=lambda kv: -kv[1]['count']):
        if r['count'] >= 5:
            tail_review.append((s, r['count'], r['word_ar_best']))
        rows.append({
            'skeleton': s, 'word_ar': r['word_ar_best'],
            'corpus_count': r['count'],
            'family': '(tail — unclassified)', 'category': 'tail',
            'pronunciation': 'hamza_default',
            'confidence': 'low',
            'baseline_auto': r['baseline_auto'],
            'conflict': ('yes' if r['baseline_auto'] in (
                'g', 'q_planted') else ''),
            'corpus_audio_evidence': '—',
            'evidence': ('[wiki] الانعكاس القاهري غير المعلم ق→ʔ + لا دليل '
                         'مضاد (ذيل تردد ' + str(r['count']) + ')'),
            'source': 'default_rule', 'notes': 'مراجعة بشرية موصى بها',
            'vowel_env': max(r['envs'], key=r['envs'].get),
            'review_flag': 'yes',
        })
        lex['hamza_default'].append({
            'word': r['word_ar_best'], 'family': '(tail)',
            'category': 'tail', 'pronunciation': 'hamza_default',
            'confidence': 'low',
            'evidence': ['[wiki] ق→ʔ الافتراضي القاهري — ذيل غير مغطى',
                         f"[corpus] {r['count']} مواضع"],
            'source': 'default_rule',
            'notes': 'لم يُغطَّ صراحةً في KB — مراجعة بشرية',
            'corpus_audio_evidence': '—',
            'corpus': {'positions': r['count'],
                       'skeletons': [s],
                       'n_skeleton_forms': 1,
                       'baseline_auto_mix': {r['baseline_auto']: r['count']}},
        })

    # ---- 5) الإحصاءات ----
    def agg(cat):
        return dict(
            families=len(lex[cat]),
            positions=sum(x['corpus']['positions'] for x in lex[cat]))

    summary = dict(
        corpus=dict(units=stats['meta']['n_units'],
                    positions=stats['meta']['n_q_positions'],
                    unique_skeletons=stats['meta']['n_unique_skeletons']),
        coverage=dict(
            explicit_kb_skeletons=len(covered),
            tail_skeletons=len(tail),
            kb_families_total=len(ENTRIES)),
        counts={c: agg(c) for c in
                ('q_default', 'g_default', 'hamza_default',
                 'context_dependent', 'uncertain')},
        conflicts=conflicts,
        tail_review_top40=tail_review[:40],
    )

    lexicon = {
        'meta': {
            'name': 'egyptian_q_default_lexicon',
            'version': 'v1.0',
            'date': '2026-10-04',
            'criterion': ('كيف ينطق المتحدث المصري القاف في هذه الكلمة '
                          'داخل كلام مصري طبيعي؟ — لا «هل الكلمة فصيحة؟»'),
            'categories': ['q_default', 'g_default', 'hamza_default',
                           'context_dependent', 'uncertain'],
            'pronunciation_tokens': {
                'q_default': "توكن 'q' (قاف أصيلة فصحى)",
                'g_default': "توكن 'v' (جيم [g] كما في good)",
                'hamza_default': "توكن '<' (همزة قاهرية)",
                'context_dependent': 'لا قاعدة واحدة بلا سياق — تُحسم '
                                     'بالسجل/المعنى أو بعلامة {ق}/{ء}/{ج}',
                'uncertain': 'أدلة غير كافية — لا تدخل q مؤكدة'},
            'methodology': [
                'مسح شامل للنصوص المصرية المشكولة (21,854 نصًا) من '
                'msa-tts-data-v1: 31,254 موضع قاف / 4,546 هيكلًا فريدًا',
                'تصنيف عائلات صرفية منسّق يدويًا مع أدلة مصدرها: توجيهات '
                'المستخدم + قياسات run1 الصوتية + سماع المستخدم + '
                'Wikipedia Egyptian Arabic phonology + talkinarabic + '
                'تحليل لغوي + قرارات baseline',
                'كل مشتق يُتحقق منه فرديًا (لا افتراض اشتقاقي) — الإدخالات '
                'اللاحقة تنسخ السابقة (override)',
                'الذيل غير المغطى -> hamza_default بقاعدة الانعكاس القاهري '
                '(Wikipedia: */q/ became [ʔ] in Cairo and the Nile Delta) '
                'بثقة منخفضة + علم مراجعة',
                'التعارض مع baseline (infer.py auto) يُعرض لا يُخفى',
            ],
            'limits': [
                'الأدلة الصوتية متوفرة لعدد محدود من الكلمات (run1 = 24 '
                'كلمة × 5 تصييرات عند متحدث واحد)؛ الباقي نصية/لغوية',
                'corpus تلفزيوني قاهري — لا يمثل صعيد مصر (ق→g ريفي) ولا '
                'اللهجات البدوية',
                'قاعدة بيئة الحركة (قِ/قُ عميقة) نصية لا صوتية وتنتج '
                'إيجابيات كاذبة موثقة (قلوب/قيمة/قوة)',
                'context_dependent يضم كلمات السجلين — corpus واحد يحوي '
                'أخبارًا ودراما معًا',
            ],
            'counts': summary['counts'],
            'baseline_note': ('هذا القاموس مقترح مراجعة بشرية — لا سياسة '
                              'سارية. قوائم infer.py الحالية لم تُمس '
                              '(baseline محفوظ كما طلب المستخدم).'),
            'sources': {
                'corpus': 'ahmedyaseen86/msa-tts-data-v1 '
                          '(niletts_tashkeel_text — 21,854 نصًا مشكولًا)',
                'acoustic': 'validation/qaf_experiment_run1.json + '
                            'QAF_NATIVE_DISCOVERY.md §8',
                'web': 'en.wikipedia.org/wiki/Egyptian_Arabic_phonology; '
                       'talkinarabic.com; eladawy.blog',
                'user_directives': '2026-10-02 (قرآن/رقم/قرش/قنطار/مقام/'
                                   'قيراط/ترقيم/قال...) + 2026-10-04 '
                                   '(قرآن/فقط/قانون q؛ قال ء)',
            },
        },
        'q_default': lex['q_default'],
        'g_default': lex['g_default'],
        'hamza_default': lex['hamza_default'],
        'context_dependent': lex['context_dependent'],
        'uncertain': lex['uncertain'],
    }
    json.dump(lexicon, open(OUT_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    # ---- 6) CSV التدقيق ----
    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'skeleton', 'word_ar', 'corpus_count', 'family', 'category',
            'pronunciation', 'confidence', 'baseline_auto', 'conflict',
            'corpus_audio_evidence', 'evidence', 'source', 'notes',
            'vowel_env', 'review_flag'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: -x['corpus_count']):
            w.writerow(r)

    # ---- 7) الملخص ----
    json.dump(summary, open(OUT_SUM, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print('=== الملخص ===')
    print('coverage: KB-covered', len(covered), '| tail', len(tail))
    for c, v in summary['counts'].items():
        print(f"{c:<20} families={v['families']:>4}  positions={v['positions']:>6}")
    print('conflicts:', len(conflicts))
    for c in conflicts[:25]:
        print('  -', c['word'], '|', c['baseline'], '->', c['new'],
              f"({c['positions']} pos)")
    print('\ntail review (top 15 by freq):')
    for s, n, w_ in tail_review[:15]:
        print(f'  {s:<16} {n:>4}  {w_}')


def _rows_for(entry, corpus_sks, words, e, sets, dicts):
    rows = []
    if corpus_sks:
        for s in corpus_sks:
            r = words[s]
            bl = r['baseline_auto']
            conflict = ''
            if bl == 'g' and entry['pronunciation'] != 'g_default':
                conflict = 'yes'
            if bl == 'q_planted' and entry['pronunciation'] not in (
                    'q_default',):
                conflict = 'yes'
            if bl == 'hamza' and entry['pronunciation'] == 'q_default':
                conflict = 'new_policy'
            rows.append({
                'skeleton': s, 'word_ar': r['word_ar_best'],
                'corpus_count': r['count'],
                'family': entry['family'], 'category': entry['category'],
                'pronunciation': entry['pronunciation'],
                'confidence': entry['confidence'],
                'baseline_auto': bl, 'conflict': conflict,
                'corpus_audio_evidence': entry['corpus_audio_evidence'],
                'evidence': ' | '.join(entry['evidence'])[:500],
                'source': entry['source'], 'notes': entry['notes'][:300],
                'vowel_env': max(r['envs'], key=r['envs'].get),
                'review_flag': ('yes' if entry['confidence'] in (
                    'low', 'medium') or conflict else ''),
            })
    else:
        # كلمة KB غير موجودة في corpus
        bl = 'absent'
        for sk in e['skels']:
            if sk in sets.get('QAF_G_SKELETONS', set()):
                bl = 'g(listed)'
                break
            if sk in sets.get('QAF_Q_SKELETONS', set()):
                bl = 'q_list'
                break
        rows.append({
            'skeleton': '/'.join(e['skels'][:4]),
            'word_ar': entry['word'], 'corpus_count': 0,
            'family': entry['family'], 'category': entry['category'],
            'pronunciation': entry['pronunciation'],
            'confidence': entry['confidence'], 'baseline_auto': bl,
            'conflict': ('yes' if (bl == 'g(listed)' and
                                   entry['pronunciation'] != 'g_default')
                         else ''),
            'corpus_audio_evidence': entry['corpus_audio_evidence'],
            'evidence': ' | '.join(entry['evidence'])[:500],
            'source': entry['source'], 'notes': entry['notes'][:300],
            'vowel_env': '—',
            'review_flag': ('yes' if entry['confidence'] in (
                'low', 'medium') else ''),
        })
    return rows


if __name__ == '__main__':
    main()
