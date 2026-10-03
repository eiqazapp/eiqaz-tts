# -*- coding: utf-8 -*-
"""qaf_study_build.py — بناء معجم «الأشكال المدروسة» للقاف الأصيلة (PATCH 9)
==========================================================================
خلفية الاكتشاف (QAF_NATIVE_DISCOVERY.md + تأكيد المستخدم 2026-10-01):
- التدريب نفسه استخدم تشكيل catt على جُمل الـcorpus → «مخرج catt داخل جملة»
  هو حرفيًا توزيع التدريب.
- الكلمة المنفردة يسقط فيها catt إلى الفتحة (قَطْعَةَ = OOD) → تحقق الهمزة.
- زرع الشكل المطابق للتدريب (قِطْعَةً) → النطق المتعلَّم (قاف أصيلة ✓ مؤكد).

المهمة هنا: لكل كلمة في قائمة Q (+ إضافات البحث الويب):
1. البحث عن جُمل corpus حقيقية تحتويها → تشكيل catt → حصاد شكل الكلمة.
2. من ليس في الـcorpus → جمل حاملة (أسلوب معلم) → catt → حصاد (مصدر: carrier).
3. تصنيف «بيئة القاف»: kasra/damma/tanween-kasr/damm = deep (قابلة للزرع
   للنطق العميق) | fatha/sukun/bare = plain (غير قابلة — سلوك أصيل).
4. الإخراج: validation/qaf_q_study_forms.json + ملخص طباعة + مقتطف Python
   جاهز للتضمين في infer.py.

مصادر قائمة الإضافات (البحث 2026-10-01 — work/qaf_research/websearch*.json):
- talkinarabic.com/egyptian/letter-qaf-in-egyptian-arabic (مصرية تُعلّم العربية):
  قصة، مقاومة، قرية، مثقف، ثقافة، اعتقد، عبقري، معقد، تعقيد، قوي(بمعنى قوي
  جسديًا فقط — homograph مع «أوي» فتُستثنى من كل القوائم)، قرآن، القاهرة
- Wikipedia «Egyptian Arabic phonology»: ق أعيد إدخالها من الفصحى للكلمات
  الدينية + جذر ث-ق-ف (ثقافة/مثقف) + للتفريق بين المترادفات (قانون، قوى)
- r/learn_arabic: «فلو الكلمة كلمة عربية فصحى شائعة بينطقوها قاف» (سجل فصحى)
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/z/my-project/work/github_repo/eiqaz-tts'
sys.path.insert(0, os.path.join(REPO, 'inference'))
sys.path.insert(0, os.path.join(REPO, 'inference', 'lib'))

import infer  # noqa: E402  (catt_vocalize + arabic_to_buckwalter عبر الحزمة)

CORPUS = os.path.join(REPO, 'tashkeel-pipeline', 'state', 'ph2_input_full.json')
Q_LIST = os.path.join(REPO, 'validation', 'qaf_q_list.json')
OUT_JSON = os.path.join(REPO, 'validation', 'qaf_q_study_forms.json')

TASH = re.compile(r'[\u064B-\u0652]')
AR_LETTER = re.compile(r'[\u0621-\u063A\u0641-\u064A]')
QAF_DIAC_BUCK = frozenset('auiFNK~o')

# ---------------------------------------------------------------------------
# إضافات البحث (غير موجودة في قائمة Q الحالية) — Tier بحثي
# word_ar, skel(buckwalter مجرد), فئة, مصدر, ملاحظة
# ---------------------------------------------------------------------------
RESEARCH_ADDS = [
    ('قطعة', 'qTEp', 'verified', 'تأكيد المستخدم 2026-10-01 — الكلمة الم flag',
     'قِطْعَةً مؤكدة سمعيًا (تجربة المستخدم) — الفجوة الحرجة: لم تكن في قائمة Q!'),
    ('قطعتين', 'qTEtyn', 'verified', 'مثنى عائلة قطع (درس المستخدم)',
     'شكل مثنى شائع في دروس القسمة'),
    ('قصة', 'qSp', 'research', 'talkinarabic',
     'قائمة كاتبها مصرية — متغيّرة في العامية (ʔiṣṣa)'),
    ('قرية', 'qryp', 'research', 'talkinarabic',
     'قاف مفتوحة في الفصحى (قَرْيَة) — بيئة plain غالبًا'),
    ('مقاومة', 'mqAwmp', 'research', 'talkinarabic', 'قاف مفتوحة (مُقَاوَمَة)'),
    ('معقد', 'mEqd', 'research', 'talkinarabic', 'قاف ساكنة مشددة (مُعَقَّد)'),
    ('تعقيد', 'tEqyd', 'research', 'talkinarabic', 'قاف مكسورة (تَعْقِيد)'),
    ('عبقري', 'Ebqry', 'research', 'talkinarabic', 'قاف ساكنة (عَبْقَرِيّ)'),
    ('اعتقد', 'AEtqd', 'research', 'talkinarabic', 'قاف مكسورة (أَعْتَقِدُ)'),
    ('ثقافة', 'vqAfp', 'research', 'talkinarabic + Wikipedia (جذر ث-ق-ف)',
     'مصدران مستقلان — قاف مفتوحة (ثَقَافَة) لكن النطق المدعي saqafa'),
    ('مثقف', 'mvqf', 'research', 'talkinarabic + Wikipedia', 'قاف ساكنة مشددة'),
    ('تثقيف', 'tvqyf', 'research', 'Wikipedia (جذر ث-ق-ف)', 'قاف مكسورة'),
    ('القاهرة', 'qAhrp', 'research', 'talkinarabic', 'قاف مفتوحة — اسم علم'),
    ('تقوى', 'twqY', 'research', 'Wikipedia (ديني)', 'قاف ساكنة (تَقْوَى)'),
    ('قيام', 'qyAm', 'research', 'ديني (صلاة القيام)', 'قاف مكسورة'),
    ('قيامة', 'qyAlp', 'research', 'ديني (يوم القيامة)', 'قاف مكسورة'),
    ('شقيق', '$qyq', 'research', 'سجل فصيح', 'قاف مكسورة (شَقِيق)'),
]

# كلمات مستثناة من أي زرع تلقائي (homograph خطر) — توثيق فقط
EXCLUDED = [
    ('قوي', 'qwy', 'homograph',
     '«أوي» العامية (intensifier) مقابل قوي الفصحى — ويكيبيديا تؤكد التفرقة '
     'بالسياق فقط؛ corpus: 287 موضعًا أغلبها العامية → استبعاد كامل. '
     'المتاح للنطق الفصيح: وضع qaf (تقريب [k]) أو إعادة التدريب.'),
]

# جمل حاملة بأسلوب المعلم (لمن ليس في الـcorpus) — نفس أسلوب دروس المستخدم
CARRIERS = [
    'شرح المعلم {w} في الدرس',
    'الدرس النهارده عن {w}',
    'بنتكلم النهارده عن ال{w} في الحصة',
]


def norm_word_ar(w):
    """تطبيع كلمة عربية خام: نزع تشكيل + ه النهاية → ة + قص حواف الترقيم."""
    w = TASH.sub('', w).strip('.,!?؛،:"\'()\u2026')
    if len(w) >= 2 and w.endswith('ه'):
        w = w[:-1] + 'ة'
    return w


def buck_skel(word_ar):
    """هيكل باكوالتير مجرد لتطبيع عربي خام (مع ه→ة النهائية)."""
    from tts_arabic.text import arabic_to_buckwalter
    w = norm_word_ar(word_ar)
    if not w or not AR_LETTER.search(w):
        return None
    b = arabic_to_buckwalter(w)
    sk = ''.join(c for c in b if c not in QAF_DIAC_BUCK)
    if sk.endswith('h') and len(sk) > 2:
        sk = sk[:-1] + 'p'
    return sk


def skel_variants(sk):
    """نفس منطق _qaf_skel_variants في infer.py (نسخة بحثية)."""
    out = {sk}
    if sk.startswith('Al') and len(sk) > 3:
        out.add(sk[2:])
    for c in ('w', 'f', 'b', 'l', 'k'):
        if sk.startswith(c) and len(sk) > 2:
            out.add(sk[1:])
            if sk[1:3] == 'Al' and len(sk) > 4:
                out.add(sk[3:])
    return out


def find_word_in_voc(voc_sent, target_skels):
    """إيجاد أول كلمة في جملة مشكولة هيكلها (بمتغيراته) من مجموعة الهدف.
    يعيد (الكلمة_كما_ظهرت_مشكولة, الهيكل_المطابق)."""
    for w in voc_sent.split():
        wn = w.strip('.,!?؛،:')
        if 'ق' not in wn and 'ٯ' not in wn:
            continue
        sk = buck_skel(wn)
        if not sk:
            continue
        for v in skel_variants(sk):
            if v in target_skels:
                return wn, v
    return None, None


def qaf_env(form_ar):
    """بيئة القاف في شكل مشكول: deep (كسرة/ضمة/تنوين كسر-ضم) أو plain.
    يعيد (env, وصف)."""
    idx = form_ar.find('ق')
    if idx < 0:
        return 'no_qaf', ''
    after = form_ar[idx + 1:idx + 2]
    if after == '\u0640':                      # تطويل قبل الحركة
        after = form_ar[idx + 1:idx + 3].replace('\u0640', '')[:1]
    m = {'\u0650': 'deep', '\u064F': 'deep',          # كسرة/ضمة
         '\u064D': 'deep', '\u064C': 'deep',          # تنوين كسر/ضم
         '\u064E': 'plain_fatha', '\u0652': 'plain_sukun',
         '~': 'plain_sukun'}                          # شدة (تعامل كسكون)
    env = m.get(after, 'plain_bare')
    return env, f'«ق{after}»'


def definitize(form_ar):
    """تحويل شكل مدروس نكرة إلى معرف: نزع التنوين وألفه (قِطْعَةً → قِطْعَة)."""
    f = re.sub(r'[\u064B-\u064D]+$', '', form_ar)
    if f.endswith('ا') and not form_ar.replace('اً', '').endswith('ا'):
        pass
    # ألف تنوين الفتح: قُرْآنًا → قُرْآن (الألف قبل ً أُسقطت معه)
    if re.search(r'ا[\u064B]$', form_ar):
        f = f[:-1]
    # نزع حركة إعراب نهائية (حالة المعرفة السكونية/الوقفية)
    f = re.sub(r'[\u064E\u064F\u0650]$', '', f)
    return f


def main():
    q_list = json.load(open(Q_LIST, encoding='utf-8'))
    targets = []          # (word_ar, skel, tier, source, note)
    seen = {}
    for e in q_list['entries']:
        targets.append((e['word_ar'], e['skel'], 'q_list',
                        e.get('rationale', ''), e.get('category', '')))
        seen[e['skel']] = e['word_ar']
    for w, sk, tier, src, note in RESEARCH_ADDS:
        if sk in seen:
            continue
        targets.append((w, sk, tier, src, note))
        seen[sk] = w

    print(f'[1/3] الهدف: {len(targets)} هيكلًا — تحميل corpus وتحمية catt ...')
    units = json.load(open(CORPUS, encoding='utf-8'))
    _ = infer.catt_vocalize('اختبار التحمية')

    # فهرسة corpus: هيكل كل كلمة → قائمة (id, جملة)
    print('[2/3] فهرسة corpus (هياكل القاف) ...')
    index = {}
    for u in units:
        txt = u.get('text') or ''
        for w in txt.split():
            if 'ق' not in w:
                continue
            sk = buck_skel(w)
            if not sk:
                continue
            for v in skel_variants(sk):
                if v in seen:
                    index.setdefault(v, []).append((u.get('id'), txt))
                    break

    # الحصاد
    print('[3/3] الحصاد عبر catt ...')
    results = {}
    for word_ar, skel, tier, src, note in targets:
        entry = {
            'word_ar': word_ar, 'skel': skel, 'tier': tier,
            'source': src, 'note': note,
            'corpus_hits': 0, 'forms': {}, 'study_form': None,
            'study_env': None, 'carrier_form': None,
        }
        target_skels = skel_variants(skel)
        sents = index.get(skel, [])[:2]
        entry['corpus_hits'] = len(index.get(skel, []))
        forms = Counter()
        for uid, txt in sents:
            try:
                voc = infer.catt_vocalize(txt)
            except Exception as ex:               # noqa: BLE001
                print(f'  [catt-err] {skel} {uid}: {ex}')
                continue
            found, matched = find_word_in_voc(voc, target_skels)
            if found:
                forms[found] += 1
        if forms:
            entry['forms'] = dict(forms.most_common(6))
        else:
            # جمل حاملة
            cforms = Counter()
            for tpl in CARRIERS:
                sent = tpl.format(w=word_ar)
                try:
                    voc = infer.catt_vocalize(sent)
                except Exception:                  # noqa: BLE001
                    continue
                found, _ = find_word_in_voc(voc, target_skels)
                if found:
                    cforms[found] += 1
            if cforms:
                entry['carrier_form'] = dict(cforms.most_common(3))
                entry['forms'] = entry['carrier_form']
        # اختيار الشكل المدروس: الأكثر شيوعًا ذا بيئة deep، وإلا الأكثر مطلقًا
        best_deep, best_any = None, None
        for f, n in sorted(entry['forms'].items(), key=lambda kv: -kv[1]):
            env, _ = qaf_env(f)
            if best_any is None:
                best_any = (f, env)
            if env == 'deep' and best_deep is None:
                best_deep = (f, env)
        chosen = best_deep or best_any
        if chosen:
            entry['study_form'], entry['study_env'] = chosen
            entry['definite_form'] = definitize(chosen[0])
        results[skel] = entry

    # ---- التقرير -----------------------------------------------------------
    deep = [e for e in results.values() if e['study_env'] == 'deep']
    plain = [e for e in results.values()
             if e['study_env'] and e['study_env'] != 'deep']
    miss = [e for e in results.values() if not e['study_form']]
    print('\n' + '=' * 70)
    print(f'بيئة deep (قابلة للزرع للنطق العميق): {len(deep)}')
    for e in deep:
        src = 'corpus' if e['corpus_hits'] else 'carrier'
        print(f"  {e['word_ar']:<12} {e['study_form']:<14} "
              f"[{src}·{e['corpus_hits']}] tier={e['tier']}")
    print(f'\nبيئة plain (سلوك أصيل — لا زرع): {len(plain)}')
    for e in plain:
        print(f"  {e['word_ar']:<12} {e['study_form']:<14} env={e['study_env']}")
    print(f'\nبلا شكل نهائيًا: {len(miss)}')
    for e in miss:
        print(f"  {e['word_ar']:<12} hits={e['corpus_hits']} forms={e['forms']}")

    meta = {
        'name': 'QAF_Q_STUDY_FORMS — الأشكال المدروسة للقاف الأصيلة',
        'version': 'study-v1.0',
        'date': '2026-10-01',
        'mechanism': (
            'زرع الشكل المطابق للتدريب (توزيع catt داخل الجملة = توزيع التدريب '
            'نفسه لأن التدريب شُكِّل بـcatt على نفس الـcorpus). الكلمة المنفردة '
            'تسقط إلى الفتحة (OOD) فتُنطق همزة؛ زرع الشكل المدروس يستدعي النطق '
            'المتعلَّم. مؤكد سمعيًا من المستخدم: قطعة/قِطْعَةً → قاف أصيلة.'),
        'env_rule': (
            'deep = كسرة/ضمة/تنوين كسر أو ضم على القاف → مرشحة للنطق العميق؛ '
            'plain = فتحة/سكون/مجردة → سلوك أصيل (همزة/جيم B) ولا تُزرع. '
            'القاعدة استنتاج من الأدلة المؤكدة (قِ/قُ → عميق، قَ → همزة) '
            'وقابلة للتنقيح بنتائج qaf_experiment.py على جهاز المستخدم.'),
        'sources': [
            'corpus نيلتس (ph2_input_full.json) عبر catt داخل جمل حقيقية',
            'جمل حاملة بأسلوب المعلم (لغير الموجود في الـcorpus)',
            'قائمة Q (qaf_q_list.json q-v1.0)',
            'بحث الويب 2026-10-01: talkinarabic.com + Wikipedia Egyptian '
            'Arabic phonology + r/learn_arabic (work/qaf_research/)',
        ],
        'n_targets': len(targets),
        'n_deep': len(deep), 'n_plain': len(plain), 'n_missing': len(miss),
        'excluded': [{'word_ar': w, 'skel': s, 'reason': r, 'note': n}
                     for w, s, r, n in EXCLUDED],
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'entries': results}, f,
                  ensure_ascii=False, indent=1)
    print(f'\nحُفظ: {OUT_JSON}')

    # مقتطف Python للتضمين في infer.py (deep فقط = معجم الزرع الإنتاجي)
    print('\n# ---- QAF_Q_STUDY_FORMS (مقتطف infer.py) ----')
    print('QAF_Q_STUDY_FORMS = {')
    for sk, e in sorted(results.items()):
        if e['study_env'] == 'deep':
            print(f"    '{sk}': '{e['study_form']}',  # {e['word_ar']}")
    print('}')


if __name__ == '__main__':
    main()
