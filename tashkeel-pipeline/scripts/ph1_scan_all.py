# -*- coding: utf-8 -*-
"""Phase 1 extension (guide v2) — corpus-wide extraction scans.

Rule 1: frozen MSA adverbs/tools — criterion: fusha adverb/particle units
        attested WITH tanween-fath somewhere in the transcripts (direct
        spoken evidence), plus non-tanween frozen fusha candidates.
Rule 2: scientific/technical terms — curated domain lexicon intersection
        + domain-salient vocabulary mining (medical/sales vs general).
Rule 3: religious terms/formulas — marker words, frozen formulas, Qur'anic
        verse fragments.

Reads:  work/prep_output/extraction.csv (21,880 units)
Writes: work/ph1_scan_frozen.json / _sci.json / _religious.json
"""
import csv
import json
import re
from collections import Counter, defaultdict

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,?!:;"()«»\u060C\u061F\u061B\u201C\u201D\u2026-'


def skel(w):
    return TASH.sub('', w).strip(PUNCT)


def strip_conj(s):
    return s[1:] if s[:1] in 'وف' and len(s) > 3 else s


def norm_hamza(s):
    return s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')


def variants(s):
    """Generate prefix-stripped lookup variants of a corpus word skeleton."""
    s = strip_conj(s)
    bases = [s]
    for p in ('وال', 'فال', 'بال', 'كال', 'لل'):
        if s.startswith(p) and len(s) - len(p) >= 3:
            bases.append(s[len(p):])
    if s.startswith('ال') and len(s) > 4:
        bases.append(s[2:])
    for c in 'وفبلك':
        if s[:1] == c and len(s) >= 4:
            bases.append(s[1:])
    out = set()
    for b in bases:
        if b:
            out.add(b)
            out.add(norm_hamza(b))
    return out


def clean_text(t):
    s = TASH.sub('', t)
    for ch in PUNCT:
        s = s.replace(ch, ' ')
    return re.sub(r'\s+', ' ', s).strip()


rows = list(csv.DictReader(open(
    '/home/z/my-project/work/prep_output/extraction.csv', encoding='utf-8')))
texts = [r['transcript'] for r in rows]
episodes = [r['episode'].split('_')[0] for r in rows]
print('episodes:', Counter(episodes))
cleaned = [clean_text(t) for t in texts]

# ================= Rule 1: frozen MSA =================
with_tan = Counter()
tan_units = set()
total_occ = Counter()
for i, t in enumerate(texts):
    for w in t.split():
        s = strip_conj(skel(w))
        if not s:
            continue
        total_occ[s] += 1
        if '\u064B' in w:
            with_tan[s] += 1
            tan_units.add(i)

frozen_all = [{'skel': s, 'tan_n': n, 'total_n': total_occ.get(s, n)}
              for s, n in with_tan.most_common()]
t_freq = [x for x in frozen_all if x['tan_n'] >= 10]
t_mid = [x for x in frozen_all if 3 <= x['tan_n'] < 10]
t_rare = [x for x in frozen_all if x['tan_n'] < 3]
print('\n== RULE 1: frozen MSA (tanween evidence) ==')
print(f'skeletons: {len(frozen_all)} total '
      f'(frequent>=10: {len(t_freq)}, mid 3-9: {len(t_mid)}, rare 1-2: {len(t_rare)})')
print(f'units with >=1 tanween adverb: {len(tan_units)}/{len(texts)}')
print(f'total tanween occurrences: {sum(x["tan_n"] for x in frozen_all)}')
print('top10 (skel, tan_n, total_n):',
      [(x['skel'], x['tan_n'], x['total_n']) for x in frozen_all[:10]])

# non-tanween frozen fusha candidates (function words / formulas)
CAND_SINGLE = ['ربما', 'أيضا', 'كذلك', 'إذن', 'حسنا', 'بالتالي', 'أما',
               'إما', 'كليا', 'عموما', 'بالفعل', 'بالتدريج']
CAND_MULTI = ['على فكرة', 'بالمناسبة', 'للأسف', 'لحسن الحظ', 'سوء الحظ',
              'على الأقل', 'على الأكثر', 'في النهاية', 'في البداية',
              'في الحقيقة', 'بشكل عام', 'من ناحية']
cand_hits = Counter()
for i, t in enumerate(texts):
    for w in t.split():
        s = skel(w)
        if not s:
            continue
        for v in variants(s):
            if v in CAND_SINGLE:
                cand_hits[v] += 1
                break


def phrase_count(phrase, allow_prefix=True):
    words = phrase.split()
    pat = r'(?:^|\s)' + (('(?:[وف])?' if allow_prefix else '')
                         + re.escape(words[0]))
    for wd in words[1:]:
        pat += r'\s+' + re.escape(wd)
    pat += r'(?=\s|$)'
    cnt = 0
    units = set()
    for i, c in enumerate(cleaned):
        n = len(re.findall(pat, c))
        if n:
            cnt += n
            units.add(i)
    return cnt, units


print('\nnon-tanween frozen candidates found:')
for c in CAND_SINGLE:
    if cand_hits.get(c):
        print(f'  {c}: {cand_hits[c]}')
for c in CAND_MULTI:
    n, uset = phrase_count(c)
    if n:
        print(f'  {c}: {n} (units {len(uset)})')

json.dump({
    'criterion': 'fusha adverb/particle units attested with tanween-fath in '
                 'the transcripts (direct spoken evidence); conjunction '
                 'prefix stripped',
    'n_skeletons': len(frozen_all), 'n_frequent': len(t_freq),
    'n_mid': len(t_mid), 'n_rare': len(t_rare),
    'units_with_tanween_adverb': len(tan_units),
    'total_tanween_occ': sum(x['tan_n'] for x in frozen_all),
    'frozen': frozen_all,
    'non_tanween_candidates': {c: (cand_hits.get(c) or phrase_count(c)[0])
                               for c in CAND_SINGLE + CAND_MULTI},
}, open('/home/z/my-project/work/ph1_scan_frozen.json', 'w',
        encoding='utf-8'), ensure_ascii=False, indent=1)

# ================= Rule 2: scientific/technical =================
SCI = {
    'physics_astronomy': [
        'طاقة', 'جاذبية', 'كهرباء', 'مغناطيس', 'مغناطيسية', 'موجات', 'موج',
        'ذرة', 'إلكترون', 'الكترون', 'بروتون', 'نيوترون', 'فوتون',
        'حرارة', 'حرارية', 'ضوء', 'ضوئية', 'صوت', 'سرعة', 'كثافة', 'ضغط',
        'فيزياء', 'فلك', 'كون', 'مجرة', 'كوكب', 'كواكب', 'نجم', 'مدار',
        'جسيم', 'جسيمات', 'إشعاع', 'اشعاع', 'تردد', 'ترددات', 'ذبذبة',
        'ذبذبات', 'مقاومة', 'جهد', 'تيار', 'شحنة', 'احتكاك', 'قصور',
        'انصهار', 'تبخر', 'تجميد', 'انفجار', 'طيف'],
    'chemistry': [
        'كيمياء', 'كيميائي', 'جزيء', 'جزيئات', 'عنصر', 'عناصر', 'مركب',
        'مركبات', 'تفاعل', 'تفاعلات', 'أكسجين', 'اوكسجين', 'هيدروجين',
        'نيتروجين', 'كربون', 'حمض', 'أحماض', 'قاعدة', 'قواعد', 'أيون',
        'أيونات', 'تأكسد', 'مذيب', 'تركيز', 'مذاب', 'ذوبان', 'غاز',
        'غازات', 'بخار', 'معادن', 'كالسيوم', 'حديد', 'زنك', 'مغنيسيوم',
        'بوتاسيوم', 'صوديوم', 'كوليسترول'],
    'biology_medicine': [
        'خلية', 'خلايا', 'نواة', 'ميتوكوندريا', 'جين', 'جينات', 'جينوم',
        'كروموسوم', 'كروموسومات', 'بروتين', 'بروتينات', 'إنزيم',
        'إنزيمات', 'انزيمات', 'هرمون', 'هرمونات', 'دوبامين', 'سيروتونين',
        'كورتيزول', 'أنسولين', 'انسولين', 'مناعة', 'مناعي', 'فيروس',
        'فيروسات', 'بكتيريا', 'ميكروب', 'ميكروبات', 'لقاح', 'لقاحات',
        'تطعيم', 'عدوى', 'التهاب', 'التهابات', 'أيض', 'أعراض', 'تشخيص',
        'علاج', 'جراحة', 'تخدير', 'مفصل', 'مفاصل', 'عظام', 'عضلات',
        'هشاشة', 'كسر', 'كسور', 'جلد', 'بشرة', 'دم', 'قلب', 'شريان',
        'شرايين', 'وريد', 'أوردة', 'هيموجلوبين', 'أنيميا', 'انيميا',
        'فيتامين', 'فيتامينات', 'سعرات', 'سعرة', 'كربوهيدرات', 'دهون',
        'معدة', 'أمعاء', 'قولون', 'كبد', 'كلى', 'غدة', 'غدد', 'درقية',
        'أدرينالين', 'عصبي', 'عصبية', 'أعصاب', 'عصب', 'مخ', 'دماغ',
        'ذاكرة', 'أرق', 'توتر', 'قلق', 'اكتئاب', 'دواء', 'أدوية', 'جرعة',
        'جرعات', 'وصفة', 'صيدلية', 'مستشفى', 'مريض', 'مرضى', 'مرض',
        'أمراض', 'وباء', 'جائحة', 'تحليل', 'تحاليل', 'أشعة', 'رنين',
        'سكان', 'سكر', 'سكري', 'أورام', 'سرطان', 'حساسية', 'تجاعيد',
        'شيخوخة', 'طبيب', 'دكتور', 'جلطات', 'جلطة', 'ضيق', 'تنفس',
        'أوكسجين', 'بول', 'براز', 'غائط', 'تعرق', 'هرمونات'],
    'technology': [
        'تكنولوجيا', 'تكنلوجيا', 'إنترنت', 'انترنت', 'حاسوب', 'كمبيوتر',
        'برمجة', 'مبرمج', 'خوارزمية', 'خوارزميات', 'بيانات', 'شبكة',
        'شبكات', 'سيرفر', 'سيرفرات', 'تطبيق', 'تطبيقات', 'برنامج',
        'برامج', 'سوفتوير', 'هاردوير', 'بكسل', 'شاشة', 'بطارية', 'رامات',
        'معالج', 'معالجات', 'سنسور', 'روبوت', 'روبوتات', 'أتمتة',
        'اتمتة', 'درون', 'لابتوب', 'موبايل', 'اتصالات', 'إلكتروني',
        'الكتروني', 'رقمي', 'رقمية', 'منصة', 'منصات', 'متجر', 'تحديث',
        'إصدار', 'اصدار', 'واجهة', 'كاميرا', 'سماعة', 'سماعات',
        'شريحة', 'شرائح', 'ذاكرة', 'فلاش', 'تنزيل', 'تسجيل'],
    'economics_business': [
        'اقتصاد', 'اقتصادي', 'تضخم', 'استثمار', 'مستثمر', 'عائد',
        'فائدة', 'سوق', 'أسواق', 'بورصة', 'أسهم', 'سهم', 'سندات',
        'سيولة', 'ربح', 'أرباح', 'خسارة', 'خسائر', 'إيرادات', 'ايرادات',
        'مصروفات', 'ميزانية', 'ضرائب', 'ضريبة', 'دخل', 'ائتمان', 'قرض',
        'قروض', 'بنك', 'بنوك', 'تمويل', 'جنيه', 'دولار', 'صرف', 'عملة',
        'عملات', 'ذهب', 'عقار', 'عقارات', 'شركة', 'شركات', 'مشروع',
        'مشاريع', 'ريادة', 'تسويق', 'مبيعات', 'عميل', 'عملاء', 'تاجر',
        'تجارة', 'مورد', 'توريد', 'مخزون', 'سعر', 'أسعار', 'تكلفة',
        'تكاليف', 'منتج', 'منتجات', 'براند', 'علامة', 'موارد', 'رواتب',
        'راتب', 'مرتب', 'تأمين', 'خطة', 'استراتيجية', 'اهداف', 'أهداف',
        'مؤشر', 'مؤشرات', 'تحليل', 'نمو', 'نسبة', 'رأس'],
    'math': [
        'رياضيات', 'معادلة', 'معادلات', 'مئوية', 'إحصاء', 'احصاء',
        'احتمالية', 'متوسط', 'حساب', 'حسابات', 'عدد', 'أعداد', 'كسر',
        'كسور', 'هندسة', 'مثلث', 'دائرة', 'زاوية', 'محور', 'رسم',
        'بياني', 'جبر', 'حساب التفاضل', 'لوغاريتم', 'نظرية', 'نظريات'],
}
SCI_MULTI = {
    'physics_astronomy': ['طاقة شمسية', 'طاقة نووية', 'الطاقة الشمسية'],
    'biology_medicine': ['حمض نووي', 'الحمض النووي', 'مضاد حيوي',
                         'المضاد الحيوي', 'مضادات حيوية', 'ضغط الدم',
                         'ضغط دم', 'جهاز عصبي', 'الجهاز العصبي',
                         'دورة دموية', 'الدورة الدموية', 'جهاز هضمي',
                         'الجهاز الهضمي', 'غدة درقية', 'الغدة الدرقية',
                         'خلايا عصبية', 'ناقل عصبي', 'مضاد أكسدة',
                         'مضادات الأكسدة', 'المناعة الذاتية', 'فيتامين د',
                         'هشاشة العظام', 'الزائدة الدودية', 'بلة؟'],
    'technology': ['ذكاء اصطناعي', 'الذكاء الاصطناعي', 'ذكاء اصطناعيا',
                   'بريد إلكتروني', 'البريد الإلكتروني', 'شبكة إنترنت',
                   'شبكات اجتماعية', 'الوسائط الرقمية', 'الواقع الافتراضي',
                   'الواقع المعزز'],
    'economics_business': ['رأس مال', 'رأس المال', 'ناتج محلي',
                           'الناتج المحلي', 'علامة تجارية', 'العلامة التجارية',
                           'عرض وطلب', 'العرض والطلب', 'قيمة مضافة',
                           'التخطيط الاستراتيجي', 'أسعار الفائدة',
                           'سوق الأسهم', 'الميزانية العمومية', 'حقوق الملكية',
                           'رأس المال'],
}
SCI_FLAT = {}
for dom, lst in SCI.items():
    for stem in lst:
        SCI_FLAT.setdefault(stem, dom)
        SCI_FLAT.setdefault('ال' + stem, dom)

sci_hits = defaultdict(Counter)
sci_units = defaultdict(set)
for i, t in enumerate(texts):
    for w in t.split():
        s = skel(w)
        if not s:
            continue
        for v in variants(s):
            if v in SCI_FLAT:
                sci_hits[SCI_FLAT[v]][v] += 1
                sci_units[SCI_FLAT[v]].add(i)
                break

sci_multi_hits = defaultdict(dict)
for dom, terms in SCI_MULTI.items():
    for term in terms:
        if '؟' in term or 'بلة' in term:
            continue
        n, uset = phrase_count(term)
        if n:
            sci_multi_hits[dom][term] = {'occ': n, 'units': len(uset)}

print('\n== RULE 2: scientific/technical ==')
tot_terms = 0
tot_occ = 0
for dom in SCI:
    singles = sci_hits[dom]
    multis = sci_multi_hits.get(dom, {})
    n_terms = len(singles) + len(multis)
    occ = sum(singles.values()) + sum(m['occ'] for m in multis.values())
    tot_terms += n_terms
    tot_occ += occ
    print(f'\n[{dom}] terms={n_terms} occ={occ} '
          f'units={len(sci_units[dom])}')
    print('  singles:', singles.most_common(12))
    if multis:
        print('  multi:', multis)

# domain-salient mining (corpus-driven, catches lexicon gaps)
dom_counters = defaultdict(Counter)
for i, t in enumerate(texts):
    d = episodes[i]
    for w in t.split():
        s = strip_conj(skel(w))
        if len(s) >= 3:
            dom_counters[d][s] += 1
for dom, base in (('medical', 'general'), ('sales', 'general')):
    dc, gc = dom_counters.get(dom, Counter()), dom_counters.get(base, Counter())
    sal = [(w, c) for w, c in dc.items() if c >= 5 and c >= 10 * gc.get(w, 0)]
    sal.sort(key=lambda x: -x[1])
    print(f'\n{dom}-salient (vs {base}, count>=5, 10x): {len(sal)}')
    print('  top40:', sal[:40])

json.dump({
    'criterion': 'domain lexicon intersection + domain-salient mining',
    'sci_hits': {d: dict(c.most_common()) for d, c in sci_hits.items()},
    'sci_multi_hits': {d: v for d, v in sci_multi_hits.items()},
    'totals': {'distinct_terms': tot_terms, 'occurrences': tot_occ},
    'domain_salient': {
        'medical_top60': sorted(
            [(w, c) for w, c in dom_counters.get('medical', Counter()).items()
             if c >= 5 and c >= 10 * dom_counters['general'].get(w, 0)],
            key=lambda x: -x[1])[:60],
        'sales_top60': sorted(
            [(w, c) for w, c in dom_counters.get('sales', Counter()).items()
             if c >= 5 and c >= 10 * dom_counters['general'].get(w, 0)],
            key=lambda x: -x[1])[:60]},
}, open('/home/z/my-project/work/ph1_scan_sci.json', 'w', encoding='utf-8'),
    ensure_ascii=False, indent=1)

# ================= Rule 3: religious =================
REL_PHRASES = [
    'الحمد لله', 'الحمدلله', 'سبحان الله', 'إن شاء الله', 'ان شاء الله',
    'ما شاء الله', 'بسم الله', 'لا إله إلا الله', 'لا اله الا الله',
    'لا حول ولا قوة إلا بالله', 'استغفر الله', 'الله أكبر', 'الله اكبر',
    'جزاك الله', 'توكلنا على الله', 'توكلت على الله', 'بإذن الله',
    'بإذن الله', 'الله يرحمه', 'الله يرحمها', 'الله ينور', 'ربنا يقوي',
    'ربنا يكرم', 'الله يسعد', 'صلى الله عليه وسلم', 'والنبي', 'يا رب',
    'الحمد لله رب العالمين', 'بسم الله الرحمن الرحيم',
    'لا حول ولا قوة الا بالله', 'ماشي يا رب', 'لله', 'والله', 'بالله',
]
REL_WORDS = [
    'الله', 'الإله', 'الرب', 'القرآن', 'قرآن', 'مصحف', 'سورة', 'آية',
    'آيات', 'دعاء', 'استغفار', 'صلاة', 'زكاة', 'صيام', 'رمضان', 'عمرة',
    'حج', 'الجنة', 'النار', 'جهنم', 'القيامة', 'الحساب', 'النبي',
    'الرسول', 'الشريعة', 'فتوى', 'حلال', 'حرام', 'إيمان', 'الشيطان',
    'إبليس', 'ملائكة', 'وحي', 'توبة', 'مغفرة', 'بركة', 'مسجد', 'كعبة',
    'أذان', 'مكة', 'الآخرة', 'اخرة', 'دين', 'ذنب', 'ثواب', 'عقاب',
    'الرحمن', 'الرحيم', 'العزيز', 'الحكيم', 'قضاء', 'الملائكة',
]
VERSES = [
    'قل هو الله أحد', 'الحمد لله رب العالمين', 'ولا تيأسوا من روح الله',
    'إن مع العسر يسرا', 'وما ذلك على الله بعزيز', 'لا يكلف الله نفسا',
    'وقل اعملوا', 'إن الله لا يغير ما بقوم', 'ومن يتق الله يجعل له مخرجا',
    'الرحمن علم القرآن', 'تبارك الذي بيده الملك', 'كل نفس ذائقة الموت',
    'وقل رب زدني علما', 'نحن نقص عليك أحسن القصص',
    'وجعلنا من الماء كل شيء حي', 'الذي خلق فسوى', 'وعد الله لا يخلف الله وعده',
    'إن الله مع الصابرين', 'وسيق الذين كفروا', 'يوم يأتي تأويله',
]

rel_word_hits = Counter()
rel_word_units = defaultdict(set)
for i, t in enumerate(texts):
    for w in t.split():
        s = skel(w)
        if not s:
            continue
        for v in variants(s):
            if v in REL_WORDS:
                rel_word_hits[v] += 1
                rel_word_units[v].add(i)
                break

rel_phrase_hits = {}
rel_phrase_units = {}
for p in REL_PHRASES:
    n, uset = phrase_count(p)
    if n:
        rel_phrase_hits[p] = n
        rel_phrase_units[p] = uset

verse_hits = {}
for v in VERSES:
    n, uset = phrase_count(v)
    if n:
        verse_hits[v] = n

all_rel_units = set()
for u in rel_word_units.values():
    all_rel_units.update(u)
for u in rel_phrase_units.values():
    all_rel_units.update(u)

print('\n== RULE 3: religious ==')
print(f'words: {dict(rel_word_hits.most_common(20))}')
print(f'\nphrases: {dict(sorted(rel_phrase_hits.items(), key=lambda x: -x[1]))}')
print(f'\nverses: {verse_hits}')
print(f'\nunits containing any religious marker: {len(all_rel_units)}/{len(texts)}')

json.dump({
    'criterion': 'marker words + frozen formulas + verse fragments '
                 '(prefix-tolerant phrase matching)',
    'word_hits': dict(rel_word_hits.most_common()),
    'phrase_hits': rel_phrase_hits,
    'verse_hits': verse_hits,
    'units_with_religious_marker': len(all_rel_units),
    'n_units_total': len(texts),
}, open('/home/z/my-project/work/ph1_scan_religious.json', 'w',
        encoding='utf-8'), ensure_ascii=False, indent=1)

print('\nDONE — JSONs saved.')
