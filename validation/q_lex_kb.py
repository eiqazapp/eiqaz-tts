#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q_lex_kb.py — قاعدة المعرفة المنسّقة لتصنيف نطق القاف في الاستعمال المصري
(معيار التصنيف: «كيف ينطق المتحدث المصري القاف في هذه الكلمة داخل كلام
مصري طبيعي؟» — لا «هل الكلمة فصيحة؟»).

مصادر الأدلة (رموز):
  [user]    توجيه المستخدم الصريح (2026-10-04: قرآن/فقط/قانون ق؛ قال وسائر
            الهمزات ء؛ رقم/قرش/قنطار/مقام/قيراط/ترقيم ج من جلسات سابقة)
  [audio]   دليل صوتي من corpus/التجارب: سماع المستخدم أو قياس run1
            (VOT/جهر الإغلاق — qaf_experiment_run1.json + §8)
  [wiki]    Wikipedia: Egyptian Arabic phonology — «*/q/ became [ʔ] in Cairo
            and the Nile Delta; /q/ reintroduced as a marginal phoneme …
            particularly relating to certain religious words, besides others
            such as those deriving from the root /θ-q-f/ (intellect/culture);
            قانون [q] في الكلام المتأني، قوى homograph»
  [talkin]  talkinarabic.com (قائمة مصرية لكلمات القاف — جلسة 2026-10-01)
  [corpus]  إحصاء corpus نيلتس (المسح الشامل: 31,254 موضعًا / 4,546 هيكلًا)
  [ling]    تحليل لغوي (الانعكاس القاهري غير المعلم: ق→ʔ؛ مجموعة الجيم
            القاهرية المغلقة؛ سجلات الاستعمال)
  [base]    قرار baseline سابق (فئة B=g أو قوائم Q في infer.py)

ترتيب المعالجة: تسلسلي — الإدخال الأحدث قد ينسخ هيكلًا أقدم (override).
الهياكل «المجردة» فقط تُدرج؛ السوابق (ال/و/ف/ب/ل/ك [+ال]) تُولد آليًا.
"""

ENTRIES = [
    # ================= Q_DEFAULT =================
    dict(fam='قرآن', cat='religious', pron='q', conf='high',
         skels=['qr|n', 'qrAn', 'qr|nA', 'qr|ny', 'qr|nyp', 'qrAny',
                'qr|nyA'],
         word='القرآن', ev=[
             '[user] توجيه صريح مرتين (2026-10-02 + 2026-10-04): ق فصحى '
             'افتراضيًا بلا لاحقة',
             '[audio] الموضعان المجردان في corpus سُمِعا قافًا عميقة بأذن '
             'المستخدم (QAF §3-أ: هيقرأ قرآن / مش قرآن نزل)',
             '[wiki] الكلمات الدينية تحتفظ بالقاف الفصحى كمارجينال في مصر',
             '[base] PATCH 11: عائلة كاملة في قائمة Q — أشكال catt موثقة'],
         audio='deep_user_confirmed ×2 (المجردة فقط؛ المعرفة 0 مواضع)',
         notes='قرار سياسة جديد: نقل العائلة من فئة B [g] إلى q افتراضيًا — '
               'تعارض مع baseline يُعرض للمراجعة. عاميًا ʔarʔān مسموعة عند '
               'بعض المتحدثين لكن السجل الديني غالب.'),
    dict(fam='فقط', cat='formal_common', pron='q', conf='medium_high',
         skels=['fqT'], word='فقط', ev=[
             '[user] توجيه صريح (2026-10-04)',
             '[corpus] 11 موضعًا (بيئة فتحة plain)',
             '[ling] كلمة سجل رسمي أصلًا — العامية تقول «بس»؛ حين تُستخدم '
             'فإن النطق الفصيح faqaṭ هو الشائع في الاستعمال المتعلم'],
         audio='غير مقيس',
         notes='تغيير سياسة: ليست في أي قائمة حاليًا (baseline=همزة). '
               'نطق faʔaṭ العامي موجود لكنه أقلية في السجل المستخدم.'),
    dict(fam='قانون', cat='legal', pron='q', conf='medium',
         skels=['qAnwn', 'qAnwny', 'qAnwnyp', 'qAnwnyA', 'qwAnyn'],
         word='قانون', ev=[
             '[user] توجيه صريح (2026-10-04)',
             '[wiki] قانون /ʔæˈnuːn/ عاميًا وتُفرَّق [qɑˈnuːn] (القانون) عن '
             '[ʔæˈnuːn] (القانون الموسيقي kanun) في الكلام المتأني',
             '[corpus] 44+15+8+7+22 موضعًا (~96 مع السوابق)',
             '[base] فئة B حاليًا [g] — تعارض يُعرض للمراجعة'],
         audio='غير مقيس لكلمة قانون ذاتها (قياس run1 للفئة B: قانون قاست g '
               'لكن ذلك كان بإجبار توكن v)',
         notes='تغيير سياسة مقترح: g→q. تذبذب حقيقي موثق (gānūn قاهري قديم '
               '/ ʔānūn عامي / qānūn متعلم-إخباري) — أذن المستخدم حكمت q. '
               'مراجعة بشرية موصى بها.'),
    dict(fam='قطع/قطعة', cat='education_everyday', pron='q', conf='high',
         skels=['qTE', 'qTEp', 'qTEtyn', 'qTEpA', 'qTAE', 'qTAEAt'],
         word='قطعة', ev=[
             '[audio] تأكيد سمعي من المستخدم («نعم قطعة و قِطْعَةً أصبحت '
             'تنطق الآن ق أصيلة فصحى» 2026-10-01)',
             '[audio] run1: خام_منفرد q? (VOT=56ms انفجار عميق صامت الإغلاق)',
             '[corpus] عائلة قطع 108 مواضع (QAF §3-أ)',
             '[base] VERIFIED — زرع auto ساري'],
         audio='q?_user_confirmed (VOT=56ms)',
         notes='أقوى حالة مثبتة للقاف العميقة داخل الكلام المصري في corpus '
               'نفسه. «قطعة» اليومية تُسمع ʔuʔʕا عند آخرين — قرار المشروع '
               'يتبع صوت corpus.'),
    dict(fam='قصة/قصص', cat='education_everyday', pron='q',
         conf='medium_high',
         skels=['qSp', 'qSS'], word='قصة', ev=[
             '[audio] run1: q? (VOT=38ms) — ترقية verified في PATCH 10',
             '[talkin] تُدرج في قائمة كلمات القاف المصرية',
             '[user] جملة «قصص{ق}» (2026-10-02) — بالعلامة',
             '[corpus] 250 موضعًا (قصة 188 + قصص 62)'],
         audio='q? (VOT=38ms) — غير مؤكدة سمعيًا بكلمة قصة ذاتها',
         notes='عاميًا ʔuṣṣا شائعة؛ الدليل الصوتي+المصدر الخارجي يرجحان q '
               'لسجل القصة/القراءة. مراجعة موصى بها.'),
    dict(fam='قطب', cat='science', pron='q', conf='medium',
         skels=['qTb', 'qTbyn'], word='قطب (الجغرافيا/الفيزياء)', ev=[
             '[audio] run1: q? (VOT=42ms) — verified في PATCH 10',
             '[ling] مصطلح علمي (القطب الشمالي/الجنوبي) — سجل تعليمي'],
         audio='q? (VOT=42ms)', notes='غير موجودة في corpus (0 مواضع).'),
    dict(fam='حقيقة', cat='epistemic_formal', pron='q', conf='medium',
         skels=['Hqyqp', 'Hqyqy', 'Hqyqyp'], word='حقيقة', ev=[
             '[audio] run1 §8.3: حقيقة خامًا قاست عمقًا (VOT=80ms) عند '
             'speaker 0 — سلوك متعلم لكل كلمة على حدة',
             '[corpus] أضخم عائلة قاف رسمية: ~949 موضعًا (حقيقة 197+ '
             'حقيقي 379+ حقيقية 373)',
             '[base] QAF_Q_TRUST — خام عميق في auto'],
         audio='deep_raw (VOT=80ms speaker 0)',
         notes='الاستعمال العامي ħaʔīʔا واسع، لكن صوت corpus (الذي سيتعلمه '
               'النموذج) يُخرجها عميقة — التصنيف يتبع صوت corpus. مراجعة.'),
    dict(fam='تقريب', cat='education_math', pron='q', conf='medium_high',
         skels=['tqryb', 'tqrybA', 'tqryby', 'tqrybyp'], word='تقريب', ev=[
             '[user] مثال صريح (جلسة المعلم — مصطلح رياضيات)',
             '[audio] run1 §8.3: تقريب قاوم الجيم والكاف — قاست q? في كل '
             'التصييرات حتى جيم_g (VOT=52ms خام)',
             '[base] قائمة Q (tqryb/tqrybA/tqryby/tqrybyp)'],
         audio='q?_resistant (VOT=52ms في كل التصييرات)',
         notes='أشدها مقاومة للاستبدال — الشكل نفسه يستدعي العمق.'),
    dict(fam='قسمة', cat='education_math', pron='q', conf='medium_high',
         skels=['qsmp', 'qsmt'], word='قسمة (الرياضيات)', ev=[
             '[user] مثال صريح (جلسة المعلم)',
             '[base] tier1 — زرع auto ساري',
             '[corpus] 4 مواضع (القسمة/تقسيمة) — نادر'],
         audio='غير مقيس', notes='عاميًا ʔisma؛ السجل التعليمي qisma.'),
    dict(fam='منطق', cat='formal', pron='q', conf='medium',
         skels=['mnTq', 'mnTqy', 'mnTqyp'], word='منطق', ev=[
             '[base] tier1 — زرع auto ساري (mnTq)',
             '[corpus] 12+108+29+11 ≈ 160 موضعًا',
             '[ling] سجل متعلم — «manṭiq» في الاستعمال الرسمي/الأكاديمي'],
         audio='غير مقيس',
         notes='انفصال صرفي مهم داخل الجذر نفسه: منطق (q سجل رسمي) مقابل '
               'منطقة (ء دائمًا — دخيل أغلبية ساحقة). لا يُعمم الحكم.'),
    dict(fam='ثقافة', cat='culture_formal', pron='q', conf='medium',
         skels=['^qAfp', '^qAfy', '^qAfyp', 'mvqf', 'tvqyf', '^qAftnA', '^qAfAt'],
         word='ثقافة', ev=[
             '[wiki] صراحة: الجذر /θ-q-f/ (الفكر والثقافة) من المواضع التي '
             'أعيدت فيها القاف /q/ في مصر',
             '[talkin] ثقافة/مثقف/تثقيف في القائمة المصرية',
             '[corpus] ~120+ موضعًا (ثقافة 53+49+16+10+8+...، ثقافةtnا 20)'],
         audio='غير مقيس',
         notes='انفصال صرفي داخل الجذر: ثقافة (q — ويكيبيديا صراحة) مقابل '
               'ثقة (ء دائمًا — الجذر ث-ق-ت). عاميًا saʔāfa مسموعة. '
               'ملاحظة توثيقية: إدخالات infer.py القديمة vqAfp/mvqf/tvqyf '
               'بحرف v لا تطابق باكوالتير (ث=^) — إدخالات ميتة لا تطابق '
               'شيئًا؛ الصحيح ^qAfp وهو الموجود أيضًا في القائمة.'),
    dict(fam='تقوى', cat='religious', pron='q', conf='medium_high',
         skels=['tqwY'], word='تقوى', ev=[
             '[talkin] في القائمة المصرية', '[base] قائمة Q',
             '[corpus] موضع واحد'],
         audio='غير مقيس', notes='سجل قرآني/ديني.'),
    dict(fam='قدوس', cat='religious_names', pron='q', conf='medium_high',
         skels=['qdws'], word='القدوس', ev=[
             '[wiki] الكلمات الدينية تحتفظ بالقاف',
             '[ling] أسماء الله الحسنى تُتلى بالقاف الفصحى في الاستعمال '
             'المصري (القدوس في الأذكار والتلاوة)'],
         audio='غير مقيس', notes='ليست في corpus (0).'),
    dict(fam='قهار', cat='religious_names', pron='q', conf='medium_high',
         skels=['qAhAr'], word='القهّار', ev=[
             '[wiki] الكلمات الدينية', '[ling] من أسماء الله'],
         audio='غير مقيس', notes='ليست في corpus (0).'),
    dict(fam='قيوم', cat='religious_names', pron='q', conf='medium_high',
         skels=['qywm'], word='القيّوم', ev=[
             '[ling] من أسماء الله (الحي القيوم)'],
         audio='غير مقيس', notes='ليست في corpus (0).'),
    dict(fam='قابض', cat='religious_names', pron='q', conf='medium_high',
         skels=['qAbD'], word='القابض', ev=[
             '[ling] من أسماء الله'],
         audio='غير مقيس', notes='ليست في corpus (0).'),
    dict(fam='قريش', cat='religious', pron='q', conf='medium_high',
         skels=['qryS'], word='قريش', ev=[
             '[ling] سورة قريش والنسب الديني — تُنطق قافًا في الاستعمال '
             'المصري (سورة قريش في التلاوة)'],
         audio='غير مقيس', notes='ليست في corpus (0).'),
    dict(fam='مستقيم', cat='education_math', pron='q', conf='medium',
         skels=['mstqym', 'mstqymp', 'AstqAmp'], word='مستقيم', ev=[
             '[base] tier1 (mstqym) — زرع auto', '[ling] سجل هندسي/رياضي',
             '[corpus] موضعان'],
         audio='غير مقيس', notes='عاميًا mustaʔīm.'),
    dict(fam='انقسام', cat='science_formal', pron='q', conf='medium',
         skels=['AnqsAm'], word='انقسام', ev=[
             '[base] tier1 — زرع auto', '[ling] مصطلح علمي (انقسام الخلية)',
             '[corpus] موضع واحد'],
         audio='غير مقيس', notes='عاميًا inʔisām.'),
    dict(fam='قطاع', cat='media_formal', pron='q', conf='medium',
         skels=['qTAE', 'qTAEAt'], word='قطاع', ev=[
             '[base] tier1 — زرع auto', '[ling] سجل إخباري/رسمي',
             '[corpus] 14+8+ مواضع'],
         audio='غير مقيس', notes='عاميًا ʔiʕāʕ نادر الاستعمال أصلاً.'),

    # ================= G_DEFAULT =================
    dict(fam='رقم', cat='everyday_numbers', pron='g', conf='high',
         skels=['rqm', 'rqmp', 'rqmnp', '>rqAm', 'ArqAm', 'rqmyn', 'trqym'],
         word='رقم', ev=[
             '[user] أمثلة الجيم الصريحة من جلسات سابقة (رقم/أرقام/ترقيم)',
             '[base] فئة B المعتمدة', '[corpus] ~200 موضع',
             '[audio] run1 §8.3: رقم قاست إزالة جهر (جهر=0.25، VOT=46ms '
             '≈ [k]-ish) — تحقق توكن v يتفاوت بالكلمة'],
         audio='devoiced_g/[k]-ish (run1) — مراجعة سمعية عند تبديل المتحدث',
         notes='ملاحظة صوتية: القياس الآلي أظهر ميلاً نحو الصمت — أذن '
               'المستخدم هي الحكم.'),
    dict(fam='قرش', cat='everyday_money', pron='g', conf='high',
         skels=['qr$', 'qr$yn'], word='قرش', ev=[
             '[user] جملة التقييم «القرش غالي»', '[base] فئة B',
             '[corpus] 6 مواضع'],
         audio='غير مقيس (جملة التقييم فقط)',
         notes='من الكلمات القاهرية القليلة الثابتة الجيم.'),
    dict(fam='مقال', cat='media', pron='g', conf='high',
         skels=['mqAl', 'mqAlAt'], word='مقال', ev=[
             '[ling] من مجموعة الجيم القاهرية المغلقة (magāl)',
             '[corpus] 24+ موضعًا (المقال)'],
         audio='غير مقيس', notes='كلمة الصحافة اليومية.'),
    dict(fam='قيراط', cat='everyday_money', pron='g', conf='high',
         skels=['qyrAT', 'qrAryT'], word='قيراط', ev=[
             '[base] فئة B', '[ling] girāṭ قاهرية (المجوهرات والأرض)'],
         audio='غير مقيس', notes='ليست في corpus.'),
    dict(fam='قنطار', cat='everyday_measure', pron='g', conf='medium_high',
         skels=['qnTAr', 'qnATyr'], word='قنطار', ev=[
             '[base] فئة B', '[ling] ganāṭar قاهرية (أسواق الجملة)'],
         audio='غير مقيس', notes='ليست في corpus.'),
    dict(fam='مقام', cat='religious_shrine', pron='g', conf='medium',
         skels=['mqAm', 'mqAmAt'], word='مقام (الضريح)', ev=[
             '[base] فئة B', '[corpus] 4 مواضع',
             '[ling] «magām» للضريح قاهرية؛ معنى المنزلة/الحال «maʔām» '
             'همزة — انفصال دلالي داخل الكلمة الواحدة'],
         audio='غير مقيس', notes='راجع حسب المعنى في المراجعة البشرية.'),

    # ================= HAMZA_DEFAULT (صريح — كثيرة التردد) =================
    dict(fam='قال/يقول', cat='everyday_verb', pron='hamza', conf='high',
         skels=['qAl', 'qAlt', 'qAlwA', '>qwl', 'qwl', 'qwlk', 'qwlh', 'qwlhA',
                'qwlwA', 'yqwl', 'yqwlk', 'yqwlh', 'yqwlhA', 'yqwlwA',
                'byqwl', 'byqwlk', 'byqwlh', 'byqwlhA', 'byqwlwA',
                'btqwl', 'btqwlk', 'btqwlhA', 'tqwl', 'nqwl', 'bnqwl',
                'nnqwl', 'wyqwl', 'wbyqwl', 'qlnA', 'qlnAh', 'qlnAhA',
                'qlt', 'qlty', 'qwlnA', 'bttqAl', 'ttqAl', 'bytqAl',
                'bqAlhA', 'bqAlnA'],
         word='قال', ev=[
             '[user] توجيه صريح (2026-10-04): قال وسائر كلمات الهمزة ء',
             '[wiki] الانعكاس القاهري غير المعلم: ق→ʔ',
             '[corpus] أضخم عائلة فعلية (~2000+ موضع مع التصريفات)'],
         audio='تنبيه §8.3: قال خامًا قاست عمقًا (VOT=70ms) عند speaker 0 — '
               'سلوك متعلم شاذ موثق؛ أذن المستخدم حكمت الهمزة والعلامة {ء} '
               'لا تفرضها على كلمة تعلمها النموذج عميقة',
         notes='قرار المستخدم: ء. القياس الشاذ يُوثق ولا يغير التصنيف.'),
    dict(fam='بقى', cat='everyday_verb', pron='hamza', conf='high',
         skels=['bqY', 'ybqY', 'bybqY', 'btbqY', 'htbqY', 'hybqY', 'wbqY',
                'wybqY', 'bnbqY', 'nbqY', 'bqt', 'wbqt', 'bqA$', 'bqt$',
                'ybqA$', 'mbqA$', 'bqyt', 'bqynA', 'bqwA', 'bybqwA'],
         word='بقى', ev=[
             '[wiki] ق→ʔ', '[corpus] 1132+ موضعًا (ثاني أعلى هيكل)'],
         audio='غير مقيس', notes='baʔa — عامية خالصة.'),
    dict(fam='وقت', cat='everyday_time', pron='hamza', conf='high',
         skels=['wqt', 'wqth', 'wqthA', 'wqtk', 'dlwqty', '>wqAt',
                'm&qt', 'm&qtp', 'wqft'], word='وقت/دلوقتي', ev=[
             '[wiki] ق→ʔ', '[corpus] ~800 موضعًا'],
         audio='غير مقيس', notes='il-waʔt / dilwaʔti.'),
    dict(fam='نقطة', cat='everyday', pron='hamza', conf='high',
         skels=['nqTp', 'nqT', 'nqAT', 'nqTtyn', 'nqTAn'], word='نقطة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~900 موضعًا'],
         audio='غير مقيس', notes='nuʔṭa — من أعلى الكلمات.'),
    dict(fam='سوق', cat='everyday', pron='hamza', conf='high',
         skels=['swq'], word='سوق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~450 موضعًا'],
         audio='غير مقيس', notes='is-sūʔ.'),
    dict(fam='عقار', cat='everyday_real_estate', pron='hamza', conf='high',
         skels=['EqAr', 'EqArAt', 'EqAry', 'EqAryp'], word='عقار', ev=[
             '[wiki] ق→ʔ', '[corpus] ~600 موضعًا'],
         audio='غير مقيس', notes='ʕaʔār.'),
    dict(fam='شقة', cat='everyday_home', pron='hamza', conf='high',
         skels=['$qp', '$qq'], word='شقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~350 موضعًا'],
         audio='غير مقيس', notes='šaʔʔa.'),
    dict(fam='منطقة', cat='everyday_geo', pron='hamza', conf='high',
         skels=['mnTqp', 'mnATq'], word='منطقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~370 موضعًا'],
         audio='غير مقيس',
         notes='مقابل منطق q — انفصال داخل الجذر ن-ط-ق موثق في مدخل منطق.'),
    dict(fam='طريقة/طريق', cat='everyday', pron='hamza', conf='high',
         skels=['Tryqp', 'Tryq', 'Trq'], word='طريقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~400 موضعًا'],
         audio='غير مقيس', notes='ṭarīʔa / ṭarīʔ.'),
    dict(fam='تقرير', cat='media', pron='hamza', conf='high',
         skels=['tqryr', 'tqryrAt'], word='تقرير', ev=[
             '[wiki] ق→ʔ', '[corpus] أعلى هيكل منفرد: 1441+ موضعًا'],
         audio='غير مقيس',
         notes='taʔrīr — الإخباري المصري نفسه يقول همزة في الكلام المتصل.'),
    dict(fam='فريق/فرق', cat='everyday', pron='hamza', conf='high',
         skels=['fryq', 'frq', 'frqAt', 'frwqAt', 'tfrq', 'yfrq', 'nfrq',
                'ftArq', 'ftArAq', 'yftArq', 'mfArqp'], word='فريق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~350 موضعًا'],
         audio='غير مقيس', notes='farīʔ / farʔ.'),
    dict(fam='عمق', cat='everyday', pron='hamza', conf='high',
         skels=['Emyq', '>Emq', 'AEmq', 'Emyqp', 'Emq', 'mEmq'], word='عمق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~200 موضعًا'],
         audio='غير مقيس', notes='ʕamīʔ/ʕamuʔ.'),
    dict(fam='عقل/أقل', cat='everyday', pron='hamza', conf='high',
         skels=['>ql', 'Eql', 'Eqlyp', 'Eqlk'], word='عقل/أقل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~250 موضعًا'],
         audio='غير مقيس', notes='ʕaʔl / aʔall.'),
    dict(fam='قد/قدام', cat='everyday', pron='hamza', conf='high',
         skels=['qd', 'qdAm', 'qdAmk', 'qdAmh', 'qdAmnA', 'lqdAm'],
         word='قد/قدام', ev=[
             '[wiki] ق→ʔ', '[corpus] ~250 موضعًا'],
         audio='غير مقيس', notes='ʔadd / ʔaddām.'),
    dict(fam='قديم', cat='everyday', pron='hamza', conf='high',
         skels=['qdym', 'qdymp', 'Alqdm'], word='قديم', ev=[
             '[wiki] ق→ʔ', '[corpus] ~120 موضعًا'],
         audio='غير مقيس', notes='ʔadīm.'),
    dict(fam='قريب', cat='everyday', pron='hamza', conf='high',
         skels=['qryb', 'qrybp', '>qrb', 'qrAb', 'qrb'], word='قريب', ev=[
             '[wiki] ق→ʔ', '[corpus] ~120 موضعًا'],
         audio='غير مقيس', notes='ʔarīb / ʔarīb mn... / ʔaʔrabb.'),
    dict(fam='قهوة', cat='everyday_food', pron='hamza', conf='high',
         skels=['qhwp'], word='قهوة', ev=[
             '[ling] eladawy.blog: «ألم (قلم) وأمر (قمر)» — ق→ء المدن',
             '[corpus] 70 موضعًا'],
         audio='غير مقيس', notes='ʔahwa.'),
    dict(fam='قلب', cat='everyday_body', pron='hamza', conf='high',
         skels=['qlb', 'qlbh'], word='قلب', ev=[
             '[wiki] ق→ʔ', '[corpus] ~100 موضعًا',
             '[ling] ʔalb — يقينية لغوية مطلقة في العامية'],
         audio='غير مقيس',
         notes='تعارض مع قائمة Q (qlb موجود فيها عبر أشكال corpus) — '
               'الإدخال اللغوي الحاسم: ء. عائلة قلوب كذلك.'),
    dict(fam='قلوب', cat='everyday_body', pron='hamza', conf='high',
         skels=['qlwb'], word='قلوب', ev=[
             '[wiki] ق→ʔ', '[corpus] موضع واحد',
             '[ling] ʔulūb — قاطعة عاميًا'],
         audio='غير مقيس',
         notes='تعارض صريح مع QAF_Q_STUDY_FORMS (قُلُوبُ ببيئة قُ عميقة '
               'قاعدة البيئة نصية لا صوتية — إيجابية كاذبة).'),
    dict(fam='قوة', cat='everyday', pron='hamza', conf='high',
         skels=['qwp', 'qwth', 'qwAEd'], word='قوة', ev=[
             '[audio] run1: قوة قاستت همزة (عامية متعلمة — QAF tier2 '
             'الموثق)', '[wiki] ق→ʔ', '[corpus] ~180 موضعًا'],
         audio='hamza_measured (run1)',
         notes='تعارض مع QAF_Q_STUDY_FORMS (قُوَّةٍ). في السياق الفيزيائي '
               'التعليمي قد تُنطق q — لكن الافتراضي ء (مقيس).'),
    dict(fam='قيمة', cat='everyday', pron='hamza', conf='high',
         skels=['qymp', 'qymth', 'qymthA', 'qym'], word='قيمة', ev=[
             '[audio] run1: قيمة قاستت همزة (251+241 موضعًا — عامية '
             'متعلمة، QAF tier2 الموثق)', '[wiki] ق→ʔ',
             '[corpus] ~600 موضعًا'],
         audio='hamza_measured (run1)',
         notes='تعارض صريح مع QAF_Q_STUDY_FORMS (قِيمَةُ ببيئة قِ). الحسم '
               'الصوتي: ء.'),
    dict(fam='قلّ/يقلل', cat='everyday_verb', pron='hamza', conf='high',
         skels=['btql', 'tql', 'yql', 'yqll', 'byqll', 'btqll', 'tqll',
                'byql', 'tqlyl', 'ql', 'qlp', 'nqll'], word='يقلل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~200 موضعًا'],
         audio='غير مقيس', notes='yiʔallil.'),
    dict(fam='علاقة', cat='everyday', pron='hamza', conf='high',
         skels=['ElAqp', 'ElAqAt', 'ElAqtnA'], word='علاقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~250 موضعًا'],
         audio='غير مقيس', notes='ʕalāʔa.'),
    dict(fam='ثقة', cat='everyday', pron='hamza', conf='high',
         skels=['^qp', 't^q', 'b^qp'], word='ثقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~160 موضعًا'],
         audio='غير مقيس',
         notes='مقابل ثقافة q — انفصال داخل جذرين متشابهين (ث-ق-ت مقابل '
               'ث-ق-ف) — ويكيبيديا تفصل صراحة.'),
    dict(fam='دقيق/دقيقة', cat='everyday_time', pron='hamza', conf='high',
         skels=['dqyq', 'dqyqp', 'dqAyq', 'dqA}q', 'bdqp', 'bdqyp'],
         word='دقيقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~300 موضعًا'],
         audio='غير مقيس',
         notes='daʔīʔa — من أكثر الكلمات اليومية. تعارض مع أشكال corpus '
               'في Q (اِلدَّقِيق — إيجابية بيئة كاذبة).'),
    dict(fam='مستقبل', cat='everyday', pron='hamza', conf='high',
         skels=['mstqbl', 'Almstqblyp'], word='مستقبل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~220 موضعًا'],
         audio='غير مقيس', notes='il-mustaʔbal.'),
    dict(fam='نقدر/يقدر', cat='everyday_verb', pron='hamza', conf='high',
         skels=['nqdr', 'yqdr', 'tqdr', 'byqdr', 'btqdr', '>qdr', 'htqdr',
                'hyqdr', 'yqdrwA', 'lqdr'], word='نقدر', ev=[
             '[wiki] ق→ʔ', '[corpus] ~300 موضعًا'],
         audio='غير مقيس', notes='niʔdar.'),
    dict(fam='يقدم', cat='everyday_verb', pron='hamza', conf='high',
         skels=['byqdm', 'btqdm', 'byqdmh', 'byqdmhA', 'btqdmh', 'btqdmhA',
                'byqdmwA', 'yqdm', 'nqdm', 'bnqdmh', 'bnqdmhA', 'tqdm',
                'mqdm', 'Almqdm'],
         word='يقدم', ev=[
             '[wiki] ق→ʔ', '[corpus] ~300 موضعًا'],
         audio='غير مقيس', notes='yiʔaddim.'),
    dict(fam='عقد/عقود', cat='everyday', pron='hamza', conf='high',
         skels=['Eqd', '>Eqd', 'Eqwd', 'Eqdp'], word='عقد', ev=[
             '[wiki] ق→ʔ', '[corpus] ~90 موضعًا'],
         audio='غير مقيس', notes='ʕaʔd / ʕuʔūd.'),
    dict(fam='حلقة', cat='everyday_tv', pron='hamza', conf='high',
         skels=['Hlqp'], word='حلقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~85 موضعًا'],
         audio='غير مقيس', notes='ħalʔa (حلقة المسلسل — corpus تلفزيوني!).'),
    dict(fam='مقارنة', cat='everyday', pron='hamza', conf='high',
         skels=['mqArnp', 'yqArn', 'nqArn'], word='مقارنة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~110 موضعًا'],
         audio='غير مقيس', notes='muʔārana.'),
    dict(fam='نقاش', cat='everyday', pron='hamza', conf='high',
         skels=['nqA$'], word='نقاش', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='naʔāš.'),
    dict(fam='قصيرة', cat='everyday', pron='hamza', conf='high',
         skels=['qSyrp', 'qSyr'], word='قصير', ev=[
             '[wiki] ق→ʔ', '[corpus] ~45 موضعًا'],
         audio='غير مقيس', notes='ʔaṣīr.'),
    dict(fam='قلق', cat='everyday', pron='hamza', conf='high',
         skels=['qlq', 'btqlq', 'mqlqp', 'mqlq'], word='قلق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~35 موضعًا'],
         audio='غير مقيس', notes='ʔalaʔ.'),
    dict(fam='سباق', cat='everyday_sport', pron='hamza', conf='high',
         skels=['sbAq'], word='سباق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~35 موضعًا'],
         audio='غير مقيس', notes='sibāʔ.'),
    dict(fam='صفقة', cat='everyday_business', pron='hamza', conf='high',
         skels=['Sfqp'], word='صفقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='ṣafʔa.'),
    dict(fam='تسويق', cat='business', pron='hamza', conf='medium_high',
         skels=['tswyq', 'tswyqyp'], word='تسويق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~110 موضعًا'],
         audio='غير مقيس',
         notes='taswīʔ عاميًا؛ الرسمي الإداري قد يقول q — مراجعة خفيفة.'),
    dict(fam='تقليدي', cat='formal', pron='hamza', conf='medium_high',
         skels=['tqlydy', 'tqlydyp', 'tqlyd'], word='تقليدي', ev=[
             '[wiki] ق→ʔ', '[corpus] ~65 موضعًا'],
         audio='غير مقيس', notes='taʔlīdī — السجل الرسمي q ممكن.'),
    dict(fam='تطبيق', cat='tech_everyday', pron='hamza', conf='medium_high',
         skels=['tTbyq', 'tTbyqAt'], word='تطبيق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~80 موضعًا'],
         audio='غير مقيس',
         notes='taṭbīʔ في الكلام المصري اليومي (المطورون أنفسهم)؛ الرسمي '
               'taṭbīq — مراجعة.'),
    dict(fam='قائمة/قائم', cat='everyday', pron='hamza', conf='high',
         skels=['qA}mp', 'qA}m', 'qA}d', 'qAym', 'qAymp'], word='قائمة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~60 موضعًا'],
         audio='غير مقيس',
         notes='ʔāʔima / ʔāʔim. تعارض مع قائمة Q (qA}mp — بيئة نصية كاذبة).'),
    dict(fam='قاعدة', cat='everyday_math', pron='hamza', conf='high',
         skels=['qAEdp', 'qAEd', 'qwAEd'], word='قاعدة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~130 موضعًا'],
         audio='غير مقيس',
         notes='ʔāʕida حتى في الشرح الرياضي غالبًا؛ الرسمي qāʕida ممكن — '
               'تعارض مع قائمة Q (qAEdp من أمثلة المستخدم القديمة).'),
    dict(fam='موقف', cat='everyday', pron='hamza', conf='high',
         skels=['mwqf', 'mwAqf'], word='موقف', ev=[
             '[wiki] ق→ʔ', '[corpus] ~75 موضعًا'],
         audio='غير مقيس',
         notes='mawʔif. تعارض مع أشكال corpus في Q (Almwqf).'),
    dict(fam='وقع/يقع', cat='everyday_verb', pron='hamza', conf='high',
         skels=['wqE', 'yqE', 'byqE', 'btqE', 'wqEt', 'tqE', 'nqE', 'bnqE', 'hnqE', 'wqwE'], word='يقع', ev=[
             '[wiki] ق→ʔ', '[corpus] ~90 موضعًا'],
         audio='غير مقيس', notes='yiʔaʕ (حدث) — مقابل واقع (مدخل مستقل).'),
    dict(fam='علق', cat='everyday_verb', pron='hamza', conf='high',
         skels=['Elq', 'wyElqwA', 'yElq'], word='يعلق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='yiʕlaʔ.'),
    dict(fam='تلاقي/لقي', cat='everyday_verb', pron='hamza', conf='high',
         skels=['lqyt', 'tlAqy', 'ylAqy', 'nlAqy', 'hnlAqy', 'bnlAqy',
                'btlAqy', 'bylAqy', 'tlAqyh', 'tlAqyhA', 'ylAqwA', 'lqY',
                'AllqTp', 'lqTp'],
         word='تلاقي', ev=[
             '[wiki] ق→ʔ', '[corpus] ~350 موضعًا'],
         audio='غير مقيس', notes='tilāʔi / laʔīt (لقطة laʔṭa كذلك).'),
    dict(fam='ينقل/نقل', cat='everyday_verb', pron='hamza', conf='high',
         skels=['yntql', 'byntql', 'nntql', 'nnql', 'nql', 'Alnql', 'nqlp',
                'Alnqlp', 'wynqlhA', 'ynqlnA', 'bynql', 'mnql'],
         word='ينقل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~150 موضعًا'],
         audio='غير مقيس', notes='yinʔal.'),
    dict(fam='ناقص', cat='everyday', pron='hamza', conf='high',
         skels=['nqS', 'nAqS'], word='ناقص', ev=[
             '[wiki] ق→ʔ', '[corpus] ~35 موضعًا'],
         audio='غير مقيس', notes='nāʔiṣ.'),
    dict(fam='خلق', cat='everyday_verb', pron='hamza', conf='high',
         skels=['xlq', 'byxlq', 'btxlq', 'yxlq', 'nxlq', 'Almxlwq'],
         word='يخلق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~140 موضعًا'],
         audio='غير مقيس', notes='yixlaʔ (المخلوق maxlūʔ كذلك).'),
    dict(fam='قليل', cat='everyday', pron='hamza', conf='high',
         skels=['qlyl', 'qlylp'], word='قليل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~70 موضعًا'],
         audio='غير مقيس', notes='ʔalīl.'),
    dict(fam='ورق', cat='everyday', pron='hamza', conf='high',
         skels=['wrq', 'wrqp'], word='ورق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~100 موضعًا'],
         audio='غير مقيس', notes='waraʔ.'),
    dict(fam='فوق', cat='everyday', pron='hamza', conf='high',
         skels=['fwq', 'lfwq', 'Alfwq'], word='فوق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~100 موضعًا'],
         audio='غير مقيس', notes='fōʔ.'),
    dict(fam='طاق', cat='everyday', pron='hamza', conf='high',
         skels=['TAqp', 'TAqth'], word='طاقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~110 موضعًا'],
         audio='غير مقيس',
         notes='ṭāʔa. تعارض مع قائمة Q (TAqp/TAqAt).'),
    dict(fam='مقابل', cat='everyday', pron='hamza', conf='high',
         skels=['mqAbl', 'btqAbl', 'mqAblp'], word='مقابل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~65 موضعًا'],
         audio='غير مقيس', notes='maʔābl.'),
    dict(fam='صدق/صديق', cat='everyday', pron='hamza', conf='high',
         skels=['Sdq', 'AlSdq', 'Sdyq', 'Sdyqp', 'SAdq', 'ySdq', 'bSdq'],
         word='صديق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~70 موضعًا'],
         audio='غير مقيس', notes='ṣadīʔ / ṣadaʔ.'),
    dict(fam='حق', cat='everyday', pron='hamza', conf='high',
         skels=['Hq', 'Hqh', 'Hqk', 'ystHq', 'tstHq', 'yHqq', 'nHqq',
                'tHqq', 'AltHqq', 'bHq'],
         word='حق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~90 موضعًا'],
         audio='غير مقيس',
         notes='ħaʔʔ عاميًا. السياق الديني/القانوني al-ħaqq q — مدخل '
               'context_dependent مستقل للحقيقة المطلقة إن لزم.'),
    dict(fam='عقبة', cat='everyday', pron='hamza', conf='high',
         skels=['Eqbp'], word='عقبة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~20 موضعًا'],
         audio='غير مقيس', notes='ʕaʔba.'),
    dict(fam='قصر', cat='everyday', pron='hamza', conf='high',
         skels=['qSr', 'qSrp'], word='قصر', ev=[
             '[wiki] ق→ʔ', '[corpus] 5 مواضع'],
         audio='غير مقيس', notes='ʔaṣr (قصر العدل ء كذلك في الكلام اليومي).'),
    dict(fam='بطاقة', cat='everyday_objects', pron='hamza', conf='high',
         skels=['bTAqp'], word='بطاقة', ev=[
             '[ling] إملاء ق لكن النطق baṭāʔa دائمًا — اصطلاح إملائي '
             'راسخ (أقوى حالة «ق مكتوبة ء منطوقة»)',
             '[corpus] موجودة في الذيل'],
         audio='غير مقيس', notes='baṭāʔa.'),
    dict(fam='زقازيق', cat='place_name', pron='hamza', conf='high',
         skels=['zqAzyq'], word='زقازيق', ev=[
             '[ling] iz-Zagāzīg — القافان كلاهما همزة'],
         audio='غير مقيس', notes='مدينة دلتاوية.'),
    dict(fam='قوله/قولي', cat='everyday', pron='hamza', conf='high',
         skels=['mqwlp'], word='مقولة', ev=[
             '[wiki] ق→ʔ', '[corpus] 10 مواضع'],
         audio='غير مقيس', notes='maʔūla.'),
    dict(fam='قاعة', cat='everyday', pron='hamza', conf='high',
         skels=['qAEp'], word='قاعة', ev=[
             '[wiki] ق→ʔ', '[corpus] 8 مواضع'],
         audio='غير مقيس', notes='ʔāʕa.'),
    dict(fam='قراءة', cat='everyday_reading', pron='hamza', conf='high',
         skels=['qrA\'p', 'qrA\'At', 'yqr>'], word='قراءة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~25 موضعًا'],
         audio='غير مقيس',
         notes='ʔirāya عاميًا؛ «قراءة القرآن» الدينية q — انظر مدخل قرآن.'),
    dict(fam='قروض/قرض', cat='finance', pron='hamza', conf='high',
         skels=['qrD', 'qrwD'], word='قرض', ev=[
             '[wiki] ق→ʔ', '[corpus] ~20 موضعًا'],
         audio='غير مقيس',
         notes='ʔarḍ / ʔurūd. «قرض حسن» الديني q — سياقي.'),
    dict(fam='أقساط', cat='finance', pron='hamza', conf='high',
         skels=['qsT', '>qsAt', '>qsAT', 'tqsyT', 'AltqsyT'], word='قسط', ev=[
             '[wiki] ق→ʔ', '[corpus] ~55 موضعًا'],
         audio='غير مقيس', notes='ʔisṭ (تقسيط taʔsīṭ).'),
    dict(fam='مقدس/قدس', cat='religious_places', pron='hamza',
         conf='medium_high',
         skels=['mqds'], word='مقدس', ev=[
             '[wiki] ق→ʔ', '[corpus] 9 مواضع'],
         audio='غير مقيس',
         notes='muʔaddas عاميًا (بيت المقدس il-muʔaddas!). القدس المدينة '
               'تُنطق غالبًا q في السجل الديني/الإخباري — مدخل مستقل أدناه.'),
    dict(fam='قفول', cat='everyday_objects', pron='hamza', conf='high',
         skels=['mqfwl', 'mqfwlp', 'yqfl', 'tqfl', 'qfl'], word='مقفول', ev=[
             '[wiki] ق→ʔ', '[corpus] ~55 موضعًا'],
         audio='غير مقيس', notes='maʔfūl.'),
    dict(fam='قُدّم/طبق', cat='everyday', pron='hamza', conf='high',
         skels=['Tbq', 'Tbqp', 'TbqAt', 'mqdm', 'qdm'], word='طبق/قدم', ev=[
             '[wiki] ق→ʔ', '[corpus] ~90 موضعًا'],
         audio='غير مقيس', notes='ṭabʔ / ʔidām (قدم القدم).'),
    dict(fam='واقف/قعد', cat='everyday_verb', pron='hamza', conf='high',
         skels=['wAqf', 'yqEd', 'tqEd', 'btqEd', 'qEdp'], word='يقعد', ev=[
             '[wiki] ق→ʔ', '[corpus] ~100 موضعًا'],
         audio='غير مقيس', notes='yiʔʕad / wāʔif.'),
    dict(fam='تثق', cat='everyday_verb', pron='hamza', conf='high',
         skels=['t^q', 'mTwqp', 'wqfp', 'twqf', 'bywqf'], word='تثق/توقف', ev=[
             '[wiki] ق→ʔ', '[corpus] ~70 موضعًا'],
         audio='غير مقيس', notes='tiθiʔ / ittaʔiff.'),
    dict(fam='فقاعة/فقر', cat='everyday', pron='hamza', conf='high',
         skels=['fqAEp', 'fqrAt', 'fqry', 'fqdAn'], word='فقاعة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~45 موضعًا'],
         audio='غير مقيس', notes='faʔʔāʕa / faʔr.'),
    dict(fam='قولون', cat='medical', pron='hamza', conf='high',
         skels=['qwlwn'], word='قولون', ev=[
             '[wiki] ق→ʔ', '[corpus] 10 مواضع'],
         audio='غير مقيس', notes='ʔolōn (قولون عصبي!).'),
    dict(fam='وثيقة/موثوق', cat='formal', pron='hamza', conf='high',
         skels=['w^yqp', 'mw^wq', 'Alw^yqp'], word='وثيقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~30 موضعًا'],
         audio='غير مقيس', notes='waθīʔa / mawθūʔ.'),
    dict(fam='رقابة/مرافق', cat='formal', pron='hamza', conf='high',
         skels=['rqAbp', 'mrAfq', 'rqbp'], word='رقابة/رقبة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~45 موضعًا'],
         audio='غير مقيس', notes='raʔāba (رقابة) / raʔba (الرقبة العنق!).'),
    dict(fam='بقية/الباقي', cat='everyday', pron='hamza', conf='high',
         skels=['bAqy', 'AlbAqy', 'AlbAqp', 'AlbqA\'', 'bqy'],
         word='باقي', ev=[
             '[wiki] ق→ʔ', '[corpus] ~60 موضعًا'],
         audio='غير مقيس',
         notes='bāʔi. تعارض مع أشكال corpus في Q (AlbAqy — بيئة كاذبة).'),
    dict(fam='يقوم', cat='everyday_verb', pron='hamza', conf='high',
         skels=['yqwm', 'tqwm', 'qAm'], word='يقوم', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='yiʔūm.'),
    dict(fam='قاتل', cat='everyday', pron='hamza', conf='high',
         skels=['qAtlp'], word='قاتلة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~20 موضعًا'],
         audio='غير مقيس', notes='ʔātila.'),
    dict(fam='قاسي', cat='everyday', pron='hamza', conf='high',
         skels=['qAsy', 'qAsyp'], word='قاسي', ev=[
             '[wiki] ق→ʔ', '[corpus] ~30 موضعًا'],
         audio='غير مقيس', notes='ʔāsī.'),
    dict(fam='النقد/نقد', cat='everyday', pron='hamza', conf='high',
         skels=['nqd', 'Alnqd'], word='نقد', ev=[
             '[wiki] ق→ʔ', '[corpus] ~15 موضعًا'],
         audio='غير مقيس',
         notes='naʔd (نقد نقدي/جحش!). النقد الفني الرسمي naqd q — سياقي '
               'خفيف؛ الافتراضي ء.'),
    dict(fam='نطاق', cat='formal', pron='hamza', conf='high',
         skels=['nTAq'], word='نطاق', ev=[
             '[wiki] ق→ʔ', '[corpus] 15 موضعًا'],
         audio='غير مقيس', notes='niṭāʔ.'),
    dict(fam='يقطر/قطرة', cat='everyday', pron='hamza', conf='high',
         skels=['qTrp', 'qTrAt'], word='قطرة', ev=[
             '[wiki] ق→ʔ', '[corpus] في الذيل'],
         audio='غير مقيس', notes='ʔaṭra — مقابل قطر الدولة/القياس (مدخل '
                                  'مستقل أدناه).'),
    dict(fam='قامة', cat='everyday', pron='hamza', conf='high',
         skels=['qAmp'], word='قامة', ev=[
             '[wiki] ق→ʔ', '[corpus] في الذيل'],
         audio='غير مقيس', notes='ʔāma (طول القامة).'),
    dict(fam='عملاق/خارق', cat='everyday', pron='hamza', conf='high',
         skels=['EmlAqp', 'EmlAq', 'xArqp'], word='عملاقة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~15 موضعًا'],
         audio='غير مقيس', notes='ʕamlaʔa / ẖāriʔa.'),
    dict(fam='مرهق', cat='everyday', pron='hamza', conf='high',
         skels=['mrhq', 'mrhqp'], word='مرهق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~20 موضعًا'],
         audio='غير مقيس', notes='murhiʔ.'),
    dict(fam='مقنع', cat='everyday', pron='hamza', conf='high',
         skels=['mqnE'], word='مقنع', ev=[
             '[wiki] ق→ʔ', '[corpus] 10 مواضع'],
         audio='غير مقيس', notes='muʔniʕ.'),
    dict(fam='لقطة', cat='everyday_media', pron='hamza', conf='high',
         skels=['lqTp'], word='لقطة', ev=[
             '[wiki] ق→ʔ', '[corpus] ~25 موضعًا'],
         audio='غير مقيس', notes='laʔṭa (لقطة تلفزيونية — corpus!).'),
    dict(fam='قبل', cat='everyday_time', pron='hamza', conf='high',
         skels=['qbl', 'qblhA', 'qblhm', 'qblhmA', 'qblk', 'qblnA'],
         word='قبل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~450 موضعًا (قبل + متفرقات)'],
         audio='غير مقيس', notes='ʔabl / ʔablamā («قبل ما» شائعة).'),
    dict(fam='قف', cat='everyday_verb', pron='hamza', conf='high',
         skels=['qf', 'yqf', 'nqf', 'tqf', 'btqf', 'htqf'], word='يقف', ev=[
             '[wiki] ق→ʔ', '[corpus] ~70 موضعًا'],
         audio='غير مقيس', notes='yiʔif / ʔuff!'),
    dict(fam='توقع', cat='everyday_verb', pron='hamza', conf='high',
         skels=['twqE', 'mtwqE', 'mtwqEp', 'twqEAt', 'AltwqEAt'], word='متوقع', ev=[
             '[wiki] ق→ʔ', '[corpus] ~55 موضعًا'],
         audio='غير مقيس', notes='yitawaʔʔaʕ — مقابل موقع tech (مستقل).'),
    dict(fam='اقترح', cat='everyday_verb', pron='hamza', conf='high',
         skels=['tqtrH', 'yqtrH', 'AqtrAH', 'tqtryH', 'mqtRH', 'mqtRHA'],
         word='اقترح', ev=[
             '[wiki] ق→ʔ', '[corpus] ~90 موضعًا'],
         audio='غير مقيس', notes='iʔtaraħ / tiʔtiraħ.'),
    dict(fam='تقيل', cat='everyday_adj', pron='hamza', conf='high',
         skels=['tqyl', 'tqylp'], word='تقيل (ثقيل)', ev=[
             '[wiki] ق→ʔ', '[corpus] 24 موضعًا'],
         audio='غير مقيس', notes='taʔīl — الصفة المصرية لثقيل.'),
    dict(fam='طقس', cat='everyday_weather', pron='hamza', conf='high',
         skels=['Tqs'], word='طقس', ev=[
             '[wiki] ق→ʔ', '[corpus] 22 موضعًا'],
         audio='غير مقيس', notes='ṭaʔs.'),
    dict(fam='حرق', cat='everyday_verb', pron='hamza', conf='high',
         skels=['Hrq', 'yHrq', 'tHrq', 'mHrwq'], word='حرق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~25 موضعًا'],
         audio='غير مقيس', notes='ħaraʔ / maħrūʔ.'),
    dict(fam='اتفق', cat='everyday_verb', pron='hamza', conf='high',
         skels=['AtfAq', 'AtfqnA', 'ytfAq', 'tfAq', 'mtfq', 'mtfqAl'], word='اتفق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='ittafaʔ (اِتّفقنا!).'),
    dict(fam='تقدير/تقديم/تقييم', cat='formal', pron='hamza', conf='medium_high',
         skels=['tqdyr', 'tqdym', 'tqyym', 'Altqdyr', 'Altqdym', 'Altqyym',
                'yqym', 'ytqym'], word='تقدير', ev=[
             '[wiki] ق→ʔ', '[corpus] ~100 موضعًا',
             '[ling] taʔdīr/taʔdīm/taʔyīm حواريًا؛ الرسمي الإداري q'],
         audio='غير مقيس',
         notes='تعارض مع قائمة Q (tqdym/tqdyr/tqdyry من أمثلة المستخدم '
               'القديمة) — يُعرض للمراجعة.'),
    dict(fam='قصد', cat='everyday_verb', pron='hamza', conf='high',
         skels=['qSd', 'tqSd', 'yqSd', 'mqSwd', 'qSAyd'], word='قصد', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='ʔuṣd / tiʔṣid (قصدي!).'),
    dict(fam='قتل', cat='everyday_verb', pron='hamza', conf='high',
         skels=['qtl', 'yqtl', 'tqtl', 'btqtl', 'mqtwl', 'qAtlp'],
         word='يقتل', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='yiʔtol / maʔtūl.'),
    dict(fam='فقد', cat='everyday_verb', pron='hamza', conf='high',
         skels=['fqd', 'byfqd', 'yfqd', 'fqdAn', 'mfqwd'], word='يفقد', ev=[
             '[wiki] ق→ʔ', '[corpus] ~40 موضعًا'],
         audio='غير مقيس', notes='yifʔad / fuqdān.'),
    dict(fam='سقف', cat='everyday_home', pron='hamza', conf='high',
         skels=['sqf', 'Asqf'], word='سقف', ev=[
             '[wiki] ق→ʔ', '[corpus] ~15 موضعًا'],
         audio='غير مقيس', notes='saʔf.'),
    dict(fam='يقين', cat='formal', pron='hamza', conf='high',
         skels=['yqyn', 'Alyqyn'], word='اليقين', ev=[
             '[wiki] ق→ʔ', '[corpus] ~15 موضعًا'],
         audio='غير مقيس', notes='yiʔīn — كلمة دينية-فكرية تُقال همزة في '
                                 'الكلام المصري المتصل.'),
    dict(fam='قابلة', cat='everyday_persons', pron='hamza', conf='high',
         skels=['qAblp'], word='قابلة (الداية)', ev=[
             '[wiki] ق→ʔ', '[corpus] 13 موضعًا'],
         audio='غير مقيس', notes='ʔabla (الحاجة القابلة!).'),
    dict(fam='وقاية', cat='medical_everyday', pron='hamza', conf='high',
         skels=['wqAyp', 'wqA}yp'], word='وقاية', ev=[
             '[wiki] ق→ʔ', '[corpus] 13 موضعًا'],
         audio='غير مقيس', notes='waʔāya (واقي الشمس!).'),
    dict(fam='تلقائي', cat='tech_formal', pron='hamza', conf='medium_high',
         skels=['tlqA}y'], word='تلقائي', ev=[
             '[wiki] ق→ʔ', '[corpus] 12 موضعًا',
             '[ling] tilʔāʔi حواريًا / tilqāʔi رسميًا تقنيًا'],
         audio='غير مقيس', notes='مراجعة خفيفة إذا رغبت في السجل التقني q.'),
    dict(fam='قاطع', cat='everyday', pron='hamza', conf='high',
         skels=['qATE'], word='قاطع', ev=[
             '[wiki] ق→ʔ', '[corpus] 11 موضعًا',
             '[ling] ʔāṭiʕ — مشتقة فاعل من جذر قطع لكن نطقها العامي همزة '
             '(تجاوز صريح لافتراض العائلة: لا يُفترض المشتق — يُتحقق)'],
         audio='غير مقيس',
         notes='مثال مباشر لطلب المستخدم: «لا تفترض تلقائيًا أن كل مشتق له '
               'نفس النطق».'),
    dict(fam='وقود', cat='everyday_energy', pron='hamza', conf='high',
         skels=['wqwd'], word='وقود', ev=[
             '[wiki] ق→ʔ', '[corpus] 10 مواضع'],
         audio='غير مقيس',
         notes='wuʔūd — تعارض مع أشكال corpus في Q (وَقُودْ — بيئة كاذبة).'),
    dict(fam='قلب (مشتقات)', cat='everyday_verb', pron='hamza', conf='high',
         skels=['yqlb', 'tqlbAt', 'mqAlbp'], word='يقلب', ev=[
             '[wiki] ق→ʔ', '[corpus] ~30 موضعًا'],
         audio='غير مقيس', notes='yiʔlib / taʔallub (تقلبات).'),
    dict(fam='تقبل', cat='formal_verb', pron='hamza', conf='high',
         skels=['tqbl', 'mqbwl'], word='تقبل/مقبول', ev=[
             '[wiki] ق→ʔ', '[corpus] ~25 موضعًا'],
         audio='غير مقيس', notes='taʔabbol / maʔbūl.'),
    dict(fam='قيد', cat='everyday_objects', pron='hamza', conf='high',
         skels=['qywd', 'qyd', 'mqyd', 'mqyyd'], word='قيود', ev=[
             '[wiki] ق→ʔ', '[corpus] ~12 موضعًا'],
         audio='غير مقيس', notes='ʔiyūd.'),
    dict(fam='قلعة', cat='everyday_places', pron='hamza', conf='high',
         skels=['qlEp', 'qlE'], word='قلعة', ev=[
             '[wiki] ق→ʔ', '[corpus] 8 مواضع'],
         audio='غير مقيس', notes='ʔalaʕa.'),
    dict(fam='حقن', cat='medical', pron='hamza', conf='high',
         skels=['Hqnp', 'Hqn', 'yHqn'], word='حقنة', ev=[
             '[wiki] ق→ʔ', '[corpus] 8 مواضع'],
         audio='غير مقيس', notes='ħaʔna.'),
    dict(fam='أقصى', cat='everyday', pron='hamza', conf='high',
         skels=['>qSY', '>qSYA'], word='أقصى', ev=[
             '[wiki] ق→ʔ', '[corpus] ~20 موضعًا'],
         audio='غير مقيس', notes='ʔaʔṣā (لأقصى حد!).'),
    dict(fam='استقبل', cat='everyday_verb', pron='hamza', conf='high',
         skels=['AstqbAl', 'ystbEl', 'mstqbl'], word='استقبال', ev=[
             '[wiki] ق→ʔ', '[corpus] ~50 موضعًا'],
         audio='غير مقيس', notes='istibʔāl (استقبلنا الضيوف!).'),
    dict(fam='متقطع', cat='everyday', pron='hamza', conf='high',
         skels=['mtqTE'], word='متقطع', ev=[
             '[wiki] ق→ʔ', '[corpus] 7 مواضع',
             '[ling] mutaʔaṭṭiʕ — مشتقة من جذر قطع (q في قطعة) لكن '
             'نطقها العامي همزة (لا يُفترض المشتق)'],
         audio='غير مقيس', notes='مثال ثانٍ لانفصال المشتقات عن العائلة.'),
    dict(fam='إقامة', cat='everyday_events', pron='hamza', conf='high',
         skels=['<qAmp', 'qAmp'], word='إقامة', ev=[
             '[wiki] ق→ʔ', '[corpus] 7 مواضع'],
         audio='غير مقيس', notes='iʔāma (إقامة الحفلات!).'),
    dict(fam='سياق', cat='formal', pron='hamza', conf='high',
         skels=['syAq', 'AlsyAq'], word='سياق', ev=[
             '[wiki] ق→ʔ', '[corpus] 7 مواضع'],
         audio='غير مقيس', notes='siyāʔ — كلمة أكاديمية تُقال همزة.'),
    dict(fam='أزرق', cat='everyday_color', pron='hamza', conf='high',
         skels=['>zrq', 'Al>zrq', 'zrq'], word='أزرق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~10 مواضع'],
         audio='غير مقيس', notes='ʔazraʔ (الجيش الأزرق!).'),
    dict(fam='شائق', cat='formal', pron='hamza', conf='medium',
         skels=['$yq'], word='شائق', ev=[
             '[wiki] ق→ʔ', '[corpus] 7 مواضع'],
         audio='غير مقيس', notes='šāʔiʔ.'),
    dict(fam='قبل... ذيل عام', cat='tail_rule', pron='hamza',
         conf='low',
         skels=[],
         word='(كل هيكل غير مغطى أعلاه)', ev=[
             '[wiki] الانعكاس القاهري غير المعلم: ق→ʔ هو الافتراضي '
             'غير المعلم في القاهرة والدلتا (corpus القاهري تلفزيوني)',
             '[ling] كل كلمة قاف جديدة في عامية القاهرة تُنطق ء إلا إن '
             'ثبت خلافها'],
         audio='—',
         notes='قاعدة الذيل: تصنيف hamza_default بثقة منخفضة + علم مراجعة '
               'بشرية لكل هيكل بتكرار >= 5 لم يُغطَّ صراحةً.'),

    # ================= CONTEXT_DEPENDENT =================
    dict(fam='قوي', cat='homograph', pron='context', conf='high',
         skels=['qwy', 'qwyp', '>qwY'], word='قوي', ev=[
             '[wiki] قوى /ˈʔæwi/ [ˈqɑwi] homograph موثق: ظرف «جدًّا» همزة '
             'دائمًا مقابل «strong» فصحى q',
             '[base] مستبعد عمدًا من قائمة Q لهذا السبب',
             '[corpus] 647+139 موضعًا (غالبها الظرف العامي)'],
         audio='غير مقيس (الظرف ء قطعًا)',
         notes='لا يجوز وضعها في قائمة واحدة (طلب المستخدم نفسه). الظرف '
               ' intensifier «قوي/قوية» = ء؛ الصفة الفصحى strong = q في '
               'السجل الرسمي فقط.'),
    dict(fam='قرار', cat='media_political', pron='context', conf='medium',
         skels=['qrAr', 'qrArAt', 'qrr', 'yqrr'], word='قرار', ev=[
             '[wiki] ق→ʔ افتراضي', '[corpus] ~280 موضعًا (كلمة إخبارية '
             'كبيرة في corpus تلفزيوني)',
             '[ling] الإخباري المتأني qarār q / الحوار المصري iʔrār ء'],
         audio='غير مقيس',
         notes='corpus يحوي كلا السجلين (أخبار + دراما) — الحسم يتطلب '
               'سماعًا موضعيًا أو سياسة سجل.'),
    dict(fam='اقتصاد', cat='media_economic', pron='context', conf='medium',
         skels=['AqtSAd', 'AqtSAdy', 'AqtSAdyp'], word='اقتصاد', ev=[
             '[wiki] ق→ʔ افتراضي',
             '[corpus] ~160 موضعًا (iqtiẓād إخباري / iʔtiẓād حواري)'],
         audio='غير مقيس',
         notes='مثل قرار — سجل إخباري مقابل عامية.'),
    dict(fam='واقع', cat='media', pron='context', conf='medium',
         skels=['wAqE', 'wAqEy', 'wAqEyp'], word='واقع', ev=[
             '[wiki] ق→ʔ', '[corpus] ~160 موضعًا',
             '[ling] il-wāqiʕ حواريًا / al-wāqiʕ إخباريًا'],
         audio='غير مقيس', notes='سجلان في corpus واحد.'),
    dict(fam='تحقيق', cat='media_legal', pron='context', conf='medium',
         skels=['tHqyq'], word='تحقيق', ev=[
             '[wiki] ق→ʔ', '[corpus] ~55 موضعًا'],
         audio='غير مقيس',
         notes='taħʔīq حواريًا (وتحقيق الشرطة) / taħqīq إخباريًا.'),
    dict(fam='استقرار', cat='media_political', pron='context', conf='medium',
         skels=['AstqrAr', 'mstqr', 'mstqrp'], word='استقرار', ev=[
             '[wiki] ق→ʔ', '[corpus] ~90 موضعًا'],
         audio='غير مقيس', notes='isteʔrār / istiqrār — إخباري مقابل عامية.'),
    dict(fam='قيادة', cat='military_formal', pron='context', conf='medium',
         skels=['qyAdp'], word='قيادة', ev=[
             '[wiki] ق→ʔ', '[corpus] 18 موضعًا'],
         audio='غير مقيس', notes='il-ʔiyāda / al-qiyāda — الرسمي العسكري q.'),
    dict(fam='حقوق', cat='legal_formal', pron='context', conf='medium',
         skels=['Hqwq', 'Hqwqy'], word='حقوق', ev=[
             '[base] tier1 — زرع auto ساري (تعارض يُعرض)',
             '[corpus] 6 مواضع فقط',
             '[ling] ħuʔūʔ il-ʔinsān عاميًا / ħuqūq إخباريًا-قانونيًا'],
         audio='غير مقيس',
         notes='تعارض مع baseline (tier1→q في auto)؛ التكرار المنخفض يسهل '
               'المراجعة اليدوية.'),
    dict(fam='قياس', cat='education_math', pron='context', conf='medium',
         skels=['qyAs', 'yqys', 'mqyAs'], word='قياس', ev=[
             '[base] قائمة Q + tier1 (qyAs)',
             '[corpus] ~20 موضعًا',
             '[ling] ʔiyās عاميًا / qiyās في الشرح التعليمي'],
         audio='غير مقيس', notes='مصطلح منهجي (توجيه المستخدم القديم) مقابل '
                                 'عامية.'),
    dict(fam='قسم', cat='education_math_verb', pron='context', conf='medium',
         skels=['qsm', 'yqsm', 'tqsm', 'tqsym', 'ynqsm', 'nqsm', 'byqsm'],
         word='قسم', ev=[
             '[base] QAF_Q_SENTENCE_SKIP موثق: ملتبسة بأفعال — catt يعطي '
             'القراءة السياقية',
             '[corpus] ~20 موضعًا',
             '[ling] ʔasm/yiʔsim عاميًا / qism قسم المدرسة وqasma القسمة '
             'في الشرح'],
         audio='غير مقيس',
         notes='ملتبسة بفعل قَسَمَ — لا قاعدة واحدة بلا سياق (موثق أصلًا '
               'في baseline).'),
    dict(fam='قدرة', cat='science_formal', pron='context', conf='medium',
         skels=['qdrp', 'qdrAt', 'qAdrp', 'qdrt', 'qdrth', 'qdrthA',
                'qdrthm', 'qAdr', 'qAdryn'], word='قدرة', ev=[
             '[base] study forms + tier1 (qdrAt) — تعارض يُعرض',
             '[corpus] ~180 موضعًا',
             '[ling] ʔidra عاميًا / qudrāt في العلوم والإعلام'],
         audio='غير مقيس', notes='مثل قياس — سجل تعليمي/إخباري مقابل عامية.'),
    dict(fam='قمة', cat='media_political', pron='context', conf='medium',
         skels=['qmp'], word='قمة', ev=[
             '[corpus] ~26 موضعًا', '[ling] ʔimmit ṭūr عامية / qimma '
             'القمة العربية إخباريًا'],
         audio='غير مقيس', notes='إخباري سياسي صرف.'),
    dict(fam='إطلاق', cat='media_formal', pron='context', conf='medium',
         skels=['TlAq', '<TlAq', 'mTlq', 'mTlqp'], word='إطلاق', ev=[
             '[corpus] ~55 موضعًا (مطلقة الطلاق ء شائعة)',
             '[ling] ʔiṭlāʔ الطلاق عاميًا / iṭlāq إطلاق القمر الصناعي '
             'إخباريًا q'],
         audio='غير مقيس',
         notes='انفصال دلالي: مطلقة (طلاق) ء ؛ إطلاق سراح/صاروخ q رسمي.'),
    dict(fam='موقع', cat='tech', pron='context', conf='medium',
         skels=['mwqE', 'mwAqE'], word='موقع', ev=[
             '[corpus] ~35 موضعًا',
             '[ling] il-mawʔiʕ حواريًا / mawqiʕ إلكتروني تقنيًا'],
         audio='غير مقيس', notes='سجل تقني متعلم مقابل عامية.'),
    dict(fam='توقيت', cat='tech_formal', pron='context', conf='medium',
         skels=['twqyt'], word='توقيت', ev=[
             '[base] tier1 — زرع auto (تعارض يُعرض)', '[corpus] 40 موضعًا',
             '[ling] tawʔīt عاميًا / tawqīt الرسمي (توقيت غرينتش)'],
         audio='غير مقيس', notes='مصطلح إعلامي/تقني.'),
    dict(fam='انتقاد/نقد فني', cat='media_cultural', pron='context',
         conf='medium',
         skels=['ntqAd', 'yntqd'], word='انتقاد', ev=[
             '[corpus] 2 مواضع (نقد) + صفر (انتقاد)',
             '[ling] intiʔād حواريًا / intiqād نقديًا'],
         audio='غير مقيس', notes='قلة المواضع تسهل الحسم اليدوي.'),
    dict(fam='مقاومة', cat='historical_media', pron='context', conf='medium',
         skels=['mqAwmp', 'mqAwmAt', 'mqAwm'], word='مقاومة', ev=[
             '[talkin] في القائمة المصرية (المناعة/المقاومة)',
             '[corpus] ~12 موضعًا',
             '[ling] muʔāwima عاميًا (المناعة) / muqāwama تاريخيًا-إعلاميًا'],
         audio='غير مقيس', notes='سجل تاريخي/إعلامي مقابل عامية.'),
    dict(fam='القاهرة', cat='place_name', pron='context', conf='medium',
         skels=['qAhrp'], word='القاهرة', ev=[
             '[corpus] 32 موضعًا',
             '[ling] المصريون يقولون «مصر» أصلًا؛ عند التسمية: il-gāhira '
             'قاهري شائع / il-ʔāhira قراءة متأنية / al-Qāhira رسمي',
             '[base] في قائمة Q (qAhrp — talkin) بينما تعليق infer.py '
             'القديم يذكرها ضمن B g — تذبذب baseline نفسه'],
         audio='غير مقيس',
         notes='ثلاثة نطقات موثقة لثلاثة سجلات — نموذج context_dependent '
               'الكامل.'),
    dict(fam='قناة', cat='media_tech', pron='context', conf='low',
         skels=['qnAp', 'qnwAt'], word='قناة', ev=[
             '[base] قائمة Q (qnAp/قنوات)',
             '[corpus] 11 موضعًا',
             '[ling] تذبذب حقيقي غير محسوم: ʔanāya حواريًا / ganāya '
             'قاهريًا تقليديًا / qanāh رسميًا — الشواهد متضاربة'],
         audio='غير مقيس',
         notes='ضعيف الثقة عن قصد — مرشح أول للمراجعة البشرية (طلب '
               'المستخدم: لا إجبار بلا دليل).'),
    dict(fam='القيامة', cat='religious_eschatology', pron='context',
         conf='medium',
         skels=['qyAmp', 'qyAm'], word='القيامة', ev=[
             '[base] study form موجود لكن خارج DEEP_OK عمدًا: «تجربتها '
             'الحية أنتجت همزة عند المستخدم» (PATCH 14)',
             '[corpus] قيام موضع واحد',
             '[ling] yōm il-ʔiyāma عاميًا / yawm al-qiyāma دينيًا'],
         audio='hamza_produced (تجربة حية موثقة PATCH 14)',
         notes='الزرع فشل سماعيًا — لا تُدرج q مؤكدة (نفس طلب المستخدم: '
               'uncertain لا يُفرض).'),
    dict(fam='القادر', cat='religious_names', pron='context', conf='medium',
         skels=['qAdr'], word='القادر', ev=[
             '[ling] اسم الله القادر q في الذكر الديني / «قادر على» '
             'العامية ʔāder ء'],
         audio='غير مقيس', notes='ملتبسة بالصفة الشائعة قديم qAdr — مدخل '
                                 'قادر اليومي hamza أعلاه مستقل.'),
    dict(fam='قبلة', cat='religious_meaning_split', pron='context',
         conf='medium',
         skels=['qblp'], word='قبلة', ev=[
             '[ling] انفصال دلالي موثق: قِبلة الصلاة qibla q في الحديث '
             'الديني / قبلة القُبلة ʔubla ء (قبلة الزواج!)'],
         audio='غير مقيس', notes='نموذج تعارض المعنى الذي طلب المستخدم '
                                 'التقاطه.'),
    dict(fam='قمر', cat='poetic', pron='context', conf='medium',
         skels=['qmr'], word='قمر', ev=[
             '[corpus] موضع واحد',
             '[ling] il-ʔamar عامية (يا قمر النداء الشعبي ء غالبًا) / '
             'qamar الغناء الكلاسيكي والأدب q'],
         audio='غير مقيس',
         notes='نداء «يا قمر» الغنائي يسمع q كثيرًا — سياقي.'),
    dict(fam='قطر', cat='place_math', pron='context', conf='medium',
         skels=['qTr'], word='قطر', ev=[
             '[base] tier2 (خُفِّض من tier1): الزرع قاس ء مرتين؛ '
             'بالتنوين q? (VOT=46ms)',
             '[corpus] 10 مواضع',
             '[ling] قطر الدولة Gaṭar قاهريًا شائع / Qaṭar إخباريًا / '
             'قطر الهندسة qiyāṭ? — لا، quṭr تعليميًا q'],
         audio='mixed_run1 (ء مرتين بلا تنوين؛ q? بالتنوين)',
         notes='تعارض صوتي موثق — سياقي بلا حسم.'),
    dict(fam='معقد/معقول', cat='formal', pron='context', conf='medium',
         skels=['mEqd', 'mEqdp', 'mEqwl', 'mEqwlp'], word='معقد', ev=[
             '[talkin] معقد/تعقيد في القائمة',
             '[corpus] ~90 موضعًا',
             '[ling] mʕaʔʔad حواريًا / muʕqad وصفًا أكاديميًا'],
         audio='غير مقيس', notes='سجلان.'),
    dict(fam='أعتقد', cat='formal_verb', pron='context', conf='medium',
         skels=['AEtqd', '>Etqd'], word='أعتقد', ev=[
             '[talkin] في القائمة', '[corpus] 50 موضعًا',
             '[ling] aʕtaʔid حواريًا / aʕtaqid حوارًا رسميًا'],
         audio='غير مقيس', notes='فعل الرأي — شائع في السجلين.'),
    dict(fam='عبقري', cat='formal', pron='context', conf='medium',
         skels=['Ebqry', 'Ebqryp', 'EbqAry'], word='عبقري', ev=[
             '[talkin] في القائمة', '[corpus] ~120 موضعًا',
             '[ling] ʕabʔari حواريًا / ʕabqarī تمجيدًا رسميًا'],
         audio='غير مقيس', notes='كلمة المدح — سجلان.'),
    dict(fam='رقمية/رقمي', cat='tech', pron='context', conf='low',
         skels=['rqmyp', 'rqmy', 'rqmyyp'], word='رقمية', ev=[
             '[corpus] ~40 موضعًا',
             '[ling] التحول الرقمي إعلاميًا raqmiyy q / بالميراث الصرفي '
             'ragmiyy g — كلاهما مسموع'],
         audio='غير مقيس',
         notes='مشتقة من عائلة g لكن السجل التقني يستعير ق الفصحى — لا '
               'افتراض اشتقاقي (طلب المستخدم).'),
    dict(fam='نطق', cat='linguistic_formal', pron='context', conf='medium',
         skels=['nTq', 'nATq'], word='نطق', ev=[
             '[corpus] 15+ مواضع (نطاق/ينطق)',
             '[ling] naṭʔ حواريًا / naṭq لغويًا وقرآنيًا (مواضع النطق '
             'علم التجويد q)'],
         audio='غير مقيس', notes='مصطلح قرائي/لغوي.'),
    dict(fam='تقنية', cat='tech', pron='context', conf='medium',
         skels=['tqnyp', 'tqnyAt'], word='تقنية', ev=[
             '[corpus] ~35 موضعًا',
             '[ling] it-taqniyya الإعلام التقني q شائع / tiʔniyya '
             'حواريًا'],
         audio='غير مقيس', notes='سجل تقني.'),
    dict(fam='توقيع', cat='formal', pron='context', conf='medium',
         skels=['twqyE'], word='توقيع', ev=[
             '[ling] tawʔīʕ يوميًا / tawqīʕ رسميًا'],
         audio='غير مقيس', notes='ذيل منخفض في corpus.'),
    dict(fam='توثيق', cat='formal', pron='context', conf='medium',
         skels=['twthq'], word='توثيق', ev=[
             '[ling] tawθīʔ يوميًا / tawthīq رسميًا'],
         audio='غير مقيس', notes='نادر في corpus.'),
    dict(fam='القدس', cat='religious_places', pron='context', conf='medium',
         skels=['qds'], word='القدس', ev=[
             '[ling] il-Quds في السجل الديني/الإخباري (القدس الشريف) / '
             'بيت il-muʔaddas عاميًا'],
         audio='غير مقيس', notes='سياسة إخبارية دينية — مراجعة.'),

    # ================= UNCERTAIN =================
    dict(fam='قنا (المدينة)', cat='place_name', pron='uncertain',
         conf='low', skels=['qnA'], word='قنا', ev=[
             '[ling] تذكر مصادر شفهية Gaʿāna/Gana قاهريًا لكن لم يُعثر '
             'على مصدر مكتوب محسم؛ ولها معنى محافظة في السجل الرسمي Qanā '
             'q'],
         audio='غير مقيس', notes='صفر مواضع corpus — يُحسم بمراجعة بشرية.'),
    dict(fam='قليوب', cat='place_name', pron='uncertain', conf='low',
         skels=['qlywb'], word='قليوب', ev=[
             '[ling] تذبذب غير موثق (ʔalūb/Galyūb)'],
         audio='غير مقيس', notes='صفر مواضع corpus.'),
    dict(fam='القناطر', cat='place_name', pron='uncertain', conf='low',
         skels=['qnTrp', 'qnATr'], word='القناطر', ev=[
             '[ling] قياسًا على قنطار g لكن بلا شاهد مباشر'],
         audio='غير مقيس', notes='صفر مواضع corpus.'),
    dict(fam='قفش', cat='slang', pron='uncertain', conf='low',
         skels=['qf$', 'qf$p'], word='قفشة', ev=[
             '[ling] عامية قاهرية حديثة — g أو ء بلا توثيق'],
         audio='غير مقيس', notes='ذيل corpus.'),
    dict(fam='أسماء أعلام قافية', cat='personal_names', pron='uncertain',
         conf='low', skels=['qAsm', 'qtbp', 'qndyl', 'yAqwt'],
         word='قاسم/قتيبة/قنديل/ياقوت', ev=[
             '[ling] أسماء الأعلام تتقلب بين القراءات (قاسم ʔāsem غالبًا؛ '
             'قنديل ʔandīl غالبًا) — لا قاعدة واحدة'],
         audio='غير مقيس',
         notes='تُحسم فرديًا في المراجعة البشرية أو بعلامة {ق}/{ء}/{ج}.'),
    dict(fam='قرطبة/قرطاج', cat='historic_places', pron='uncertain',
         conf='low', skels=['qrTbp', 'qrTAj'], word='قرطبة', ev=[
             '[ling] مرجع تاريخي رسمي q في السجل الثقافي؛ لا وجود عامي '
             'فعلي'],
         audio='غير مقيس', notes='صفر مواضع corpus.'),
]
