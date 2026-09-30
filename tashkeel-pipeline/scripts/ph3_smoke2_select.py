# -*- coding: utf-8 -*-
"""بناء مدخل عينة الدخان الرسمية (الدليل v2.1) — 14 وحدة.

12 وحدة قاعدية تغطي عائلات القرارات الثمانية + وحدة تصحيح تفريغ
(1_00768_01 صالة الحديد — اختبار آلية §ك-5) + وحدة حجر (0_04186_00 مص —
اختبار تخطي §ل-5 بلا LLM).
"""
import csv
import json

UNITS = [
    # (id, ما تختبره)
    ('0_05699_00', 'مرساة قديمة — الوحدة الفاشلة في الدخان الداخلي (كثافة 1.21) — إعادة اختبار'),
    ('1_06985_02', 'مرساة قديمة — فاصلتان عربيتان ، + محوّل ترقيم'),
    ('0_04717_01', 'مرساة قديمة — إن/اوي/همزة وصل'),
    ('1_05209_00', '؟ عربية (تحويل) + يا رب (Tier A) + اوي'),
    ('1_02373_00', 'والله (Tier A) + ربنا يكرموا → رَبِّنَا (§ك-4)'),
    ('0_01276_00', 'ربنا يكرم → رَبِّنَا يِكْرِمْ (المنقولة من Tier A)'),
    ('1_00955_01', 'بربنا → بِرَبِّنَا + دين = دِين (مفهومًا) + انتشارا (تنوين) وقبولا (بلا تنوين — خارج القائمة)'),
    ('1_02614_01', 'دين = دَيْن ماليًا (مستبعدات)'),
    ('0_08127_01', 'لا قدر الله (Tier A جديدة) + ، ASCII + ؟ (eval)'),
    ('0_06079_00', 'لقدر الله، (Tier A جديدة) + تماماً، (تنوين + فاصلة) + ؟'),
    ('0_03547_01', 'بركة البيت → بَرَكَة (مستبعدات) + أرقام 2024 + جداً'),
    ('0_01179_00', 'رمضان → رَمَضَانْ بالقواعد العامة (§ك)'),
    ('1_00768_01', 'آلية تصحيح التفريغ: صلاة → صالة الحديد (§ك-5)'),
    ('0_04186_00', 'آلية الحجر: مص (§ل-5) — بلا LLM إطلاقًا'),
]

rows = {r['utt']: r for r in csv.DictReader(
    open('/home/z/my-project/work/prep_output/extraction.csv', encoding='utf-8'))}

out = []
notes = {}
for uid, note in UNITS:
    r = rows[uid]
    out.append({'id': uid, 'text': r['transcript'],
                'n_tokens_recorded': int(r['n_tokens']),
                'split': r['split']})
    notes[uid] = note

path = '/home/z/my-project/work/ph3_smoke2_input.json'
json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(notes, open('/home/z/my-project/work/ph3_smoke2_notes.json', 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{len(out)} units written to {path}')
for u, n in UNITS:
    print(f'  {u} [{rows[u]["split"]}] {n[:60]}')
