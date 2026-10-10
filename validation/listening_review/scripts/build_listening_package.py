#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_listening_package.py — حزمة الاستماع المعمّاة.

لكل زوج (S1-S8): تعيين عشوائي (بذرة مسجلة) أي الملفين A وأيهما B،
نسخ باسمين مجهولين L0n_A/L0n_B، وكتابة:
- README البروتوكول (بلا مفتاح)
- نموذج التقييم CSV (بلا تلميحات تكشف الجهة)
- المفتاح في ملف منفصل لا يُسلَّم للمستمع (work/listening/)
"""
import csv
import json
import random
import shutil

PAIRS = [
    ('L01', 'S1', 'بِيَكْتِب الدرس'),
    ('L02', 'S2', 'هو بيقول للطالب'),
    ('L03', 'S3', 'دلوقتي هنبدأ الدرس يا عمر'),
    ('L04', 'S4', 'أنا مش فاهم السؤال ده'),
    ('L05', 'S5', 'بكرة هنعمل تجربة جديدة'),
    ('L06', 'S6', 'هو بيقول كده'),
    ('L07', 'S7', 'مش عارف أعمل إيه'),
    ('L08', 'S8', 'هو هنا'),
]
MAIN_DIR = '/home/z/my-project/work/listening/main'
FIX_DIR = '/home/z/my-project/work/listening/fix'
PKG = '/home/z/my-project/download/listening_review'
KEY_PATH = '/home/z/my-project/work/listening/blinding_key.json'

SEED = 20261010  # بذرة مسجلة لإعادة إنتاج التعيين — المفتاح لا يُنشر قبل المراجعة


def main():
    rng = random.Random(SEED)
    key = {'seed': SEED, 'mapping': {}}
    for lid, sid, sent in PAIRS:
        fix_is_a = rng.random() < 0.5
        main_wav = f'{MAIN_DIR}/{sid}.wav'
        fix_wav = f'{FIX_DIR}/{sid}.wav'
        a_src, b_src = (fix_wav, main_wav) if fix_is_a else (main_wav, fix_wav)
        shutil.copyfile(a_src, f'{PKG}/{lid}_A.wav')
        shutil.copyfile(b_src, f'{PKG}/{lid}_B.wav')
        key['mapping'][lid] = {
            'sentence': sent, 'source_id': sid,
            'A_is': 'fix' if fix_is_a else 'main',
            'B_is': 'main' if fix_is_a else 'fix',
        }
        print(f'{lid}: A={"fix" if fix_is_a else "main":4s} | {sent}')

    with open(KEY_PATH, 'w', encoding='utf-8') as f:
        json.dump(key, f, ensure_ascii=False, indent=1)

    # نموذج التقييم — أعمدة حسب مواصفة المهمة
    with open(f'{PKG}/evaluation_form.csv', 'w', encoding='utf-8-sig',
              newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'الحالة', 'الجملة المدخلة',
            'طبيعية النطق المصري A (1-5)', 'طبيعية النطق المصري B (1-5)',
            'صحة الحركات الداخلية A (1-5)', 'صحة الحركات الداخلية B (1-5)',
            'وضوح الكلمات A (1-5)', 'وضوح الكلمات B (1-5)',
            'سلامة نطق الكلمات العامية A (1-5)',
            'سلامة نطق الكلمات العامية B (1-5)',
            'أخطاء أو أصوات غريبة A (وصف/لا)',
            'أخطاء أو أصوات غريبة B (وصف/لا)',
            'التفضيل (A / B / لا فرق واضح)',
            'ملاحظات نوعية (اذكر الكلمات)',
        ])
        for lid, sid, sent in PAIRS:
            w.writerow([lid, sent] + [''] * 12)

    print(f'\nPackage: {PKG}')
    print(f'Key (لا يُسلَّم للمستمع): {KEY_PATH}')


if __name__ == '__main__':
    main()
