#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""listening_regression.py — مراجعة الانحدار على مستوى الكلمة.

لكل زوج (main/fix) ولكل كلمة (محاذاة موضعية):
- نص متطابق → توكنات متطابقة حتميًا (نفس دالة التوكن في النسختين —
  get_tokenizer متطابقة بايت-بايت، مُتحقق منها).
- نص مختلف → توكنات الكلمة في كل نسخة + فرق حقيقي.

يثبت: (1) لا إسقاط كلمات، (2) الكلمات غير المستهدفة لا تتغير نطقًا،
(3) كل تغيير نطق محصور في كلمات مستهدفة (قائمة det-v1.3 + هو) أو
استكمال عاري (bug الأصل)، (4) فروق ترتيب العلامات بلا أثر نطقي.
"""
import difflib
import json
import sys

INF = '/home/z/my-project/work/github_repo/eiqaz-tts/inference'
sys.path.insert(0, INF)
sys.path.insert(0, INF + '/lib')
import os
os.chdir(INF)
import infer  # noqa: E402

MAIN = json.load(open('/home/z/my-project/work/listening/main/gen_report.json',
                      encoding='utf-8'))
FIX = json.load(open('/home/z/my-project/work/listening/fix/gen_report.json',
                     encoding='utf-8'))
_, toks_egy, _ = infer.get_tokenizer()

# الكلمات المتوقعة تغيرها الإصلاح (مرجع det-v1.3 + قاعدة هو + استكمال العاري)
EXPECTED_TARGETS = {
    'هو', 'بيقول', 'بيكتب', 'دلوقتي', 'هنبدأ', 'هنا', 'بكرة',
    'الدرس',  # استكمال العاري (bug الأصل)
}
AR_DIAC = set(
    '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670'
    '\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DF\u06E0'
    '\u06E1\u06E2\u06E3\u06E4\u06E5\u06E6\u06E7\u06E8\u06EA'
    '\u06EB\u06EC\u06ED')


def skel(w):
    return ''.join(ch for ch in w if ch not in AR_DIAC)


def wtoks(w):
    return list(toks_egy(w))


def main():
    mb = {c['id']: c for c in MAIN['cases']}
    fb = {c['id']: c for c in FIX['cases']}
    results = []
    print('=' * 72)
    print('مراجعة الانحدار كلمةً كلمة (توكنات النطق لكل كلمة)')
    print('=' * 72)
    for cid in [f'S{i}' for i in range(1, 9)]:
        mws = mb[cid]['final_text'].split()
        fws = fb[cid]['final_text'].split()
        assert len(mws) == len(fws), f'{cid}: عدد كلمات مختلف!'
        rows = []
        for i, (mw, fw) in enumerate(zip(mws, fws)):
            tm, tf = wtoks(mw), wtoks(fw)
            same_text = mw == fw
            same_tok = tm == tf
            if same_text and same_tok:
                status = 'ثابتة'
            elif not same_text and same_tok:
                status = 'فرق-نصي-فقط'  # مثل ترتيب العلامات
            else:
                status = 'تغير-نطق'
            rows.append({'word': mw, 'word_fix': fw, 'status': status,
                         'tok_main': tm, 'tok_fix': tf})
        changed = [r for r in rows if r['status'] != 'ثابتة']
        n_pron = [r for r in rows if r['status'] == 'تغير-نطق']
        unexpected = [r for r in n_pron
                      if skel(r['word']) not in EXPECTED_TARGETS
                      and skel(r['word_fix'])
                      not in EXPECTED_TARGETS]
        results.append({
            'id': cid, 'input': mb[cid]['input'],
            'n_words': len(mws),
            'words_unchanged': len(rows) - len(changed),
            'word_rows': rows,
            'pronunciation_changed_words': [
                {'main': r['word'], 'fix': r['word_fix']}
                for r in n_pron],
            'text_only_diff_words': [
                {'main': r['word'], 'fix': r['word_fix']}
                for r in rows if r['status'] == 'فرق-نصي-فقط'],
            'unexpected_pron_changes': [
                {'main': r['word'], 'fix': r['word_fix']}
                for r in unexpected],
        })
        print(f"\n[{cid}] {mb[cid]['input']}  ({len(mws)} كلمة)")
        for r in rows:
            mark = {'ثابتة': '  =', 'فرق-نصي-فقط': ' ~',
                    'تغير-نطق': ' ✗'}[r['status']]
            print(f"  {mark} {r['word']:18s} → {r['word_fix']:18s}"
                  f"  [{' '.join(r['tok_main'])}]"
                  f" → [{' '.join(r['tok_fix'])}]")
        if unexpected:
            print(f"  ⚠ كلمات تغير نطقها خارج المجموعة المستهدفة:")
            for r in unexpected:
                print(f"      {r['word']} → {r['word_fix']}")
        else:
            print('  ✓ كل تغييرات النطق داخل المجموعة المستهدفة')

    with open('/home/z/my-project/work/listening/regression_words.json',
              'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\nSaved: /home/z/my-project/work/listening/regression_words.json')


if __name__ == '__main__':
    main()
