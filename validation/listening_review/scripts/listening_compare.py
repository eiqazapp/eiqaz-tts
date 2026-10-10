#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""listening_compare.py — مقارنة أزواج الاستماع + إثبات منشأ التسجيلات القديمة.

1) إثبات المنشأ: مخرجات إعادة التوليد (main worktree / fix checkout) مقابل
   التسجيلات القديمة المحفوظة في download/audit_audio/ — بصمات MD5.
2) مقارنة كل زوج (main مقابل fix): النص النهائي، فرق الحركات كلمةً كلمة،
   التوكنات، هوية WAV البايتية، المدة.
3) الضوابط: S7 يجب ألا تتغير؛ الكلمات غير المستهدفة يجب ألا تتغير.
"""
import hashlib
import json
import os
import re
import sys
import unicodedata

MAIN_DIR = '/home/z/my-project/work/listening/main'
FIX_DIR = '/home/z/my-project/work/listening/fix'
OLD_DIR = '/home/z/my-project/download/audit_audio'

AR_DIAC = set(
    '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670'
    '\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DF\u06E0'
    '\u06E1\u06E2\u06E3\u06E4\u06E5\u06E6\u06E7\u06E8\u06EA'
    '\u06EB\u06EC\u06ED')


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def strip_diac(w):
    return ''.join(ch for ch in w if ch not in AR_DIAC)


def norm_marks(w):
    """توحيد ترتيب العلامات المركبة (شدة+فتحة) — NFC يرتبها قانونيًا."""
    return unicodedata.normalize('NFC', w)


def diac_of(w):
    """حركات الكلمة كأزواج (حرف، حركاته) — الحركة تتبع الحرف السابق."""
    out = []
    for ch in norm_marks(w):
        if ch in AR_DIAC:
            if out:
                out[-1] = (out[-1][0], out[-1][1] + ch)
        else:
            out.append((ch, ''))
    return out


def mark_order_only(bw, aw):
    """هل الفرق بين الكلمتين ترتيب علامات فقط (نطق متطابق)؟"""
    return norm_marks(bw) == norm_marks(aw) and bw != aw


def word_diff(before_w, after_w):
    """أي الحركات تغيرت بين نسختي الكلمة نفسها (نفس الهيكل)."""
    if mark_order_only(before_w, after_w):
        return {'kind': 'mark_order_only', 'before': before_w,
                'after': after_w,
                'note': 'ترتيب العلامات المركبة فقط — النطق متطابق بعد NFC'}
    if strip_diac(norm_marks(before_w)) != strip_diac(norm_marks(after_w)):
        return {'kind': 'structure', 'before': before_w, 'after': after_w}
    b, a = diac_of(before_w), diac_of(after_w)
    changes = {}
    for (ch, bd), (_, ad) in zip(b, a):
        if bd != ad:
            changes[ch] = [bd or '—', ad or '—']
    if changes:
        return {'kind': 'vowels', 'before': before_w, 'after': after_w,
                'changed_letters': changes}
    return None


def main():
    main_rep = json.load(open(f'{MAIN_DIR}/gen_report.json', encoding='utf-8'))
    fix_rep = json.load(open(f'{FIX_DIR}/gen_report.json', encoding='utf-8'))
    mb = {c['id']: c for c in main_rep['cases']}
    fb = {c['id']: c for c in fix_rep['cases']}

    # ---------- 1) إثبات منشأ التسجيلات القديمة ----------
    old_map = {  # (old_id) -> (new_id, side)
        'A1': 'A1_OLD', 'A2': 'S1', 'A3': 'S3', 'A4': 'S7', 'A5': 'S8'}
    provenance = []
    print('=' * 70)
    print('أ) إثبات منشأ التسجيلات القديمة (إعادة توليد مقابل محفوظ)')
    print('=' * 70)
    for old_id, new_id in old_map.items():
        for side, rep_dir in (('after_fix', FIX_DIR), ('before_main', MAIN_DIR)):
            old_p = f'{OLD_DIR}/{old_id}_{side}.wav'
            new_p = f'{rep_dir}/{new_id}.wav'
            if not (os.path.exists(old_p) and os.path.exists(new_p)):
                provenance.append({'old': old_p, 'new': new_p,
                                   'status': 'missing'})
                continue
            same = md5(old_p) == md5(new_p)
            provenance.append({
                'old_file': os.path.basename(old_p),
                'regenerated_from': (
                    'fix' if side == 'after_fix' else 'main') +
                f'@{new_id}',
                'md5_identical': same,
                'old_md5': md5(old_p)[:10], 'new_md5': md5(new_p)[:10]})
            print(f"  {os.path.basename(old_p):24s} vs {('fix' if side=='after_fix' else 'main'):4s}@{new_id:7s}"
                  f" → {'متطابق بتّيًا ✓' if same else 'مختلف ✗'}")

    # ---------- 2) مقارنة الأزواج ----------
    pairs = []
    print()
    print('=' * 70)
    print('ب) مقارنة أزواج الاستماع (main مقابل fix — نفس الإعدادات)')
    print('=' * 70)
    for cid in [f'S{i}' for i in range(1, 9)]:
        m, f_ = mb[cid], fb[cid]
        mws, fws = m['final_text'].split(), f_['final_text'].split()
        words_equal_len = len(mws) == len(fws)
        word_changes = []
        if words_equal_len:
            for bw, aw in zip(mws, fws):
                d = word_diff(bw, aw)
                if d:
                    word_changes.append(d)
        wav_same = md5(m['wav']) == md5(f_['wav'])
        tok_same = m['tokens'] == f_['tokens']
        pairs.append({
            'id': cid, 'input': m['input'],
            'main_text': m['final_text'], 'fix_text': f_['final_text'],
            'n_words_main': len(mws), 'n_words_fix': len(fws),
            'words_preserved': words_equal_len,
            'word_changes': word_changes,
            'tokens_identical': tok_same,
            'tokens_changed_positions': (
                None if tok_same else
                [i for i, (a, b) in enumerate(
                    zip(m['tokens'], f_['tokens'])) if a != b]),
            'wav_identical': wav_same,
            'main_dur': m['duration_sec'], 'fix_dur': f_['duration_sec'],
            'main_n_tokens': m['n_tokens'], 'fix_n_tokens': f_['n_tokens'],
        })
        tag = 'متطابق' if wav_same else 'مختلف'
        print(f"[{cid}] {m['input']}")
        print(f"   main: {m['final_text']}  ({m['duration_sec']}s)")
        print(f"   fix : {f_['final_text']}  ({f_['duration_sec']}s)")
        print(f"   WAV {tag} | توكن {m['n_tokens']}→{f_['n_tokens']}"
              f" | كلمات متغيرة: {len(word_changes)}")
        for wc in word_changes:
            print(f"     • {wc['before']} → {wc['after']}"
                  f"  {json.dumps(wc.get('changed_letters', {}), ensure_ascii=False)}")

    out = {
        'provenance_old_recordings': provenance,
        'pairs': pairs,
        'environment': {
            'main_git_sha': main_rep['git_sha'],
            'fix_git_sha': fix_rep['git_sha'],
            'main_torch': main_rep['torch'], 'fix_torch': fix_rep['torch'],
            'settings': fix_rep['settings'],
        },
    }
    with open('/home/z/my-project/work/listening/comparison.json', 'w',
              encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print('\nSaved: /home/z/my-project/work/listening/comparison.json')


if __name__ == '__main__':
    main()
