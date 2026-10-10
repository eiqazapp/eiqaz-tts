#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""listening_gen.py — مولّد أزواج الاستماع قبل/بعد (نسخة موحدة لبيئتين).

استخدام: python3 listening_gen.py <INF_DIR> <OUT_DIR>

يعمل المسار الكامل من كود النسخة الموجودة في <INF_DIR>:
  النص → prepare_text_rich(auto, egy) → synthesize(نفس الإعدادات) → WAV

الحالات: 8 جمل مجموعة الاستماع + حالة A1 القديمة (وضع egyptian) لإثبات
منشأ التسجيلات القديمة عبر إعادة التوليد ومقارنة البصمات.

إعدادات موثقة داخل التقرير + SHA الكود + نظافة شجرة git — لإثبات أن
الفرق الوحيد بين التشغيلين هو نسخة الكود (التشكيل فقط؛ كود التوليد
الصوتي متطابق بايت-بايت بين origin/main والفرع — موثق في التقرير).
"""
import hashlib
import json
import os
import subprocess
import sys

INF_DIR = os.path.abspath(sys.argv[1])
OUT_DIR = os.path.abspath(sys.argv[2])
for p in (INF_DIR, os.path.join(INF_DIR, 'lib')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(INF_DIR)

import infer  # noqa: E402  — يُستورد من INF_DIR المحدد حصراً

CKPT = os.path.join(INF_DIR, 'checkpoints', 'states_cont_180516.pth')

# (id, نص خام, وضع, لهجة) — الوضع auto/egy هو افتراضي التطبيق؛
# A1_OLD بإعدادها الأصلي (egyptian) لمطابقة تسجيلات التدقيق القديمة.
CASES = [
    ('S1', 'بِيَكْتِب الدرس', 'auto', 'egy'),
    ('S2', 'هو بيقول للطالب', 'auto', 'egy'),
    ('S3', 'دلوقتي هنبدأ الدرس يا عمر', 'auto', 'egy'),
    ('S4', 'أنا مش فاهم السؤال ده', 'auto', 'egy'),
    ('S5', 'بكرة هنعمل تجربة جديدة', 'auto', 'egy'),
    ('S6', 'هو بيقول كده', 'auto', 'egy'),
    ('S7', 'مش عارف أعمل إيه', 'auto', 'egy'),
    ('S8', 'هو هنا', 'auto', 'egy'),
    ('A1_OLD', 'هو بيقول كده', 'egyptian', 'egy'),
]

# نفس إعدادات التدقيق الأصلي — لا تغيير في أي عامل
SETTINGS = {
    'checkpoint': 'states_cont_180516.pth (git blob b0656ac7 — unmodified)',
    'speaker': 0,
    'pace': 1.0,
    'denoise': 0.005,
    'peak_normalize': True,
    'sample_rate': 22050,
}


def git_info():
    """SHA الكود المشغَّل + نظافة الشجرة — إثبات إصدار الكود تشغيليًا."""
    sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=INF_DIR, text=True).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=INF_DIR, text=True).strip()
    return sha, dirty


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    sha, dirty = git_info()
    import torch
    print(f'INF_DIR  = {INF_DIR}')
    print(f'git sha  = {sha[:12]}  | dirty={bool(dirty)}')
    print(f'torch    = {torch.__version__}')

    model, it = infer.load_model(CKPT)
    _, toks_egy, _ = infer.get_tokenizer()
    print(f'model loaded (iter={it})')

    rows = []
    for cid, text, mode, dialect in CASES:
        res = infer.prepare_text_rich(text, mode, dialect)
        final_text = res['text']
        st = res.get('stages') or {}
        wav = os.path.join(OUT_DIR, f'{cid}.wav')
        n_ids, n_secs = infer.synthesize(
            model, final_text, dialect, speaker=0, pace=1.0,
            out_path=wav, denoise=0.005, peak_normalize=True)
        row = {
            'id': cid, 'input': text, 'mode': mode, 'dialect': dialect,
            'final_text': final_text,
            'n_tokens': n_ids, 'tokens': list(toks_egy(final_text)),
            'duration_sec': round(n_secs, 3), 'wav': wav,
            'wav_md5': md5(wav),
            # معلومات المراحل — متوفرة في الفرع فقط (main بلا stages)
            'effective_mode': st.get('effective_mode', res.get('mode')),
            'normalized': st.get('normalized', res.get('normalized')),
            'after_catt': st.get('after_catt'),
            'after_det': st.get('after_det'),
            'after_merge': st.get('after_merge'),
            'merge_fallback': st.get('merge_info', {}).get('fallback')
            if isinstance(st.get('merge_info'), dict) else
            res.get('merge_fallback'),
        }
        rows.append(row)
        print(f"[{cid}] {text}")
        print(f"    → {final_text}  ({n_ids} tok, {n_secs:.2f}s)")

    report = {
        'inf_dir': INF_DIR,
        'git_sha': sha, 'git_dirty': bool(dirty),
        'torch': torch.__version__,
        'settings': SETTINGS,
        'method': 'المسار الكامل من كود هذه النسخة فقط: prepare_text_rich '
                  '+ synthesize بنفس الإعدادات. الفرضية: كود التوليد '
                  'الصوتي متطابق بين النسختين (مُتحقق منه بالفروق '
                  'البايتية) — الفرق الوحيد مسار التشكيل.',
        'cases': rows,
    }
    out = os.path.join(OUT_DIR, 'gen_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\nReport: {out}')


if __name__ == '__main__':
    main()
