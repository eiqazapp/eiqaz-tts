#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msa_build_package.py — التحقق الشامل + بناء حزمة Kaggle النهائية.

1. تحقق: كل سجل له ملف wav موجود بالمدة الصحيحة، ولا ملفات يتيمة.
2. دمج كل السجلات → filelist.csv + filelist.jsonl.
3. إحصاءات شاملة (ساعات، تعرض قاف، متحدثون) → META.json.
4. بناء دليل كاغل: نقل الأصوات + نسخ filelists + ph4 + السكربتات +
   README + dataset-metadata.json.
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import soundfile as sf

BASE = '/home/z/my-project/work'
PREP = f'{BASE}/msa_prep'
KAG = f'{BASE}/kaggle_msa_dataset'
PH4 = f'{BASE}/ph4_dataset'
TOOLS = '/home/z/my-project/scripts'

# ---------- 1) تحميل كل السجلات ----------
records = []
for f in sorted(glob.glob(f'{PREP}/records/clartts_*.json')):
    records += json.load(open(f, encoding='utf-8'))
records += json.load(open(f'{PREP}/records/cv_records.json', encoding='utf-8'))

# توحيد التسمية: test → eval (اصطلاح ph4/NileTTS)
for r in records:
    if r['split'] == 'test':
        r['split'] = 'eval'

# ترتيب مستقر: clartts ثم cv، بترتيب utt
records.sort(key=lambda r: (r['utt'].startswith('cvf'), r['utt']))
assert len({r['utt'] for r in records}) == len(records), 'duplicate utt!'


def wav_root():
    """دليل جذر الأصوات — PREP أو KAG (بعد النقل)."""
    for root in (PREP, KAG):
        if os.path.isdir(f'{root}/clartts/wavs'):
            return root
    raise SystemExit('no wav root found!')


WR = wav_root()
print('wav root:', WR)

# ---------- 2) التحقق من الملفات ----------
missing, dur_mismatch, orphans = [], [], []
valid = []
for r in records:
    p = f"{WR}/{r['wav']}"
    if not os.path.exists(p):
        missing.append(r['utt'])
        continue
    info = sf.info(p)
    file_dur = info.frames / info.samplerate
    if abs(file_dur - r['duration_s']) > 0.05:
        dur_mismatch.append((r['utt'], file_dur, r['duration_s']))
        continue
    if info.samplerate != 22050 or info.channels != 1 \
            or info.subtype != 'PCM_16':
        dur_mismatch.append((r['utt'], 'fmt', f"{info.samplerate}/"
                            f"{info.channels}/{info.subtype}"))
        continue
    valid.append(r)

wav_files = set()
for d in ('clartts/wavs', 'cv_female/wavs'):
    dp = f'{WR}/{d}'
    if os.path.isdir(dp):
        wav_files |= {f'{d}/{fn}' for fn in os.listdir(dp)}
referenced = {r['wav'] for r in valid}
orphans = wav_files - referenced

print(f'records={len(records)} valid={len(valid)} missing={len(missing)} '
      f'dur_mismatch={len(dur_mismatch)} orphans={len(orphans)}')
if missing:
    print('  missing sample:', missing[:5])
if dur_mismatch:
    print('  mismatch sample:', dur_mismatch[:5])
if orphans:
    print('  orphan sample:', sorted(orphans)[:5])
assert not missing and not dur_mismatch, 'FAILED validation'

# ---------- 3) الإحصاءات ----------
def stats(rs, label):
    h = sum(r['duration_s'] for r in rs) / 3600
    q = sum(r['n_q'] for r in rs)
    tok = sum(r['n_tokens'] for r in rs)
    spk = {}
    for r in rs:
        s = spk.setdefault(r['speaker'], {'clips': 0, 'hours': 0.0})
        s['clips'] += 1
        s['hours'] += r['duration_s'] / 3600
    out = {
        'label': label, 'clips': len(rs), 'hours': round(h, 2),
        'q_tokens': q, 'tokens': tok,
        'q_share_pct': round(100 * q / max(1, tok), 2),
        'train_pool_ok': sum(1 for r in rs if r['train_pool_ok']),
        'split_train': sum(1 for r in rs if r['split'] == 'train'),
        'split_eval': sum(1 for r in rs if r['split'] == 'eval'),
        'speakers': {k: {'clips': v['clips'],
                         'hours': round(v['hours'], 2)}
                     for k, v in spk.items()},
    }
    return out

clt = [r for r in valid if r['utt'].startswith('clt')]
cvf = [r for r in valid if r['utt'].startswith('cvf')]
s_clt, s_cvf = stats(clt, 'ClArTTS (male MSA)'), stats(cvf, 'CV female MSA')
print(json.dumps({'clartts': s_clt, 'cv': s_cvf}, ensure_ascii=False,
                 indent=1))

# ---------- 4) كتابة filelists ----------
os.makedirs(KAG, exist_ok=True)
FIELDS = ['utt', 'wav', 'split', 'dialect', 'speaker', 'text', 'text_raw',
          'n_tokens', 'train_pool_ok', 'n_q', 'tashkeel_density',
          'duration_s', 'sr_orig', 'clip_id', 'source', 'license']
with open(f'{PREP}/filelist.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
    w.writeheader()
    for r in valid:
        w.writerow(r)
with open(f'{PREP}/filelist.jsonl', 'w', encoding='utf-8') as f:
    for r in valid:
        f.write(json.dumps({k: r[k] for k in FIELDS},
                           ensure_ascii=False) + '\n')

# ---------- 5) META ----------
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

meta = {
    'name': 'msa-tts-data-v1',
    'generated_at': '2026-10-03',
    'purpose': ('MSA (فصحى) training data for the Eiqaz merged MixerTTS '
                'training — teaches the untrained q token (قاف أصيلة) via '
                'the toks_ms path and provides an MSA register.'),
    'format': {
        'audio': 'wav 22050Hz mono PCM16, peak<=0.95 (NileTTS-prep rule)',
        'text': 'full tashkeel; punctuation ، → , and ؟ → ? only '
                '(ph2 convention); dialect=msa → toks_ms tokenizer',
        'token_caps': 'write<=180; train_pool_ok<=160 (NileTTS rules)',
    },
    'clartts': s_clt,
    'cv_female': s_cvf,
    'total': {
        'clips': len(valid),
        'hours': round(sum(r['duration_s'] for r in valid) / 3600, 2),
        'q_tokens': s_clt['q_tokens'] + s_cvf['q_tokens'],
        'q_share_pct': round(
            100 * (s_clt['q_tokens'] + s_cvf['q_tokens']) /
            max(1, s_clt['tokens'] + s_cvf['tokens']), 2),
    },
    'sources': {
        'MBZUAI/ClArTTS': 'CC-BY-4.0 — classical male, hand-diacritized',
        'MohamedRashad/common-voice-18-arabic':
            'CC0-1.0 (Common Voice 18 Arabic mirror) — top-4 female '
            'speakers; texts diacritized locally with catt (catt_eo.onnx)',
    },
    'suggested_merged_speaker_ids': {
        '0': 'NileTTS SPEAKER_01 (male, Egyptian)',
        '1': 'NileTTS SPEAKER_02 (female, Egyptian)',
        '2': 'clartts_male',
        '3': 'cvf_78c954e3', '4': 'cvf_5f810213',
        '5': 'cvf_cf4d8f89', '6': 'cvf_fc3b87e3',
    },
    'filelist_csv_sha256': sha(f'{PREP}/filelist.csv'),
    'filelist_jsonl_sha256': sha(f'{PREP}/filelist.jsonl'),
}
json.dump(meta, open(f'{PREP}/META.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---------- 6) بناء دليل Kaggle ----------
# نقل الأصوات (بدون نسخ مكلفة) — idempotent وآمن:
# النقل فقط عندما يوجد المصدر في PREP؛ إن كانت موجودة في KAG فلا تمسها.
for d in ('clartts', 'cv_female'):
    src, dst = f'{PREP}/{d}', f'{KAG}/{d}'
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
for fn in ('filelist.csv', 'filelist.jsonl', 'META.json'):
    shutil.copy2(f'{PREP}/{fn}', f'{KAG}/{fn}')

# حزمة نصوص Egyptian (ph4) — حماية إضافية
if os.path.exists(f'{KAG}/niletts_tashkeel_text'):
    shutil.rmtree(f'{KAG}/niletts_tashkeel_text')
shutil.copytree(PH4, f'{KAG}/niletts_tashkeel_text')

# سكربتات التوليد — قابلية إعادة الإنتاج
os.makedirs(f'{KAG}/tools', exist_ok=True)
for s in ('msa_prep_clartts.py', 'msa_prep_cv.py', 'msa_build_package.py'):
    shutil.copy2(f'{TOOLS}/{s}', f'{KAG}/tools/{s}')

# scan/catt_cache للشفافية
shutil.copy2(f'{PREP}/cv_female_scan.json', f'{KAG}/cv_female_scan.json')

# ---------- 7) بطاقة README ----------
t = meta['total']
readme = f"""<div dir="rtl">

# بيانات تدريب الفصحى لـ Eiqaz TTS — msa-tts-data-v1

بيانات صوتية ونصية **بالعربية الفصحى المشكولة بالكامل**، أُعدت لتدريج
توكن القاف الأصيلة `q` (غير المدرَّب في نموذج NileTTS الحالي: 0/1.59M)
وإضافة سجل صوتي فصحي إلى التدريب المدمج المصري+الفصحى (مشروع Eiqaz).

## المحتوى

| المصدر | المتحدث | المقطع | الساعات | الرخصة | ملاحظة النص |
|---|---|---|---|---|---|
| MBZUAI/ClArTTS (كلاسيكي) | ذكر واحد | {s_clt['clips']:,} | {s_clt['hours']}h | CC-BY-4.0 | مشكول يدويًا (أصلي) |
| Common Voice 18 العربي (مرآة) | 4 إناث | {s_cvf['clips']:,} | {s_cvf['hours']}h | CC0-1.0 | مشكول بـ catt محليًا |
| **الإجمالي** | 5 | **{t['clips']:,}** | **{t['hours']}h** | — | — |

**تعرض القاف**: {t['q_tokens']:,} توكن q ({t['q_share_pct']}% من كل التوكنات) —
هذا هو الوقود المباشر لهدف «ق فصحى في كلمات مثل القرآن، فقط، قطعة».

## البنية

```
filelist.csv / filelist.jsonl   ← كل الوحدات (utt, wav, split, dialect=msa,
                                   speaker, text مشكول, text_raw, n_tokens,
                                   train_pool_ok, n_q, duration_s, ...)
clartts/wavs/clt_*.wav          ← {s_clt['clips']:,} ملفًا (22050Hz mono PCM16)
cv_female/wavs/cvf_*.wav        ← {s_cvf['clips']:,} ملفًا (22050Hz mono PCM16)
niletts_tashkeel_text/          ← حزمة نصوص الجانب المصري (ph4: 21,854 وحدة
                                   مشكولة بمحرك LLM — للدمج في التدريب)
tools/                          ← سكربتات التوليد (قابلية إعادة الإنتاج)
META.json, cv_female_scan.json  ← الإحصاءات والاختيار
```

## المواصفات (متوافقة حرفيًا مع خط NileTTS-prep)

- **الصوت**: 22050Hz mono PCM16، ذروة ≤ 0.95
- **النص**: تشكيل كامل؛ تحويل الترقيم `، → ,` و`؟ → ?` فقط (اصطلاح ph2)؛
  لا ASCII alpha؛ ≥ 10 حروف عربية؛ ≥ 3 ثوانٍ
- **الحدود**: توكنات ≤ 180 (كتابة)؛ `train_pool_ok` = توكنات ≤ 160
- **التوكنيزر المقترح للتدريب**: مسار فصحى `toks_ms` (يحفظ توكن q بإذن الله) —
  مقابل مسار مصري `toks_egy` (يحوله لهمزة `<`)

## معرفات المتحدثين المقترحة للتدريب المدمج

`0` = NileTTS ذكر مصري · `1` = NileTTS أنثى مصرية · `2` = clartts_male ·
`3` = cvf_78c954e3 · `4` = cvf_5f810213 · `5` = cvf_cf4d8f89 ·
`6` = cvf_fc3b87e3

## العزو (مطلوب بموجب CC-BY-4.0 لجزء ClArTTS)

- **ClArTTS**: MBZUAI — Classical Arabic TTS corpus (CC-BY-4.0)، كما هو على
  Hugging Face `MBZUAI/ClArTTS`.
- **Common Voice 18 Arabic**: مساهمو Mozilla Common Voice (CC0-1.0)، عبر
  المرآة `MohamedRashad/common-voice-18-arabic`؛ اختيرت أفضل 4 متحدثات
  (`female_feminine`) بالمدة الفعلية المقيسة، وأشكلت نصوصهن محليًا بنموذج
  catt (`catt_eo.onnx`) — النص الأصلي محفوظ في عمود `text_raw`.
- **niletts_tashkeel_text**: ناتج خط تشكيل LLM داخلي على نصوص
  `KickItLikeShika/NileTTS-dataset` (لا صوت هنا — نص فقط).

> ملاحظة اختيار المتحدثات: أعلى متحدثة = 1,283 مقطعًا/1.38h مقيسة فعليًا
> (تقدير «5.4h» في تخطيط سابق كان خاطئًا)؛ ضُمت أفضل 4 لموازنة ذكر
> ClArTTS (11.4h).

</div>

---
Generated 2026-10-03 by the Eiqaz tashkeel/training pipeline (scripts:
msa_prep_clartts.py, msa_prep_cv.py, msa_build_package.py — included in
`tools/`). All floats verified: every record's wav exists with matching
duration/format (see META.json).
"""
open(f'{KAG}/README.md', 'w', encoding='utf-8').write(readme)

ds_meta = {
    'title': 'Eiqaz MSA TTS Training Data v1',
    'id': 'ahmedyaseen86/msa-tts-data-v1',
    'licenses': [{'name': 'CC-BY-4.0'}],
    'private': True,
    'keywords': ['tts', 'arabic', 'msa', 'tashkeel', 'speech'],
}
json.dump(ds_meta, open(f'{KAG}/dataset-metadata.json', 'w'), indent=1)

print('\nKaggle dataset dir built at', KAG)
total_mb = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(KAG) for f in fs) / 1e6
print(f'total size: {total_mb:.0f} MB')
print('validation PASSED — ready for upload')
