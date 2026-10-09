#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""توليد بيانات خط النص لمنفذ JS (textpipe.js) — Eiqaz v1.

يستخرج كل المعاجم والقوائم والرموز من eqz_tokens.py/eqz_text.py نفسها
(المصدر الوحيد للحقيقة) ويكتبها JSON/JS — لا نسخ يدوي يخطئ أبدًا.

المخرجات:
  web-exp/models/textpipe_data.json + .js — ثوابت سياسة القاف v1 + الرموز
  web-exp/tests/expected_tokens.json — التوكنز المتوقعة (وضع manual الحتمي)
  web-exp/tests/expected_catt.json   — مخرجات catt_eo المتوقعة
  web-exp/tests/expected_normalize.json — تطبيع eqz_text المتوقع (مرجع فوري)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(INF_DIR)
WEBEXP = os.path.join(REPO, 'web-exp')

sys.path.insert(0, INF_DIR)
sys.path.insert(0, os.path.join(INF_DIR, 'lib'))
sys.path.insert(0, os.path.join(INF_DIR, 'lib', 'mixer_repo'))

import infer  # noqa: E402


def main():
    import eqz_tokens
    from tts_arabic.text.symbols import symbols as SYMBOLS

    data = {
        'version': 2,
        'generated_from': ('inference/lib/eqz_tokens.py + eqz_text.py + '
                           'tts_arabic.text.symbols (Eiqaz v1)'),
        'EGY_SOUND_MAP': eqz_tokens.EGY_SOUND_MAP,
        'TRAIN_MAX_TOKENS': infer.TRAIN_MAX_TOKENS,
        'symbols': SYMBOLS,
        'FORCED_Q_SKELETONS': sorted(eqz_tokens.FORCED_Q_SKELETONS),
        'FORCED_G_SKELETONS': sorted(eqz_tokens.FORCED_G_SKELETONS),
        'QAF_MARKER_MAP': dict(eqz_tokens._QAF_MARKER_MAP),
        'ALLOWED_SPEAKERS': list(infer.ALLOWED_SPEAKERS),
        'EGYPTIAN_SPEAKERS': list(infer.EGYPTIAN_SPEAKERS),
    }
    os.makedirs(os.path.join(WEBEXP, 'models'), exist_ok=True)
    out = os.path.join(WEBEXP, 'models', 'textpipe_data.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    out_js = os.path.join(WEBEXP, 'models', 'textpipe_data.js')
    with open(out_js, 'w', encoding='utf-8') as f:
        f.write('// مولَّد آليًا بواسطة inference/onnx/gen_textpipe_data.py '
                '— لا تُعدَّل يدويًا\n')
        f.write('globalThis.TEXTPIPE_DATA = ')
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        f.write(';\n')
    print('written:', out, '+', out_js)

    # ---- مراجع التطبيع (eqz_text مباشرة — فحص فوري بلا نماذج) --------------
    import eqz_text
    norm_cases = [
        'عندي 5 كتب', 'عمره 10 سنين', 'عام 2026', '250 جنيه', '12345',
        '٥ كتب', 'استهلكت 80% من الباقة', '3.5', '5.4',
        '192.168.1.1', '1,000 جنيه', 'meeting بكرة', '{ق}انون',
        '{ق}قانون', 'هل أنت جاهز؟', 'دلوقتي هنبدأ الدرس، يا عمر.',
        'خلينا الساعة 7 الصبح ونرجع 3 العصر', 'عندي 2500000 جنيه',
        'الالتحاق بويندوز 11 ممتاز', 'نزل الupdate الجديد',
    ]
    norm_expected = []
    for raw in norm_cases:
        for dialect in ('egy', 'msa'):
            norm_expected.append({
                'raw': raw, 'dialect': dialect,
                'normalized': eqz_text.normalize_text(raw, dialect),
            })
    outn = os.path.join(WEBEXP, 'tests', 'expected_normalize.json')
    os.makedirs(os.path.join(WEBEXP, 'tests'), exist_ok=True)
    with open(outn, 'w', encoding='utf-8') as f:
        json.dump(norm_expected, f, ensure_ascii=False, indent=1)
    print(f'written: {outn} ({len(norm_expected)} حالة)')

    # ---- التوكنز المتوقعة (وضع manual — بلا تشكيل آلي: حتمي) ---------------
    with open(os.path.join(HERE, 'golden_texts.json'), encoding='utf-8') as f:
        gold = json.load(f)
    toks_ms, toks_egy, ids_of = infer.get_tokenizer()
    expected = []
    for item in gold['items'] + gold.get('extra_parity', []):
        try:
            res = infer.prepare_text_rich(item['text'], 'manual',
                                          item['dialect'])
            toks = toks_ms(res['text']) if item['dialect'] == 'msa' \
                else toks_egy(res['text'])
            ids = ids_of(toks)
            expected.append({
                'id': item['id'], 'text': item['text'],
                'dialect': item['dialect'],
                'prepared': res['text'], 'tokens': toks, 'ids': ids,
            })
        except Exception as e:                                  # noqa: BLE001
            expected.append({'id': item['id'], 'text': item['text'],
                             'dialect': item['dialect'],
                             'error': f'{type(e).__name__}: {e}'})
    out2 = os.path.join(WEBEXP, 'tests', 'expected_tokens.json')
    with open(out2, 'w', encoding='utf-8') as f:
        json.dump(expected, f, ensure_ascii=False, indent=1)
    n_ok = sum(1 for e in expected if 'ids' in e)
    print(f'written: {out2} ({n_ok}/{len(expected)} ok)')

    # ---- مخرجات catt المتوقعة ----------------------------------------------
    catt_out = []
    for item in gold['items']:
        try:
            v = infer.catt_vocalize(item['text'])
            catt_out.append({'id': item['id'], 'text': item['text'],
                             'vocalized': ' '.join(v.split())})
        except Exception as e:                                  # noqa: BLE001
            catt_out.append({'id': item['id'], 'text': item['text'],
                             'error': f'{type(e).__name__}: {e}'})
    out3 = os.path.join(WEBEXP, 'tests', 'expected_catt.json')
    with open(out3, 'w', encoding='utf-8') as f:
        json.dump(catt_out, f, ensure_ascii=False, indent=1)
    print(f'written: {out3} ({sum(1 for c in catt_out if "vocalized" in c)}'
          f'/{len(catt_out)} ok)')


if __name__ == '__main__':
    main()
