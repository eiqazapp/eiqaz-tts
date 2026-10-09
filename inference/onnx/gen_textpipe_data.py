#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""توليد بيانات خط النص لمنفذ JS (textpipe.js).

يستخرج كل المعاجم والقوائم والرموز منinfer.py نفسه (المصدر الوحيد
للحقيقة) ويكتبها JSON — لا نسخ يدوي يخطئ أبدًا.

المخرجات:
  web-exp/models/textpipe_data.json — كل ثوابت القاف + الرموز + الخرائط
  web-exp/tests/expected_tokens.json — التوكنز المتوقعة للنصوص الذهبية
                                        (مسار never — بلا تشكيل catt)
  web-exp/tests/expected_catt.json   — مخرجات catt_eo المتوقعة للنصوص الذهبية
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INF_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(INF_DIR)
WEBEXP = os.path.join(REPO, 'web-exp')

sys.path.insert(0, INF_DIR)
sys.path.insert(0, os.path.join(INF_DIR, 'lib', 'mixer_repo'))

import infer  # noqa: E402


def main():
    from tts_arabic.text.symbols import symbols as SYMBOLS

    data = {
        'version': 1,
        'generated_from': 'inference/infer.py + tts_arabic.text.symbols',
        'EGY_TOKEN_MAP': infer.EGY_TOKEN_MAP,
        'TRAIN_MAX_TOKENS': infer.TRAIN_MAX_TOKENS,
        'symbols': SYMBOLS,
        'QAF_G_SKELETONS': sorted(infer.QAF_G_SKELETONS),
        'QAF_Q_SKELETONS': sorted(infer.QAF_Q_SKELETONS),
        'QAF_Q_STUDY_FORMS': dict(infer.QAF_Q_STUDY_FORMS),
        'QAF_Q_AFFIX_OK': sorted(infer.QAF_Q_AFFIX_OK),
        'QAF_Q_CORPUS_FORMS': dict(infer.QAF_Q_CORPUS_FORMS),
        'QAF_Q_TRUST': sorted(infer.QAF_Q_TRUST),
        'QAF_Q_VERIFIED': sorted(infer.QAF_Q_VERIFIED),
        'QAF_Q_TIER1': sorted(infer.QAF_Q_TIER1),
        'QAF_Q_SENTENCE_SKIP': sorted(infer.QAF_Q_SENTENCE_SKIP),
        'QAF_Q_DEEP_OK': sorted(infer.QAF_Q_DEEP_OK),
        'QAF_MARKER_MAP': dict(infer._QAF_MARKER_MAP),
        'SUN_LETTERS': sorted(infer._SUN_LETTERS),
        'QAF_CLITICS': list(infer._QAF_CLITICS),
        'QAF_DIAC': 'auiFNK~o',
    }
    os.makedirs(os.path.join(WEBEXP, 'models'), exist_ok=True)
    out = os.path.join(WEBEXP, 'models', 'textpipe_data.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    # نسخة سكريبت كلاسيكي (توافق أوسع مع متصفحات الهواتف — بلا ES modules)
    out_js = os.path.join(WEBEXP, 'models', 'textpipe_data.js')
    with open(out_js, 'w', encoding='utf-8') as f:
        f.write('// مولَّد آليًا بواسطة inference/onnx/gen_textpipe_data.py '
                '— لا تُعدَّل يدويًا\n')
        f.write('globalThis.TEXTPIPE_DATA = ')
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        f.write(';\n')
    print('written:', out, '+', out_js)

    # ---- التوكنز المتوقعة (مسار never — بلا catt) -------------------------
    with open(os.path.join(HERE, 'golden_texts.json'), encoding='utf-8') as f:
        gold = json.load(f)
    expected = []
    for item in gold['items'] + gold.get('extra_parity', []):
        try:
            # مسار الإنتاج نفسه (كما في synthesize): التحضير يضبط
            # qaf_actions/qaf_native وتُمرَّر للمرمِّز صراحةً
            res = infer.prepare_text_rich(item['text'], 'never',
                                          item['dialect'], 'auto')
            acts = res['qaf_actions'] or None
            nat = frozenset(res['qaf_native']) or None
            if item['dialect'] == 'msa':
                toks = infer.get_msa_synthesis_tokens(res['text'], 'auto',
                                                      acts)
                _, _, ids_of = infer.get_tokenizer('auto', acts, nat)
            else:
                _, toks_egy, ids_of = infer.get_tokenizer('auto', acts, nat)
                toks = toks_egy(res['text'])
            ids = ids_of(toks)
            expected.append({
                'id': item['id'], 'text': item['text'],
                'dialect': item['dialect'],
                'prepared': res['text'], 'tokens': toks, 'ids': ids,
                'qaf_actions': res['qaf_actions'],
                'qaf_native': list(res['qaf_native']),
            })
        except Exception as e:                                  # noqa: BLE001
            expected.append({'id': item['id'], 'text': item['text'],
                             'dialect': item['dialect'],
                             'error': f'{type(e).__name__}: {e}'})
    os.makedirs(os.path.join(WEBEXP, 'tests'), exist_ok=True)
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
