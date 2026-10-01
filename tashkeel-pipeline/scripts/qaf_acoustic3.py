# -*- coding: utf-8 -*-
"""qaf_acoustic3.py — تحليل نهائي حاسم بمرشحات نطاقية حقيقية.

المنهج:
  - مرشحات Butterworth (sosfilt) لنطاقات: جهر 60-300 / منخفض 250-1000 /
    متوسط 1000-2000 / عالي 2000-5000 Hz
  - كشف بدايات: طاقة النطاق 300-5000 (نافذة 8ms/خطوة 2ms) مع شرط إغلاق
    (هبوط للطاقة قبل الحدث)
  - لكل حدث (نافذة انفجار 16ms بعد البداية):
      * توزيع الطاقة عبر النطاقات الثلاثة → مكان الانفجار
      * جهر الإغلاق: نسبة نطاق الجهر/الكل في 40ms قبل البداية
      * VOT: حتى جهر >0.4 مستقرًا
  - مطابقة يدوية بالترتيب مع تسلسل الجملة (الجمل قصيرة ومعروفة)

المخرج المرجعي داخل كل ملف: جيم مصرية [g] (توكن v مدرَّب) للمقارنة
المباشرة مع القاف في نفس الجملة/نفس المتحدث.
"""
import json

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 22050


def load(path):
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x / (np.abs(x).max() + 1e-9)


def bandpass(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=SR, output='sos')
    return sosfiltfilt(sos, x)


def env(x, frame=0.008, hop=0.002):
    n = int((len(x) / SR - frame) / hop) + 1
    L = int(frame * SR)
    out = np.zeros(n)
    for i in range(n):
        s = i * int(hop * SR)
        out[i] = float(np.sqrt(np.sum(x[s:s + L] ** 2))) if s + L <= len(x) else 0.0
    return out


def analyze(path, label, expect):
    x = load(path)
    y_burst = bandpass(x, 300, 5000)     # للكشف
    y_voice = bandpass(x, 60, 300)       # الجهر
    b_lo = bandpass(x, 250, 1000)        # انفجار منخفض (q-ish)
    b_mid = bandpass(x, 1000, 2000)      # وسطي
    b_hi = bandpass(x, 2000, 5000)       # عالي (k/t-ish)
    y_all = np.abs(x)

    e_b = env(y_burst)
    hop = 0.002
    n = len(e_b)
    thr = np.percentile(e_b, 85) + 1e-9

    events = []
    i = 25  # تجاهل أول 50ms
    while i < n - 8:
        # بداية: قفزة كبيرة بعد إغلاق هادئ (الإغلاق: أدنى من 25% من الحدث)
        cur = e_b[i:i + 4].max()
        pre = e_b[max(0, i - 20):max(1, i - 2)].min()
        if cur > 8 * (pre + 1e-12) and cur > thr:
            t = i * hop
            # نافذة الانفجار: 16ms بعد البداية
            s = int(t * SR)
            w = int(0.016 * SR)
            seg_lo = np.sum(b_lo[s:s + w] ** 2)
            seg_mid = np.sum(b_mid[s:s + w] ** 2)
            seg_hi = np.sum(b_hi[s:s + w] ** 2)
            tot = seg_lo + seg_mid + seg_hi + 1e-12
            # جهر الإغلاق (40ms قبل)
            ps = max(0, s - int(0.040 * SR))
            voice_pre = float(np.sum(y_voice[ps:s] ** 2) /
                              (np.sum(y_all[ps:s] ** 2) + 1e-12))
            # VOT
            vot = None
            for j in range(s, min(len(x) - 3, s + int(0.12 * SR)), int(0.002 * SR)):
                L = int(0.010 * SR)
                v = float(np.sum(y_voice[j:j + L] ** 2) /
                          (np.sum(y_all[j:j + L] ** 2) + 1e-12))
                if v > 0.40:
                    vot = round((j - s) / SR * 1000)
                    break
            events.append({
                't': round(t, 2),
                'lo': round(100 * seg_lo / tot),
                'mid': round(100 * seg_mid / tot),
                'hi': round(100 * seg_hi / tot),
                'closure_voice': round(voice_pre, 2),
                'vot_ms': vot,
            })
            i += 12
        else:
            i += 1

    print(f'\n=== {label} :: {path.split("/")[-1]}')
    print(f'    متوقع: {expect}')
    for ev in events:
        dom = max(('lo', ev['lo']), ('mid', ev['mid']), ('hi', ev['hi']),
                  key=lambda kv: kv[1])[0]
        if ev['closure_voice'] > 0.30:
            v = 'إغلاق مجهور'
        else:
            v = 'إغلاق صامت'
        print(f"    [{ev['t']:>6.2f}s] lo={ev['lo']:>3}% mid={ev['mid']:>3}%"
              f" hi={ev['hi']:>3}% | {v} (جهر={ev['closure_voice']:.2f})"
              f" | VOT={ev['vot_ms']}ms | المهيمن={dom}"
              f" → {interp(ev, dom)}")
    return events


def interp(ev, dom):
    if ev['closure_voice'] > 0.30:
        return 'g/b/d (مجهور)'
    if dom == 'hi':
        return 'k/t/T (عالي صامت)'
    if dom == 'lo':
        return 'q؟ (منخفض صامت)' if (ev['vot_ms'] or 0) >= 25 else 'q?/T? (منخفض صامت VOT قصير)'
    return 'وسطي صامت'


def main():
    base = ('/home/z/my-project/work/github_repo/eiqaz-tts/training/'
            'run-artifacts/samples')
    targets = [
        (f'{base}/user_eval/speaker_0/difficult_words_00.wav', 'spk0-فوق',
         'الجمل(g) شايل الحمل فوق(q) الجبل(g) — مرجعا [g] حول (q)'),
        (f'{base}/user_eval/speaker_1/difficult_words_00.wav', 'spk1-فوق',
         'نفسها أنثى'),
        (f'{base}/user_eval/speaker_0/difficult_words_01.wav', 'spk0-قرش',
         'القرش(q1) غالي والقلب(q2) لب(b) بيوجع(b+g) قمر(q3)'),
        (f'{base}/user_eval/speaker_1/difficult_words_01.wav', 'spk1-قرش',
         'نفسها أنثى'),
        (f'{base}/user_eval/speaker_0/difficult_words_02.wav', 'spk0-مرجع',
         'ظرف الثلاجة(t) فيه تلات(t) كوبايين(k) شاي بلح(b) — لا قاف: خط أساس'),
        (f'{base}/egyptian_scratch/speaker_0/egy_07.wav', 'spk0-عقل',
         'الزحمة في الميترو(t) النهارده عقل(q) تخلي(t)'),
        (f'{base}/egyptian_scratch/speaker_0/egy_09.wav', 'spk0-نقعد',
         'عشان نقعد(q) مع بعض(b)'),
        (f'{base}/original_msa_reference/speaker_3/msa_02.wav', 'MSA-قراءة',
         'على القراءة(q) الصحيحة — النموذج الفصيح الأصلي: مرجع [q] حقيقي'),
    ]
    out = {}
    for path, label, expect in targets:
        try:
            out[label] = analyze(path, label, expect)
        except Exception as e:
            print(f'\n=== {label} — خطأ: {e}')
    with open('/home/z/my-project/work/qaf_research/'
              'acoustic_burst_scan3.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\nv3 حُفظ: work/qaf_research/acoustic_burst_scan3.json')


if __name__ == '__main__':
    main()
