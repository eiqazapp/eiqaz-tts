# PATCHES.md — توثيق التعديلات على الكود المضمّن

## المبدأ العام

كل كود النموذج (`lib/mixer_repo`) ومكتبة المعالجة النصية
(`lib/tts_arabic`) منسوخ **كما هو** من حزمة التدريب الأصلية
(mixer-tts-scratch-data). أُجريت 5 تعديلات جراحية فقط، كلها من فئة
واحدة: **جعل استيرادات مسار-التدريب-فقط اختيارية** كي تعمل الحزمة على
Windows/CPU بتوزيعة مينيمالية (6 مكتبات) بدل تثبيت سلسلة
librosa/numba/matplotlib/transformers/gdown الثقيلة — كلها غير مستخدمة
في مسار التوليد إطلاقًا.

لا يوجد أي تعديل على: بنية النموذج، الأوزان، معادلات الترميز، خريطة
التوكنز المصرية، أو أي حساب رقمي.

---

## التعديلات الخمسة

### PATCH 1/4 — `lib/mixer_repo/models/mixer_tts/mixer_tts.py`
- **قبل:** `import transformers` (استيراد علوي إلزامي).
- **بعد:** استيراد محمي بـtry/except يضبط `transformers = None` عند غيابه.
- **السبب:** transformers لا يُستخدم إلا داخل `_get_lm_embeddings()`
  و`_get_lm_padding_value()` (تكييف LM)، و`self.cond_on_lm_embeddings`
  مثبّت على `False` في `__init__` بالكود الأصلي نفسه — الدالتان لا
  تُستدعيان أبدًا في الاستدلال.

### PATCH 2/4 — `lib/mixer_repo/models/mixer_tts/modules/helpers.py`
- **قبل:** استيراد علوي لـ`librosa` و`matplotlib.pylab` و`from numba import jit, prange`.
- **بعد:** استيرادان محميان (librosa/matplotlib يصبحان `None` عند
  الغياب) + بديل pure-python لدالكوريتور `@jit` و`prange = range`.
- **السبب:** هذه المكتبات تستخدمها فقط دوال التدريب/التسجيل في نفس
  الملف: `griffin_lim`، `log_audio_to_tb`، دوال `plot_*`، وتنصيف MAS
  المُسرَّع بـnumba في `binarize_attention_parallel`. مسار
  `model.infer()` لا يستدعي أيًّا منها (يستخدم `get_mask_from_lengths`
  و`regulate_len` فقط — كلاهما torch صافي). البديل يحافظ على عمل دوال
  MAS حتى بدون numba (أبطأ — وغير مطلوب أصلًا للاستدلال).

### PATCH 3/4 — `lib/tts_arabic/vocalizer/models/core.py`
- **قبل:** `import gdown` استيراد علوي (يجرّ معه requests/beautifulsoup4...).
- **بعد:** الاستيراد نُقل داخل فرع التنزيل في `get_model_path()` —
  وهو فرع لا يُنفَّذ أبدًا لأن `data/catt_eo.onnx` مضمّن في الحزمة.
- **ملاحظة:** `packaging.version` بقيت كما هي (مكتبة ميكروسكوبية
  مضمّنة في requirements).

### PATCH 4/4 — `lib/tts_arabic/models/core.py`
- نفس فكرة PATCH 3/4 لملف الشقيق الخاص بنماذج TTS (غير مستخدم في
  حزمة الاستدلال هذه لكنه يُستورد عبر `tts_arabic/__init__.py`).

### PATCH 5/5 (تجميلي) — `lib/tts_arabic/text/phonetise_buckwalter.py`
- **قبل:** `re.sub("(\S)(\.|\?|,|!)", "\\1 \\2", utterance)` — يطبع
  SyntaxWarning على Python ≥ 3.12 (تسلسل هروب غير معروف في نص عادي).
- **بعد:** الشكل raw-string المكافئ تمامًا:
  `re.sub(r"(\S)(\.|\?|,|!)", r"\1 \2", utterance)`.
- **السبب:** إزالة تحذير مطبوع على stderr في كل تشغيل. الدلالة
  الرياضية للـregex متطابقة حرفيًا (تم التحقق بتطابق التوكنز قبل/بعد).

---

## ما لم يُضمَّ من المستودع الأصلي (ولم يُحذف أي شيء يستخدمه التوليد)

- `mixer_repo/utils/` و`mixer_repo/models/common/loss.py` و
  `modules/data_function.py`: ملفات مسار التدريب فقط (الميل،
  BetaBinomialInterpolator، خسائر GAN). استيرادها يجرّ librosa/scipy
  إلزاميًا، ولا يستوردها أي ملف في سلسلة التوليد.
- أوزان `mixer128_pytorch.pth` (نموذج المؤلف الأصلي): مخرجات
  المقارنة A/B فقط — ليست جزءًا من التوليد من checkpoint الخاص بك.

---

## إثبات عدم تأثر السلوك الحسابي

تمت الموازاة على نفس الجهاز ونفس الـcheckpoint (`states_79590.pth`)
بين مسار كود النواة الأصلي (نسخة scratch_data غير المعدلة) وبين هذه
الحزمة، لنفس الجملة والمتحدث:

| القياس | النتيجة |
|---|---|
| توكنز الإدخال (ids) | **متطابقة تمامًا** |
| ميل النموذج (80×T) | **متطابقة bit-by-bit** (فرق أقصى = 0.0) |
| WAV النهائي (بعد vocos + التطبيع + PCM_16) | **متطابق bit-by-bit** (ارتباط 1.000000) |

وبين مخرج هذه الحزمة على CPU وعينة النواة النهائية المولّدة على GPU
(نفس الـcheckpoint 79590، نفس الجملة egy_00): طول متطابق (38400 عينة)
وارتباط موجي **1.000**.

كما أعيد إنتاج نفس المخرج في بيئة نظيفة (venv جديد، torch 2.5.1+cpu
بدل 2.14) بفارق أقصى 6×10⁻⁵ (فروق أنوية conv بين إصدارات torch — دون
مستوى الإدراك السمعي).
