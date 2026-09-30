ضع ملفات الـcheckpoint هنا (checkpoints)
=========================================

انسخ من مخرجات نواة التدريب على Kaggle (Output) الملفات التي تريد
التوليد منها، وأهمها:

    states_79590.pth      <- نقطة النهاية (iter 79590) — النموذج الرباعي-ساعاتي

أي ملف من الصيغتين يعمل:
    states_79590.pth   (snapshot — نموذج فقط)
    states.pth         (rolling — يحوي أيضًا حالات المُحسِّن، والسكريبت
                        يقرأ منه جزء "model" فقط كما هو)

بعد وضع الملف هنا يمكنك التشغيل بلا تحديد مسار:
    python infer.py --text "..." --out out.wav
(يُختار أحدث states_*.pth تلقائيًا)

أو بتحديد صريح:
    python infer.py --checkpoint states_79590.pth --text "..." --out out.wav
