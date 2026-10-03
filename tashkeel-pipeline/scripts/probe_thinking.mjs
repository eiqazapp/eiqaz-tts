import ZAI from 'z-ai-web-dev-sdk';
const zai = await ZAI.create();
const cases = [
  'قبل أي حاجة تانية. نمسك ورقة و الم و نسأل نفسنا شوية أسئلة أساسية.',
  'أهمها طبعاً, زي ما الكل عارف, القرب من الشغل و من الأهل.',
];
const sys = 'أنت خبير تشكيل النصوص المصرية. أضف التشكيل الكامل (فتحة/ضمة/كسرة/سكون/شدة) كما تُنطق بالمصرية: كل «ال» أول كلمة = اِلْ بكسرة تحت الألف. انقل كل حرف ومسافة وعلامة ترقيم حرفيًا (الفاصلة اللاتينية , تبقى لاتينية). لا تنوين إلا على الظروف المنطوقة (جدا فعلا طبعا...). ة بلا حركة. أعد السطر المشكول فقط بلا أي شرح.';
for (const c of cases) {
  const completion = await zai.chat.completions.create({
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: 'شكّل هذا السطر تشكيلًا كاملًا (تحويل وليس نسخًا):\n\n' + c },
    ],
    thinking: { type: 'enabled' },
  });
  console.log('IN :', c);
  console.log('OUT:', completion.choices?.[0]?.message?.content?.trim());
  console.log('---');
}
