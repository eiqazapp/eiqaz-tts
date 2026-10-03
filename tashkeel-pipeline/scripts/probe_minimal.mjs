// probe_minimal.mjs — minimal-prompt fallback test for stubborn units
import ZAI from 'z-ai-web-dev-sdk';
const zai = await ZAI.create();
const t = 'قبل أي حاجة تانية. نمسك ورقة و الم و نسأل نفسنا شوية أسئلة أساسية.';
const MIN_SYS = `شكّل النص المصري تشكيلًا كاملًا كما يُنطق (قاهري): اِلْ التعريف بكسرة تحت الألف، السكون في نهايات الكلمات، لا إعراب ولا تنوين إلا للظروف المنطوقة (جدا فعلا طبعا)، ة بلا حركة، بيقول/بتقول/هيروح بالأسلوب المصري.
انسخ كل حرف ومسافة وترقيم كما هو حرفيًا وأضف الحركات فقط. أعد السطر المشكول فقط بلا شرح.`;
const completion = await zai.chat.completions.create({
  messages: [
    { role: 'system', content: MIN_SYS },
    { role: 'user', content: `شكّل هذا السطر (تحويل كامل وليس نسخًا):\n\n${t}` },
  ],
  thinking: { type: 'disabled' },
});
console.log('OUT:', completion.choices?.[0]?.message?.content?.trim());
