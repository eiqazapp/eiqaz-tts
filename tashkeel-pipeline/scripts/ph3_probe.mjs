// ph3_probe.mjs — diagnose why the engine echo'd instead of diacritizing.
// Tests variants on 2 sentences, prints raw completions.
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'node:fs';

const GUIDE = fs.readFileSync('/home/z/my-project/work/tashkeel_guide_v1.md', 'utf8');
const SENT = [
  'النقطة دي بالزات بتفسر نجاح شراكات',
  'التسرع ده ممكن يخليك تشتري بسعر أغلى من المفروض, أو في مكان مش مناسب ليك,',
];

const CONTRACT = `
---

## عقد الإخراج (إلزامي — يُنفَّذ حرفيًا)

- ستستقبل جملًا مرقّمة (1. 2. 3. ...). أعد كل جملة مشكولةً بالكامل كما تُنطق
  بالمصرية، بنفس الترقيم الرقمي وبنفس الترتيب — سطر واحد لكل جملة، ولا شيء غيرها.
- ممنوع: أي شرح أو مقدمة أو خاتمة أو علامات تنصيص أو تنسيق ماركداون — أسطر
  مرقّمة فقط.
- انسخ كل حرف ومسافة ورقم ورمز ترقيم كما ورد حرفيًا وأضِف الحركات فقط وفق الدليل أعلاه.
- التنوين: فقط على الظروف المنصوصة في قائمة §ب المغلقة.`;

const FEWSHOT = `أمثلة مرجعية للمطلوب (خام ← مشكول):
- خام: انه بقى بيزنس مودل ناجح ومش بس بيوفر فرص عمل للشباب
- مشكول: اِنُّهْ بَقَى بِيزْنِسْ مُودِلْ نَاجِحْ وِمُشْ بَسْ بِيُوَفِّرْ فُرَصْ عَمَلْ لِلشَّبَابْ

- خام: هي دي فكرة اوية جدا.
- مشكول: هِيَ دِي فِكْرَة اوِّيَة جِدَاً.
`;

const VARIANTS = {
  v1_system_role: {
    messages: (u) => [
      { role: 'system', content: GUIDE + CONTRACT },
      { role: 'user', content: u },
    ],
  },
  v2_assistant_role_strong: {
    messages: (u) => [
      { role: 'assistant', content: GUIDE + CONTRACT },
      { role: 'user', content: u },
    ],
  },
  v3_fewshot: {
    messages: (u) => [
      { role: 'assistant', content: FEWSHOT + GUIDE + CONTRACT },
      { role: 'user', content: u },
    ],
  },
};

const userMsg = (sents) => `المهمة: إضافة التشكيل (الفتحة/الضمة/الكسرة/السكون/الشدة) إلى الجمل التالية كما تُنطق بالمصرية — هذا تحويل وليس نسخًا للنص.

شكّل الجمل التالية بنفس ترقيمها وترتيبها (سطر مرقّم لكل جملة، بلا أي شرح):

${sents.map((s, i) => `${i + 1}. ${s}`).join('\n')}`;

const zai = await ZAI.create();
for (const [name, v] of Object.entries(VARIANTS)) {
  console.log(`\n########## ${name} ##########`);
  try {
    const completion = await zai.chat.completions.create({
      messages: v.messages(userMsg(SENT)),
      thinking: { type: 'disabled' },
    });
    console.log(completion.choices?.[0]?.message?.content);
  } catch (e) { console.log('ERROR:', e.message); }
}
