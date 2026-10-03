import ZAI from 'z-ai-web-dev-sdk';
const zai = await ZAI.create();
// test 1: does max_tokens get respected? (tiny limit -> visible truncation)
try {
  const c1 = await zai.chat.completions.create({
    messages: [{ role: 'user', content: 'اكتب من 1 إلى 30، رقمًا في كل سطر' }],
    max_tokens: 25,
    thinking: { type: 'disabled' },
  });
  const t1 = c1.choices?.[0]?.message?.content || '';
  console.log('max_tokens=25 -> lines:', t1.split('\n').filter(Boolean).length, '| finish:', c1.choices?.[0]?.finish_reason ?? c1.choices?.[0]?.message?.finish_reason ?? JSON.stringify(c1.choices?.[0]).slice(0,120));
} catch (e) { console.log('max_tokens param REJECTED:', String(e.message).slice(0,150)); }
// test 2: default cap estimation
try {
  const c2 = await zai.chat.completions.create({
    messages: [{ role: 'user', content: 'اكتب من 1 إلى 400، رقمًا في كل سطر، لا شيء غيرها' }],
    thinking: { type: 'disabled' },
  });
  const t2 = c2.choices?.[0]?.message?.content || '';
  console.log('default -> lines:', t2.split('\n').filter(Boolean).length, '| last:', JSON.stringify(t2.slice(-30)));
} catch (e) { console.log('default probe fail:', String(e.message).slice(0,100)); }
