import ZAI from 'z-ai-web-dev-sdk';
const t0 = Date.now();
try {
  const zai = await ZAI.create();
  const c = await zai.chat.completions.create({
    messages: [{ role: 'user', content: 'رد بكلمة واحدة: تمام' }],
    thinking: { type: 'disabled' },
  });
  console.log('OK', `${((Date.now() - t0) / 1000).toFixed(1)}s`,
    JSON.stringify(c.choices?.[0]?.message?.content).slice(0, 40));
} catch (e) {
  console.log('FAIL', `${((Date.now() - t0) / 1000).toFixed(1)}s`,
    String(e.message).slice(0, 120));
}
