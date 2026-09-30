// probe_usage.mjs — tiny probe: is the API quota available, and does the
// completion object expose `usage`? (needed for hard token accounting in v5.2)
import ZAI from 'z-ai-web-dev-sdk';

const GUIDE = (await import('node:fs')).default.readFileSync(
  '/home/z/my-project/work/tashkeel_guide_v2.md', 'utf8');
console.log('guide chars:', GUIDE.length);

const zai = await ZAI.create();
const t0 = Date.now();
try {
  const c = await zai.chat.completions.create({
    messages: [
      { role: 'system', content: 'أنت مساعد تشكيل نصوص عربية. أعد الجملة مشكولة فقط.' },
      { role: 'user', content: 'شكّل: الرجل ده جاي' },
    ],
    thinking: { type: 'disabled' },
  });
  console.log('OK in', Date.now() - t0, 'ms');
  console.log('content:', JSON.stringify(c.choices?.[0]?.message?.content));
  console.log('usage:', JSON.stringify(c.usage ?? null));
  console.log('model:', c.model ?? null);
  console.log('keys:', Object.keys(c).join(','));
} catch (e) {
  console.log('FAILED in', Date.now() - t0, 'ms');
  console.log('error:', String(e.message).slice(0, 300));
}
