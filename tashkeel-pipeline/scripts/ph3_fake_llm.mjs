// ph3_fake_llm.mjs — TEST-ONLY fake LLM for offline dry tests of
// ph3_engine_v52.mjs (never used in production; the engine activates it
// only when env PH3_FAKE_LLM points at this file).
// Plan file (env PH3_FAKE_PLAN): JSON array of response strings, consumed
// in call order. An entry starting with "THROW:" makes the call fail with
// the remainder as the error message (simulates API errors).
// Every call is appended to the log file (env PH3_FAKE_LOG) as one JSON
// line: {"call":N,"messages":[...]} for post-run assertions.
import fs from 'node:fs';

const plan = JSON.parse(fs.readFileSync(process.env.PH3_FAKE_PLAN, 'utf8'));
const logPath = process.env.PH3_FAKE_LOG;
let n = 0;

export async function fakeLLM(messages) {
  const idx = n++;
  fs.appendFileSync(logPath, JSON.stringify({ call: idx + 1, messages }) + '\n');
  if (idx >= plan.length) throw new Error(`fake plan exhausted at call ${idx + 1}`);
  const entry = plan[idx];
  if (typeof entry === 'string' && entry.startsWith('THROW:')) {
    throw new Error(entry.slice(6));
  }
  return { choices: [{ message: { content: entry } }], model: 'fake-test' };
}
