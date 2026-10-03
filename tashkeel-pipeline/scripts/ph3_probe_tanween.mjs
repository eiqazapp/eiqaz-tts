import fs from 'node:fs';
import assert from 'node:assert';
const TANWEEN_ADVERBS = new Set(
  JSON.parse(fs.readFileSync('/home/z/my-project/work/tanween_closed_list.json', 'utf8')).words);

// EXACT v5.1 engine tanweenCheck
const STRIP = /[\u064B-\u0652\u0653-\u0655\u0670]/g;
function tanweenCheck(out) {
  if (/[\u064C\u064D]/.test(out)) return { ok: false, reason: 'damm/kasr' };
  for (const w of out.split(/\s+/)) {
    if (w.includes('\u064B')) {
      let skel = w.replace(STRIP, '').replace(/[^\u0621-\u064A\u0640]/g, '');
      if (!TANWEEN_ADVERBS.has(skel)
          && (skel[0] === 'و' || skel[0] === 'ف') && skel.length > 3
          && TANWEEN_ADVERBS.has(skel.slice(1))) skel = skel.slice(1);
      if (!TANWEEN_ADVERBS.has(skel))
        return { ok: false, reason: `outside: ${w} skel=${skel}` };
    }
  }
  return { ok: true };
}

// 1) the three real v5 failures must now PASS
const d = JSON.parse(fs.readFileSync('/home/z/my-project/work/ph3_diag_out_diag.json', 'utf8'));
let n = 0;
for (const it of d.diag) {
  if (!it.reason || !it.reason.includes('تنوين')) continue;
  const r = tanweenCheck(it.outAfterRestore);
  console.log(it.id, '->', JSON.stringify(r));
  assert.ok(r.ok, 'real case should pass after fix'); n++;
}
console.log(`real v5-failure cases now passing: ${n}/${n}`);

// 2) synthetic matrix
const cases = [
  ['وَعَمِلْ فِعْلَاً كُوَيِّسْ', true,  'فعلا in list'],
  ['عَمِلْ فَوْرَاً', true,  'فورا in list'],
  ['وَعَمِلْ وِفِعْلَاً', true,  'وفعلا in list directly'],
  ['فَأَوَّلَاً نِتْكَلِّمْ', true,  'ف + أولا (prefix path)'],
  ['وِجِدَاً بِيْتَكَلِّمْ', true,  'و + جدا (prefix path)'],
  ['شَرِبَ مَايَاً كَتِيرْ', false, 'مايا not adverb'],
  ['قَبُوْلَاً بِسْ', false, 'قبولا not in list (smoke precedent)'],
  ['فَشَرِبَ مَايَاً كَتِيرْ', false, 'ف + مايا still rejected'],
];
for (const [txt, exp, label] of cases) {
  const r = tanweenCheck(txt);
  console.log((r.ok === exp ? 'PASS' : 'FAIL'), label, '->', JSON.stringify(r.reason ?? 'ok'));
  assert.equal(r.ok, exp, label);
}
console.log('ALL TANWEEN PROBES PASS (v5.1)');
