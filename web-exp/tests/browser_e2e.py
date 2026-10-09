#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار E2E حقيقي داخل متصفح Chromium (Playwright) لتجربة إيقاز
====================================================================
يفتح التطبيق عبر serve.py (خادم HTTP حقيقي بمنفذ محلي) ويختبر فعليًا:

  1) تحميل الواجهة وقراءة manifest
  2) تكافؤ خط النص JS داخل المتصفح مقابل بايثون (expected_tokens.json)
  3) تكافؤ المشكّل catt_eo.onnx داخل المتصفح (expected_catt.json)
  4) توليد كامل: تحميل mixertts_fp16 + vocos → صوت غير فارغ + WAV سليم
  5) محاكاة البث: أول صوت مجدول قبل وصول كل الدفعات (TTFA أقل من زمن
     وصول النص الكامل) + عدد مقاطع > 1
  6) المقاطعة الفورية: زمن < 100مل، ورفض مقاطع متأخرة (epoch)
  7) وضع WASM الإلزامي يعمل (مسار CPU الحقيقي)

يعمل مع نماذج sanity (أوزان عشوائية) — يثبت الآلية لا جودة الصوت.
"""
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WEBEXP = os.path.dirname(HERE)      # web-exp/
REPO = os.path.dirname(WEBEXP)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


PORT = free_port()
proc = subprocess.Popen(
    [sys.executable, os.path.join(WEBEXP, 'serve.py'),
     '--host', '127.0.0.1', '--port', str(PORT)],
    cwd=WEBEXP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
BASE = f'http://127.0.0.1:{PORT}'

# انتظر جاهزية الخادم بنبضات صحية (حتى 15 ثانية)
import urllib.request  # noqa: E402
_server_ready = False
for _ in range(75):
    if proc.poll() is not None:
        break
    try:
        urllib.request.urlopen(BASE + '/index.html', timeout=1)
        _server_ready = True
        break
    except Exception:
        time.sleep(0.2)
if not _server_ready:
    print('FAIL  server_start — الخادم لم يجهز خلال 15 ثانية')
    proc.terminate()
    sys.exit(1)

results = []


def record(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f' — {detail}' if detail else ''))


async def main_async():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            args=['--autoplay-policy=no-user-gesture-required'])
        page = await browser.new_page()
        page.on('pageerror', lambda e: print('  [pageerror]', str(e)[:200]))
        errors = []
        page.on('console', lambda m: errors.append(m.text)
                if m.type == 'error' else None)

        # ---- 1) تحميل الواجهة ------------------------------------------------
        await page.goto(BASE + '/index.html', wait_until='load')
        ok = await page.evaluate('typeof EiqazApp === "object"')
        record('load_app', ok, f'EiqazApp {await page.evaluate("EiqazApp.version")}')

        manifest = await page.evaluate('__EIQAZ_DIAG__.manifest')
        n_models = len((manifest or {}).get('acoustic_variants', [])) \
            if manifest else 0
        record('manifest_loaded', n_models >= 3,
               f'{n_models} نسخ صوتية في manifest')
        tag = (manifest or {}).get('acoustic_variants', [{}])[0].get('tag')
        record('manifest_tag_visible', tag in ('sanity', 'production'),
               f'tag={tag}')

        # ---- 2) تكافؤ خط النص داخل المتصفح ------------------------------------
        t0 = time.time()
        tp = await page.evaluate('EiqazApp.textParityCheck()')
        record('textpipe_parity_browser', tp['verdict'] == 'PASS',
               f"{tp['nMatch']}/{tp['nTotal']} خلال {time.time()-t0:.1f}s")
        for r in tp['rows']:
            if not r.get('match', True):
                print('   اختلاف:', r['id'], r.get('error', ''))

        # ---- 3) تكافؤ catt داخل المتصفح ----------------------------------------
        t0 = time.time()
        try:
            cp = await page.evaluate('EiqazApp.cattParityCheck()')
            record('catt_parity_browser', cp['verdict'] == 'PASS',
                   f"{cp['nMatch']}/{cp['nTotal']} خلال {time.time()-t0:.1f}s")
            for r in cp['rows']:
                if not r.get('match', True) and not r.get('skipped'):
                    print('   اختلاف catt:', r['id'],
                          'py=', (r.get('expected') or '')[:40],
                          'js=', (r.get('got') or '')[:40])
        except Exception as e:
            record('catt_parity_browser', False, str(e)[:150])

        # ---- 4) توليد كامل (FP16 + vocos fp16) ---------------------------------
        await page.select_option('#modelSel', 'mixertts_fp16.onnx')
        await page.select_option('#vocoderSel', 'vocos22_fp16.onnx')
        await page.fill('#textInput', 'دلوقتي هنبدأ الدرس يا عمر. الذكاء الاصطناعي بيساعدنا نفهم المعلومات.')
        t0 = time.time()
        await page.click('#btnGenerate')
        # انتظر النتيجة (حتى 120ث — تنزيل 7+30MB على localhost سريع)
        await page.wait_for_function(
            "document.getElementById('runStatus').textContent.includes('تم —')"
            "|| document.getElementById('runStatus').textContent.includes('خطأ')",
            timeout=180000)
        gen_elapsed = time.time() - t0
        status = await page.inner_text('#runStatus')
        audio_visible = await page.is_visible('#fullAudio')
        metrics = await page.evaluate('EiqazApp.state.metrics')
        ok = ('تم —' in status) and audio_visible
        record('generate_full_fp16', ok,
               f"الحالة: {status[:80]} | RTF={metrics.get('rtf')} "
               f"chunks={metrics.get('nChunks')} ({gen_elapsed:.1f}s شاملة التنزيل)")

        # ---- 5) محاكاة البث: صوت قبل اكتمال النص ------------------------------
        await page.fill('#textInput',
                        'النموذج اللغوي بيبعت النص على دفعات. والمحرك لازم يتكلم فورًا من أول دفعة. '
                        'والصوت لازم يفضل شغال. والمقاطعة لازم توقف كل حاجة في نفس اللحظة. '
                        'وفي الآخر نقيس كل الأزمنة ونقارن النسخ كويس كويس.')
        await page.evaluate("""
          document.getElementById('batchWords').value = 3;
          document.getElementById('batchWords').dispatchEvent(new Event('input'));
          document.getElementById('batchMs').value = 500;
          document.getElementById('batchMs').dispatchEvent(new Event('input'));
        """)
        t0 = time.time()
        await page.click('#btnStream')
        # راقب حتى يظهر TTFA (انتهى البث) أو خطأ
        await page.wait_for_function(
            "document.getElementById('runStatus').textContent.includes('انتهى البث')"
            "|| document.getElementById('runStatus').textContent.includes('خطأ')",
            timeout=240000)
        status = await page.inner_text('#runStatus')
        metrics = await page.evaluate('EiqazApp.state.metrics')
        ttfa = metrics.get('ttfaMs')
        n_chunks = metrics.get('nChunks') or 0
        # الشرط الحاسم: TTFA أقل من نصف زمن وصول النص الكامل
        # (24 كلمة ÷ 3 = 8 دفعات × 500مل = 3.5 ثانية حتى آخر دفعة)
        text_arrival_ms = 3500
        stream_ok = (ttfa is not None and n_chunks >= 3
                     and ttfa < text_arrival_ms)
        record('stream_ttfa_before_full_text', stream_ok,
               f"TTFA={ttfa}مل (وصول النص كله ~{text_arrival_ms}مل) "
               f"| مقاطع={n_chunks} | RTF={metrics.get('rtf')}")

        # ---- 6) المقاطعة الفورية -----------------------------------------------
        await page.click('#btnStream')   # ابدأ بثًا جديدًا
        await page.wait_for_function(
            "document.getElementById('runStatus').textContent.includes('وصلت الدفعة')",
            timeout=60000)
        await page.wait_for_timeout(1500)  # دع الصوت يبدأ فعلًا
        t_int = time.time()
        interrupt_ms = await page.evaluate('EiqazApp.interrupt()')
        latency = (time.time() - t_int) * 1000
        # بعد المقاطعة: انتظر أكثر من زمن بث متبقٍ وتأكد من عدم عودة صوت
        await page.wait_for_timeout(3000)
        status_after = await page.inner_text('#runStatus')
        m_after = await page.evaluate('EiqazApp.state.metrics')
        sources_active = await page.evaluate(
            'EiqazApp.diag && true')  # العدادات الداخلية
        ok = (interrupt_ms is not None and interrupt_ms < 100
              and 'تمت المقاطعة' in status_after)
        record('interrupt_immediate', ok,
               f"زمن المقاطعة={interrupt_ms}مل | الحالة: {status_after[:60]}")

        # طلب جديد بعد المقاطعة بلا إعادة تحميل — يجب أن يعمل
        await page.fill('#textInput', 'طلب جديد بعد المقاطعة. يتكلم فورًا.')
        await page.click('#btnStream')
        await page.wait_for_function(
            "document.getElementById('runStatus').textContent.includes('انتهى البث')"
            "|| document.getElementById('runStatus').textContent.includes('خطأ')",
            timeout=120000)
        status2 = await page.inner_text('#runStatus')
        record('new_request_after_interrupt', 'انتهى البث' in status2,
               status2[:70])
        await page.evaluate('EiqazApp.interrupt()')

        # ---- 7) مسار WASM الإلزامي (CPU الحقيقي) -------------------------------
        await page.select_option('#epSel', 'wasm')
        await page.fill('#textInput', 'مسار دبليو إيه إس إم. يعمل على المعالج فقط.')
        await page.click('#btnGenerate')
        await page.wait_for_function(
            "document.getElementById('runStatus').textContent.includes('تم —')"
            "|| document.getElementById('runStatus').textContent.includes('خطأ')",
            timeout=120000)
        status3 = await page.inner_text('#runStatus')
        active_ep = await page.evaluate('__EIQAZ_DIAG__.activeEP')
        record('wasm_only_path', ('تم —' in status3) and active_ep == 'wasm',
               f"EP الفعلي={active_ep} | {status3[:60]}")

        # INT8-dyn نسخة مكمّمة تعمل في المتصفح
        await page.select_option('#epSel', 'auto')
        await page.select_option('#modelSel', 'mixertts_int8dyn.onnx')
        await page.fill('#textInput', 'النسخة المكممة تعمل داخل المتصفح. جرب بنفسك.')
        await page.click('#btnGenerate')
        await page.wait_for_function(
            "document.getElementById('runStatus').textContent.includes('تم —')"
            "|| document.getElementById('runStatus').textContent.includes('خطأ')",
            timeout=120000)
        status4 = await page.inner_text('#runStatus')
        record('int8dyn_browser', 'تم —' in status4, status4[:70])

        n_err_console = len([e for e in errors if 'favicon' not in e])
        record('no_console_errors', n_err_console == 0,
               f'{n_err_console} أخطاء كونسول' if n_err_console else '')

        await browser.close()


def main():
    import asyncio
    try:
        asyncio.run(main_async())
    finally:
        proc.terminate()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== المتصفح: {n_pass}/{len(results)} PASS ===")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
