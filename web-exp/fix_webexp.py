#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_webexp.py (v2) — ترقيع تجربة إيقاز داخل المتصفح (web-exp)
====================================================================
الأخطاء الأربعة المعالجة:

1) البث المباشر يعمل مع النقطة "." فقط:
   دفعات النص تُقصّ عند النقطة، فنصٌ بفواصل "،" فقط يتحول إلى دفعة واحدة
   ضخمة → لا بث. الترقيع يجعل حدود الدفعات كل علامات الترقيم:
   . ، ؛ ؟ ! , ; ? …  + السطر الجديد
   (لا يشمل ":" لأنها داخل الجملة عادةً مثل "قال: يا عمر")

2) لا يخرج أي صوت على متصفح الهاتف (كروم أندرويد):
   كروم الهاتف يبدأ AudioContext معلقًا (suspended) حتى أول إيماءة
   مستخدم، والتطبيق ينشئه أثناء تحميل الصفحة قبل أي لمسة فيبقى معلقًا
   للأبد. الترقيع يلفّ منشئ AudioContext ويسجل كل السياقات، وأول نقرة/
   لمسة على الصفحة تستأنفها فورًا (إدراج في index.html قبل app.js
   + نسخة احتياطية نهاية app.js).

3) تحميل النموذج على الهاتف بطيء جدًا والعدّاد يهبط ويصعد:
   الخادم أحادي الخيط → المتصفح ينزل الملفات الكبيرة واحدًا تلو الآخر
   (ومعها يتعطل تقدم العدّاد بين الملفات)، ولا رؤوس تخزين مؤقت → كل
   زيارة تعيد تنزيل مئات الميغابايت من جديد. الترقيع:
   - HTTPServer/TCPServer → ThreadingHTTPServer/ThreadingTCPServer
     (تنزيل متوازٍ + استجابة لأكثر من جهاز في نفس اللحظة)
   - رؤوس Cache-Control: النماذج/WASM/vendor لمدة يوم كامل،
     وapp.js/HTML بلا كاش حتى تصلك أي تحديثات لاحقة فورًا.
   ملاحظة: أول زيارة على الهاتف تظل بطيئة بحسب سرعة الواي فاي والنسخة
   المختارة (اختر int8static على الهاتف) — لكن كل زيارة بعدها فورية
   من الكاش، والعدّاد لن يتذبذب بين الملفات المتوازية.

4) بانر serve.py لا يُطبع فور التشغيل في Git Bash:
   mintty يشغّل بايثون عبر أنبوب لا طرف، فيتحول الإخراج إلى تخزين كتليّ،
   ولا يظهر البانر إلا عند أول طلب من المتصفح. الترقيع يفعّل الطباعة
   السطرية الفورية (reconfigure line_buffering).

(+) إسكات 404 favicon.ico برد 204.

سياسة الأمان:
   - نسخة احتياطية لكل ملف قبل تعديله: <الملف>.iqzbak_<الوقت>
   - يطبع فرقًا (diff) لكل تغيير
   - لا يلمس إطلاقًا: textpipe.js / catt.js / vendor/ / models/
     (خط نطق الإنتاج المرجعي)
   - إعادة التشغيل آمنة: يتخطى ما رُقّع مسبقًا (حتى لو رُقّع بالإصدار v1)

التشغيل (ضع الملف داخل مجلد web-exp بجوار serve.py):
    python fix_webexp.py
أو من أي مكان:
    python fix_webexp.py "D:/Eiqaz Core/eiqaz-tts/web-exp"

بعدها:
    1) أعد تشغيل الخادم: python serve.py   → البانر يظهر فورًا
    2) على الحاسوب: Ctrl+F5 مرة واحدة (تجاوز الكاش القديم لapp.js)
    3) على الهاتف: أعد تحميل الصفحة مرتين — الأولى تنزيل عادي إلى الكاش
       وكل زيارة بعدها شبه فورية
    4) جرّب نص الفواصل (يُطبع أسفل المخرجات) → الصوت يبدأ من أول دفعة
"""

import codecs
import datetime
import difflib
import os
import re
import shutil
import sys

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------
# حدود الدفعات الجديدة: كل علامات الترقيم عدا ":" (داخل الجملة)
# ---------------------------------------------------------------
PUNCT_CHARS = ".!?,;،؛؟…"
PUNCT_JS = PUNCT_CHARS + "\\n"   # داخل فئة JS مع \n

HELPERS_JS = (
    "/* ===== [EIQAZ-PATCH] حدود الدفعات: كل علامات الترقيم (لا النقطة فقط) =====\n"
    "   أُضيفت آليًا نهاية الملف — تصريحات دوال مرفوعة فلا مشكلة ترتيب */\n"
    "function IQZ_HAS_PUNCT(s) {\n"
    "  return /[" + PUNCT_JS + "]/.test(String(s == null ? '' : s));\n"
    "}\n"
    "function IQZ_LAST_PUNCT(s) {\n"
    "  s = String(s == null ? '' : s);\n"
    "  for (var i = s.length - 1; i >= 0; i--) {\n"
    "    if (IQZ_HAS_PUNCT(s.charAt(i))) return i;\n"
    "  }\n"
    "  return -1;\n"
    "}\n"
    "function IQZ_ENDS_PUNCT(s) {\n"
    "  s = String(s == null ? '' : s);\n"
    "  return s.length > 0 && IQZ_HAS_PUNCT(s.charAt(s.length - 1));\n"
    "}\n"
    "/* ===== نهاية [EIQAZ-PATCH] ===== */\n"
)

# ---------------------------------------------------------------
# فتح قفل الصوت على الهاتف — يُدرج في index.html (المسار الأساسي،
# قبل تحميل app.js) وتنسخ نسخة احتياطية نهاية app.js (حارس runtime
# window.__IQZ_AUDIO_UNLOCK__ يمنع التثبيت المزدوج)
# ---------------------------------------------------------------
AUDIO_UNLOCK_JS = (
    "/* ===== [EIQAZ-AUDIO-UNLOCK] فتح قفل الصوت على متصفح الهاتف =====\n"
    "   كروم الهاتف يبدأ AudioContext معلقًا (suspended) حتى أول إيماءة\n"
    "   مستخدم، والتطبيق ينشئه أثناء تحميل الصفحة قبل أي لمسة فيبقى\n"
    "   معلقًا إلى الأبد: لا يخرج أي صوت (تحويل كامل أو بث).\n"
    "   المعالجة: (1) تلفّ منشئ AudioContext فتُسجَّل كل سياقات جديدة\n"
    "             (2) تُسجَّل السياقات القديمة عند أول createBufferSource\n"
    "             (3) أول نقرة أو لمسة على الصفحة تستأنف كل السياقات */\n"
    "(function () {\n"
    "  if (window.__IQZ_AUDIO_UNLOCK__) return;\n"
    "  window.__IQZ_AUDIO_UNLOCK__ = 1;\n"
    "  var EVENTS = ['pointerdown', 'touchstart', 'touchend', 'mousedown', 'click', 'keydown'];\n"
    "  var contexts = [];\n"
    "  function resumeAll() {\n"
    "    for (var i = 0; i < contexts.length; i++) {\n"
    "      try {\n"
    "        var c = contexts[i];\n"
    "        if (c && c.state === 'suspended') {\n"
    "          var p = c.resume();\n"
    "          if (p && p.catch) p.catch(function () {});\n"
    "        }\n"
    "      } catch (e) {}\n"
    "    }\n"
    "  }\n"
    "  for (var i = 0; i < EVENTS.length; i++) {\n"
    "    document.addEventListener(EVENTS[i], resumeAll, { capture: true, passive: true });\n"
    "  }\n"
    "  function track(ctx) {\n"
    "    try {\n"
    "      if (ctx && contexts.indexOf(ctx) === -1) contexts.push(ctx);\n"
    "    } catch (e) {}\n"
    "  }\n"
    "  function wrapCtor(name) {\n"
    "    var Original = window[name];\n"
    "    if (typeof Original !== 'function') return;\n"
    "    try {\n"
    "      var Wrapped = class extends Original {\n"
    "        constructor() {\n"
    "          super(...arguments);\n"
    "          track(this);\n"
    "          resumeAll();\n"
    "        }\n"
    "      };\n"
    "      try { Object.defineProperty(Wrapped, 'name', { value: name }); } catch (e) {}\n"
    "      window[name] = Wrapped;\n"
    "    } catch (e) {}\n"
    "  }\n"
    "  wrapCtor('AudioContext');\n"
    "  wrapCtor('webkitAudioContext');\n"
    "  try {\n"
    "    var BAC = window.BaseAudioContext;\n"
    "    if (BAC && BAC.prototype && BAC.prototype.createBufferSource) {\n"
    "      var origCBS = BAC.prototype.createBufferSource;\n"
    "      BAC.prototype.createBufferSource = function () {\n"
    "        track(this);\n"
    "        return origCBS.apply(this, arguments);\n"
    "      };\n"
    "    }\n"
    "  } catch (e) {}\n"
    "})();\n"
)

# ================================================================
# أدوات قراءة/كتابة محافظة على الترميز ونهايات الأسطر
# ================================================================

def smart_read(path):
    with open(path, "rb") as f:
        raw = f.read()
    bom = raw.startswith(codecs.BOM_UTF8)
    encodings = (["utf-8-sig"] if bom else []) + ["utf-8", "cp1256", "latin-1"]
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise SystemExit("[خطأ] تعذر فك ترميز: " + path)


def smart_write(path, text, enc):
    with open(path, "w", encoding=enc, newline="") as f:
        f.write(text)


def newlines_of(text):
    return "\r\n" if "\r\n" in text else "\n"


def print_diff(name, old, new, limit=160):
    diff = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=name + " (قبل)", tofile=name + " (بعد)", lineterm=""))
    if not diff:
        print("   (لا فرق)")
        return
    for ln in diff[:limit]:
        print("   " + ln)
    if len(diff) > limit:
        print("   ... + %d سطر فرق إضافي" % (len(diff) - limit))


def _append_block(src, block, nl):
    """إلحاق كتلة JS نهاية الملف مع الحفاظ على نهايات الأسطر"""
    return src.rstrip("\r\n \t") + nl + nl + block.replace("\n", nl) + nl

# ================================================================
# app.js — (أ) توسيع حدود الدفعات  (ب) فتح قفل الصوت (احتياط)
# ================================================================

ALLOWED_CLASS_CHARS = set(".!?;\\sn")


def extend_js_class(cls):
    """توسيع فئة حروف JS بعلامات الترقيم الناقصة.
    تعيد الفئة الجديدة أو None إذا كانت غير جملية (مثل [0-9.]) أو موسعة سابقًا."""
    body = cls[1:] if cls.startswith("^") else cls
    if any(c in body for c in "،؛؟"):
        return None                      # علامات عربية موجودة — لا شيء لنفعله
    compact = body.replace("\\", "")
    if not set(compact) <= ALLOWED_CLASS_CHARS:
        return None                      # فئة غير جملية (أرقام/حروف) — لا تُمس
    if "." not in compact:
        return None                      # لا علاقة لها بالنقطة
    add = "".join(c for c in PUNCT_CHARS if c not in compact)
    if "\\n" not in body and "n" not in compact and "\\s" not in body:
        add += "\\n"
    return ("^" if cls.startswith("^") else "") + body + add


Q = "['\"]"

RE_SPLIT_STR = re.compile(r"\.split\(\s*(" + Q + r")\.[ ]*\1\s*\)")
RE_SPLIT_CLS = re.compile(r"(\.split\(\s*)/\[([^\]\n]*)\](\+|\*|\{\d+(?:,\d*)?\})?([^/\n]*)/([a-zA-Z]*)\s*\)")
RE_SPLIT_ESC = re.compile(r"(\.split\(\s*)/\\\.(\+|\*)?([^/\n]*)/([a-zA-Z]*)\s*\)")
RE_INCLUDES = re.compile(r"([A-Za-z_$][\w$.]*)\.includes\(\s*(" + Q + r")\.[ ]*\2\s*\)")
RE_LASTIDX = re.compile(r"([A-Za-z_$][\w$.]*)\.lastIndexOf\(\s*(" + Q + r")\.[ ]*\2\s*\)")
RE_ENDSWITH = re.compile(r"([A-Za-z_$][\w$.]*)\.endsWith\(\s*(" + Q + r")\.[ ]*\2\s*\)")
RE_IDX_EQ_M1 = re.compile(r"([A-Za-z_$][\w$.]*)\.indexOf\(\s*(" + Q + r")\.[ ]*\2\s*\)\s*={2,3}\s*-1")
RE_IDX_NE_M1 = re.compile(r"([A-Za-z_$][\w$.]*)\.indexOf\(\s*(" + Q + r")\.[ ]*\2\s*\)\s*!==?\s*-1")
RE_IDX_GT_M1 = re.compile(r"([A-Za-z_$][\w$.]*)\.indexOf\(\s*(" + Q + r")\.[ ]*\2\s*\)\s*>\s*-1")
RE_IDX_GE_0 = re.compile(r"([A-Za-z_$][\w$.]*)\.indexOf\(\s*(" + Q + r")\.[ ]*\2\s*\)\s*>=\s*0")
RE_CH_EQ = re.compile(r"([A-Za-z_$][\w$.]*)\s*={2,3}\s*(" + Q + r")\.\2")
RE_CH_NE = re.compile(r"([A-Za-z_$][\w$.]*)\s*!==?\s*(" + Q + r")\.\2")
RE_MATCH_TWO = re.compile(r"(\.match(?:All)?\(\s*)/\[([^\]\n]*)\](\+|\*)\[([^\]\n]*)\](\+|\*)?([^/\n]*)/([a-zA-Z]*)\s*\)")
RE_MATCH_ONE = re.compile(r"(\.match(?:All)?\(\s*)/\[([^\]\n]*)\](\+|\*)?([^/\n]*)/([a-zA-Z]*)\s*\)")
RE_MATCH_ESC = re.compile(r"(\.match(?:All)?\(\s*)/\[([^\]\n]*)\](\+|\*)\\\.(\+|\*|\?)?([^/\n]*)/([a-zA-Z]*)\s*\)")

# ---- أنماط web-exp الفعلية (المستودع): lookbehind + اختبار الاكتمال ----
RE_LB_SPLIT = re.compile(r"(\.split\(\s*)/\(\?<=\[([^\]\n]*)\]\)([^/\n]*)/([a-zA-Z]*)\s*\)")
RE_TEST_END = re.compile(r"/\[([^\]\n]*)\](\\s\*\$)/\.test\(")

# ---- أنماط web-exp الفعلية: تسخين الصوت (نصوص حرفية من المستودع) ----
RE_GETCTX_TAIL = re.compile(
    r"([ \t]*if \(audioCtx\.state === 'suspended'\) audioCtx\.resume\(\);\r?\n"
    r"[ \t]*return audioCtx;\r?\n[ \t]*\})")
CLICK_GEN_OLD = "els.btnGenerate.addEventListener('click', () => generateFull());"
CLICK_STR_OLD = "els.btnStream.addEventListener('click', () => startStream());"
CLICK_REP_OLD = "els.btnReplay.addEventListener('click', () => replay());"
PLAY_OLD = "els.fullAudio.play().catch(() => { });"

WARMUP_FN_JS = (
    "\n"
    "  // [EIQAZ-AUDIO-WARMUP] فتح قفل صوت الهاتف: كروم الجوال يبدأ AudioContext\n"
    "  // معلقًا (suspended) حتى إيماءة مستخدم، وصلاحية الإيماءة تنتهي أثناء\n"
    "  // انتظار التوليد الطويل — فاستئناف getCtx() بعد الأوامر غير المتزامنة\n"
    "  // يُحجب ويبقى الصوت صامتًا للأبد. الحل: إنشاء/استئناف السياق متزامنًا\n"
    "  // داخل حدث النقر نفسه قبل أي await (نستدعيها من مستمعي الأزرار).\n"
    "  function warmupAudio() {\n"
    "    try { getCtx(); } catch (e) { }\n"
    "  }\n"
)

SENT_END_CHARS = set(".!?؟…")
SENT_PAUSE_CHARS = "،؛,;"

RE_REPORT = re.compile(
    r"\b(split|match|matchAll|exec|includes|indexOf|lastIndexOf|endsWith|"
    r"startsWith|search|replace|charCodeAt)\s*\(")


def _extend_sentence_class(cls):
    """توسيع فئة حدود الجملة الحقيقية [.!؟?…] بالفواصل ،؛,;
    (تعيد None إن لم تكن فئة نهايات جمل أو كانت موسعة سابقًا)"""
    body = cls[1:] if cls.startswith("^") else cls
    compact = body.replace("\\", "")
    have = set(compact)
    if not (have & SENT_END_CHARS):
        return None                     # ليست فئة نهايات جمل
    if not have <= (SENT_END_CHARS | set(SENT_PAUSE_CHARS) | set("\\sn")):
        return None                     # أحرف غير متوقعة — لا تُمس
    missing = "".join(c for c in SENT_PAUSE_CHARS if c not in have)
    if not missing:
        return None                     # موسعة سابقًا (idempotent)
    return body + missing


def _lb_split_repl(m):
    new = _extend_sentence_class(m.group(2))
    if new is None:
        return m.group(0)
    tail = m.group(3) or "\\s+"
    flags = m.group(4) or ""
    return m.group(1) + "/(?<=[" + new + "])" + tail + "/" + flags + ")"


def _test_end_repl(m):
    new = _extend_sentence_class(m.group(1))
    if new is None:
        return m.group(0)
    return "/[" + new + "]" + m.group(2) + "/.test("


def _split_str_repl(m):
    return ".split(/[" + PUNCT_JS + "]+\\s*/)"


def _split_cls_repl(m):
    new = extend_js_class(m.group(2))
    if new is None:
        return m.group(0)
    q = m.group(3) or "+"
    tail = m.group(4) or "\\s*"
    flags = m.group(5) or ""
    return m.group(1) + "/[" + new + "]" + q + tail + "/" + flags + ")"


def _split_esc_repl(m):
    tail = m.group(3) or "\\s*"
    flags = m.group(4) or ""
    return m.group(1) + "/[" + PUNCT_JS + "]+" + tail + "/" + flags + ")"


def _match_two_repl(m):
    pre, c1, q1, c2, q2, tail, flags = m.groups()
    n1, n2 = extend_js_class(c1), extend_js_class(c2)
    if n1 is None and n2 is None:
        return m.group(0)
    n1 = n1 if n1 is not None else c1
    n2 = n2 if n2 is not None else c2
    return (pre + "/[" + n1 + "]" + q1 + "[" + n2 + "]" + (q2 or "") +
            (tail or "") + "/" + (flags or "") + ")")


def _match_one_repl(m):
    pre, c, q, tail, flags = m.groups()
    n = extend_js_class(c)
    if n is None:
        return m.group(0)
    return pre + "/[" + n + "]" + (q or "") + (tail or "") + "/" + (flags or "") + ")"


def _match_esc_repl(m):
    pre, c, q, qdot, tail, flags = m.groups()
    n = extend_js_class(c)
    if n is None:
        return m.group(0)
    return (pre + "/[" + n + "]" + (q or "") + "[" + PUNCT_JS + "]" + (qdot or "") +
            (tail or "") + "/" + (flags or "") + ")")


def _has(m):
    return "IQZ_HAS_PUNCT(" + m.group(1) + ")"


def _not_has(m):
    return "!IQZ_HAS_PUNCT(" + m.group(1) + ")"


def _last(m):
    return "IQZ_LAST_PUNCT(" + m.group(1) + ")"


def _ends(m):
    return "IQZ_ENDS_PUNCT(" + m.group(1) + ")"


def _apply_regexes(src, log, rules):
    """تطبيق قائمة قواعد (وصف، نمط، بديل) مع عدّ المواقع المتغيرة فعليًا"""
    n_total = 0
    for desc, pattern, repl in rules:
        cnt = [0]

        def fn(m):
            out = repl(m)
            if out != m.group(0):
                cnt[0] += 1
            return out

        src = pattern.sub(fn, src)
        if cnt[0]:
            log.append("[OK] " + desc + " — " + str(cnt[0]) + " موضع")
            n_total += cnt[0]
    return src, n_total


def _patch_batches(src, log):
    """قسم حدود الدفعات — يعيد (النص، هل تغيّر شيء)"""
    state = {"src": src, "n": 0}

    def apply(desc, pattern, repl):
        count = [0]

        def fn(m):
            out = repl(m)
            if out != m.group(0):
                count[0] += 1
            return out

        state["src"] = pattern.sub(fn, state["src"])
        if count[0]:
            log.append("[OK] " + desc + " — " + str(count[0]) + " موضع")
            state["n"] += count[0]

    apply("split على نص النقطة '.'", RE_SPLIT_STR, _split_str_repl)
    apply("split على فئة [.]", RE_SPLIT_CLS, _split_cls_repl)
    apply("split على \\.", RE_SPLIT_ESC, _split_esc_repl)
    apply("match بفئتين [^..][..]", RE_MATCH_TWO, _match_two_repl)
    apply("match بفئة واحدة", RE_MATCH_ONE, _match_one_repl)
    apply("match مع \\.", RE_MATCH_ESC, _match_esc_repl)
    apply("endsWith('.')", RE_ENDSWITH, _ends)
    apply("lastIndexOf('.')", RE_LASTIDX, _last)
    apply("includes('.')", RE_INCLUDES, _has)
    apply("indexOf('.') === -1", RE_IDX_EQ_M1, _not_has)
    apply("indexOf('.') !== -1", RE_IDX_NE_M1, _has)
    apply("indexOf('.') > -1", RE_IDX_GT_M1, _has)
    apply("indexOf('.') >= 0", RE_IDX_GE_0, _has)
    apply("مقارنة حرف === '.'", RE_CH_EQ, _has)
    apply("مقارنة حرف !== '.'", RE_CH_NE, _not_has)
    apply("قصّ lookbehind عند حدود الجمل (،؛,; تُضاف)", RE_LB_SPLIT, _lb_split_repl)
    apply("اختبار اكتمال الجملة .test([..]$)", RE_TEST_END, _test_end_repl)

    out = state["src"]
    if state["n"] == 0:
        log.append("[تنبيه] لم أجد نمط تقسيم معروفًا في app.js — راجع القائمة أدناه")
        return out, False

    if ("IQZ_HAS_PUNCT(" in out or "IQZ_LAST_PUNCT(" in out
            or "IQZ_ENDS_PUNCT(" in out):
        out = _append_block(out, HELPERS_JS, newlines_of(out))
        log.append("[OK] دوال المساعدة IQZ_* أُلحقت نهاية app.js")
    return out, True


def _patch_warmup(src, log, nl):
    """تسخين سياق الصوت داخل النقر (أنماط web-exp الفعلية حرفيًا)"""
    if "EIQAZ-AUDIO-WARMUP" in src:
        log.append("[تخطي] تسخين الصوت موجود مسبقًا في app.js")
        return src, False
    changed = False

    m = RE_GETCTX_TAIL.search(src)
    fn_ok = bool(m) or "function warmupAudio" in src
    if m:
        src = src[:m.end()] + WARMUP_FN_JS.replace("\n", nl) + src[m.end():]
        log.append("[OK] warmupAudio() أُدرجت بعد getCtx()")
        changed = True
    elif fn_ok:
        pass
    else:
        log.append("[معلومة] لم أجد نمط getCtx — لن يُضاف تسخين الأزرار "
                   "(الإصلاح الكامل عبر سحب المستودع)")

    if fn_ok:
        pairs = [
            (CLICK_GEN_OLD,
             "els.btnGenerate.addEventListener('click', "
             "() => { warmupAudio(); generateFull(); });"),
            (CLICK_STR_OLD,
             "els.btnStream.addEventListener('click', "
             "() => { warmupAudio(); startStream(); });"),
            (CLICK_REP_OLD,
             "els.btnReplay.addEventListener('click', "
             "() => { warmupAudio(); replay(); });"),
            (PLAY_OLD,
             "els.fullAudio.play().catch(() => {"
             "\n        try { scheduleWave(final, myEpoch); } catch (e) { }"
             "\n      });"),
        ]
        n_w = 0
        for old, new in pairs:
            if old in src:
                src = src.replace(old, new.replace("\n", nl), 1)
                n_w += 1
        if n_w:
            log.append("[OK] تسخين الصوت مربوط بـ%d من مستمعي الأزرار/التشغيل"
                       % n_w)
            changed = True
    return src, changed


def patch_app_js(src):
    """يعيد (النص الجديد، سجل التغييرات، هل تغيّر شيء)"""
    log = []
    out = src
    changed = False
    nl = newlines_of(src)

    # ---------- (أ) حدود الدفعات: كل علامات الترقيم ----------
    if "IQZ_HAS_PUNCT" in out:
        log.append("[تخطي] حدود الدفعات مرقّعة مسبقًا (IQZ_* موجودة)")
    else:
        out, ch = _patch_batches(out, log)
        changed = changed or ch

    # ---------- (ب) تسخين الصوت داخل النقر (أنماط المستودع) ----------
    out, ch = _patch_warmup(out, log, nl)
    changed = changed or ch

    # ---------- (ج) فتح قفل الصوت (نسخة احتياطية نهاية app.js) ----------
    # المسار الأساسي هو الإدراج في index.html قبل app.js؛ هذه النسخة
    # احتياط للصفحات التي تعذّر إدراجها هناك. حارس window.__IQZ_AUDIO_UNLOCK__
    # في وقت التشغيل يمنع أي تثبيت مزدوج مهما وُجدت النسختان.
    if "__IQZ_AUDIO_UNLOCK__" in out:
        log.append("[تخطي] فتح قفل الصوت موجود مسبقًا في app.js")
    else:
        out = _append_block(out, AUDIO_UNLOCK_JS, nl)
        log.append("[OK] [EIQAZ-AUDIO-UNLOCK] أُلحق نهاية app.js (نسخة احتياطية)")
        changed = True

    return out, log, changed

# ================================================================
# serve.py — طباعة فورية + favicon + خيوط متعددة + رؤوس كاش
# ================================================================

def _insert_index(lines):
    """أين ندرج إصلاح الطباعة: بعد shebang/التعليقات/Docstring/from __future__"""
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s == "" or s.startswith("#"):
            i += 1
        else:
            break
    if i < n:
        s = lines[i].lstrip()
        for q in ('"""', "'''"):
            if s.startswith(q):
                if s.count(q) >= 2 and len(s) > 3:
                    i += 1
                else:
                    i += 1
                    while i < n and q not in lines[i]:
                        i += 1
                    i += 1
                break
    while i < n:
        s = lines[i].strip()
        if s.startswith("from __future__") or s == "" or s.startswith("#"):
            i += 1
        else:
            break
    return i


def _patch_line_buffer(src, log, nl):
    """[1] طباعة سطرية فورية للبانر"""
    if "IQZ_LINE_BUFFER" in src:
        log.append("[تخطي] إصلاح الطباعة موجود مسبقًا")
        return src, False
    lines = src.splitlines(True)
    stripped = [l.rstrip("\r\n") for l in lines]
    i = _insert_index(stripped)
    fix = nl.join([
        "# --- [IQZ_LINE_BUFFER] طباعة فورية: إصلاح تأخر البانر في Git Bash ---",
        "import sys as _iqz_sys",
        "for _iqz_s in (_iqz_sys.stdout, _iqz_sys.stderr):",
        "    try:",
        "        _iqz_s.reconfigure(line_buffering=True)",
        "    except Exception:",
        "        pass",
        "# --- نهاية [IQZ_LINE_BUFFER] ---",
        "",
        "",
    ])
    lines.insert(i, fix)
    log.append("[OK] طباعة سطرية فورية أُدرجت (بعد السطر %d)" % i)
    return "".join(lines), True


def _patch_reconfigure(src, log):
    """[1-ب] إضافة line_buffering إلى reconfigure موجود (serve.py الفعلي)"""
    if "line_buffering" in src:
        return src, False
    if "reconfigure(" not in src:
        return src, False
    cnt = [0]

    def fn(m):
        args = m.group(1).rstrip()
        if "encoding" not in args:
            return m.group(0)
        cnt[0] += 1
        return "reconfigure(" + args + ", line_buffering=True)"

    src = re.sub(r"reconfigure\(([^()\n]*)\)", fn, src)
    if cnt[0]:
        log.append("[OK] line_buffering=True أُضيفت إلى reconfigure الموجود "
                   "(بانر فوري في Git Bash)")
        return src, True
    return src, False


def _patch_favicon(src, log, nl):
    """[2] رد 204 على favicon.ico لإسكات 404"""
    if "favicon" in src:
        log.append("[تخطي] معالجة favicon موجودة مسبقًا")
        return src, False
    m = re.search(r"^([ \t]*)def do_GET\s*\([^)]*\)\s*:", src, re.M)
    if m:
        after = src[m.end():]
        mm = re.search(r"^([ \t]+)\S", after, re.M)
        indent = mm.group(1) if mm else m.group(1) + "    "
        # فحص favicon فقط في بداية do_GET — جسم الدالة الأصلي يواصل العمل
        stub = (indent + "if self.path.split('?', 1)[0] == '/favicon.ico':" + nl +
                indent + "    self.send_response(204)" + nl +
                indent + "    self.end_headers()" + nl +
                indent + "    return" + nl)
        le = src.find(nl, m.end())
        if le == -1:
            src = src + nl + stub
        else:
            at = le + len(nl)
            src = src[:at] + stub + src[at:]
        log.append("[OK] favicon.ico يُرد بـ204 (إسكات 404 — مع فحص معامِلات GET)")
        return src, True
    # لا يوجد do_GET — نضيف دالة كاملة داخل صنف المعالج
    m_cls = re.search(
        r"^class\s+\w+\s*\(\s*[^)\n]*(?:SimpleHTTPRequestHandler"
        r"|BaseHTTPRequestHandler)[^)\n]*\s*\)\s*:", src, re.M)
    if m_cls:
        after = src[m_cls.end():]
        mm = re.search(r"^([ \t]+)\S", after, re.M)
        ind = mm.group(1) if mm else "    "
        method = nl.join([
            "",
            ind + "def do_GET(self):",
            ind + "    # [IQZ_FAVICON] إسكات 404 favicon.ico (يطلبه المتصفح تلقائيًا)",
            ind + "    if self.path.split('?', 1)[0] == '/favicon.ico':",
            ind + "        self.send_response(204)",
            ind + "        self.end_headers()",
            ind + "        return",
            ind + "    return super().do_GET()",
        ])
        le = src.find(nl, m_cls.end())
        at = (le + len(nl)) if le != -1 else len(src)
        src = src[:at] + method + nl + src[at:]
        log.append("[OK] favicon.ico يُرد بـ204 (دالة do_GET جديدة في المعالج)")
        return src, True
    log.append("[معلومة] لم أجد do_GET ولا صنف معالج — تعذر إسكات favicon")
    return src, False


def _patch_threading(src, log):
    """[3] خادم متعدد الخيوط: تنزيل متوازٍ للنماذج على الهاتف"""
    if any(k in src for k in ("ThreadingHTTPServer", "ThreadingTCPServer",
                              "ThreadingMixIn")):
        log.append("[تخطي] الخادم متعدد الخيوط موجود مسبقًا")
        return src, False
    n_http = len(re.findall(r"\bHTTPServer\b", src))
    if n_http:
        # يعالج معًا: from http.server import HTTPServer  و  http.server.HTTPServer(...)
        # و  HTTPServer((host, port), ...)  و  class X(HTTPServer)
        src = re.sub(r"\bHTTPServer\b", "ThreadingHTTPServer", src)
        log.append("[OK] HTTPServer → ThreadingHTTPServer (%d موضعًا) — "
                   "تنزيل متوازٍ + أكثر من جهاز في نفس اللحظة" % n_http)
        return src, True
    n_tcp = len(re.findall(r"\bTCPServer\b", src))
    if n_tcp and ("SimpleHTTPRequestHandler" in src or "BaseHTTPRequestHandler" in src):
        src = re.sub(r"\bTCPServer\b", "ThreadingTCPServer", src)
        log.append("[OK] TCPServer → ThreadingTCPServer (%d موضعًا) — تنزيل متوازٍ" % n_tcp)
        return src, True
    log.append("[تنبيه] لم أجد HTTPServer/TCPServer — ربما بنية خادم مختلفة "
               "(لن تتأثر بقية الإصلاحات)")
    return src, False


def _cache_body(indent):
    """أسطر رؤوس الكاش بمسافة بادئة معطاة (indent = مسافة جسم الدالة)"""
    return [
        indent + "# --- [IQZ_CACHE] تخزين مؤقت للملفات الثقيلة (نماذج/WASM/vendor) ---",
        indent + "# app.js و HTML يبقيان no-cache حتى تصلك أي تحديثات لاحقة فورًا",
        indent + "try:",
        indent + "    _p = self.path.split('?', 1)[0]",
        indent + "    _heavy = (_p.lower().endswith(('.onnx', '.wasm', '.mjs', '.woff2', '.data'))",
        indent + "              or _p.startswith('/vendor/') or _p.startswith('/models/'))",
        indent + "    self.send_header('Cache-Control',",
        indent + "                     'public, max-age=86400' if _heavy else 'no-cache')",
        indent + "except Exception:",
        indent + "    pass",
    ]


def _patch_cache(src, log, nl):
    """[4] رؤوس Cache-Control: الثقيل يومًا كاملًا، وapp.js/HTML بلا كاش
    حالة 0: يوجد Cache-Control: no-store (serve.py الفعلي) → استبدال ذكي
    حالة أ: لا يوجد end_headers → نضيف دالة جديدة قبل do_GET
    حالة ب: end_headers موجود أصلًا (مثل رؤوس COOP/COEP) → ندرج أسطرنا
            في بداية جسمها بدل التخطي"""
    if "IQZ_CACHE" in src:
        log.append("[تخطي] رؤوس التخزين المؤقت موجودة مسبقًا")
        return src, False

    # ---- حالة 0: استبدال no-store (كان يعيد تنزيل النماذج كل زيارة) ----
    m_ns = re.search(
        r"(?:^[ \t]*#[^\n]*تخزين مؤقت[^\n]*\r?\n)?"
        r"^([ \t]*)self\.send_header\(\s*(['\"])Cache-Control\2\s*,\s*"
        r"(['\"])no-store\3\s*\)[ \t]*\r?\n",
        src, re.M)
    if m_ns:
        indent = m_ns.group(1)
        smart = nl.join(_cache_body(indent))
        src = src[:m_ns.start()] + smart + nl + src[m_ns.end():]
        log.append("[OK] no-store استُبدل بكاش ذكي: نماذج/WASM/vendor يومًا، "
                   "app.js/HTML بلا كاش (كان يعيد تنزيل كل شيء كل زيارة)")
        return src, True

    m_eh = re.search(r"^([ \t]*)def end_headers\s*\(\s*(self)?\s*\)\s*:", src, re.M)
    if m_eh:
        d_indent = m_eh.group(1)
        after = src[m_eh.end():]
        mm = re.search(r"^([ \t]+)\S", after, re.M)
        body_indent = mm.group(1) if mm else d_indent + "    "
        body = nl.join(_cache_body(body_indent))
        le = src.find(nl, m_eh.end())
        if le == -1:
            src = src + nl + body + nl
        else:
            at = le + len(nl)
            src = src[:at] + body + nl + nl + src[at:]
        log.append("[OK] رؤوس Cache-Control أُدرجت داخل end_headers الموجود "
                   "(بجوار رؤوس COOP/COEP إن كانت هناك)")
        return src, True

    m = re.search(r"^([ \t]*)def do_GET\s*\([^)]*\)\s*:", src, re.M)
    if not m:
        log.append("[تنبيه] لم أجد do_GET/end_headers — تعذر إضافة رؤوس الكاش")
        return src, False
    indent = m.group(1)
    method = nl.join(
        [indent + "def end_headers(self):"] + _cache_body(indent + "    ") +
        [indent + "    super().end_headers()", ""])
    # m.start() يقع عند بداية مسافة بادئة def do_GET نفسها → نغلق بـ nl
    src = src[:m.start()] + method + nl + src[m.start():]
    log.append("[OK] رؤوس Cache-Control: نماذج/WASM/vendor لمدة يوم، app.js/HTML بلا كاش")
    return src, True


def patch_serve_py(src):
    """يعيد (النص الجديد، سجل التغييرات، هل تغيّر شيء)"""
    original = src
    log = []
    changed = False
    nl = newlines_of(src)

    # [1-أ] إن وُجد reconfigure موجود نضيف line_buffering إليه مباشرة،
    # وإلا أدرجنا كتلة IQZ_LINE_BUFFER منفصلة (المسار العام)
    src, ch = _patch_reconfigure(src, log)
    changed = changed or ch
    if not ch:
        src, ch = _patch_line_buffer(src, log, nl)
        changed = changed or ch
    src, ch = _patch_favicon(src, log, nl)
    changed = changed or ch
    src, ch = _patch_threading(src, log)
    changed = changed or ch
    src, ch = _patch_cache(src, log, nl)
    changed = changed or ch

    # ---- شبكة أمان: لا نكتب serve.py أبدًا إن أنتج الترقيع كودًا مكسورًا ----
    if changed:
        try:
            compile(src, "<serve.py>", "exec")
        except SyntaxError as e:
            log.append("[خطأ-سلامة] الترقيع أنتج كودًا غير صالح (سطر %s: %s) — "
                       "أُعيد الملف الأصلي دون أي تعديل" % (e.lineno, e.msg))
            return original, log, False

    for kw, why in [
        ("hashlib", "قد يحسب بصمات الملفات عند الإقلاع → بطء بدء التشغيل"),
        ("getfqdn", "استعلام DNS قد يعلّق البدء عدة ثوانٍ"),
        ("gethostbyname", "تحليل اسم المضيف قد يبطئ البدء"),
    ]:
        if kw in src:
            log.append("[تنبيه-أداء] " + kw + ": " + why)
    return src, log, changed


# ================================================================
# index.html — حقن فتح قفل الصوت قبل أول سكربت
# ================================================================

def patch_index_html(src):
    """يعيد (النص الجديد، سجل التغييرات، هل تغيّر شيء)"""
    log = []
    if "__IQZ_AUDIO_UNLOCK__" in src:
        log.append("[تخطي] فتح قفل الصوت موجود مسبقًا في index.html")
        return src, log, False
    nl = newlines_of(src)
    block = ("<script>" + nl + AUDIO_UNLOCK_JS.replace("\n", nl) + nl +
             "</script>" + nl)

    m = re.search(r"[ \t]*<script\b", src, re.I)
    at, where = (m.start(), "قبل أول وسم <script>") if m else (None, None)
    if at is None:
        m = re.search(r"</head\s*>", src, re.I)
        at, where = (m.start(), "قبل </head>") if m else (None, None)
    if at is None:
        m = re.search(r"</body\s*>", src, re.I)
        at, where = (m.start(), "قبل </body>") if m else (None, None)
    if at is None:
        at, where = len(src.rstrip()), "نهاية الملف"

    src = src[:at] + block + src[at:]
    log.append("[OK] سكربت [EIQAZ-AUDIO-UNLOCK] أُدرج في index.html (%s) — "
               "قبل تحميل app.js (المسار الأساسي لفتح صوت الهاتف)" % where)
    return src, log, True


# ================================================================
# العرض والتنفيذ
# ================================================================

def _report_lines(src, limit=30):
    n = 0
    for i, line in enumerate(src.splitlines(), 1):
        if RE_REPORT.search(line):
            print("   %4d| %s" % (i, line.strip()[:120]))
            n += 1
            if n >= limit:
                print("   ... (اقتُصر العرض على %d سطر)" % limit)
                break
    if n == 0:
        print("   (لا أسطر مطابقة)")


def _apply(target, name, patcher):
    """قراءة + ترقيع + نسخة احتياطية + كتابة + فرق. يعيد True إن تغيّر شيء"""
    src, enc = smart_read(target)
    new, log, changed = patcher(src)
    for l in log:
        print("   " + l)
    if not changed:
        return src, new, False
    bak = target + ".iqzbak_" + STAMP
    shutil.copy2(target, bak)
    smart_write(target, new, enc)
    print("   [نسخة احتياطية] " + os.path.basename(bak))
    print("   ---- الفرق ----")
    print_diff(name, src, new)
    return src, new, True


def main():
    target = (os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
              else os.path.dirname(os.path.abspath(__file__)))
    print("=" * 64)
    print(" ترقيع تجربة الويب (web-exp) v2")
    print(" 1) حدود الدفعات: كل علامات الترقيم   2) فتح قفل صوت الهاتف")
    print(" 3) خادم متعدد الخيوط + رؤوس كاش       4) بانر فوري + إسكات favicon")
    print(" المجلد الهدف: " + target)
    print("=" * 64)

    app_js = os.path.join(target, "app.js")
    serve_py = os.path.join(target, "serve.py")
    index_html = os.path.join(target, "index.html")

    missing = [p for p in (app_js, serve_py) if not os.path.exists(p)]
    if missing:
        print("[خطأ] لم أجد: " + ", ".join(os.path.basename(p) for p in missing))
        print("       ضع fix_webexp.py داخل مجلد web-exp (بجوار serve.py) ثم أعد")
        print("       التشغيل، أو مرّر المسار:")
        print('       python fix_webexp.py "D:/Eiqaz Core/eiqaz-tts/web-exp"')
        sys.exit(1)

    # ---------- [1/4] app.js ----------
    print("\n[1/4] app.js — (أ) حدود الدفعات عند كل علامات الترقيم")
    print("                (ب) نسخة احتياطية لفتح قفل الصوت")
    _, app_new, app_changed = _apply(app_js, "app.js", patch_app_js)
    if app_changed:
        print("   ---- أسطر التقسيم/البوابات النهائية في app.js ----")
        _report_lines(app_new)
    else:
        print("   ---- أسطر التقسيم/البوابات الحالية في app.js ----")
        _report_lines(app_new)
        if "IQZ_HAS_PUNCT" not in app_new:
            print("   >> إن ظل البث لا يعمل بعد التجربة: انسخ الأسطر أعلاه وأرسلها لي")

    # ---------- [2/4] serve.py ----------
    print("\n[2/4] serve.py — بانر فوري + خيوط متعددة + رؤوس كاش + favicon")
    _, _, _ = _apply(serve_py, "serve.py", patch_serve_py)

    # ---------- [3/4] index.html ----------
    if os.path.exists(index_html):
        print("\n[3/4] index.html — حقن فتح قفل صوت الهاتف قبل أول سكربت")
        _, _, _ = _apply(index_html, "index.html", patch_index_html)
        h, _ = smart_read(index_html)
        hits = sum(1 for line in h.splitlines()
                   if RE_REPORT.search(line) and "__IQZ" not in line
                   and "AUDIO_UNLOCK" not in line)
        if hits:
            print("   [معلومة] وُجد %d سطر منطق تقسيم/بحث داخل index.html (لم يُمسّ)" % hits)
    else:
        print("\n[3/4] index.html غير موجود — الاعتماد على نسخة app.js الاحتياطية")

    # ---------- [4/4] الخلاصة ----------
    print("\n" + "=" * 64)
    print("ملفات غير ممسوسة عمدًا (خط إنتاج النطق المرجعي):")
    print("  textpipe.js, catt.js, vendor/, models/")
    print("=" * 64)
    print("الخطوات التالية على جهازك:")
    print("  1) أعد تشغيل الخادم:   python serve.py")
    print("     → البانر يجب أن يظهر فورًا (بدون انتظار فتح المتصفح)")
    print("  2) على الحاسوب: Ctrl+F5 مرة واحدة فقط (تجاوز الكاش القديم لapp.js؛")
    print("     بعد هذا الترقيع تصلك التحديثات القادمة بلا Ctrl+F5)")
    print("  3) على الهاتف: أعد تحميل الصفحة مرتين —")
    print("     المرة الأولى تنزيل عادي إلى الكاش، وكل زيارة بعدها شبه فورية")
    print("     وأول نقرة/لمسة على الصفحة تفتح قفل الصوت تلقائيًا")
    print("  4) لسرعة أعلى على الهاتف: اختر mixertts_int8static من قائمة النسخ")
    print("     (التحويل على WASM أحادي الخيط أسرع بكثير مع النسخة الأصغر)")
    print("  5) WebGPU/الخيوط المتعددة على الهاتف تتطلب https — إن كان خادمك")
    print("     يدعمها شغّل: python serve.py --https  وافتح رابط https الظاهر")
    print()
    print("  6) جرّب نص الفواصل (البث كان لا يعمل معه):")
    print('     "النموذج اللغوي بيبعت النص على دفعات، والمحرك لازم يتكلم')
    print('      فورًا من أول دفعة، والصوت لازم يفضل شغال بدون انتظار،')
    print('      والمقاطعة لازم توقف كل حاجة في نفس اللحظة، وفي الآخر')
    print('      نقيس كل الأزمنة ونقارن النسخ."')
    print("     المتوقع: 5 دفعات (قصّ عند كل ،) — الصوت يبدأ بعد الدفعة")
    print("     الأولى «...على دفعات،» وليس بعد النص كاملًا.")
    print()
    print("ملاحظات:")
    print("  - إن ظهر البانر فورًا لكن البدء نفسه ظل بطيئًا جرّب: python -u serve.py")
    print("    وأرسل لي نتيجته (عندها السبب شيء آخر مثل DNS أو بصمات الملفات).")
    print("  - إن لم يجد السكربت نمط التقسيم في app.js فستظهر لك أسطر مرقّمة —")
    print("    انسخها وأرسلها لي لأرقّعها يدويًا بدقة.")
    print("  - العدّاد في أول زيارة على الهاتف يظل بطيئًا بحسب الواي فاي؛ تذبذبه")
    print("    بين الملفات يجب أن يختفي مع التنزيل المتوازي (خيوط متعددة).")


if __name__ == "__main__":
    main()
