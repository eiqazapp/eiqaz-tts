/**
 * app.js — محرك تجربة إيقاز داخل المتصفح (ONNX Runtime Web)
 * ===========================================================================
 * المسار: نص → [تشكيل catt اختياري] → خط النص (textpipe.js — مكافئ بايثون
 * المثبت باختبار تكافؤ) → توكنز → MixerTTS.onnx → mel → Vocos.onnx → موجة
 * → تشغيل تدريجي (Web Audio) مع طابور ومقاطعة فورية حقيقية (epoch-based).
 *
 * وضع «توليد كامل» يطابق webapp.py الإنتاجي: تقسيم تلقائي + توحيد نبرة
 * (RMS وسيط + تلاشي حواف + فاصل 220ms + تطبيع ذروة واحد).
 * وضع «محاكاة البث» يشتغل بالمثل لكن سببيًا (الوسيط الجاري) لأن المستقبل
 * مجهول أثناء البث — الفرق موثق في التقرير.
 */
(function () {
  'use strict';

  const APP_VERSION = '1.0.0';
  const SAMPLE_RATE = 22050;
  const CHUNK_GAP_S = 0.22;
  const EDGE_FADE_S = 0.008;

  // مراجع DOM
  const $ = (id) => document.getElementById(id);
  const els = {
    modelSel: $('modelSel'), vocoderSel: $('vocoderSel'), epSel: $('epSel'),
    speakerSel: $('speakerSel'), dialectSel: $('dialectSel'),
    paceRange: $('paceRange'), paceVal: $('paceVal'),
    diacritizeSel: $('diacritizeSel'),
    modelHint: $('modelHint'), vocoderHint: $('vocoderHint'), epHint: $('epHint'),
    sanityWarn: $('sanityWarn'), loadModelStatus: $('loadModelStatus'),
    textInput: $('textInput'), runStatus: $('runStatus'), metrics: $('metrics'),
    fullAudio: $('fullAudio'), llmBox: $('llmBox'), chunkList: $('chunkList'),
    diagPanel: $('diagPanel'), compatPanel: $('compatPanel'),
    btnGenerate: $('btnGenerate'), btnStream: $('btnStream'),
    btnInterrupt: $('btnInterrupt'), btnStop: $('btnStop'),
    btnReplay: $('btnReplay'),
    batchWords: $('batchWords'), batchMs: $('batchMs'),
    batchWordsVal: $('batchWordsVal'), batchMsVal: $('batchMsVal'),
    presetShort: $('presetShort'), presetLong: $('presetLong'),
    presetQaf: $('presetQaf'), presetMSA: $('presetMSA'),
    verBadge: $('verBadge'),
  };
  els.verBadge.textContent = 'v' + APP_VERSION;

  // ========================================================================
  // الحالة العامة (معروضة للتشخيص والاختبارات)
  // ========================================================================
  const diag = {
    appVersion: APP_VERSION,
    ortVersion: null,
    manifest: null,
    manifestError: null,
    models: {},
    requestedEP: null,
    activeEP: null,
    epFallback: null,
    webgpuAdapter: null,
    crossOriginIsolated: typeof crossOriginIsolated !== 'undefined'
      ? crossOriginIsolated : false,
    lastError: null,
    sessions: {},
  };
  globalThis.__EIQAZ_DIAG__ = diag;

  const state = {
    epoch: 0,                 // عداد الطلبات — أساس المقاطعة
    mode: null,               // 'full' | 'stream'
    busy: false,
    chunks: [],               // مقاطع البث المولدة {text, wave, buffer, ...}
    metrics: {},
  };

  // ========================================================================
  // أدوات
  // ========================================================================
  function fmtBytes(n) {
    if (n > 1e6) return (n / 1e6).toFixed(1) + ' MB';
    if (n > 1e3) return (n / 1e3).toFixed(0) + ' KB';
    return n + ' B';
  }
  function setStatus(msg, cls) {
    els.runStatus.textContent = msg;
    els.runStatus.className = 'status' + (cls ? ' ' + cls : '');
  }
  function setLoadStatus(msg, cls) {
    els.loadModelStatus.textContent = msg;
    els.loadModelStatus.className = 'status' + (cls ? ' ' + cls : '');
  }
  function renderMetrics(m) {
    state.metrics = m;
    const rows = [
      ['TTFA', m.ttfaMs != null ? (m.ttfaMs / 1000).toFixed(2) + ' ث' : '—'],
      ['زمن التوليد', m.genS != null ? m.genS.toFixed(2) + ' ث' : '—'],
      ['مدة الصوت', m.audioS != null ? m.audioS.toFixed(1) + ' ث' : '—'],
      ['RTF', m.rtf != null ? m.rtf.toFixed(3) : '—'],
      ['مقاطع', m.nChunks != null ? String(m.nChunks) : '—'],
      ['توكنز', m.nTokens != null ? String(m.nTokens) : '—'],
      ['زمن المقاطعة', m.interruptMs != null ? m.interruptMs + ' م‌ل' : '—'],
      ['أخطاء', String(m.errors || 0)],
    ];
    els.metrics.innerHTML = rows.map(([k, v]) =>
      `<div class="metric"><div class="v">${v}</div><div class="k">${k}</div></div>`
    ).join('');
  }
  function renderDiag() {
    const sess = Object.entries(diag.sessions).map(([k, v]) =>
      `  ${k}: ${v.file} [${v.ep}] load=${(v.loadMs / 1000).toFixed(2)}s`
      + (v.fetchMs ? ` fetch=${(v.fetchMs / 1000).toFixed(2)}s` : ''));
    els.diagPanel.textContent =
      `Eiqaz Web Experiment ${APP_VERSION}\n`
      + `onnxruntime-web: ${diag.ortVersion || '؟'}\n`
      + `WebGPU adapter: ${diag.webgpuAdapter === null ? 'قيد الفحص…'
        : diag.webgpuAdapter ? 'متاح' : 'غير متاح'}\n`
      + `crossOriginIsolated (خيوط متعددة): ${diag.crossOriginIsolated}\n`
      + `مسار التنفيذ المطلوب: ${diag.requestedEP || '—'} | الفعلي: ${diag.activeEP || '—'}`
      + (diag.epFallback ? ` (رجع احتياطيًا: ${diag.epFallback})` : '') + '\n'
      + `الجلسات:\n${sess.join('\n') || '  —'}\n`
      + (diag.lastError ? `آخر خطأ: ${diag.lastError}\n` : '')
      + `ذاكرة JS: ${perfMemoryText()}`;
  }
  function perfMemoryText() {
    if (performance.memory) {
      return fmtBytes(performance.memory.usedJSHeapSize) + ' / '
        + fmtBytes(performance.memory.jsHeapSizeLimit) + ' (كروم فقط)';
    }
    return 'غير متاح في هذا المتصفح';
  }
  function renderCompat() {
    const c = {
      'WebGPU (navigator.gpu)': typeof navigator.gpu !== 'undefined',
      'WASM SIMD': wasmFeatureDetect(),
      'SharedArrayBuffer (خيوط)': typeof SharedArrayBuffer !== 'undefined',
      'AudioContext 22.05kHz': true,
      'عزل المصدر (COOP/COEP)': diag.crossOriginIsolated,
    };
    els.compatPanel.textContent = Object.entries(c)
      .map(([k, v]) => `${v ? '✓' : '✗'} ${k}`).join('\n');
  }
  function wasmFeatureDetect() {
    try {
      // كشف SIMD عبر WebAssembly.validate لتعليمة v128
      return WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
        10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
    } catch (e) { return false; }
  }

  async function detectWebGPU() {
    if (diag.webgpuAdapter !== null) return diag.webgpuAdapter;
    try {
      if (typeof navigator.gpu === 'undefined') {
        diag.webgpuAdapter = false;
      } else {
        const adapter = await navigator.gpu.requestAdapter();
        diag.webgpuAdapter = !!adapter;
      }
    } catch (e) {
      diag.webgpuAdapter = false;
    }
    renderDiag(); renderCompat();
    return diag.webgpuAdapter;
  }

  // ========================================================================
  // تحميل ORT والجلسات
  // ========================================================================
  function configureOrt() {
    if (diag.ortVersion) return;
    const ort = globalThis.ort;
    ort.env.wasm.wasmPaths = 'vendor/ort/';
    ort.env.logLevel = 'error';
    // خيوط متعددة تتطلب عزل المصدر — وإلا خيط واحد (يعمل لكن أبطأ)
    ort.env.wasm.numThreads = diag.crossOriginIsolated
      ? Math.min(4, (navigator.hardwareConcurrency || 4)) : 1;
    diag.ortVersion = ort.version || '1.30.x';
    renderDiag();
  }

  async function fetchWithProgress(url, onProgress) {
    const t0 = performance.now();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`فشل تنزيل ${url}: HTTP ${resp.status}`);
    const total = +(resp.headers.get('content-length') || 0);
    if (!resp.body || !total) {
      const buf = await resp.arrayBuffer();
      onProgress && onProgress(buf.byteLength, buf.byteLength);
      return { buf, fetchMs: performance.now() - t0 };
    }
    const reader = resp.body.getReader();
    const parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      got += value.byteLength;
      onProgress && onProgress(got, total);
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const p of parts) { buf.set(p, off); off += p.byteLength; }
    return { buf: buf.buffer, fetchMs: performance.now() - t0 };
  }

  const sessionCache = new Map();   // key → session info

  async function createSession(file, epList, label, onProgress) {
    const ort = globalThis.ort;
    const key = file + '|' + epList.join(',');
    if (sessionCache.has(key)) return sessionCache.get(key);

    const { buf, fetchMs } = await fetchWithProgress('models/' + file,
      (got, total) => onProgress && onProgress(label, got, total));

    const t0 = performance.now();
    let session = null, usedEp = epList[0];
    try {
      session = await ort.InferenceSession.create(
        new Uint8Array(buf), { executionProviders: epList });
    } catch (e) {
      if (epList.length > 1 && epList[0] === 'webgpu') {
        usedEp = 'wasm';
        diag.epFallback = 'webgpu→wasm';
        session = await ort.InferenceSession.create(
          new Uint8Array(buf), { executionProviders: ['wasm'] });
      } else {
        throw e;
      }
    }
    const info = { session, file, ep: usedEp, fetchMs,
      loadMs: performance.now() - t0 };
    sessionCache.set(key, info);
    return info;
  }

  // اختيار قائمة مسارات التنفيذ
  function epListFor() {
    const sel = els.epSel.value;
    diag.requestedEP = sel;
    if (sel === 'wasm') return ['wasm'];
    if (sel === 'webgpu') return ['webgpu'];
    return ['webgpu', 'wasm'];    // auto
  }

  async function ensureSessions(onProgress) {
    configureOrt();
    const modelFile = els.modelSel.value;
    const vocFile = els.vocoderSel.value;
    if (!modelFile || !vocFile) throw new Error('لا توجد نماذج — شغّل سكريبت البناء أولًا');
    const eps = epListFor();

    const mix = await createSession(modelFile, eps, 'النموذج الصوتي', onProgress);
    diag.sessions.mixer = mix;
    diag.activeEP = mix.ep;
    const voc = await createSession(vocFile, ['wasm'], 'المُصوِّت', onProgress);
    diag.sessions.vocoder = voc;
    renderDiag();
    return { mix, voc };
  }

  let cattInfo = null;
  async function ensureCatt(onProgress) {
    if (cattInfo) return cattInfo;
    const ort = globalThis.ort;
    const { buf, fetchMs } = await fetchWithProgress('models/catt_eo.onnx',
      (got, total) => onProgress && onProgress('المشكل', got, total));
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(
      new Uint8Array(buf), { executionProviders: ['wasm'] });
    cattInfo = { session, file: 'catt_eo.onnx', ep: 'wasm', fetchMs,
      loadMs: performance.now() - t0 };
    diag.sessions.catt = cattInfo;
    renderDiag();
    return cattInfo;
  }

  // ========================================================================
  // مشغل الصوت (Web Audio) — طابور + مقاطعة فورية
  // ========================================================================
  let audioCtx = null;
  const player = {
    nextTime: 0,
    sources: new Set(),
    playing: false,
    stopAll() {
      for (const s of this.sources) {
        try { s.stop(0); } catch (e) { }
      }
      this.sources.clear();
      this.playing = false;
      this.nextTime = 0;
    },
  };

  function getCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return audioCtx;
  }

  /**
   * جدولة موجة للطابور. تعيد زمن البدء الفعلي.
   * fadeEdges: تلاشي 8ms عند الحواف (منع نقرات الوصل — PATCH 13)
   */
  function scheduleWave(wave, myEpoch) {
    if (myEpoch !== state.epoch) return null;   // طلب قديم — إسقاط
    const ctx = getCtx();
    const buf = ctx.createBuffer(1, wave.length, SAMPLE_RATE);
    buf.copyToChannel(new Float32Array(wave), 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const start = Math.max(ctx.currentTime + 0.02, player.nextTime || ctx.currentTime + 0.02);
    src.start(start);
    player.sources.add(src);
    src.onended = () => { player.sources.delete(src); };
    player.nextTime = start + buf.duration + CHUNK_GAP_S;
    player.playing = true;
    return start;
  }

  // معالجة موجة مقطع (سببية — توحيد جهارة جارٍ مثل PATCH 13 لكن بلا معرفة المستقبل)
  const rmsHistory = [];
  function processChunkWave(wave) {
    const w = Float32Array.from(wave);
    const fade = Math.max(1, Math.floor(EDGE_FADE_S * SAMPLE_RATE));
    if (w.length > 2 * fade) {
      for (let i = 0; i < fade; i++) {
        w[i] *= i / fade;
        w[w.length - 1 - i] *= i / fade;
      }
    }
    // تعادل RMS نحو وسيط المقاطع السابقة (سببي — حدود 0.4–2.5 كالإنتاج)
    let rms = 0;
    for (let i = 0; i < w.length; i++) rms += w[i] * w[i];
    rms = Math.sqrt(rms / Math.max(1, w.length));
    if (rmsHistory.length > 0 && rms > 1e-6) {
      const sorted = [...rmsHistory].sort((a, b) => a - b);
      const target = sorted[Math.floor(sorted.length / 2)];
      if (target > 1e-6) {
        const g = Math.min(Math.max(target / rms, 0.4), 2.5);
        for (let i = 0; i < w.length; i++) w[i] *= g;
      }
    }
    rmsHistory.push(rms);
    return w;
  }

  // توحيد غير سببي (وضع التوليد الكامل — مطابق _unify_chunks_tone الإنتاجي)
  function unifyChunksTone(waves) {
    const n = waves.length;
    if (n === 0) return null;
    const arrs = waves.map(w => Float32Array.from(w));
    const rms = arrs.map(w => {
      let s = 0;
      for (let i = 0; i < w.length; i++) s += w[i] * w[i];
      return Math.sqrt(s / Math.max(1, w.length));
    });
    if (n > 1) {
      const target = [...rms].sort((a, b) => a - b)[Math.floor(n / 2)];
      if (target > 1e-6) {
        for (let i = 0; i < n; i++) {
          if (rms[i] > 1e-6) {
            const g = Math.min(Math.max(target / rms[i], 0.4), 2.5);
            for (let j = 0; j < arrs[i].length; j++) arrs[i][j] *= g;
          }
        }
      }
    }
    const fade = Math.max(1, Math.floor(EDGE_FADE_S * SAMPLE_RATE));
    for (const w of arrs) {
      if (w.length > 2 * fade) {
        for (let i = 0; i < fade; i++) {
          w[i] *= i / fade;
          w[w.length - 1 - i] *= i / fade;
        }
      }
    }
    const gap = new Float32Array(Math.floor(CHUNK_GAP_S * SAMPLE_RATE));
    let out = arrs[0];
    for (let i = 1; i < n; i++) {
      const joined = new Float32Array(out.length + gap.length + arrs[i].length);
      joined.set(out, 0); joined.set(gap, out.length);
      joined.set(arrs[i], out.length + gap.length);
      out = joined;
    }
    let peak = 0;
    for (let i = 0; i < out.length; i++) peak = Math.max(peak, Math.abs(out[i]));
    if (peak > 1e-6) for (let i = 0; i < out.length; i++) out[i] *= 0.9 / peak;
    return out;
  }

  function wavBlobFromFloat32(wave) {
    const n = wave.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE');
    ws(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, SAMPLE_RATE, true);
    v.setUint32(28, SAMPLE_RATE * 2, true); v.setUint16(32, 2, true);
    v.setUint16(34, 16, true); ws(36, 'data'); v.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, wave[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  // ========================================================================
  // توليد مقطع واحد: نص خام → موجة (مطابق مسار الإنتاج)
  // ========================================================================
  async function synthChunk(rawText, opts, myEpoch) {
    if (myEpoch !== state.epoch) return null;
    const TP = globalThis.TextPipe;
    const { mix, voc } = opts.sessions;

    // المسار الجديد (Eiqaz v1): prepareTextRich يقوم بالتطبيع كله
    // (أرقام→كلمات، ترقيم، نقحرة) والتشكيل حسب الوضع — catt يُحمَّل
    // كسولًا فقط إن حسم الوضع التلقائي تشكيلًا (نص غير مشكول).
    // الوضع egyptian الكامل (قواعد det) بايثون فقط — هنا بلا det (موثق).
    const cattFn = async (t) => {
      const catt = await ensureCatt(opts.onProgress);
      if (myEpoch !== state.epoch) throw new Error('canceled');
      return globalThis.CattTashkeel.predict(catt.session, t);
    };
    const res = await TP.prepareTextRich(rawText, opts.diacritize,
      opts.dialect, cattFn);
    if (myEpoch !== state.epoch) return null;

    const toks = opts.dialect === 'msa'
      ? TP.toksMs(res.text) : TP.toksEgy(res.text);
    const ids = TP.tokensToIds(toks);
    if (ids.length < 2) throw new Error('مقطع قصير جدًا بعد الترميز');
    if (myEpoch !== state.epoch) return null;

    // النموذج الصوتي
    const ort = globalThis.ort;
    const tMel = performance.now();
    const melRes = await mix.session.run({
      'text': new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      'pace': new ort.Tensor('float32', new Float32Array([opts.pace]), []),
      'speaker': new ort.Tensor('int64', BigInt64Array.from([BigInt(opts.speaker)]), []),
    });
    const mel = melRes.mel;                    // [1, 80, T']
    if (myEpoch !== state.epoch) return null;
    const melMs = performance.now() - tMel;

    // المُصوِّت — يعالج دفعات 30 إطارًا لحظيًا؟ لا: صوت المقطع قصير — تشغيل كامل
    const tWav = performance.now();
    const wavRes = await voc.session.run({
      'mel_spec': mel,
      'denoise': new ort.Tensor('float32', new Float32Array([0.005]), [1]),
    });
    const vocMs = performance.now() - tWav;
    const wave = wavRes[Object.keys(wavRes)[0]];
    // مخرج vocos [1, T] — الدفعة 1 فالبيانات هي الصف الأول مباشرة
    // (وحتى لو كانت [T] فالبيانات نفسها)
    const waveF = new Float32Array(wave.data);

    return {
      prepared: res.text, tokens: toks, ids, wave: waveF,
      nTokens: ids.length,
      audioS: waveF.length / SAMPLE_RATE,
      melMs, vocMs,
      melFrames: mel.dims[mel.dims.length - 1],
    };
  }

  // ========================================================================
  // وضع التوليد الكامل (مطابق webapp.py)
  // ========================================================================
  async function generateFull() {
    const myEpoch = ++state.epoch;
    state.mode = 'full'; state.busy = true;
    state.chunks = []; rmsHistory.length = 0;
    player.stopAll();
    els.btnGenerate.disabled = true; els.btnStream.disabled = true;
    els.btnInterrupt.disabled = false; els.btnStop.disabled = true;
    els.btnReplay.disabled = true;
    els.fullAudio.style.display = 'none';
    els.chunkList.innerHTML = '';
    renderMetrics({});
    setStatus('تحميل النماذج…', null);

    const t0 = performance.now();
    try {
      const sessions = await ensureSessions((label, got, total) => {
        setLoadStatus(`تنزيل ${label}: ${fmtBytes(got)} / ${fmtBytes(total)}`);
      });
      const loadS = (performance.now() - t0) / 1000;
      setLoadStatus(
        `النموذج جاهز [${sessions.mix.ep}] — تحميل ${(sessions.mix.loadMs / 1000).toFixed(2)}ث`
        + ` + تنزيل ${(sessions.mix.fetchMs / 1000).toFixed(2)}ث`, 'ok');

      const raw = els.textInput.value.trim();
      const TP = globalThis.TextPipe;
      const dialect = els.dialectSel.value;
      const diacritize = els.diacritizeSel.value;
      setStatus('تقسيم النص وتوليد المقاطع…', null);

      const chunks = TP.splitIntoChunks(raw, dialect);
      if (!chunks.length) throw new Error('النص لا يحتوي حروفًا عربية');
      const opts = {
        sessions, dialect, diacritize,
        speaker: +els.speakerSel.value, pace: +els.paceRange.value,
      };

      const waves = [];
      let nTokens = 0, melMs = 0, vocMs = 0;
      const tGen = performance.now();
      for (let i = 0; i < chunks.length; i++) {
        if (myEpoch !== state.epoch) { setStatus('أُلغي الطلب.', null); return; }
        setStatus(`توليد المقطع ${i + 1} من ${chunks.length}…`, null);
        const c = await synthChunk(chunks[i], opts, myEpoch);
        if (!c) return;
        waves.push(c.wave);
        nTokens += c.nTokens; melMs += c.melMs; vocMs += c.vocMs;
        addChunkRow(i + 1, chunks[i], c, 'queued');
      }
      const genS = (performance.now() - tGen) / 1000;

      // توحيد النبرة غير السببي — مطابق الإنتاج
      const final = unifyChunksTone(waves);
      const audioS = final.length / SAMPLE_RATE;
      const blob = wavBlobFromFloat32(final);
      els.fullAudio.src = URL.createObjectURL(blob);
      els.fullAudio.style.display = 'block';
      els.fullAudio.play().catch(() => { });

      renderMetrics({
        ttfaMs: null,           // الوضع الكامل لا يقيس TTFA بث — البث يقيسه
        genS: genS + loadS, audioS, nTokens,
        nChunks: waves.length,
        rtf: audioS > 0 ? genS / audioS : null, errors: 0,
      });
      setStatus(`تم — ${waves.length} مقطعًا · ${audioS.toFixed(1)} ثانية صوت`
        + ` · RTF ${(genS / audioS).toFixed(3)}`, 'ok');
      state.chunks = waves.map((w, i) => ({ text: chunks[i], wave: w }));
      els.btnReplay.disabled = false; els.btnStop.disabled = false;
    } catch (e) {
      diag.lastError = String(e);
      setStatus('خطأ: ' + e.message, 'err');
      renderDiag();
      renderMetrics({ errors: 1 });
    } finally {
      state.busy = false;
      els.btnGenerate.disabled = false; els.btnStream.disabled = false;
      els.btnInterrupt.disabled = true;
    }
  }

  // ========================================================================
  // وضع محاكاة البث (LLM → دفعات نص → نطق تدريجي فوري)
  // ========================================================================
  async function startStream() {
    const myEpoch = ++state.epoch;
    state.mode = 'stream'; state.busy = true;
    state.chunks = []; rmsHistory.length = 0;
    player.stopAll(); player.nextTime = 0;
    els.btnGenerate.disabled = true; els.btnStream.disabled = true;
    els.btnInterrupt.disabled = false; els.btnStop.disabled = false;
    els.fullAudio.style.display = 'none';
    els.chunkList.innerHTML = '';
    renderMetrics({});
    els.llmBox.innerHTML = '<span class="old">سيظهر وصول النص هنا دفعة بدفعة…</span>';
    setStatus('تحميل النماذج…', null);

    const t0 = performance.now();
    const m = {
      firstBatchAt: null, firstAudioAt: null, genS: 0, audioS: 0,
      nTokens: 0, nChunks: 0, errors: 0, batchDelays: [],
    };
    try {
      const sessions = await ensureSessions((label, got, total) => {
        setLoadStatus(`تنزيل ${label}: ${fmtBytes(got)} / ${fmtBytes(total)}`);
      });
      setLoadStatus(
        `النموذج جاهز [${sessions.mix.ep}] — تحميل ${(sessions.mix.loadMs / 1000).toFixed(2)}ث`, 'ok');

      const raw = els.textInput.value.trim();
      const TP = globalThis.TextPipe;
      const dialect = els.dialectSel.value;
      const diacritize = els.diacritizeSel.value;
      const opts = {
        sessions, dialect, diacritize,
        speaker: +els.speakerSel.value, pace: +els.paceRange.value,
      };

      // ---- إعداد الدفعات -------------------------------------------------
      const words = raw.split(/\s+/).filter(Boolean);
      const batchWords = +els.batchWords.value;
      const batchMs = +els.batchMs.value;
      const batches = [];
      for (let i = 0; i < words.length; i += batchWords) {
        batches.push(words.slice(i, i + batchWords).join(' '));
      }

      let arrived = '';
      let synthPointer = 0;          // ما استُهلك في توليد المقاطع
      let pending = [];              // جمل كاملة بانتظار التوليد
      let synthBusy = false;
      let done = false;

      const updateLlmBox = (newPart) => {
        if (myEpoch !== state.epoch) return;
        const old = arrived.slice(0, arrived.length - newPart.length);
        els.llmBox.innerHTML =
          `<span class="old">${escapeHtml(old)}</span>`
          + `<span class="new">${escapeHtml(newPart)}</span>`;
      };

      // ---- مضخة التوليد: جملة كاملة → مقطع فوري ----------------------------
      const pump = async () => {
        if (synthBusy || !pending.length) return;
        synthBusy = true;
        try {
          while (pending.length) {
            if (myEpoch !== state.epoch) return;
            const sentence = pending.shift();
            setStatus(`توليد مقطع فوريًا: «${sentence.slice(0, 40)}…»`, null);
            const tC = performance.now();
            const c = await synthChunk(sentence, opts, myEpoch);
            if (!c) return;                       // طلب ملغى
            m.genS += (performance.now() - tC) / 1000;
            m.nTokens += c.nTokens; m.nChunks++;
            m.audioS += c.audioS;

            const w = processChunkWave(c.wave);
            const startAt = scheduleWave(w, myEpoch);
            if (startAt === null) return;
            if (m.firstAudioAt === null) {
              m.firstAudioAt = performance.now();
            }
            state.chunks.push({ text: sentence, wave: w });
            addChunkRow(m.nChunks, sentence, c, 'playing');
          }
        } catch (e) {
          if (myEpoch === state.epoch) {
            m.errors++; diag.lastError = String(e);
            setStatus('خطأ أثناء البث: ' + e.message, 'err');
            renderDiag();
          }
        } finally {
          synthBusy = false;
          maybeFinish();
        }
      };

      const maybeFinish = () => {
        if (myEpoch !== state.epoch) return;
        if (done && !pending.length && !synthBusy) {
          const ttfaMs = (m.firstBatchAt !== null && m.firstAudioAt !== null)
            ? m.firstAudioAt - m.firstBatchAt : null;
          renderMetrics({
            ttfaMs, genS: m.genS, audioS: m.audioS, nTokens: m.nTokens,
            nChunks: m.nChunks,
            rtf: m.audioS > 0 ? m.genS / m.audioS : null,
            errors: m.errors,
          });
          setStatus(`انتهى البث — TTFA ${ttfaMs ? (ttfaMs / 1000).toFixed(2) : '؟'}ث`
            + ` · ${m.nChunks} مقطعًا · RTF ${m.audioS > 0 ? (m.genS / m.audioS).toFixed(3) : '؟'}`,
            m.errors ? 'err' : 'ok');
          els.btnReplay.disabled = false;
        }
      };

      // وصول الدفعات
      for (let i = 0; i < batches.length; i++) {
        if (myEpoch !== state.epoch) return;
        if (i === 0) m.firstBatchAt = performance.now();
        arrived += (i ? ' ' : '') + batches[i];
        updateLlmBox(batches[i]);
        setStatus(`وصلت الدفعة ${i + 1}/${batches.length} — النموذج اللغوي يرسل…`, null);

        // اكتشاف الجمل الكاملة الجديدة (قص عند نهايات الجمل فقط —
        // العلامات {ق} تصل ملتصقة بكلماتها داخل الجملة الكاملة)
        const tail = arrived.slice(synthPointer);
        const parts = tail.split(/(?<=[.!؟?…])\s+/);
        const complete = parts.slice(0, -1);
        const lastIsComplete = /[.!؟?…]\s*$/.test(tail);
        if (complete.length || lastIsComplete) {
          if (lastIsComplete) complete.push(parts[parts.length - 1]);
          for (const c of complete) {
            const cc = c.trim();
            if (cc) pending.push(cc);
            synthPointer += c.length + 1;
          }
          pump();     // توليد غير متزامن — الصوت يبدأ فورًا
        }
        await sleep(batchMs);
      }

      // دفعة أخيرة: ما تبقى بلا نهاية جملة
      if (myEpoch !== state.epoch) return;
      const tailRest = arrived.slice(synthPointer).trim();
      if (tailRest) pending.push(tailRest);
      done = true;
      pump();
      maybeFinish();

    } catch (e) {
      diag.lastError = String(e);
      setStatus('خطأ: ' + e.message, 'err');
      renderDiag();
    } finally {
      state.busy = false;
      els.btnGenerate.disabled = false; els.btnStream.disabled = false;
      els.btnInterrupt.disabled = true;
    }
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function addChunkRow(idx, text, c, cls) {
    const div = document.createElement('div');
    div.className = 'chunk ' + (cls || '');
    div.dataset.idx = idx;
    div.innerHTML =
      `<span class="idx">مقطع ${idx}</span>`
      + `<span class="txt">${escapeHtml(text)}</span>`
      + `<span class="meta">${c.nTokens} توكن · ${c.audioS.toFixed(1)}ث`
      + ` · mel ${c.melMs.toFixed(0)}مل + voc ${c.vocMs.toFixed(0)}مل</span>`;
    els.chunkList.prepend(div);
  }

  // ========================================================================
  // المقاطعة / الإيقاف / إعادة التشغيل
  // ========================================================================
  function interrupt() {
    const t0 = performance.now();
    state.epoch++;               // إبطال كل الاستمرارات الجارية والطوابير
    player.stopAll();
    rmsHistory.length = 0;
    state.mode = null; state.busy = false;
    for (const el of [els.btnGenerate, els.btnStream]) el.disabled = false;
    els.btnInterrupt.disabled = true;
    els.btnStop.disabled = true;
    const ms = Math.round(performance.now() - t0);
    setStatus(`تمت المقاطعة فورًا (${ms} مللي ثانية) — الصوت متوقف، والدفعات`
      + ` المتأخرة ستُرفض. جاهز لطلب جديد بلا إعادة تحميل.`, 'ok');
    const mm = Object.assign({}, state.metrics);
    mm.interruptMs = ms;
    renderMetrics(mm);
    // تعليم المقاطع الملغاة بصريًا
    for (const el of els.chunkList.querySelectorAll('.chunk')) {
      if (el.classList.contains('playing')) el.classList.add('cut');
    }
    return ms;
  }

  function stopPlayback() {
    player.stopAll();
    setStatus('أُوقف التشغيل (المقاطع المولدة محفوظة — يمكن إعادة التشغيل).', null);
    els.btnReplay.disabled = state.chunks.length === 0;
  }

  function replay() {
    if (!state.chunks.length) return;
    state.epoch++;               // منع أي استمرارات قديمة
    player.stopAll(); player.nextTime = 0;
    rmsHistory.length = 0;
    let t = 0;
    for (const ch of state.chunks) {
      const w = processChunkWave(ch.wave);
      scheduleWaveAt(w, t);
      t += w.length / SAMPLE_RATE + CHUNK_GAP_S;
    }
    setStatus(`إعادة تشغيل ${state.chunks.length} مقطعًا من المخزن.`, null);
  }

  function scheduleWaveAt(wave, at) {
    const ctx = getCtx();
    const buf = ctx.createBuffer(1, wave.length, SAMPLE_RATE);
    buf.copyToChannel(new Float32Array(wave), 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const start = ctx.currentTime + 0.05 + at;
    src.start(start);
    player.sources.add(src);
    src.onended = () => { player.sources.delete(src); };
    player.nextTime = Math.max(player.nextTime, start + buf.duration);
  }

  // ========================================================================
  // بناء الواجهة من manifest
  // ========================================================================
  async function initFromManifest() {
    try {
      const resp = await fetch('models/manifest.json');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      diag.manifest = await resp.json();
    } catch (e) {
      diag.manifestError = String(e);
      els.modelSel.innerHTML = '<option value="">— لا يوجد manifest —</option>';
      setLoadStatus('لا يوجد models/manifest.json — شغّل سكريبت بناء النماذج'
        + ' (inference/onnx/export_onnx.py ثم quantize_onnx.py) أولًا.', 'err');
      renderDiag();
      return;
    }
    const acoustic = diag.manifest.acoustic_variants
      .filter(v => v.status !== 'failed' && v.file);
    for (const v of acoustic) {
      const o = document.createElement('option');
      o.value = v.file;
      const sz = (v.size_bytes / 1e6).toFixed(1);
      const verd = v.parity && v.parity.verdict ? v.parity.verdict : '';
      o.textContent = `${v.id.toUpperCase()} — ${sz}MB`
        + (verd ? ` [${verd === 'reference' ? 'مرجع' : verd}]` : '');
      o.dataset.tag = v.tag || '';
      els.modelSel.appendChild(o);
    }
    // الأفضل افتراضيًا: fp16 (نجح تكافؤه) وإلا fp32
    const fp16 = acoustic.find(v => v.id === 'fp16' && v.parity
      && v.parity.verdict === 'PASS');
    if (fp16) els.modelSel.value = fp16.file;

    const voc = (diag.manifest.other_models || []).filter(
      v => v.role === 'vocoder' && v.status !== 'failed'
      && v.parity && v.parity.verdict !== 'FAIL');
    for (const v of voc) {
      const o = document.createElement('option');
      o.value = v.file;
      o.textContent = `${v.id.toUpperCase()} — ${(v.size_bytes / 1e6).toFixed(1)}MB`;
      els.vocoderSel.appendChild(o);
    }
    updateModelHints();
    const isSanity = acoustic.some(v => v.tag === 'sanity');
    els.sanityWarn.classList.toggle('on', !!isSanity);
    renderDiag();
  }

  function updateModelHints() {
    const mv = els.modelSel.selectedOptions[0];
    if (mv) {
      const v = (diag.manifest.acoustic_variants || [])
        .find(x => x.file === mv.value);
      els.modelHint.textContent = v && v.notes_ar ? v.notes_ar : '';
      els.modelSel.dataset.tag = mv.dataset.tag || '';
    }
    const vv = els.vocoderSel.selectedOptions[0];
    if (vv) {
      const v = (diag.manifest.other_models || [])
        .find(x => x.file === vv.value);
      els.vocoderHint.textContent = v && v.notes_ar ? v.notes_ar : '';
    }
  }

  // ========================================================================
  // واجهة الاختبارات الآلية (Playwright) — تكافؤ catt والنص داخل المتصفح
  // ========================================================================
  async function cattParityCheck() {
    const catt = await ensureCatt();
    if (!catt) throw new Error('catt غير محمّل');
    let expected;
    try {
      const r = await fetch('tests/expected_catt.json');
      expected = await r.json();
    } catch (e) {
      throw new Error('expected_catt.json غير موجود: ' + e.message);
    }
    const rows = [];
    for (const item of expected) {
      if (item.error) { rows.push({ id: item.id, skipped: true }); continue; }
      const got = await globalThis.CattTashkeel.predict(catt.session, item.text);
      rows.push({
        id: item.id,
        match: got === item.vocalized,
        expected: item.vocalized, got,
      });
    }
    const nMatch = rows.filter(r => r.match).length;
    return { nMatch, nTotal: rows.length, rows,
      verdict: nMatch === rows.length ? 'PASS' : 'FAIL' };
  }

  async function textParityCheck() {
    let expected;
    try {
      const r = await fetch('tests/expected_tokens.json');
      expected = await r.json();
    } catch (e) {
      throw new Error('expected_tokens.json غير موجود: ' + e.message);
    }
    const TP = globalThis.TextPipe;
    const rows = [];
    for (const item of expected) {
      if (item.error) { rows.push({ id: item.id, skipped: true }); continue; }
      try {
        const res = await TP.prepareTextRich(item.text, 'manual',
          item.dialect, null);
        const toks = item.dialect === 'msa'
          ? TP.toksMs(res.text) : TP.toksEgy(res.text);
        const ids = TP.tokensToIds(toks);
        rows.push({
          id: item.id,
          match: JSON.stringify(ids) === JSON.stringify(item.ids),
        });
      } catch (e) {
        rows.push({ id: item.id, match: false, error: String(e) });
      }
    }
    const nMatch = rows.filter(r => r.match).length;
    return { nMatch, nTotal: rows.length, rows,
      verdict: nMatch === rows.length ? 'PASS' : 'FAIL' };
  }

  // ========================================================================
  // ربط الواجهة
  // ========================================================================
  els.btnGenerate.addEventListener('click', () => generateFull());
  els.btnStream.addEventListener('click', () => startStream());
  els.btnInterrupt.addEventListener('click', () => interrupt());
  els.btnStop.addEventListener('click', () => stopPlayback());
  els.btnReplay.addEventListener('click', () => replay());

  els.paceRange.addEventListener('input', () => {
    els.paceVal.textContent = (+els.paceRange.value).toFixed(2);
  });
  els.batchWords.addEventListener('input', () => {
    els.batchWordsVal.textContent = els.batchWords.value;
  });
  els.batchMs.addEventListener('input', () => {
    els.batchMsVal.textContent = els.batchMs.value;
  });
  els.modelSel.addEventListener('change', updateModelHints);
  els.vocoderSel.addEventListener('change', updateModelHints);
  els.epSel.addEventListener('change', () => {
    diag.requestedEP = els.epSel.value;
    renderDiag();
  });

  const PRESETS = {
    presetShort: 'خلينا نبدأ. الموضوع بسيط. جرب بنفسك. واسمع الفرق. قبل وبعد.',
    presetLong: 'النموذج اللغوي بيبعت النص على دفعات، والمحرك لازم يتكلم فورًا من أول دفعة، والصوت لازم يفضل شغال بدون انتظار، والمقاطعة لازم توقف كل حاجة في نفس اللحظة، وفي الآخر نقيس كل الأزمنة ونقارن النسخ.',
    presetQaf: 'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا. القرآن{ق} فيه قصص{ق} كتير.',
    presetMSA: 'سَنَتَعَلَّمُ الْيَوْمَ مَعْنَى التَّفْكِيرِ الْمَنْطِقِيِّ وَأَهَمِّيَّتَهُ فِي حَيَاتِنَا الْيَوْمِيَّةِ.',
  };
  for (const [id, txt] of Object.entries(PRESETS)) {
    els[id].addEventListener('click', () => { els.textInput.value = txt; });
  }

  // ========================================================================
  // واجهة عامة للاختبارات
  // ========================================================================
  globalThis.EiqazApp = {
    version: APP_VERSION,
    state, diag,
    generateFull, startStream, interrupt, stop: stopPlayback, replay,
    ensureSessions, ensureCatt,
    cattParityCheck, textParityCheck,
    synthChunk, processChunkWave, unifyChunksTone, wavBlobFromFloat32,
  };

  // ---- تشغيل أولي ---------------------------------------------------------
  detectWebGPU();
  renderCompat();
  initFromManifest();
  renderDiag();
})();
