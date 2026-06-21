/* InstructFX2FX demo — waveform players, A/B compare, gradient-descent scrub. */
(() => {
  "use strict";
  const elExamples = document.getElementById("examples");
  if (!elExamples) return;

  const PLAY  = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  const PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
  const RES = 480;
  const PEAK_CACHE = new Map();
  let actx = null;
  const decodeCtx = () => (actx ||= new (window.AudioContext || window.webkitAudioContext)());

  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const FX_NAME = { rev: "reverb", reverb: "reverb", eq: "EQ", comp: "comp", dist: "dist", delay: "delay", pitchshift: "pitch", bitcrush: "crush" };
  const fxName = f => FX_NAME[f] || f;
  const ROUTE = {
    "Initialize-only":            { label: "init",        cls: "init"   },
    "Reuse-and-optimize":         { label: "reuse + opt", cls: "reuse"  },
    "Mixed reuse-and-initialize": { label: "extend",      cls: "extend" },
  };
  const hasGD = t => Array.isArray(t.trajectory) && t.trajectory.some(c => /iter_?\d+/.test(c.label || ""));
  const cpLabel = c => { const l = c.label || ""; if (l === "start") return "init"; if (l === "end" || l === "final") return "converged";
    const m = /iter_?(\d+)/.exec(l); return m ? "iter " + m[1] : l; };
  const fmt = s => { if (!isFinite(s) || s < 0) s = 0; const m = Math.floor(s / 60), x = Math.floor(s % 60); return m + ":" + String(x).padStart(2, "0"); };

  async function loadPeaks(url) {
    if (PEAK_CACHE.has(url)) return PEAK_CACHE.get(url);
    const p = (async () => {
      const ab = await (await fetch(url)).arrayBuffer();
      const buf = await decodeCtx().decodeAudioData(ab);
      const ch0 = buf.getChannelData(0);
      const ch1 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : null;
      const peaks = new Float32Array(RES);
      const block = Math.max(1, Math.floor(ch0.length / RES));
      let mx = 0;
      for (let i = 0; i < RES; i++) {
        let m = 0; const s = i * block, e = Math.min(ch0.length, s + block);
        for (let j = s; j < e; j++) { let v = Math.abs(ch0[j]); if (ch1) { const w = Math.abs(ch1[j]); if (w > v) v = w; } if (v > m) m = v; }
        peaks[i] = m; if (m > mx) mx = m;
      }
      if (mx > 0) for (let i = 0; i < RES; i++) peaks[i] /= mx;
      return { peaks, duration: buf.duration };
    })();
    PEAK_CACHE.set(url, p);
    return p;
  }

  let CURRENT = null;
  function claim(w) { if (CURRENT && CURRENT !== w) CURRENT.audio.pause(); CURRENT = w; }

  class Wave {
    constructor(root) {
      this.canvas = root.querySelector("canvas.wf");
      this.btn    = root.querySelector("[data-play]");
      this.time   = root.querySelector("[data-time]");
      this.load   = root.querySelector("[data-load]");
      this.ctx    = this.canvas.getContext("2d");
      this.audio  = new Audio(); this.audio.preload = "none";
      this.peaks = null; this.dur = 0; this.url = null; this.progress = 0; this.raf = 0; this.token = 0;
      this._seek = null; this._drag = false;

      this.btn.addEventListener("click", () => this.toggle());
      this.audio.addEventListener("play",  () => { this.setIcon(true);  this.tick(); claim(this); });
      this.audio.addEventListener("pause", () => { this.setIcon(false); cancelAnimationFrame(this.raf); });
      this.audio.addEventListener("ended", () => { this.setIcon(false); this.progress = 1; this.draw(); });
      this.audio.addEventListener("loadedmetadata", () => { if (this._seek != null) { try { this.audio.currentTime = this._seek * this.audio.duration; } catch (e) {} this._seek = null; } this.updTime(); });
      const seek = e => {
        const r = this.canvas.getBoundingClientRect();
        const f = Math.max(0, Math.min(1, ((e.touches ? e.touches[0].clientX : e.clientX) - r.left) / r.width));
        this.progress = f; this.draw();
        if (this.audio.readyState >= 1 && isFinite(this.audio.duration)) { this.audio.currentTime = f * this.audio.duration; this.updTime(); }
        else { this._seek = f; if (!this.audio.src) this.audio.src = this.url; }
      };
      this.canvas.addEventListener("pointerdown", e => { this.canvas.setPointerCapture(e.pointerId); this._drag = true; seek(e); });
      this.canvas.addEventListener("pointermove", e => { if (this._drag) seek(e); });
      this.canvas.addEventListener("pointerup",   () => { this._drag = false; });
      new ResizeObserver(() => this.draw()).observe(this.canvas);
    }
    setIcon(on) { this.btn.innerHTML = on ? PAUSE : PLAY; this.btn.setAttribute("aria-pressed", on ? "true" : "false"); }
    toggle() { if (!this.url) return; if (this.audio.paused) { if (!this.audio.src) this.audio.src = this.url; this.audio.play().catch(() => {}); } else this.audio.pause(); }
    tick() { cancelAnimationFrame(this.raf); const step = () => { if (isFinite(this.audio.duration) && this.audio.duration > 0) { this.progress = this.audio.currentTime / this.audio.duration; this.draw(); this.updTime(); } if (!this.audio.paused) this.raf = requestAnimationFrame(step); }; this.raf = requestAnimationFrame(step); }
    updTime() { const d = isFinite(this.audio.duration) ? this.audio.duration : this.dur; this.time.textContent = fmt(this.audio.currentTime || this.progress * d) + " / " + fmt(d); }
    async set(url, opts) {
      const keep = opts && opts.keep;
      const tok = ++this.token;
      const wasPlaying = !this.audio.paused;
      const frac = keep ? this.progress : 0;
      this.url = url;
      this.audio.pause();
      this.audio.src = url;
      this._seek = frac;
      if (wasPlaying) this.audio.play().catch(() => {});
      else this.progress = frac;
      this.load.classList.add("on");
      try { const { peaks, duration } = await loadPeaks(url); if (tok !== this.token) return; this.peaks = peaks; this.dur = duration; }
      catch (e) { if (tok === this.token) this.peaks = null; }
      this.load.classList.remove("on");
      this.draw(); this.updTime();
    }
    draw() {
      const c = this.canvas, ctx = this.ctx, dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth, H = c.clientHeight;
      if (!W || !H) return;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const cs = getComputedStyle(document.documentElement);
      const cWave = cs.getPropertyValue("--wave").trim() || "#bbb";
      const cPlay = cs.getPropertyValue("--accent").trim() || "#2f4b86";
      const mid = H / 2, bw = 2, gap = 1, n = Math.max(16, Math.floor(W / (bw + gap)));
      const px = this.progress * W;
      for (let i = 0; i < n; i++) {
        const t0 = Math.floor(i / n * RES), t1 = Math.max(t0 + 1, Math.floor((i + 1) / n * RES));
        let v = 0; if (this.peaks) for (let j = t0; j < t1 && j < RES; j++) { if (this.peaks[j] > v) v = this.peaks[j]; }
        if (!this.peaks) v = 0.05;
        const h = Math.max(1.5, v * (H - 4)), x = i * (bw + gap);
        ctx.fillStyle = (x + bw) <= px ? cPlay : cWave;
        ctx.fillRect(x, mid - h / 2, bw, h);
      }
      if (this.progress > 0 && this.progress < 1) { ctx.fillStyle = cPlay; ctx.fillRect(px - 0.75, 1, 1.5, H - 2); }
    }
  }

  function chainHtml(t) {
    const nw = new Set(t.fx_new || []);
    return (t.fx_chain || []).map(f =>
      `<span class="fx${nw.has(f) ? " new" : ""}">${esc(fxName(f))}${nw.has(f) ? '<span class="tag">+new</span>' : ""}</span>`
    ).join('<span class="arrow">→</span>');
  }

  function playerHtml(gd, traj, ariaLabel) {
    return `
      <div class="player">
        <div class="pl-row">
          <button class="pbtn" data-play type="button" aria-pressed="false" aria-label="${ariaLabel}">${PLAY}</button>
          <div class="wf-wrap"><canvas class="wf"></canvas><span class="wf-load" data-load>decoding</span></div>
          <span class="ptime" data-time>0:00 / 0:00</span>
        </div>
        <div class="pl-controls">
          <div class="seg" role="group" aria-label="Compare">
            <button type="button" data-ab="dry">dry</button>
            <button type="button" data-ab="result" aria-pressed="true">result</button>
          </div>
          ${gd ? `
          <div class="gd">
            <div class="gd-head"><span class="gd-label">gradient descent</span><span class="gd-now" data-now>converged</span></div>
            <input class="slider" type="range" min="0" max="${traj.length - 1}" value="${traj.length - 1}" step="1" data-slider aria-label="Optimization checkpoint" />
            <div class="gd-axis"><span>init</span><span>iterations →</span><span>converged</span></div>
          </div>` : ``}
        </div>
      </div>`;
  }

  function turnHtml(t, source) {
    const r = ROUTE[t.routing] || { label: t.routing, cls: "" };
    const gd = hasGD(t), traj = gd ? t.trajectory : null;
    return `
      <div class="turn" data-source="${esc(source)}" data-result="${esc(t.result)}">
        <span class="turn-node">${String(t.n).padStart(2, "0")}</span>
        <p class="instruction">${esc(t.instruction)}</p>
        <div class="turn-meta">
          <span class="route ${r.cls}"><i></i>${esc(r.label)}</span>
          <span class="chain">${chainHtml(t)}</span>
        </div>
        ${playerHtml(gd, traj, "Play result")}
      </div>`;
  }

  function dryPlayerHtml() {
    return `
      <div class="player" data-dry>
        <div class="pl-row">
          <button class="pbtn" data-play type="button" aria-pressed="false" aria-label="Play dry input">${PLAY}</button>
          <div class="wf-wrap"><canvas class="wf"></canvas><span class="wf-load" data-load>decoding</span></div>
          <span class="ptime" data-time>0:00 / 0:00</span>
        </div>
      </div>`;
  }

  function sessionHtml(ex, i) {
    return `
      <div class="session">
        <div class="ses-top"><span class="ses-id">S${String(i + 1).padStart(2, "0")}</span><h3>${esc(ex.title)}</h3><span class="ses-meta">${ex.n_turns} turns · ${esc(ex.llm)}</span></div>
        <p class="ses-blurb">${esc(ex.blurb || "")}</p>
        ${ex.source ? `<div class="dry"><span class="dry-tag">dry input</span>${dryPlayerHtml()}</div>` : ""}
        <div class="thread">${ex.turns.map(t => turnHtml(t, ex.source)).join("")}</div>
      </div>`;
  }

  function lazy(el, cb) {
    const io = new IntersectionObserver((es, obs) => { es.forEach(e => { if (e.isIntersecting) { cb(); obs.disconnect(); } }); }, { rootMargin: "260px" });
    io.observe(el);
  }

  function wireTurn(turnEl, traj) {
    const source = turnEl.dataset.source, result = turnEl.dataset.result;
    const w = new Wave(turnEl.querySelector(".player"));
    const segBtns = [...turnEl.querySelectorAll("[data-ab]")];
    const slider = turnEl.querySelector("[data-slider]");
    const now = turnEl.querySelector("[data-now]");
    const setSeg = key => segBtns.forEach(b => b.setAttribute("aria-pressed", b.dataset.ab === key ? "true" : "false"));
    const clearSeg = () => segBtns.forEach(b => b.setAttribute("aria-pressed", "false"));
    segBtns.forEach(b => b.addEventListener("click", () => {
      setSeg(b.dataset.ab);
      if (b.dataset.ab === "dry") w.set(source, { keep: true });
      else { w.set(result, { keep: true }); if (slider) { slider.value = slider.max; if (now) now.textContent = "converged"; } }
    }));
    if (slider && traj) {
      slider.addEventListener("input", () => { const i = +slider.value; clearSeg(); if (now) now.textContent = cpLabel(traj[i]); w.set(traj[i].audio, { keep: true }); if (i === traj.length - 1) setSeg("result"); });
    }
    lazy(turnEl, () => w.set(result));
  }

  function fallback(data) {
    elExamples.innerHTML = data.examples.map((ex, i) => `
      <div class="session">
        <div class="ses-top"><span class="ses-id">S${String(i + 1).padStart(2, "0")}</span><h3>${esc(ex.title)}</h3><span class="ses-meta">${ex.n_turns} turns · ${esc(ex.llm)}</span></div>
        <p class="ses-blurb">${esc(ex.blurb || "")}</p>
        ${ex.source ? `<div class="dry"><span class="dry-tag">dry input</span><audio controls preload="none" src="${esc(ex.source)}" style="width:100%"></audio></div>` : ""}
        <div class="thread">${ex.turns.map(t => { const r = ROUTE[t.routing] || { label: t.routing, cls: "" }; return `
          <div class="turn"><span class="turn-node">${String(t.n).padStart(2, "0")}</span>
            <p class="instruction">${esc(t.instruction)}</p>
            <div class="turn-meta"><span class="route ${r.cls}"><i></i>${esc(r.label)}</span><span class="chain">${chainHtml(t)}</span></div>
            <audio controls preload="none" src="${esc(t.result)}" style="width:100%"></audio>
          </div>`; }).join("")}</div>
      </div>`).join("");
  }

  (async () => {
    let data;
    try { data = await (await fetch("data.json")).json(); }
    catch (e) { elExamples.innerHTML = `<div class="loading">Could not load <code>data.json</code>. Serve this folder over HTTP (for example <code>python&nbsp;-m&nbsp;http.server</code>).</div>`; return; }
    if (!(window.AudioContext || window.webkitAudioContext)) { fallback(data); return; }
    try {
      elExamples.innerHTML = data.examples.map(sessionHtml).join("");
      const sessions = [...elExamples.querySelectorAll(".session")];
      data.examples.forEach((ex, si) => {
        const ses = sessions[si];
        const dryEl = ses.querySelector("[data-dry]");
        if (dryEl) { const dw = new Wave(dryEl); lazy(dryEl, () => dw.set(ex.source)); }
        const turnEls = [...ses.querySelectorAll(".turn")];
        ex.turns.forEach((t, ti) => wireTurn(turnEls[ti], hasGD(t) ? t.trajectory : null));
      });
    } catch (e) { fallback(data); }
  })();

  // scrollspy
  (() => {
    const links = [...document.querySelectorAll(".tnav a")];
    if (!links.length) return;
    const map = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) { links.forEach(l => l.classList.remove("on")); const a = map.get(e.target.id); if (a) a.classList.add("on"); } });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("section[id]").forEach(s => io.observe(s));
  })();
})();
