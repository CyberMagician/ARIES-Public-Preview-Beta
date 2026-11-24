/* ARIES Public Preview Beta — Apple HIG Frontend Controller
   - First-Launch Graphical Model Setup (with Orbital Canvas + SVG rings)
   - Prompt, Persona & Inference Settings Sheet
   - Rich Markdown (Mermaid.js, KaTeX, Highlight.js, Sandboxed Code Runner, Canvas)
   - Fullscreen Image Lightbox
*/

(function () {
  "use strict";

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatBytes(b) {
    if (!b || b <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let v = b;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(i >= 2 ? 2 : 0)} ${units[i]}`;
  }

  // 1. Orbital Canvas Animation for First-Launch Setup
  let orbitalPct = 0;
  function startOrbitalAnimation() {
    const canvas = document.getElementById("ariesOrbitalCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let t = 0;
    function draw() {
      if (!document.getElementById("ariesSetupOverlay")?.classList.contains("open")) {
        requestAnimationFrame(draw);
        return;
      }
      t += 0.03;
      const w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2;
      ctx.clearRect(0, 0, w, h);

      // Outer progress arc
      ctx.beginPath();
      ctx.arc(cx, cy, 36, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * Math.max(0.04, orbitalPct / 100)));
      ctx.strokeStyle = "#5AC8FA";
      ctx.lineWidth = 3.5;
      ctx.lineCap = "round";
      ctx.stroke();

      // Orbiting nodes
      for (let i = 0; i < 3; i++) {
        const ang = t * (1 + i * 0.35) + (i * Math.PI * 2) / 3;
        const r = 18 + i * 6;
        const x = cx + Math.cos(ang) * r;
        const y = cy + Math.sin(ang) * r * 0.7;
        ctx.beginPath();
        ctx.arc(x, y, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "#0A84FF" : i === 1 ? "#5AC8FA" : "#30D158";
        ctx.fill();
      }
      requestAnimationFrame(draw);
    }
    draw();
  }

  // 2. Setup Overlay & Model Downloader Polling
  let setupPollTimer = null;
  async function refreshSetupStatus() {
    try {
      const res = await fetch("/api/setup/status");
      const data = await res.json();
      orbitalPct = data.percent || 0;

      const fill = document.getElementById("ariesAggFill");
      const stats = document.getElementById("ariesAggStats");
      const grid = document.getElementById("ariesSetupGrid");
      if (fill) fill.style.width = `${data.percent}%`;
      if (stats) {
        const eta = data.eta_seconds > 0 ? ` · ETA ${data.eta_seconds}s` : "";
        const spd = data.speed_mbps > 0 ? ` · ${data.speed_mbps} MB/s` : "";
        stats.textContent = `${formatBytes(data.downloaded_bytes)} / ${formatBytes(data.total_bytes)} (${data.percent}%)${spd}${eta}`;
      }
      if (grid && Array.isArray(data.models)) {
        grid.innerHTML = data.models.map((m) => {
          const r = 16;
          const c = 2 * Math.PI * r;
          const off = c - (c * (m.percent || 0)) / 100;
          const stroke = m.status === "ready" ? "#30D158" : m.status === "error" ? "#FF453A" : "#0A84FF";
          return `
            <div class="aries-model-tile">
              <svg class="aries-ring-svg" viewBox="0 0 40 40">
                <circle cx="20" cy="20" r="${r}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="3.5"/>
                <circle cx="20" cy="20" r="${r}" fill="none" stroke="${stroke}" stroke-width="3.5"
                        stroke-dasharray="${c}" stroke-dashoffset="${off}" stroke-linecap="round"/>
              </svg>
              <div style="min-width:0;flex:1;">
                <div style="font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(m.name)}</div>
                <div style="font-size:11px;color:#9ba3b4;">${escapeHtml(m.role)} · ${escapeHtml(m.license)}</div>
                <div style="font-size:10.5px;color:${stroke};margin-top:2px;">${escapeHtml(m.status.toUpperCase())} (${m.percent}%)</div>
              </div>
            </div>`;
        }).join("");
      }

      if (data.all_ready && setupPollTimer) {
        clearInterval(setupPollTimer);
        setupPollTimer = null;
      }
      return data;
    } catch (e) {
      return null;
    }
  }

  window.openAriesSetup = async function () {
    const ov = document.getElementById("ariesSetupOverlay");
    if (ov) ov.classList.add("open");
    await refreshSetupStatus();
    if (!setupPollTimer) setupPollTimer = setInterval(refreshSetupStatus, 1000);
  };

  window.closeAriesSetup = function () {
    const ov = document.getElementById("ariesSetupOverlay");
    if (ov) ov.classList.remove("open");
    if (setupPollTimer) { clearInterval(setupPollTimer); setupPollTimer = null; }
  };

  window.startAriesModelDownload = async function () {
    await fetch("/api/setup/start", { method: "POST" });
    await refreshSetupStatus();
    if (!setupPollTimer) setupPollTimer = setInterval(refreshSetupStatus, 900);
  };

  window.cancelAriesModelDownload = async function () {
    await fetch("/api/setup/cancel", { method: "POST" });
    await refreshSetupStatus();
  };

  window.linkLocalModelsFolder = async function () {
    const inp = document.getElementById("ariesLocalModelsPath");
    const path = inp ? inp.value.trim() : "";
    if (!path) return;
    const res = await fetch("/api/setup/link-local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await res.json();
    await refreshSetupStatus();
    if (data.ok && data.status?.all_ready) {
      setTimeout(window.closeAriesSetup, 700);
    }
  };

  // 3. Canvas & Lightbox Helpers
  window.openAriesCanvas = function (title, htmlContent) {
    const drawer = document.getElementById("ariesCanvasDrawer");
    const frame = document.getElementById("ariesCanvasFrame");
    const ttl = document.getElementById("ariesCanvasTitle");
    if (!drawer || !frame) return;
    if (ttl) ttl.textContent = title || "Live Preview";
    frame.srcdoc = htmlContent;
    drawer.classList.add("open");
  };

  window.closeAriesCanvas = function () {
    document.getElementById("ariesCanvasDrawer")?.classList.remove("open");
  };

  window.openAriesLightbox = function (src) {
    const lb = document.getElementById("ariesLightbox");
    const img = document.getElementById("ariesLightboxImg");
    if (lb && img) {
      img.src = src;
      lb.classList.add("open");
    }
  };

  window.closeAriesLightbox = function () {
    document.getElementById("ariesLightbox")?.classList.remove("open");
  };

  window.runSandboxedCodeBlock = async function (btn, lang, encodedCode) {
    const code = decodeURIComponent(encodedCode);
    const wrap = btn.closest(".hig-code-wrap");
    let outEl = wrap.querySelector(".hig-sandbox-out");
    if (!outEl) {
      outEl = document.createElement("div");
      outEl.className = "hig-sandbox-out";
      wrap.appendChild(outEl);
    }
    outEl.textContent = "Running in /private/tmp sandbox...";
    try {
      const res = await fetch("/api/sandbox/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: lang, code }),
      });
      const data = await res.json();
      outEl.style.color = data.ok ? "#30D158" : "#FF453A";
      outEl.textContent = (data.stdout || "") + (data.stderr ? `\n${data.stderr}` : "") || "(no output)";
    } catch (e) {
      outEl.style.color = "#FF453A";
      outEl.textContent = String(e);
    }
  };

  // 4. Enhance Chat Code Blocks, Mermaid, Highlight.js, & KaTeX
  function sanitizeMermaidCode(raw) {
    return String(raw || "").replace(/(\b[A-Za-z0-9_]+)\[([^"\[\]\n]*[()][^"\[\]\n]*)\]/g, '$1["$2"]');
  }

  function buildCanvasDoc(lang, code) {
    if (lang === "mermaid") {
      const clean = sanitizeMermaidCode(code);
      return `<!DOCTYPE html><html><head><meta charset="utf-8"><script src="/static/vendor/mermaid.min.js"><\/script></head><body style="margin:0;padding:20px;background:#0b0f14;color:#f5f7fa;display:flex;justify-content:center;"><div class="mermaid">${escapeHtml(clean)}</div><script>mermaid.initialize({startOnLoad:true,theme:"dark"});<\/script></body></html>`;
    }
    if (lang === "css") {
      return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>${code}</style></head><body style="margin:20px;font-family:-apple-system,sans-serif;background:#0b0f14;color:#f5f7fa;"><button class="red-button">Preview Button</button><div class="preview-box">CSS Preview</div></body></html>`;
    }
    if (/<html[\s>]/i.test(code)) return code;
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:20px;font-family:-apple-system,sans-serif;background:#0b0f14;color:#f5f7fa;}</style></head><body>${code}</body></html>`;
  }

  function enhanceChatBlocks() {
    document.querySelectorAll(".edge-code-block").forEach((block) => {
      if (block.dataset.higEnhanced === "1") return;
      const langEl = block.querySelector(".edge-code-lang");
      const codeEl = block.querySelector("pre code");
      if (!langEl || !codeEl) return;
      block.dataset.higEnhanced = "1";

      const lang = (langEl.textContent || "text").trim().toLowerCase();
      const rawCode = codeEl.textContent || "";

      langEl.style.display = "flex";
      langEl.style.alignItems = "center";
      langEl.style.justifyContent = "space-between";
      langEl.innerHTML = `<span>${escapeHtml(lang || "code")}</span><span class="hig-code-actions"></span>`;
      const actions = langEl.querySelector(".hig-code-actions");

      const copyBtn = document.createElement("button");
      copyBtn.className = "hig-mini-btn";
      copyBtn.textContent = "Copy";
      copyBtn.onclick = () => {
        navigator.clipboard?.writeText(rawCode);
        copyBtn.textContent = "Copied";
        setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
      };
      actions.appendChild(copyBtn);

      if (["html", "svg", "mermaid", "xml", "css"].includes(lang)) {
        const prevBtn = document.createElement("button");
        prevBtn.className = "hig-mini-btn";
        prevBtn.textContent = "Preview in Canvas";
        prevBtn.onclick = () => window.openAriesCanvas(`${lang.toUpperCase()} Preview`, buildCanvasDoc(lang, rawCode));
        actions.appendChild(prevBtn);
      }

      if (["python", "py", "bash", "sh", "zsh", "js", "javascript"].includes(lang)) {
        const runBtn = document.createElement("button");
        runBtn.className = "hig-mini-btn";
        runBtn.textContent = "Run";
        runBtn.onclick = () => window.runSandboxedCodeBlock(runBtn, lang, encodeURIComponent(rawCode));
        actions.appendChild(runBtn);
      }

      if (window.hljs) {
        try { window.hljs.highlightElement(codeEl); } catch (e) {}
      }

      if (lang === "mermaid" && window.mermaid) {
        const box = document.createElement("div");
        box.className = "hig-mermaid-box";
        const id = "mmd_" + Math.random().toString(36).slice(2, 9);
        window.mermaid.render(id, sanitizeMermaidCode(rawCode)).then((res) => {
          box.innerHTML = res.svg;
          block.insertAdjacentElement("afterend", box);
        }).catch(() => {});
      }

      if (["latex", "math", "katex"].includes(lang) && window.katex) {
        const mathBox = document.createElement("div");
        mathBox.className = "hig-mermaid-box";
        mathBox.innerHTML = window.katex.renderToString(rawCode, { displayMode: true, throwOnError: false });
        block.insertAdjacentElement("afterend", mathBox);
      }
    });

    if (window.katex) {
      document.querySelectorAll(".msg-body").forEach((bodyEl) => {
        const walker = document.createTreeWalker(bodyEl, NodeFilter.SHOW_TEXT);
        const nodes = [];
        let n;
        while ((n = walker.nextNode())) {
          const p = n.parentElement;
          if (!p || p.closest(".edge-code-block, pre, code, .katex, .hig-mermaid-box, script, style")) continue;
          if (/(\$\$|\$|\\\[|\\\()/.test(n.nodeValue || "")) nodes.push(n);
        }
        nodes.forEach((textNode) => {
          const raw = textNode.nodeValue || "";
          const replaced = raw
            .replace(/\$\$([^$]+)\$\$/g, (_, e) => window.katex.renderToString(e, { displayMode: true, throwOnError: false }))
            .replace(/\\\[([\s\S]+?)\\\]/g, (_, e) => window.katex.renderToString(e, { displayMode: true, throwOnError: false }))
            .replace(/\\\(([\s\S]+?)\\\)/g, (_, e) => window.katex.renderToString(e, { displayMode: false, throwOnError: false }))
            .replace(/\$([^$\n]+)\$/g, (_, e) => window.katex.renderToString(e, { displayMode: false, throwOnError: false }));
          if (replaced !== raw) {
            const span = document.createElement("span");
            span.innerHTML = replaced;
            textNode.parentNode?.replaceChild(span, textNode);
          }
        });
      });
    }
  }

  // 5. Initialize on DOMContentLoaded
  document.addEventListener("DOMContentLoaded", async () => {
    startOrbitalAnimation();
    if (window.mermaid) {
      try { window.mermaid.initialize({ startOnLoad: false, theme: "dark" }); } catch (e) {}
    }
    let enhanceTimer = null;
    const obs = new MutationObserver(() => {
      clearTimeout(enhanceTimer);
      enhanceTimer = setTimeout(enhanceChatBlocks, 120);
    });
    obs.observe(document.body, { childList: true, subtree: true });
    enhanceChatBlocks();

    const st = await refreshSetupStatus();
    if (st && !st.all_ready) {
      window.openAriesSetup();
    }
  });
})();
