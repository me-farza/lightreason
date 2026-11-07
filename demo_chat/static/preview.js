/* Real browser rendering for generated single-file HTML/CSS/JavaScript.
 * Generated content is assigned only to an opaque, sandboxed iframe srcdoc.
 * The host must additionally send CSP `frame-src about:` to prevent the
 * sandboxed frame navigating itself to a network URL.
 */
(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LightreasonPreview = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  const CHANNEL = "lightreason-preview-v1";
  const POLICY = [
    "default-src 'none'", "script-src 'unsafe-inline'", "style-src 'unsafe-inline'",
    "img-src data: blob:", "font-src data:", "connect-src 'none'", "frame-src 'none'",
    "worker-src 'none'", "object-src 'none'", "base-uri 'none'", "form-action 'none'",
    "media-src data: blob:", "manifest-src 'none'", "prefetch-src 'none'", "navigate-to 'none'"
  ].join("; ");

  function extractHTML(rawText) {
    const raw = String(rawText == null ? "" : rawText).replace(/^\uFEFF/, "");
    // Inspect all opening fences because a prose preamble can include another
    // language. Only an explicitly HTML fence, or a clearly HTML bare fence,
    // is a renderable website. Unclosed fences are useful during streaming.
    const fence = /(?:^|\n)[ \t]*(`{3,}|~{3,})([^\n\r]*)[\r]?\n/g;
    let match;
    while ((match = fence.exec(raw))) {
      const marker = match[1];
      const language = match[2].trim().toLowerCase().split(/\s+/)[0];
      const start = fence.lastIndex;
      const closing = new RegExp("(?:^|\\n)[ \\t]*" + marker[0] + "{" + marker.length + ",}[ \\t]*(?=\\r?$)", "m");
      const tail = raw.slice(start);
      const end = closing.exec(tail);
      const code = (end ? tail.slice(0, end.index) : tail).trim();
      if (language === "html" || language === "htm" ||
          (!language && /^\s*(?:<!doctype\s+html\b|<(?:html|head|body|style|main|section|div|header|article|button|canvas)\b)/i.test(code))) {
        return { code, language: "html", fenced: true, fenceClosed: !!end, supported: code.length > 0 };
      }
      // Skip a complete non-HTML block rather than accidentally treating a
      // line inside its string literals as the next opening fence.
      if (end) fence.lastIndex = start + end.index + end[0].length;
      else break;
    }
    const rawStart = /(?:^|\n)[ \t]*(<!doctype\s+html\b|<(?:html|head|body|style|main|section|div|header|article|button|canvas)\b)/i.exec(raw);
    const start = rawStart ? rawStart.index + rawStart[0].length - rawStart[1].length : -1;
    if (start >= 0 && !/^[\s\S]*?(?:^|\n)[ \t]*`{3,}\s*(?:python|javascript|js|typescript|tsx|jsx|css|sh|bash|json)\b/im.test(raw.slice(0, start))) {
      let code = raw.slice(start).trim();
      // A full document's closing HTML tag is an unambiguous boundary before
      // a model's explanatory prose. Fragments remain literal code.
      const closes = Array.from(code.matchAll(/<\/html\s*>/ig));
      const close = closes.length ? closes[closes.length - 1] : null;
      if (close) code = code.slice(0, close.index + close[0].length);
      return { code, language: "html", fenced: false, fenceClosed: false, supported: true };
    }
    return { code: "", language: null, fenced: false, fenceClosed: false, supported: false };
  }

  function analyzeHTML(value) {
    const code = String(value || "");
    const issues = [];
    const opened = new Set(), closed = new Set();
    let fullDocument = false;
    let position = 0;
    function tagEnd(start) {
      let quote = null;
      for (let i = start; i < code.length; i += 1) {
        const char = code[i];
        if (quote) { if (char === quote) quote = null; }
        else if (char === '"' || char === "'") quote = char;
        else if (char === ">") return i;
      }
      return -1;
    }
    while (position < code.length) {
      const start = code.indexOf("<", position);
      if (start < 0) break;
      if (code.slice(start, start + 4) === "<!--") {
        const end = code.indexOf("-->", start + 4);
        if (end < 0) { issues.push("Unfinished HTML comment"); break; }
        position = end + 3; continue;
      }
      const tail = code.slice(start);
      if (/^<!doctype\s+html\b/i.test(tail)) {
        fullDocument = true;
        const end = tagEnd(start + 2);
        if (end < 0) { issues.push("Unfinished HTML doctype"); break; }
        position = end + 1; continue;
      }
      if (code.slice(start, start + 9) === "<![CDATA[") {
        const end = code.indexOf("]]>", start + 9);
        if (end < 0) { issues.push("Unfinished CDATA section"); break; }
        position = end + 3; continue;
      }
      const tag = /^<(\/?)([a-z][\w:-]*)(?=[\s/>]|$)/i.exec(tail);
      if (!tag) { position = start + 1; continue; }
      const name = tag[2].toLowerCase();
      const end = tagEnd(start + tag[0].length);
      if (end < 0) { issues.push(`Unfinished <${tag[1]}${name}> tag`); break; }
      const closing = tag[1] === "/";
      if (closing) closed.add(name);
      else opened.add(name);
      if (name === "html") fullDocument = true;
      position = end + 1;
      if (!closing && ["script", "style", "textarea", "title"].includes(name)) {
        // HTML raw-text closing rules apply even inside JavaScript strings.
        // Ignore apparent html/body tags inside these raw-text elements.
        const closingTag = new RegExp("<\\/" + name + "(?=[\\s/>])", "ig");
        closingTag.lastIndex = position;
        const rawEnd = closingTag.exec(code);
        if (!rawEnd) { issues.push(`Missing </${name}>`); break; }
        const rawEndClose = tagEnd(rawEnd.index + rawEnd[0].length);
        if (rawEndClose < 0) { issues.push(`Unfinished </${name}> tag`); break; }
        closed.add(name);
        position = rawEndClose + 1;
      }
    }
    if (fullDocument) {
      // Optional HTML tag omission remains supported: require closing tags
      // only for html/head/body elements explicitly opened by the source.
      for (const name of ["head", "body", "html"]) {
        if (opened.has(name) && !closed.has(name)) issues.push(`Missing </${name}>`);
      }
    }
    return { fullDocument, complete: issues.length === 0, issues: [...new Set(issues)] };
  }

  function calculateViewport(settings) {
    const width = settings.width == null ? null : settings.width;
    if (width !== null && ![375, 768, 1024].includes(width)) throw new Error("Choose responsive, 375, 768, or 1024 pixel preview width");
    const fit = settings.fit !== false;
    const availableWidth = Math.max(1, Number(settings.availableWidth) || width || 600);
    const availableHeight = Math.max(200, Number(settings.availableHeight) || 600);
    const frameWidth = width || availableWidth;
    const scale = width && fit ? Math.min(1, availableWidth / frameWidth) : 1;
    return { width, fit, availableWidth, availableHeight, frameWidth,
      frameHeight: Math.ceil(availableHeight / scale), scale };
  }

  function buildDocument(code, nonce, renderId) {
    if (!/^[a-f0-9]{32}$/.test(nonce) || !Number.isSafeInteger(renderId) || renderId < 1) {
      throw new Error("Invalid preview render identity");
    }
    const instrumentation = `(function(){"use strict";
const nonce=${JSON.stringify(nonce)},renderId=${renderId},channel=${JSON.stringify(CHANNEL)};
function send(type,details){parent.postMessage(Object.assign({channel:channel,nonce:nonce,renderId:renderId,type:type},details||{}),"*");}
addEventListener("error",function(e){send("error",{message:String(e.message||"The generated page raised an error.").slice(0,1000)});});
addEventListener("unhandledrejection",function(e){let r=e.reason;send("error",{message:String(r&&r.message||r||"Unhandled JavaScript rejection").slice(0,1000)});});
addEventListener("securitypolicyviolation",function(e){send("blocked",{directive:String(e.effectiveDirective||"resource"),message:"This preview uses local inline code only; an external resource was blocked."});});
document.addEventListener("submit",function(e){e.preventDefault();},true);
document.addEventListener("click",function(e){let a=e.target&&e.target.closest&&e.target.closest("a[href]");if(a&&!/^#/.test(a.getAttribute("href")||"")){e.preventDefault();}},true);
let observer=null,paintPending=false,contentPainted=false;
function hasContent(){let b=document.body;if(!b)return false;if((b.innerText||"").trim())return true;let all=b.querySelectorAll("*");for(let i=0;i<all.length;i++){let e=all[i];if(/^(SCRIPT|STYLE|META|LINK|TITLE)$/.test(e.tagName))continue;let r=e.getBoundingClientRect(),s=getComputedStyle(e);if(r.width>0&&r.height>0&&s.display!=="none"&&s.visibility!=="hidden"&&s.opacity!=="0")return true;}let s=getComputedStyle(b);return s.backgroundColor!=="rgba(0, 0, 0, 0)"&&s.backgroundColor!=="transparent";}
function painted(){if(paintPending||contentPainted)return;paintPending=true;requestAnimationFrame(function(){requestAnimationFrame(function(){paintPending=false;let visible=hasContent();send("paint",{hasContent:visible});if(visible){contentPainted=true;if(observer)observer.disconnect();}});});}
function ready(){painted();observer=new MutationObserver(painted);if(document.body)observer.observe(document.body,{childList:true,subtree:true,attributes:true,characterData:true});}
if(document.readyState==="loading")addEventListener("DOMContentLoaded",ready,{once:true});else ready();
})();`;
    // Our CSP and instrumentation are parsed before any generated byte. Extra
    // generated meta policies can only restrict this policy, never relax it.
    // Nested html/head/body tags are handled by the browser's HTML parser.
    return "<!doctype html><html><head><meta charset=\"utf-8\">" +
      "<meta http-equiv=\"Content-Security-Policy\" content=\"" + POLICY + "\">" +
      "<meta name=\"referrer\" content=\"no-referrer\">" +
      "<meta http-equiv=\"x-dns-prefetch-control\" content=\"off\">" +
      "<script>" + instrumentation + "<\/script></head><body>" + String(code) + "</body></html>";
  }

  function create(options) {
    const iframe = options && options.iframe;
    if (!iframe || typeof iframe.setAttribute !== "function") throw new Error("A preview iframe is required");
    const host = options.hostWindow || window;
    const clock = options.clock || (() => host.performance.now());
    const delay = Number.isFinite(options.throttleMs) ? Math.max(50, options.throttleMs) : 300;
    const timers = options.timers || host;
    iframe.setAttribute("sandbox", "allow-scripts");
    iframe.setAttribute("referrerpolicy", "no-referrer");
    iframe.setAttribute("allow", "camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'");

    let disposed = false;
    let timer = null;
    let generation = 0;
    let renderId = 0;
    let nonce = null;
    let latestCode = "";
    let lastRenderedCode = null;
    let lastRenderStarted = 0;
    let latest = { final: false, modelComplete: true, stopReason: null };
    let structure = analyzeHTML("");
    let active = null;
    let viewportSettings = { width: null, fit: true };
    let viewportState = calculateViewport(viewportSettings);
    let resizeObserver = null;

    function applyViewport() {
      const container = options.viewportContainer;
      if (!container || !iframe.style || !container.style) return viewportState;
      const width = Number(container.clientWidth);
      // A hidden studio view is not a zero-width device. ResizeObserver will
      // apply its actual dimensions once the view becomes visible again.
      if (!(width > 0)) return viewportState;
      viewportState = calculateViewport({ ...viewportSettings, availableWidth: width,
        availableHeight: Number(container.clientHeight) || 600 });
      container.style.position = "relative";
      container.style.minWidth = "0";
      container.style.overflow = viewportSettings.fit ? "hidden" : "auto";
      iframe.style.position = "absolute";
      iframe.style.top = "0"; iframe.style.left = "0";
      iframe.style.minWidth = "0"; iframe.style.minHeight = "0";
      iframe.style.width = `${viewportState.frameWidth}px`;
      iframe.style.height = `${viewportState.frameHeight}px`;
      iframe.style.transformOrigin = "top left";
      iframe.style.transform = viewportState.scale === 1 ? "none" : `scale(${viewportState.scale})`;
      return viewportState;
    }

    function viewport(settings) {
      settings = settings || {};
      const width = Object.prototype.hasOwnProperty.call(settings, "width") ? settings.width : viewportSettings.width;
      const fit = Object.prototype.hasOwnProperty.call(settings, "fit") ? settings.fit : viewportSettings.fit;
      if (typeof fit !== "boolean") throw new Error("Use an explicit preview fit setting");
      calculateViewport({ width, fit });
      viewportSettings = { width: width == null ? null : width, fit };
      viewportState = { ...viewportState, ...viewportSettings };
      return { ...applyViewport() };
    }

    function getState() {
      return { code: latestCode, final: latest.final, modelComplete: latest.modelComplete,
        complete: latest.modelComplete && structure.complete, structuralIssues: [...structure.issues],
        fullDocument: structure.fullDocument, renderError: active && active.error || null,
        resourceRestriction: active && active.blocked || null,
        painted: !!(active && active.painted),
        ready: !!(active && active.final && active.complete && active.painted && !active.error && !active.blocked),
        renderId: active && active.renderId || null, viewport: { ...viewportState } };
    }

    function status(label, detail) {
      if (options.statusElement) options.statusElement.textContent = label;
      if (options.detailElement) options.detailElement.textContent = detail || "";
    }

    function cancelTimer() {
      if (timer !== null) timers.clearTimeout(timer);
      timer = null;
    }

    function freshNonce() {
      const crypto = options.crypto || host.crypto;
      if (!crypto || typeof crypto.getRandomValues !== "function") throw new Error("Secure browser randomness is required for the preview");
      const words = crypto.getRandomValues(new Uint32Array(4));
      return Array.from(words, value => value.toString(16).padStart(8, "0")).join("");
    }

    function render() {
      cancelTimer();
      if (disposed || !latestCode) return;
      const key = generation;
      const code = latestCode;
      nonce = freshNonce();
      renderId += 1;
      lastRenderStarted = clock();
      lastRenderedCode = code;
      active = { nonce, renderId, generation: key, code, final: latest.final,
                 modelComplete: latest.modelComplete, complete: latest.modelComplete && structure.complete,
                 structuralIssues: [...structure.issues], stopReason: latest.stopReason,
                 started: lastRenderStarted, painted: false, error: null, blocked: null };
      if (latest.final && !structure.complete) status("Incomplete HTML", structure.issues.join(" · ") + ". The available code is shown without inventing the missing content.");
      else status(latest.final ? "Rendering preview…" : "Updating preview…", "Rendering the generated HTML, CSS and JavaScript in the browser.");
      iframe.srcdoc = buildDocument(code, nonce, renderId);
      applyViewport();
    }

    function onMessage(event) {
      if (disposed || !active || event.source !== iframe.contentWindow || event.origin !== "null") return;
      const data = event.data;
      if (!data || data.channel !== CHANNEL || data.nonce !== active.nonce ||
          data.renderId !== active.renderId || active.generation !== generation) return;
      if (data.type === "error") {
        const message = typeof data.message === "string" ? data.message.slice(0, 1000) : "The generated JavaScript raised an error.";
        active.error = message;
        status(active.final ? "Render error" : "Updating preview…", active.final ? message : "The streamed code is still incomplete. The browser will retry as more code arrives.");
        if (typeof options.onError === "function") options.onError({ message, final: active.final, renderId: active.renderId });
      } else if (data.type === "blocked") {
        active.blocked = typeof data.message === "string" ? data.message.slice(0, 1000) : "An external resource was blocked.";
        if (active.final && !active.error && active.structuralIssues.length === 0) status("Preview rendered with restrictions", active.blocked);
      } else if (data.type === "paint" && !active.painted) {
        if (data.hasContent !== true) {
          if (!active.error && !(active.final && active.structuralIssues.length)) status(active.final ? "No visible page yet" : "Waiting for page content…", "The browser parsed the HTML, but no visible page content has appeared yet.");
          return;
        }
        active.painted = true;
        const elapsedMs = Math.max(0, clock() - active.started);
        if (!active.error) {
          if (!active.final) status("Live preview", "Updating as the model generates more code.");
          else if (active.structuralIssues.length) status("Incomplete HTML", active.structuralIssues.join(" · ") + ". The browser rendered only the available code.");
          else if (!active.complete) status("Preview from partial code", "Generation reached its context or output limit. The browser rendered the available code.");
          else if (active.blocked) status("Preview rendered with restrictions", active.blocked);
          else status("Preview ready", "Generated HTML, CSS and JavaScript rendered in the browser.");
        }
        if (typeof options.onPaint === "function") options.onPaint({ elapsedMs, renderId: active.renderId,
          final: active.final, complete: active.complete, code: active.code, hasError: !!active.error,
          structuralIssues: [...active.structuralIssues], hasRestrictions: !!active.blocked, hasContent: true });
      }
    }
    host.addEventListener("message", onMessage);

    function update(rawText, state) {
      if (disposed) return;
      state = state || {};
      latest = { final: state.final === true, modelComplete: state.complete !== false, stopReason: state.stopReason || null };
      const extracted = extractHTML(rawText);
      latestCode = extracted.supported ? extracted.code : "";
      structure = analyzeHTML(latestCode);
      if (!latestCode) {
        cancelTimer();
        active = null;
        nonce = null;
        // Leave the previous rendered page visible during a new code preamble.
        // It is never returned as the newly generated code.
        status(latest.final ? "No web preview" : "Waiting for HTML…", latest.final
          ? "This response does not contain a single-file HTML page. The generated text remains visible in the code panel."
          : "The current page stays visible until new HTML arrives.");
        return;
      }
      if (latestCode === lastRenderedCode && active && active.generation === generation && !latest.final) return;
      if (latest.final) return render();
      const elapsed = clock() - lastRenderStarted;
      if (!active || active.generation !== generation || elapsed >= delay) render();
      else if (timer === null) {
        const currentGeneration = generation;
        timer = timers.setTimeout(() => {
          timer = null;
          if (currentGeneration === generation) render();
        }, Math.max(0, delay - elapsed));
      }
    }

    function begin(settings) {
      if (disposed) return;
      cancelTimer();
      generation += 1;
      nonce = null;
      active = null;
      latestCode = "";
      lastRenderedCode = null;
      lastRenderStarted = 0;
      latest = { final: false, modelComplete: true, stopReason: null };
      structure = analyzeHTML("");
      status("Waiting for HTML…", "The current page stays visible until the next generated page arrives.");
      if (settings && settings.keepPreview === false) iframe.srcdoc = "";
    }

    function reset() { begin({ keepPreview: false }); }
    function dispose() {
      if (disposed) return;
      cancelTimer();
      disposed = true;
      active = null;
      if (resizeObserver) resizeObserver.disconnect();
      host.removeEventListener("message", onMessage);
      iframe.srcdoc = "";
    }
    const Observer = options.ResizeObserver || host.ResizeObserver;
    if (options.viewportContainer && typeof Observer === "function") {
      resizeObserver = new Observer(applyViewport);
      resizeObserver.observe(options.viewportContainer);
    }
    applyViewport();
    status("Waiting for HTML…", "Ask for a self-contained HTML page with inline CSS and JavaScript.");
    return { update, begin, reset, getCode: () => latestCode, getState, viewport, dispose };
  }

  return { create, extractHTML, analyzeHTML, calculateViewport, buildDocument, policy: POLICY, channel: CHANNEL };
});
