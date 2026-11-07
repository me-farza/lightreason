/* CPU-only contract tests; fake frames never run generated JavaScript. */
"use strict";
const assert = require("node:assert/strict");
const preview = require("./static/preview.js");
let passed = 0;
function test(name, callback) {
  callback(); passed += 1; process.stdout.write(`PASS ${name}\n`);
}

function fixture(extra = {}) {
  let now = 0, random = 0, nextTimer = 0;
  const jobs = new Map(), listeners = new Set(), writes = [], paints = [], errors = [];
  const frame = { attributes: {}, contentWindow: {}, style: {},
    setAttribute(key, value) { this.attributes[key] = value; },
    set srcdoc(value) { this.last = value; writes.push(value); }, get srcdoc() { return this.last; } };
  const host = {
    performance: { now: () => now },
    crypto: { getRandomValues(values) { for (let i = 0; i < values.length; i++) values[i] = ++random; return values; } },
    addEventListener(name, listener) { assert.equal(name, "message"); listeners.add(listener); },
    removeEventListener(name, listener) { assert.equal(name, "message"); listeners.delete(listener); },
    setTimeout(callback, ms) { const id = ++nextTimer; jobs.set(id, { callback, at: now + ms }); return id; },
    clearTimeout(id) { jobs.delete(id); }
  };
  const status = { textContent: "" }, detail = { textContent: "" };
  const controller = preview.create({ iframe: frame, hostWindow: host,
    statusElement: status, detailElement: detail,
    onPaint: event => paints.push(event), onError: event => errors.push(event), ...extra });
  function identity() {
    const match = /const nonce="([a-f0-9]{32})",renderId=(\d+)/.exec(frame.srcdoc);
    assert.ok(match); return { nonce: match[1], renderId: Number(match[2]) };
  }
  function message(type, extra = {}, outer = {}) {
    const data = { channel: preview.channel, ...identity(), type, ...extra };
    const event = { source: frame.contentWindow, origin: "null", data, ...outer };
    for (const listener of listeners) listener(event);
  }
  function advance(ms) {
    now += ms;
    let ready;
    while ((ready = [...jobs].find(([, job]) => job.at <= now))) {
      jobs.delete(ready[0]); ready[1].callback();
    }
  }
  return { frame, writes, jobs, paints, errors, listeners, status, detail, controller,
    identity, message, advance, setTime(value) { now = value; } };
}

test("extracts a complete HTML fence and excludes explanation", () => {
  const result = preview.extractHTML("Here is your page:\n```HTML\n<!doctype html><h1>Hello</h1>\n```\nExplanation.");
  assert.equal(result.code, "<!doctype html><h1>Hello</h1>");
  assert.equal(result.fenceClosed, true);
});
test("extracts an unfinished HTML fence while streaming", () => {
  const result = preview.extractHTML("```html\n<html><style>body{background:");
  assert.equal(result.code, "<html><style>body{background:");
  assert.equal(result.fenceClosed, false);
});
test("finds HTML after a different complete fenced block", () => {
  const result = preview.extractHTML("```python\nprint('demo')\n```\n```html\n<button>Hi</button>\n```");
  assert.equal(result.code, "<button>Hi</button>");
});
test("recognizes raw documents/fragments and excludes trailing prose", () => {
  assert.equal(preview.extractHTML("Some prose\n<!doctype html><html><body>Hi</body></html>\nExplanation").code,
    "<!doctype html><html><body>Hi</body></html>");
  assert.equal(preview.extractHTML("<main>Hello</main>").supported, true);
  assert.equal(preview.extractHTML("```\n<style>body{color:red}</style><h1>Hello</h1>\n```").supported, true);
});
test("never pretends Python, JS strings, prose or CSS alone are HTML", () => {
  for (const value of ["print('hello')", "```python\nprint('<div>hi</div>')\n```", "Use a <div> element.",
    "const page = '<div>Hello</div>';", "```css\nbody { color:red }\n```", "```js\nconst html = `<main>Hello</main>`;\n```"])
    assert.equal(preview.extractHTML(value).supported, false, value);
});
test("trusted CSP/instrumentation is always before generated content", () => {
  const code = '<meta http-equiv="Content-Security-Policy" content="default-src *"><base href="https://example.invalid/"><script>fetch("https://example.invalid")</script>';
  const document = preview.buildDocument(code, "a".repeat(32), 1);
  assert.ok(document.indexOf(preview.policy) < document.indexOf(code));
  for (const directive of ["default-src 'none'", "connect-src 'none'", "base-uri 'none'", "form-action 'none'", "worker-src 'none'"])
    assert.ok(document.includes(directive));
  assert.ok(document.includes(code), "Generated code remains unmodified inside its sandbox.");
  assert.throws(() => preview.buildDocument("<h1>x</h1>", '"</script>', 1));
});
test("enforces opaque iframe sandbox and never inserts host HTML", () => {
  const f = fixture(); f.controller.update("<main>Hello</main>");
  assert.equal(f.frame.attributes.sandbox, "allow-scripts");
  assert.equal(f.frame.attributes.referrerpolicy, "no-referrer");
  assert.equal(f.writes.length, 1);
  assert.equal("innerHTML" in f.status, false);
});
test("coalesces streaming updates and final render cancels pending work", () => {
  const f = fixture();
  f.controller.update("<main>A</main>"); f.advance(50);
  f.controller.update("<main>AB</main>"); f.advance(50);
  f.controller.update("<main>ABC</main>");
  assert.equal(f.writes.length, 1); assert.equal(f.jobs.size, 1);
  f.advance(200); assert.equal(f.writes.length, 2);
  assert.ok(f.frame.srcdoc.includes("<main>ABC</main>"));
  f.controller.update("<main>ABCD</main>");
  assert.equal(f.jobs.size, 1);
  f.controller.update("<main>ABCDE</main>", { final: true });
  assert.equal(f.writes.length, 3); assert.equal(f.jobs.size, 0);
  f.advance(500); assert.equal(f.writes.length, 3);
});
test("only current opaque-frame messages can update preview timing", () => {
  const f = fixture(); f.controller.update("<main>Hello</main>", { final: true });
  f.message("paint", { hasContent: true }, { source: {} });
  f.message("paint", { hasContent: true }, { origin: "http://127.0.0.1:8765" });
  f.message("paint", { hasContent: true, nonce: "b".repeat(32) });
  assert.equal(f.paints.length, 0);
  f.advance(17); f.message("paint", { hasContent: true });
  assert.equal(f.paints.length, 1); assert.equal(f.paints[0].elapsedMs, 17);
  assert.equal(f.status.textContent, "Preview ready");
  f.message("paint", { hasContent: true }); assert.equal(f.paints.length, 1);
});
test("blank early HTML does not count as first visible preview", () => {
  const f = fixture(); f.controller.update("<!doctype html><html>");
  f.message("paint", { hasContent: false }); assert.equal(f.paints.length, 0);
  assert.equal(f.status.textContent, "Waiting for page content…");
  f.message("paint", { hasContent: true }); assert.equal(f.paints.length, 1);
});
test("partial script errors stay updating; final script errors are explicit", () => {
  const f = fixture(); f.controller.update("<main>x</main><script>let x=");
  f.message("error", { message: "Unexpected end of input" });
  assert.equal(f.status.textContent, "Updating preview…"); assert.equal(f.errors[0].final, false);
  f.controller.update("<main>x</main><script>let x=</script>", { final: true });
  f.message("error", { message: "Unexpected token <script>" }); f.message("paint", { hasContent: true });
  assert.equal(f.status.textContent, "Render error"); assert.equal(f.detail.textContent, "Unexpected token <script>");
  assert.equal(f.errors[1].final, true);
});
test("a context-limited response is labeled partial even if HTML parses", () => {
  const f = fixture(); f.controller.update("<main>Partial</main>", { final: true, complete: false, stopReason: "context_limit" });
  f.message("paint", { hasContent: true });
  assert.equal(f.status.textContent, "Preview from partial code"); assert.equal(f.paints[0].complete, false);
});
test("begin preserves previous page, resets extraction and ignores stale paint", () => {
  const f = fixture(); f.controller.update("<main>Old</main>", { final: true });
  const old = f.identity(), previous = f.frame.srcdoc;
  f.controller.begin(); assert.equal(f.frame.srcdoc, previous); assert.equal(f.controller.getCode(), "");
  f.message("paint", { ...old, hasContent: true }); assert.equal(f.paints.length, 0);
  f.controller.update("I am updating your page..."); assert.equal(f.controller.getCode(), "");
  assert.equal(f.frame.srcdoc, previous);
  f.controller.update("<main>New</main>", { final: true });
  f.message("paint", { ...old, hasContent: true }); assert.equal(f.paints.length, 0);
  f.message("paint", { hasContent: true }); assert.equal(f.paints.length, 1);
});
test("unsupported final code invalidates old messages without masquerading as HTML", () => {
  const f = fixture(); f.controller.update("<main>Old</main>");
  f.controller.update("```python\nprint('hi')\n```", { final: true });
  f.message("paint", { hasContent: true });
  assert.equal(f.status.textContent, "No web preview"); assert.equal(f.paints.length, 0);
  assert.equal(f.controller.getCode(), "");
});
test("reset/dispose clear frame and stale work", () => {
  const f = fixture(); f.controller.update("<main>Old</main>");
  f.controller.update("<main>New</main>"); assert.equal(f.jobs.size, 1);
  f.controller.reset(); assert.equal(f.frame.srcdoc, ""); assert.equal(f.jobs.size, 0);
  f.controller.update("<main>Final</main>"); f.controller.dispose();
  assert.equal(f.frame.srcdoc, ""); assert.equal(f.listeners.size, 0);
  const count = f.writes.length; f.controller.update("<main>Ignored</main>"); assert.equal(f.writes.length, count);
});
test("recognizes structurally unfinished full documents without running code", () => {
  const result = preview.analyzeHTML('<!doctype html><html><head><style>body{color:red}</style></head><body><button>Hello</button><script>let n=');
  assert.equal(result.complete, false);
  assert.deepEqual(result.issues, ["Missing </script>", "Missing </body>", "Missing </html>"]);
  assert.deepEqual(preview.analyzeHTML('<!doctype html><html><body><div title="unfinished').issues,
    ["Unfinished <div> tag", "Missing </body>", "Missing </html>"]);
  assert.equal(preview.analyzeHTML("<!doctype html").complete, false);
});
test("does not mistake raw text and comments for document closing tags", () => {
  const result = preview.analyzeHTML('<!doctype html><html><head><style>body:after{content:"<body>"}</style></head><body><!-- <html><body> --><script>const x="</body></html>";</script></body></html>');
  assert.equal(result.complete, true);
  assert.equal(preview.analyzeHTML('<main>Hello</main>').complete, true);
  assert.equal(preview.analyzeHTML('<!doctype html><h1>Optional enclosing tags</h1>').complete, true);
  assert.equal(preview.analyzeHTML('<main>Hello</main><style>body{color:red').complete, false);
});
test("an EOS is insufficient for Preview ready when HTML is unfinished", () => {
  const f = fixture();
  f.controller.update('<!doctype html><html><body><button>Hello</button>', { final: true, complete: true });
  assert.equal(f.status.textContent, "Incomplete HTML");
  f.message("paint", { hasContent: true });
  assert.equal(f.status.textContent, "Incomplete HTML");
  assert.equal(f.paints[0].complete, false);
  assert.equal(f.controller.getState().ready, false);
  assert.equal(f.controller.getState().modelComplete, true);
  assert.deepEqual(f.controller.getState().structuralIssues, ["Missing </body>", "Missing </html>"]);
});
test("getState distinguishes parsed content, runtime errors, and restrictions", () => {
  const f = fixture(); f.controller.update("<main>Hello</main>", { final: true });
  assert.equal(f.controller.getState().ready, false);
  f.message("paint", { hasContent: true }); assert.equal(f.controller.getState().ready, true);
  f.message("error", { message: "Actual JavaScript error" });
  assert.equal(f.controller.getState().renderError, "Actual JavaScript error");
  assert.equal(f.controller.getState().ready, false);
  f.controller.update("<main>Hello again</main>", { final: true });
  f.message("blocked", { message: "External dependency blocked" });
  f.message("paint", { hasContent: true });
  assert.equal(f.controller.getState().ready, false);
  assert.equal(f.paints[f.paints.length - 1].hasRestrictions, true);
});
test("viewport calculations preserve true device width while fitting pane", () => {
  const desktop = preview.calculateViewport({ width: 1024, fit: true, availableWidth: 420, availableHeight: 600 });
  assert.equal(desktop.frameWidth, 1024); assert.equal(desktop.scale, 420 / 1024);
  assert.equal(desktop.frameWidth * desktop.scale, 420);
  assert.ok(desktop.frameHeight * desktop.scale >= 600);
  const responsive = preview.calculateViewport({ width: null, fit: true, availableWidth: 420, availableHeight: 600 });
  assert.equal(responsive.frameWidth, 420); assert.equal(responsive.scale, 1);
  const full = preview.calculateViewport({ width: 1024, fit: false, availableWidth: 420 });
  assert.equal(full.frameWidth, 1024); assert.equal(full.scale, 1);
  assert.throws(() => preview.calculateViewport({ width: 1000000 }));
});
test("ResizeObserver fits presets and responsive pages without rerendering code", () => {
  const container = { clientWidth: 500, clientHeight: 600, style: {} };
  let observer;
  class Observer {
    constructor(callback) { this.callback = callback; this.disconnected = false; observer = this; }
    observe(value) { assert.equal(value, container); }
    disconnect() { this.disconnected = true; }
  }
  const f = fixture({ viewportContainer: container, ResizeObserver: Observer });
  assert.equal(f.frame.style.width, "500px");
  f.controller.update("<main>Keep this interactive page</main>", { final: true });
  const count = f.writes.length;
  const fitted = f.controller.viewport({ width: 1024, fit: true });
  assert.equal(fitted.scale, 500 / 1024); assert.equal(f.frame.style.width, "1024px");
  assert.equal(container.style.overflow, "hidden");
  container.clientWidth = 375; observer.callback();
  assert.equal(f.controller.getState().viewport.scale, 375 / 1024);
  f.controller.viewport({ fit: false }); assert.equal(f.frame.style.transform, "none");
  assert.equal(container.style.overflow, "auto");
  f.controller.viewport({ width: null, fit: true });
  assert.equal(f.frame.style.width, "375px"); assert.equal(f.frame.style.transform, "none");
  assert.equal(f.writes.length, count, "Changing the viewport must not rerun generated JavaScript.");
  assert.throws(() => f.controller.viewport({ fit: "yes" }));
  f.controller.dispose(); assert.equal(observer.disconnected, true);
});
test("hidden views retain dimensions until they become visible", () => {
  const container = { clientWidth: 500, clientHeight: 600, style: {} };
  let resize;
  class Observer { constructor(callback) { resize = callback; } observe() {} disconnect() {} }
  const f = fixture({ viewportContainer: container, ResizeObserver: Observer });
  f.controller.viewport({ width: 768, fit: true });
  container.clientWidth = 0; resize();
  assert.equal(f.controller.getState().viewport.availableWidth, 500);
  assert.equal(f.frame.style.width, "768px");
  container.clientWidth = 320; resize();
  assert.equal(f.controller.getState().viewport.availableWidth, 320);
  assert.equal(f.controller.getState().viewport.scale, 320 / 768);
});
process.stdout.write(`${passed} preview contract tests passed.\n`);
