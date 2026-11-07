"use strict";

const $ = (id) => document.getElementById(id);
const form = $("compare-form");
const runButton = $("run-button");
const modeSelect = $("mode");
const errorBox = $("error");
const WEB_PROMPT = "Build a compact weather dashboard as one complete HTML file with inline CSS and JavaScript. Use a dark theme, a city search field and sample weather data. Keep it concise. Return only the HTML.";
const DEFAULT_DESIGN_PROMPT = "You build responsive browser apps. Return one complete HTML document in a single html code block, with inline CSS and JavaScript. Use no external libraries, fonts, images, APIs or network calls. Make visible controls work. Use box-sizing:border-box, responsive layouts, width:100% and max-width constraints; avoid fixed minimum widths and horizontal overflow. For a new app, build the requested app from scratch. For an edit, return the complete updated document. Keep code concise.";
// Keep this instruction in sync with JSON_PATCH_SYSTEM_PROMPT in edit_patch.py.
const JSON_PATCH_SYSTEM_PROMPT = 'Edit a self-contained HTML page. Return only a JSON object with keys "search" and "replace", both strings. ' +
  'The search string must be a small exact substring that occurs once in the original HTML. ' +
  'The replace string contains just the updated code. Preserve all unrelated code. ' +
  'For example, {"search":"count += 1;","replace":"count += 3;"} changes the increment by replacing one statement. ' +
  'For behavior changes, find and edit the existing JavaScript statement. Avoid new scripts or duplicate handlers. ' +
  'If a small edit cannot implement the request, return {"rewrite":true}.';
let designSystemPrompt = DEFAULT_DESIGN_PROMPT;
let designSystemEnabled = true;
const MODEL_NAME = "Qwen2.5-3B-Instruct";
const RUNNER_NAMES = { original: "Qwen · Standard runner", optimized: "Qwen · Lightreason Turbo", strict: "Qwen · Lightreason Exact" };
const RUNNER_LABELS = { original: "Standard runner", optimized: "Lightreason Turbo", strict: "Lightreason Exact" };
let viewMode = "build";
const viewSettings = { build: { prompt: WEB_PROMPT, mode: "optimized", budget: "auto" }, compare: { prompt: WEB_PROMPT, mode: "optimized", budget: "auto" } };
const resultByView = { build: null, compare: null };
let inFlight = false;
let serviceReady = false;
let serviceBusy = false;
let lastResult = null;
let automaticSetup = null;
let currentEvents = [];
let currentRequest = null;
let currentStatus = {};
const outputs = { baseline: "", candidate: "", single: "" };
const clientFirstToken = { baseline: null, candidate: null, single: null };
let requestSentAt = null;
let clientTotalS = null;
let acceptedPageCode = "";
let acceptedPageComplete = true;
let singleCompleted = false;
let previewRun = null;
let previewRunCounter = 0;
const updateHistory = [];

function formatNumber(value, digits = 2) {
  return Number.isFinite(Number(value)) && value !== null ? Number(value).toFixed(digits) : "—";
}

function setMetric(id, value, unit) {
  const el = $(id);
  el.replaceChildren(document.createTextNode(String(value)));
  const small = document.createElement("small");
  small.textContent = ` ${unit}`;
  el.append(small);
}

function setActivity(message) { if (message) $("activity").textContent = message; }
function showError(message) { errorBox.textContent = message; errorBox.hidden = false; }
function runnerName(mode = modeSelect.value) { return RUNNER_NAMES[mode] || "Lightreason"; }

function renderOutput(runner, text) {
  const el = runner === "single" ? $("single-code") : $(`${runner}-output`);
  const follow = el.scrollHeight - el.clientHeight - el.scrollTop < 40;
  if (runner === "single") el.value = text;
  else el.textContent = text;
  if (follow) el.scrollTop = el.scrollHeight;
}

function selectedRequest() {
  const action = $("app-action").value;
  const patch = action === "patch" && viewMode === "build";
  const request = { prompt: $("prompt").value.trim(), app_action: patch ? "edit" : action, system_prompt: designSystemEnabled ? designSystemPrompt.trim() : "", mode: modeSelect.value, execution: "sequential", max_new_tokens: $("max-tokens").value === "auto" ? null : Number($("max-tokens").value) };
  if (request.app_action === "edit" && acceptedPageCode) request.previous_code = acceptedPageCode;
  if (patch) {
    request.patch_format = "json";
    request.system_prompt = JSON_PATCH_SYSTEM_PROMPT;
  }
  return request;
}

function patchWireRequest(request) {
  const source = request.previous_code;
  if (!preview || !source || !acceptedPageComplete) throw new Error("Patch edit needs a complete rendered page.");
  const editorCode = window.LightreasonPreview.extractHTML($("single-code").value).code;
  if (editorCode !== source.replace(/\r\n?/g, "\n")) throw new Error("Render your local code edits before requesting a patch.");
  const longestBackticks = (source.match(/`+/g) || []).reduce((length, run) => Math.max(length, run.length), 0);
  const fence = "`".repeat(Math.max(3, longestBackticks + 1));
  const wire = { ...request, app_action: "new", prompt: `Current HTML:\n${fence}html\n${source}\n${fence}\n\nRequested edit:\n${request.prompt}` };
  delete wire.previous_code;
  delete wire.patch_format;
  return wire;
}

function applyPatchResult(source, response) {
  if (typeof response !== "string") throw new Error("The patch response is missing its JSON text.");
  const text = response.replace(/\r\n/g, "\n").trim();
  const fenced = /^```(?:json)?\n([\s\S]*)\n```$/.exec(text);
  const raw = fenced ? fenced[1].trim() : text;
  let fields;
  try { fields = JSON.parse(raw); }
  catch (_) { throw new Error("Qwen did not return a valid JSON patch."); }
  if (fields?.rewrite === true) throw new Error("Qwen requested a full rewrite. Choose Edit current app for this change.");
  // JSON.parse accepts duplicate keys. Require exactly two string pairs in
  // addition to validating the parsed fields so duplicate keys cannot pass.
  const jsonString = /"(?:[^"\\]|\\[\s\S])*"/.source;
  const pair = `${jsonString}\\s*:\\s*${jsonString}`;
  const twoPairs = new RegExp(`^\\{\\s*${pair}\\s*,\\s*${pair}\\s*\\}$`);
  if (!twoPairs.test(raw) || typeof fields?.search !== "string" || typeof fields?.replace !== "string" || !fields.search.trim()) {
    throw new Error("Expected one nonempty search string and one replacement string.");
  }
  const { search, replace } = fields;
  const start = source.indexOf(search);
  if (start < 0 || source.indexOf(search, start + 1) !== -1) throw new Error("The patch must match exactly one place in the current HTML.");
  const updated = source.slice(0, start) + replace + source.slice(start + search.length);
  if (updated.trim() === source) throw new Error("The patch made no change to the page.");
  const { extractHTML, analyzeHTML } = window.LightreasonPreview;
  if (extractHTML(updated).code !== updated.trim() || !updated.trim() || !analyzeHTML(updated).complete) {
    throw new Error("The patch would leave the page with incomplete or unsupported HTML.");
  }
  return updated;
}

function updateButton() {
  runButton.disabled = inFlight || !serviceReady || serviceBusy || !modeSelect.value || !$("prompt").value.trim();
  const buildLabel = $("app-action").value === "patch" ? "Generate patch" : "Generate page";
  $("run-button-text").textContent = inFlight ? (viewMode === "build" ? "Generating…" : "Comparing…") : serviceBusy ? "GPU is busy" : serviceReady ? (viewMode === "build" ? buildLabel : "Run comparison") : "Starting Qwen…";
  $("render-code").disabled = inFlight || !$("single-code").value.trim();
  $("clear-page").disabled = inFlight || (!acceptedPageCode && !$("single-code").value);
}

function lockControls(locked) {
  for (const id of ["mode", "max-tokens", "prompt", "app-action", "design-button", "view-build", "view-compare", "single-code"]) $(id).disabled = locked;
  for (const button of document.querySelectorAll("[data-prompt]")) button.disabled = locked;
}

function updateMode() {
  const label = RUNNER_LABELS[modeSelect.value] || "Lightreason";
  $("candidate-title").textContent = MODEL_NAME;
  $("candidate-kicker").textContent = label;
  $("candidate-output").setAttribute("aria-label", `${MODEL_NAME} through ${label}`);
  if (viewMode === "build" && (!resultByView.build || inFlight)) {
    $("single-title").textContent = MODEL_NAME;
    $("single-runner-label").textContent = label;
  }
  const descriptions = {
    original: "Qwen2.5-3B-Instruct uses the standard runner. Browser rendering is timed separately from generation.",
    strict: "Qwen2.5-3B-Instruct uses Lightreason Exact. This runner preserved token outputs in our previous tests; Compare runners checks matching for your prompt.",
    optimized: "Qwen2.5-3B-Instruct uses Lightreason Turbo. Its execution can produce floating-point differences; Compare runners reports exact token matching."
  };
  $("mode-note").textContent = descriptions[modeSelect.value];
}

function saveViewSettings() { viewSettings[viewMode] = { prompt: $("prompt").value, mode: modeSelect.value, budget: $("max-tokens").value }; }
function syncModeOptions() {
  const available = currentStatus.available_modes;
  const selected = modeSelect.value;
  for (const option of modeSelect.options) {
    const allowed = (viewMode === "build" || option.value !== "original") &&
      (!Array.isArray(available) || available.includes(option.value));
    option.hidden = !allowed;
    option.disabled = !allowed;
  }
  if (!modeSelect.selectedOptions[0] || modeSelect.selectedOptions[0].disabled) {
    const enabled = [...modeSelect.options].filter(option => !option.disabled);
    modeSelect.value = enabled.find(option => option.value === "strict")?.value || enabled[0]?.value || "";
  }
  return modeSelect.value !== selected;
}

function setView(view) {
  if (inFlight || view === viewMode) return;
  saveViewSettings(); viewMode = view;
  const settings = viewSettings[view];
  $("prompt").value = settings.prompt;
  modeSelect.value = view === "compare" && settings.mode === "original" ? "optimized" : settings.mode;
  syncModeOptions();
  $("max-tokens").value = settings.budget;
  $("build-view").hidden = view !== "build"; $("compare-view").hidden = view !== "compare";
  $("build-examples").hidden = view !== "build"; $("compare-examples").hidden = view !== "compare";
  for (const item of ["build", "compare"]) { $( `view-${item}` ).classList.toggle("active", item === view); $(`view-${item}`).setAttribute("aria-selected", String(item === view)); }
  $("page-title").replaceChildren(document.createTextNode(view === "build" ? "From prompt " : "One prompt. "));
  const span = document.createElement("span"); span.textContent = view === "build" ? "to preview." : "Two runners."; $("page-title").append(span);
  $("intro-copy").textContent = view === "build" ? "Watch the code arrive and the page take shape. Choose a runner, build a page, then ask for the next change." : "Compare the standard and Lightreason runners using the same Qwen2.5-3B-Instruct model and the same prompt.";
  $("prompt-label").textContent = view === "build" ? "WHAT SHOULD WE BUILD?" : "YOUR PROMPT";
  $("mode-label").textContent = view === "build" ? "Qwen runner" : "Compare Qwen runners";
  $("execution-note").textContent = view === "build" ? "The code streams from the selected runner. The browser renders the HTML, CSS and JavaScript in an isolated live preview." : "The standard runner streams first, then Lightreason on the same GPU. Both use Qwen2.5-3B-Instruct and receive the same prompt and page context.";
  lastResult = resultByView[view]; $("timing-detail").hidden = true; errorBox.hidden = true;
  updateMode(); updateContextUI();
  setActivity(view === "build" ? "Describe a page or the next change and click Generate page." : "Enter a prompt to compare both runners with the same Qwen model.");
}

function resetPanel(runner, keepEditor = false) {
  outputs[runner] = "";
  if (!keepEditor) renderOutput(runner, "");
  $(`${runner}-panel`).classList.remove("running");
  $(`${runner}-state`).textContent = runner === "baseline" ? "Ready to start first" : runner === "single" ? "Ready to generate" : "Starts after standard runner";
  $(`${runner}-end`).textContent = "—"; $(`${runner}-first`).textContent = ""; $(`${runner}-latency`).textContent = ""; $(`${runner}-latency`).hidden = true;
  clientFirstToken[runner] = null;
  for (const [key, unit] of [["seconds", "s"], ["rate", "tok/s"], ["tokens", "tokens"]]) setMetric(`${runner}-${key}`, "—", unit);
}
function revealResults(view) {
  const reducedMotion = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  $(view === "build" ? "build-view" : "compare-view").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
}

function resetForRun() {
  errorBox.hidden = true; $("timing-detail").hidden = true;
  if (viewMode === "build") {
    const patch = currentRequest.patch_format === "json";
    resetPanel("single", patch); singleCompleted = false;
    $("single-title").textContent = MODEL_NAME;
    $("single-runner-label").textContent = RUNNER_LABELS[currentRequest.mode];
    setMetric("preview-first", "—", "s from click"); setMetric("preview-render-time", "—", "ms");
    previewRun = { id: ++previewRunCounter, submittedAt: requestSentAt, firstPaintS: null, renderMs: null, request: { ...currentRequest }, manual: false };
    if (currentRequest.app_action === "new") { acceptedPageCode = ""; acceptedPageComplete = true; updateContextUI(); }
    if (preview && !patch) { preview.begin({ keepPreview: currentRequest.app_action === "edit" }); $("preview-status").textContent = currentRequest.app_action === "edit" ? "Updating app…" : "Building new app…"; $("preview-detail").textContent = currentRequest.app_action === "edit" ? "The current page remains visible while the updated HTML is generated." : "Generating the requested app from scratch. Its preview appears when new HTML arrives."; }
    $("single-export-button").disabled = true;
  } else {
    resetPanel("baseline"); resetPanel("candidate"); updateMode();
    setMetric("speedup", "—", "×"); $("speedup-detail").textContent = "Waiting for both measured runs.";
    $("rate-ratio").textContent = "—"; $("exact-match").textContent = "Comparing…"; $("exact-match").className = "";
    $("match-detail").textContent = "Compared using token IDs"; $("export-button").disabled = true;
  }
}

function applyStatus(status) {
  currentStatus = { ...currentStatus, ...status };
  serviceReady = Boolean(currentStatus.ready); serviceBusy = Boolean(currentStatus.busy);
  if (syncModeOptions()) updateMode();
  if (currentStatus.gpu) $("gpu-name").textContent = currentStatus.gpu;
  $("model-name").textContent = `Same model: ${MODEL_NAME} · BF16`;
  $("connection").classList.toggle("ready", serviceReady); $("connection").classList.remove("error-state");
  $("connection-text").textContent = inFlight || serviceBusy ? "Qwen is running" : serviceReady ? "Qwen connected" : "Starting Qwen…";
  if (!inFlight && (!lastResult || !serviceReady)) setActivity(serviceReady ? "Ready. Enter a prompt to generate a page or compare runners." : "Starting Qwen…");
  updateButton();
}
async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`GPU service returned HTTP ${response.status}`);
    applyStatus(await response.json());
  } catch (_) {
    serviceReady = false; updateButton(); $("connection").classList.remove("ready"); $("connection").classList.add("error-state");
    $("connection-text").textContent = "Qwen service unavailable";
    if (!inFlight) setActivity("Cannot reach the local Qwen service. Start it again; this page reconnects automatically.");
  } finally { window.setTimeout(pollStatus, 2500); }
}

function observedClientTotal() {
  if (clientTotalS === null && requestSentAt !== null) clientTotalS = (performance.now() - requestSentAt) / 1000;
  return clientTotalS;
}

function showComparison(event) {
  setMetric("speedup", formatNumber(event.speedup), "×"); const same = event.exact_match === true;
  $("speedup-detail").textContent = same ? "Standard runner time ÷ Lightreason time. Same Qwen model and output token IDs." : "Completion-time ratio. Different outputs or lengths can affect this ratio.";
  if (Number.isFinite(Number(event.speedup)) && Number(event.speedup) < 1) $("speedup-detail").textContent += " The standard runner was faster on this run.";
  $("rate-ratio").textContent = `${formatNumber(event.token_rate_speedup)}×`; $("exact-match").textContent = same ? "Identical token IDs" : "Outputs differ";
  $("exact-match").className = same ? "same" : "different";
  $("match-detail").textContent = same ? `${event.baseline_tokens ?? "All"} output tokens matched` : `${event.prefix_match_tokens ?? 0} matching initial tokens · inspect both answers`;
  setActivity(same ? "Comparison complete. Both measured outputs match exactly." : "Comparison complete. Both answers are shown; different outputs require review.");
  const timing = [`Total from click: ${formatNumber(observedClientTotal(), 3)} s`];
  if (Number.isFinite(event.setup_s)) timing.push(`Runner handoff: ${formatNumber(event.setup_s, 3)} s`);
  timing.push(`Same Qwen model · Standard runner → ${RUNNER_LABELS[currentRequest.mode]}`);
  $("timing-detail").textContent = timing.join(" · "); $("timing-detail").hidden = false;
}
function stopLabel(metrics) { return metrics.stop_reason === "context_limit" ? "Context limit reached" : metrics.finished_eos === true ? "Natural end" : metrics.finished_eos === false ? "Output limit reached" : "Finished"; }

function updateRunnerMetrics(runner, metrics, tokenIds) {
  setMetric(`${runner}-seconds`, formatNumber(metrics.seconds, 3), "s"); setMetric(`${runner}-rate`, formatNumber(metrics.tokens_per_second, 1), "tok/s"); setMetric(`${runner}-tokens`, metrics.tokens ?? tokenIds?.length ?? "—", "tokens");
  $(`${runner}-end`).textContent = stopLabel(metrics);
  if (Number.isFinite(metrics.time_to_first_token_s)) $(`${runner}-first`).textContent = `Generation first token ${formatNumber(metrics.time_to_first_token_s, 3)} s`;
  if (clientFirstToken[runner] === null && Number.isFinite(metrics.request_to_first_token_s)) { $(`${runner}-latency`).textContent = `Request to first token: ${formatNumber(metrics.request_to_first_token_s, 3)} s`; $(`${runner}-latency`).hidden = false; }
}
function markFirstToken(runner) {
  if (clientFirstToken[runner] === null && requestSentAt !== null) { clientFirstToken[runner] = (performance.now() - requestSentAt) / 1000; $(`${runner}-latency`).textContent = `From click to first token: ${formatNumber(clientFirstToken[runner], 3)} s`; $(`${runner}-latency`).hidden = false; }
}
function updatePreview(text, final, metrics = {}) {
  if (!preview) return;
  preview.update(text, { final, complete: final ? metrics.finished_eos === true : true, stopReason: metrics.stop_reason || null });
}
function addHistory(event) {
  if (!previewRun || previewRun.manual) return;
  const row = { id: previewRun.id, prompt: currentRequest.prompt, app_action: currentRequest.app_action, patch_format: currentRequest.patch_format, mode: currentRequest.mode, generation_s: event.metrics?.seconds ?? null, first_token_s: clientFirstToken.single, first_preview_s: previewRun.firstPaintS, output_tokens: event.metrics?.tokens, stop_reason: event.metrics?.stop_reason };
  updateHistory.unshift(row); if (updateHistory.length > 8) updateHistory.pop(); renderHistory();
}
function renderHistory() {
  const list = $("update-history"); list.replaceChildren();
  if (!updateHistory.length) { const empty = document.createElement("li"); empty.className = "history-empty"; empty.textContent = "Build a page, then ask for a change to compare update times."; list.append(empty); return; }
  for (const row of updateHistory) {
    const item = document.createElement("li"); const description = document.createElement("div"); const title = document.createElement("strong"); title.textContent = row.prompt.length > 85 ? row.prompt.slice(0, 82) + "…" : row.prompt;
    const action = row.patch_format === "json" ? "Patch edit" : row.app_action === "edit" ? "Edit current app" : "New app";
    const detail = document.createElement("span"); detail.textContent = `${runnerName(row.mode)} · ${action} · ${row.output_tokens ?? "—"} tokens · ${row.stop_reason === "context_limit" ? "context limit" : row.stop_reason === "eos" ? "natural end" : "output limit"}`;
    description.append(title, detail); const timing = document.createElement("strong"); timing.className = "history-time"; timing.textContent = `${formatNumber(row.generation_s, 3)} s`; item.append(description, timing); list.append(item);
  }
}
function completeSingle(event) {
  if (singleCompleted) return; singleCompleted = true;
  const patch = currentRequest.patch_format === "json";
  const text = event.text ?? outputs.single; outputs.single = text; renderOutput("single", text); updateRunnerMetrics("single", event.metrics || {}, event.token_ids);
  $("single-panel").classList.remove("running"); $("single-state").textContent = patch ? "Patch applied" : "Generation complete";
  updatePreview(text, true, event.metrics || {}); addHistory(event);
  setActivity(patch ? "Patch applied. The browser is rendering the updated page." : "Generation complete. The preview renders the generated page; browser render time is separate.");
}

function handleEvent(event) {
  currentEvents.push(event); const runner = event.runner; const known = runner === "baseline" || runner === "candidate" || runner === "single";
  const patch = currentRequest.patch_format === "json";
  switch (event.type) {
    case "status":
      if (event.ready !== undefined || event.state || event.gpu) applyStatus(event);
      if (/prepar|warm|loading|starting/i.test(event.state || "")) setActivity(`Starting ${runnerName(currentRequest?.mode)}…`);
      else setActivity(event.message || event.detail);
      break;
    case "started":
      if (!known) break; $(`${runner}-panel`).classList.add("running"); $(`${runner}-state`).textContent = "Generating live";
      setActivity(runner === "single" ? (patch ? "Qwen is proposing one replacement. The current page remains available." : `${runnerName(currentRequest.mode)} is generating your page.`) : runner === "baseline" ? "Qwen is streaming through the standard runner first." : `${runnerName(currentRequest.mode)} is streaming now.`); break;
    case "token":
      if (!known) break;
      outputs[runner] = event.cumulative ? (event.text || "") : outputs[runner] + (event.text || "");
      if (!patch) renderOutput(runner, outputs[runner]);
      markFirstToken(runner);
      if (Number.isFinite(event.tokens)) setMetric(`${runner}-tokens`, event.tokens, "tokens");
      if (Number.isFinite(event.elapsed_s) && event.elapsed_s > 0) { setMetric(`${runner}-seconds`, formatNumber(event.elapsed_s, 3), "s"); if (Number.isFinite(event.tokens)) setMetric(`${runner}-rate`, formatNumber(event.tokens / event.elapsed_s, 1), "tok/s"); }
      if (runner === "single" && !patch) updatePreview(outputs.single, false); break;
    case "completed":
      if (!known) break;
      if (runner === "single") { if (!patch) completeSingle(event); break; }
      outputs[runner] = event.text ?? outputs[runner]; renderOutput(runner, outputs[runner]); updateRunnerMetrics(runner, event.metrics || {}, event.token_ids); $(`${runner}-panel`).classList.remove("running"); $(`${runner}-state`).textContent = "Complete";
      if (runner === "baseline") setActivity("The standard runner finished. The same Qwen model starts through Lightreason next."); break;
    case "generation":
      if (!patch && !singleCompleted) completeSingle({ ...event, runner: "single" });
      $("timing-detail").textContent = `Total from click: ${formatNumber(observedClientTotal(), 3)} s · Runner: ${runnerName(currentRequest.mode)}`; $("timing-detail").hidden = false; break;
    case "comparison": showComparison(event); break;
    case "error": throw new Error(event.message || "The GPU request could not finish.");
    case "done": if (event.success === false) throw new Error("The GPU request did not finish."); break;
    default: if (event.message) setActivity(event.message);
  }
}

async function consumeNDJSON(response, onEvent = handleEvent) {
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = "";
  for (;;) { const { value, done } = await reader.read(); pending += decoder.decode(value || new Uint8Array(), { stream: !done }); let next;
    while ((next = pending.indexOf("\n")) >= 0) { const line = pending.slice(0, next).trim(); pending = pending.slice(next + 1); if (line) onEvent(JSON.parse(line)); }
    if (done) break;
  }
  if (pending.trim()) onEvent(JSON.parse(pending));
}
async function responseError(response, prefix) { let message = `${prefix} HTTP ${response.status}.`; try { const body = await response.json(); message = body.error || body.message || message; } catch (_) {} return new Error(message); }



async function sendRun(endpoint, request) {
  const body = JSON.stringify(request);
  const headers = { "Content-Type": "application/json", "Accept": "application/x-ndjson" };
  const send = () => fetch(endpoint, { method: "POST", headers, body });
  let response = await send();
  if (response.status !== 409) return response;
  let message = "The GPU is busy. Try again when the current request finishes.";
  try { const payload = await response.json(); message = payload.error || payload.message || message; } catch (_) {}
  if (!message.includes("Prepare this demo configuration")) throw new Error(message);

  setActivity(`Starting ${runnerName(request.mode)}… setting up GPU execution.`);
  const setupStarted = performance.now();
  const setupEvents = [];
  let configured = false;
  let finished = false;
  const setupResponse = await fetch("/api/prepare", { method: "POST", headers, body });
  if (!setupResponse.ok) throw await responseError(setupResponse, "Qwen setup returned");
  await consumeNDJSON(setupResponse, (event) => {
    setupEvents.push(event);
    if (event.type === "token" || event.type === "token_ids" || event.type === "completed" || event.type === "generation" || event.type === "comparison") throw new Error("Unexpected generation during GPU setup.");
    if (event.type === "error") throw new Error(event.message || "Qwen could not start.");
    if (event.type === "prepared") configured = event.ready === true || event.prepared === true || event.demo_ready === true;
    if (event.type === "done") { if (event.success === false) throw new Error("Qwen setup did not finish."); finished = true; }
    if (event.type === "status") setActivity(`Starting ${runnerName(request.mode)}… setting up GPU execution.`);
  });
  if (!configured || !finished) throw new Error("Qwen setup ended before execution was ready.");
  automaticSetup = { seconds: (performance.now() - setupStarted) / 1000, events: setupEvents };
  // Retry once. The first 409 did not generate an answer; setup never used the
  // true prompt as model input. The retried request is the identical snapshot.
  return send();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault(); const submittedAt = performance.now();
  if (inFlight || !serviceReady || serviceBusy) return;
  currentRequest = selectedRequest(); if (!currentRequest.prompt) return;
  const patch = currentRequest.patch_format === "json";
  const editorAtSubmit = patch ? $("single-code").value : null;
  let wireRequest;
  try { wireRequest = patch ? patchWireRequest(currentRequest) : currentRequest; }
  catch (error) { showError(error.message); return; }
  const requestView = viewMode; currentEvents = []; automaticSetup = null; clientTotalS = null; inFlight = true; requestSentAt = submittedAt; resetForRun(); updateButton(); lockControls(true); revealResults(requestView);
  setActivity(requestView === "build" ? `Starting ${runnerName(currentRequest.mode)}.` : "Starting Qwen through the standard runner first.");
  try {
    const response = await sendRun(requestView === "build" ? "/api/generate" : "/api/compare", wireRequest);
    if (!response.ok) throw await responseError(response, "GPU service returned");
    await consumeNDJSON(response);
    const completeType = requestView === "build" ? "generation" : "comparison";
    const completed = currentEvents.find(item => item.type === completeType);
    if (!completed) throw new Error("The stream ended before the complete measured result arrived.");
    if (patch) {
      const done = currentEvents.at(-1);
      if (done?.type !== "done" || done.success !== true || completed.metrics?.finished_eos !== true) {
        throw new Error("The patch did not finish. Try again with a larger output limit or use Edit current app.");
      }
      const source = currentRequest.previous_code;
      if (acceptedPageCode !== source || $("single-code").value !== editorAtSubmit) throw new Error("The source changed during generation. Request a new patch.");
      const updated = applyPatchResult(source, completed.text);
      preview.begin({ keepPreview: true });
      completeSingle({ ...completed, text: updated });
    }
    lastResult = { recorded_at: new Date().toISOString(), view: requestView, request: currentRequest, service: currentStatus, automatic_setup: automaticSetup, client_total_s: observedClientTotal(), client_observed_first_token_s: { ...clientFirstToken }, preview: previewRun ? { first_preview_s: previewRun.firstPaintS, latest_render_ms: previewRun.renderMs } : null, events: currentEvents };
    resultByView[requestView] = lastResult; $(requestView === "build" ? "single-export-button" : "export-button").disabled = false;
  } catch (error) {
    showError(error.message);
    setActivity(patch ? "Patch rejected. Your current page and editor are intact." : "The request did not finish. Partial output is not a complete measured result.");
    if (patch) $("single-export-button").disabled = !resultByView.build;
    for (const runner of requestView === "build" ? ["single"] : ["baseline", "candidate"]) { $(`${runner}-panel`).classList.remove("running"); if ($(`${runner}-state`).textContent !== "Complete" && $(`${runner}-state`).textContent !== "Generation complete") $(`${runner}-state`).textContent = "Not completed"; }
  } finally { inFlight = false; serviceBusy = false; lockControls(false); updateButton(); }
});

function updateContextUI() {
  const available = Boolean(acceptedPageCode);
  const action = $("app-action");
  const editOption = action.querySelector('option[value="edit"]');
  const patchOption = action.querySelector('option[value="patch"]');
  const patchAvailable = viewMode === "build" && available && acceptedPageComplete && Boolean(preview);
  editOption.disabled = !available;
  editOption.textContent = acceptedPageComplete ? "Edit current app" : "Edit current app (partial)";
  patchOption.hidden = viewMode !== "build";
  patchOption.disabled = !patchAvailable;
  if (!patchAvailable && action.value === "patch") action.value = available ? "edit" : "new";
  if (!available && action.value === "edit") action.value = "new";
  const editing = available && action.value !== "new";
  $("context-control").hidden = !editing;
  $("app-action-hint").textContent = action.value === "patch"
    ? "Patch edit uses dedicated instructions for one exact replacement. Applying it reloads the preview."
    : editing ? "Edit current app sends the current HTML with your requested change." : "New app starts fresh. The current page is not sent with your prompt.";
  $("context-label-text").textContent = acceptedPageComplete ? "Editing the current app" : "Editing partial or incomplete page code";
  $("context-size").textContent = editing ? `${acceptedPageCode.length.toLocaleString()} characters included` : "";
  $("context-code").textContent = editing ? acceptedPageCode : "";
  updateButton();
}
function acceptPreviewCode(code, complete = true) {
  if (!code || (code === acceptedPageCode && complete === acceptedPageComplete)) return;
  acceptedPageCode = code; acceptedPageComplete = complete;
  // A generated page becomes available for explicit editing. It never turns
  // the next prompt into an edit or sends old page code without that choice.
  updateContextUI();
}

const preview = window.LightreasonPreview ? window.LightreasonPreview.create({
  iframe: $("web-preview"), viewportContainer: $("preview-viewport"), statusElement: $("preview-status"), detailElement: $("preview-detail"),
  onPaint(event) {
    setMetric("preview-render-time", formatNumber(event.elapsedMs, 1), "ms");
    if (previewRun && !previewRun.manual) {
      previewRun.renderMs = event.elapsedMs;
      if (previewRun.firstPaintS === null) { previewRun.firstPaintS = (performance.now() - previewRun.submittedAt) / 1000; setMetric("preview-first", formatNumber(previewRun.firstPaintS, 3), "s from click"); }
      const history = updateHistory.find((row) => row.id === previewRun.id); if (history) history.first_preview_s = previewRun.firstPaintS;
      if (resultByView.build?.request === currentRequest && resultByView.build.preview) { resultByView.build.preview.first_preview_s = previewRun.firstPaintS; resultByView.build.preview.latest_render_ms = event.elapsedMs; }
    }
    if (event.final) acceptPreviewCode(event.code || preview.getCode(), event.complete !== false && !event.hasError);
  },
  onError(event) { if (event.final) setActivity("The browser reported a preview error. You can edit the code or ask the model to fix the page."); }
}) : null;
if (!preview) { $("preview-status").textContent = "Preview unavailable"; $("preview-detail").textContent = "The preview module did not load. Code generation can still be measured."; }

$("render-code").addEventListener("click", () => {
  if (!preview || inFlight) return;
  previewRun = { id: ++previewRunCounter, manual: true }; preview.begin({ keepPreview: true }); preview.update($("single-code").value, { final: true, complete: true });
  $("single-state").textContent = "Edited locally"; setActivity("Rendering your edited code. Existing generation timings describe the previous model output.");
});
$("clear-page").addEventListener("click", () => { if (inFlight) return; acceptedPageCode = ""; acceptedPageComplete = true; $("app-action").value = "new"; $("single-code").value = ""; outputs.single = ""; previewRun = null; if (preview) preview.reset(); updateContextUI(); updateButton(); });
$("single-code").addEventListener("input", updateButton);
$("app-action").addEventListener("change", updateContextUI);
for (const button of document.querySelectorAll("[data-prompt]")) button.addEventListener("click", () => { $("prompt").value = button.dataset.prompt; updateButton(); $("prompt").focus(); });
function saveResult() { const result = resultByView[viewMode]; if (!result) return; const blob = new Blob([JSON.stringify(result, null, 2) + "\n"], { type: "application/json" }); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = `lightreason-${viewMode}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000); }
$("export-button").addEventListener("click", saveResult); $("single-export-button").addEventListener("click", saveResult);
$("view-build").addEventListener("click", () => setView("build")); $("view-compare").addEventListener("click", () => setView("compare"));
modeSelect.addEventListener("change", () => { if (viewMode === "compare") { resetPanel("candidate"); resultByView.compare = null; $("export-button").disabled = true; $("exact-match").textContent = "Not measured yet"; setMetric("speedup", "—", "×"); } updateMode(); updateButton(); });
$("max-tokens").addEventListener("change", () => { updateButton(); });
$("prompt").addEventListener("input", () => { updateButton(); });

function updatePreviewViewport() {
  if (!preview || typeof preview.viewport !== "function") return;
  const value = $("preview-viewport-size").value;
  preview.viewport({ width: value === "responsive" ? null : Number(value), fit: $("preview-fit").checked });
}
$("preview-viewport-size").addEventListener("change", updatePreviewViewport);
$("preview-fit").addEventListener("change", updatePreviewViewport);
function updateDesignState() { $("design-state").textContent = designSystemEnabled && designSystemPrompt.trim() ? "On" : "Off"; }
$("design-button").addEventListener("click", () => {
  $("design-system-prompt").value = designSystemPrompt;
  $("design-enabled").checked = designSystemEnabled;
  $("design-dialog").showModal();
});
$("design-cancel").addEventListener("click", () => $("design-dialog").close());
$("design-reset").addEventListener("click", () => { $("design-system-prompt").value = DEFAULT_DESIGN_PROMPT; $("design-enabled").checked = true; });
$("design-save").addEventListener("click", () => {
  designSystemPrompt = $("design-system-prompt").value.trim();
  designSystemEnabled = $("design-enabled").checked;
  $("design-dialog").close(); updateDesignState();
  setActivity("Design instructions updated. The same system prompt will be used by each runner.");
});
updateDesignState(); updatePreviewViewport();

updateMode(); updateContextUI(); pollStatus();
