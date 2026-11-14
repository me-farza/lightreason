"use strict";

const byId = (id) => document.getElementById(id);
const composer = byId("composer");
const input = byId("message");
const status = byId("chat-status");
const thread = byId("messages");
const modePicker = byId("memory-mode");
const attachment = byId("attachment");
const dreamNotePanel = byId("dream-note-panel");
const dreamNoteInput = byId("dream-note-input");
const dreamNoteStatus = byId("dream-note-status");
const dreamSample = {
  text: "When aligning the demo optics, check the reference first.",
  source: "Synthetic Dream sample",
  requestId: "95f61653-b040-43fd-89de-aee7123ccb42",
};
const chatInput = ["input", "Chat context", "The current message and recent turns."];
const geminiFlow = [
  chatInput,
  ["memory", "Local 3B recall", "Checks for a learned rule that fits."],
  ["decode", "Decode memory", "Turns recalled vectors into a short text cue."],
  ["answer", "Gemini", "Writes the reply using chat and any cue."],
];
const chatFlows = {
  gemini: geminiFlow,
  vector: [chatInput, ["memory", "Local 3B recall", "Passes relevant vectors directly to the reasoner."],
    ["answer", "Local Qwen 35B", "Writes the reply."]],
  text: [chatInput, ["memory", "Saved evidence", "Finds relevant excerpts in saved notes."],
    ["answer", "Local model", "Writes the reply using chat and any excerpts."]],
  off: [chatInput, ["answer", "Local model", "Writes the reply from chat context."]],
  remote: [chatInput, ["answer", "OpenRouter", "Writes the reply from chat context."]],
};
const inventoryFlow = [
  ["input", "Your request", "Ask to search the wiki or list saved memory."],
  ["memory", "Memory wiki", "Look up saved evidence on this server."],
  ["answer", "Local reply", "Show matching notes or the memory inventory."],
];
const night = [
  ["capture", "Review notes", "Select useful lessons from queued messages and files."],
  ["questions", "Draft questions", "Gemini can draft questions grounded in the notes."],
  ["train", "Train and test", "Train the local recall policy and check what it remembers."],
  ["promote", "Save results", "Keep passing updates and archive evidence in the wiki."],
];
const exampleButtons = [...byId("chat-examples").querySelectorAll("button")];
const conversations = {local: [], remote: []};
const availableModes = new Set();
const mountPath = new URL(document.baseURI).pathname;
const storageKey = (name) => mountPath === "/" ? "vicena." + name : "vicena." + mountPath + "." + name;
const localChatKey = storageKey("localChat.v1");
const maxStoredMessages = 40;
const reportNames = {
  running: "Dream is running.",
  complete: "Dream complete.",
  failed: "Dream stopped.",
};
const localModelNotes = {
  cold: "The local model will warm up on the first message.",
  starting: "The local model is warming up.",
  ready: "The local model is ready.",
  failed: "The local model needs to restart on the next message.",
  unavailable: "The local model is unavailable.",
};
let remoteCapture = null;
let pendingCapture = null;
let localModelStatus;
let readinessTimer;
let waitingForLocal = false;
let shownChannel;
let dreaming = false;
let busy = false;
let uploading = false;
let savingDreamNote = false;
let dreamNoteRequest = null;
let sampleSaved = false;
let starting = false;
let didFailToStart = false;
let dreamLoaded = false;
let reportKey;
let dreamQuestion = "";
let dreamRevision = 0;
let entryNight = 0;
let ranDuringVisit = false;
let completedDuringVisit = false;
let pollTimer;
let samplingTimer;
let samplingAttempt;
let watchSamplingAttempt;
let dream = {status: "idle", events: []};
const flows = {};
const samplingRuns = new Map();
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const route = (mode) => mode === "remote" ? "remote" : "local";
const usesLearnedMemory = (mode) => mode === "vector" || mode === "gemini";
const usesLocalAnswer = (mode) => route(mode) === "local" && mode !== "gemini";

function learnedMode() {
  if (usesLearnedMemory(modePicker.value)) return modePicker.value;
  if (availableModes.has("gemini")) return "gemini";
  return availableModes.has("vector") ? "vector" : "";
}

function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content !== undefined) element.textContent = content;
  if (className) element.className = className;
  return element;
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) {
    const error = new Error(value.error || "Vicena could not complete the request.");
    error.captureId = value.capture_id;
    throw error;
  }
  return value;
}

function persistLocalChat() {
  try {
    sessionStorage.setItem(localChatKey, JSON.stringify({
      transcript: conversations.local.slice(-maxStoredMessages),
      pendingCapture,
    }));
  } catch {}
}

function showPendingCapture() {
  input.value = pendingCapture?.content || "";
  byId("remember").checked = true;
  dreamQuestion = "";
  if (pendingCapture) {
    status.textContent = pendingCapture.captureId ?
      "Message saved for Dream. Send again to retry the reply." :
      "Send again to retry saving this message and get a reply.";
  }
}

function restoreLocalChat() {
  let stored;
  try { stored = JSON.parse(sessionStorage.getItem(localChatKey)); } catch {}
  if (!stored) return;
  if (Array.isArray(stored.transcript)) {
    const transcript = stored.transcript.slice(-maxStoredMessages);
    if (transcript.length % 2 === 0 && transcript.every((message, index) =>
      message?.role === (index % 2 === 0 ? "user" : "assistant") && typeof message.content === "string")) {
      conversations.local = transcript;
      for (const message of transcript) {
        line(message.role, message.content, "local", message.metadata, false);
      }
    }
  }
  const pending = stored.pendingCapture;
  if (typeof pending?.content === "string" && /^[a-f0-9-]{36}$/.test(pending.requestId)) {
    pendingCapture = pending;
  }
}

function setLocalStatus(value) {
  localModelStatus = value;
  byId("local-route-note").textContent = localModelNotes[value] || "";
  if (waitingForLocal) {
    status.textContent = value === "starting" || value === "cold" ?
      "Waiting for the local model to warm up…" : "Waiting for Vicena…";
  }
}

function watchLocalStatus() {
  clearTimeout(readinessTimer);
  if (!waitingForLocal && (!usesLocalAnswer(modePicker.value) || localModelStatus !== "starting")) return;
  readinessTimer = setTimeout(async () => {
    try {
      const result = await api("api/chat/status");
      setLocalStatus(result.local_status);
    } catch {}
    watchLocalStatus();
  }, 2000);
}

function renderFlow() {
  const event = dream.events?.at(-1);
  const stage = dream.stage || event?.stage;
  const nightStage = {review: "questions", complete: "promote"}[stage] || stage;
  const mode = modePicker.value;
  const chatFlow = flows[mode] || {};
  const active = dreaming ? (dream.status === "idle" || starting ? "capture" : nightStage) : chatFlow.stage;
  const steps = dreaming ? night : chatFlow.localOnly ? inventoryFlow : (chatFlows[mode] || geminiFlow);
  byId("flow-title").textContent = dreaming ? "Dream workflow" : "Chat workflow";
  const activeIndex = steps.findIndex(([name]) => name === active);
  const counts = dream.counts || {};
  const notes = dream.intake?.captured_count ?? dream.entries?.length;
  const queued = counts.pending || 0;
  const lessonCount = dream.questions?.length;
  const updates = dream.training?.promotions;
  const stageNotes = {
    capture: notes ? quantity(notes, "note used") : queued ? quantity(queued, "note waiting") : "",
    questions: lessonCount != null && activeIndex >= 1 ? quantity(lessonCount, "question") : "",
    train: dream.status === "complete" && dream.training?.optimizer_steps === 0 ? "No training steps" :
      dream.training?.optimizer_steps != null ? quantity(dream.training.optimizer_steps, "optimizer step") : "",
    promote: updates != null && dream.status === "complete" ? quantity(updates, "memory update") : "",
  };
  byId("flow-track").replaceChildren(...steps.map(([name, label, description], index) => {
    const item = node("li", undefined, "flow-step");
    if (index < activeIndex) item.dataset.state = "done";
    if (!dreaming && name === "decode" && chatFlow.decoded === false && active === "answer") {
      item.dataset.state = "skipped";
      description = "Skipped because local memory found no match.";
    }
    if (name === active) {
      item.dataset.state = "current";
      item.setAttribute("aria-current", "step");
    }
    const informative = node("span");
    informative.append(node("strong", label), node("small", description));
    if (dreaming && !starting && stageNotes[name]) informative.append(node("small", stageNotes[name], "flow-detail"));
    const dot = node("span", undefined, "flow-dot");
    dot.setAttribute("aria-hidden", "true");
    item.append(dot, informative);
    return item;
  }));
  const message = dreaming ? (starting ? "Starting Dream…" : dream.status === "running" ?
    event?.message || "Training local memory" : dream.counts?.pending ? "Ready to run" : "Ready for notes") :
    chatFlow.message || "Ready for a message";
  byId("flow-status").textContent = message;
}

function controls() {
  const training = dream.status === "running";
  const running = training || starting;
  byId("send").disabled = !availableModes.has(modePicker.value) || busy || uploading || running;
  modePicker.disabled = busy || availableModes.size === 0;
  input.disabled = busy;
  byId("remember").disabled = busy;
  byId("new-chat").disabled = busy || uploading || savingDreamNote || (dreaming && running);
  byId("dream-toggle").disabled = !dreamLoaded || busy || uploading || savingDreamNote || (dreaming && running);
  attachment.disabled = busy || uploading;
  const pending = dream.counts?.pending || 0;
  const retry = didFailToStart || dream.status === "failed" || (!pending && (dream.skill?.retryable || heldMemory(dream)));
  const next = completedDuringVisit && pending > 0 && !retry;
  const run = byId("run-dream");
  run.disabled = !dreamLoaded || busy || uploading || savingDreamNote || running || !(pending || retry);
  run.dataset.retry = String(retry && !didFailToStart);
  run.textContent = running ? "Dream running…" : retry ? "Retry Dream" : next ? "Next batch" : "Run Dream";
  byId("dream-queue").textContent = pending ? quantity(pending, "note queued", "notes queued") :
    !running && !retry ? "Queue a note to enable Run Dream." : "";
  const noteLocked = busy || uploading || savingDreamNote || running;
  dreamNoteInput.disabled = noteLocked;
  byId("save-dream-note").disabled = noteLocked || !dreamNoteInput.value.trim();
  byId("try-dream-sample").disabled = noteLocked || sampleSaved;
  byId("morning-questions").querySelectorAll("button").forEach((button) => {
    button.disabled = busy || running || Boolean(pendingCapture) || !learnedMode();
  });
  for (const button of exampleButtons) {
    const needsMemory = button.dataset.example !== "teach";
    const evidenceMode = button.dataset.example === "wiki" && availableModes.has("text");
    button.disabled = byId("send").disabled || Boolean(pendingCapture || remoteCapture) ||
      (needsMemory && !learnedMode() && !evidenceMode);
  }
}

function showConversation() {
  const remote = modePicker.value === "remote";
  const channel = route(modePicker.value);
  shownChannel = channel;
  const turns = [...thread.querySelectorAll(".chat-turn")];
  for (const turn of turns) turn.hidden = turn.getAttribute("data-channel") !== channel;
  byId("welcome").hidden = turns.some((turn) => !turn.hidden);
  byId("remote-disclosure").hidden = !remote;
  byId("gemini-disclosure").hidden = modePicker.value !== "gemini";
  byId("local-route-note").hidden = !usesLocalAnswer(modePicker.value);
  watchLocalStatus();
  status.textContent = "";
  controls();
  renderFlow();
}

function view(nightMode) {
  dreaming = nightMode;
  document.body.classList.toggle("is-dreaming", dreaming);
  byId("dream-toggle").setAttribute("aria-pressed", String(dreaming));
  byId("dream-symbol").textContent = dreaming ? "☀" : "☾";
  byId("dream-label").textContent = dreaming ? "Morning" : "Dream";
  byId("dream-view").hidden = !dreaming;
  thread.hidden = dreaming;
  composer.hidden = dreaming;
  controls();
  renderFlow();
  if (!dreaming) flushSampling();
  try { sessionStorage.setItem(storageKey("dreamView"), String(dreaming)); } catch {}
  if (dreaming) byId("dream-title").focus();
  else input.focus();
}

function heldMemory(data) {
  return data.training?.policy_runs?.some((run) => run.status === "held");
}

function quantity(count, singular, plural = singular + "s") {
  return count + " " + (count === 1 ? singular : plural);
}

function dreamStatus(data) {
  if (data.error) return data.error;
  if (data.status === "idle") return "";
  if (data.status !== "complete") return reportNames[data.status] || "";
  const updates = data.training?.promotions || 0;
  if (heldMemory(data) || data.skill?.retryable) {
    return updates > 0 ? "Dream saved validated updates and held candidates that failed checks." :
      "Dream held candidates that failed checks. Local 3B memory is unchanged.";
  }
  if (updates > 0) return "Dream complete. Local 3B memory updated.";
  if (data.intake?.status === "needs_detail") return "Saved as evidence. Add a concrete lesson to train local memory.";
  if (data.intake?.status === "evidence_only") return "This batch produced evidence questions. Local 3B memory is unchanged.";
  if (!data.questions?.length && !data.counts?.pending) return "";
  if (data.count_run_promoted > 0) return "Dream complete. Material saved as evidence. Local 3B memory is unchanged.";
  return "Dream complete. Local 3B memory is unchanged.";
}

function setDetails(id, items, empty) {
  const container = byId(id);
  const detailKey = (detail) => detail.dataset.key || detail.firstElementChild.textContent;
  const expanded = new Map([...container.querySelectorAll("details")]
    .map((detail) => [detailKey(detail), detail.open]));
  for (const item of items) {
    const descendants = [...item.querySelectorAll("details")];
    const details = item.matches("details") ? [item, ...descendants] : descendants;
    for (const detail of details) {
      detail.open = expanded.get(detailKey(detail)) ?? detail.open;
    }
  }
  container.replaceChildren(...(items.length || !empty ? items : [node("p", empty, "dream-caption")]));
}

function scoredAction(value) {
  if (value === null) return "Abstain";
  if (typeof value === "string") return value.replaceAll("_", " ");
  return JSON.stringify(value);
}

function renderCheck(check) {
  const item = node("details");
  item.append(node("summary", check.question || "Retention check"));
  if (check.split) item.append(node("span", splitLabel(check.split), "dream-category"));
  for (const [key, label] of [["expected", "Expected action"], ["before", "Before Dream"], ["actual", "After Dream"]]) {
    if (check[key] !== undefined) item.append(node("p", label + " · " + scoredAction(check[key])));
  }
  if (check.answer) item.append(node("p", "Expected memory value · " + check.answer));
  if (Number.isFinite(check.reward_after) && Number.isFinite(check.reward_before)) {
    item.append(node("p", "Reward " + check.reward_before + " → " + check.reward_after, "dream-caption"));
  } else if (Number.isFinite(check.reward)) {
    item.append(node("p", "Reward " + check.reward, "dream-caption"));
  }
  if (check.source) item.append(node("p", check.source, "dream-caption"));
  return item;
}

function splitLabel(split) {
  return {training: "Training question", "held out": "Held-out check", retention: "Retention check"}[split] || split;
}

function renderPlan(plan, checks) {
  const list = node("ol", undefined, "dream-plan");
  for (const row of plan) {
    const item = node("li");
    item.append(node("span", splitLabel(row.split) + (row.origin === "gemini" ? " · Gemini" : ""), "dream-category"));
    const result = checks.find((check) => check.question === row.question && check.split === row.split);
    if (result) {
      item.append(renderCheck(result));
    } else {
      item.append(node("strong", row.question));
      if (row.expected !== undefined) item.append(node("p", "Expected action · " + scoredAction(row.expected)));
      if (row.answer) item.append(node("p", "Expected memory value · " + row.answer));
    }
    list.append(item);
  }
  return list;
}

function chartElement(tag, attributes, content) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (content != null) element.textContent = content;
  return element;
}

function samplingChart(points, {metric, label, min, max, ticks, theme, live}) {
  const observed = points.filter((point) => Number.isFinite(point[metric]));
  if (!observed.length) return null;
  const latest = observed.at(-1);
  const firstStep = points[0].step;
  const lastStep = points.at(-1).step;
  const endStep = live ? Math.max(32, Math.ceil(lastStep / 32) * 32) : lastStep;
  const x = (step) => firstStep === endStep ? 186 : 38 + 296 * (step - firstStep) / (endStep - firstStep);
  const y = (value) => 84 - 73 * (Math.max(min, Math.min(max, value)) - min) / (max - min);
  const figure = node("figure", undefined, "dream-chart " + theme);
  const caption = node("figcaption", label);
  caption.append(node("strong", latest[metric].toFixed(3)));
  const description = label + ", " + observed.length + " measured checkpoints. Final shown step " +
    latest.step + ", value " + latest[metric].toFixed(3) + ".";
  const chart = chartElement("svg", {viewBox: "0 0 358 121", role: "img", "aria-label": description});
  for (const [value, tick] of ticks) {
    const pos = y(value);
    chart.append(chartElement("line", {x1: 38, x2: 334, y1: pos, y2: pos, class: "chart-gridline"}),
      chartElement("text", {x: 31, y: pos + 3, "text-anchor": "end", class: "chart-label"}, tick));
  }
  for (const step of new Set([firstStep, endStep])) {
    chart.append(chartElement("text", {x: x(step), y: 99, "text-anchor": "middle", class: "chart-label"}, step));
  }
  chart.append(chartElement("text", {x: 186, y: 116, "text-anchor": "middle", class: "chart-label"}, "Optimizer step"));
  const vertices = observed.map((point) => [x(point.step), y(point[metric])]);
  if (vertices.length > 1) {
    chart.append(chartElement("polyline", {points: vertices.map((point) => point.join(",")).join(" "), class: "chart-trend"}));
  }
  for (const [index, [cx, cy]] of vertices.entries()) {
    const point = observed[index];
    const newest = live && index === vertices.length - 1;
    const dot = chartElement("circle", {cx, cy, r: newest ? 4 : 2.7,
      class: newest ? "chart-point chart-latest" : "chart-point"});
    dot.append(chartElement("title", {}, "Step " + point.step + " · " + point[metric].toFixed(3)));
    chart.append(dot);
  }
  figure.append(caption, chart);
  return figure;
}

function renderSampling(points, live) {
  if (!points.length) return null;
  const block = node("div", undefined, "dream-sampling");
  const charts = node("div", undefined, "dream-chart-grid");
  const reward = samplingChart(points, {metric: "mean_reward", label: "Sampled mean reward", min: -1, max: 1,
    ticks: [[1, "1"], [0, "0"], [-1, "−1"]], theme: "reward", live});
  const entropy = samplingChart(points, {metric: "mean_entropy_nats", label: "Sampled policy entropy (nats)", min: 0,
    max: Math.LN2, ticks: [[Math.LN2, "0.69"], [Math.LN2 / 2, "0.35"], [0, "0"]], theme: "entropy", live});
  charts.append(...[reward, entropy].filter(Boolean));
  block.append(charts);
  if (entropy) block.append(node("p", "Lower entropy means more decisive actions. Validation checks determine whether Dream saves an update."));
  const details = node("details");
  details.append(node("summary", "Sampled checkpoints"));
  const table = node("table");
  const header = node("tr");
  const showSuccess = points.some((point) => Number.isFinite(point.success_rate));
  const columns = ["Step", "Samples", "Reward", ...(entropy ? ["Entropy"] : []), ...(showSuccess ? ["Success"] : [])];
  for (const label of columns) {
    const cell = node("th", label);
    cell.scope = "col";
    header.append(cell);
  }
  const head = node("thead");
  head.append(header);
  const body = node("tbody");
  const fixed = (number) => Number.isFinite(number) ? number.toFixed(3) : "—";
  for (const point of points) {
    const row = node("tr");
    const values = [point.step, point.sample_count ?? "—", fixed(point.mean_reward)];
    if (entropy) values.push(fixed(point.mean_entropy_nats));
    if (showSuccess) values.push(Number.isFinite(point.success_rate) ? Math.round(point.success_rate * 100) + "%" : "—");
    row.append(...values.map((value) => node("td", value)));
    body.append(row);
  }
  table.append(head, body);
  details.append(table);
  block.append(details);
  return block;
}

function samplingKey(run) {
  return JSON.stringify([run.candidate_id ?? [run.key, run.source], run.discarded_optimizer_steps || 0,
    Boolean(run.planner_fallback)]);
}

function showSampling(slot, state) {
  const sampling = renderSampling(state.points.slice(0, state.visible), state.live || state.visible < state.points.length);
  if (sampling) {
    const details = sampling.querySelector("details");
    details.dataset.key = slot.dataset.samplingKey + "/sampled-checkpoints";
    details.open = slot.querySelector("details")?.open || false;
  }
  slot.replaceChildren(...(sampling ? [sampling] : []));
}

function updateSamplingPanels(changed) {
  for (const slot of byId("dream-checks").querySelectorAll("[data-sampling-key]")) {
    const key = slot.dataset.samplingKey;
    const state = !changed || changed.has(key) ? samplingRuns.get(key) : null;
    if (state) showSampling(slot, state);
  }
}

function flushSampling() {
  clearTimeout(samplingTimer);
  samplingTimer = undefined;
  for (const state of samplingRuns.values()) {
    state.visible = state.points.length;
    state.live = false;
  }
  updateSamplingPanels();
}
document.addEventListener("visibilitychange", () => { if (document.hidden) flushSampling(); });
reducedMotion.addEventListener("change", () => { if (reducedMotion.matches) flushSampling(); });

function playSampling() {
  if (samplingTimer || ![...samplingRuns.values()].some((state) => state.visible < state.points.length)) return;
  samplingTimer = setTimeout(() => {
    samplingTimer = undefined;
    if (!dreaming || document.hidden || reducedMotion.matches) {
      flushSampling();
      return;
    }
    const changed = new Set();
    for (const [key, state] of samplingRuns) {
      if (state.visible < state.points.length) {
        state.visible += 1;
        changed.add(key);
      }
    }
    updateSamplingPanels(changed);
    playSampling();
  }, dream.status === "running" ? 125 : 80);
}

function syncSampling(data, started) {
  const attempt = JSON.stringify([data.run_id, data.night, data.attempt_id]);
  if (attempt !== samplingAttempt || started) {
    clearTimeout(samplingTimer);
    samplingTimer = undefined;
    samplingRuns.clear();
    samplingAttempt = attempt;
  }
  if (data.status === "running" || started) watchSamplingAttempt = attempt;
  const animate = dreaming && !document.hidden && !reducedMotion.matches && watchSamplingAttempt === attempt;
  const active = new Set();
  for (const run of data.training?.policy_runs || []) {
    const key = samplingKey(run);
    active.add(key);
    const sampling = run.sampling || {};
    const checkpoints = sampling.points?.length ? sampling.points : [sampling.first, sampling.last];
    const points = [...new Map(checkpoints.filter((point) => Number.isFinite(point?.step))
      .map((point) => [point.step, point])).values()].sort((left, right) => left.step - right.step);
    const previous = samplingRuns.get(key);
    const visible = animate ? Math.min(points.length, Math.max(1, previous?.visible || 0)) : points.length;
    samplingRuns.set(key, {points, visible, live: animate && data.status === "running" &&
      (run.status === "training" || run.status === "validating")});
  }
  for (const key of samplingRuns.keys()) {
    if (!active.has(key)) samplingRuns.delete(key);
  }
  if (animate) playSampling();
}

function renderTraining(training) {
  const panel = byId("dream-training");
  panel.hidden = !training && dream.stage !== "train";
  if (!training) {
    byId("dream-training-status").textContent = panel.hidden ? "" : "Preparing training and validation questions…";
    byId("dream-checks").replaceChildren();
    return;
  }
  const metrics = [
    ["attempts", "replay attempts"], ["optimizer_steps", "optimizer steps"],
    ["promotions", "memory updates"],
  ];
  byId("dream-training-status").textContent = training.reason || metrics
    .filter(([key]) => Number.isFinite(training[key]))
    .map(([key, label]) => training[key] + " " + label).join(" · ") ||
    (dream.status === "running" ? "Training. Checks appear as they finish." : "");

  const labels = {
    training: "Training", validating: "Validating", promoted: "Saved", held: "Held", skipped: "Skipped", unchanged: "Unchanged",
  };
  const runs = (training.policy_runs || []).map((run) => {
    const plan = run.plan || [];
    const item = node("details", undefined, "dream-policy");
    item.dataset.key = JSON.stringify([dream.night, dream.attempt_id, run.candidate_id ?? [run.key, run.source]]);
    item.open = run.status === "training" || run.status === "validating" || plan.length > 0;
    const summary = [run.algorithm, labels[run.status] || run.status];
    if (Number.isFinite(run.optimizer_steps)) summary.push(run.optimizer_steps + " optimizer steps");
    const title = node("summary", run.key);
    title.append(node("small", summary.filter(Boolean).join(" · "), "dream-policy-meta"));
    item.append(title);
    const slot = node("div");
    slot.dataset.samplingKey = samplingKey(run);
    const samples = samplingRuns.get(slot.dataset.samplingKey);
    if (samples) showSampling(slot, samples);
    item.append(slot);
    if (run.checks?.length && Number.isFinite(run.before_passed) && Number.isFinite(run.after_passed)) {
      item.append(node("p", "Validation checks passed: " + run.before_passed + " → " + run.after_passed,
        "dream-validation"));
    }
    if (run.source) item.append(node("p", run.source, "dream-caption"));
    if (run.reason) item.append(node("p", run.reason, "dream-caption"));
    if (run.planner_fallback) item.append(node("p", run.planner_fallback, "dream-caption"));
    if (run.answer) item.append(node("p", "Expected answer · " + run.answer));
    const checks = run.checks || [];
    if (plan.length || checks.length) {
      const questions = node("details");
      questions.dataset.key = item.dataset.key + "/questions";
      questions.open = run.status === "training" || run.status === "validating";
      questions.append(node("summary", "Training and validation questions"));
      if (plan.length) questions.append(renderPlan(plan, checks));
      questions.append(...checks.filter((check) => !plan.some((row) => row.question === check.question &&
        row.split === check.split)).map(renderCheck));
      item.append(questions);
    }
    return item;
  });
  const checks = [...(training.recall_checks || []), ...(!runs.length ? training.procedure_checks || [] : [])].map(renderCheck);
  setDetails("dream-checks", [...runs, ...checks]);
}

function renderMorningQuestions(data) {
  const runs = data.training?.policy_runs || [];
  const passed = runs.filter((run) => run.status === "promoted")
    .flatMap((run) => run.checks || [])
    .filter((check) => check.split === "held out" && check.expected === "recall" &&
      (check.reward_after ?? check.reward) > 0);
  const recalled = (data.training?.recall_checks || []).filter((check) => check.reward > 0);
  const candidates = passed.length || recalled.length || runs.length ? [...passed, ...recalled] :
    (data.questions || []).filter((item) => item.neural_eligible === true);
  const prompts = [...new Set(candidates
    .map((item) => item.question).filter((value) => typeof value === "string" && value.length > 0))].slice(0, 3);
  byId("morning-preview").hidden = data.status !== "complete" || !data.training?.promotions || !prompts.length;
  byId("morning-questions").replaceChildren(...prompts.map((prompt) => {
    const button = node("button", prompt);
    button.type = "button";
    button.addEventListener("click", () => {
      const mode = learnedMode();
      if (!mode || pendingCapture) return;
      if (modePicker.value !== mode) {
        modePicker.value = mode;
        showConversation();
      }
      input.value = prompt;
      byId("remember").checked = true;
      dreamQuestion = prompt;
      input.focus();
    });
    return button;
  }));
}

function renderDreamQuestions(data) {
  const questions = data.questions || [];
  const followups = data.clarifications || [];
  const inProgress = data.status === "running";
  const categories = {process: "Procedure", experiment: "Experiment", preference: "Preference", profile: "Profile"};
  const items = questions.map((item) => {
    const card = node("article", undefined, "dream-question");
    const type = categories[item.category] || "Lesson";
    const target = item.neural_eligible === false ? "Saved evidence" : "For local 3B";
    card.append(node("span", [type, target, item.split].filter(Boolean).join(" · "), "dream-category"),
      node("h3", item.question));
    if (item.answer) {
      const answer = node("p", undefined, "dream-answer");
      answer.append(node("span", "Expected answer"), node("strong", item.answer));
      card.append(answer);
    }
    if (item.selection_reason) card.append(node("p", item.selection_reason));
    if (item.source || item.observation || item.reward != null && !item.selection_reason) {
      const details = node("details");
      details.append(node("summary", item.source ? "Source · " + item.source : "Scoring"));
      if (item.observation) details.append(node("p", item.observation));
      if (item.origin === "gemini_extraction") details.append(node("p", "Gemini proposed this lesson from the quoted note."));
      if (item.reward != null && !item.selection_reason) {
        details.append(node("p", typeof item.reward === "number" ? "Reward " + item.reward : item.reward));
      }
      card.append(details);
    }
    return card;
  });
  let empty = "No approved notes yet.";
  if (inProgress) empty = "Reading approved notes. Questions will appear as Vicena finds concrete lessons.";
  else if (data.intake?.status === "needs_detail") empty = "Vicena found no concrete lesson in this batch.";
  else if (data.counts?.pending) empty = quantity(data.counts.pending, "approved note") + " waiting to be processed.";
  else if (data.selection?.conversation_only && !data.entries?.length) {
    empty = "This batch contained chat-only turns.";
  } else if (data.status === "complete" && data.counts?.captures) {
    empty = "The previous Dream produced no questions. Add one concrete lesson to try again.";
  } else if (data.status === "failed") empty = "Dream stopped before it could produce questions.";
  setDetails("dream-questions", items, empty);
  byId("dream-followups").hidden = !followups.length;
  byId("dream-clarifications").replaceChildren(...followups.map((followup) => {
    const item = node("li");
    item.append(node("strong", followup.question));
    if (followup.source) item.append(node("p", followup.source));
    return item;
  }));
  if (!inProgress && (!questions.length || followups.length || data.intake?.status === "evidence_only")) {
    dreamNotePanel.open = true;
  }
}

function renderDream(data, started = false) {
  dream = data;
  if (ranDuringVisit && data.status === "complete") completedDuringVisit = true;
  if (data.entries?.some((entry) => entry.source === dreamSample.source)) markSampleSaved();
  byId("dream-status").textContent = dreamStatus(data);
  const key = JSON.stringify(data);
  if (key === reportKey && !started) {
    controls();
    return;
  }
  reportKey = key;
  renderFlow();
  byId("dream-title").textContent = data.night > 0 ? "Dream " + data.night : "Dream";
  const {conversation_only: chatTurns = 0, excluded_lessons: excluded = 0,
    training_lessons: training = 0} = data.selection || {};
  const skipped = [
    chatTurns > 0 ? quantity(chatTurns, "chat-only turn") : "",
    excluded > 0 ? quantity(excluded, "lesson") : "",
  ].filter(Boolean);
  byId("dream-selection").textContent = skipped.length ?
    "Skipped " + skipped.join(" and ") + ". " + quantity(training, "training candidate") + "." : "";
  byId("dream-selection").hidden = !skipped.length;
  renderDreamQuestions(data);
  syncSampling(data, started);
  renderTraining(data.training);
  renderMorningQuestions(data);
  controls();

  const counts = [[data.intake?.captured_count ?? data.entries?.length, "Notes in this Dream"],
    [data.counts?.pending, "Notes waiting"], [data.intake?.recognized_lessons ?? data.counts?.questions, "Questions from notes"],
    [data.intake?.trainable_lessons ?? data.counts?.neural_examples, "Candidates for local 3B"]];
  byId("dream-counts").replaceChildren(...counts.filter(([value]) => Number.isFinite(value)).map(([value, label]) => {
    const item = node("li");
    item.append(node("strong", String(value)), node("span", label));
    return item;
  }));
  const history = (data.history || []).filter((run) => run.night > 0 && run.night < data.night);
  byId("dream-history").hidden = !history.length;
  byId("dream-history-list").replaceChildren(...history.map((run) => {
    const item = node("li");
    item.append(node("strong", "Dream " + run.night), node("span", run.status === "failed" ?
      "Stopped" : run.neural_updates + " memory updates"));
    return item;
  }));
  const events = (data.events || []).map((event) => node("li", event.message));
  byId("dream-events").replaceChildren(...(events.length ? events : [node("li", "No activity yet.")]));
  setDetails("dream-entries", (data.entries || []).map((item) => {
    const detail = node("details");
    detail.append(node("summary", item.source || "Approved message"), node("p", item.text || ""));
    if (item.category) detail.append(node("span", item.category, "dream-category"));
    return detail;
  }), "Save a chat message or add a TXT or Markdown file to give Dream material.");
}

async function refreshDream() {
  clearTimeout(pollTimer);
  const revision = ++dreamRevision;
  try {
    const value = await api("api/dream");
    if (revision === dreamRevision) {
      dreamLoaded = true;
      renderDream(value);
    }
  } catch (error) {
    if (revision === dreamRevision) {
      dreamLoaded = true;
      controls();
      if (dreaming) byId("dream-status").textContent = error.message;
    }
  } finally {
    if (revision === dreamRevision) scheduleDreamRefresh();
  }
}

function scheduleDreamRefresh() {
  if (dream.status === "running") {
    const watchingTraining = dreaming && !document.hidden && dream.stage === "train";
    pollTimer = setTimeout(refreshDream, watchingTraining ? 320 : 1000);
  }
}

async function startDream(retry = false) {
  if (busy || uploading || savingDreamNote || starting || dream.status === "running") return;
  dreamRevision += 1;
  clearTimeout(pollTimer);
  const previous = dream.night || 0;
  const previousAttempt = dream.attempt_id;
  starting = true;
  didFailToStart = false;
  byId("dream-status").textContent = "Starting Dream…";
  controls();
  renderFlow();
  try {
    const result = await api("api/dream", retry ? {retry: true} : {});
    starting = false;
    renderDream(result, result.status === "running" || result.night > previous ||
      (Boolean(result.attempt_id) && result.attempt_id !== previousAttempt));
    if (result.status === "running" || result.night > previous ||
        (retry && result.status === "complete" && !heldMemory(result) && !result.skill?.retryable)) {
      ranDuringVisit = true;
      if (result.status === "complete") completedDuringVisit = true;
      try { sessionStorage.setItem(storageKey("dreamRanDuringVisit"), "true"); } catch {}
    }
    scheduleDreamRefresh();
  } catch (error) {
    didFailToStart = true;
    byId("dream-status").textContent = error.message;
  } finally {
    starting = false;
    controls();
    renderFlow();
  }
}

function enterDream() {
  entryNight = dream.night || 0;
  ranDuringVisit = false;
  completedDuringVisit = false;
  try {
    sessionStorage.setItem(storageKey("dreamEntryNight"), String(entryNight));
    sessionStorage.removeItem(storageKey("dreamRanDuringVisit"));
  } catch {}
  view(true);
  if (dream.counts?.pending && dream.questions?.length && !dream.clarifications?.length && !dreamNoteInput.value &&
      dream.intake?.status !== "evidence_only") dreamNotePanel.open = false;
  refreshDream();
}

function enterMorning() {
  if (dream.status === "complete" && (ranDuringVisit || dream.night > entryNight) &&
      !pendingCapture && !input.value.trim()) {
    resetConversation("local");
    if (route(modePicker.value) === "remote" && learnedMode()) modePicker.value = learnedMode();
    showConversation();
  }
  view(false);
}

byId("dream-toggle").addEventListener("click", () => {
  if (dreaming) enterMorning();
  else enterDream();
});

byId("run-dream").addEventListener("click", (event) => {
  if (!event.currentTarget.disabled) startDream(event.currentTarget.dataset.retry === "true");
});
function markSampleSaved() {
  if (sampleSaved) return;
  sampleSaved = true;
  try { sessionStorage.setItem(storageKey("dreamSampleSaved"), "true"); } catch {}
  byId("sample-dream-help").textContent = "The synthetic sample has been saved.";
}

async function saveDreamNote({text, source, requestId}, isSample = false) {
  savingDreamNote = true;
  controls();
  dreamNoteStatus.textContent = "Queueing note locally…";
  try {
    await api("api/chat/save", {text, source, kind: "message", capture_request_id: requestId});
    if (isSample) markSampleSaved();
    else {
      dreamNoteInput.value = "";
      dreamNoteRequest = null;
    }
    dreamNoteStatus.textContent = isSample ? "Synthetic sample queued." : "Note queued.";
    await refreshDream();
  } catch (error) {
    dreamNoteStatus.textContent = error.message;
    return;
  } finally {
    savingDreamNote = false;
    controls();
  }
}

byId("dream-note-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (byId("save-dream-note").disabled) return;
  const text = dreamNoteInput.value.trim();
  if (text !== dreamNoteRequest?.text) {
    dreamNoteRequest = {text, source: "Dream note", requestId: crypto.randomUUID()};
  }
  saveDreamNote(dreamNoteRequest);
});
dreamNoteInput.addEventListener("input", controls);
byId("try-dream-sample").addEventListener("click", () => {
  if (!byId("try-dream-sample").disabled) saveDreamNote(dreamSample, true);
});

function line(role, content, channel, metadata, scroll = true) {
  byId("welcome").hidden = true;
  const item = node("article", undefined, "chat-turn " + role);
  item.setAttribute("data-channel", channel);
  item.append(node("strong", role === "user" ? "You" : "Vicena"), node("p", content));
  if (metadata) {
    if (metadata.local_only === true) {
      const link = node("a", "Open Memory wiki");
      link.href = "wiki";
      const action = node("p");
      action.append(link);
      item.append(action);
    }
    const details = node("details");
    const rows = metadata.injected_rows ?? metadata.vector_rows;
    const sources = metadata.sources || [];
    details.append(node("summary", "Inspect memory"));
    const mode = metadata.memory_mode || "No saved notes used";
    const provider = {openrouter: "OpenRouter", gemini: "Google Gemini"}[metadata.provider];
    const count = Number.isInteger(rows) && metadata.mode === "vector" ?
      rows + (rows === 1 ? " vector passed" : " vectors passed") : null;
    details.append(node("p", [metadata.model, provider, mode, count].filter(Boolean).join(" · ")));
    if (metadata.provider === "gemini" && typeof metadata.memory_readout === "string" && metadata.memory_readout) {
      details.append(node("p", "Summary sent to Gemini"), node("p", metadata.memory_readout));
    }
    if (Array.isArray(metadata.actions) && metadata.actions.length) {
      const actions = node("ol");
      for (const action of metadata.actions) {
        actions.append(node("li", `${action.step} · ${action.outcome}`));
      }
      details.append(actions);
    }
    const receipt = metadata.vector_receipt;
    if (receipt?.sha256) {
      details.append(node("p", "35B prefill receipt"), node("code", receipt.sha256, "memory-hash"));
    }
    if (receipt?.restored === true) details.append(node("p", "Read from restored local memory"));
    if (receipt?.state_sha256) {
      details.append(node("p", "Local memory checkpoint"), node("code", receipt.state_sha256, "memory-hash"));
    }
    for (const source of sources) {
      if (metadata.local_only && source.memory_id) {
        const link = node("a", source.title || source.source);
        link.href = "wiki#" + encodeURIComponent(source.memory_id);
        details.append(link, node("p", source.content));
      } else {
        details.append(node("p", source.source + " · " + source.content));
      }
    }
    item.append(details);
  }
  thread.append(item);
  if (scroll) item.scrollIntoView({block: "end"});
  return item;
}

function recent(turns, providerContext) {
  // Complete turns are stored as adjacent user/assistant pairs. Drop both
  // messages when either one marks the exchange as local-only.
  const isLocalOnly = (message) => message?.local_only === true || message?.metadata?.local_only === true;
  const context = providerContext ? turns.filter((message, index) => {
    const partner = index % 2 === 0 ? turns[index + 1] : turns[index - 1];
    return !isLocalOnly(message) && !isLocalOnly(partner);
  }) : turns;
  let bytes = 0;
  const selected = [];
  for (const {role, content} of context.slice(-17).reverse()) {
    const message = {role, content};
    bytes += new TextEncoder().encode(JSON.stringify(message)).length;
    if (bytes > 15000) break;
    selected.unshift(message);
  }
  if (selected[0]?.role === "assistant") selected.shift();
  return selected;
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || byId("send").disabled) return;
  busy = true;
  controls();
  input.value = "";
  const mode = modePicker.value;
  const channel = route(mode);
  const item = line("user", content, channel);
  const check = usesLearnedMemory(mode) && Boolean(dreamQuestion);
  dreamQuestion = "";
  const userMessage = {role: "user", content};
  const messages = [...conversations[channel], userMessage];
  const save = byId("remember").checked;
  let approved;
  flows[mode] = {stage: "input", message: "Sending message…"};
  renderFlow();
  try {
    if (save) {
      const previous = channel === "local" ? pendingCapture : remoteCapture;
      approved = previous?.content === content ? previous : {content, requestId: crypto.randomUUID()};
      if (channel === "local") {
        pendingCapture = approved;
        persistLocalChat();
      } else {
        remoteCapture = approved;
      }
    }
    if (mode === "remote") {
      status.textContent = "Waiting for OpenRouter…";
    } else if (mode === "gemini") {
      status.textContent = "Checking local memory and waiting for Gemini…";
    } else {
      waitingForLocal = true;
      setLocalStatus(localModelStatus);
      watchLocalStatus();
    }
    const context = check ? [userMessage] : recent(messages, mode === "gemini" || mode === "remote");
    const request = {messages: context, mode, save};
    if (approved) request.capture_request_id = approved.requestId;
    const result = await api("api/chat", request);
    byId("remember").checked = true;
    if (channel === "remote") remoteCapture = null;
    const metadata = {...result, mode};
    const assistantMessage = {role: "assistant", content: result.answer, metadata};
    const localOnly = result.local_only === true;
    if (localOnly) {
      userMessage.local_only = true;
      assistantMessage.local_only = true;
    }
    conversations[channel] = [...messages, assistantMessage];
    if (channel === "local") {
      if (pendingCapture?.content === content) pendingCapture = null;
      persistLocalChat();
      if (!localOnly && usesLocalAnswer(mode)) setLocalStatus("ready");
    }
    line("assistant", result.answer, channel, metadata);
    const rows = result.injected_rows ?? result.vector_rows;
    flows[mode] = {stage: "answer", localOnly, decoded: Boolean(result.memory_readout), message: [
      result.model, mode === "vector" && Number.isInteger(rows) ?
        rows + (rows === 1 ? " memory vector" : " memory vectors") : result.memory_mode,
    ].filter(Boolean).join(" · ")};
    status.textContent = result.capture_id ? "Queued for Dream review." : "Dream capture skipped for this message.";
    if (result.capture_id) refreshDream();
  } catch (error) {
    if (approved && error.captureId) {
      approved.captureId = error.captureId;
      if (channel === "local") persistLocalChat();
      refreshDream();
    }
    item.remove();
    byId("welcome").hidden = conversations[channel].length > 0;
    input.value = content;
    if (check) dreamQuestion = content;
    status.textContent = error.message;
    flows[mode] = {stage: null, message: approved?.captureId ?
      "Message saved for Dream. Reply incomplete." : "Message incomplete"};
    if (usesLocalAnswer(mode)) {
      api("api/chat/status").then((result) => setLocalStatus(result.local_status)).catch(() => {});
    }
  } finally {
    waitingForLocal = false;
    watchLocalStatus();
    busy = false;
    controls();
    renderFlow();
    if (!dreaming) input.focus();
  }
});

for (const button of exampleButtons) {
  button.addEventListener("click", () => {
    const example = button.dataset.example;
    const teaching = example === "teach";
    const mode = teaching ? modePicker.value : (learnedMode() || "text");
    if (modePicker.value !== mode) {
      modePicker.value = mode;
      showConversation();
    }
    input.value = button.querySelector(".example-prompt").textContent;
    byId("remember").checked = teaching;
    dreamQuestion = example === "recall" ? input.value : "";
    input.focus();
  });
}

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

function resetConversation(channel) {
  conversations[channel] = [];
  if (channel === "local") {
    pendingCapture = null;
    persistLocalChat();
  } else {
    remoteCapture = null;
  }
  input.value = "";
  byId("remember").checked = true;
  dreamQuestion = "";
  thread.querySelectorAll(".chat-turn").forEach((item) => {
    if (item.getAttribute("data-channel") === channel) item.remove();
  });
  for (const mode of availableModes) {
    if (route(mode) === channel) delete flows[mode];
  }
}

byId("new-chat").addEventListener("click", () => {
  resetConversation(route(modePicker.value));
  showConversation();
  view(false);
});

attachment.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  uploading = true;
  controls();
  status.textContent = "Saving document for Dream…";
  try {
    if (!/\.(txt|md)$/i.test(file.name)) throw new Error("Choose a TXT or Markdown file.");
    if (file.size > 12000) throw new Error("Choose a text or Markdown file under 12 KB.");
    await api("api/chat/save", {text: await file.text(), source: file.name, kind: "document"});
    status.textContent = "Saved " + file.name + " for Dream.";
    refreshDream();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    event.target.value = "";
    uploading = false;
    controls();
  }
});

modePicker.addEventListener("change", () => {
  const changedChannel = route(modePicker.value) !== shownChannel;
  showConversation();
  if (!changedChannel) return;
  byId("remember").checked = true;
  if (shownChannel === "local") {
    showPendingCapture();
  } else {
    input.value = "";
    dreamQuestion = "";
  }
});
restoreLocalChat();
renderDream(dream);
try {
  if (sessionStorage.getItem(storageKey("dreamSampleSaved")) === "true") markSampleSaved();
  entryNight = Number(sessionStorage.getItem(storageKey("dreamEntryNight"))) || 0;
  ranDuringVisit = sessionStorage.getItem(storageKey("dreamRanDuringVisit")) === "true";
  if (sessionStorage.getItem(storageKey("dreamView")) === "true") view(true);
} catch {}
const chatUrl = new URL(location.href);
if (chatUrl.searchParams.get("view") === "dream") {
  chatUrl.searchParams.delete("view");
  history.replaceState(null, "", chatUrl.href);
  view(true);
}
refreshDream();
api("api/chat/status").then((data) => {
  document.querySelectorAll(".dream-planner-disclosure").forEach((note) => { note.hidden = !data.gemini_available; });
  const options = [];
  function addMode(value, label) {
    const option = node("option", label);
    option.value = value;
    options.push(option);
    availableModes.add(value);
  }
  if (data.gemini_available) addMode("gemini", "Gemini · local learned memory");
  if (data.remote_available) addMode("remote", "Quick chat · OpenRouter");
  if (data.vector_available) addMode("vector", "Local 35B · neural vectors");
  if (data.local_available ?? data.available) {
    addMode("text", "Local · saved evidence");
    addMode("off", "Local · memory off");
  }
  if (options.length) modePicker.replaceChildren(...options);
  const preferredMode = data.gemini_available ? "gemini" :
    availableModes.has("vector") ? "vector" : data.default_mode;
  if (availableModes.has(preferredMode)) modePicker.value = preferredMode;
  byId("chat-options").hidden = options.length < 2;
  setLocalStatus(data.local_status);
  watchLocalStatus();
  showConversation();
  if (route(modePicker.value) === "local") showPendingCapture();
  else byId("remember").checked = true;
  if (!options.length) status.textContent = "No chat model is available.";
}).catch((error) => { status.textContent = error.message; });
