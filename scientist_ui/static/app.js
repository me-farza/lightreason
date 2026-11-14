"use strict";

const workflowSelect = document.getElementById("workflow");
const currentButton = document.getElementById("view-current");
const originalButton = document.getElementById("view-initial");
const correctionButton = document.getElementById("apply-correction");
const notebookElement = document.getElementById("notebook");
const message = document.getElementById("workspace-message");
const workspaceError = document.getElementById("workspace-error");
const queryForm = document.getElementById("lookup-form");
const querySelect = document.getElementById("example");
const questionInput = document.getElementById("question");
const queryButton = document.getElementById("run-query");
const resultElement = document.getElementById("lookup-result");
const resultStatus = document.getElementById("lookup-status");
const resultAnswer = document.getElementById("lookup-answer");
const resultSources = document.getElementById("lookup-sources");
const timestampFormat = new Intl.DateTimeFormat("en-GB", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: "UTC",
});

let notebook = null;
let queryRequest = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text ?? "";
  return node;
}

function timestamp(value, className) {
  const time = element("time", className);
  time.dateTime = value;
  time.textContent = timestampFormat.format(new Date(value)) + " UTC";
  return time;
}

async function request(path, body, signal) {
  const options = { signal };
  if (body) {
    options.method = "POST";
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Notebook request failed.");
  return data;
}

function connection(state, label) {
  document.getElementById("connection").className = "status-badge " + state;
  document.getElementById("connection-text").textContent = label;
}

function showError(text = "") {
  workspaceError.textContent = text;
  workspaceError.hidden = !text;
}

function clearResult() {
  queryRequest?.abort();
  queryRequest = null;
  queryButton.disabled = !notebook;
  resultElement.setAttribute("aria-busy", "false");
  resultStatus.hidden = true;
  resultAnswer.hidden = true;
  resultAnswer.textContent = "";
  resultSources.replaceChildren();
}

function showResultStatus(kind, text) {
  resultStatus.className = "evidence-status " + kind;
  resultStatus.textContent = text;
  resultStatus.hidden = false;
}

function sourceDetails(record, includeExcerpt = true) {
  const details = element("details", "record-source");
  details.append(element("summary", "", "Source " + record.source));
  if (includeExcerpt) details.append(element("blockquote", "", record.content));
  details.append(timestamp(record.timestamp, ""), element("div", "evidence-id", record.memory_id));
  return details;
}

function renderProtocol() {
  const badge = document.getElementById("protocol-badge");
  const isCorrected = Boolean(notebook.protocol.supersedes_id);
  badge.textContent = isCorrected ? "Corrected revision" : "Original revision";
  badge.className = "panel-badge" + (isCorrected ? " corrected" : "");
  badge.hidden = false;
  const previousText = notebook.correction.previous.content;
  const steps = notebook.steps.map((step) => {
    const row = element("li", "step");
    if (isCorrected && !previousText.includes(step.text)) row.classList.add("corrected");
    const number = element("span", "step-num", step.number);
    number.setAttribute("aria-hidden", "true");
    row.append(number, element("span", "step-text", step.text));
    return row;
  });
  document.getElementById("protocol-steps").replaceChildren(...steps);
  document.getElementById("protocol-source").replaceChildren(sourceDetails(notebook.protocol));
}

function revisionEntry(record, value, label, active) {
  const entry = element("div", "ct-entry " + (active ? "current" : "historical"));
  const dot = element("span", "ct-dot");
  dot.setAttribute("aria-hidden", "true");
  const content = element("div", "ct-content");
  const metadata = element("div", "ct-meta");
  const badge = element("span", "ct-label " + (active ? "current" : "superseded"), label);
  metadata.append(badge, timestamp(record.timestamp, "ct-ts"));
  content.append(metadata, element("div", "ct-body", value), sourceDetails(record));
  entry.append(dot, content);
  return entry;
}

function renderCorrection() {
  const correction = notebook.correction;
  const timeline = document.getElementById("correction-history");
  const previousActive = !correction.applied || notebook.view === "initial";
  const previousLabel = previousActive ? "Selected original" : "Original";
  const entries = [revisionEntry(correction.previous, correction.previous_value, previousLabel, previousActive)];
  if (correction.current) {
    const selected = notebook.view === "current";
    entries.unshift(revisionEntry(correction.current, correction.next_value, selected ? "Selected correction" : "Correction applied", selected));
  } else {
    const pending = element("div", "ct-entry historical");
    const dot = element("span", "ct-dot");
    dot.setAttribute("aria-hidden", "true");
    const content = element("div", "ct-content");
    const metadata = element("div", "ct-meta");
    metadata.append(element("span", "ct-label superseded", "Recorded correction"), timestamp(correction.next_timestamp, "ct-ts"));
    content.append(metadata, element("div", "ct-body", correction.next_value), element("div", "ct-source", correction.next_source));
    pending.append(dot, content);
    entries.unshift(pending);
  }
  timeline.replaceChildren(...entries);
  correctionButton.hidden = correction.applied;
}

const recordLabels = {
  controls: "Controls",
  ellipsometry: "Measured outcomes",
  enclosure: "Humidity exception",
  humidity: "Humidity exception",
  outcome: "Measured outcomes",
  precursor: "Experiment design",
  reference: "Controls",
  sol: "Experiment design",
};

function renderRecords() {
  const rows = notebook.observations.filter((record) => record.memory_id !== notebook.protocol.memory_id);
  document.getElementById("records").replaceChildren(...rows.map((record) => {
    const row = element("div", "design-row");
    const key = recordLabels[record.slot_key] || record.slot_key.replaceAll("_", " ");
    const text = element("div", "design-value", record.content);
    text.append(sourceDetails(record, false));
    row.append(element("div", "design-slot", key), text);
    return row;
  }));
}

function renderExamples() {
  const options = notebook.queries
    .filter((example) => example.view === notebook.view &&
      (!example.requires_correction || notebook.correction.applied))
    .map((example) => new Option(example.question, example.id));
  querySelect.replaceChildren(new Option("Write your own", ""), ...options);
  questionInput.value = "";
}

function renderNotebook(data) {
  notebook = data;
  workflowSelect.value = notebook.workflow.id;
  for (const [button, selected] of [[currentButton, notebook.view === "current"], [originalButton, notebook.view === "initial"]]) {
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  renderProtocol();
  renderCorrection();
  renderRecords();
  renderExamples();
  const snapshotLabel = notebook.view === "initial" ? "Original snapshot" : "Current snapshot";
  message.textContent = snapshotLabel;
  if (notebook.as_of) message.append(document.createTextNode(" · "), timestamp(notebook.as_of, ""));
  connection("ready", "Local evidence store");
}

function setBusy(busy) {
  notebookElement.setAttribute("aria-busy", String(busy));
  for (const control of [workflowSelect, currentButton, originalButton]) control.disabled = busy;
  for (const control of [querySelect, questionInput, queryButton]) control.disabled = busy || !notebook;
  correctionButton.disabled = busy || !notebook || notebook.correction.applied;
}

async function loadNotebook(workflow, view, apply = false) {
  clearResult();
  setBusy(true);
  showError();
  try {
    const params = new URLSearchParams({ workflow, view });
    const data = apply
      ? await request("/api/correction", { workflow })
      : await request("/api/notebook?" + params);
    renderNotebook(data);
    if (apply) message.append(document.createTextNode(" · Correction saved"));
  } catch (error) {
    if (notebook) workflowSelect.value = notebook.workflow.id;
    else connection("error", "Notebook unavailable");
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

function renderSources(sources) {
  resultSources.replaceChildren(...sources.map((record) => {
    const card = element("article", "evidence-item");
    card.append(sourceDetails(record));
    return card;
  }));
}

async function runQuery(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!notebook || !question) return;
  clearResult();
  const controller = new AbortController();
  queryRequest = controller;
  queryButton.disabled = true;
  resultElement.setAttribute("aria-busy", "true");
  showResultStatus("unknown", "Searching the selected revision…");
  try {
    const body = { workflow: notebook.workflow.id, view: notebook.view };
    if (querySelect.value) body.query = querySelect.value;
    else body.question = question;
    const result = await request("/api/recall", body, controller.signal);
    if (queryRequest !== controller) return;
    const labels = {
      supported: ["supported", "Supported by local sources"],
      matches: ["partial", "Matching records"],
      insufficient: ["unknown", "Insufficient evidence"],
    };
    showResultStatus(...(labels[result.status] || labels.insufficient));
    resultAnswer.textContent = result.answer;
    resultAnswer.hidden = !result.answer;
    renderSources(result.sources);
  } catch (error) {
    if (queryRequest === controller) showResultStatus("unknown", error.message);
  } finally {
    if (queryRequest === controller) {
      queryRequest = null;
      queryButton.disabled = false;
      resultElement.setAttribute("aria-busy", "false");
    }
  }
}

workflowSelect.addEventListener("change", () => loadNotebook(workflowSelect.value, "current"));
currentButton.addEventListener("click", () => loadNotebook(workflowSelect.value, "current"));
originalButton.addEventListener("click", () => loadNotebook(workflowSelect.value, "initial"));
correctionButton.addEventListener("click", () => loadNotebook(workflowSelect.value, "current", true));
querySelect.addEventListener("change", () => {
  questionInput.value = querySelect.value ? querySelect.selectedOptions[0].textContent : "";
  questionInput.focus();
});
questionInput.addEventListener("input", () => {
  querySelect.value = "";
});
queryForm.addEventListener("submit", runQuery);

async function start() {
  try {
    const { workflows } = await request("/api/workflows");
    workflowSelect.replaceChildren(...workflows.map((flow) => new Option(flow.title, flow.id)));
    if (!workflows.length) throw new Error("No experiment workflows are available.");
    await loadNotebook(workflows[0].id, "current");
  } catch (error) {
    connection("error", "Notebook unavailable");
    notebookElement.setAttribute("aria-busy", "false");
    showError(error.message);
  }
}

start();
