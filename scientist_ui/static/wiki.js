"use strict";

const byId = (id) => document.getElementById(id);
const search = byId("wiki-search");
const formatter = new Intl.DateTimeFormat(undefined, {dateStyle: "medium"});
const categories = {process: "Procedure", experiment: "Experiment", preference: "Preference", profile: "Profile"};
let notes = [];
let index = new Map();
let activeTag = "";
let currentNote = null;
let pendingCount;

function element(tag, content, className) {
  const value = document.createElement(tag);
  if (content != null) value.textContent = content;
  if (className) value.className = className;
  return value;
}

function date(timestamp) {
  const value = new Date(timestamp);
  const time = element("time");
  if (timestamp && Number.isFinite(value.valueOf())) {
    time.dateTime = timestamp;
    time.textContent = formatter.format(value);
  }
  return time;
}

function noteLink(note, description) {
  const link = element("a");
  link.href = "#" + encodeURIComponent(note.id);
  link.append(element("span", note.title || "Untitled note", "wiki-link-title"));
  const subtitle = element("span", undefined, "wiki-link-subtitle");
  if (description) subtitle.append(document.createTextNode(description));
  const moment = date(note.timestamp);
  if (moment.textContent) subtitle.append(moment);
  link.append(subtitle);
  return link;
}

function showState(title, description = "", retry = false) {
  byId("wiki-note").hidden = true;
  byId("wiki-state-title").textContent = title;
  byId("wiki-state-description").textContent = description;
  byId("wiki-retry").hidden = !retry;
  byId("wiki-state").hidden = false;
}

function renderNote(note) {
  currentNote = note;
  const kind = categories[note.category] || note.category || "Note";
  const superseded = Boolean(note.superseded_by_id || note.status === "superseded");
  byId("wiki-note-meta").replaceChildren(element("span", kind), date(note.timestamp));
  if (superseded) byId("wiki-note-meta").append(element("span", "Superseded", "wiki-superseded"));
  byId("wiki-note-title").textContent = note.title || "Untitled note";
  byId("wiki-summary").textContent = note.summary || "";
  byId("wiki-summary").hidden = !note.summary;
  byId("wiki-provenance").hidden = !/^gemini(?:\b|$)/i.test(String(note.organized_by || ""));
  byId("wiki-source-name").textContent = note.source || "Saved note";
  byId("wiki-original").textContent = note.content || "";
  byId("wiki-tags").replaceChildren(...note.tags.map((tag) => {
    const item = element("li");
    const button = element("button", tag);
    button.type = "button";
    button.setAttribute("aria-pressed", String(tag.toLowerCase() === activeTag));
    button.addEventListener("click", () => {
      activeTag = activeTag === tag.toLowerCase() ? "" : tag.toLowerCase();
      render();
    });
    item.append(button);
    return item;
  }));
  const corrections = [
    [note.superseded_by_id, "Replaced by"],
    [note.supersedes_id, "Replaces"],
  ].filter(([id]) => id);
  byId("wiki-correction-list").replaceChildren(...corrections.map(([id, description]) => {
    const item = element("li");
    const related = index.get(id);
    item.append(related ? noteLink(related, description) :
      element("span", description + " a note outside the current index.", "wiki-missing"));
    return item;
  }));
  byId("wiki-corrections").hidden = !corrections.length;
  const topicTags = new Set(note.tags.map((tag) => tag.toLowerCase()));
  const related = notes.filter((item) => item.id !== note.id && item.tags.some((tag) => topicTags.has(tag.toLowerCase())));
  byId("wiki-related-list").replaceChildren(...related.slice(0, 8).map((item) => {
    const entry = element("li");
    entry.append(noteLink(item, item.tags.filter((tag) => topicTags.has(tag.toLowerCase())).join(" · ")));
    return entry;
  }));
  byId("wiki-related").hidden = related.length === 0;
  byId("wiki-state").hidden = true;
  byId("wiki-note").hidden = false;
}

function hashId() {
  try { return decodeURIComponent(window.location.hash.slice(1)); } catch { return ""; }
}

function render(requested = hashId()) {
  const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = notes.filter((note) =>
    (!activeTag || note.tags.some((tag) => tag.toLowerCase() === activeTag)) &&
    terms.every((term) => note.searchText.includes(term)));
  const selected = matches.find((note) => note.id === requested) || matches[0];
  const activeTopic = byId("wiki-tag-filter");
  activeTopic.hidden = !activeTag;
  activeTopic.textContent = activeTag ? "Clear topic · " + activeTag : "";
  byId("wiki-count").textContent = String(matches.length);
  byId("wiki-list").replaceChildren(...matches.map((note) => {
    const link = noteLink(note, categories[note.category] || note.category || "");
    if (note === selected) link.setAttribute("aria-current", "page");
    return link;
  }));
  if (!notes.length) {
    showState("No saved notes yet", pendingCount > 0 ?
      "Queued notes will appear here if Dream saves them as evidence." :
      "Save a message or document in chat, then run Dream to add it here.");
  } else if (!selected) {
    showState("No matching notes", "Try different words or clear the topic filter.");
  } else {
    renderNote(selected);
  }
}

async function load() {
  search.disabled = true;
  byId("wiki-queue").hidden = true;
  showState("Loading notes…");
  try {
    const response = await fetch("/api/wiki", {cache: "no-store"});
    if (!response.ok) throw new Error("Wiki unavailable");
    const payload = await response.json();
    if (!Array.isArray(payload.notes)) throw new Error("Invalid wiki");
    const pending = payload.pending_count;
    pendingCount = Number.isSafeInteger(pending) && pending >= 0 ? pending : undefined;
    if (pendingCount !== undefined) {
      byId("wiki-queue-count").textContent = pendingCount === 1 ? "1 note waiting for Dream" :
        pendingCount + " notes waiting for Dream";
      byId("wiki-queue-guidance").textContent = pendingCount > 0 ?
        "Choose Run Dream to review the queue and file notes here as evidence." :
        "Save a message or document in chat to add it to the queue.";
      byId("wiki-queue").hidden = false;
    }
    notes = payload.notes.filter((note) => typeof note?.id === "string" && note.id).map((note) => {
      const tags = Array.isArray(note.tags) ? note.tags.filter((tag) => typeof tag === "string" && tag.trim()) : [];
      const searchText = [note.title, note.summary, note.source, note.category, note.content, ...tags].join(" ").toLowerCase();
      return {...note, tags, searchText};
    });
    index = new Map(notes.map((note) => [note.id, note]));
    search.disabled = notes.length === 0;
    render();
    if (!window.location.hash && currentNote) {
      history.replaceState(null, "", "#" + encodeURIComponent(currentNote.id));
    }
  } catch {
    showState("Could not load notes", "The local wiki is unavailable.", true);
  }
}

search.addEventListener("input", () => render());
byId("wiki-tag-filter").addEventListener("click", () => {
  activeTag = "";
  render();
  search.focus();
});
byId("wiki-retry").addEventListener("click", load);
window.addEventListener("hashchange", () => {
  const id = hashId();
  if (!search.disabled && index.has(id)) {
    activeTag = "";
    search.value = "";
  }
  render(id);
});
document.addEventListener("click", (event) => {
  const link = event.target.closest("a[href^='#']");
  if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const id = decodeURIComponent(link.hash.slice(1));
  if (!index.has(id)) return;
  event.preventDefault();
  if (link.hash !== window.location.hash) history.pushState(null, "", link.hash);
  if (link.closest(".wiki-note")) {
    activeTag = "";
    search.value = "";
  }
  render(id);
  byId("wiki-note-title").focus();
});
load();
