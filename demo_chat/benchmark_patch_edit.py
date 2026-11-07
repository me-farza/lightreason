"""Compare full-file and exact-patch edit prompts on the running local Studio.

Both paths use the selected runner and Until finished. The requests differ in
edit instructions and output format. Shape preparation, HTTP completion, model
generation, and local patch application have separate timings.
"""
from __future__ import annotations

import argparse
import hashlib
from html import unescape
from http.client import HTTPConnection
import json
from pathlib import Path
import re
import statistics
import time
from urllib.parse import urlsplit

if __package__:
    from .edit_patch import PATCH_SYSTEM_PROMPT, apply_model_patch, build_patch_prompt
    from .messages import DEFAULT_DESIGN_PROMPT
else:
    from edit_patch import PATCH_SYSTEM_PROMPT, apply_model_patch, build_patch_prompt
    from messages import DEFAULT_DESIGN_PROMPT


SOURCE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Focus dashboard</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #edf2f7; color: #172437; font: 16px system-ui, sans-serif; }
    main { width: min(100% - 2rem, 44rem); margin: 3rem auto; }
    h1 { margin: .25rem 0; font-size: clamp(1.8rem, 6vw, 3rem); }
    .tag { color: #53647a; font-size: .85rem; letter-spacing: .12em; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 1rem; }
    .card { padding: 1.25rem; background: white; border: 1px solid #d8e1eb; border-radius: 1rem; }
    .count { display: block; font-size: 2.5rem; line-height: 1.1; }
    .actions { display: flex; flex-wrap: wrap; gap: .65rem; margin: 1rem 0; }
    button { padding: .65rem .85rem; border: 0; border-radius: .5rem; cursor: pointer; }
    .primary { background: #2563eb; color: white; }
    .secondary { background: #e7edf5; color: #172437; }
    output { display: block; min-height: 1.5em; color: #53647a; }
  </style>
</head>
<body>
  <main>
    <span class="tag">DAILY FOCUS</span>
    <h1>Focus dashboard</h1>
    <p>Track completed focus rounds during this session.</p>
    <section class="grid" aria-label="Session summary">
      <article class="card"><strong class="count" id="rounds">0</strong>Completed rounds</article>
      <article class="card"><strong class="count">25</strong>Minutes per round</article>
    </section>
    <div class="actions">
      <button class="primary" id="add" type="button">Complete a round</button>
      <button class="secondary" id="reset" type="button">Reset</button>
    </div>
    <output id="message" aria-live="polite">Ready</output>
  </main>
  <script>
    let rounds = 0;
    const roundsNode = document.getElementById("rounds");
    const addButton = document.getElementById("add");
    const resetButton = document.getElementById("reset");
    const messageNode = document.getElementById("message");
    function updateStatus(message) {
      roundsNode.textContent = rounds;
      messageNode.textContent = message;
    }
    addButton.addEventListener("click", () => {
      rounds += 1;
      updateStatus("Round complete");
    });
    resetButton.addEventListener("click", () => {
      rounds = 0;
      updateStatus("Counter reset");
    });
  </script>
</body>
</html>"""

CASES = (
    ("text", "Change only the main heading from Focus dashboard to Study dashboard.",
     "<h1>Focus dashboard</h1>", "<h1>Study dashboard</h1>", "Study dashboard"),
    ("css", "Change only the primary button background from #2563eb to #0f766e.",
     ".primary { background: #2563eb; color: white; }", ".primary { background: #0f766e; color: white; }", "#0f766e"),
    ("javascript", "Change only the Complete a round button so each click adds 2 rounds instead of 1. Keep Reset working as before.",
     "rounds += 1;", "rounds += 2;", "rounds += 2"),
)
MAX_LINE_BYTES = 2 * 1024 * 1024


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def patch_protocol(name):
    if name == "markers":
        return PATCH_SYSTEM_PROMPT, apply_model_patch
    if __package__:
        from .edit_patch import JSON_PATCH_SYSTEM_PROMPT, apply_json_model_patch
    else:
        from edit_patch import JSON_PATCH_SYSTEM_PROMPT, apply_json_model_patch
    return JSON_PATCH_SYSTEM_PROMPT, apply_json_model_patch


def request_payloads(instruction, mode, patch_system_prompt=PATCH_SYSTEM_PROMPT):
    common = {"mode": mode, "execution": "sequential", "max_new_tokens": None}
    return {
        "whole": {**common, "app_action": "edit", "prompt": instruction, "previous_code": SOURCE,
                  "system_prompt": DEFAULT_DESIGN_PROMPT},
        "patch": {**common, "app_action": "new", "prompt": build_patch_prompt(SOURCE, instruction),
                  "system_prompt": patch_system_prompt},
    }


def loopback_address(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or
            parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Use the local Studio HTTP origin, such as http://127.0.0.1:8765")
    return parsed.hostname, parsed.port or 80


def post(address, route, payload, timeout, *, stream=False, changed_literal=None):
    """Read the actual loopback response with a per-request deadline."""
    began = time.perf_counter()
    connection = HTTPConnection(*address, timeout=timeout)
    response = None
    try:
        connection.request("POST", route, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            {"Content-Type": "application/json", "Accept": "application/x-ndjson" if stream else "application/json"})
        transport = connection.sock
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(65536).decode("utf-8", errors="replace")
            raise RuntimeError(f"{route} returned HTTP {response.status}: {detail}")
        if not stream:
            raw = response.read(MAX_LINE_BYTES + 1)
            if len(raw) > MAX_LINE_BYTES:
                raise RuntimeError("Readiness response exceeded the byte limit")
            return json.loads(raw), time.perf_counter() - began
        if not response.getheader("Content-Type", "").startswith("application/x-ndjson"):
            raise RuntimeError(f"{route} did not return NDJSON")
        selected, counts = [], {}
        first_token_s = None
        first_changed_literal_s = None
        for _ in range(8192):
            remaining = timeout - (time.perf_counter() - began)
            if remaining <= 0:
                raise TimeoutError(f"{route} exceeded its request deadline")
            if transport is not None:
                transport.settimeout(remaining)
            line = response.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_LINE_BYTES:
                raise RuntimeError("NDJSON line exceeded the byte limit")
            event = json.loads(line)
            kind = event.get("type")
            counts[kind] = counts.get(kind, 0) + 1
            if kind in {"token", "token_ids"}:
                observed = time.perf_counter() - began
                if first_token_s is None:
                    first_token_s = observed
                if (changed_literal and first_changed_literal_s is None and event.get("cumulative") is True and
                        isinstance(event.get("text"), str) and changed_literal in event["text"]):
                    first_changed_literal_s = observed
            else:
                selected.append(event)
            if kind == "done":
                return {"events": selected, "event_counts": counts, "client_first_token_s": first_token_s,
                    "client_first_changed_literal_in_cumulative_output_s": first_changed_literal_s}, time.perf_counter() - began
        raise RuntimeError(f"{route} ended before its final done event")
    finally:
        if response is not None:
            response.close()
        connection.close()


def event_of(result, kind):
    rows = [event for event in result["events"] if event.get("type") == kind]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {kind} event, received {len(rows)}")
    return rows[0]


def check_done(result):
    errors = [event.get("message") for event in result["events"] if event.get("type") == "error"]
    if errors or event_of(result, "done").get("success") is not True:
        raise RuntimeError("Studio request failed: " + "; ".join(map(str, errors)))


def ready(address, payload, timeout):
    deadline = time.perf_counter() + timeout
    began = time.perf_counter()
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Studio remained busy before this edit")
        state, _ = post(address, "/api/check-ready", payload, remaining)
        if not state.get("busy"):
            break
        time.sleep(min(.10, remaining))
    if state.get("until_finished") is not True or type(state.get("resolved_max_new_tokens")) is not int:
        raise RuntimeError("Studio must be ready and support the Until finished output policy")
    return {key: state.get(key) for key in ("prepared", "prompt_tokens", "resolved_max_new_tokens", "setup_s")}, time.perf_counter() - began


def warm(address, payload, timeout):
    before, check_s = ready(address, payload, timeout)
    result = {"before": before, "readiness_wall_s": check_s, "synthetic_prepare": None}
    if before["prepared"] is True:
        return result
    prepared, wall = post(address, "/api/prepare", payload, timeout, stream=True)
    check_done(prepared)
    if set(prepared["event_counts"]) & {"token", "token_ids", "completed", "generation", "comparison"}:
        raise RuntimeError("Preparation unexpectedly generated an answer")
    event = event_of(prepared, "prepared")
    if event.get("synthetic_only") is not True or event.get("answers_cached") is not False:
        raise RuntimeError("Preparation did not certify synthetic inputs and an empty answer cache")
    after, check_s = ready(address, payload, timeout)
    if (after["prepared"] is not True or before["prompt_tokens"] != after["prompt_tokens"] or
            before["resolved_max_new_tokens"] != after["resolved_max_new_tokens"]):
        raise RuntimeError("Preparation failed to warm the same public request shape")
    result.update(after=after, after_readiness_wall_s=check_s,
                  synthetic_prepare={"client_wall_s": wall, "server_setup_s": event.get("setup_s"),
                                     "synthetic_only": True, "answers_cached": False})
    return result


def extract_document(text):
    fences = re.findall(r"^[ \t]*```html?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\r?$", text, flags=re.I | re.M | re.S)
    if len(fences) == 1:
        return fences[0].strip()
    raw = text.strip()
    if not fences and re.match(r"<(?:!doctype\s+html|html)\b", raw, re.I):
        return raw
    raise ValueError("Expected a complete HTML document or one closed HTML code block")


def fixture_checks(html, case):
    """Conservative static checks for this fixture's requested behavior."""
    heading = "Study dashboard" if case == "text" else "Focus dashboard"
    color = "#0f766e" if case == "css" else "#2563eb"
    step = "2" if case == "javascript" else "1"
    heading_match = re.search(r"<h1\b[^>]*>(.*?)</h1\s*>", html, re.I | re.S)
    primary = re.search(r"\.primary\s*\{([^{}]+)\}", html, re.I)
    increment = re.search(r"\brounds\s*(?:\+=\s*" + step + r"\b|=\s*rounds\s*\+\s*" + step + r"\b)", html)
    if step == "1" and not increment:
        increment = re.search(r"\brounds\s*\+\+", html)
    return {
        "complete_document": bool(re.match(r"<(?:!doctype\s+html|html)\b", html, re.I) and
            re.search(r"</html\s*>\s*$", html, re.I) and all(re.search(r"</" + tag + r"\s*>", html, re.I)
                for tag in ("head", "body", "style", "script"))),
        "expected_heading": bool(heading_match and unescape(heading_match[1].strip()) == heading),
        "expected_primary_background": bool(primary and re.search(r"\bbackground(?:-color)?\s*:\s*" + color + r"\b", primary[1], re.I)),
        "expected_round_increment": bool(increment),
        "original_title_preserved": bool(re.search(r"<title>\s*Focus dashboard\s*</title>", html, re.I)),
        "counter_buttons_and_live_message_retained": all(re.search(r"\bid\s*=\s*['\"]" + name + r"['\"]", html)
            for name in ("rounds", "add", "reset", "message")) and "Complete a round" in html and "Reset" in html,
        "reset_and_click_handlers_retained": len(re.findall(r"\.addEventListener\s*\(\s*['\"]click['\"]", html)) >= 2 and
            len(re.findall(r"\brounds\s*=\s*0\b", html)) >= 2,
        "focus_summary_retained": all(label in html for label in ("Completed rounds", "Minutes per round", "25")),
    }


def assess(kind, case, raw, expected, native_eos, patch_parser=apply_model_patch):
    result = {"success": False, "exact_expected_source": False, "patch_count": None}
    if not native_eos:
        return {**result, "error": "Generation reached its output or context limit"}
    try:
        if kind == "patch":
            html, count = patch_parser(SOURCE, raw)
            result["patch_count"] = count
        else:
            html = extract_document(raw)
        exact = html == expected
        checks = fixture_checks(html, case)
        return {**result, "html": html, "exact_expected_source": exact, "static_fixture_checks": checks,
                "success": exact if kind == "patch" else all(checks.values())}
    except ValueError as error:
        return {**result, "error": str(error)}


def generate(address, payload, timeout, kind, case, expected, changed_literal, patch_parser=apply_model_patch):
    stream, wall = post(address, "/api/generate", payload, timeout, stream=True, changed_literal=changed_literal)
    check_done(stream)
    generation = event_of(stream, "generation")
    completed = event_of(stream, "completed")
    metrics = generation["metrics"]
    tokens = generation["token_ids"]
    if (generation.get("integrity_passed") is not True or generation.get("same_original_target_weights") is not True or
            metrics.get("until_finished") is not True or metrics.get("output_limit_requested") is not None or
            completed.get("token_ids") != tokens or completed.get("text") != generation.get("text") or
            not tokens or len(tokens) != metrics["tokens"] or stream["client_first_token_s"] is None):
        raise RuntimeError("Generation did not produce a complete integrity-checked live stream")
    began = time.perf_counter()
    assessment = assess(kind, case, generation["text"], expected, metrics.get("finished_eos") is True, patch_parser)
    return {"generated_tokens": len(tokens), "generation_s": metrics["seconds"], "client_wall_s": wall,
        "client_first_token_s": stream["client_first_token_s"], "model_first_token_s": metrics.get("time_to_first_token_s"),
        "client_first_changed_literal_in_cumulative_output_s": stream["client_first_changed_literal_in_cumulative_output_s"],
        "server_auto_setup_required": generation.get("auto_setup_required"),
        "server_auto_setup_s": generation.get("auto_setup_s"), "native_eos": metrics.get("finished_eos"),
        "stop_reason": metrics.get("stop_reason"), "context_capacity_new_tokens": metrics.get("context_capacity_new_tokens"),
        "target_sha256": generation.get("target_sha256"), "local_assessment_s": time.perf_counter() - began,
        "event_counts": stream["event_counts"], "raw_model_output": generation["text"], "raw_token_ids": tokens,
        "assessment": assessment}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", choices=("original", "strict", "optimized"), default="optimized")
    parser.add_argument("--patch-format", choices=("markers", "json"), default="markers")
    parser.add_argument("--case", choices=tuple(case[0] for case in CASES), help="Run one fixture edit instead of all three")
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists() or not 5 <= args.timeout <= 600:
        parser.error("Use a fresh output path and a timeout between 5 and 600 seconds")
    address = loopback_address(args.url)
    patch_system_prompt, patch_parser = patch_protocol(args.patch_format)
    report = {"schema": "lightreason-live-whole-vs-exact-patch-edit-v1", "complete": False, "arguments": vars(args),
        "source": SOURCE, "source_sha256": digest(SOURCE), "source_characters": len(SOURCE),
        "output_policy": "Until finished for both requests; resolved context capacities reported per shape",
        "setup_policy": "Check-ready and synthetic-only preparation outside generation timings",
        "quality_policy": "Exact patched source equality and conservative static full-file fixture checks",
        "first_changed_literal_scope": "Client-observed text in cumulative NDJSON, distinct from browser paint and completed edit",
        "performance_scope": "Different edit prompts and output lengths on one runner", "cases": [], "pairs": []}
    save(output, report)
    try:
        for index, (case, instruction, before, after, literal) in enumerate(CASES):
            if args.case is not None and case != args.case:
                continue
            if SOURCE.count(before) != 1:
                raise ValueError("Fixture edit must identify exactly one source segment")
            expected = SOURCE.replace(before, after, 1)
            payloads = request_payloads(instruction, args.mode, patch_system_prompt)
            record = {"case": case, "instruction": instruction, "expected_source_sha256": digest(expected),
                      "expected_replacement": {"search": before, "replace": after}, "changed_literal": literal,
                      "requests": payloads, "warmup": {}}
            report["cases"].append(record)
            initial_order = ("whole", "patch") if index % 2 == 0 else ("patch", "whole")
            for kind in initial_order:
                record["warmup"][kind] = warm(address, payloads[kind], args.timeout)
                save(output, report)
            for repeat in range(args.repeats):
                order = initial_order if repeat % 2 == 0 else tuple(reversed(initial_order))
                pair = {"case": case, "repeat": repeat, "order": list(order), "runs": {}}
                report["pairs"].append(pair)
                for kind in order:
                    preparation = warm(address, payloads[kind], args.timeout)
                    run = generate(address, payloads[kind], args.timeout, kind, case, expected, literal, patch_parser)
                    run["preparation_before_run"] = preparation
                    pair["runs"][kind] = run
                    save(output, report)
                whole, patch = pair["runs"]["whole"], pair["runs"]["patch"]
                if not whole["target_sha256"] or whole["target_sha256"] != patch["target_sha256"]:
                    raise RuntimeError("Edit pair did not attest to the same target weights")
                pair.update(observed_whole_over_patch_generation_ratio=whole["generation_s"] / patch["generation_s"],
                    observed_whole_over_patch_client_ratio=whole["client_wall_s"] / patch["client_wall_s"],
                    whole_over_patch_output_token_ratio=whole["generated_tokens"] / patch["generated_tokens"],
                    both_edit_checks_passed=whole["assessment"]["success"] and patch["assessment"]["success"],
                    both_generation_requests_already_warm=all(run["server_auto_setup_required"] is False
                                                            for run in (whole, patch)))
                save(output, report)
        eligible = [pair for pair in report["pairs"] if pair["both_edit_checks_passed"] and pair["both_generation_requests_already_warm"]]
        report["summary"] = {"pairs": len(report["pairs"]), "warm_pairs_with_both_edit_checks_passed": len(eligible),
            "successful_warm_generation_ratio_median": statistics.median(
                pair["observed_whole_over_patch_generation_ratio"] for pair in eligible) if eligible else None,
            "successful_warm_client_ratio_median": statistics.median(
                pair["observed_whole_over_patch_client_ratio"] for pair in eligible) if eligible else None}
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(output, report)
    print(json.dumps({"out": str(output), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
