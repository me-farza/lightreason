"""Teach and verify the synthetic two-day rehearsal through Vicena's local API."""

import argparse
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


def request(base, path, body=None, *, timeout=15):
    data = None if body is None else json.dumps(body).encode()
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        response = json.loads(error.read())
        capture = f" Saved teaching turn: {response['capture_id']}." if response.get("capture_id") else ""
        raise RuntimeError(f"{path}: {response.get('error', error.reason)}{capture}") from error


def write_report(path, report):
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def chat_status(base, mode):
    status = request(base, "/api/chat/status")
    if not status.get(f"{mode}_available"):
        raise RuntimeError(f"The {mode} chat route is unavailable.")
    return status


def teach(base, directory, mode):
    chat_status(base, mode)
    plan = request(base, "/api/rehearsal")
    if plan["report"] or request(base, "/api/dream")["counts"]["captures"]:
        raise ValueError("Start with a fresh rehearsal database and checkpoint directory.")
    messages = []
    for day, lessons in enumerate(plan["days"], 1):
        print(f"Sending Day {day} to Vicena with {mode} chat.", flush=True)
        messages.append({"role": "user", "content": "\n".join(lessons)})
        reply = request(base, "/api/chat", {
            "mode": mode, "messages": messages, "save": True, "capture_request_id": str(uuid4()),
        }, timeout=660)
        if not reply.get("capture_id"):
            raise RuntimeError(f"Day {day} teaching turn was not saved.")
        messages.append({"role": "assistant", "content": reply["answer"]})
        status = request(base, "/api/dream", {})
        deadline = monotonic() + 360
        last_stage = None
        while status["status"] == "running":
            if status.get("stage") != last_stage:
                last_stage = status.get("stage")
                print(f"Day {day} Dream: {last_stage}", flush=True)
            if monotonic() >= deadline:
                raise TimeoutError(f"Day {day} Dream timed out.")
            sleep(2)
            status = request(base, "/api/dream")
        plan = request(base, "/api/rehearsal")
        checkpoint = plan["report"].get(f"day{day}")
        if status["status"] != "complete" or not checkpoint or not checkpoint["promoted"]:
            raise RuntimeError(f"Day {day} did not promote: {json.dumps(checkpoint or status)}")
        print(f"Day {day} promoted after {checkpoint['optimizer_steps']} optimizer steps.", flush=True)
    write_report(directory / "trained.json", plan)


def matches_lesson(expected, answer):
    patterns = {
        "sealed_blank": (r"\bsealed blank\b", r"\b(?:ambient|environmental)\b"),
        "blind_scorer": (r"\b(?:blind\w*|hid(?:e|ing)|mask\w*|anonym\w*)\b",
                         r"\b(?:bias|expectation\w*)\b"),
    }
    return expected is None or (expected in patterns and all(
        re.search(pattern, answer, re.I) for pattern in patterns[expected]
    ))


def verify(base, directory, mode):
    status = chat_status(base, mode)
    gemini = mode == "gemini"
    model = status["gemini_model"] if gemini else "Qwen3.6-35B-A3B"
    before = json.loads((directory / "trained.json").read_text())
    after = request(base, "/api/rehearsal?evaluate=1", timeout=240)
    digest = before["report"]["day2"]["checkpoint_sha256"]
    if (before["run_id"] == after["run_id"] or not digest
            or after["report"]["day2"]["checkpoint_sha256"] != digest):
        raise RuntimeError("Expected a new service instance with the same Day 2 checkpoint.")
    checks = after["report"]["local_checks"]
    if {row["id"] for row in checks} != {"transfer", "retention", "abstention"} or len(checks) != 3:
        raise RuntimeError("The expected three local skill checks were not available.")
    results = []
    for check in checks:
        print(f"Asking {model} to verify {check['id']}.", flush=True)
        reply = request(base, "/api/chat", {"mode": mode, "save": False,
            "messages": [{"role": "user", "content": check["question"]}]}, timeout=660)
        rows, receipt = reply["injected_rows"], reply["vector_receipt"]
        supplied = check["expected"] is not None
        if gemini:
            readout = reply["memory_readout"]
            decoded = isinstance(readout, str) and matches_lesson(check["expected"], readout) if supplied else readout is None
            handoff = reply["provider"] == "gemini" and rows == 0 and receipt is None and decoded
            detail = "decoded readout" if readout else "no readout"
        else:
            exact = (bool(check["vector_sha256"]) and receipt == {
                "rows": check["vector_rows"], "sha256": check["vector_sha256"],
            }) if supplied else (receipt is None and check["vector_sha256"] is None)
            handoff = (rows > 0) == supplied and rows == check["vector_rows"] and exact
            detail = f"{rows} vectors"
        passed = (reply["run_id"] == after["run_id"] and reply["model"] == model
                  and reply["sources"] == [] and check["expected"] == check["actual"] and handoff
                  and matches_lesson(check["expected"], reply["answer"]))
        result = {**check, "answer": reply["answer"], "injected_rows": rows, "receipt": receipt, "passed": passed}
        if gemini:
            result["memory_readout"] = readout
        results.append(result)
        print(f"{check['id']}: {'passed' if passed else 'failed'} ({detail})", flush=True)
    report = {"verified": all(row["passed"] for row in results), "checkpoint_sha256": digest,
              "before_run_id": before["run_id"], "after_run_id": after["run_id"], "restart_checks": results}
    proof_name = "gemini-restart-proof.json" if gemini else "restart-proof.json"
    write_report(directory / proof_name, report)
    if not report["verified"]:
        raise RuntimeError(f"At least one restart check failed. See {proof_name}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teach", "verify"))
    parser.add_argument("--mode", choices=("vector", "gemini"), default="vector")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    args = parser.parse_args()
    endpoint = urlsplit(args.base_url.rstrip("/"))
    if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"}
            or endpoint.path or endpoint.query or endpoint.fragment or endpoint.username is not None):
        parser.error("Use a local Vicena HTTP address.")
    try:
        args.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        replay = teach if args.stage == "teach" else verify
        replay(endpoint.geturl(), args.run_dir, args.mode)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
