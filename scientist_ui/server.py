"""Serve the local Vicena Lightreason notebook and its synthetic evidence store."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
from stat import S_ISREG
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from lightreason_memory.scientist_fixtures import WORKFLOWS, seed_workflow_store
from lightreason_memory.store import Store


_ASSETS = {
    "/": ("static/chat.html", "text/html; charset=utf-8"),
    "/notebook": ("static/index.html", "text/html; charset=utf-8"),
    "/wiki": ("static/wiki.html", "text/html; charset=utf-8"),
    "/architecture": ("static/architecture.html", "text/html; charset=utf-8"),
    "/static/architecture.css": ("static/architecture.css", "text/css; charset=utf-8"),
    "/static/wiki.css": ("static/wiki.css", "text/css; charset=utf-8"),
    "/static/wiki.js": ("static/wiki.js", "text/javascript; charset=utf-8"),
    "/static/chat.js": ("static/chat.js", "text/javascript; charset=utf-8"),
    "/static/style.css": ("static/style.css", "text/css; charset=utf-8"),
    "/static/app.js": ("static/app.js", "text/javascript; charset=utf-8"),
    "/static/logo/lightreason.svg": ("static/logo/lightreason.svg", "image/svg+xml"),
    "/static/logo/favicon.ico": ("static/logo/favicon.ico", "image/x-icon"),
    "/static/logo/favicon-32x32.png": ("static/logo/favicon-32x32.png", "image/png"),
    "/static/logo/android-chrome-192x192.png": ("static/logo/android-chrome-192x192.png", "image/png"),
}
_STEP_MARKER = re.compile(r"\bStep (\d+)\s+")
_MAX_BODY_BYTES = 16_384
_MAX_REHEARSAL_PROOF_BYTES = 128 * 1024
_CAPTURE_REQUEST_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WORD = re.compile(r"[^\W_]+")


def _workflow_summary(fixture):
    return {"id": fixture.workflow_id, "title": fixture.title, "domain": fixture.domain}


def _steps(content):
    fragments = _STEP_MARKER.split(content)[1:]
    return [
        {"number": int(number), "text": step.strip()}
        for number, step in zip(fragments[::2], fragments[1::2])
    ]


def _snapshot_time(fixture, view):
    if view == "initial":
        return fixture.initial_as_of
    if view != "current":
        raise ValueError("view must be current or initial")
    return None


def _verified_restart_checks(path, checkpoint, checks):
    """Read recorded restart results for the current policy and rehearsal."""
    try:
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_REHEARSAL_PROOF_BYTES + 1)
        if len(encoded) > _MAX_REHEARSAL_PROOF_BYTES:
            return None
        proof = json.loads(encoded)
    except (OSError, ValueError, RecursionError):
        return None

    if not isinstance(proof, dict) or proof.get("verified") is not True or proof.get("checkpoint_sha256") != checkpoint:
        return None
    before, after = proof.get("before_run_id"), proof.get("after_run_id")
    if not all(isinstance(run, str) and _RUN_ID.fullmatch(run) for run in (before, after)) or before == after:
        return None
    rows = proof.get("restart_checks")
    if not isinstance(rows, list) or len(rows) != len(checks) or any(not isinstance(row, dict) for row in rows):
        return None

    by_id = {row["id"]: row for row in rows if isinstance(row.get("id"), str)}
    for case in checks:
        row = by_id.get(case["id"], {})
        action = case["day_two"]
        if ("expected" not in row or "actual" not in row or row.get("expected") != action
                or row.get("actual") != action or row.get("question") != case["question"]
                or row.get("passed") is not True or not isinstance(row.get("answer"), str)
                or not row["answer"].strip()):
            return None
        count = row.get("injected_rows")
        digest = row.get("vector_sha256")
        receipt = row.get("receipt")
        if type(count) is not int or type(row.get("vector_rows")) is not int or count != row["vector_rows"]:
            return None
        if action is None:
            if count != 0 or receipt is not None or digest is not None:
                return None
        elif (count <= 0 or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
              or not isinstance(receipt, dict) or type(receipt.get("rows")) is not int
              or receipt["rows"] != count or receipt.get("sha256") != digest):
            return None
    return rows


class Notebook:
    """Scope the local demo store by workflow and read revisions at event time."""

    def __init__(self, db_path):
        self._stores = {}
        self._observations = {}
        for workflow, fixture in WORKFLOWS.items():
            store = Store(db_path, "scientist-notebook-demo", workflow)
            seeded = seed_workflow_store(store, fixture, apply_correction=False)
            self._stores[workflow] = store
            self._observations[workflow] = seeded["records"]

    def workflows(self):
        return {"workflows": [_workflow_summary(fixture) for fixture in WORKFLOWS.values()]}

    def _scope(self, workflow):
        if not isinstance(workflow, str) or workflow not in WORKFLOWS:
            raise KeyError("workflow not found")
        return WORKFLOWS[workflow], self._stores[workflow], self._observations[workflow]

    def snapshot(self, workflow, view="current"):
        fixture, store, by_event = self._scope(workflow)
        as_of = _snapshot_time(fixture, view)
        correction = fixture.correction
        original = by_event[correction.target_event_id]
        previous = store.get(original["memory_id"], as_of=fixture.initial_as_of)

        # An idempotent write can occur from another request or process. Look up
        # the active protocol in the store on every read.
        try:
            active = store.get(original["memory_id"])
        except KeyError:
            active, = (
                row for row in store.search(correction.replacement)
                if row["supersedes_id"] == original["memory_id"]
            )
        applied = active["memory_id"] != previous["memory_id"]
        protocol = previous if as_of else active
        observations = []
        for observation in fixture.observations:
            if observation.is_transient:
                continue
            if observation.event_id == correction.target_event_id:
                record = protocol
            else:
                record = store.get(by_event[observation.event_id]["memory_id"], as_of=as_of)
            observations.append({**record, "role": observation.role, "slot_key": observation.slot_key})

        return {
            "workflow": _workflow_summary(fixture),
            "view": view,
            "as_of": as_of,
            "protocol": protocol,
            "steps": _steps(protocol["content"]),
            "observations": observations,
            "correction": {
                "applied": applied,
                "previous": previous,
                "current": active if applied else None,
                "description": f"{correction.previous_compact_value} → {correction.compact_value}",
                "previous_value": correction.previous_compact_value,
                "next_value": correction.compact_value,
                "next_source": correction.source,
                "next_timestamp": correction.timestamp,
            },
            "queries": [
                {"id": query.name, "question": query.question, "route": query.route,
                 "view": "initial" if query.as_of else "current",
                 "requires_correction": query.as_of is None and not query.abstain}
                for query in fixture.queries
            ],
        }

    def correct(self, workflow):
        fixture, store, by_event = self._scope(workflow)
        correction = fixture.correction
        store.correct(by_event[correction.target_event_id]["memory_id"], correction.replacement,
                      correction.source, correction.timestamp)
        return self.snapshot(workflow)

    def evidence(self, workflow, view, memory_id):
        fixture, store, _ = self._scope(workflow)
        return store.get(memory_id, as_of=_snapshot_time(fixture, view))

    def recall(self, workflow, view, *, query=None, question=None):
        fixture, store, _ = self._scope(workflow)
        as_of = _snapshot_time(fixture, view)
        if query is None:
            sources = [row for row in store.search(question, as_of=as_of, limit=5) if row["is_fact_evidence"]]
            return self._result(sources, status="matches")

        preset = next((preset for preset in fixture.queries if preset.name == query), None)
        if preset is None:
            raise KeyError("notebook query not found")
        if question is not None:
            raise ValueError("provide either a query preset or a question")
        if view != ("initial" if preset.as_of else "current"):
            raise ValueError("use the preset in its corresponding notebook snapshot")

        # A missing-measurement example uses its focused search terms and only
        # returns a record when it includes both the measurement and its subject.
        if preset.abstain:
            terms = set(_WORD.findall(preset.memory_query.casefold()))
            sources = [
                row for row in store.search(preset.memory_query, as_of=as_of)
                if row["is_fact_evidence"] and terms.issubset(_WORD.findall(row["content"].casefold()))
            ]
            return self._result(sources, status="matches")

        # Presets demonstrate the two kinds of evidence views: an exact protocol
        # lookup and the documented inputs for a planning question. Values in the
        # readout come from the selected revision.
        snapshot = self.snapshot(workflow, view)
        if preset.as_of is None and not snapshot["correction"]["applied"]:
            return self._result([], unavailable="Apply the recorded correction to see the updated protocol.")
        if preset.route == "explicit_lookup":
            return self._result([snapshot["protocol"]])
        sources = [row for row in snapshot["observations"] if row["role"] != "measurement"]
        return self._result(sources)

    @staticmethod
    def _result(sources, status="supported", *, unavailable="Insufficient evidence in this notebook snapshot."):
        return {
            "status": status if sources else "insufficient",
            "mode": "local_evidence_lookup",
            "answer": "\n\n".join(row["content"] for row in sources) if sources else unavailable,
            "sources": sources,
        }


class NotebookServer(ThreadingHTTPServer):
    def __init__(self, port, notebook, chat=None, *, dream=None):
        self.notebook = notebook
        self.chat = chat
        self.dream = dream
        self.run_id = uuid4().hex
        super().__init__(("127.0.0.1", port), NotebookHandler)


class NotebookHandler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=HTTPStatus.OK):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _request(self, dispatch):
        port = self.server.server_port
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} or (
            origin is not None and origin != f"http://{host}"
        ):
            self._json({"error": "Open the notebook through its local address."}, HTTPStatus.FORBIDDEN)
            return
        try:
            dispatch()
        except KeyError as exc:
            self._json({"error": exc.args[0]}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _dream(self, *, start=False, retry=False):
        try:
            if self.server.dream is None:
                raise RuntimeError("Dream unavailable")
            dream = self.server.dream
            report = dream.start(retry=retry) if start else dream.status()
            self._json({**report, "run_id": self.server.run_id})
        except (OSError, RuntimeError):
            self._json({"error": "Dream is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)

    def do_GET(self):
        self._request(self._get)

    def _get(self):
        request = urlsplit(self.path)
        if request.path in _ASSETS:
            asset, content_type = _ASSETS[request.path]
            try:
                body = (Path(__file__).resolve().parent / asset).read_bytes()
            except FileNotFoundError as exc:
                raise KeyError("notebook asset not found") from exc
            self._send(HTTPStatus.OK, body, content_type)
            return

        notebook = self.server.notebook
        query = {key: values[-1] for key, values in parse_qs(request.query).items()}
        workflow = query.get("workflow")
        view = query.get("view", "current")
        if request.path == "/api/chat/status":
            chat = self.server.chat
            vector = chat is not None and chat.vector_available
            remote = chat is not None and chat.remote_available
            gemini = chat is not None and chat.gemini_available
            self._json({"available": chat is not None, "vector_available": vector, "gemini_available": gemini,
                        "gemini_model": chat.gemini_model if gemini else None,
                        "remote_available": remote, "local_available": chat is not None and chat.local_available,
                        "default_mode": chat.default_mode if chat is not None else "remote",
                        "remote_model": chat.openrouter_model if remote else None,
                        "local_status": chat.local_status if chat is not None else "unavailable",
                        "run_id": self.server.run_id,
                        "model": chat.model_name if chat is not None else None})
        elif request.path == "/api/rehearsal":
            from .skill_rehearsal import CHECKS, DAYS, training_batch

            dream = self.server.dream
            report = dream.rehearsal_report() if dream is not None else {}
            day_two = report.get("day2", {})
            checkpoint = day_two.get("checkpoint_sha256")
            proof = getattr(self.server, "rehearsal_proof", None)
            evaluate = query.get("evaluate") == "1"
            if (day_two.get("promoted") is True and isinstance(checkpoint, str) and _SHA256.fullmatch(checkpoint)
                    and dream.trainer is not None and (evaluate or proof is not None)):
                try:
                    current = dream.trainer.skill_checkpoint_digest()
                except (OSError, ValueError):
                    current = None
                if current == checkpoint:
                    if evaluate:
                        batch = training_batch([{"text": lesson} for day in DAYS for lesson in day])
                        report["local_checks"] = dream.trainer.evaluate_skills(batch)
                    if proof is not None:
                        rows = _verified_restart_checks(proof, checkpoint, CHECKS)
                        if rows is not None:
                            report.update(restart_checks=rows, verified=True, proof_source="recorded")
            self._json({"days": DAYS, "checks": CHECKS, "report": report, "run_id": self.server.run_id})
        elif request.path == "/api/dream":
            self._dream()
        elif request.path == "/api/wiki":
            dream = self.server.dream
            notes, pending = dream.wiki_snapshot() if dream else ([], 0)
            self._json({"notes": notes, "pending_count": pending, "source": "local"})
        elif request.path == "/api/workflows":
            self._json(notebook.workflows())
        elif request.path == "/api/notebook":
            self._json(notebook.snapshot(workflow, view))
        elif request.path == "/api/evidence":
            self._json(notebook.evidence(workflow, view, query.get("id")))
        else:
            raise KeyError("notebook endpoint not found")

    def _chat(self, body):
        chat = self.server.chat
        mode = body.get("mode", chat.default_mode)
        messages = chat.validate_request(body.get("messages"), mode)
        save = body.get("save", True)
        if type(save) is not bool:
            raise ValueError("Set save to true or false.")
        capture_request_id = body.get("capture_request_id")
        if capture_request_id is not None and (
            not save or not isinstance(capture_request_id, str) or not _CAPTURE_REQUEST_ID.fullmatch(capture_request_id)
        ):
            raise ValueError("Provide a valid capture request ID only when saving a message.")

        capture_id = None
        if save:
            try:
                capture_id = chat.save(messages[-1]["content"], "Saved chat message", capture_request_id=capture_request_id)
            except (OSError, sqlite3.Error) as error:
                self.log_error("Capture save failed (%s)", type(error).__name__)
                self._json({"error": "The message could not be saved for Dream. Try again."},
                           HTTPStatus.SERVICE_UNAVAILABLE)
                return

        try:
            reply = chat.reply(messages, mode=mode)
        except (OSError, RuntimeError, ValueError) as error:
            if isinstance(error, ValueError):
                detail = str(error)
                status = HTTPStatus.BAD_REQUEST
            else:
                provider = {"remote": "OpenRouter", "gemini": "Gemini chat with local memory"}.get(mode, "The local model")
                self.log_error("%s request failed (%s)", provider, type(error).__name__)
                detail = f"{provider} could not answer. Try again."
                status = HTTPStatus.SERVICE_UNAVAILABLE
            response = {"error": detail}
            if capture_id is not None:
                response.update(error=f"{detail} Your message was saved for Dream.", capture_id=capture_id)
            self._json(response, status)
        else:
            self._json({**reply, "run_id": self.server.run_id, "capture_id": capture_id})

    def do_POST(self):
        self._request(self._post)

    def _post(self):
        if self.headers.get_content_type() != "application/json":
            self._json({"error": "Send an application/json request."}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not 1 <= length <= _MAX_BODY_BYTES:
            raise ValueError("request body must be between 1 and 16384 bytes")
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")

        notebook = self.server.notebook
        workflow = body.get("workflow")
        if self.path == "/api/correction":
            self._json(notebook.correct(workflow))
        elif self.path == "/api/dream":
            retry = body.get("retry", False)
            if type(retry) is not bool:
                raise ValueError("Set retry to true or false.")
            self._dream(start=True, retry=retry)
        elif self.path == "/api/recall":
            self._json(notebook.recall(workflow, body.get("view", "current"),
                                       query=body.get("query"), question=body.get("question")))
        elif self.path in {"/api/chat", "/api/chat/save"}:
            if self.server.chat is None:
                self._json({"error": "The local chat model is offline."}, HTTPStatus.SERVICE_UNAVAILABLE)
            elif self.path == "/api/chat":
                self._chat(body)
            else:
                self._save_for_dream(body)
        else:
            raise KeyError("notebook endpoint not found")

    def _save_for_dream(self, body):
        capture_request_id = body.get("capture_request_id")
        if capture_request_id is not None and (
            not isinstance(capture_request_id, str) or not _CAPTURE_REQUEST_ID.fullmatch(capture_request_id)
        ):
            raise ValueError("Provide a valid capture request ID.")
        try:
            capture = self.server.chat.save(body.get("text"), body.get("source"), kind=body.get("kind", "message"),
                                            capture_request_id=capture_request_id)
        except (OSError, sqlite3.Error) as error:
            self.log_error("Capture save failed (%s)", type(error).__name__)
            self._json({"error": "The note could not be saved for Dream. Try again."}, HTTPStatus.SERVICE_UNAVAILABLE)
        else:
            self._json({"memory_id": capture})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--db", type=Path, default=Path.home() / ".lightreason" / "scientist-demo.sqlite3")
    parser.add_argument("--chat-model", type=Path, help="pinned local Qwen2.5-3B checkpoint")
    parser.add_argument("--chat-model-35b", type=Path)
    parser.add_argument("--chat-stage-35b", type=Path)
    parser.add_argument("--openrouter-key-file", type=Path, help="owner-only credential file")
    parser.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_MODEL"))
    parser.add_argument("--gemini-model", default=os.environ.get("VICENA_GEMINI_MODEL"))
    parser.add_argument("--gemini-key-file", type=Path, help="owner-only credential file")
    parser.add_argument("--dream-destination-model", type=Path, help="pinned destination embeddings and tokenizer")
    parser.add_argument("--dream-codec", type=Path)
    parser.add_argument("--dream-policy", type=Path)
    parser.add_argument("--dream-state", type=Path)
    parser.add_argument("--rehearsal-proof", type=Path)
    options = parser.parse_args(argv)
    if bool(options.chat_model_35b) != bool(options.chat_stage_35b):
        parser.error("Set both --chat-model-35b and --chat-stage-35b.")
    dream_paths = (options.dream_codec, options.dream_policy, options.dream_state)
    destination = options.dream_destination_model or options.chat_model_35b
    if (any(dream_paths) or options.dream_destination_model) and not all((*dream_paths, options.chat_model, destination)):
        parser.error("The neural trainer requires the 3B checkpoint, destination embeddings, codec, policy, and state.")
    if options.chat_model_35b and options.dream_destination_model and (
        options.chat_model_35b.expanduser().resolve() != options.dream_destination_model.expanduser().resolve()
    ):
        parser.error("The neural memory destination must match the local 35B checkpoint.")
    gemini = None
    if options.gemini_model or options.gemini_key_file:
        if not options.gemini_model or not all(dream_paths):
            parser.error("Gemini chat requires a Gemini model and the local Dream trainer.")
        key = os.environ.get("GEMINI_API_KEY")
        if options.gemini_key_file:
            try:
                fd = os.open(options.gemini_key_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, encoding="utf-8") as stream:
                    info = os.fstat(stream.fileno())
                    if not S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                        parser.error("The Gemini key file must be owner-only.")
                    key = stream.read(4096).strip()
            except (OSError, ValueError):
                parser.error("Could not read the Gemini key file.")
        if not key:
            parser.error("Set GEMINI_API_KEY or supply --gemini-key-file for Gemini chat.")
        from .gemini_chat import GeminiReasoner

        try:
            gemini = GeminiReasoner(options.gemini_model, key)
        except ValueError as error:
            parser.error(str(error))

    from .dream import Dream, PolicyTrainer

    trainer = None
    if all(dream_paths):
        trainer = PolicyTrainer(options.chat_model, *dream_paths,
                                destination_model_path=destination)
    planner = None
    if gemini:
        from .dream_planner import DreamPlanner

        planner = DreamPlanner(gemini)
    dream = Dream(options.db, trainer=trainer, planner=planner)
    chat = None
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if options.openrouter_key_file:
        credential = options.openrouter_key_file
        try:
            if credential.stat().st_mode & 0o077:
                parser.error("The OpenRouter key file must be owner-only.")
            openrouter_key = credential.read_text().strip()
        except OSError:
            parser.error("Could not read the OpenRouter key file.")
    remote_model = options.openrouter_model if openrouter_key else None
    if options.chat_model or options.chat_model_35b or remote_model or gemini:
        from .chat import LocalChat

        chat = LocalChat(options.db, options.chat_model, dream=dream, model_35b=options.chat_model_35b,
                         stage_35b=options.chat_stage_35b,
                         vectors=trainer.vectors if trainer and options.chat_model_35b else None,
                         openrouter_model=remote_model, openrouter_api_key=openrouter_key if remote_model else None,
                         gemini=gemini, memory_readout=trainer.readout if gemini else None)
    try:
        with NotebookServer(options.port, Notebook(options.db), chat=chat, dream=dream) as server:
            server.rehearsal_proof = options.rehearsal_proof
            print(f"Vicena Lightreason at http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if chat is not None:
            chat.close()


if __name__ == "__main__":
    main()
