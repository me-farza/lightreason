"""Start the isolated Vicena demo backend and its invite gateway."""

from contextlib import ExitStack
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from stat import S_ISREG
from subprocess import Popen, TimeoutExpired
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import json
import os
import re
import signal
import sys


APP_PORT = 8783
GATEWAY_PORT = 8080
DATA = Path("/data")
MODELS = Path("/models")
GEMINI_KEY = Path("/run/vicena/gemini-key")
INVITE = Path("/run/vicena/invite")
DEFAULT_MODEL = "gemini-3.8-flash"


def runtime_secret(name, mounted, folder):
    value = os.environ.pop(name, "").strip()
    if not value:
        try:
            source = os.open(mounted, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            raise ValueError(f"Set {name} or mount {mounted}.") from None
        with os.fdopen(source, encoding="ascii") as file:
            meta = os.fstat(file.fileno())
            if not S_ISREG(meta.st_mode) or meta.st_uid != os.geteuid() or meta.st_mode & 0o077:
                raise ValueError(f"{mounted.name} must be an owner-only file.")
            value = file.read(4097).strip()
    if len(value) > 4096 or not value.isascii() or any(character.isspace() for character in value):
        raise ValueError(f"{name} has an invalid format.")
    if name == "VICENA_DEMO_INVITE" and not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", value):
        raise ValueError("Use a random URL-safe demo invite with at least 32 characters.")
    if not value:
        raise ValueError(f"Set {name} or mount {mounted}.")
    target = folder / mounted.name
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as file:
        file.write(value)
    return target


def require_assets():
    source = MODELS / "qwen3b"
    destination = MODELS / "qwen35b-input"
    codec = MODELS / "codec.safetensors"
    policy = MODELS / "policy.safetensors"
    for folder, files in (
        (source, ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "model.safetensors.index.json")),
        (destination, ("config.json", "tokenizer.json", "tokenizer_config.json",
                       "model.safetensors.index.json", "lightreason-download.json")),
    ):
        for name in files:
            if not (folder / name).is_file():
                raise ValueError(f"Missing model artifact {folder / name}.")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    if not index or any(not (source / name).is_file() for name in set(index.values())):
        raise ValueError("Missing a pinned Qwen 3B weight shard.")
    index = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
    embed_shard = index.get("model.language_model.embed_tokens.weight")
    if embed_shard is None or not (destination / embed_shard).is_file():
        raise ValueError("Missing the pinned Qwen 35B input-embedding shard.")
    for file in (codec, policy):
        if not file.is_file():
            raise ValueError(f"Missing model artifact {file}.")
    return source, destination, codec, policy


def state_path():
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    if not any(row.split()[4] == str(DATA) for row in mounts):
        raise ValueError("Mount persistent storage at /data.")
    folder = DATA / "demo"
    folder.mkdir(mode=0o700, exist_ok=True)
    meta = folder.lstat()
    if not folder.is_dir() or folder.is_symlink() or meta.st_uid != os.geteuid() or meta.st_mode & 0o077:
        raise ValueError("The /data/demo directory must be private and owned by UID 10001.")
    return folder


def get(port, path, host):
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", path, headers={"Host": host})
        response = connection.getresponse()
        return response.status, response.read(65536)
    finally:
        connection.close()


def healthcheck():
    from scientist_ui.demo_gateway import public_host

    try:
        host = public_host(os.environ["VICENA_PUBLIC_ORIGIN"])
        backend_status, backend_reply = get(APP_PORT, "/api/chat/status", f"127.0.0.1:{APP_PORT}")
        gateway_status, _ = get(GATEWAY_PORT, os.environ.get("VICENA_PUBLIC_PATH", "").rstrip("/") + "/api/wiki", host)
        return 0 if backend_status == 200 and json.loads(backend_reply)["gemini_available"] and gateway_status == 401 else 1
    except (HTTPException, OSError, ValueError, KeyError):
        return 1


def wait_for_app(app, stopping):
    deadline = monotonic() + 40
    while not stopping.is_set() and app.poll() is None and monotonic() < deadline:
        try:
            status, body = get(APP_PORT, "/api/chat/status", f"127.0.0.1:{APP_PORT}")
            if status == 200 and json.loads(body)["gemini_available"]:
                return True
        except (HTTPException, OSError, ValueError, KeyError):
            pass
        stopping.wait(0.25)
    return False


def stop_children(children):
    for process in children:
        if process.poll() is None:
            process.terminate()
    deadline = monotonic() + 8
    for process in children:
        try:
            process.wait(timeout=max(0, deadline - monotonic()))
        except TimeoutExpired:
            process.kill()
            process.wait()


def run():
    from scientist_ui.demo_gateway import public_host

    os.umask(0o077)
    origin = os.environ.get("VICENA_PUBLIC_ORIGIN", "")
    public_host(origin)
    public_path = os.environ.get("VICENA_PUBLIC_PATH", "").rstrip("/")
    if public_path and not re.fullmatch(r"/[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*", public_path):
        raise ValueError("VICENA_PUBLIC_PATH must be an absolute URL path such as /lightreason.")
    source, destination, codec, policy = require_assets()
    state = state_path()
    stopping = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_args: stopping.set())

    with ExitStack() as stack:
        keys = Path(stack.enter_context(TemporaryDirectory(prefix="vicena-runtime-")))
        gemini_key = runtime_secret("GEMINI_API_KEY", GEMINI_KEY, keys)
        invite = runtime_secret("VICENA_DEMO_INVITE", INVITE, keys)
        environment = {name: value for name, value in os.environ.items() if not name.startswith("OPENROUTER_")}
        children = []
        stack.callback(stop_children, children)
        app = Popen([
            sys.executable, "-m", "scientist_ui.server",
            "--port", str(APP_PORT), "--db", str(state / "wiki.sqlite3"),
            "--chat-model", str(source), "--gemini-model", os.environ.get("VICENA_GEMINI_MODEL", DEFAULT_MODEL),
            "--gemini-key-file", str(gemini_key), "--dream-codec", str(codec), "--dream-policy", str(policy),
            "--dream-destination-model", str(destination), "--dream-state", str(state / "personal.safetensors"),
        ], env=environment)
        children.append(app)
        if not wait_for_app(app, stopping):
            if stopping.is_set():
                return 0
            raise RuntimeError("Vicena backend did not become available.")
        gateway = Popen([
            sys.executable, "-m", "scientist_ui.demo_gateway", "--bind", "0.0.0.0",
            "--port", str(GATEWAY_PORT), "--upstream-port", str(APP_PORT),
            "--public-origin", origin, "--invite-file", str(invite),
            *(["--public-path", public_path] if public_path else []),
        ], env=environment)
        children.append(gateway)
        while not stopping.wait(0.25):
            if any(process.poll() is not None for process in children):
                raise RuntimeError("Vicena backend or gateway exited.")
    return 0


if __name__ == "__main__":
    try:
        code = healthcheck() if sys.argv[1:] == ["--healthcheck"] else run()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Vicena startup: {error}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
