"""Invite-only HTTPS gateway for a shared fictional Vicena demo workspace."""

from argparse import ArgumentParser
from collections import deque
from hashlib import sha256
from hmac import compare_digest, digest
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from secrets import token_hex
from stat import S_IMODE, S_ISDIR, S_ISREG
from tempfile import NamedTemporaryFile
from threading import BoundedSemaphore, Lock
from time import monotonic, time
from urllib.parse import parse_qs, urlsplit


_COOKIE = "__Host-vicena-demo"
_PATH = re.compile(r"/[a-z][a-z0-9-]{0,62}")
_MAX_BODY = 16_384
_MAX_RESPONSE = 4 * 1024 * 1024
_SESSION_SECONDS = 6 * 60 * 60
_SESSION_KEY_BYTES = 32
SHORT_INVITE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_SHORT_INVITE_GROUP = rf"[{SHORT_INVITE_ALPHABET}]{{4}}"
_INVITE = re.compile(rf"{_SHORT_INVITE_GROUP}(?:-{_SHORT_INVITE_GROUP}){{2}}|[A-Za-z0-9_-]{{32,128}}")
INVITE_FORMAT_ERROR = "Use a random XXXX-XXXX-XXXX code or a URL-safe invite of 32 to 128 characters."
_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TOKEN = re.compile(r"(\d{1,12})\.([a-f0-9]{32})\.([a-f0-9]{64})")
_JSON = "application/json; charset=utf-8"
_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
_GET = frozenset({
    "/", "/architecture", "/notebook", "/wiki",
    "/static/architecture.css", "/static/style.css", "/static/chat.js", "/static/app.js",
    "/static/wiki.css", "/static/wiki.js", "/static/logo/lightreason.svg",
    "/static/logo/favicon.ico", "/static/logo/favicon-32x32.png",
    "/static/logo/android-chrome-192x192.png",
    "/api/chat/status", "/api/dream", "/api/wiki", "/api/workflows", "/api/notebook",
})
_NOTEBOOK_QUERY = {"workflow", "view"}
_POST = frozenset({"/api/chat", "/api/chat/save", "/api/dream", "/api/correction", "/api/recall"})
_LIMITS = {
    "invalid_invite": (24, 60),
    "/api/chat": (24, 300),
    "/api/chat/save": (24, 300),
    "/api/dream": (4, 600),
    "/api/correction": (12, 300),
    "/api/recall": (24, 300),
}
_LANDING = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<base href="/">
<title>Vicena Lightreason demo</title><link rel="stylesheet" href="_demo/style.css">
<script src="_demo/access.js" defer></script></head>
<body><main><h1>Vicena Lightreason</h1>
<form><label for="invite">Invite code</label><input id="invite" type="password" required maxlength="128"
autocomplete="off"><button type="submit">Open demo</button></form><p id="status" role="status"></p></main></body></html>"""
_ACCESS = b"""'use strict';
const form = document.querySelector('form');
const input = document.getElementById('invite');
const status = document.getElementById('status');
const invite = new URLSearchParams(location.hash.slice(1)).get('invite');
const home = new URL(document.baseURI);
const current = new URL(location.href);
if (current.hash) {
  current.hash = '';
  history.replaceState(null, '', current);
}
if (invite) input.value = invite;
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;
  status.textContent = 'Checking invite...';
  try {
    const response = await fetch(new URL('_demo/session', home), {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({invite: input.value.trim()})
    });
    if (!response.ok) {
      status.textContent = response.status === 429 ? 'Try again in a minute.' : 'This invite did not grant access.';
      return;
    }
    input.value = '';
    location.replace(current);
  } catch {
    status.textContent = 'Demo access is unavailable. Try again.';
  } finally {
    button.disabled = false;
  }
});
"""
_STYLE = b"""html{color-scheme:dark;background:#111619;color:#f2f1e9;font:16px/1.6 system-ui,sans-serif}
body{margin:0;padding:1rem}main{background:#181f21;border:1px solid #323f42;border-radius:14px;
max-width:34rem;margin:clamp(2rem,10vh,7rem) auto;padding:clamp(1.3rem,4vw,2.5rem)}
h1{font-size:clamp(1.8rem,6vw,2.7rem);line-height:1.1;margin:0 0 1rem}p{color:#b9c5bf}
form{display:grid;gap:.45rem;margin-top:1.4rem}input,button{font:inherit;padding:.55rem .7rem;border-radius:7px}
input{border:1px solid #667573;background:#111619}button{color:#111619;background:#f1d07e;border:0;
font-weight:650;cursor:pointer}button:disabled{opacity:.55}#status{min-height:1.6em;margin-bottom:0}"""
_PUBLIC = {
    "/_demo/access.js": (_ACCESS, "text/javascript; charset=utf-8"),
    "/_demo/style.css": (_STYLE, "text/css; charset=utf-8"),
}


def public_host(origin):
    """Accept one canonical HTTPS origin with no port override or path."""
    parsed = urlsplit(origin)
    if not _HOST.fullmatch(parsed.netloc) or origin != f"https://{parsed.netloc}":
        raise ValueError("Set the exact public origin as https://<hostname>.")
    return parsed.netloc


def invite_is_valid(value):
    return isinstance(value, str) and _INVITE.fullmatch(value) is not None


def read_invite(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, encoding="ascii") as file:
        info = os.fstat(file.fileno())
        if not S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Invite file must be owner-only.")
        invite = file.read(200).strip()
    if not invite_is_valid(invite):
        raise ValueError(INVITE_FORMAT_ERROR)
    return sha256(invite.encode("ascii")).digest()


def load_session_key(path):
    """Read or atomically create the private key shared by gateway restarts."""
    parent = path.parent.lstat()
    if not S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid() or parent.st_mode & 0o077:
        raise ValueError("The session key directory must be owner-only.")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(os.urandom(_SESSION_KEY_BYTES))
                file.flush()
                os.fsync(file.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass  # Another gateway published its key first.
            finally:
                temporary.unlink()
        fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as file:
        info = os.fstat(file.fileno())
        if not S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("The session key must be an owner-only regular file.")
        key = file.read(_SESSION_KEY_BYTES + 1)
    if len(key) != _SESSION_KEY_BYTES:
        raise ValueError("The session key must be exactly 32 bytes.")
    return key


class DemoGateway(ThreadingHTTPServer):
    def __init__(self, port, upstream_port, public_origin, invite_hash, *,
                 bind="127.0.0.1", public_path="", session_key=None):
        if bind not in ("127.0.0.1", "0.0.0.0"):
            raise ValueError("Use a loopback or container bind address.")
        if public_path and not _PATH.fullmatch(public_path):
            raise ValueError("Use a single path segment such as /lightreason.")
        self.public_path = public_path
        self.cookie_name = "__Secure-vicena-demo" if public_path else _COOKIE
        self.base = public_path + "/"
        self.public_host = public_host(public_origin)
        self.public_origin = public_origin
        self.upstream_port = upstream_port
        self.invite_hash = invite_hash
        root_key = session_key if session_key is not None else os.urandom(_SESSION_KEY_BYTES)
        if not isinstance(root_key, bytes) or len(root_key) != _SESSION_KEY_BYTES:
            raise ValueError("The session key must be exactly 32 bytes.")
        scope = f"vicena-demo-session/v1\0{public_origin}\0{self.base}\0".encode("ascii")
        self.secret = digest(root_key, scope + invite_hash, "sha256")
        self.budget_lock = Lock()
        self.usage = {key: deque() for key in _LIMITS}
        self.connections = BoundedSemaphore(16)
        self.upstream_calls = BoundedSemaphore(5)
        super().__init__((bind, port), DemoHandler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def consume(self, key):
        limit, period = _LIMITS[key]
        with self.budget_lock:
            now = monotonic()
            window = self.usage[key]
            while window and now - window[0] >= period:
                window.popleft()
            if len(window) >= limit:
                return False
            window.append(now)
            return True

    def issue_token(self):
        payload = f"{int(time()) + _SESSION_SECONDS}.{token_hex(16)}"
        return f"{payload}.{self.sign(payload)}"

    def sign(self, payload):
        return digest(self.secret, payload.encode("ascii"), "sha256").hex()

    def valid_token(self, token):
        match = _TOKEN.fullmatch(token)
        if not match:
            return False
        expires, nonce, signature = match.groups()
        return int(expires) > int(time()) and compare_digest(self.sign(f"{expires}.{nonce}"), signature)


class DemoHandler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(10)
        super().setup()

    def log_message(self, format, *args):
        # Request targets can contain credentials. Avoid access logging here.
        pass

    def _send(self, status, body=b"", content_type=_JSON, *, cookie=None):
        if content_type.startswith("text/html") and self.server.public_path:
            base = f'<base href="{self.server.base}">'.encode("ascii")
            body = body.replace(b'<base href="/">', base, 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CSP)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send(status, json.dumps({"error": message}).encode("utf-8"))

    def _valid_origin(self, *, required=False):
        host = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        return (host == [self.server.public_host] and
                (origins == [self.server.public_origin] or (not origins and not required)))

    def _session(self):
        cookies = SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
        except CookieError:
            return False
        name = self.server.cookie_name
        return name in cookies and self.server.valid_token(cookies[name].value)

    def _body(self, limit):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1:
            raise ValueError("Expected one JSON body.")
        if not re.fullmatch(r"[0-9]{1,6}", lengths[0]) or not 1 <= int(lengths[0]) <= limit:
            raise ValueError("Invalid request body size.")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Use an application/json body.")
        length = int(lengths[0])
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("Incomplete request body.")
        return data

    def _login(self):
        try:
            value = json.loads(self._body(256))
            invite = value.get("invite") if isinstance(value, dict) else None
        except (ValueError, OSError):
            self._error(HTTPStatus.BAD_REQUEST, "Provide an invite code.")
            return
        if not invite_is_valid(invite) or not compare_digest(
            sha256(invite.encode("ascii")).digest(), self.server.invite_hash,
        ):
            if self.server.consume("invalid_invite"):
                self._error(HTTPStatus.UNAUTHORIZED, "Invalid invite.")
            else:
                self._error(HTTPStatus.TOO_MANY_REQUESTS, "Try again in a minute.")
            return
        cookie = (f"{self.server.cookie_name}={self.server.issue_token()}; Path={self.server.base}; "
                  f"Max-Age={_SESSION_SECONDS}; "
                  "Secure; HttpOnly; SameSite=Strict")
        self._send(HTTPStatus.OK, b'{"ok":true}', cookie=cookie)

    def _proxy(self, path, body=None):
        if not self.server.upstream_calls.acquire(blocking=False):
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "The demo is busy. Try again shortly.")
            return
        port = self.server.upstream_port
        connection = HTTPConnection("127.0.0.1", port, timeout=48)
        headers = {"Host": f"127.0.0.1:{port}"}
        if body is not None:
            headers["Content-Type"] = _JSON
            headers["Origin"] = f"http://127.0.0.1:{port}"
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read(_MAX_RESPONSE + 1)
            content_type = response.getheader("Content-Type", _JSON)
            if len(data) > _MAX_RESPONSE or response.status not in {200, 400, 403, 404, 409, 415, 429, 500, 503}:
                raise ValueError("Invalid upstream response.")
        except (OSError, HTTPException, ValueError):
            self._error(HTTPStatus.BAD_GATEWAY, "The demo backend is unavailable.")
            return
        finally:
            connection.close()
            self.server.upstream_calls.release()
        self._send(response.status, data, content_type)

    def _target(self):
        # Validate the entire origin-form target before forwarding it.
        if not self.path.startswith("/") or self.path.startswith("//") or len(self.path) > 256:
            return "", ""
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            return "", ""
        if not parsed.path.startswith(self.server.base):
            return "", ""
        path = parsed.path.removeprefix(self.server.public_path)
        if parsed.query:
            if path == "/" and self.command == "GET" and parsed.query == "view=dream":
                return "/", "/"
            if path != "/api/notebook" or self.command != "GET":
                return "", ""
            try:
                params = parse_qs(parsed.query, strict_parsing=True, max_num_fields=2)
            except ValueError:
                return "", ""
            if set(params) - _NOTEBOOK_QUERY or any(
                len(values) != 1 or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", values[0])
                for values in params.values()
            ):
                return "", ""
        return path, path + ("?" + parsed.query if parsed.query else "")

    def do_GET(self):
        if not self._valid_origin():
            self._error(HTTPStatus.FORBIDDEN, "Open the published HTTPS demo address.")
            return
        path, target = self._target()
        if path in _PUBLIC:
            payload, content_type = _PUBLIC[path]
            self._send(HTTPStatus.OK, payload, content_type)
        elif path == "/" and not self._session():
            self._send(HTTPStatus.OK, _LANDING, "text/html; charset=utf-8")
        elif not self._session():
            self._error(HTTPStatus.UNAUTHORIZED, "An invite is required.")
        elif path in _GET:
            self._proxy(target)
        else:
            self._error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self):
        if not self._valid_origin(required=True):
            self._error(HTTPStatus.FORBIDDEN, "Open the published HTTPS demo address.")
            return
        path, target = self._target()
        if path == "/_demo/session":
            self._login()
            return
        if not self._session():
            self._error(HTTPStatus.UNAUTHORIZED, "An invite is required.")
            return
        if path not in _POST:
            self._error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        try:
            body = self._body(_MAX_BODY)
        except (ValueError, OSError):
            self._error(HTTPStatus.BAD_REQUEST, "Send a JSON body of at most 16384 bytes.")
            return
        if not self.server.consume(path):
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "The shared demo limit has been reached. Try later.")
            return
        self._proxy(target, body)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8784)
    parser.add_argument("--bind", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--public-path", default="")
    parser.add_argument("--upstream-port", type=int, default=8783)
    parser.add_argument("--public-origin", required=True)
    parser.add_argument("--invite-file", required=True)
    parser.add_argument("--session-key-file", type=Path, help="Private signing key persisted across restarts.")
    options = parser.parse_args()
    if not (1 <= options.port <= 65535 and 1 <= options.upstream_port <= 65535 and options.port != options.upstream_port):
        parser.error("Use different valid ports.")
    try:
        invite_hash = read_invite(options.invite_file)
        session_key = load_session_key(options.session_key_file) if options.session_key_file else None
        with DemoGateway(options.port, options.upstream_port, options.public_origin, invite_hash,
                         bind=options.bind, public_path=options.public_path, session_key=session_key) as server:
            print(f"Vicena demo gateway listening on {options.bind}:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
