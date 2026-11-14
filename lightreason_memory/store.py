"""Transactional local memory records scoped to a configured user and workspace."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
from heapq import nlargest
import os
from pathlib import Path
import re
import sqlite3
import stat
from uuid import uuid4


KINDS = frozenset({"user_report", "tool_observation", "assistant_proposal", "inference"})
FACT_EVIDENCE_KINDS = frozenset({"user_report", "tool_observation"})
MAX_CONTENT_LENGTH = 16_384
MAX_QUESTION_LENGTH = 2_048
_WORDS = re.compile(r"[^\W_]+")
_STOP_WORDS = frozenset(
    "a an and are as at be by can did do does for from had has have how i in is it me of on "
    "not or please should tell than that the their there these they this to was were what when "
    "where which who why will with would you your".split()
)
_SCOPE_FILTER = "m.user_id = ? AND m.workspace_id = ?"


class EventConflictError(ValueError):
    """An event ID in this scope already identifies a different observation."""


def _required_text(value, field, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\0" in value:
        raise ValueError(f"{field} must be nonblank text of at most {maximum} characters")
    return value


def _timestamp(value):
    try:
        if isinstance(value, str) and len(value) <= 64:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, datetime) and value.utcoffset() is not None:
            return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (OverflowError, ValueError):
        pass
    raise ValueError("timestamp must be an ISO 8601 date and time with a UTC offset")


def _terms(text):
    return dict.fromkeys(word for word in _WORDS.findall(text.casefold()) if word not in _STOP_WORDS)


def _visibility(as_of):
    if as_of is None:
        return "m.status = 'active'", ()
    boundary = _timestamp(as_of)
    return (
        "m.timestamp <= ? AND (m.superseded_by_id IS NULL OR "
        "(SELECT s.timestamp FROM memory_records s WHERE s.memory_id = m.superseded_by_id "
        "AND s.user_id = m.user_id AND s.workspace_id = m.workspace_id) > ?)",
        (boundary, boundary),
    )


def _record(row, historical=False):
    record = {
        key: row[key] for key in (
            "memory_id", "event_id", "content", "source", "timestamp", "recorded_at", "kind",
            "status", "supersedes_id", "superseded_by_id",
        )
    }
    record["is_fact_evidence"] = record["kind"] in FACT_EVIDENCE_KINDS
    if historical:
        record["status"] = "active"
        record["superseded_by_id"] = None
    return record


class Store:
    """SQLite storage with event-time queries against the history known today.

    ``as_of`` includes records whose event time is at or before the boundary.
    A correction takes effect at its own event time, including at that instant.
    Reads without ``as_of`` use the latest recorded version, regardless of event
    time. ``recorded_at`` captures when each version was written to the database.
    """

    def __init__(self, path, user, workspace):
        self._scope = (_required_text(user, "user", 128), _required_text(workspace, "workspace", 128))
        self._path = Path(path).expanduser().absolute()
        if self._path.name == ":memory:":
            raise ValueError("a persistent database file path is required")
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise PermissionError("memory database must be an owner-only regular file owned by the service user")
        finally:
            os.close(fd)

        with self._connect(write=True) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    seq INTEGER PRIMARY KEY,
                    memory_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    event_id TEXT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN (
                        'user_report', 'tool_observation', 'assistant_proposal', 'inference'
                    )),
                    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
                    supersedes_id TEXT UNIQUE REFERENCES memory_records(memory_id),
                    superseded_by_id TEXT UNIQUE REFERENCES memory_records(memory_id),
                    CHECK ((status = 'active') = (superseded_by_id IS NULL)),
                    UNIQUE (user_id, workspace_id, event_id)
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS memory_scope_timestamp
                ON memory_records(user_id, workspace_id, timestamp)
            """)
            self._fts_available = self._setup_fts(db)

    @contextmanager
    def _connect(self, *, write=False):
        with closing(sqlite3.connect(self._path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db

    @staticmethod
    def _setup_fts(db):
        try:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content, tokenize='porter unicode61')")
            db.execute("""
                INSERT INTO memory_fts(rowid, content)
                SELECT seq, content FROM memory_records
                WHERE seq > (SELECT coalesce(max(rowid), 0) FROM memory_fts)
            """)
        except sqlite3.OperationalError as error:
            if "no such module: fts5" in str(error):
                return False
            raise
        return True

    def _insert(self, db, *, event_id=None, content, source, timestamp, kind, supersedes_id=None):
        memory_id = uuid4().hex
        cursor = db.execute("""
            INSERT INTO memory_records (
                memory_id, user_id, workspace_id, event_id, content, source, timestamp,
                recorded_at, kind, supersedes_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, *self._scope, event_id, content, source, timestamp,
              _timestamp(datetime.now(timezone.utc)), kind, supersedes_id))
        if self._fts_available:
            db.execute("INSERT INTO memory_fts(rowid, content) VALUES (?, ?)", (cursor.lastrowid, content))
        return self._find(db, memory_id)

    def _find(self, db, memory_id):
        return db.execute(
            f"SELECT m.* FROM memory_records m WHERE {_SCOPE_FILTER} AND m.memory_id = ?",
            (*self._scope, memory_id),
        ).fetchone()

    def observe(self, event_id, content, source, timestamp, kind):
        """Persist an observation. Exact retries return the original record."""
        event_id = _required_text(event_id, "event_id", 256)
        content = _required_text(content, "content", MAX_CONTENT_LENGTH)
        source = _required_text(source, "source", 1024)
        timestamp = _timestamp(timestamp)
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(sorted(KINDS))}")
        with self._connect(write=True) as db:
            existing = db.execute(
                f"SELECT m.* FROM memory_records m WHERE {_SCOPE_FILTER} AND m.event_id = ?",
                (*self._scope, event_id),
            ).fetchone()
            if existing is None:
                return _record(self._insert(db, event_id=event_id, content=content, source=source,
                                            timestamp=timestamp, kind=kind))
            if (existing["content"], existing["source"], existing["timestamp"], existing["kind"]) != (
                content, source, timestamp, kind,
            ):
                raise EventConflictError(f"event ID {event_id!r} already identifies a different observation")
            return _record(existing)

    def correct(self, memory_id, replacement, source, timestamp):
        """Supersede a version while retaining its original evidence and provenance."""
        memory_id = _required_text(memory_id, "memory_id", 128)
        replacement = _required_text(replacement, "replacement", MAX_CONTENT_LENGTH)
        source = _required_text(source, "source", 1024)
        timestamp = _timestamp(timestamp)
        with self._connect(write=True) as db:
            original = self._find(db, memory_id)
            if original is None:
                raise KeyError("memory record not found in this scope")
            if timestamp < original["timestamp"]:
                raise ValueError("correction timestamp precedes the original event time")
            if original["superseded_by_id"] is not None:
                successor = self._find(db, original["superseded_by_id"])
                if (successor["content"], successor["source"], successor["timestamp"]) == (
                    replacement, source, timestamp,
                ):
                    return _record(successor)
                raise ValueError("memory record has already been superseded")
            successor = self._insert(db, content=replacement, source=source, timestamp=timestamp,
                                     kind=original["kind"], supersedes_id=memory_id)
            db.execute("""
                UPDATE memory_records SET status = 'superseded', superseded_by_id = ?
                WHERE user_id = ? AND workspace_id = ? AND memory_id = ?
            """, (successor["memory_id"], *self._scope, memory_id))
            return _record(successor)

    def get(self, memory_id, as_of=None):
        """Get a visible version in this scope, or raise KeyError."""
        memory_id = _required_text(memory_id, "memory_id", 128)
        visible, dates = _visibility(as_of)
        with self._connect() as db:
            row = db.execute(f"""
                SELECT m.* FROM memory_records m WHERE {_SCOPE_FILTER}
                AND m.memory_id = ? AND {visible}
            """, (*self._scope, memory_id, *dates)).fetchone()
        if row is None:
            raise KeyError("memory record not found in this scope and time range")
        return _record(row, historical=as_of is not None)

    def search(self, question, as_of=None, limit=8):
        """Find scoped evidence by matching query terms, with stable rank and tie breaks."""
        question = _required_text(question, "question", MAX_QUESTION_LENGTH)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        visible, dates = _visibility(as_of)
        terms = tuple(_terms(question))[:32]
        if not terms:
            return []
        with self._connect() as db:
            if self._fts_available:
                placeholders = ", ".join("(?)" for _ in terms)
                phrases = tuple(f'"{term}"' for term in terms)
                rows = db.execute(f"""
                    WITH query_terms(term) AS (VALUES {placeholders})
                    SELECT m.*, count(*) AS matched FROM query_terms
                    JOIN memory_fts ON memory_fts MATCH query_terms.term
                    JOIN memory_records m ON m.seq = memory_fts.rowid
                    WHERE {_SCOPE_FILTER} AND {visible}
                    GROUP BY m.seq
                    ORDER BY (m.kind IN ('user_report', 'tool_observation')) DESC,
                             matched DESC, m.timestamp DESC, m.seq DESC
                    LIMIT ?
                """, (*phrases, *self._scope, *dates, limit)).fetchall()
            else:
                candidates = db.execute(
                    f"SELECT m.* FROM memory_records m WHERE {_SCOPE_FILTER} AND {visible}",
                    (*self._scope, *dates),
                )
                query_terms = set(terms)
                matches = ((len(query_terms.intersection(_terms(row["content"]))), row) for row in candidates)
                top = nlargest(limit, ((count, row) for count, row in matches if count), key=lambda item: (
                    item[1]["kind"] in FACT_EVIDENCE_KINDS, item[0], item[1]["timestamp"], item[1]["seq"],
                ))
                rows = [row for _, row in top]
        return [_record(row, historical=as_of is not None) for row in rows]
