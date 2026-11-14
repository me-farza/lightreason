"""CPU tests for local memory persistence, history, and retrieval."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lightreason_memory.store import EventConflictError, Store


START = "2026-10-01T12:00:00+00:00"
REVISION = "2026-10-02T12:00:00+00:00"
LATEST = "2026-10-03T12:00:00+00:00"


class StoreTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "private" / "memory.sqlite3"
        self.store = Store(self.path, "user", "workspace")

    def add(self, event="reading", text="The lab recorded 22 watts for the sensor.",
            timestamp=START, kind="tool_observation"):
        return self.store.observe(event, text, "synthetic meter receipt", timestamp, kind)

    def test_observation_survives_reopen_and_duplicate_is_idempotent(self):
        first = self.add()
        reopened = Store(self.path, "user", "workspace")
        duplicate = reopened.observe("reading", first["content"], first["source"],
                                     "2026-10-01T05:00:00-07:00", "tool_observation")
        self.assertEqual(first, duplicate)
        self.assertEqual(reopened.search("watts"), [first])
        self.assertEqual(reopened.get(first["memory_id"]), first)
        self.assertEqual(first["timestamp"], "2026-10-01T12:00:00.000000+00:00")
        self.assertNotEqual(first["recorded_at"], first["timestamp"])

    def test_event_id_conflicts_are_explicit_and_leave_the_original(self):
        first = self.add()
        for arguments in (
            ("Different content", first["source"], START, "tool_observation"),
            (first["content"], "different source", START, "tool_observation"),
            (first["content"], first["source"], REVISION, "tool_observation"),
            (first["content"], first["source"], START, "user_report"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(EventConflictError):
                self.store.observe("reading", *arguments)
        self.assertEqual(self.store.search("sensor"), [first])

    def test_concurrent_retries_share_one_record(self):
        def write(_):
            return self.add()["memory_id"]

        with ThreadPoolExecutor(max_workers=8) as threads:
            ids = list(threads.map(write, range(16)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.store.search("watts")), 1)

    def test_corrections_preserve_scoped_event_time_history(self):
        first = self.add(kind="user_report")
        revised = self.store.correct(first["memory_id"], "The sensor used 24 watts.", "corrected report", REVISION)
        latest = self.store.correct(revised["memory_id"], "The sensor used 23 watts.", "final report", LATEST)

        self.assertEqual(revised["supersedes_id"], first["memory_id"])
        self.assertEqual(latest["supersedes_id"], revised["memory_id"])
        self.assertEqual(latest["kind"], "user_report")
        self.assertEqual(latest["source"], "final report")
        self.assertIsNone(latest["event_id"])
        self.assertEqual(self.store.search("sensor watts"), [latest])
        self.assertEqual(self.store.search("22"), [])
        self.assertEqual(self.store.search("sensor watts", as_of="2026-10-01T11:59:59Z"), [])
        self.assertEqual(self.store.search("sensor watts", as_of=START), [first])
        self.assertEqual(self.store.search("sensor watts", as_of="2026-10-02T04:59:59-07:00"), [first])
        self.assertEqual(self.store.search("sensor watts", as_of=REVISION), [revised])
        self.assertEqual(self.store.search("sensor watts", as_of=LATEST), [latest])
        self.assertEqual(self.store.get(first["memory_id"], as_of=START), first)
        with self.assertRaises(KeyError):
            self.store.get(first["memory_id"])
        with self.assertRaises(KeyError):
            self.store.get(revised["memory_id"], as_of=START)
        receipt = self.add(kind="user_report")
        self.assertEqual(receipt["status"], "superseded")
        self.assertEqual(receipt["superseded_by_id"], revised["memory_id"])

    def test_correction_retries_and_invalid_timelines(self):
        original = self.add(timestamp=REVISION)
        with self.assertRaises(ValueError):
            self.store.correct(original["memory_id"], "Earlier sensor result", "test", START)
        successor = self.store.correct(original["memory_id"], "Revised sensor result", "test", REVISION)
        repeated = self.store.correct(original["memory_id"], "Revised sensor result", "test", REVISION)
        self.assertEqual(successor, repeated)
        self.assertEqual(self.store.search("sensor", as_of=REVISION), [successor])
        with self.assertRaises(ValueError):
            self.store.correct(original["memory_id"], "Conflicting sensor result", "test", LATEST)
        self.assertEqual(self.store.search("sensor"), [successor])

    def test_user_and_workspace_isolation_including_shared_event_ids(self):
        first = self.add()
        other_user = Store(self.path, "another user", "workspace")
        other_workspace = Store(self.path, "user", "another workspace")
        for other in (other_user, other_workspace):
            self.assertEqual(other.search("watts"), [])
            with self.assertRaises(KeyError):
                other.get(first["memory_id"], as_of=LATEST)
            with self.assertRaises(KeyError):
                other.correct(first["memory_id"], "Outside this scope", "test", REVISION)
            own = other.observe("reading", "An independent sensor reading", "test", START, "user_report")
            self.assertNotEqual(own["memory_id"], first["memory_id"])
            self.assertEqual(other.search("sensor"), [own])
        self.assertEqual(self.store.search("sensor"), [first])

    def test_retrieval_ranks_by_provenance_relevance_time_and_insertion(self):
        low = self.add("low", "Agreement was measured.", LATEST)
        older = self.add("older", "Output agreement was complete.")
        current = self.add("current", "Complete agreement for output.", REVISION, "user_report")
        tied = self.add("tied", "Complete output agreement.", REVISION)
        proposal = self.add("proposal", "Proposal for complete output agreement.", LATEST, "assistant_proposal")
        inference = self.add("inference", "Inferred complete output agreement.", LATEST, "inference")
        expected = [tied, current, older, low, inference, proposal]
        for _ in range(3):
            self.assertEqual(self.store.search("Which preserved complete output agreement?"), expected)
        self.assertEqual(self.store.search("output agreement", limit=2), expected[:2])
        self.assertEqual(self.store.search("astronomy"), [])
        self.assertEqual(self.store.search("Which was it?"), [])
        self.assertTrue(current["is_fact_evidence"])
        self.assertFalse(proposal["is_fact_evidence"])
        self.assertFalse(inference["is_fact_evidence"])
        edited_proposal = self.store.correct(proposal["memory_id"], "Revised proposal for agreement.", "test", LATEST)
        self.assertFalse(edited_proposal["is_fact_evidence"])

    def test_fallback_persists_and_fts_reindexes_when_available(self):
        fallback_path = self.path.parent / "fallback.sqlite3"
        with patch.object(Store, "_setup_fts", return_value=False):
            fallback = Store(fallback_path, "user", "workspace")
            saved = fallback.observe("sensor", "Three sensor readings.", "fixture", START, "tool_observation")
            self.assertEqual(fallback.search("sensor"), [saved])
        reopened = Store(fallback_path, "user", "workspace")
        self.assertEqual(reopened.search("sensor"), [saved])
        if reopened._fts_available:
            self.assertEqual(reopened.search("sensors"), [saved])
            with sqlite3.connect(fallback_path) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM memory_fts").fetchone()[0], 1)

    def test_database_is_private_and_rejects_symlinks_and_public_files(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        linked = self.path.with_name("linked.sqlite3")
        linked.symlink_to(self.path)
        with self.assertRaises(OSError):
            Store(linked, "user", "workspace")
        self.path.chmod(0o644)
        with self.assertRaises(PermissionError):
            Store(self.path, "user", "workspace")

    def test_timestamp_and_input_bounds(self):
        for invalid in ("yesterday", "2026-10-01", "2026-10-01T12:00:00", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.add(timestamp=invalid)
        with self.assertRaises(ValueError):
            self.add(kind="confirmed")
        with self.assertRaises(ValueError):
            self.add(text="x" * 16_385)
        with self.assertRaises(ValueError):
            self.store.search("x" * 2049)
        with self.assertRaises(ValueError):
            self.store.search("sensor", limit=True)
        with self.assertRaises(ValueError):
            self.store.search("sensor", as_of="2026-10-02")


if __name__ == "__main__":
    unittest.main()
