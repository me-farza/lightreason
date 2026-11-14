"""Unit tests for synthetic materials-science and calibration workflow fixtures."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lightreason_memory.scientist_fixtures import (
    TIO2_UNSEEN_VARIATION,
    UNRELATED_CONTROL_OBSERVATIONS,
    WORKFLOWS,
    ZNO_WORKFLOW,
    score_response,
    seed_workflow_store,
    verify_fixtures,
)
from lightreason_memory.store import Store


class ScientistFixtureTests(unittest.TestCase):
    def test_verify_fixtures_passes_store_versioning_and_transient_filtering(self):
        report = verify_fixtures()
        self.assertTrue(report["verified"])
        self.assertEqual(set(report["workflows"]), {"zno-spin-xrd-v1", "tio2-dip-raman-v1"})
        self.assertEqual(report["workflows"]["zno-spin-xrd-v1"]["durable_records_seeded"], 5)
        self.assertEqual(report["workflows"]["zno-spin-xrd-v1"]["transient_skipped"], 1)
        self.assertEqual(report["unrelated_controls"], 1)

    def test_compact_values_fit_multi_slot_prefill_budget(self):
        for fixture in WORKFLOWS.values():
            durable = [obs for obs in fixture.observations if not obs.is_transient]
            self.assertGreaterEqual(len(durable), 5)
            for obs in durable:
                self.assertLessEqual(len(obs.compact_value.split()), 3)
            self.assertLessEqual(len(fixture.correction.compact_value.split()), 3)

    def test_rubric_scoring_distinguishes_stale_vs_corrected(self):
        by_name = {q.name: q for q in ZNO_WORKFLOW.queries}
        paraphrase = by_name["propose_after_correction_paraphrase"]

        good = (
            "Use 0.20 M precursor on fused silica with uncoated blank and NIST SRM 640f checks. "
            "Purge the spin bowl with dry nitrogen for 120 s at 62% RH, and anneal at 425 C "
            "with a 3 C/min ramp for 45 min."
        )
        stale = (
            "Run the furnace anneal at 350 C with a 10 C/min ramp, purge with dry nitrogen, "
            "and include the blank and SRM 640f standard."
        )
        self.assertTrue(score_response(paraphrase, good)["passed"])
        stale_score = score_response(paraphrase, stale)
        self.assertFalse(stale_score["passed"])
        self.assertFalse(stale_score["matched"]["corrected 425 C target"])
        self.assertIn("stale 350 C recommended as current", stale_score["violations"])

        absent = by_name["absent_hall_mobility"]
        self.assertTrue(score_response(absent, "Unknown: Hall mobility was not measured for ZNO-24-A.")["passed"])
        self.assertFalse(score_response(absent, "Hall mobility was 18.4 cm^2/V-s.")["passed"])

    def test_citations_require_scoped_factual_support(self):
        rubric = next(q for q in ZNO_WORKFLOW.queries if q.name == "explicit_lookup_current")
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "lab.sqlite3"
            store = Store(db, "scientist", "zno")
            seeded = seed_workflow_store(store)
            corrected = seeded["correction"]
            original = seeded["records"][ZNO_WORKFLOW.correction.target_event_id]
            outcome = seeded["records"]["zno-outcome-01"]
            controls = seeded["records"]["zno-controls-01"]
            answer = corrected["content"]
            current_id = corrected["memory_id"]
            non_factual = store.observe(
                "proposal-copy", answer, "assistant", corrected["timestamp"], "assistant_proposal",
            )
            other_scope = Store(db, "scientist", "other")
            foreign = other_scope.observe(
                "foreign-copy", answer, "other-scope", corrected["timestamp"], "user_report",
            )

            self.assertFalse(score_response(rubric, answer, source_ids=[current_id])["passed"])
            self.assertTrue(score_response(rubric, answer, source_ids=[current_id], evidence_store=store)["passed"])
            for ids in (
                [],
                ["mem-1"],
                [""],
                [original["memory_id"]],
                [outcome["memory_id"]],
                [controls["memory_id"]],
                [foreign["memory_id"]],
                [non_factual["memory_id"]],
                [current_id, outcome["memory_id"]],
                [current_id, "mem-1"],
            ):
                with self.subTest(ids=ids):
                    self.assertFalse(score_response(rubric, answer, source_ids=ids, evidence_store=store)["citation_ok"])

    def test_historical_citation_supports_step_four_at_its_event_time(self):
        rubric = next(q for q in ZNO_WORKFLOW.queries if q.name == "explicit_lookup_historical")
        with TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "lab.sqlite3", "scientist", "zno")
            seeded = seed_workflow_store(store)
            original = seeded["records"][ZNO_WORKFLOW.correction.target_event_id]
            wrong_step = store.observe(
                "wrong-step", "Step 3 anneal at 350 C using 10 C/min. Step 4 measure resistance.",
                "sop:alternative", original["timestamp"], "user_report",
            )
            answer = "Step 4 said to anneal at 350 C with a 10 C/min ramp."
            ids = [original["memory_id"]]
            self.assertTrue(score_response(rubric, answer, source_ids=ids, evidence_store=store)["passed"])
            for wrong_record in (seeded["correction"], seeded["records"]["zno-outcome-01"], wrong_step):
                result = score_response(rubric, answer, source_ids=[wrong_record["memory_id"]], evidence_store=store)
                self.assertFalse(result["citation_ok"])

    def test_unseen_tio2_variation_and_unrelated_control_remain_disjoint(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "lab.sqlite3"
            zno_store = Store(db, "scientist", "zno")
            tio2_store = Store(db, "scientist", "tio2")
            seed_workflow_store(zno_store, ZNO_WORKFLOW)
            seed_workflow_store(tio2_store, TIO2_UNSEEN_VARIATION)
            for ctrl in UNRELATED_CONTROL_OBSERVATIONS:
                ctrl_store = Store(db, "scientist", "unrelated")
                ctrl_store.observe(ctrl.event_id, ctrl.content, ctrl.source, ctrl.timestamp, ctrl.kind)

            self.assertEqual(zno_store.search("520.7 Raman anatase"), [])
            self.assertEqual(tio2_store.search("SRM 640f zinc"), [])
            self.assertEqual(ctrl_store.search("ZnO anneal fused silica"), [])


if __name__ == "__main__":
    unittest.main()
