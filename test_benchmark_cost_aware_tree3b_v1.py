"""CPU protocol checks; these never claim model/GPU correctness."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import benchmark_cost_aware_tree3b_v1 as bench
import profile_cost_aware_tree3b_v1 as profile


def run(tokens=(2, 3, 151645), seconds=1., native=True):
    return {"token_ids": list(tokens), "external_wall_s": seconds,
            "finished_eos": native, "generated_tokens": len(tokens),
            "internal_metrics": {"cycles": 2, "decode_s": seconds/2}}


class ProtocolTests(unittest.TestCase):
    def test_fixed_suite_splits_and_caps(self):
        suite, dev = bench.load_cases("optimization_20261106/suite.json", "development")
        _, held = bench.load_cases("optimization_20261106/suite.json", "heldout")
        self.assertEqual((len(dev), len(held)), (6, 6))
        self.assertFalse({row["id"] for row in dev} & {row["id"] for row in held})
        self.assertTrue(all(row["max_new_tokens"] == 1024 for row in dev+held))
        with self.assertRaises(RuntimeError):
            bench.load_cases("optimization_20261106/suite.json", "development", [held[0]["id"]])

    def test_suite_mutation_rejected_before_any_runtime_import(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/"suite.json"
            suite = json.loads(Path("optimization_20261106/suite.json").read_text())
            suite["cases"][0]["max_new_tokens"] = 128
            target.write_text(json.dumps(suite))
            with self.assertRaises(RuntimeError):
                bench.load_cases(target)

    def test_exact_alternating_order(self):
        names = ["baseline", "current", "candidate"]
        self.assertEqual(bench.alternating_order(names, 0), names)
        self.assertEqual(bench.alternating_order(names, 1), names[::-1])
        self.assertEqual(bench.alternating_order(names, 2), names)
        with self.assertRaises(RuntimeError):
            bench.alternating_order(["current", "current"], 0)

    def test_first_eos_and_unchanged_budget_policy(self):
        eos = {151645}
        self.assertTrue(bench.stop_valid([1, 151645], "eos", eos, 1024))
        self.assertTrue(bench.stop_valid([1, 2], "max_new_tokens", eos, 2))
        for values, reason in (([], "eos"), ([1, 151645, 2], "eos"),
                              ([1, 2], "max_new_tokens"), ([True, 151645], "eos")):
            self.assertFalse(bench.stop_valid(values, reason, eos, 1024))

    def test_numerical_differences_are_reported_never_mislabeled_exact(self):
        equal = bench.output_comparison(run(seconds=10), run())
        self.assertTrue(equal["exact_token_match"])
        self.assertEqual(equal["same_output_latency_speedup"], 10.)
        unequal = bench.output_comparison(run(seconds=10), run((2, 4, 151645)))
        self.assertFalse(unequal["exact_token_match"])
        self.assertEqual(unequal["matching_prefix_tokens"], 1)
        self.assertIsNone(unequal["same_output_latency_speedup"])
        self.assertFalse(unequal["quality_preservation_inferred_from_target_verification"])

    def test_streaming_real_consumer_and_cumulative_output(self):
        class Tokenizer:
            def decode(self, values, skip_special_tokens):
                return "é"+":".join(map(str, values))
        sink = bench.StreamingSink(Tokenizer())
        began = 0.
        sink.emit([1])
        sink.emit([1, 2])
        sink.close()
        self.assertEqual(sink.last_tokens, [1, 2])
        self.assertEqual(sink.metrics(began)["events"], 2)
        self.assertGreater(sink.serialized_bytes, 0)
        self.assertFalse(sink.metrics(began)["HTTP_transport_and_browser_rendering_included"])

    def test_consumer_error_propagates(self):
        class Tokenizer:
            def decode(self, *args, **kwargs):
                raise ValueError("decode fixture")
        sink = bench.StreamingSink(Tokenizer())
        sink.emit([1])
        with self.assertRaisesRegex(ValueError, "decode fixture"):
            sink.close()

    def test_n64_cache_parity_gate_rejects_uncommitted_state_mismatch(self):
        cache = {"position": 8, "anchor": 151645, "layers": [{"keys": "a", "values": "b"}], "features": "c"}
        good = bench.validate_control_parity(run(), run(), cache, deepcopy(cache))
        self.assertTrue(good["passed"])
        bad = deepcopy(cache)
        bad["layers"][0]["keys"] = "changed"
        self.assertFalse(bench.validate_control_parity(run(), run(), cache, bad)["passed"])
        self.assertFalse(bench.validate_control_parity(run(), run((8, 151645)), cache, cache)["passed"])

    def test_summary_never_mixes_actual_shapes_or_modes(self):
        rows = []
        for streaming in (False, True):
            for packed, seconds in ((16, 2.), (64, 1.)):
                rows.append({"streaming": streaming, "packed_budget": packed,
                    "runs": {"baseline": run(seconds=10), "current": run(seconds=1.2),
                             "candidate": run(seconds=seconds)}})
        summary = bench.summarize(rows, ["baseline", "current", "candidate"], completed=True, integrity_passed=True)
        self.assertEqual(len(summary["modes"]), 4)
        self.assertEqual(summary["modes"]["non_streaming:N16"]["comparisons"]["baseline_over_candidate"]["same_output_speedup_median"], 5.)
        self.assertEqual(summary["modes"]["non_streaming:N64"]["comparisons"]["baseline_over_candidate"]["same_output_speedup_median"], 10.)
        self.assertFalse(summary["general_quality_preservation_certified"])

    def test_failure_and_missing_eos_invalidate_same_output_speedup(self):
        rows = [{"streaming": False, "runs": {"baseline": run(seconds=10), "candidate": run(native=False)}}]
        for complete, integrity in ((True, True), (False, True), (True, False)):
            summary = bench.summarize(rows, ["baseline", "candidate"], completed=complete, integrity_passed=integrity)
            group = summary["modes"]["non_streaming"]["comparisons"]["baseline_over_candidate"]
            self.assertFalse(group["same_output_speedup_eligible"])
            self.assertIsNone(group["same_output_speedup_median"])

    def test_frozen_manifest_detects_runtime_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"original.py").write_text("original")
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"target_sha256": profile.TARGET_SHA,
                "files": {"original.py": profile.sha(root/"original.py")}}))
            original_root = profile.ROOT
            try:
                profile.ROOT = root
                self.assertTrue(profile.verify_frozen_manifest(manifest)["all_frozen_control_sources_unchanged"])
                (root/"original.py").write_text("modified")
                with self.assertRaises(RuntimeError):
                    profile.verify_frozen_manifest(manifest)
            finally:
                profile.ROOT = original_root

    def test_coordination_keeps_nested_runtime_and_operator_times_separate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"trace.json"
            path.write_text(json.dumps({"traceEvents": [
                {"ph": "X", "name": "aten::item", "dur": 12},
                {"ph": "X", "name": "cudaStreamSynchronize", "dur": 10},
                {"ph": "X", "name": "cudaStreamSynchronize", "dur": 9},
            ]}))
            report = profile.trace_coordination(path)
            self.assertEqual(report["operations"]["cudaStreamSynchronize"]["calls"], 2)
            self.assertTrue(report["nested_ATen_and_runtime_totals_overlap_do_not_add"])


if __name__ == "__main__":
    unittest.main()
