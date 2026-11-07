"""CPU-only quality checks for the fixed cost-aware-tree experiment.

Answer keys live ONLY here. The inference suite is input-only and importing
this module is never necessary in a model generation process. These finite
checks supplement token identity and native-EOS reports; they do not certify
arbitrary-prompt quality or unchanged reasoning.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


EXPECTED_JSON = {
    "dev-math-inventory": {"remaining": 90, "revenue": 780},
    "dev-json-net": {"items": [{"id": "a", "net": 7}, {"id": "b", "net": 8},
                              {"id": "c", "net": 15}], "total": 30},
    "dev-logic-order": {"order": ["Nova", "Maple", "Cedar", "Birch"]},
    "heldout-math-tickets": {"adults": 31, "children": 31},
    "heldout-json-stock": {"stock": [{"item": "orange", "available": 27},
                                     {"item": "apple", "available": 22},
                                     {"item": "pear", "available": 20}],
                           "total_available": 69},
    "heldout-logic-arrivals": {"order": ["Kestrel", "Iris", "Juniper", "Larch"]},
}
EXPECTED_INTEGER = {"dev-math-machines": 729, "heldout-math-water": 360}
FUNCTION_NAMES = {"dev-code-prefix": "prefix_sums", "dev-code-unique": "first_unique",
                  "heldout-code-dedupe": "stable_dedupe",
                  "heldout-code-intervals": "merge_intervals"}


def _same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_json(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _last_json_object(text):
    decoder = json.JSONDecoder()
    found = None
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found = value, start, end
        index = end
    return found


def _schema_matches(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_schema_matches(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_schema_matches(a, b) for a, b in zip(actual, expected))
    return True


def _function_tests(case_id):
    if case_id == "dev-code-prefix":
        inputs = [[], [0], [3, -1, 4], [-5, 5, 0, 7], list(range(40)), [1, -1] * 25]
        expected = []
        for values in inputs:
            total = 0
            row = []
            for value in values:
                total += value
                row.append(total)
            expected.append(row)
    elif case_id == "dev-code-unique":
        inputs = [[], [4], [2, 3, 2, 4, 3], [5, 5, 7, 7], [0, -1, 0, 2],
                  [1, 2, 3, 1, 2, 3, 4], list(range(40)) + list(range(39))]
        expected = [next((v for v in values if values.count(v) == 1), None) for values in inputs]
    elif case_id == "heldout-code-dedupe":
        inputs = [[], ["a"], ["pear", "apple", "pear", "plum", "apple"],
                  ["A", "a", "A", "a"], ["", "", "x"], ["λ", "猫", "λ", "é"],
                  [str(i % 11) for i in range(60)]]
        expected = [list(dict.fromkeys(values)) for values in inputs]
    elif case_id == "heldout-code-intervals":
        inputs = [[], [[1, 2]], [[1, 3], [3, 5]], [[8, 10], [1, 3], [2, 6], [15, 18]],
                  [[1, 10], [2, 4], [6, 8]], [[-5, -3], [-3, 0], [2, 2]],
                  [[5, 5], [1, 1], [3, 3]], [[4, 7], [1, 4], [7, 9], [0, 0]]]
        expected = []
        for intervals in inputs:
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            expected.append(merged)
    else:
        raise KeyError(case_id)
    return [{"input": value, "expected": answer} for value, answer in zip(inputs, expected)]


# The generated functions run in a separate bounded CPU process. Only function
# definitions are retained; module examples/imports are never executed. Dunder
# access and attributes outside the small list/dict/string-method allowlist are
# rejected before execution. This is not presented as a general Python sandbox.
_FUNCTION_WORKER = r'''
import __future__, ast, copy, json, resource, sys
payload = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
if sys.platform.startswith("linux"):
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
tree = ast.parse(payload["code"])
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
allowed_attributes = {"append", "extend", "pop", "sort", "copy", "count", "index", "get", "items", "values", "keys", "setdefault", "add", "update", "discard", "fromkeys"}
for function in functions:
    if function.decorator_list:
        raise ValueError("Decorators are not admitted")
    for node in ast.walk(function):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.ClassDef)):
            raise ValueError("Only ordinary pure functions are admitted")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ValueError("Dunder names are not admitted")
        if isinstance(node, ast.Attribute) and node.attr not in allowed_attributes:
            raise ValueError("Attribute outside the bounded method allowlist: " + node.attr)
safe = {"len":len,"range":range,"enumerate":enumerate,"sum":sum,"sorted":sorted,"list":list,"dict":dict,"set":set,"tuple":tuple,"zip":zip,"min":min,"max":max,"abs":abs,"all":all,"any":any,"bool":bool,"int":int,"str":str,"reversed":reversed,"next":next,"iter":iter}
namespace = {"__builtins__": safe}
module = ast.Module(body=functions, type_ignores=[])
exec(compile(module, "<bounded-generated-functions>", "exec", flags=__future__.annotations.compiler_flag), namespace)
function = namespace[payload["name"]]
results = []
for row in payload["tests"]:
    value = copy.deepcopy(row["input"])
    before = copy.deepcopy(value)
    result = function(value)
    results.append({"behavior_pass": result == row["expected"], "input_unchanged": value == before})
print(json.dumps({"checks":results}))
'''


def _grade_code(case_id, text):
    name = FUNCTION_NAMES[case_id]
    fenced = list(re.finditer(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.I | re.S))
    if not fenced:
        return {"semantic_pass": False, "strict_format_pass": False,
                "details": {"reason": "Expected one complete Python code block", "function": name}}
    code = fenced[0].group(1)
    tests = _function_tests(case_id)
    try:
        ast.parse(code)
        process = subprocess.run([sys.executable, "-I", "-c", _FUNCTION_WORKER],
                                 input=json.dumps({"code": code, "name": name, "tests": tests}),
                                 text=True, capture_output=True, timeout=3)
        if process.returncode:
            raise ValueError(process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "Bounded function process failed")
        checks = json.loads(process.stdout)["checks"]
        passed = len(checks) == len(tests) and all(row["behavior_pass"] and row["input_unchanged"] for row in checks)
        return {"semantic_pass": passed, "strict_format_pass": len(fenced) == 1,
                "details": {"function": name, "checks": checks, "tests": len(tests),
                            "generated_module_examples_executed": False}}
    except (ValueError, SyntaxError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        return {"semantic_pass": False, "strict_format_pass": len(fenced) == 1,
                "details": {"function": name, "reason": str(error)[:400], "tests": len(tests)}}


def grade_case(case_id, text):
    """Return separate finite semantic and requested-format checks."""
    if not isinstance(text, str):
        raise TypeError("Grade decoded assistant text")
    if case_id in FUNCTION_NAMES:
        return _grade_code(case_id, text)
    if case_id in EXPECTED_INTEGER:
        match = re.search(r"FINAL:\s*([+-]?\d+)\s*\.?\s*$", text, flags=re.I)
        exact = re.search(r"FINAL:\s*([+-]?\d+)\s*$", text)
        return {"semantic_pass": bool(match and int(match.group(1)) == EXPECTED_INTEGER[case_id]),
                "strict_format_pass": exact is not None,
                "details": {"observed_final": int(match.group(1)) if match else None}}
    if case_id not in EXPECTED_JSON:
        raise KeyError("Unknown fixed quality case: " + case_id)
    found = _last_json_object(text)
    if found is None:
        return {"semantic_pass": False, "strict_format_pass": False,
                "details": {"reason": "No complete JSON object"}}
    actual, start, end = found
    expected = EXPECTED_JSON[case_id]
    trailing_clean = not text[end:].strip()
    json_only = case_id in ("dev-json-net", "heldout-json-stock")
    format_ok = _schema_matches(actual, expected) and trailing_clean and (not json_only or not text[:start].strip())
    return {"semantic_pass": _same_json(actual, expected), "strict_format_pass": format_ok,
            "details": {"observed": actual, "trailing_text": bool(text[end:].strip()), "JSON_only_required": json_only}}


def grade_resultset(suite, records):
    """Grade records with id/case_id and text; do not pass this into inference."""
    cases = {row["id"]: row for row in suite["cases"]}
    rows = []
    for record in records:
        case_id = record.get("case_id", record.get("id"))
        if case_id not in cases:
            raise KeyError("Result does not belong to fixed suite: " + str(case_id))
        row = {"id": case_id, "split": cases[case_id]["split"], **grade_case(case_id, record["text"])}
        for key in ("engine", "repeat", "streaming", "stop_reason", "finished_eos"):
            if key in record:
                row[key] = record[key]
        rows.append(row)
    return {"schema": "lightreason-cost-aware-finite-quality-v1", "rows": rows,
            "summary": {"records": len(rows), "semantic_passes": sum(row["semantic_pass"] for row in rows),
                        "strict_format_passes": sum(row["strict_format_pass"] for row in rows)},
            "arbitrary_prompt_quality_certified": False,
            "gold_or_reference_outputs_used_for_inference": False}


def self_test():
    for key, expected in EXPECTED_JSON.items():
        text = json.dumps(expected)
        assert grade_case(key, text)["semantic_pass"] and grade_case(key, text)["strict_format_pass"]
        wrong = {**expected, "unexpected": 1}
        assert not grade_case(key, json.dumps(wrong))["semantic_pass"]
    assert grade_case("dev-math-machines", "Calculation.\nFINAL: 729")["semantic_pass"]
    assert not grade_case("dev-math-machines", "FINAL: 728")["semantic_pass"]
    assert grade_case("heldout-math-water", "FINAL: 360.")["semantic_pass"]
    assert not grade_case("heldout-math-water", "FINAL: 360.")["strict_format_pass"]
    assert grade_case("dev-json-net", "```json\n" + json.dumps(EXPECTED_JSON["dev-json-net"]) + "\n```")["semantic_pass"]
    assert not grade_case("dev-json-net", "```json\n" + json.dumps(EXPECTED_JSON["dev-json-net"]) + "\n```")["strict_format_pass"]
    code = {
        "dev-code-prefix": "def prefix_sums(values):\n out=[]\n total=0\n for value in values:\n  total+=value\n  out.append(total)\n return out\n",
        "dev-code-unique": "def first_unique(values):\n return next((v for v in values if values.count(v)==1),None)\n",
        "heldout-code-dedupe": "def stable_dedupe(words):\n return list(dict.fromkeys(words))\n",
        "heldout-code-intervals": "def merge_intervals(intervals):\n out=[]\n for a,b in sorted(intervals):\n  if out and a<=out[-1][1]:\n   out[-1][1]=max(out[-1][1],b)\n  else:\n   out.append([a,b])\n return out\n",
    }
    for key, source in code.items():
        result = grade_case(key, "```python\n" + source + "```\nBrief explanation.")
        assert result["semantic_pass"], (key, result)
    assert not grade_case("dev-code-prefix", "```python\ndef prefix_sums(values):\n return values\n```")["semantic_pass"]
    assert not grade_case("dev-code-prefix", "```python\ndef prefix_sums(values):\n while True: pass\n```")["semantic_pass"]
    return {"CPU_quality_contract_passed": True, "GPU_work_performed": False,
            "heldout_model_outputs_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--suite", default="optimization_20261106/suite.json")
    parser.add_argument("--results")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if not args.results or not args.out:
            parser.error("Use --self-test, or --results and --out for local grading")
        suite_path = Path(args.suite)
        suite = json.loads(suite_path.read_text())
        data = json.loads(Path(args.results).read_text())
        records = data if isinstance(data, list) else data["records"]
        output = grade_resultset(suite, records)
        output["suite_sha256"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
        Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
