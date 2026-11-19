"""Offline grading, provenance, reporting and frozen-reference regression tests."""
from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_bench.grading import (DIMENSIONS, aggregate, aggregate_records,
    apply_review_overlays, evidence_index, grade_annotation, load_review_overlays,
    parse_annotation, validate_annotation, verdict)
from trajectory_bench.reporting import build_report, render_html, write_reports


def evidence():
    return {
        "domain": "retail", "policy": "Ask for confirmation before a refund.\nRefund to the confirmed method.",
        "history": [
            {"ref": "h:0", "role": "user", "content": "Refund order A to the credit card.", "customer_visible": False},
            {"ref": "h:1", "role": "assistant", "content": "The refund request was submitted.", "customer_visible": True},
        ],
        "actual_execution": [{"ref": "e:0", "name": "return_order", "response": {"status": "submitted"}}],
        "mechanical_state_differences": [{"event_ref": "e:0", "state_ref": "s:0", "changes": []}],
        "offline_database_facts": {"orders": {"A": {"status": "return requested"}}},
    }


def annotation(state="pass"):
    scores = {dim: {"status": state, "reason": "已核对用户要求和工具结果。", "refs": ["h:0", "e:0"], "requirement_ids": ["Q1"], "issue_ids": []} for dim in DIMENSIONS}
    value = {
        "groups": [{"group_id": "G1", "goal": "办理订单退款", "object": "A", "request_refs": ["h:0"],
                    "revisions": [], "requirements": [{"id": "Q1", "content": "退款至确认方式", "basis": "Refund to the confirmed method.", "refs": ["h:0", "policy:2"]}],
                    "scores": scores, "end_state": "退款申请已提交"}],
        "issues": [], "evidence_links": [{"from": "h:0", "to": "e:0", "purpose": "对照请求与办理结果"}], "auxiliary_notes": [],
    }
    return value


def fail_annotation():
    value = annotation()
    value["groups"][0]["scores"]["CONSENT"].update(status="fail", issue_ids=["I1"])
    value["issues"] = [{"issue_id": "I1", "title": "未确认退款方式", "basis": "Ask for confirmation before a refund.",
                        "refs": ["h:0", "e:0", "policy:1"], "root_cause": "agent", "severity": "substantive", "recovered": False,
                        "effect": "使用未确认的方式提交退款申请"}]
    return value


class ValidationTests(unittest.TestCase):
    def test_valid_and_na_without_refs(self):
        value = annotation()
        value["groups"][0]["scores"]["CONSENT"] = {"status": "na", "reason": "本项不要求确认"}
        result = validate_annotation(value, evidence())
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["evidence_checks"]["semantic_source_review"], "not_automated")

    def test_fail_needs_substantive_issue_and_real_source(self):
        self.assertTrue(validate_annotation(fail_annotation(), evidence())["valid"])
        value = fail_annotation()
        value["issues"][0]["severity"] = "minor"
        self.assertIn("fail_requires_substantive_issue", " ".join(validate_annotation(value, evidence())["errors"]))
        value = fail_annotation()
        value["groups"][0]["scores"]["CONSENT"]["refs"] = []
        self.assertIn("source_evidence_required", " ".join(validate_annotation(value, evidence())["errors"]))
        value = fail_annotation()
        value["issues"][0]["refs"] = ["policy:999", "db:invented", "s:999", "h:999"]
        self.assertEqual(sum("nonexistent_ref" in item for item in validate_annotation(value, evidence())["errors"]), 4)

    def test_status_reason_dimensions_and_requirement_links(self):
        value = annotation()
        del value["groups"][0]["scores"]["FACT"]
        value["groups"][0]["scores"]["RULE"].update(status="maybe", reason="", requirement_ids=["Q9"])
        errors = " ".join(validate_annotation(value, evidence())["errors"])
        for code in ("exactly_six_dimensions", "invalid_status", "nonempty_string", "unknown_id:Q9"):
            self.assertIn(code, errors)

    def test_duplicate_group_issue_and_hidden_fields(self):
        value = fail_annotation()
        value["groups"].append(copy.deepcopy(value["groups"][0]))
        value["issues"].append(copy.deepcopy(value["issues"][0]))
        value["original_reward"] = 1
        value["groups"][0]["hidden_answers"] = {"result": "pass"}
        errors = " ".join(validate_annotation(value, evidence())["errors"])
        self.assertIn("duplicate_group_id", errors)
        self.assertIn("duplicate_issue_id", errors)
        self.assertEqual(errors.count("forbidden_answer_or_original_score"), 2)

    def test_exact_quote_and_precise_database_refs(self):
        value = annotation()
        value["evidence_quotes"] = [{"ref": "e:0", "quote": "submitted"}]
        value["groups"][0]["scores"]["OUTCOME"]["refs"] = ["s:0", "db:/orders/A/status"]
        self.assertTrue(validate_annotation(value, evidence())["valid"])
        value["evidence_quotes"][0]["quote"] = "paid in full"
        self.assertIn("quote_not_found_at_ref", " ".join(validate_annotation(value, evidence())["errors"]))
        self.assertNotIn("s:999", evidence_index(evidence()))

    def test_tool_payload_cannot_define_extra_citation_locations(self):
        packet = evidence()
        packet["actual_execution"][0]["response"]["ref"] = "h:999"
        self.assertNotIn("h:999", evidence_index(packet))
        value = fail_annotation()
        value["issues"][0]["refs"] = ["h:999"]
        self.assertFalse(validate_annotation(value, packet)["valid"])

    def test_applicable_empty_refs_warning_retains_legacy_record(self):
        value = annotation()
        value["groups"][0]["scores"]["RULE"]["refs"] = []
        result = validate_annotation(value, evidence())
        self.assertTrue(result["valid"])
        self.assertIn("empty_applicable_evidence", " ".join(result["warnings"]))
        del value["groups"][0]["scores"]["RULE"]["refs"]
        self.assertFalse(validate_annotation(value, evidence())["valid"])

    def test_strict_json_raw_failures_are_preserved(self):
        raw = '```json\n{"groups": []}\n```'
        result = parse_annotation(raw, evidence())
        record = grade_annotation("c1", result, evidence())
        self.assertEqual(record["raw_text"], raw)
        self.assertEqual(record["automatic"]["status"], "technical_unscored")
        self.assertIsNone(record["automatic"]["verdict"])
        for raw in ('{"groups":[],"groups":[]}', '{"groups":[],"x":NaN}'):
            self.assertTrue(parse_annotation(raw)["validation_errors"])

    def test_business_unknown_is_scored_technical_is_separate(self):
        unknown = grade_annotation("c1", annotation("unknown"), evidence())
        technical = grade_annotation("c2", None, evidence())
        summary = aggregate_records([unknown, technical])
        self.assertEqual(summary["trajectories"]["unknown"], 1)
        self.assertEqual(summary["technical_unscored"], 1)
        self.assertEqual(summary["trajectories"]["n"], 1)
        self.assertEqual(summary["trajectories"]["valid_denominator"], 0)
        self.assertIsNone(summary["trajectories"]["pass_rate_decided"])

    def test_capture_only_is_not_technical_failure_or_business_unknown(self):
        record = {"case_id": "capture", "status": "captured_unscored", "metadata": {}}
        summary = aggregate_records([record])
        self.assertEqual(summary["captured_unscored"], 1)
        self.assertEqual(summary["technical_unscored"], 0)
        self.assertEqual(summary["unreviewed"], 0)
        self.assertEqual(summary["trajectories"]["n"], 0)


class AggregationTests(unittest.TestCase):
    def test_hierarchy_is_not_equal_averaging(self):
        a = annotation()
        a["case_id"] = "a"
        second = copy.deepcopy(a["groups"][0])
        second["group_id"] = "G2"
        second["scores"]["REPORT"]["status"] = "unknown"
        a["groups"].append(second)
        b = annotation()
        b["case_id"] = "b"
        b["groups"][0]["scores"]["RULE"]["status"] = "fail"
        result = aggregate([a, b])
        self.assertEqual(result["trajectories"]["unknown"], 1)
        self.assertEqual(result["trajectories"]["fail"], 1)
        self.assertEqual(result["business_groups"]["n"], 3)
        self.assertEqual(result["dimensions"]["RULE"]["n"], 2)
        self.assertEqual(result["dimensions"]["RULE"]["pass_rate_decided"], .5)
        self.assertEqual(verdict(["na", "na"]), "na")
        self.assertEqual(verdict(["unknown", "fail"]), "fail")
        for states in ([], ["corrupt"]):
            with self.assertRaises(ValueError):
                verdict(states)

    def test_empty_groups_duplicates_and_malformed_dimensions_rejected(self):
        value = {"case_id": "a", **annotation()}
        for rows in ([value, value], [{"case_id": "a", "groups": []}]):
            with self.assertRaises(ValueError):
                aggregate(rows)
        del value["groups"][0]["scores"]["FACT"]
        with self.assertRaises(ValueError):
            aggregate([value])

    def test_frozen_reference_540_reproduces_all_counts(self):
        workspace = Path(__file__).resolve().parents[1]
        materials = workspace / "reference_data"
        if not materials.exists():
            self.skipTest("Read-only frozen reference materials are outside the distributed package")
        source = materials / "reference" / "annotations.jsonl"
        import hashlib
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        expected = json.loads((materials / "statistics" / "reproduced_summary.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(aggregate(rows), expected)
        self.assertEqual(expected["trajectories"], {"n": 540, "pass": 291, "fail": 220, "unknown": 29, "na": 0, "pass_rate_decided": 291 / 511})
        self.assertEqual(expected["business_groups"]["n"], 874)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


class ReviewAndReportTests(unittest.TestCase):
    def test_provenance_overlay_does_not_rewrite_automatic(self):
        original = grade_annotation("a", fail_annotation(), evidence(), {"domain": "retail", "original_reward": 1})
        untouched = copy.deepcopy(original)
        overlay = {"case_id": "a", "reviewer": "reviewer-1", "reviewed_at": "2026-11-13T09:00:00+08:00",
                   "review_type": "assistant", "source_reviewed": True, "reason": "对照完整原记录后更正", "annotation": annotation()}
        result = apply_review_overlays([original], [overlay], {"a": evidence()})
        self.assertEqual(original, untouched)
        self.assertEqual(result[0]["automatic"], original["automatic"])
        self.assertEqual(result[0]["reviewed"]["verdict"], "pass")
        self.assertEqual(result[0]["raw_annotation"], original["raw_annotation"])
        self.assertEqual(result[0]["review_history"][0]["overlay"], overlay)
        self.assertEqual(result[0]["reviewed"]["provenance"]["review_type"], "assistant")
        with self.assertRaises(ValueError):
            apply_review_overlays(result, [overlay])

    def test_patch_and_partial_review_coverage(self):
        records = [grade_annotation("a", annotation(), evidence()), grade_annotation("b", annotation(), evidence())]
        overlay = {"case_id": "a", "reviewer": "r", "reviewed_at": "2026-11-13", "reason": "结果依据缺失", "review_type": "human",
                   "changes": [{"group_id": "G1", "dimension": "OUTCOME", "status": "unknown", "reason": "结果字段缺失", "refs": ["e:0"]}]}
        result = apply_review_overlays(records, [overlay], {"a": evidence()})
        summary = aggregate_records(result, "reviewed")
        self.assertEqual(summary["scored_cases"], 1)
        self.assertEqual(summary["unreviewed"], 1)
        self.assertEqual(summary["trajectories"]["unknown"], 1)
        self.assertEqual(summary["trajectories"]["pass"], 0)
        overlay["reviewer"] = ""
        with self.assertRaises(ValueError):
            apply_review_overlays(records, [overlay])

    def test_overlay_invalid_refs_cannot_be_marked_reviewed(self):
        record = grade_annotation("a", annotation(), evidence())
        overlay = {"case_id": "a", "reviewer": "r", "reviewed_at": "2026-11-13", "reason": "复核", "review_type": "human",
                   "changes": [{"group_id": "G1", "dimension": "RULE", "refs": ["h:999"]}]}
        with self.assertRaisesRegex(ValueError, "nonexistent_ref"):
            apply_review_overlays([record], [overlay], {"a": evidence()})

    def test_report_artifacts_and_escaped_raw_failures(self):
        value = annotation()
        value["groups"][0]["goal"] = '=HYPERLINK("https://bad")'
        value["groups"][0]["scores"]["RULE"]["reason"] = "<script>alert(1)</script>"
        records = [grade_annotation("a", value, evidence(), {"domain": "retail", "original_reward": 1}),
                   grade_annotation("b", "not json <script>bad</script>", evidence(), {"original_reward": 1})]
        records[0]["artifacts"] = {"bad": "javascript:alert(1)", "evidence": "cases/a/evidence.json"}
        with tempfile.TemporaryDirectory() as temp:
            paths = write_reports(records, temp, include_markdown=True)
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(report["layers"]["automatic"]["technical_unscored"], 1)
            self.assertEqual(report["layers"]["reviewed"]["unreviewed"], 2)
            page = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertIn("业务规则与对象", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>", page)
            self.assertNotIn("javascript:alert", page)
            self.assertIn("cases/a/evidence.json", page)
            self.assertEqual(report["records"][1]["raw_text"], records[1]["raw_text"])
            with Path(paths["groups_csv"]).open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertTrue(row["goal"].startswith("'="))
            self.assertEqual(json.loads(Path(paths["records"]).read_text(encoding="utf-8").splitlines()[1])["raw_text"], records[1]["raw_text"])

    def test_review_jsonl_load_has_line_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.jsonl"
            path.write_text('{"case_id":"a"}\nwrong\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_review_overlays(path)


if __name__ == "__main__":
    unittest.main()
