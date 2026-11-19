import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trajectory_bench.evidence import EvidenceError, compile_evidence, load_evidence


def episode():
    """Small captured episode, including an unsent text/tool-call prefix."""
    return {
        "schema_version": "trajectory_bench.native_episode.v1",
        "domain": "example",
        "domain_policy": "# Policy\nObtain confirmation before modifying a record.\nKeep exact receipt evidence.",
        "tools": [{"type": "function", "function": {
            "name": "change", "description": "Change a record.",
            "parameters": {"type": "object", "properties": {"record": {"type": "string"}}}}}],
        "task": {"evaluation_criteria": {"answer": "PRIVATE_EXPECTED_ANSWER"}},
        "model_alias": "PRIVATE_MODEL", "original_score": 1, "reward": 1,
        "source_labels": ["PRIVATE_LABEL"], "user_system_prompt": "PRIVATE_USER_PLAN",
        "messages": [
            {"role": "system", "content": "PRIVATE_SYSTEM_TASK"},
            {"role": "user", "content": "Change record Z; yes, please proceed."},
            {"role": "assistant", "content": "The change is already complete.",
             "tool_calls": [{"id": "c1", "name": "change", "arguments": {"record": "Z"}}]},
            {"role": "tool", "id": "c1", "requestor": "assistant",
             "content": '{"success":true}', "error": False},
            {"role": "assistant", "content": "The tool returned success.", "customer_visible": True},
        ],
        "tool_events": [{"event_index": 0, "name": "change", "requestor": "assistant",
                         "tool_call_id": "c1", "arguments": {"record": "Z"},
                         "response": {"success": True}, "state_before": "old_hash", "state_after": "new_hash",
                         "state_changes": [{"path": ["records", "Z", "status"], "before": "old", "after": "new",
                                            "before_exists": True, "after_exists": True}]}],
        "initial_state": {"agent": {"records": {"Z": {"status": "old"}}}},
        "final_state": {"agent": {"records": {"Z": {"status": "new"}}}},
    }


class EvidenceTests(unittest.TestCase):
    def test_raw_packet_keeps_policy_tools_order_and_drops_private_metadata(self):
        raw = episode()
        untouched = copy.deepcopy(raw)
        packet = compile_evidence(raw)
        self.assertEqual(raw, untouched)
        self.assertEqual(packet["policy"], raw["domain_policy"])
        self.assertEqual(packet["available_tool_schemas"], raw["tools"])
        self.assertEqual([h["ref"] for h in packet["history"]], ["h:1", "h:2", "h:3", "h:4"])
        serialized = json.dumps(packet)
        for secret in ("PRIVATE_EXPECTED_ANSWER", "PRIVATE_MODEL", "PRIVATE_LABEL", "PRIVATE_USER_PLAN", "PRIVATE_SYSTEM_TASK"):
            self.assertNotIn(secret, serialized)
        for key in ("model_alias", "original_score", "reward", "task", "source_labels"):
            self.assertNotIn(key, packet)

    def test_assistant_tool_prefix_is_preserved_but_never_delivered(self):
        raw = episode()
        raw["messages"][2]["customer_visible"] = True
        packet = compile_evidence(raw)
        prefix = packet["history"][1]
        self.assertEqual(prefix["content"], "The change is already complete.")
        self.assertFalse(prefix["customer_visible"])
        self.assertTrue(packet["history"][-1]["customer_visible"])

    def test_id_actor_arguments_and_receipt_link_verified(self):
        packet = compile_evidence(episode())
        event = packet["actual_execution"][0]
        self.assertEqual(event["call_ref"], "h:2")
        self.assertEqual(event["call_index"], 0)
        self.assertEqual(event["receipt_ref"], "h:3")
        self.assertTrue(event["receipt_exact"])
        self.assertEqual(event["pairing"], "tool_call_id")
        self.assertTrue(event["backend_state_changed"])
        self.assertFalse(event["returned_error"])
        self.assertEqual(event["execution_status"], "returned")
        self.assertTrue(packet["capture_integrity"]["verified"])
        self.assertEqual(packet["mechanical_state_differences"][0]["changes"][0]["after"], "new")

    def test_event_id_mismatch_does_not_pair_by_similar_arguments(self):
        raw = episode()
        raw["tool_events"][0]["tool_call_id"] = "different"
        packet = compile_evidence(raw)
        self.assertIsNone(packet["actual_execution"][0]["call_ref"])
        self.assertEqual(packet["capture_integrity"]["unmatched_structured_calls"], 1)
        self.assertFalse(packet["capture_integrity"]["verified"])

    def test_user_tools_are_not_agent_calls_or_agent_observations(self):
        raw = episode()
        raw["messages"][2]["role"] = "user"
        raw["messages"][3]["requestor"] = "user"
        raw["tool_events"][0]["requestor"] = "user"
        packet = compile_evidence(raw)
        self.assertEqual(packet["actual_execution"][0]["requestor"], "user")
        self.assertFalse(packet["history"][1]["agent_observed"])
        self.assertFalse(packet["history"][2]["agent_observed"])

    def test_explicit_unobserved_receipt_is_not_overridden(self):
        raw = episode()
        raw["messages"][3]["agent_observed"] = False
        packet = compile_evidence(raw)
        self.assertFalse(packet["history"][2]["agent_observed"])

    def test_missing_receipt_actor_inferred_from_matching_call_only(self):
        raw = episode()
        del raw["messages"][3]["requestor"]
        packet = compile_evidence(raw)
        self.assertEqual(packet["history"][2]["requestor"], "assistant")
        self.assertTrue(packet["history"][2]["agent_observed"])

    def test_rejected_attempt_not_invented_as_successful_event(self):
        raw = episode()
        raw["tool_events"] = []
        raw["messages"][3].update(content="Error: confirmation required", error=True, execution_status="error")
        packet = compile_evidence(raw)
        self.assertEqual(packet["actual_execution"], [])
        details = packet["capture_integrity"]["unmatched_call_details"]
        self.assertEqual(len(details), 1)
        self.assertTrue(details[0]["returned_error"])
        self.assertFalse(packet["capture_integrity"]["verified"])

    def test_returned_error_event_is_retained(self):
        raw = episode()
        raw["messages"][3].update(content="Error: invalid record", error=True)
        raw["tool_events"][0].update(response="Error: invalid record", error_type="ValueError")
        packet = compile_evidence(raw)
        event = packet["actual_execution"][0]
        self.assertTrue(event["returned_error"])
        self.assertEqual(event["execution_status"], "error")
        self.assertTrue(event["receipt_exact"])

    def test_legacy_repeated_call_skips_rejection_missing_from_event_log(self):
        raw = episode()
        retry_call = copy.deepcopy(raw["messages"][2])
        retry_call["tool_calls"][0]["id"] = "c2"
        retry_receipt = copy.deepcopy(raw["messages"][3])
        retry_receipt["id"] = "c2"
        raw["messages"][3].update(content="Error: retry", error=True)
        raw["messages"][4:4] = [retry_call, retry_receipt]
        del raw["tool_events"][0]["tool_call_id"]
        packet = compile_evidence(raw)
        self.assertEqual(packet["actual_execution"][0]["call_ref"], "h:4")
        self.assertEqual(packet["actual_execution"][0]["receipt_ref"], "h:5")
        self.assertEqual(packet["capture_integrity"]["unmatched_call_details"][0]["tool_call_id"], "c1")

    def test_absent_states_remain_unknown_and_have_integrity_note(self):
        raw = episode()
        raw.pop("initial_state")
        raw.pop("final_state")
        for key in ("state_before", "state_after", "state_changes"):
            raw["tool_events"][0].pop(key)
        packet = compile_evidence(raw)
        self.assertIsNone(packet["actual_execution"][0]["backend_state_changed"])
        self.assertEqual(packet["mechanical_state_differences"], [])
        self.assertTrue(any("snapshots" in n for n in packet["capture_integrity"]["notes"]))

    def test_openai_function_call_arguments_are_parsed_without_changing_receipt(self):
        raw = episode()
        raw["messages"][2]["tool_calls"] = [{"id": "c1", "type": "function", "function": {
            "name": "change", "arguments": '{"record":"Z"}'}}]
        packet = compile_evidence(raw)
        self.assertEqual(packet["history"][1]["tool_calls"][0]["arguments"], {"record": "Z"})
        self.assertEqual(packet["history"][2]["content"], '{"success":true}')

    def test_normalized_packet_roundtrip_preserves_main_evidence(self):
        first = compile_evidence(episode())
        imported = compile_evidence(first)
        for key in ("domain", "policy", "available_tool_schemas", "history", "actual_execution", "offline_database_facts", "mechanical_state_differences"):
            self.assertEqual(imported[key], first[key], key)
        self.assertEqual(imported["evidence_provenance"]["conversion"], "normalized_import")

    def test_normalized_forged_refs_rejected(self):
        for location in ("history", "event", "receipt", "snapshot"):
            with self.subTest(location=location):
                raw = compile_evidence(episode())
                if location == "history":
                    raw["history"][0]["ref"] = "h:999"
                elif location == "event":
                    raw["actual_execution"][0]["call_ref"] = "h:999"
                elif location == "receipt":
                    raw["actual_execution"][0]["receipt_ref"] = "h:999"
                else:
                    raw["offline_database_facts"]["recorded_state_snapshots"] = [{"ref": "s:99", "actual_event_ref": "e:99"}]
                with self.assertRaises(EvidenceError):
                    compile_evidence(raw)

    def test_exact_file_hash_and_file_object_disagreement(self):
        raw = episode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            packet = load_evidence(path, expected_sha256=expected)
            self.assertTrue(packet["evidence_provenance"]["expected_hash_verified"])
            self.assertNotIn(str(path), json.dumps(packet))
            with self.assertRaises(EvidenceError):
                load_evidence(path, expected_sha256="0" * 64)
            changed = copy.deepcopy(raw)
            changed["domain"] = "another"
            with self.assertRaises(EvidenceError):
                compile_evidence(changed, source_path=path)

    def test_missing_policy_and_conflicting_message_copies_fail(self):
        raw = episode()
        del raw["domain_policy"]
        with self.assertRaises(EvidenceError):
            compile_evidence(raw)
        raw = episode()
        raw["simulation"] = {"messages": []}
        with self.assertRaises(EvidenceError):
            compile_evidence(raw)

    def test_db_replay_claim_is_not_new_verification(self):
        raw = compile_evidence(episode())
        raw["offline_database_facts"]["replayed_actual_state"] = {
            "all_recorded_hashes_match": True, "post_operation_db_objects": [{
                "ref": "s:0", "actual_event_ref": "e:0", "before_object": {"amount": "1.20"},
                "after_object": {"amount": "1.25"}, "private_answer": "DO_NOT_COPY"}],
            "private_answer": "DO_NOT_COPY"}
        packet = compile_evidence(raw)
        self.assertNotIn("DO_NOT_COPY", json.dumps(packet))
        self.assertTrue(any("source claims" in n for n in packet["capture_integrity"]["notes"]))

    def test_whole_db_snapshots_omitted_unless_explicitly_requested(self):
        raw = episode()
        raw["initial_state"] = {"agent": {"available": True, "hash": "initial-hash",
                                           "snapshot": {"unqueried_record": "UNQUERIED_DATA"}}}
        raw["final_state"] = copy.deepcopy(raw["initial_state"])
        packet = compile_evidence(raw)
        self.assertNotIn("UNQUERIED_DATA", json.dumps(packet))
        self.assertTrue(packet["offline_database_facts"]["initial_state"]["agent"]["snapshot_omitted"])
        self.assertEqual(packet["offline_database_facts"]["initial_state"]["agent"]["hash"], "initial-hash")
        forensic = compile_evidence(raw, include_state_snapshots=True)
        self.assertIn("UNQUERIED_DATA", json.dumps(forensic))

    def test_compact_opaque_state_is_idempotent_on_import(self):
        raw = episode()
        raw["tool_events"][0].update(state_before={"known": "before", "unqueried": "UNQUERIED"},
                                      state_after={"known": "after", "unqueried": "UNQUERIED"})
        del raw["tool_events"][0]["state_changes"]
        packet = compile_evidence(raw)
        self.assertNotIn("UNQUERIED", json.dumps(packet))
        self.assertEqual(packet["mechanical_state_differences"][0]["changes"],
                         [{"path": "/known", "change": "changed", "before": "before", "after": "after"}])
        imported = compile_evidence(packet)
        self.assertEqual(imported["actual_execution"], packet["actual_execution"])

    def test_initialization_events_separate_from_live_agent_execution(self):
        raw = episode()
        raw["tool_events"][0]["event_index"] = 1
        raw["tool_events"].insert(0, {"event_index": 0, "phase": "initialization", "name": "setup",
                                      "requestor": "assistant", "arguments": {}, "response": "ready",
                                      "state_before": "pre", "state_after": "post"})
        packet = compile_evidence(raw)
        self.assertEqual([e["ref"] for e in packet["actual_execution"]], ["e:1"])
        self.assertEqual(packet["offline_database_facts"]["initialization_execution"][0]["name"], "setup")
        self.assertTrue(packet["capture_integrity"]["verified"])
        imported = compile_evidence(packet)
        self.assertEqual(imported["offline_database_facts"]["initialization_execution"],
                         packet["offline_database_facts"]["initialization_execution"])

    def test_effective_agent_policy_preserves_additional_language_guidance(self):
        raw = episode()
        raw["agent_policy"] = raw["domain_policy"] + "\nRespond in English."
        self.assertEqual(compile_evidence(raw)["policy"], raw["agent_policy"])

    def test_actual_environment_error_receipt_preserved_without_fake_response(self):
        raw = episode()
        raw["messages"][3].update(content="Error: refused", error=True)
        event = raw["tool_events"][0]
        event.pop("response")
        event["error_type"] = "ValueError"
        event["tool_message"] = copy.deepcopy(raw["messages"][3])
        packet = compile_evidence(raw)
        captured = packet["actual_execution"][0]
        self.assertNotIn("response", captured)
        self.assertEqual(captured["environment_receipt"]["content"], "Error: refused")
        self.assertTrue(captured["returned_error"])
        self.assertTrue(captured["receipt_exact"])
        self.assertTrue(packet["capture_integrity"]["verified"])

    def test_capture_failure_and_truncation_metadata_are_not_success(self):
        raw = episode()
        raw["status"] = "error"
        raw["termination_reason"] = "max_steps"
        raw["capture_integrity"] = {"capture_errors": [{"type": "missing_hook", "private": "SECRET"}]}
        packet = compile_evidence(raw)
        self.assertFalse(packet["capture_integrity"]["verified"])
        self.assertEqual(packet["capture_integrity"]["episode_status"], "error")
        self.assertEqual(packet["capture_integrity"]["termination_reason"], "max_steps")
        self.assertEqual(packet["capture_integrity"]["capture_errors"], [{"type": "missing_hook"}])
        self.assertNotIn("SECRET", json.dumps(packet))

    def test_failed_state_capture_does_not_prove_no_change(self):
        raw = episode()
        missing = {"agent": {"available": True, "hash": None, "hash_error_type": "Unavailable"},
                   "user": {"available": False, "hash": None}}
        raw["tool_events"][0].update(state_before=missing, state_after=copy.deepcopy(missing))
        raw["tool_events"][0].pop("state_changes")
        packet = compile_evidence(raw)
        self.assertIsNone(packet["actual_execution"][0]["backend_state_changed"])


if __name__ == "__main__":
    unittest.main()
