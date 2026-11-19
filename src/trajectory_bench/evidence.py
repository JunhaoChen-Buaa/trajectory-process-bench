"""Compile source-only grading evidence from captured or normalized episodes.

This module does not run a simulator, infer user intent, or import experimental
scripts.  In particular, a tool request, a returned receipt, and an actual
execution event are different observations.  Missing observations stay missing.
The explicit field lists keep task answers, rewards and model metadata outside
the grading packet; the original episode should be retained separately.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "trajectory_bench.evidence.v1"


class EvidenceError(ValueError):
    """The source is malformed or a requested integrity check failed."""


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("Evidence must consist of finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _parsed(value: Any) -> Any:
    """Compare JSON/python-repr receipts without altering the original text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return value


def _object(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{description} must be an object")
    return value


def _list(value: Any, description: str) -> list:
    if not isinstance(value, list):
        raise EvidenceError(f"{description} must be an array")
    return value


def _index(value: Any, default: int, description: str) -> int:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise EvidenceError(f"{description} must be a nonnegative integer")
    return result


def _actor(value: Any) -> str | None:
    if value is None:
        return None
    if value not in ("assistant", "user"):
        raise EvidenceError("Tool requestor must be assistant, user, or null")
    return value


def _call(raw: Any, actor: str, notes: list[str]) -> dict:
    source = _object(raw, "Tool call")
    function = source.get("function", source)
    function = _object(function, "Tool function")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise EvidenceError("Tool calls need a nonempty name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        parsed = _parsed(arguments)
        if isinstance(parsed, dict):
            arguments = parsed
        else:
            notes.append("A tool argument string could not be parsed as an object; original text retained.")
    elif not isinstance(arguments, dict):
        raise EvidenceError("Tool arguments must be an object or their original string")
    call_id = source.get("id", source.get("tool_call_id"))
    if call_id is not None and not isinstance(call_id, str):
        raise EvidenceError("Tool call IDs must be strings")
    return {"id": call_id, "name": name, "arguments": _clone(arguments),
            "requestor": _actor(source.get("requestor")) or actor}


def _history(messages: list, normalized: bool, notes: list[str]) -> list[dict]:
    rows: list[dict] = []
    previous = -1
    for position, item in enumerate(messages):
        source = _object(item, "History message")
        role = source.get("role")
        # Simulator/system prompts can contain the private task and answers.
        if role in ("system", "developer"):
            notes.append("Private system/developer message omitted; supplied domain policy is kept separately.")
            continue
        if role not in ("assistant", "user", "tool"):
            raise EvidenceError(f"Unsupported history role: {role!r}")
        i = _index(source.get("i") if normalized else None, position, "History index")
        if i <= previous:
            raise EvidenceError("History indices must be unique and increasing")
        previous = i
        if normalized and source.get("ref", f"h:{i}") != f"h:{i}":
            raise EvidenceError("History ref disagrees with its index")
        requestor = _actor(source.get("requestor"))
        calls = [_call(c, role, notes) for c in _list(source.get("tool_calls") or [], "Tool calls")]
        if calls and role == "tool":
            raise EvidenceError("A tool receipt cannot itself request tool calls")
        public = role == "assistant" and not calls
        if type(source.get("customer_visible")) is bool:
            if calls and source["customer_visible"]:
                notes.append(f"{i}: text accompanying tool requests is retained as unsent content.")
            public = public and source["customer_visible"]
        row = {"i": i, "ref": f"h:{i}", "role": role,
               "requestor": requestor, "customer_visible": public,
               "agent_observed": role == "user" and not calls,
               "_observation_explicit": type(source.get("agent_observed")) is bool}
        if "content" in source:
            row["content"] = _clone(source["content"])
        else:
            row["content"] = ""
        if calls:
            row["tool_calls"] = calls
        if role == "tool":
            row["tool_call_id"] = source.get("tool_call_id") or source.get("id")
            for key in ("name", "error", "execution_status"):
                if key in source:
                    row[key] = _clone(source[key])
        if type(source.get("agent_observed")) is bool:
            row["agent_observed"] = source["agent_observed"]
        rows.append(row)
    return rows


def _calls_and_receipts(history: list[dict], notes: list[str]) -> tuple[list[dict], list[str]]:
    calls: list[dict] = []
    unmatched_receipts = []
    for row in history:
        for number, call in enumerate(row.get("tool_calls", [])):
            calls.append({"call": call, "history": row, "call_index": number, "receipt": None})
        if row["role"] != "tool":
            continue
        call_id = row.get("tool_call_id")
        candidates = [c for c in calls if c["receipt"] is None and call_id is not None
                      and c["call"]["id"] == call_id
                      and (row.get("requestor") is None or c["call"]["requestor"] == row["requestor"])
                      and (not row.get("name") or row["name"] == c["call"]["name"])]
        if len(candidates) == 1:
            candidate = candidates[0]
            candidate["receipt"] = row
            if row["requestor"] is None:
                row["requestor"] = candidate["call"]["requestor"]
            if not row["_observation_explicit"]:
                row["agent_observed"] = row["requestor"] == "assistant"
        else:
            unmatched_receipts.append(row["ref"])
            notes.append(f"{row['ref']}: tool receipt has no unique preceding structured call.")
    return calls, unmatched_receipts


def _has_error(source: Mapping[str, Any]) -> bool:
    if source.get("error") or source.get("error_type"):
        return True
    if str(source.get("execution_status", "")).lower() in ("error", "failed", "failure", "rejected"):
        return True
    value = _parsed(source.get("response", source.get("content")))
    if isinstance(value, dict):
        return bool(value.get("error")) or value.get("success") is False
    return isinstance(value, str) and value.lstrip().lower().startswith(("error:", "error(", "error "))


def _state_changed(source: Mapping[str, Any]) -> bool | None:
    if source.get("state_before") is not None and source.get("state_after") is not None:
        before, after = source["state_before"], source["state_after"]
        if (isinstance(before, dict) and isinstance(after, dict)
                and set(before).issubset({"agent", "user"}) and set(after).issubset({"agent", "user"})):
            comparisons = []
            unknown = False
            for role in set(before) | set(after):
                left, right = before.get(role, {}), after.get(role, {})
                if not isinstance(left, dict) or not isinstance(right, dict):
                    unknown = True
                elif left.get("available") is False and right.get("available") is False:
                    continue
                elif left.get("hash") is not None and right.get("hash") is not None:
                    comparisons.append(left["hash"] != right["hash"])
                elif "snapshot" in left and "snapshot" in right:
                    comparisons.append(left["snapshot"] != right["snapshot"])
                else:
                    unknown = True
            if any(comparisons):
                return True
            return None if unknown or not comparisons else False
        return before != after
    flag = source.get("backend_state_changed")
    return flag if type(flag) is bool else None


def _compact_state(value: Any, include_snapshots: bool) -> Any:
    if include_snapshots or not isinstance(value, (dict, list)):
        return _clone(value)
    if isinstance(value, dict) and value.get("snapshot_omitted") is True and "snapshot_sha256" in value:
        return {"snapshot_omitted": True, "snapshot_sha256": value["snapshot_sha256"]}
    if isinstance(value, dict) and set(value).issubset({"agent", "user"}):
        compact = {}
        for role, row in value.items():
            row = _object(row, "Environment state")
            compact[role] = {k: _clone(row[k]) for k in
                             ("available", "hash", "hash_error_type", "snapshot_error_type",
                              "snapshot_omitted", "snapshot_sha256") if k in row}
            if "snapshot" in row:
                compact[role].update(snapshot_omitted=True, snapshot_sha256=_hash(row["snapshot"]))
            elif not compact[role] and row:
                compact[role].update(snapshot_omitted=True, snapshot_sha256=_hash(row))
        return compact
    # An opaque whole-state object is not a source for guessed entity identities.
    return {"snapshot_omitted": True, "snapshot_sha256": _hash(value)}


def _events(raw_events: list, calls: list[dict], tool_types: Mapping[str, Any],
            notes: list[str], include_snapshots: bool) -> tuple[list[dict], list[dict]]:
    events = []
    matched: set[int] = set()
    snapshots = []
    last_index = -1
    for number, item in enumerate(raw_events):
        source = _object(item, "Execution event")
        e = _index(source.get("e", source.get("event_index")), number, "Execution index")
        if e <= last_index:
            raise EvidenceError("Execution indices must be unique and increasing")
        last_index = e
        if source.get("ref", f"e:{e}") != f"e:{e}":
            raise EvidenceError("Execution ref disagrees with its index")
        name = source.get("name")
        if not isinstance(name, str) or not name:
            raise EvidenceError("Execution events need a nonempty tool name")
        actor = _actor(source.get("requestor"))
        arguments = _parsed(source.get("arguments", {}))
        call_id = source.get("tool_call_id")
        candidates = [(i, c) for i, c in enumerate(calls) if i not in matched
                      and c["call"]["name"] == name and c["call"]["arguments"] == arguments
                      and (actor is None or c["call"]["requestor"] == actor)]
        if call_id is not None:
            candidates = [(i, c) for i, c in candidates if c["call"]["id"] == call_id]
        if source.get("call_ref"):
            candidates = [(i, c) for i, c in candidates if c["history"]["ref"] == source["call_ref"]
                          and (source.get("call_index") is None or c["call_index"] == source["call_index"])]
            if not candidates:
                raise EvidenceError("Recorded execution call_ref cannot be verified against history")
        exact = [(i, c) for i, c in candidates if c["receipt"] is not None
                 and "response" in source
                 and _parsed(c["receipt"].get("content")) == _parsed(source["response"])]
        if exact:
            candidates = exact
        elif call_id is None and len(candidates) > 1:
            # A rejected attempt absent from the successful event ledger must not
            # consume the later successful event just because arguments repeat.
            without_error = [(i, c) for i, c in candidates
                             if c["receipt"] is None or not _has_error(c["receipt"])]
            if without_error:
                candidates = without_error
        chosen = candidates[0] if candidates else None
        if len(candidates) > 1:
            notes.append(f"e:{e}: repeated matching calls paired in recorded order; no unique event call ID.")
        receipt = chosen[1]["receipt"] if chosen else None
        if chosen:
            matched.add(chosen[0])
            actor = actor or chosen[1]["call"]["requestor"]
        else:
            notes.append(f"e:{e}: actual event has no matching structured call; it is retained unpaired.")
        if source.get("receipt_ref") and (receipt is None or source["receipt_ref"] != receipt["ref"]):
            raise EvidenceError("Recorded execution receipt_ref cannot be verified against history")
        response_present = "response" in source
        recorded_receipt = source.get("tool_message", source.get("environment_receipt"))
        if recorded_receipt is not None:
            recorded_receipt = _object(recorded_receipt, "Environment tool receipt")
        receipt_content_present = response_present or (recorded_receipt is not None and "content" in recorded_receipt)
        receipt_value = (recorded_receipt["content"] if recorded_receipt is not None and "content" in recorded_receipt
                         else source.get("response"))
        error = _has_error(source) or (receipt is not None and _has_error(receipt))
        if recorded_receipt is not None:
            error = error or _has_error(recorded_receipt)
        row = {"e": e, "ref": f"e:{e}", "name": name, "requestor": actor,
               "arguments": _clone(arguments), "backend_state_changed": _state_changed(source),
               "tool_type": source.get("tool_type", tool_types.get(name)),
               "call_ref": chosen[1]["history"]["ref"] if chosen else None,
               "call_index": chosen[1]["call_index"] if chosen else None,
               "tool_call_id": chosen[1]["call"]["id"] if chosen else call_id,
               "receipt_ref": receipt["ref"] if receipt else None,
               "receipt_exact": (receipt_content_present and _parsed(receipt.get("content")) == _parsed(receipt_value)) if receipt else None,
               "returned_error": bool(error),
               "execution_status": "error" if error else "returned" if response_present else "unknown",
               "pairing": "tool_call_id" if chosen and call_id is not None else "structured_source_match" if chosen else "unpaired"}
        for key in ("response", "error", "error_type", "state_before", "state_after", "phase"):
            if key in source:
                row[key] = (_compact_state(source[key], include_snapshots)
                            if key in ("state_before", "state_after") else _clone(source[key]))
        if recorded_receipt is not None:
            row["environment_receipt"] = {k: _clone(recorded_receipt[k]) for k in
                                          ("id", "tool_call_id", "role", "requestor", "content", "error", "execution_status")
                                          if k in recorded_receipt}
        if "state_changes" in source:
            changes = []
            for change in _list(source["state_changes"], "Captured state changes"):
                change = _object(change, "Captured state change")
                if not isinstance(change.get("path"), (str, list)):
                    raise EvidenceError("Captured state change requires a source path")
                changes.append({k: _clone(change[k]) for k in
                                ("path", "change", "before", "after", "before_exists", "after_exists") if k in change})
            row["state_changes"] = changes
        if row["backend_state_changed"] is None:
            notes.append(f"e:{e}: before/after state unavailable; state change is unknown.")
        if receipt and response_present and not row["receipt_exact"]:
            notes.append(f"e:{e}: actual response differs from the delivered tool receipt.")
        if isinstance(source.get("state_before"), (dict, list)) or isinstance(source.get("state_after"), (dict, list)):
            if include_snapshots:
                snapshots.append({"ref": f"s:{e}", "actual_event_ref": f"e:{e}",
                                  **{key: _clone(source[key]) for key in ("state_before", "state_after") if key in source},
                                  "verification": "Recorded snapshots; not independently replayed by this converter."})
            elif "state_changes" not in source and "state_before" in source and "state_after" in source:
                row["state_changes"] = _difference(source["state_before"], source["state_after"])
        events.append(row)
    unmatched = [{"call_ref": c["history"]["ref"], "call_index": c["call_index"],
                  "tool_call_id": c["call"]["id"], "name": c["call"]["name"],
                  "requestor": c["call"]["requestor"],
                  "receipt_ref": c["receipt"]["ref"] if c["receipt"] else None,
                  "returned_error": _has_error(c["receipt"]) if c["receipt"] else None}
                 for i, c in enumerate(calls) if i not in matched]
    if unmatched:
        notes.append("Structured calls without captured execution events remain attempts, not invented successful executions.")
    return events, [unmatched, snapshots]


def _difference(before: Any, after: Any, path: str = "") -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        output = []
        for key in sorted(set(before) | set(after)):
            pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before:
                output.append({"path": pointer, "change": "added", "after": _clone(after[key])})
            elif key not in after:
                output.append({"path": pointer, "change": "removed", "before": _clone(before[key])})
            else:
                output.extend(_difference(before[key], after[key], pointer))
        return output
    if before != after:
        return [{"path": path, "change": "changed", "before": _clone(before), "after": _clone(after)}]
    return []


_FACT_KEYS = ("initial_phone_line_bindings", "catalogue_variants_referenced_in_actual_calls")
_REPLAY_KEYS = ("supported_domain", "events_replayed", "all_recorded_hashes_match",
                "post_operation_db_objects", "offline_flight_options", "error", "note")
_SNAPSHOT_KEYS = ("ref", "actual_event_ref", "state_before_hash", "state_after_hash",
                  "before_object", "after_object", "new_payment_entries")


def _offline(source: Mapping[str, Any], normalized: bool, snapshots: list[dict],
             event_refs: set[str], notes: list[str], include_snapshots: bool) -> dict:
    facts: dict = {"visibility": "Offline captured facts; not necessarily observed by the agent or delivered to the user."}
    if normalized:
        original = _object(source.get("offline_database_facts", {}), "Offline database facts")
        for key in _FACT_KEYS:
            if key in original:
                facts[key] = _clone(original[key])
        replay = original.get("replayed_actual_state")
        if replay is not None:
            replay = _object(replay, "Recorded replay facts")
            facts["replayed_actual_state"] = {k: _clone(replay[k]) for k in _REPLAY_KEYS if k in replay and k != "post_operation_db_objects"}
            states = []
            for item in _list(replay.get("post_operation_db_objects", []), "Recorded database snapshots"):
                item = _object(item, "Recorded database snapshot")
                if item.get("actual_event_ref") not in event_refs:
                    raise EvidenceError("State snapshot references an absent execution event")
                states.append({k: _clone(item[k]) for k in _SNAPSHOT_KEYS if k in item})
            facts["replayed_actual_state"]["post_operation_db_objects"] = states
            notes.append("Imported replay/hash claims are retained source claims; this converter does not replay the original database.")
        for key in ("initial_state", "final_state"):
            if key in original:
                facts[key] = _compact_state(original[key], include_snapshots)
        for item in _list(original.get("recorded_state_snapshots", []), "Captured state snapshots"):
            item = _object(item, "Captured state snapshot")
            if item.get("actual_event_ref") not in event_refs:
                raise EvidenceError("Captured snapshot references an absent execution event")
            if not any(s["actual_event_ref"] == item["actual_event_ref"] for s in snapshots):
                if include_snapshots:
                    snapshots.append({k: _clone(item[k]) for k in
                                      ("ref", "actual_event_ref", "state_before", "state_after", "verification") if k in item})
    if snapshots:
        facts["recorded_state_snapshots"] = snapshots
    # Native captures guarantee these are environment snapshots, not task specs.
    if str(source.get("schema_version", "")).startswith("trajectory_bench.native_episode."):
        for key in ("initial_state", "final_state"):
            if source.get(key) is not None:
                facts[key] = _compact_state(source[key], include_snapshots)
                if not include_snapshots:
                    notes.append("Whole initial/final database snapshots omitted from grading evidence; recorded hashes and event deltas retained in this packet, full snapshots in source.")
    if not snapshots and not facts.get("replayed_actual_state", {}).get("post_operation_db_objects"):
        notes.append("No object-level before/after state snapshots are supplied; actual stored effects may be uncheckable.")
    return facts


def compile_evidence(episode: Mapping[str, Any], *, source_path: str | Path | None = None,
                     expected_sha256: str | None = None,
                     include_state_snapshots: bool = False) -> dict:
    """Return an answer-free packet, preserving source order and honest gaps.

    ``expected_sha256`` verifies exact file bytes when ``source_path`` is supplied,
    otherwise the canonical JSON hash of ``episode``.  For normalized imports,
    structural links are rechecked; inherited replay claims are not certified.
    The path itself is never sent to the grading model (it can identify a model).
    Whole environment snapshots are omitted by default; compact captured changes
    and hashes remain.  Set ``include_state_snapshots`` for local forensic use.
    """
    source = _object(episode, "Episode")
    canonical_hash = _hash(source)
    digest = canonical_hash
    hash_kind = "canonical_json"
    if source_path is not None:
        raw_bytes = Path(source_path).read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        hash_kind = "file_bytes"
        try:
            loaded = json.loads(raw_bytes.decode("utf-8-sig"))
        except (UnicodeError, ValueError) as exc:
            raise EvidenceError("Source file is not valid UTF-8 JSON") from exc
        if _hash(loaded) != canonical_hash:
            raise EvidenceError("Source file does not contain the supplied episode")
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise EvidenceError("Source SHA-256 does not match the expected value")
    normalized = "history" in source and "policy" in source
    notes: list[str] = []
    domain = source.get("domain")
    if not isinstance(domain, str) or not domain:
        raise EvidenceError("Episode requires a nonempty domain")
    policy = source.get("agent_policy", source.get("domain_policy", source.get("policy")))
    if not isinstance(policy, str) or not policy.strip():
        raise EvidenceError("Full domain policy is required; it cannot be inferred from private prompts")
    schemas = source.get("available_tool_schemas", source.get("tools", []))
    schemas = _list(schemas, "Available tool schemas")
    if not schemas:
        notes.append("No tool schemas were captured; tool constraints cannot be reconstructed by this converter.")
    if normalized:
        messages = source["history"]
        raw_events = source.get("actual_execution", source.get("actual_tool_events", []))
    else:
        simulation = _object(source.get("simulation", {}), "Simulation")
        messages = simulation.get("messages", source.get("messages"))
        if messages is None:
            raise EvidenceError("Episode requires captured messages")
        if "messages" in source and "messages" in simulation and source["messages"] != simulation["messages"]:
            raise EvidenceError("Episode messages and simulation.messages disagree")
        raw_events = source.get("tool_events", [])
    history = _history(_list(messages, "History"), normalized, notes)
    calls, unmatched_receipts = _calls_and_receipts(history, notes)
    raw_events = _list(raw_events, "Execution events")
    initialization = [e for e in raw_events if _object(e, "Execution event").get("phase") == "initialization"]
    trial_events = [e for e in raw_events if e.get("phase") != "initialization"]
    types = _object(source.get("tool_types", {}), "Tool types")
    events, (unmatched_calls, snapshots) = _events(trial_events, calls, types, notes, include_state_snapshots)
    offline = _offline(source, normalized, snapshots, {e["ref"] for e in events}, notes, include_state_snapshots)
    if normalized and (source.get("execution_field_comparisons") or source.get("operation_navigation") or source.get("query_navigation")):
        notes.append("Imported derived comparison/navigation summaries are not copied as conclusions; source history, execution records, catalogue and database facts remain available.")
    if initialization:
        startup, _ = _events(initialization, [], types, [], include_state_snapshots)
        offline["initialization_execution"] = startup
        notes.append("Initialization/replayed setup events are separate offline history, not live Agent trial actions.")
    elif normalized and source.get("offline_database_facts", {}).get("initialization_execution"):
        startup, _ = _events(_list(source["offline_database_facts"]["initialization_execution"], "Initialization events"),
                             [], types, [], include_state_snapshots)
        offline["initialization_execution"] = startup
    differences = []
    for snapshot in offline.get("replayed_actual_state", {}).get("post_operation_db_objects", []):
        if "before_object" in snapshot and "after_object" in snapshot:
            differences.append({"event_ref": snapshot["actual_event_ref"], "state_ref": snapshot.get("ref"),
                                "changes": _difference(snapshot["before_object"], snapshot["after_object"]),
                                "new_payment_entries": _clone(snapshot.get("new_payment_entries", []))})
    for snapshot in snapshots:
        if "state_before" in snapshot and "state_after" in snapshot:
            differences.append({"event_ref": snapshot["actual_event_ref"], "state_ref": snapshot["ref"],
                                "changes": _difference(snapshot["state_before"], snapshot["state_after"])})
    for event in events:
        if "state_changes" in event:
            differences.append({"event_ref": event["ref"], "state_ref": None,
                                "changes": _clone(event["state_changes"]),
                                "verification": "Captured event delta; not independently replayed by this converter."})
    if not raw_events and calls:
        notes.append("No execution event ledger supplied; tool receipts alone do not certify backend state changes.")
    capture_claim = _object(source.get("capture_integrity", {}), "Capture integrity")
    capture_errors = [{"type": item["type"]} for item in
                      _list(capture_claim.get("capture_errors", []), "Capture errors")
                      if isinstance(item, dict) and isinstance(item.get("type"), str)]
    if capture_errors:
        notes.append("Source instrumentation reported capture errors; retained separately from business violations.")
    episode_status = source.get("status", capture_claim.get("episode_status"))
    if episode_status not in (None, "complete", "error", "started"):
        episode_status = "other_recorded_status"
    termination = source.get("termination_reason", capture_claim.get("termination_reason"))
    if termination is None and isinstance(source.get("simulation"), dict):
        termination = source["simulation"].get("termination_reason")
    for row in history:
        row.pop("_observation_explicit", None)
    return {"schema_version": SCHEMA_VERSION, "domain": domain, "policy": policy,
            "available_tool_schemas": _clone(schemas), "history": history,
            "actual_execution": events, "offline_database_facts": offline,
            "mechanical_state_differences": differences,
            "capture_integrity": {"verified": not unmatched_calls and not unmatched_receipts and not capture_errors
                                  and all(e["call_ref"] and e["receipt_ref"] and e["receipt_exact"] for e in events),
                                  "verification_scope": "Structural call/event/receipt consistency only; not business correctness or independent database replay.",
                                  "unmatched_structured_calls": len(unmatched_calls),
                                  "unmatched_call_details": unmatched_calls,
                                  "unmatched_receipts": unmatched_receipts,
                                  "capture_errors": capture_errors,
                                  "episode_status": episode_status,
                                  "termination_reason": termination,
                                  "notes": list(dict.fromkeys(notes))},
            "evidence_provenance": {"conversion": "normalized_import" if normalized else "captured_episode",
                                    "source_sha256": digest, "hash_kind": hash_kind,
                                    "expected_hash_verified": expected_sha256 is not None,
                                    "canonical_source_sha256": canonical_hash}}


def load_evidence(path: str | Path, *, expected_sha256: str | None = None,
                  include_state_snapshots: bool = False) -> dict:
    """Load UTF-8 JSON and verify exact source bytes before conversion."""
    source_path = Path(path)
    try:
        episode = json.loads(source_path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise EvidenceError("Source file is not valid UTF-8 JSON") from exc
    return compile_evidence(episode, source_path=source_path, expected_sha256=expected_sha256,
                            include_state_snapshots=include_state_snapshots)


# Small discoverable aliases for integration code.
from_episode = compile_evidence
packet = compile_evidence
