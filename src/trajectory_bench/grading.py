"""Validate source-linked annotations and aggregate categorical process verdicts.

No model calls or reference-answer lookup occur here.  Reference existence and
literal quotations can be checked mechanically; whether evidence supports a
business judgement still requires source review.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DIMENSIONS = ("RULE", "CONSENT", "OUTCOME", "REPORT", "FACT", "STEPS")
DIMS = DIMENSIONS
STATES = ("pass", "fail", "unknown", "na")
_FORBIDDEN = {
    "original_reward", "original_score", "reward_info", "reference_answer",
    "reference_answers", "hidden_answer", "hidden_answers", "ground_truth",
    "gold_label", "expected_actions", "evaluation_criteria",
}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def evidence_index(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Index only actual source locations, never accept arbitrary ref prefixes."""
    result: dict[str, Any] = {}

    # Only compiler-owned locations define references. A tool response may
    # itself contain a user-controlled {"ref": "h:999"}; that is data, not a
    # new source location and must not make an invented citation valid.
    for section in ("history", "actual_execution"):
        records = evidence.get(section, [])
        if isinstance(records, list):
            for record in records:
                if isinstance(record, Mapping) and _nonempty(record.get("ref")):
                    result[record["ref"]] = record
    records = evidence.get("mechanical_state_differences", [])
    if isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping) and _nonempty(record.get("state_ref")):
                result[record["state_ref"]] = record
    if isinstance(evidence.get("policy"), str):
        result["policy"] = evidence["policy"]
        # Line references are one-based and must refer to a real policy line.
        for number, line in enumerate(evidence["policy"].splitlines(), 1):
            result[f"policy:{number}"] = line
    for key, prefix in (("offline_database_facts", "db"), ("final_state", "state")):
        if key in evidence:
            result[prefix] = evidence[key]
    # A JSON pointer is a precise location, not a wildcard db:/state: exemption.
    def pointers(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            children = value.items()
        elif isinstance(value, list):
            children = enumerate(value)
        else:
            return
        for key, child in children:
            part = str(key).replace("~", "~0").replace("/", "~1")
            ref = f"{prefix}/{part}"
            result[ref] = child
            pointers(child, ref)
    for key, prefix in (("offline_database_facts", "db:"), ("final_state", "state:")):
        if key in evidence:
            pointers(evidence[key], prefix)
    return result


def validate_annotation(value: Any, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return structural errors and mechanical evidence flags, without fixing input.

    Empty refs on N/A are allowed.  An applicable score needs a refs array;
    pass/unknown with an empty array remain visible warnings for compatibility
    with the published frozen annotations.  A fail must have real cited source
    locations and a corresponding substantive issue.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checked_refs = 0
    checked_quotes = 0
    index = evidence_index(evidence) if evidence is not None else None

    def error(path: str, code: str) -> None:
        errors.append(f"{path}:{code}")

    def forbidden(obj: Any, path: str = "annotation") -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                if str(key).lower() in _FORBIDDEN:
                    error(f"{path}.{key}", "forbidden_answer_or_original_score")
                forbidden(child, f"{path}.{key}")
        elif isinstance(obj, list):
            for number, child in enumerate(obj):
                forbidden(child, f"{path}[{number}]")

    def refs(obj: Mapping[str, Any], path: str, *, required: bool = False, nonempty: bool = False, key: str = "refs") -> list[str]:
        nonlocal checked_refs
        val = obj.get(key)
        if val is None and not required:
            return []
        if not isinstance(val, list) or any(not _nonempty(r) for r in val):
            error(f"{path}.{key}", "expected_string_array")
            return []
        if nonempty and not val:
            error(f"{path}.{key}", "source_evidence_required")
        if index is not None:
            for ref in val:
                checked_refs += 1
                if ref not in index:
                    error(f"{path}.{key}", f"nonexistent_ref:{ref}")
        return val

    def quotes(obj: Mapping[str, Any], path: str) -> None:
        nonlocal checked_quotes
        for key in ("quotes", "evidence_quotes"):
            if key not in obj:
                continue
            if not isinstance(obj[key], list):
                error(f"{path}.{key}", "expected_array")
                continue
            for number, quote in enumerate(obj[key]):
                qpath = f"{path}.{key}[{number}]"
                if not isinstance(quote, Mapping) or not _nonempty(quote.get("ref")) or not _nonempty(quote.get("quote")):
                    error(qpath, "expected_ref_and_literal_quote")
                elif index is not None:
                    checked_quotes += 1
                    if quote["ref"] not in index:
                        error(qpath, f"nonexistent_ref:{quote['ref']}")
                    elif quote["quote"] not in _json_text(index[quote["ref"]]):
                        error(qpath, "quote_not_found_at_ref")

    if not isinstance(value, Mapping):
        errors.append("annotation:expected_object")
        value = {}
    forbidden(value)
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        errors.append("groups:nonempty_array_required")
        groups = []
    raw_issues = value.get("issues", [])
    if not isinstance(raw_issues, list):
        errors.append("issues:expected_array")
        raw_issues = []
    issues: dict[str, Mapping[str, Any]] = {}
    for number, issue in enumerate(raw_issues):
        path = f"issues[{number}]"
        if not isinstance(issue, Mapping):
            error(path, "expected_object")
            continue
        iid = issue.get("issue_id")
        if not _nonempty(iid):
            error(path, "issue_id_required")
        elif iid in issues:
            error(path, "duplicate_issue_id")
        else:
            issues[iid] = issue
        for key in ("title", "basis", "effect"):
            if not _nonempty(issue.get(key)):
                error(f"{path}.{key}", "nonempty_string_required")
        if issue.get("severity") not in ("substantive", "minor"):
            error(f"{path}.severity", "invalid_severity")
        if issue.get("root_cause") not in ("agent", "tool", "user_simulator", "mixed", "unknown"):
            error(f"{path}.root_cause", "invalid_root_cause")
        if not isinstance(issue.get("recovered"), bool):
            error(f"{path}.recovered", "boolean_required")
        refs(issue, path, required=True, nonempty=issue.get("severity") == "substantive")
        quotes(issue, path)
    gids: set[str] = set()
    for number, group in enumerate(groups):
        path = f"groups[{number}]"
        if not isinstance(group, Mapping):
            error(path, "expected_object")
            continue
        gid = group.get("group_id")
        if not _nonempty(gid):
            error(f"{path}.group_id", "nonempty_string_required")
        elif gid in gids:
            error(f"{path}.group_id", "duplicate_group_id")
        else:
            gids.add(gid)
        for key in ("goal", "object"):
            if not _nonempty(group.get(key)):
                error(f"{path}.{key}", "nonempty_string_required")
        refs(group, path, key="request_refs")
        requirements = group.get("requirements")
        qids: set[str] = set()
        if requirements is None:
            warnings.append(f"{path}.requirements:not_separately_recorded")
        elif not isinstance(requirements, list):
            error(f"{path}.requirements", "expected_array")
        else:
            for qnumber, req in enumerate(requirements):
                qpath = f"{path}.requirements[{qnumber}]"
                if not isinstance(req, Mapping):
                    error(qpath, "expected_object")
                    continue
                qid = req.get("id")
                if not _nonempty(qid) or qid in qids:
                    error(qpath, "missing_or_duplicate_requirement_id")
                else:
                    qids.add(qid)
                for key in ("content", "basis"):
                    if not _nonempty(req.get(key)):
                        error(f"{qpath}.{key}", "nonempty_string_required")
                refs(req, qpath)
        revisions = group.get("revisions", [])
        if not isinstance(revisions, list):
            error(f"{path}.revisions", "expected_array")
        else:
            for rnumber, revision in enumerate(revisions):
                rpath = f"{path}.revisions[{rnumber}]"
                if not isinstance(revision, Mapping) or not _nonempty(revision.get("change")):
                    error(rpath, "change_required")
                elif not _nonempty(revision.get("ref")) or (index is not None and revision["ref"] not in index):
                    error(rpath, "nonexistent_revision_ref")
        scores = group.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(DIMENSIONS):
            error(f"{path}.scores", "exactly_six_dimensions_required")
        if not isinstance(scores, Mapping):
            continue
        for dim, score in scores.items():
            spath = f"{path}.scores.{dim}"
            if not isinstance(score, Mapping):
                error(spath, "expected_object")
                continue
            state = score.get("status")
            if state not in STATES:
                error(f"{spath}.status", "invalid_status")
            if not _nonempty(score.get("reason")):
                error(f"{spath}.reason", "nonempty_string_required")
            citations = refs(score, spath, required=state != "na", nonempty=state == "fail")
            if state in ("pass", "unknown") and not citations:
                warnings.append(f"{spath}.refs:empty_applicable_evidence")
            for key in ("requirement_ids", "issue_ids"):
                ids = score.get(key, [])
                if not isinstance(ids, list) or any(not _nonempty(i) for i in ids):
                    error(f"{spath}.{key}", "expected_string_array")
                    continue
                known = qids if key == "requirement_ids" else set(issues)
                if key == "requirement_ids" and requirements is None:
                    continue
                for identity in ids:
                    if identity not in known:
                        error(f"{spath}.{key}", f"unknown_id:{identity}")
            if state == "fail":
                iids = score.get("issue_ids", [])
                if not isinstance(iids, list) or not iids:
                    error(spath, "fail_requires_substantive_issue")
                elif not any(issues.get(iid, {}).get("severity") == "substantive" for iid in iids if isinstance(iid, str)):
                    error(spath, "fail_requires_substantive_issue")
            quotes(score, spath)
    links = value.get("evidence_links", [])
    if not isinstance(links, list):
        errors.append("evidence_links:expected_array")
    else:
        for number, link in enumerate(links):
            path = f"evidence_links[{number}]"
            if not isinstance(link, Mapping):
                error(path, "expected_object")
            else:
                for key in ("from", "to"):
                    if not _nonempty(link.get(key)) or (index is not None and link[key] not in index):
                        error(f"{path}.{key}", "nonexistent_ref")
                if not _nonempty(link.get("purpose")):
                    error(f"{path}.purpose", "nonempty_string_required")
    notes = value.get("auxiliary_notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, (str, Mapping)) for note in notes):
        errors.append("auxiliary_notes:expected_array")
    quotes(value, "annotation")
    if evidence is None:
        warnings.append("evidence_not_supplied:source_refs_not_checked")
    return {
        "valid": not errors, "errors": errors, "warnings": warnings,
        "evidence_checks": {
            "source_supplied": evidence is not None, "refs_checked": checked_refs,
            "literal_quotes_checked": checked_quotes,
            "semantic_source_review": "not_automated",
        },
    }


def validate(value: Any, evidence: Mapping[str, Any] | None = None) -> list[str]:
    """Compatibility helper returning errors only."""
    return validate_annotation(value, evidence)["errors"]


def parse_annotation(raw: Any, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Parse strict JSON; retain the original text, including malformed replies."""
    raw_text = raw if isinstance(raw, str) else None
    parsed = copy.deepcopy(raw) if isinstance(raw, Mapping) else None
    parse_errors: list[str] = []
    if isinstance(raw, str):
        try:
            # Duplicate keys otherwise disappear silently in json.loads.
            def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result = {}
                for key, val in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key: {key}")
                    result[key] = val
                return result
            parsed = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(f"nonfinite JSON number: {val}")))
        except (ValueError, TypeError) as exc:
            parse_errors.append(f"json_parse:{exc}")
    elif not isinstance(raw, Mapping):
        parse_errors.append("json_parse:expected JSON text or object")
    validation = validate_annotation(parsed, evidence)
    validation["errors"] = parse_errors + validation["errors"]
    validation["valid"] = not validation["errors"]
    return {"parsed": parsed, "raw_text": raw_text, "validation_errors": validation["errors"], "validation": validation}


def verdict(states: Iterable[str]) -> str:
    """Fail dominates unknown; N/A never contributes an artificial pass."""
    values = list(states)
    if set(values) - set(STATES):
        raise ValueError("Unrecognized status")
    if not values:
        raise ValueError("Cannot aggregate empty statuses")
    if "fail" in values:
        return "fail"
    if "unknown" in values:
        return "unknown"
    return "pass" if "pass" in values else "na"


def count(states: Iterable[str]) -> dict[str, Any]:
    counts = Counter(states)
    if set(counts) - set(STATES):
        raise ValueError("Unrecognized status")
    denominator = counts["pass"] + counts["fail"]
    return {"n": sum(counts.values()), **{s: counts[s] for s in STATES},
            "pass_rate_decided": counts["pass"] / denominator if denominator else None}


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce frozen aggregation, without rewriting or validating old labels.

    Each dimension is collapsed across a trajectory's groups, not averaged over
    groups or over dimensions.  Use grade_annotation first for new predictions.
    """
    seen: set[str] = set()
    trajectories: list[str] = []
    groups: list[str] = []
    dimensions: dict[str, list[str]] = {dim: [] for dim in DIMENSIONS}
    for row in rows:
        case_id = row["case_id"]
        if not _nonempty(case_id) or case_id in seen:
            raise ValueError("Missing or duplicate case_id")
        seen.add(case_id)
        if not isinstance(row.get("groups"), list) or not row["groups"]:
            raise ValueError("Missing business groups; an empty record is not pass")
        group_states = []
        for group in row["groups"]:
            if set(group["scores"]) != set(DIMENSIONS):
                raise ValueError("Incomplete scoring dimensions")
            state = verdict(group["scores"][dim]["status"] for dim in DIMENSIONS)
            groups.append(state)
            group_states.append(state)
        trajectories.append(verdict(group_states))
        for dim in DIMENSIONS:
            dimensions[dim].append(verdict(g["scores"][dim]["status"] for g in row["groups"]))
    return {"trajectories": count(trajectories), "business_groups": count(groups),
            "dimensions": {dim: count(values) for dim, values in dimensions.items()}}


def grade_annotation(case_id: str, annotation: Any, evidence: Mapping[str, Any] | None = None,
                     metadata: Mapping[str, Any] | None = None, layer: str = "automatic") -> dict[str, Any]:
    """Make a case record; invalid output is technical-unscored, never unknown."""
    if not _nonempty(case_id):
        raise ValueError("case_id must be a stable nonempty string")
    if layer != "automatic":
        raise ValueError("Create reviewed layers through a provenance-bearing overlay")
    # Accept parse_annotation's result as well as a raw model reply.
    if isinstance(annotation, Mapping) and "parsed" in annotation and "validation" in annotation:
        parsed_result = parse_annotation(annotation.get("raw_text") if annotation.get("raw_text") is not None else annotation.get("parsed"), evidence)
    else:
        parsed_result = parse_annotation(annotation, evidence)
    parsed = parsed_result["parsed"]
    if isinstance(parsed, Mapping) and parsed.get("case_id", case_id) != case_id:
        parsed_result["validation"]["errors"].append("case_id:model_output_mismatch")
        parsed_result["validation"]["valid"] = False
    valid = parsed_result["validation"]["valid"]
    status = "scored" if valid else "technical_unscored"
    state = verdict(verdict(g["scores"][d]["status"] for d in DIMENSIONS) for g in parsed["groups"]) if valid else None
    automatic = {"status": status, "verdict": state, "annotation": copy.deepcopy(parsed),
                 "validation": parsed_result["validation"], "source_reviewed": False}
    return {"case_id": case_id, "metadata": copy.deepcopy(dict(metadata or {})),
            "raw_annotation": copy.deepcopy(parsed), "raw_text": parsed_result["raw_text"],
            "automatic": automatic, "reviewed": None, "review_history": []}


def load_review_overlays(path: str | Path) -> list[dict[str, Any]]:
    """Read explicit review decisions.  Does not overwrite annotations on disk."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Review JSONL line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Review JSONL line {number}: object required")
            rows.append(row)
    return rows


def apply_review_overlays(records: Iterable[Mapping[str, Any]], overlays: Iterable[Mapping[str, Any]],
                          evidence_by_case: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Apply full annotations or explicit cell changes, preserving raw and auto.

    Overlay fields: case_id, reviewer, reviewed_at, reason, review_type (e.g.
    human / assistant / model_assisted), and annotation OR changes.  An optional
    source_reviewed boolean describes the reviewer's declared work, not an
    automatically certified human gold standard.  Changes identify group_id,
    dimension and replacement score fields; issues may be explicitly replaced.
    """
    result = [copy.deepcopy(dict(record)) for record in records]
    by_id: dict[str, dict[str, Any]] = {}
    for record in result:
        if record["case_id"] in by_id:
            raise ValueError("Duplicate case_id in records")
        by_id[record["case_id"]] = record
    seen_review_ids = {h["review_id"] for r in result for h in r.get("review_history", [])}
    for overlay in overlays:
        if not isinstance(overlay, Mapping):
            raise ValueError("Review overlay must be an object")
        case_id = overlay.get("case_id")
        if case_id not in by_id:
            raise ValueError(f"Unknown review case_id: {case_id}")
        for key in ("reviewer", "reviewed_at", "reason", "review_type"):
            if not _nonempty(overlay.get(key)):
                raise ValueError(f"Review {case_id}: nonempty {key} required")
        if "source_reviewed" in overlay and not isinstance(overlay["source_reviewed"], bool):
            raise ValueError("source_reviewed must be a boolean")
        digest = hashlib.sha256(json.dumps(overlay, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        review_id = overlay.get("review_id", digest)
        if not _nonempty(review_id) or review_id in seen_review_ids:
            raise ValueError("Missing or duplicate review_id")
        seen_review_ids.add(review_id)
        record = by_id[case_id]
        if ("annotation" in overlay) == ("changes" in overlay):
            raise ValueError("Review must provide exactly one of annotation or changes")
        if "annotation" in overlay:
            annotation = copy.deepcopy(overlay["annotation"])
        else:
            base = record.get("reviewed") or record["automatic"]
            annotation = copy.deepcopy(base.get("annotation"))
            if not isinstance(annotation, Mapping) or not isinstance(annotation.get("groups"), list):
                raise ValueError("Cannot patch an unparsed annotation; provide a complete annotation")
            changes = overlay["changes"]
            if not isinstance(changes, list):
                raise ValueError("Review changes must be an array")
            groups = {g["group_id"]: g for g in annotation["groups"]}
            for change in changes:
                if not isinstance(change, Mapping) or change.get("group_id") not in groups or change.get("dimension") not in DIMENSIONS:
                    raise ValueError("Review change must name an existing group and dimension")
                replacement = {k: copy.deepcopy(v) for k, v in change.items() if k not in ("group_id", "dimension")}
                groups[change["group_id"]]["scores"][change["dimension"]].update(replacement)
            if "issues" in overlay:
                annotation["issues"] = copy.deepcopy(overlay["issues"])
        evidence = evidence_by_case.get(case_id) if evidence_by_case is not None else None
        reviewed_record = grade_annotation(case_id, annotation, evidence, record.get("metadata"))
        if reviewed_record["automatic"]["status"] != "scored":
            raise ValueError(f"Invalid reviewed annotation {case_id}: " + "; ".join(reviewed_record["automatic"]["validation"]["errors"]))
        provenance = {k: copy.deepcopy(v) for k, v in overlay.items() if k not in ("annotation", "changes", "issues")}
        provenance.update(review_id=review_id, overlay_sha256=digest)
        reviewed = reviewed_record["automatic"]
        reviewed.update(source_reviewed=overlay.get("source_reviewed", False), provenance=provenance)
        record["reviewed"] = reviewed
        record.setdefault("review_history", []).append({"review_id": review_id, "overlay_sha256": digest,
                                                       "overlay": copy.deepcopy(dict(overlay)), "annotation": copy.deepcopy(annotation)})
    return result


def aggregate_records(records: Iterable[Mapping[str, Any]], layer: str = "automatic") -> dict[str, Any]:
    """Summarise one explicitly named layer with coverage and denominators."""
    if layer not in ("automatic", "reviewed"):
        raise ValueError("layer must be automatic or reviewed")
    materialized = list(records)
    seen: set[str] = set()
    scored = []
    technical = []
    captured = []
    absent = []
    warning_cases = []
    source_reviewed = 0
    slices: dict[str, dict[str, list[Mapping[str, Any]]]] = {key: defaultdict(list) for key in ("domain", "model_alias", "structure")}
    for record in materialized:
        cid = record["case_id"]
        if cid in seen:
            raise ValueError("Duplicate case_id")
        seen.add(cid)
        entry = record.get(layer)
        if entry is None:
            if layer == "automatic" and record.get("status") == "captured_unscored":
                captured.append(cid)
            elif layer == "automatic":
                technical.append(cid)
            else:
                absent.append(cid)
            continue
        if entry.get("status") != "scored":
            technical.append(cid)
            continue
        row = {"case_id": cid, **entry["annotation"]}
        row["case_id"] = cid
        scored.append(row)
        if entry.get("validation", {}).get("warnings"):
            warning_cases.append(cid)
        if entry.get("source_reviewed") is True:
            source_reviewed += 1
        metadata = record.get("metadata", {})
        for key in ("domain", "model_alias"):
            slices[key][str(metadata.get(key, "unspecified"))].append(row)
        slices["structure"]["single" if len(row["groups"]) == 1 else "multiple"].append(row)
    def with_denominators(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        result = aggregate(rows)
        for counts in [result["trajectories"], result["business_groups"], *result["dimensions"].values()]:
            counts["valid_denominator"] = counts["pass"] + counts["fail"]
        return result
    summary = with_denominators(scored)
    summary.update(layer=layer, cases_total=len(materialized), scored_cases=len(scored),
                   technical_unscored=len(technical), technical_unscored_case_ids=technical,
                   captured_unscored=len(captured), captured_unscored_case_ids=captured,
                   unreviewed=len(absent), unreviewed_case_ids=absent,
                   source_reviewed_cases=source_reviewed, evidence_warning_cases=len(warning_cases),
                   evidence_warning_case_ids=warning_cases,
                   breakdowns={key: {name: with_denominators(rows) for name, rows in sorted(values.items())} for key, values in slices.items()})
    return summary
