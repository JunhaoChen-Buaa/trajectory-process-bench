"""Command-line entry. Running business tasks is always an explicit selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, load_json, save_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trajectory-bench", description="Run an Agent or score saved interaction trajectories")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Agent → native tasks → captured trajectory → process score → reports")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--domain", required=True, choices=["airline", "retail", "telecom"])
    run.add_argument("--task-id", action="append", required=True, help="Repeat for multiple task IDs")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-steps", type=int, default=80)
    run.add_argument("--max-errors", type=int, default=5)
    run.add_argument("--agent-factory", help="Importable module:function accepting tools, domain_policy and kwargs")
    run.add_argument("--capture-only", action="store_true", help="Save trajectory without a judge API call")
    suite = sub.add_parser("suite", help="Run the fixed 45-task suite across all three domains")
    suite.add_argument("--config", required=True, type=Path)
    suite.add_argument("--output", required=True, type=Path)
    suite.add_argument("--trials", type=int, default=1)
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--max-steps", type=int, default=200)
    suite.add_argument("--max-errors", type=int, default=10)
    suite.add_argument("--agent-factory")
    suite.add_argument("--limit", type=int, help="Optional first N fixed tasks for installation smoke testing")
    suite.add_argument("--capture-only", action="store_true")
    score = sub.add_parser("score", help="Grade existing raw episodes or normalized evidence packets")
    score.add_argument("--config", required=True, type=Path)
    score.add_argument("--input", action="append", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    imp = sub.add_parser("convert", help="Convert one raw episode to source-only grading evidence, no API")
    imp.add_argument("--input", required=True, type=Path)
    imp.add_argument("--output", required=True, type=Path)
    imp.add_argument("--sha256", help="Optional expected source-byte SHA256")
    rep = sub.add_parser("report", help="Rebuild reports from saved automatic records; optional explicit source reviews")
    rep.add_argument("--records", required=True, type=Path)
    rep.add_argument("--reviews", type=Path, help="JSONL explicit review overlays; automatic answers remain unchanged")
    rep.add_argument("--output", required=True, type=Path)
    agg = sub.add_parser("aggregate", help="Aggregate complete annotation JSONL with frozen hierarchy, zero API")
    agg.add_argument("--annotations", required=True, type=Path)
    agg.add_argument("--output", required=True, type=Path)
    sub.add_parser("tasks", help="List the 45 paper tasks, no API")
    return p


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("run", "suite"):
            from .workflow import native_run, resource_text
            if args.trials < 1 or args.max_steps < 1 or args.max_errors < 1:
                raise ValueError("trials/max-steps/max-errors must be positive")
            choices = None
            if args.command == "suite":
                tasks = json.loads(resource_text("task_index.json"))
                if args.limit is not None and args.limit < 1:
                    raise ValueError("limit must be positive")
                choices = [(row["domain"], str(row["task_id"])) for row in tasks[:args.limit]]
            rows = native_run(load_config(args.config), args.output,
                              args.domain if args.command == "run" else "suite",
                              args.task_id if args.command == "run" else [],
                              args.trials, args.seed, args.max_steps, args.max_errors,
                              args.agent_factory, args.capture_only, _task_selection=choices)
            failed = sum(row.get("automatic", {}).get("status") == "technical_unscored" for row in rows)
            print(json.dumps({"records": len(rows), "output": str(args.output.resolve()),
                              "status": "completed_with_errors" if failed else "completed", "technical_unscored": failed,
                              "report": str((args.output / "report.html").resolve())}, ensure_ascii=False))
            if failed:
                return 1
        elif args.command == "score":
            from .workflow import score_files
            rows = score_files(args.input, load_config(args.config), args.output)
            print(json.dumps({"records": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))
            if any(row.get("automatic", {}).get("status") == "technical_unscored" for row in rows):
                return 1
        elif args.command == "convert":
            from .evidence import load_evidence
            save_json(args.output, load_evidence(args.input, expected_sha256=args.sha256))
            print(str(args.output.resolve()))
        elif args.command == "aggregate":
            from .grading import aggregate
            summary = aggregate(read_jsonl(args.annotations))
            save_json(args.output, summary)
            print(json.dumps(summary, ensure_ascii=False))
        elif args.command == "report":
            from .grading import apply_review_overlays
            from .reporting import write_reports
            rows = read_jsonl(args.records)
            # Artifact paths originally belong to the records directory, even
            # when an explicitly reviewed report is written somewhere else.
            import os
            for row in rows:
                for name, value in list(row.get("artifacts", {}).items()):
                    path = Path(value)
                    if not path.is_absolute():
                        path = args.records.resolve().parent / path
                    row["artifacts"][name] = os.path.relpath(path, args.output.resolve())
            if args.reviews:
                from .config import digest
                evidence = {}
                for row in rows:
                    target = row.get("artifacts", {}).get("evidence")
                    if target:
                        path = Path(target)
                        if not path.is_absolute():
                            path = args.output.resolve() / path
                        # Collapse .. before opening: on POSIX the report output
                        # directory may not exist yet while reviews are loaded.
                        packet = load_json(path.resolve())
                        if row.get("evidence_sha256") and digest(packet) != row["evidence_sha256"]:
                            raise ValueError(f"Evidence hash differs for {row['case_id']}")
                        evidence[row["case_id"]] = packet
                rows = apply_review_overlays(rows, read_jsonl(args.reviews), evidence_by_case=evidence)
            result = write_reports(rows, args.output)
            print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False))
        elif args.command == "tasks":
            from .workflow import resource_text
            print(resource_text("task_index.json"))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
