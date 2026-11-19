"""Run stages with immutable inputs, auditable outputs and bounded API use."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.resources import files
import json
import hashlib
from pathlib import Path
from typing import Callable

from . import __version__
from .config import digest, load_json, safe_config, save_json
from .evidence import compile_evidence, load_evidence
from .llm import ChatClient


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resource_text(name: str) -> str:
    return files("trajectory_bench").joinpath("resources", name).read_text(encoding="utf-8-sig")


def lock_run(output: Path, config: dict, selection: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    view = {"config": safe_config(config), "selection": selection,
            "rubric_sha256": digest(resource_text("rubric_v6.txt")),
            "package_version": __version__}
    code_dir = Path(__file__).parent
    view["method_files"] = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
          for name in ("native.py", "native_completion.py", "evidence.py", "grading.py", "workflow.py", "llm.py")}
    backend_cfg = config.get("native", {})
    if backend_cfg.get("backend_root") and selection.get("mode") == "native":
        backend_root = Path(backend_cfg["backend_root"])
        inputs = list((backend_root / "src" / "tau2").rglob("*.py"))
        data_root = Path(backend_cfg.get("data_dir", backend_root / "data"))
        inputs += [p for p in (data_root / "tau2").rglob("*")
                   if p.is_file() and p.suffix.lower() in (".json", ".toml", ".txt", ".md")]
        view["backend_inputs_sha256"] = digest([(str(p.relative_to(backend_root)) if p.is_relative_to(backend_root) else str(p),
                          hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)])
    fingerprint = digest(view)
    path = output / "manifest.json"
    if path.exists():
        old = load_json(path)
        if old.get("settings_sha256") != fingerprint:
            raise ValueError("Existing run has different settings/inputs; use a new output directory")
        return old
    manifest = dict(view, settings_sha256=fingerprint, created_at=timestamp(),
                    reference_scope="Native task environment and tool protocol; no cross-environment validation",
                    scoring="automatic model assessment; source review is a separate optional overlay")
    save_json(path, manifest)
    return manifest


def score_packet(case_id: str, packet: dict, cfg: dict, output: Path,
                 metadata: dict | None = None, client_factory: Callable = ChatClient) -> dict:
    from .grading import grade_annotation, parse_annotation

    case_dir = output / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    packet_path = case_dir / "evidence.json"
    packet_hash = digest(packet)
    if packet_path.exists() and digest(load_json(packet_path)) != packet_hash:
        raise ValueError(f"Saved evidence differs for {case_id}")
    save_json(packet_path, packet)
    annotation_path = case_dir / "automatic.json"
    if annotation_path.exists():
        saved = load_json(annotation_path)
        if saved.get("evidence_sha256") != packet_hash:
            raise ValueError(f"Saved score input differs for {case_id}")
        return saved
    if "judge" not in cfg:
        raise ValueError("Scoring requires a judge model configuration")
    source_status = packet.get("capture_integrity", {}).get("episode_status")
    if source_status is not None and source_status != "complete":
        record = grade_annotation(case_id, None, evidence=packet, metadata=metadata or {})
        record["technical_error"] = {"type": "IncompleteEpisode", "detail": "Source execution did not finish; no judge API call made"}
        record["evidence_sha256"] = packet_hash
        record["artifacts"] = {"evidence": f"cases/{case_id}/evidence.json"}
        save_json(annotation_path, record)
        return record
    messages = [{"role": "system", "content": resource_text("rubric_v6.txt")},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
    try:
        response = client_factory(cfg["judge"], case_dir / "api_judge").call(messages, request_id="score")
        raw = response["choices"][0]["message"].get("content") or ""
        save_json(case_dir / "judge_response.json", response)
        parsed = parse_annotation(raw, evidence=packet)
        record = grade_annotation(case_id, parsed, evidence=packet, metadata=metadata or {})
        record["response_parse"] = {k: v for k, v in parsed.items() if k != "parsed"}
        record["judge"] = {"provider": cfg["judge"].get("provider", "openai_compatible"),
                           "requested_model": cfg["judge"]["model"], "actual_model": response.get("model"),
                           "options": cfg["judge"].get("options", {})}
    except Exception as exc:
        # An invalid answer is not retried or relabelled a passing trajectory.
        record = grade_annotation(case_id, None, evidence=packet, metadata=metadata or {})
        record["technical_error"] = {"type": type(exc).__name__, "detail": str(exc)[:800]}
    record["evidence_sha256"] = packet_hash
    record["artifacts"] = {"evidence": f"cases/{case_id}/evidence.json",
                           "automatic": f"cases/{case_id}/automatic.json"}
    save_json(annotation_path, record)
    return record


def write_run_reports(records: list[dict], output: Path) -> dict:
    from .reporting import write_reports

    dest = output / "records.jsonl"
    temp = dest.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    temp.replace(dest)
    paths = write_reports(records, output)
    save_json(output / "completion.json", {"finished_at": timestamp(), "records": len(records),
              "report_paths": {k: str(v) for k, v in paths.items()}})
    return paths


def score_files(paths: list[Path], cfg: dict, output: Path) -> list[dict]:
    selection = {"mode": "import", "inputs": [{"path": str(p.resolve()),
                    "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()} for p in paths]}
    lock_run(output, cfg, selection)
    records = []
    for index, path in enumerate(paths, 1):
        case_id = f"import{index:04d}"
        packet = load_evidence(path)
        record = score_packet(case_id, packet, cfg, output,
                              {"domain": packet.get("domain"), "source_file": path.name})
        record["artifacts"]["source"] = str(path.resolve())
        save_json(output / "cases" / case_id / "automatic.json", record)
        records.append(record)
        write_run_reports(records, output)
    return records


def native_run(cfg: dict, output: Path, domain: str, task_ids: list[str],
               trials: int = 1, seed: int = 42, max_steps: int = 80, max_errors: int = 5,
               factory: str | None = None, capture_only: bool = False,
               _task_selection: list[tuple[str, str]] | None = None) -> list[dict]:
    from .native import NativeRunSpec, run_native
    from .native_completion import make_completion
    import importlib

    for role in ("user",) if factory else ("agent", "user"):
        if role not in cfg:
            raise ValueError(f"Native run needs {role} configuration")
    if not capture_only and "judge" not in cfg:
        raise ValueError("Native scoring run needs judge configuration")
    choices = _task_selection if _task_selection is not None else [(domain, tid) for tid in task_ids]
    selection = dict(mode="native", trials=trials, seed=seed,
                     max_steps=max_steps, max_errors=max_errors, factory=factory, capture_only=capture_only)
    selection.update({"tasks": choices} if _task_selection is not None else {"domain": domain, "task_ids": task_ids})
    loaded_factory = None
    if factory:
        from .native import prepare_backend
        native_cfg = cfg.get("native", {})
        prepare_backend(NativeRunSpec(domain=choices[0][0], task_ids=[choices[0][1]],
                        agent_model="bench-agent", user_model="bench-user",
                        backend_root=native_cfg.get("backend_root"), data_dir=native_cfg.get("data_dir")))
        module, sep, name = factory.partition(":")
        if not sep:
            raise ValueError("--agent-factory expects importable_module:factory_function")
        loaded_factory = getattr(importlib.import_module(module), name)
        module_path = getattr(importlib.import_module(module), "__file__", None)
        if module_path:
            selection["factory_source_sha256"] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    lock_run(output, cfg, selection)
    records = []
    for domain, task_id in choices:
        for repetition in range(1, trials + 1):
            case_id = f"run{len(records)+1:04d}"
            case_dir = output / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            raw_path = case_dir / "episode.json"
            metadata = {"domain": domain, "task_id": task_id, "repetition": repetition,
                        "agent_model": cfg.get("agent", {}).get("model", factory),
                        "model_alias": factory or cfg.get("agent", {}).get("model"),
                        "user_model": cfg["user"]["model"]}
            if raw_path.exists():
                raw = load_json(raw_path)
                if raw.get("status") != "complete":
                    raise RuntimeError(f"Saved native execution is incomplete at {case_id}; inspect episode.json (not graded)")
            else:
                # Distinct role aliases permit different providers using the same model name.
                completion = make_completion(cfg, case_dir / "api_business")
                native_cfg = cfg.get("native", {})
                spec = NativeRunSpec(domain=domain, task_ids=[task_id], agent_model="bench-agent",
                    user_model="bench-user", trials=1, seed=seed + repetition - 1,
                    max_steps=max_steps, max_errors=max_errors,
                    llm_args_agent=cfg.get("agent", {}).get("options", {}),
                    llm_args_user=cfg["user"].get("options", {}),
                    backend_root=native_cfg.get("backend_root"), data_dir=native_cfg.get("data_dir"))
                try:
                    episodes = run_native(spec, completion=completion, agent_factory=loaded_factory)
                    if len(episodes) != 1:
                        raise RuntimeError("Native runner must return exactly one episode per selected trial")
                    raw = episodes[0]
                    raw["benchmark_metadata"] = metadata
                    save_json(raw_path, raw)
                    if raw.get("status") != "complete":
                        raise RuntimeError(f"Native task execution failed: {raw.get('error', {}).get('type', 'see episode.json')}")
                except Exception as exc:
                    from .grading import grade_annotation
                    record = grade_annotation(case_id, None, metadata=metadata)
                    record["technical_error"] = {"type": type(exc).__name__, "detail": str(exc)[:800]}
                    save_json(case_dir / "run_error.json", record)
                    records.append(record)
                    write_run_reports(records, output)
                    # Fail fast: a broken adapter must not spend an entire suite budget.
                    raise RuntimeError(f"Native run failed at {case_id}; error and API receipts saved") from exc
            packet = compile_evidence(raw)
            if capture_only:
                save_json(case_dir / "evidence.json", packet)
                record = {"case_id": case_id, "status": "captured_unscored", "metadata": metadata}
            else:
                record = score_packet(case_id, packet, cfg, output, metadata)
            record.setdefault("artifacts", {})["source"] = f"cases/{case_id}/episode.json"
            records.append(record)
            write_run_reports(records, output)
    return records
