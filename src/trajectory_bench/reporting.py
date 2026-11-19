"""Offline JSON/CSV/HTML reports for automatic and reviewed score layers."""
from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .grading import DIMENSIONS, aggregate_records, verdict

DIMENSION_LABELS = {
    "RULE": "业务规则与对象", "CONSENT": "用户确认", "OUTCOME": "实际办理结果",
    "REPORT": "完成说明", "FACT": "业务事实说明", "STEPS": "必要步骤",
}
STATUS_LABELS = {"pass": "通过", "fail": "违反", "unknown": "待定", "na": "不适用",
                 "technical_unscored": "技术失败，未评分", "captured_unscored": "仅采集，未请求评分", "scored": "已评分", None: "未复核"}
LAYER_LABELS = {"automatic": "自动初评", "reviewed": "复核后评分"}


def _original_comparison(records: list[Mapping[str, Any]], layer: str) -> dict[str, Any]:
    result = {"metadata_available": 0, "original_success": 0, "original_failure": 0,
              "original_success_process": {s: 0 for s in ("pass", "fail", "unknown", "na", "technical_unscored", "captured_unscored", "unreviewed")}}
    for record in records:
        metadata = record.get("metadata", {})
        reward = metadata.get("original_reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            continue
        result["metadata_available"] += 1
        if reward == 1:
            result["original_success"] += 1
            entry = record.get(layer)
            if entry is None:
                state = "unreviewed" if layer == "reviewed" else "captured_unscored" if record.get("status") == "captured_unscored" else "technical_unscored"
            else:
                state = entry.get("verdict") if entry.get("status") == "scored" else "technical_unscored"
            result["original_success_process"][state] += 1
        else:
            result["original_failure"] += 1
    denominator = result["original_success"]
    result["original_success_violation_rate"] = result["original_success_process"]["fail"] / denominator if denominator else None
    result["note"] = "原任务分数仅来自独立元数据，不进入过程判分；该比例以有原成功元数据的全部轨迹为分母，技术失败和未复核单列。"
    return result


def build_report(records: Iterable[Mapping[str, Any]], *, title: str = "交互轨迹过程可靠性评测报告") -> dict[str, Any]:
    rows = list(records)
    layers = {layer: aggregate_records(rows, layer) for layer in LAYER_LABELS}
    for layer, summary in layers.items():
        summary["original_result_comparison"] = _original_comparison(rows, layer)
    return {
        "schema_version": "1.0", "title": title,
        "aggregation_rule": "分项→业务组→轨迹：违反优先，其次待定；其余适用项通过才通过；全部不适用则不适用。分项按轨迹跨组汇总，不作等权平均。",
        "rate_denominator": "通过率 = 通过 / (通过 + 违反)；待定、不适用、技术失败和未复核均单列。",
        "evidence_limit": "程序只核对格式、引用存在性及显式提供的原文短引。引用存在不等于业务判断成立，源证据支持仍需复核。",
        "review_limit": "复核层只统计提供了明确复核决定的记录，不自动混入初评分；复核者、时间、原因和原始评分均保留，不自动称独立人工金标准。",
        "layers": layers, "records": rows,
    }


def _rate(value: Any) -> str:
    return "—" if value is None else f"{value:.1%}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _badge(state: str | None) -> str:
    return f'<span class="badge {_escape(state or "absent")}">{_escape(STATUS_LABELS.get(state, state))}</span>'


def _counts_table(summary: Mapping[str, Any]) -> str:
    lines = ['<table><thead><tr><th>统计对象</th><th>总数</th><th>通过</th><th>违反</th><th>待定</th><th>不适用</th><th>有效分母</th><th>通过率</th></tr></thead><tbody>']
    counts = [("轨迹", summary["trajectories"]), ("业务组", summary["business_groups"])]
    counts.extend((DIMENSION_LABELS[d], summary["dimensions"][d]) for d in DIMENSIONS)
    for label, item in counts:
        denominator = item["pass"] + item["fail"]
        lines.append(f'<tr><th>{label}</th>' + "".join(f'<td>{item[k]}</td>' for k in ("n", "pass", "fail", "unknown", "na")) + f'<td>{denominator}</td><td>{_rate(item["pass_rate_decided"])}</td></tr>')
    return "".join(lines) + "</tbody></table>"


def _safe_artifacts(artifacts: Any) -> str:
    if not isinstance(artifacts, Mapping):
        return ""
    links = []
    for label, raw in artifacts.items():
        if not isinstance(raw, (str, Path)):
            continue
        value = str(raw)
        # No annotation-supplied javascript/data links in a standalone report.
        windows_path = len(value) > 2 and value[1] == ":" and value[2] in "\\/"
        scheme = urlsplit(value).scheme.lower()
        if scheme and scheme not in ("http", "https", "file") and not windows_path:
            continue
        if windows_path:
            value = "file:///" + value.replace("\\", "/")
        elif Path(value).is_absolute() and not scheme:
            value = Path(value).as_uri()
        links.append(f'<a href="{_escape(value)}">{_escape(label)}</a>')
    return '<p class="artifacts">记录文件：' + " · ".join(links) + "</p>" if links else ""


def _breakdowns_html(summary: Mapping[str, Any]) -> str:
    chunks = ["<details><summary>按模型、场景与业务数量汇总</summary>"]
    labels = {"model_alias": "模型", "domain": "场景", "structure": "业务数量"}
    names = {"single": "单业务", "multiple": "多业务", "unspecified": "未记录"}
    for key in ("model_alias", "domain", "structure"):
        chunks.append(f'<h4>{labels[key]}</h4><table><thead><tr><th>分组</th><th>已评分轨迹</th><th>通过</th><th>违反</th><th>待定</th><th>不适用</th><th>有效分母</th><th>通过率</th></tr></thead><tbody>')
        for name, values in summary["breakdowns"][key].items():
            item = values["trajectories"]
            chunks.append(f'<tr><th>{_escape(names.get(name, name))}</th>' + "".join(f'<td>{item[k]}</td>' for k in ("n", "pass", "fail", "unknown", "na", "valid_denominator")) + f'<td>{_rate(item["pass_rate_decided"])}</td></tr>')
        chunks.append("</tbody></table>")
    return "".join(chunks) + "</details>"


def _annotation_html(entry: Mapping[str, Any] | None) -> str:
    if entry is None:
        return "<p>尚未提供复核决定。</p>"
    chunks = [f'<p>{_badge(entry.get("verdict") if entry.get("status") == "scored" else "technical_unscored")}</p>']
    provenance = entry.get("provenance")
    if provenance:
        chunks.append('<p class="muted">复核记录：' + _escape(json.dumps(provenance, ensure_ascii=False, sort_keys=True)) + "</p>")
    validation = entry.get("validation", {})
    if validation.get("errors") or validation.get("warnings"):
        chunks.append('<details class="flags"><summary>格式与证据标记</summary><ul>')
        for kind in ("errors", "warnings"):
            chunks.extend(f'<li>{_escape(kind)}：{_escape(flag)}</li>' for flag in validation.get(kind, []))
        chunks.append("</ul></details>")
    annotation = entry.get("annotation")
    if not isinstance(annotation, Mapping):
        return "".join(chunks)
    groups = annotation.get("groups", [])
    if not isinstance(groups, list):
        groups = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        chunks.append(f'<h4>{_escape(group.get("group_id", ""))} {_escape(group.get("goal", ""))}</h4><p>主要对象：{_escape(group.get("object", ""))}</p>')
        if group.get("end_state"):
            chunks.append(f'<p>实际状态：{_escape(group["end_state"])}</p>')
        chunks.append('<table class="scores"><thead><tr><th>方面</th><th>判断</th><th>理由与引用</th></tr></thead><tbody>')
        scores = group.get("scores", {})
        if not isinstance(scores, Mapping):
            scores = {}
        for dim in DIMENSIONS:
            score = scores.get(dim, {})
            if not isinstance(score, Mapping):
                score = {}
            citations = score.get("refs", [])
            if not isinstance(citations, list):
                citations = []
            chunks.append(f'<tr><th>{DIMENSION_LABELS[dim]}<small>{dim}</small></th><td>{_badge(score.get("status"))}</td><td>{_escape(score.get("reason", "缺少该项"))}<div class="refs">引用：{_escape("、".join(map(str, citations)) or "无")}</div></td></tr>')
        chunks.append("</tbody></table>")
    issues = annotation.get("issues", [])
    if isinstance(issues, list) and issues:
        chunks.append('<h4>问题依据与后果</h4><ul>')
        for issue in issues:
            if isinstance(issue, Mapping):
                chunks.append(f'<li><strong>{_escape(issue.get("issue_id", ""))} {_escape(issue.get("title", ""))}</strong>（{_escape(issue.get("severity", ""))}；原因：{_escape(issue.get("root_cause", ""))}；已恢复：{_escape(issue.get("recovered", ""))}）<p>依据：{_escape(issue.get("basis", ""))}</p><p>后果：{_escape(issue.get("effect", ""))}</p><p class="refs">引用：{_escape(issue.get("refs", []))}</p></li>')
        chunks.append("</ul>")
    return "".join(chunks)


def render_html(report: Mapping[str, Any]) -> str:
    """Render one self-contained Chinese report; all supplied text is escaped."""
    chunks = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
              f'<title>{_escape(report["title"])}</title>',
              '<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1160px;margin:auto;padding:32px;line-height:1.65;color:#172437;background:#f7f9fc}h1{font-size:1.9rem}h2{margin-top:2rem}table{border-collapse:collapse;width:100%;background:white;margin:1rem 0}td,th{border:1px solid #d7dfe9;padding:.55rem;text-align:left;vertical-align:top}thead{background:#eaf0f7}small,.muted,.refs{display:block;color:#526277;font-size:.88rem}.badge{display:inline-block;border-radius:5px;padding:.1rem .5rem;white-space:nowrap;background:#e8edf3}.pass{background:#daf0e5;color:#174c35}.fail{background:#fbe0df;color:#882723}.unknown{background:#fff0ca;color:#775600}.technical_unscored{background:#ecdff4;color:#603778}.card,details.case{background:white;border:1px solid #d7dfe9;border-radius:9px;padding:1rem 1.2rem;margin:1rem 0}summary{cursor:pointer;font-weight:600}.scores th:first-child{width:12%}.scores td:nth-child(2){width:10%}code,pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1759a5}.flags{margin:.5rem 0}.artifacts{font-size:.9rem}@media(max-width:700px){body{padding:12px}table{font-size:.84rem}td,th{padding:.3rem}}@media print{body{background:white;padding:0}details.case{break-inside:avoid}summary{display:block}}</style></head><body>',
              f'<h1>{_escape(report["title"])}</h1><p>{_escape(report["aggregation_rule"])}</p><p>{_escape(report["rate_denominator"])}</p><p class="muted">{_escape(report["evidence_limit"])} {_escape(report["review_limit"])}</p>']
    for layer, summary in report["layers"].items():
        chunks.append(f'<section><h2>{LAYER_LABELS[layer]}</h2><div class="card"><p>共{summary["cases_total"]}条；已评分{summary["scored_cases"]}条，技术失败未评分{summary["technical_unscored"]}条，仅采集未评分{summary["captured_unscored"]}条，未提供本层评分{summary["unreviewed"]}条。证据标记涉及{summary["evidence_warning_cases"]}条；声明完成源复核{summary["source_reviewed_cases"]}条。</p>')
        chunks.append(_counts_table(summary))
        chunks.append(_breakdowns_html(summary))
        comparison = summary["original_result_comparison"]
        if comparison["metadata_available"]:
            counts = comparison["original_success_process"]
            chunks.append(f'<p>原任务成功{comparison["original_success"]}条：过程通过{counts["pass"]}、违反{counts["fail"]}、待定{counts["unknown"]}、不适用{counts["na"]}、技术失败{counts["technical_unscored"]}、仅采集{counts["captured_unscored"]}、未复核{counts["unreviewed"]}。原成功中的过程违反比例为{_rate(comparison["original_success_violation_rate"])}。</p>')
        chunks.append("</div></section>")
    chunks.append("<h2>逐条记录与证据索引</h2><p>展开查看两层评分。自动初评的违反是候选判断；格式或引用检查通过，不代表已确认业务问题。</p>")
    for record in report["records"]:
        meta = record.get("metadata", {})
        label = " / ".join(str(meta[k]) for k in ("domain", "model_alias", "task_id", "repetition") if k in meta)
        chunks.append(f'<details class="case"><summary>{_escape(record["case_id"])} {_escape(label)}</summary>')
        chunks.append(_safe_artifacts(record.get("artifacts")))
        for layer in LAYER_LABELS:
            chunks.append(f'<h3>{LAYER_LABELS[layer]}</h3>')
            if layer == "automatic" and record.get("status") == "captured_unscored":
                chunks.append("<p>仅采集原始轨迹，未请求过程评分。</p>")
            else:
                chunks.append(_annotation_html(record.get(layer)))
        raw_text = record.get("raw_text")
        if raw_text is None:
            raw_text = record.get("response_parse", {}).get("raw_text")
        if raw_text is not None:
            chunks.append(f'<details><summary>原始模型输出（保留原样）</summary><pre>{_escape(raw_text)}</pre></details>')
        chunks.append("</details>")
    chunks.append("</body></html>")
    return "".join(chunks)


def _cell(value: Any) -> Any:
    """Avoid spreadsheet formula execution when users open the CSV."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in fieldnames})


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [f'# {report["title"]}', "", report["aggregation_rule"], "", report["rate_denominator"], "", report["evidence_limit"], "", report["review_limit"]]
    for layer, summary in report["layers"].items():
        lines.extend(["", f'## {LAYER_LABELS[layer]}', "", f'共{summary["cases_total"]}条；已评分{summary["scored_cases"]}，技术失败{summary["technical_unscored"]}，仅采集未评分{summary["captured_unscored"]}，未复核{summary["unreviewed"]}。', "", "| 对象 | 通过 | 违反 | 待定 | 不适用 | 有效分母 | 通过率 |", "|---|---:|---:|---:|---:|---:|---:|"])
        items = [("轨迹", summary["trajectories"]), ("业务组", summary["business_groups"])] + [(DIMENSION_LABELS[d], summary["dimensions"][d]) for d in DIMENSIONS]
        lines.extend(f'| {label} | {item["pass"]} | {item["fail"]} | {item["unknown"]} | {item["na"]} | {item["pass"] + item["fail"]} | {_rate(item["pass_rate_decided"])} |' for label, item in items)
    lines.extend(["", "逐条证据、复核来源和原始输出见 report.html 与 records.jsonl。", ""])
    return "\n".join(lines)


def write_reports(records: Iterable[Mapping[str, Any]], out_dir: str | Path, include_markdown: bool = False,
                  *, title: str = "交互轨迹过程可靠性评测报告", layer: str | None = None) -> dict[str, str]:
    """Write reusable artifacts; the two score layers always stay explicit.

    ``layer`` is accepted to support callers selecting their principal layer;
    both layers remain in the deliverables so reviews cannot erase raw scores.
    """
    if layer not in (None, "automatic", "reviewed"):
        raise ValueError("layer must be automatic, reviewed, or None")
    rows = list(records)
    report = build_report(rows, title=title)
    report["principal_layer"] = layer
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {"json": target / "report.json", "csv": target / "cases.csv", "groups_csv": target / "group_scores.csv",
             "dimensions_csv": target / "dimensions.csv", "html": target / "report.html", "records": target / "records.jsonl"}
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["records"].write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rows), encoding="utf-8")
    case_rows = []
    group_rows = []
    for record in rows:
        meta = record.get("metadata", {})
        case_row = {"case_id": record["case_id"], **{key: meta.get(key, "") for key in ("domain", "model_alias", "task_id", "repetition", "original_reward")}}
        for score_layer in LAYER_LABELS:
            entry = record.get(score_layer)
            case_row[f"{score_layer}_status"] = ("unreviewed" if score_layer == "reviewed" else "captured_unscored" if record.get("status") == "captured_unscored" else "technical_unscored") if entry is None else entry["status"]
            case_row[f"{score_layer}_verdict"] = entry.get("verdict", "") if entry else ""
            validation = entry.get("validation", {}) if entry else {}
            case_row[f"{score_layer}_errors"] = validation.get("errors", [])
            case_row[f"{score_layer}_warnings"] = validation.get("warnings", [])
            case_row[f"{score_layer}_provenance"] = entry.get("provenance", {}) if entry else {}
            annotation = entry.get("annotation") if entry else None
            if not isinstance(annotation, Mapping) or not isinstance(annotation.get("groups"), list):
                continue
            for group in annotation["groups"]:
                if not isinstance(group, Mapping) or not isinstance(group.get("scores"), Mapping):
                    continue
                for dim in DIMENSIONS:
                    score = group["scores"].get(dim, {})
                    if isinstance(score, Mapping):
                        group_rows.append({"case_id": record["case_id"], "layer": score_layer, "scoring_status": entry["status"],
                                           "group_id": group.get("group_id", ""), "goal": group.get("goal", ""), "object": group.get("object", ""),
                                           "dimension": dim, **{k: score.get(k, "") for k in ("status", "reason", "refs", "requirement_ids", "issue_ids")}})
        case_rows.append(case_row)
    case_fields = ["case_id", "domain", "model_alias", "task_id", "repetition", "original_reward"]
    case_fields += [f"{score_layer}_{key}" for score_layer in LAYER_LABELS for key in ("status", "verdict", "errors", "warnings", "provenance")]
    _write_csv(paths["csv"], case_fields, case_rows)
    _write_csv(paths["groups_csv"], ["case_id", "layer", "scoring_status", "group_id", "goal", "object", "dimension", "status", "reason", "refs", "requirement_ids", "issue_ids"], group_rows)
    dimension_rows = []
    for score_layer, summary in report["layers"].items():
        for dim, counts in summary["dimensions"].items():
            dimension_rows.append({"layer": score_layer, "dimension": dim, **counts})
    _write_csv(paths["dimensions_csv"], ["layer", "dimension", "n", "pass", "fail", "unknown", "na", "valid_denominator", "pass_rate_decided"], dimension_rows)
    paths["html"].write_text(render_html(report), encoding="utf-8")
    if include_markdown:
        paths["markdown"] = target / "report.md"
        paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    return {key: str(path.resolve()) for key, path in paths.items()}
