# Trajectory, score, and review formats

## Captured episodes

The runner saves `episode.json` with:

| Field | Meaning |
|---|---|
| `schema_version` | `trajectory_bench.native_episode.v1` |
| `status` | `complete` or `error`: software execution status, not a process verdict |
| `domain`, `task_id` | Task identifier |
| `agent_policy`, `tools`, `user_tools` | Policy and available tool definitions |
| `messages` | Ordered messages; `customer_visible` distinguishes delivered agent replies from internal tool-call prefixes |
| `tool_events` | Tool name, call identifier, actor, arguments, actual response/error, and available state differences |
| `initial_state`, `final_state` | Captured state for source review; the full database is not automatically passed to the scorer |
| `capture_integrity` | Pairing, missing-record, and capture diagnostics |

## Importing another framework

`score` accepts captured episodes and normalized evidence packets. A minimal generic episode is shown below. Values and tool names are illustrative, not a task to be solved or an expected answer.

```json
{
  "domain": "retail",
  "policy": "Obtain user confirmation before cancelling an order.",
  "available_tool_schemas": [],
  "messages": [
    {"role": "user", "content": "Cancel order A."},
    {"role": "assistant", "content": "May I cancel order A?", "customer_visible": true},
    {"role": "user", "content": "Yes."},
    {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "cancel_order", "arguments": "{\"order_id\":\"A\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "requestor": "assistant", "content": "{\"status\":\"cancelled\"}"},
    {"role": "assistant", "content": "Order A has been cancelled.", "customer_visible": true}
  ],
  "tool_events": [
    {"tool_call_id": "c1", "name": "cancel_order", "requestor": "assistant", "arguments": {"order_id": "A"}, "response": {"status": "cancelled"}}
  ]
}
```

For real use, include the actual tool schemas and available state evidence. A response alone provides only the evidence it contains; the converter does not invent a successful database write. Preserve failed and rejected requests, as well as successful ones. Distinguish user-side actions from agent-side actions.

Normalized evidence uses `h:N` for messages and `e:N` for execution events. Original task scores, hidden user instructions, and reference labels are excluded from scoring inputs. Internal text not delivered to the user is excluded when assessing public claims.

## Automatic annotations

`groups` lists independently assessable requests. Each group records a goal, object, request locations, applicable requirements, and six entries in `scores`. Each score has `status`, `reason`, `refs`, `requirement_ids`, and `issue_ids`.

The status codes are `pass`, `fail`, `unknown`, and `na`. A `fail` must identify a substantive violation supported by source references. A single issue can affect several dimensions but should use one issue identifier.

`issues` records the requirement, supporting source, cause attribution, consequence, and recovery. Minor errors and pure interface-format issues are recorded separately in `auxiliary_notes`. A structural or reference check passing does not establish that the model's interpretation is correct. Raw answers and validation errors are retained.

Historical reference annotations and current runtime outputs have different conventions in a few optional fields. See [FORMAT_COMPATIBILITY.md](FORMAT_COMPATIBILITY.md) before importing annotations between them.

## Separate source-review layer

The `automatic` field preserves the initial assessment. `reviewed` is added only when an explicit review decision is supplied. Review decisions are JSONL records, for example:

```json
{"case_id":"run0001","changes":[],"reviewer":"reviewer-name","reviewed_at":"2026-11-19T12:00:00+08:00","reason":"Checked the user confirmation and execution records; no change needed.","review_type":"model_assisted_source_review","source_reviewed":true}
```

An unchanged judgment still needs an explicit review record. For a complete revision, provide the full `annotation`; an empty set of groups is not a pass. A partial revision uses `changes` with `group_id`, `dimension`, and replacement score fields. Supply complete `issues` when adding or changing issues.

```console
python -X utf8 run_bench.py report --records results/run/records.jsonl --reviews reviews.jsonl --output results/reviewed
```

The report command checks the evidence hash used by a review. Initial outputs are not overwritten, and unreviewed records are not silently added to reviewed totals. The reviewer's identity, review type, time, and reason remain in provenance.

## Output files

`manifest.json` stores runtime/rubric settings. Each case has its captured `episode.json`, scorer-facing `evidence.json`, raw scoring response, and `automatic.json`. Business-model and scorer requests are saved in `api_business/` and `api_judge/`. Summary outputs include `records.jsonl`, `report.json`, `cases.csv`, `group_scores.csv`, `dimensions.csv`, and `report.html`.

New runs can contain credentials or private task data supplied by their users. Review those outputs before redistributing them. No successful model answer is resampled merely because the judgment is unfavorable; technical scoring failures remain separate from unknown business judgments.
