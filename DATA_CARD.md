# Data card

## Purpose and coverage

The reference collection supports process reliability evaluation of tool-using agents. It records whether an agent's actions and statements satisfied the business policy and the user's instructions at the time of each action.

There are **45 tasks**, with 15 each in retail, airline, and telecom. Each task was run three times with each of four agent configurations, giving **540 trajectories**. Source review identified **874 request groups**. These are 540 runs of 45 tasks, not 540 distinct tasks.

Tasks, synthetic business databases, tools, and policies originate from the bundled τ²-bench v2.0.0 snapshot. The selected task identifiers are in `reference_data/tasks/task_index.json`; the declared upstream commit is `989d3285ae669e49503324bdb9981bf14fccedeb`.

## How the records were produced

1. The task subset was selected by business category before the reported agent runs. The original environment, initial state, policy, and tools were retained.
2. The public LLMAgent ran each task three times using Qwen3.5-9B, Llama3.1-8B, DeepSeek Flash, or GLM-5.3-Flash. The user simulator was fixed to local Qwen 27B. Quantization and inference settings differ between configurations and are preserved in `paper_configuration.json`.
3. GLM-5.3-Flash initially grouped requests and assigned process judgments from the policy and actual interaction records. Hidden user instructions, original task scores, and historical reference judgments were not scoring inputs.
4. All 540 trajectories were checked against source evidence with assistant/model assistance. Initial judgments and reviewed annotations remain separate. Thirteen incomplete initial outputs were completed from the source records for the reference collection.

Reference review is **not independent human gold annotation or double annotation**. Among the 527 complete initial outputs, initial and reviewed trajectory verdicts agree in 459 cases (87.1%). This is agreement within the reviewed collection, not independent-test accuracy.

## Files and their intended use

| Path | Contents | Use |
|---|---|---|
| `reference_data/inputs/trNNN.json` | Policy, ordered interaction evidence, execution events, available mechanical facts | Import into `score`; inspect evidence |
| `reference_data/reference/annotations.jsonl` | Reviewed request groups, requirements, six dimension judgments, issues, and references | Reproduce reference statistics with `aggregate` |
| `reference_data/run_index.json` | Model/domain/task/repetition mapping, input hashes, original task score | Group results; compare with original evaluation after scoring |
| `reference_data/tasks/` | Selected upstream task records and task index | Run the environment and user simulator |
| `reference_data/policies/`, `reference_data/requirements/` | Original policies and policy-location indices | Locate business requirements |
| `reference_data/scoring/` | Historical rubric, annotation schema, aggregation helper | Inspect historical conventions |
| `reference_data/statistics/` | Frozen counts, including unknown and not-applicable counts | Reproduce the paper's summaries |
| `analysis_source/initial_scores540/` | First saved automatic scoring containers, including incomplete outputs | Compare initial assessments with reviewed judgments |
| `analysis_source/comparison_sources.json` | Experiment metadata and both original/published score-file hashes | Verify the comparison inputs |
| `examples/replay/retail100/` | A separate captured software-acceptance example | Exercise import and reporting without API calls |

The 540 reference trajectories are distributed as normalized evidence packets, not as the original full provider transport logs or a full database dump for every run. Do not assume that every intermediate database value is available. The saved scoring containers retain parsed answers, validation failures, settings, and usage metadata; private local file pointers were removed as described below.

Task files contain hidden user-simulation instructions and original evaluation conditions. They belong to the environment, **not to the process scorer's input**. Likewise, reference labels and original task scores must not be passed to a new scorer. The runner constructs source-only evidence for scoring.

## Judgments and aggregation

Each request group has six dimensions: `RULE`, `CONSENT`, `OUTCOME`, `REPORT`, `FACT`, and `STEPS`. Their meanings are in the main [README](README.md). Codes are:

- `pass`: the applicable requirements are supported as satisfied;
- `fail`: there is an evidenced substantive violation;
- `unknown`: missing evidence or a policy ambiguity prevents a definite judgment;
- `na`: no requirement for that dimension applies.

A violation takes precedence over unknown when combining dimensions into a group, and when combining groups into a trajectory. With no violation, an unknown makes the result undetermined. All applicable items must pass for the result to pass. An all-`na` result is kept separate.

Pass rates use **pass / (pass + fail)**. Unknown and not-applicable cases do not enter that denominator. A zero denominator produces no rate. Technical scoring failures are not ordinary unknown business judgments.

The reviewed reference totals are 291 passing, 220 violating, and 29 undetermined trajectories; 506 passing, 321 violating, and 47 undetermined request groups. The original task evaluation marked 340 runs successful, 66 of which have reviewed process violations. Original task success is a separate outcome label, not part of process scoring.

## Languages and format compatibility

Dialogue, policies, and tool records are in English. The frozen scoring prompt and most annotation explanations are in Chinese; JSON keys, dimension codes, and status codes are stable English identifiers. Historical annotations retain their original conventions. The current runtime checks a stricter output format than the original annotation protocol. See [docs/FORMAT_COMPATIBILITY.md](docs/FORMAT_COMPATIBILITY.md); a format difference is not by itself a new business error.

`reference_data/使用说明.txt` is retained as historical documentation. Its statement that the old data-only bundle lacked a client does not describe this repository, which includes `run_bench.py` and the current workflow. Use the root README and `docs/` for current instructions.

## Privacy and publication transformations

The task databases contain fictional people, accounts, addresses, telephone numbers, and payment identifiers used to simulate business operations. They are benchmark fixtures, not real customer transaction records. For example, the VPN address in telecom tool results is a simulated environment value, not a private experiment server address.

This publication copy excludes local API key files, credentials, execution environments, generated verification output, and redundant installation runs. All 540 reference evidence packets and reviewed annotations are unchanged from the validated delivery.

Two metadata-only transformations were applied to other files:

1. The `raw_path` field was removed from 540 saved automatic-score containers. It named a private local file that is not distributed. Parsed model answers, validation errors, and model settings were not changed.
2. Two machine-specific backend paths in the standalone replay episode were replaced with repository-relative paths. Its messages, tool events, and simulated state were not changed.

`analysis_source/comparison_sources.json` retains the original container hashes and supplies separate hashes for the published containers. The comparison reader verifies the published hashes. See [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) and `PUBLICATION_PROVENANCE.json`.

## Appropriate interpretation

This collection covers English text interactions in three simulated domains. It does not establish performance in arbitrary agent environments, real customer deployments, speech interfaces, or other languages. Reusing fixed database entities across tasks also differs from sampling independent real-world cases.

New runs produce automatic model judgments, while the reported reference results use reviewed judgments. Automatic scoring can be wrong; structural and citation checks do not establish that a business interpretation is correct. Keep output layers and technical failures separate when comparing systems.

The task configurations differ in model size, quantization, provider, and reasoning settings. Their results describe the tested configurations, not an isolated effect of model family. Multi-request and single-request subsets also differ in task/model composition; their pass-rate gap is not a controlled causal estimate of request count.

## License

See [DATA_LICENSE.md](DATA_LICENSE.md) and [NOTICE.md](NOTICE.md). Preserve upstream attribution when redistributing task materials.
