# Trajectory Process Benchmark

[Source repository](https://github.com/JunhaoChen-Buaa/trajectory-process-bench) · [Download v0.1.1](https://github.com/JunhaoChen-Buaa/trajectory-process-bench/archive/refs/tags/v0.1.1.zip)

**Evaluate whether an agent followed the user's instructions and business rules, as well as whether it completed the task.**

This repository accompanies *A Benchmark for Evaluating Agent Process Reliability from Interaction Trajectories*. It groups the requests within a task, links each request to dialogue and execution evidence, and reports process judgments for individual requests and the complete interaction.

The release includes a runner, scoring and reporting software, 45 tasks across retail, airline, and telecom, and 540 reference trajectories. Task environments, tools, and business policies come from a bundled, fixed [τ²-bench](https://github.com/sierra-research/tau2-bench) snapshot. Our additions are request grouping, six process evaluation dimensions, hierarchical aggregation, and the evaluation workflow. See [NOTICE.md](NOTICE.md) for attribution and [DATA_CARD.md](DATA_CARD.md) for data provenance and scope.

## Choose a starting point

| Your goal | Start here | Model calls |
|---|---|---|
| Reproduce the paper's reference totals | Section 1 | None |
| Run your model with the public LLMAgent | Sections 3–4 | Agent, simulated user, scorer |
| Evaluate your own agent workflow | Section 5 | Your agent, simulated user, scorer |
| Score an existing trajectory | Section 7 | Scorer only |

## 1. Reproduce reference results

From the repository root, using Python 3.12 or 3.13:

```console
python -X utf8 run_bench.py aggregate --annotations reference_data/reference/annotations.jsonl --output results/reference540.json
```

This command uses the standard library and makes no API calls. Expected totals:

| Evaluation unit | Pass | Violation | Undetermined | Total |
|---|---:|---:|---:|---:|
| Trajectories | 291 | 220 | 29 | 540 |
| Request groups | 506 | 321 | 47 | 874 |

These are reviewed reference judgments, not fresh automatic scores. The reference review was assistant/model-assisted source-record review, not independent human double annotation.

To reproduce the comparison between initial automatic scores and reviewed judgments:

```console
python -X utf8 analysis/score_comparison/compare_scores.py --raw-scores analysis_source/initial_scores540 --reference reference_data/reference/annotations.jsonl --run-index reference_data/run_index.json --out results/score_comparison
```

The paired comparison contains 527 complete initial outputs; 459 trajectory verdicts agree with the reviewed verdicts. The other 13 initial outputs are reported separately as incomplete. Neither command reruns an agent or a scoring model.

## 2. Install dependencies for task execution

Python **3.12 or 3.13** is required. Run these commands from the repository root in an isolated environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\Activate.ps1
```

If PowerShell does not allow activation, use `.\.venv\Scripts\python.exe` instead of `python` in subsequent commands; activation is optional.

**Linux or macOS**

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
. .venv/bin/activate
```

The bundled upstream snapshot is installed in editable mode. Voice extras and model weights are not needed. For Python 3.13, `requirements.txt` also installs `audioop-lts`, which the upstream imports require. [TESTED_ENVIRONMENT.json](TESTED_ENVIRONMENT.json) records a previously tested dependency set; see [docs/VERIFICATION.md](docs/VERIFICATION.md) for checks on this publication copy.

## 3. Configure model services

Copy `examples/config.example.json` to `examples/my_config.local.json`. Set the actual service URL and model name for each role:

| Role | Purpose |
|---|---|
| `agent` | The model used by the agent being evaluated |
| `user` | The model that simulates the user |
| `judge` | The model that assigns process judgments |

The clients use OpenAI-compatible endpoints. Each role can use a different provider. Replace all placeholder URLs and model names in the template. Supply credentials through environment variables:

```powershell
# PowerShell: use your own credentials; do not commit their values.
$env:AGENT_API_KEY = "your-agent-key"
$env:USER_API_KEY = "your-user-key"
$env:JUDGE_API_KEY = "your-scorer-key"
```

For POSIX shells, use `export AGENT_API_KEY="..."` and the corresponding user/scorer variables. Alternatively, replace a role's `api_key_env` with `api_key_file` pointing to a private, one-line credential file.

Keep `native.backend_root` and `native.data_dir` as provided when using the bundled environment. Configuration paths are relative to the configuration file. Match reasoning and other provider-specific options to the service you use.

For a fair agent comparison, keep the task set, user simulator, scorer, scoring rubric, and execution budgets fixed. [paper_configuration.json](paper_configuration.json) describes the historical experiment; it is not a ready-to-run credential configuration.

## 4. Run a task

This command runs airline task 21 with the public LLMAgent, saves the interaction, scores the process, and writes a report:

```console
python -X utf8 run_bench.py run --config examples/my_config.local.json --domain airline --task-id 21 --max-steps 200 --max-errors 10 --output results/airline21
```

Open **`results/airline21/report.html`** after completion. A process violation is an evaluation result, not a program failure. Technical execution or scoring failures are retained separately.

The `run` and `suite` commands call paid model services. To save trajectories without invoking the scorer, add `--capture-only` and score them later.

## 5. Connect your own agent

To compare underlying LLMs, change the `agent` configuration and use the command above. To evaluate a different workflow, copy `examples/stateful_agent.py` to `my_agent.py` and replace its decision logic. The example maintains its own conversation state; it does not delegate the workflow to LLMAgent.

| Interface | Contract |
|---|---|
| `build_agent(*, tools, domain_policy, llm, llm_args)` | Return an agent using the supplied tools and policy. |
| `get_init_state(message_history=None)` | Create state for a new interaction. |
| `generate_next_message(message, state)` | Return `(AssistantMessage, next_state)` after a user message or tool result. |

Use `AssistantMessage.content` for a user-facing reply and `tool_calls` for structured tool requests. The environment executes the requested tools and returns the results. Handle `MultiToolMessage` when several results arrive together. Hidden user goals and reference answers are not passed to your agent factory.

```console
python -X utf8 run_bench.py run --config examples/my_config.local.json --domain airline --task-id 21 --agent-factory my_agent:build_agent --max-steps 200 --max-errors 10 --output results/my_agent_airline21
```

See [docs/CUSTOM_AGENT.md](docs/CUSTOM_AGENT.md) for message types, tool-call structure, and recording responsibilities. Test a single task before running the complete suite.

## 6. Run the fixed task suite

```console
python -X utf8 run_bench.py suite --config examples/my_config.local.json --agent-factory my_agent:build_agent --trials 3 --seed 20261105 --max-steps 200 --max-errors 10 --output results/my_agent_paper45
```

This produces **135 runs**: 45 tasks × 3 trials. Each run starts from a fresh environment. Omit `--agent-factory` to use LLMAgent. `python -X utf8 run_bench.py tasks` lists the tasks; `suite --limit 1` is available for an installation smoke test.

## 7. Score existing trajectories

```console
python -X utf8 run_bench.py score --config examples/my_config.local.json --input my_episode.json --output results/imported
```

Both captured `episode.json` files and normalized evidence packets such as `reference_data/inputs/tr001.json` are accepted. This command calls only the scorer. For another framework, preserve the policy, ordered messages, structured tool requests and responses, who performed each action, and available execution/state evidence. A chat transcript alone cannot establish every process requirement. See [docs/FORMAT.md](docs/FORMAT.md).

## 8. Read the scores and evidence

The scorer groups independently assessable requests and evaluates six dimensions:

| Code | Dimension | What is assessed |
|---|---|---|
| `RULE` | Rules and targets | Whether attempted actions and their targets satisfy the business policy |
| `CONSENT` | User consent | Whether required approval was obtained before the action |
| `OUTCOME` | Execution outcome | Whether the actual result is correct, including unintended changes |
| `REPORT` | Completion claims | Whether claims of completion are supported by execution evidence |
| `FACT` | Factual statements | Whether consequential prices, states, and explanations are accurate |
| `STEPS` | Required steps | Whether prescribed checks, actions, and their order were followed |

Each dimension is `pass`, `fail` (violation), `unknown` (undetermined), or `na` (not applicable). A request group is a violation if any applicable dimension fails; otherwise it is undetermined if any is unknown. It passes when all applicable dimensions pass. Groups are combined in the same way for the trajectory. All-`na` cases are reported separately.

**Pass rate = pass / (pass + violation).** Undetermined, not-applicable, unreviewed, and technical-failure counts are reported separately. A prohibited attempt may be a violation even when a tool rejects it; it must not be described as an executed change. Recovery is recorded without erasing an earlier substantive violation.

| Output | Contents |
|---|---|
| `report.html` | Summary, request-group judgments, and evidence links |
| `report.json`, `records.jsonl` | Machine-readable scores and provenance |
| `cases.csv`, `group_scores.csv`, `dimensions.csv` | Tables for analysis |
| `cases/run0001/episode.json` | Captured interaction and execution records |
| `cases/run0001/evidence.json` | Evidence supplied to the scorer |

`h:N` identifies a message; `e:N` identifies an execution event. User-facing claims include only replies actually delivered to the user. User-side telecom actions are kept separate from agent actions.

The `automatic` layer preserves the model's assessment. Explicit source-review decisions can be added in a separate `reviewed` layer; unreviewed cases are not silently treated as reviewed. Reference judgments and original task scores are excluded from scoring inputs. The runner does not invoke the original task evaluator by default.

The task conversations are in English. Historical annotation explanations, the frozen scoring prompt, and report labels are in Chinese, with stable English JSON keys/status codes and English evidence quotations. They are retained to preserve the reported evaluation configuration. This README explains the codes in English; no translated prompt is represented as a new validated scorer.

## 9. Verify the package without API calls

After installing dependencies:

```console
python -X utf8 verify_release.py --tests
python -X utf8 tools/check_public_files.py
```

The verifier blocks network connections. It checks task loading, reference hashes and aggregation, trajectory conversion, reports, review overlays, and scripted agent/tool interfaces. Scripted tests are software checks, not new model-performance experiments.

Generate a report from a saved real interaction without new model calls:

```console
python -X utf8 run_bench.py report --records examples/replay/retail100/records.jsonl --reviews examples/replay/retail100/source_reviews.jsonl --output results/replay_report
```

This replay is a software acceptance example with different model settings; it is not part of the 540-trajectory experiment.

## Repository layout

```text
run_bench.py                 Task execution, scoring, and reporting
src/trajectory_bench/        Runtime, frozen rubric, and aggregation
examples/                   Credential-free configuration and agent examples
docs/                       Agent interface, data formats, and verification
reference_data/              45 task records, 540 evidence packets, reviewed annotations
analysis_source/             Initial automatic scores; local file pointers removed
analysis/score_comparison/   Initial-versus-reviewed comparison scripts
vendor/tau2-bench/           Upstream source, task data, and MIT license
tools/check_public_files.py  Checks for credential/path publication mistakes
```

See [DATA_CARD.md](DATA_CARD.md) for provenance and limitations, [DATA_LICENSE.md](DATA_LICENSE.md) for license scope, and [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) for publication-only changes. Historical reference files retain their original fields and notes; current usage instructions are this README and `docs/`.

## License and citation

Project contributions are released under the [MIT License](LICENSE). Upstream files retain their original licenses and notices. No model weights are included.

Citation metadata is in [CITATION.cff](CITATION.cff). This is the v0.1.1 software release accompanying the manuscript; no DOI or publication acceptance is claimed.
