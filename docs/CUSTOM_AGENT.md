# Connect an agent workflow

There are two ways to evaluate an agent:

- Change the `agent` model settings and use the public LLMAgent.
- Implement your own workflow and load it with `--agent-factory module:build_agent`.

The second option is described here. It uses the same task environment, tools, user simulator, capture, and scoring as the default agent.

## Start from the example

Copy `examples/stateful_agent.py` to `my_agent.py` in the repository root. The example maintains its own history and chooses the next response through the configured model client. It does not call LLMAgent internally or read reference labels.

Replace its planning, memory, or tool-selection logic with your own implementation. Keep these methods:

```python
def build_agent(*, tools, domain_policy, llm, llm_args):
    # Return an instance implementing the methods below.
    ...

def get_init_state(self, message_history=None):
    # New state for one interaction.
    ...

def generate_next_message(self, message, state):
    # Return an AssistantMessage and the updated state.
    return reply, next_state
```

The factory receives keyword arguments. `tools` describes the available tools, `domain_policy` is the task's business policy, and `llm`/`llm_args` describe the configured model service. The factory does not receive the hidden user goal or expected actions.

## Message and tool protocol

Use message classes from `tau2.data_model.message`:

- A normal user message or `ToolMessage` is supplied through `message`.
- A `MultiToolMessage` contains several `tool_messages`; preserve all of them in the state.
- Return an `AssistantMessage` with `content` to speak to the user.
- Return structured `tool_calls` to request tools. Each `ToolCall` has an `id`, `name`, and argument dictionary.

The environment executes tool requests and returns their results on subsequent calls. Do not directly execute environment tools inside the agent: that can bypass capture and lose the ordering needed for process evaluation. Follow the environment's half-duplex interaction protocol.

The bundled example uses `tau2.utils.llm_utils.generate`, so it automatically uses the configured agent service and the benchmark's request recording. An agent may use a different internal model client; its authentication and internal request logging are then the agent author's responsibility. Environment-side tool execution is still captured.

## Run the agent

From the repository root:

```console
python -X utf8 run_bench.py run --config examples/my_config.local.json --domain airline --task-id 21 --agent-factory my_agent:build_agent --max-steps 200 --max-errors 10 --output results/my_agent_airline21
```

Your module must be importable in the active Python environment. For a packaged implementation, install it into that environment and use its module path. Check a single task before using `suite`.

The run produces `report.html`, `records.jsonl`, machine-readable tables, and captured episodes. Examine the report's evidence links and `cases/run0001/evidence.json` to confirm that the interaction was captured correctly. Agent violations should be reported as process outcomes; they do not prevent a complete software run from producing a report.

## Comparing workflows

Keep tasks, repetition count, user simulator, scorer, rubric, and step/error budgets fixed. Record custom framework and model settings alongside the report. Do not expose hidden task goals, original task scores, or reference annotations to the evaluated agent or scorer.

The provided adapter covers English text interactions using this task/tool protocol. Other environments require an adapter or a trajectory conversion that preserves the evidence described in [FORMAT.md](FORMAT.md).
