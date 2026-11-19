"""Run and capture the public tau2 half-duplex backend without editing it.

The backend is optional and imported only by :func:`run_native`.  A caller can
provide a LiteLLM-compatible ``completion(**kwargs)`` callable, so credentials
and provider routing never have to enter tau2's persisted configuration.  Runs
are serialized because tau2's completion entry point is a process global.

``evaluation='none'`` runs the real orchestrator without running an additional
official judge. ``env`` and ``action`` explicitly request deterministic official
checks.  No experiment module, answer label, frozen result or secret file is
imported by this module.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import json
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "trajectory_bench.native_episode.v1"
_BACKEND_LOCK = threading.RLock()
_ROLE = contextvars.ContextVar("trajectory_bench_native_role", default="unknown")
_SECRET_FIELDS = frozenset({
    "api_key", "apikey", "authorization", "proxy_authorization", "access_token",
    "refresh_token", "secret", "client_secret", "password", "headers",
    "extra_headers", "default_headers",
})


class NativeDependencyError(RuntimeError):
    """The selected optional backend/runtime cannot be used in this process."""


@dataclass(frozen=True)
class NativeRunSpec:
    """Explicit English task selection; no default model or remote endpoint.

    ``backend_root`` is the tau2 project directory containing ``src/tau2`` and
    ``data``. Omit it only to use an already installed tau2 package. Python and
    dependencies must satisfy that backend's pyproject (pinned v2: 3.12--3.13).
    Custom implementations register themselves in ``plugins`` or can be
    registered by the caller before this function. Custom factories receive
    ``tools, domain_policy, llm, llm_args`` only, never the hidden task/answers.
    """

    domain: str
    task_ids: Sequence[str]
    agent_model: str
    user_model: str
    agent: str = "llm_agent"
    user: str = "user_simulator"
    trials: int = 1
    seed: int = 42
    max_steps: int = 80
    max_errors: int = 5
    llm_args_agent: Mapping[str, Any] = field(default_factory=lambda: {"temperature": 0})
    llm_args_user: Mapping[str, Any] = field(default_factory=lambda: {"temperature": 0})
    backend_root: str | Path | None = None
    data_dir: str | Path | None = None
    task_split: str | None = "base"
    task_set: str | None = None
    plugins: Sequence[str] = ()
    evaluation: str = "none"
    snapshots: bool = True
    timeout: float | None = None
    enforce_communication_protocol: bool = False

    def __post_init__(self) -> None:
        ids = tuple(str(item) for item in self.task_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("task_ids must be a nonempty explicit list with no duplicates")
        if self.trials < 1 or self.max_steps < 1 or self.max_errors < 1:
            raise ValueError("trials, max_steps and max_errors must be positive")
        if not self.agent_model or not self.user_model:
            raise ValueError("agent_model and user_model must be supplied explicitly")
        if self.evaluation not in {"none", "env", "action"}:
            raise ValueError("evaluation must be none, env, or action; model judges are separate")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
        object.__setattr__(self, "task_ids", ids)
        object.__setattr__(self, "plugins", tuple(self.plugins))
        for args in (self.llm_args_agent, self.llm_args_user):
            _reject_secrets(args)


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _SECRET_FIELDS:
                raise ValueError("Supply credentials through the injected completion, not llm_args")
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)


def _json(value: Any) -> Any:
    """Detach JSON values; never stringify opaque objects or exception messages."""
    if hasattr(value, "model_dump"):
        return _json(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): ("[REDACTED]" if str(key).lower() in _SECRET_FIELDS else _json(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json(value.to_dict())
    raise TypeError(f"Cannot capture unsupported value type {type(value).__name__}")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _error(error: BaseException) -> dict[str, Any]:
    # Provider exceptions may include credentials or request headers in str(e).
    return {
        "type": type(error).__name__,
        "frames": [{"file": item.filename, "line": item.lineno, "function": item.name}
                   for item in traceback.extract_tb(error.__traceback__)],
    }


def _state(environment: Any, *, snapshots: bool = True) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for role, attribute, hash_method in (
        ("agent", "tools", "get_db_hash"),
        ("user", "user_tools", "get_user_db_hash"),
    ):
        toolkit = getattr(environment, attribute, None)
        db = getattr(toolkit, "db", None)
        row: dict[str, Any] = {"available": db is not None}
        try:
            method = getattr(environment, hash_method, None)
            row["hash"] = method() if method is not None else None
        except Exception as error:
            row["hash_error_type"] = type(error).__name__
        if snapshots and db is not None:
            try:
                row["snapshot"] = _json(db)
            except Exception as error:
                row["snapshot_error_type"] = type(error).__name__
        state[role] = row
    return state


def _hashes(state: dict[str, Any]) -> dict[str, Any]:
    return {role: {key: value for key, value in row.items() if key != "snapshot"}
            for role, row in state.items()}


def _changes(before: Any, after: Any, path: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        output = []
        for key in sorted(before.keys() | after.keys()):
            if key in before and key in after:
                output.extend(_changes(before[key], after[key], (*path, key)))
            else:
                output.append({"path": [*path, key], "before_exists": key in before,
                               "after_exists": key in after, "before": before.get(key),
                               "after": after.get(key)})
        return output
    # Lists remain whole: insertion/deletion must not invent element pairing.
    return [{"path": list(path), "before_exists": True, "after_exists": True,
             "before": before, "after": after}]


def _message_key(message: Any) -> str:
    row = _json(message)
    return _digest({key: row.get(key) for key in (
        "role", "timestamp", "id", "content", "tool_calls", "requestor")})


class EpisodeCapture:
    """Scoped instrumentation on one already-built half-duplex orchestrator.

    Captures an actual tool attempt even when the tool raises. ``tool_message``
    is the environment's real returned receipt, including its error flag. State
    deltas are observed, not reconstructed from task answers. Instrumentation
    restores the original instance attributes on exit, including error exits.
    """

    def __init__(self, orchestrator: Any, *, snapshots: bool = True):
        self.orchestrator = orchestrator
        self.environment = orchestrator.environment
        self.snapshots = snapshots
        self.events: list[dict[str, Any]] = []
        self.initial_state: dict[str, Any] | None = None
        self.final_state: dict[str, Any] | None = None
        self.capture_errors: list[dict[str, str]] = []
        self._originals: list[tuple[Any, str, bool, Any]] = []
        self._current_call: dict[str, Any] | None = None
        self._initializing = False
        self._received: dict[str, set[str]] = {"agent": set(), "user": set()}

    def _patch(self, obj: Any, name: str, value: Any) -> None:
        own = name in getattr(obj, "__dict__", {})
        self._originals.append((obj, name, own, getattr(obj, name)))
        setattr(obj, name, value)

    def _observe(self, role: str, message: Any) -> None:
        if message is None:
            return
        children = getattr(message, "tool_messages", None)
        if children is not None:
            for child in children:
                self._observe(role, child)
        else:
            self._received[role].add(_message_key(message))

    def _participant(self, participant: Any, role: str) -> None:
        for name in ("generate_next_message", "stop"):
            original = getattr(participant, name, None)
            if original is None:
                continue

            def wrapped(*args: Any, _original=original, **kwargs: Any) -> Any:
                message = args[0] if args else kwargs.get("message")
                self._observe(role, message)
                token = _ROLE.set(role)
                try:
                    return _original(*args, **kwargs)
                finally:
                    _ROLE.reset(token)

            self._patch(participant, name, wrapped)

    def __enter__(self) -> "EpisodeCapture":
        env = self.environment
        original_call = env.make_tool_call
        original_response = env.get_response
        original_initialize = self.orchestrator.initialize

        def capture_call(tool_name: str, requestor: str = "assistant", **kwargs: Any) -> Any:
            before = _state(env, snapshots=self.snapshots)
            event = {
                "event_index": len(self.events), "name": tool_name,
                "requestor": requestor, "arguments": _json(kwargs),
                "tool_call_id": (self._current_call or {}).get("id"),
                "phase": "initialization" if self._initializing else "trial",
                "trajectory_length": len(self.orchestrator.trajectory),
                "state_before": _hashes(before),
            }
            self.events.append(event)
            started = time.monotonic()
            try:
                result = original_call(tool_name, requestor=requestor, **kwargs)
                try:
                    event["response"] = _json(result)
                except Exception as error:
                    # Capture must never turn a successful real operation into
                    # a failed tool receipt just because its value is opaque.
                    event["response_capture_error_type"] = type(error).__name__
                    self.capture_errors.append({"type": "tool_response_not_serializable"})
                event["status"] = "returned"
                return result
            except Exception as error:
                event["status"] = "raised"
                event["error_type"] = type(error).__name__
                raise
            finally:
                after = _state(env, snapshots=self.snapshots)
                event["state_after"] = _hashes(after)
                event["state_changes"] = _changes(before, after)
                event["elapsed_seconds"] = round(time.monotonic() - started, 6)

        def capture_response(message: Any) -> Any:
            previous = self._current_call
            self._current_call = _json(message)
            start = len(self.events)
            before = _state(env, snapshots=self.snapshots)
            try:
                response = original_response(message)
                # get_response syncs toolkits after make_tool_call. Keep the
                # actual post-sync state rather than assuming a write succeeded.
                after = _state(env, snapshots=self.snapshots)
                matching = self.events[start:]
                if matching:
                    matching[-1]["tool_message"] = _json(response)
                    matching[-1]["state_after"] = _hashes(after)
                    matching[-1]["state_changes"] = _changes(before, after)
                else:
                    self.capture_errors.append({"type": "response_without_make_tool_call"})
                return response
            finally:
                self._current_call = previous

        def capture_initialize(*args: Any, **kwargs: Any) -> Any:
            self._initializing = True
            try:
                return original_initialize(*args, **kwargs)
            finally:
                self._initializing = False
                self.initial_state = _state(env, snapshots=self.snapshots)

        try:
            self._patch(env, "make_tool_call", capture_call)
            self._patch(env, "get_response", capture_response)
            self._patch(self.orchestrator, "initialize", capture_initialize)
            self._participant(self.orchestrator.agent, "agent")
            self._participant(self.orchestrator.user, "user")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused: Any) -> None:
        self.final_state = _state(self.environment, snapshots=self.snapshots)
        for obj, name, own, original in reversed(self._originals):
            if own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self._originals.clear()

    def messages(self, messages: Sequence[Any]) -> list[dict[str, Any]]:
        output = []
        for message in messages:
            row = _json(message)
            key = _message_key(message)
            if row.get("role") == "assistant":
                row["customer_visible"] = key in self._received["user"]
            if row.get("role") in {"user", "tool"}:
                row["agent_observed"] = key in self._received["agent"]
            output.append(row)
        return output

    def integrity(self, messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
        calls: list[tuple[int, dict[str, Any]]] = []
        receipts: dict[tuple[Any, Any], list[tuple[int, dict[str, Any]]]] = {}
        for index, message in enumerate(messages):
            if message.get("role") in {"assistant", "user"}:
                for call in message.get("tool_calls") or []:
                    calls.append((index, {**call, "requestor": message["role"]}))
            elif message.get("role") == "tool":
                receipts.setdefault((message.get("requestor"), message.get("id")), []).append((index, message))
        unused = set(range(len(self.events)))
        used_receipts: set[int] = set()
        unmatched, pairs = [], []
        for message_index, call in calls:
            matches = [index for index in sorted(unused)
                       if self.events[index].get("tool_call_id") == call.get("id")
                       and self.events[index]["requestor"] == call["requestor"]
                       and self.events[index]["name"] == call.get("name")
                       and self.events[index]["arguments"] == call.get("arguments")]
            key = (call["requestor"], call.get("id"))
            following = [(i, receipt) for i, receipt in receipts.get(key, [])
                         if i > message_index and i not in used_receipts]
            if matches:
                event_index = matches[0]
                unused.remove(event_index)
                expected = self.events[event_index].get("tool_message")
                exact = [(i, receipt) for i, receipt in following
                         if expected is not None and _message_key(receipt) == _message_key(expected)]
                receipt_index = exact[0][0] if exact else None
                if receipt_index is not None:
                    used_receipts.add(receipt_index)
                pairs.append({"message_index": message_index, "tool_call_id": call.get("id"),
                              "event_index": event_index,
                              "receipt_message_index": receipt_index})
            else:
                rejected = following and following[0][1].get("execution_status") == "rejected"
                if rejected:
                    used_receipts.add(following[0][0])
                unmatched.append({"message_index": message_index, "call": call,
                                  "reason": "orchestrator_rejected" if rejected else "no_captured_execution"})
        orphan_receipts = sorted(index for rows in receipts.values() for index, _ in rows
                                 if index not in used_receipts)
        return {
            "actual_tool_events": len(self.events), "structured_tool_calls": len(calls),
            "paired_calls": len(pairs), "pairs": pairs,
            "unmatched_structured_calls": len(unmatched), "unexecuted_calls": unmatched,
            "events_without_structured_call": sorted(unused),
            "orphan_tool_receipt_indices": orphan_receipts,
            "capture_errors": list(self.capture_errors),
            "verified": not unmatched and not unused and not orphan_receipts and not self.capture_errors
                        and all(pair["receipt_message_index"] is not None for pair in pairs),
            "scope": "Exact call id/requestor/name/arguments and following receipt; no success inference",
        }


@contextmanager
def _completion_scope(completion: Callable[..., Any] | None,
                      events: list[dict[str, Any]]) -> Iterator[None]:
    from tau2.utils import llm_utils

    original = llm_utils.completion
    chosen = completion or original

    def wrapped(**kwargs: Any) -> Any:
        event: dict[str, Any] = {"role": _ROLE.get(), "requested_model": kwargs.get("model")}
        events.append(event)
        started = time.monotonic()
        try:
            result = chosen(**kwargs)
            if isinstance(result, Mapping):
                # Keep the transport provider-neutral. Tau2 itself expects
                # LiteLLM's attribute-access response object.
                from litellm import ModelResponse

                payload = dict(result)
                payload.setdefault("model", kwargs.get("model"))
                result = ModelResponse(**payload)
            event["returned_model"] = getattr(result, "model", None)
            event["usage"] = _json(getattr(result, "usage", None))
            choices = getattr(result, "choices", [])
            event["finish_reason"] = getattr(choices[0], "finish_reason", None) if choices else None
            event["status"] = "returned"
            return result
        except Exception as error:
            event["status"] = "error"
            event["error_type"] = type(error).__name__
            raise
        finally:
            event["elapsed_seconds"] = round(time.monotonic() - started, 6)

    llm_utils.completion = wrapped
    try:
        yield
    finally:
        llm_utils.completion = original


def _load_backend(spec: NativeRunSpec) -> dict[str, Any]:
    root = Path(spec.backend_root).resolve() if spec.backend_root else None
    if root is not None:
        source = root / "src"
        if not (source / "tau2" / "__init__.py").is_file():
            raise NativeDependencyError("backend_root must contain src/tau2/__init__.py")
        loaded = sys.modules.get("tau2")
        if loaded is not None and Path(loaded.__file__).resolve().parent != source / "tau2":
            raise NativeDependencyError("A different tau2 backend is already imported; use a fresh process")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    data = Path(spec.data_dir).resolve() if spec.data_dir else (root / "data" if root else None)
    if data is not None:
        if not data.is_dir():
            raise NativeDependencyError("The selected tau2 data directory does not exist")
        loaded_utils = sys.modules.get("tau2.utils.utils")
        if loaded_utils is not None and Path(loaded_utils.DATA_DIR).resolve() != data:
            raise NativeDependencyError("tau2 already bound a different data directory; use a fresh process")
        os.environ["TAU2_DATA_DIR"] = str(data)
    # Disable LiteLLM's model-map download and telemetry before its lazy import.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    try:
        tau2 = importlib.import_module("tau2")
        from tau2.data_model.simulation import TextRunConfig
        from tau2.multilingual.runtime import load_run_language
        from tau2.runner.build import build_text_orchestrator
        from tau2.runner.simulation import run_simulation
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.registry import registry
    except (ImportError, SyntaxError) as error:
        missing = getattr(error, "name", None)
        raise NativeDependencyError(
            "tau2 backend needs Python 3.12--3.13 and its runtime dependencies; "
            "from the benchmark directory run: python -m pip install -r requirements.txt. "
            f"Import failure: {type(error).__name__}; missing module: {missing or 'see backend installation'}. "
            "Python 3.13 also requires audioop-lts."
        ) from None
    for plugin in spec.plugins:
        importlib.import_module(plugin)
    return {
        "TextRunConfig": TextRunConfig, "load_run_language": load_run_language,
        "build": build_text_orchestrator, "run_simulation": run_simulation,
        "EvaluationType": EvaluationType,
        "registry": registry,
        "source": str(Path(tau2.__file__).resolve()), "data_dir": str(data) if data else None,
    }


def _tools(environment: Any, *, user: bool = False) -> list[dict[str, Any]]:
    toolkit = getattr(environment, "user_tools" if user else "tools", None)
    if toolkit is None:
        return []
    method = environment.get_user_tools if user else environment.get_tools
    return [_json(tool.openai_schema) for tool in method()]


def prepare_backend(spec: NativeRunSpec) -> None:
    """Make the explicit optional backend importable before loading an Agent.

    Custom adapter modules often subclass tau2 classes at module scope. Call
    this before importing such a module when the backend is local source rather
    than an installed package. This loads dependencies only; no task is run and
    no model request is made.
    """
    with _BACKEND_LOCK:
        _load_backend(spec)


def run_native(spec: NativeRunSpec, *, completion: Callable[..., Any] | None = None,
               agent_factory: Callable[..., Any] | None = None,
               on_episode: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Run selected real tasks, reconstructing environment/participants per trial.

    The injected completion returns an OpenAI response dict or LiteLLM
    ``ModelResponse``; its arguments are ordinary OpenAI-style messages/tools. The callable
    owns provider credentials/routing/retries. Nothing calls a remote API when
    importing this module or constructing a spec. Trials are independent fresh
    tau2 environments; a failed trial remains an error episode, never a success.

    ``on_episode`` is called after each trial for durable CLI storage. It is not
    swallowed: output/write failures should stop the caller explicitly.
    """
    with _BACKEND_LOCK:
        backend = _load_backend(spec)
        selected_factory = agent_factory
        if spec.agent != "llm_agent" and selected_factory is None:
            if spec.agent in {"llm_agent_gt", "llm_agent_solo"}:
                raise ValueError("Ground-truth and solo agents are outside the ordinary interactive benchmark")
            selected_factory = backend["registry"].get_agent_factory(spec.agent)
            if selected_factory is None:
                raise ValueError("The requested custom Agent has no registered factory")
        config = backend["TextRunConfig"](
            domain=spec.domain, language="en", task_ids=list(spec.task_ids),
            task_set_name=spec.task_set, task_split_name=spec.task_split,
            # Build the standard scaffold, then install a custom participant
            # using the safe public inputs. Upstream build_agent passes a full
            # hidden task to factories; that is intentionally avoided here.
            agent="llm_agent", user=spec.user,
            llm_agent=spec.agent_model, llm_user=spec.user_model,
            llm_args_agent=dict(spec.llm_args_agent), llm_args_user=dict(spec.llm_args_user),
            seed=spec.seed, max_steps=spec.max_steps, max_errors=spec.max_errors,
            timeout=spec.timeout, enforce_communication_protocol=spec.enforce_communication_protocol,
        )
        try:
            loaded = backend["load_run_language"](config)
        except UnicodeDecodeError:
            raise NativeDependencyError(
                "The backend reads UTF-8 policy files with the process default encoding. "
                "On Windows launch Python with -X utf8 (or set PYTHONUTF8=1 before launch)."
            ) from None
        selected = {str(task.id): task for task in loaded.tasks}
        missing = [task_id for task_id in spec.task_ids if task_id not in selected]
        if missing:
            raise ValueError(f"Requested English task ids not found: {missing}")
        episodes = []
        for task_id in spec.task_ids:
            for trial in range(spec.trials):
                task = selected[task_id].model_copy(deep=True)
                initial = getattr(task, "initial_state", None)
                if initial is not None and getattr(initial, "message_history", None):
                    raise NativeDependencyError("Seeded message_history delivery is not instrumented; this runner supports fresh-conversation tasks only")
                episode_id = str(uuid.uuid4())
                seed = spec.seed + trial
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION, "id": episode_id,
                    "domain": spec.domain, "task_id": task_id, "trial": trial,
                    "seed": seed, "language": "en", "status": "started",
                    "task": _json(task), "config": _json(config),
                    "backend": {"name": "tau2", "source": backend["source"],
                                "data_dir": backend["data_dir"]},
                    "official_evaluation": spec.evaluation,
                    "requested_agent": spec.agent,
                    "api_calls": [], "tool_events": [], "messages": [],
                }
                started = time.monotonic()
                capture = None
                orchestrator = None
                try:
                    with _completion_scope(completion, record["api_calls"]):
                        # build_text_orchestrator constructs a new environment
                        # and participants every time; initialize applies task
                        # initial state before the first live conversation step.
                        orchestrator = backend["build"](config, task, seed=seed,
                                                         simulation_id=episode_id)
                        environment = orchestrator.environment
                        if selected_factory is not None:
                            # Match the public factory's documented kwargs.
                            # The ordinary builder made no model call; replace
                            # its participant before any state initialization.
                            orchestrator.agent = selected_factory(
                                tools=environment.get_tools(),
                                domain_policy=getattr(orchestrator.agent, "domain_policy", environment.get_policy()),
                                llm=spec.agent_model, llm_args=dict(spec.llm_args_agent),
                            )
                            if not hasattr(orchestrator.agent, "generate_next_message"):
                                raise TypeError("agent_factory must return a half-duplex Agent")
                        record.update(
                            domain_policy=environment.get_policy(),
                            agent_policy=getattr(orchestrator.agent, "domain_policy", environment.get_policy()),
                            tools=_tools(environment), user_tools=_tools(environment, user=True),
                            agent_class=f"{type(orchestrator.agent).__module__}.{type(orchestrator.agent).__name__}",
                            user_class=f"{type(orchestrator.user).__module__}.{type(orchestrator.user).__name__}",
                        )
                        capture = EpisodeCapture(orchestrator, snapshots=spec.snapshots)
                        with capture:
                            if spec.evaluation == "none":
                                simulation = orchestrator.run()
                            else:
                                simulation = backend["run_simulation"](
                                    orchestrator,
                                    evaluation_type=backend["EvaluationType"](spec.evaluation),
                                )
                        record["simulation"] = _json(simulation)
                        record["status"] = "complete"
                        record["termination_reason"] = record["simulation"].get("termination_reason")
                except Exception as error:
                    record["status"] = "error"
                    record["error"] = _error(error)
                finally:
                    if capture is not None:
                        actual_messages = (getattr(orchestrator, "trajectory", []) or [])
                        if "simulation" in record:
                            # Preserve native final chronological ordering.
                            actual_messages = simulation.messages
                        messages = capture.messages(actual_messages)
                        record.update(
                            messages=messages, tool_events=capture.events,
                            initial_state=capture.initial_state, final_state=capture.final_state,
                            capture_integrity=capture.integrity(messages),
                        )
                        if "simulation" in record:
                            record["simulation"]["messages"] = messages
                    record["elapsed_seconds"] = round(time.monotonic() - started, 6)
                episodes.append(record)
                if on_episode is not None:
                    on_episode(record)
        return episodes


__all__ = ["NativeRunSpec", "NativeDependencyError", "EpisodeCapture", "prepare_backend", "run_native"]
