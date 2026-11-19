"""No-network capture tests, plus optional real-backend/scripted-model smoke.

The smoke exercises the actual public LLMAgent, UserSimulator, orchestrator and
mock tools. Its completion is scripted; it is not a real model quality test.
Run with the backend environment: python -m unittest discover -s tests -v.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from trajectory_bench.native import EpisodeCapture, NativeRunSpec, run_native


class Message:
    def __init__(self, **row):
        self.row = row
        for key, value in row.items():
            setattr(self, key, value)

    def model_dump(self, **kwargs):
        return copy.deepcopy(self.row)


class FakeEnvironment:
    def __init__(self):
        self.tools = SimpleNamespace(db={"balance": 10})
        self.user_tools = SimpleNamespace(db={"read": False})

    def get_db_hash(self):
        return json.dumps(self.tools.db, sort_keys=True)

    def get_user_db_hash(self):
        return json.dumps(self.user_tools.db, sort_keys=True)

    def make_tool_call(self, tool_name, requestor="assistant", **kwargs):
        if tool_name == "fail_after_write":
            self.tools.db["balance"] -= 1
            raise ValueError("genuine tool error")
        if requestor == "user":
            self.user_tools.db["read"] = True
            return "read"
        self.tools.db["balance"] += kwargs["amount"]
        return {"balance": self.tools.db["balance"]}

    def get_response(self, message):
        try:
            result = self.make_tool_call(message.name, requestor=message.requestor,
                                         **message.arguments)
            error = False
        except Exception:
            result, error = "Error: genuine tool error", True
        return Message(role="tool", id=message.id, requestor=message.requestor,
                       content=json.dumps(result), error=error)


class Participant:
    def generate_next_message(self, message, state):
        return message, state

    def stop(self, message=None, state=None):
        pass


class FakeOrchestrator:
    def __init__(self):
        self.environment = FakeEnvironment()
        self.agent, self.user = Participant(), Participant()
        self.trajectory = []

    def initialize(self):
        self.environment.tools.db["balance"] = 20


class CaptureTests(unittest.TestCase):
    def test_actual_failed_write_user_call_and_restore(self):
        orch = FakeOrchestrator()
        original = orch.environment.make_tool_call
        with EpisodeCapture(orch) as capture:
            orch.initialize()
            for index, (name, role, arguments) in enumerate([
                ("write", "assistant", {"amount": 4}),
                ("fail_after_write", "assistant", {}),
                ("read", "user", {}),
            ]):
                call = Message(id=f"call{index}", name=name, requestor=role, arguments=arguments)
                orch.trajectory.append(Message(role=role, content=None,
                    tool_calls=[call.model_dump()], timestamp=f"t{index}"))
                response = orch.environment.get_response(call)
                orch.trajectory.append(response)
        self.assertEqual(orch.environment.make_tool_call, original)
        self.assertEqual(capture.initial_state["agent"]["snapshot"]["balance"], 20)
        self.assertEqual(capture.final_state["agent"]["snapshot"]["balance"], 23)
        self.assertEqual([event["requestor"] for event in capture.events],
                         ["assistant", "assistant", "user"])
        self.assertEqual(capture.events[1]["status"], "raised")
        self.assertTrue(capture.events[1]["tool_message"]["error"])
        self.assertTrue(capture.events[1]["state_changes"])
        self.assertTrue(capture.integrity(capture.messages(orch.trajectory))["verified"])

    def test_delivery_flags_do_not_publish_unsent_prefix(self):
        orch = FakeOrchestrator()
        public = Message(role="assistant", content="Submitted.", timestamp="a", tool_calls=None)
        unsent = Message(role="assistant", content="Thinking prefix", timestamp="b",
                         tool_calls=[{"id": "c", "name": "write", "arguments": {}}])
        with EpisodeCapture(orch) as capture:
            orch.user.generate_next_message(public, None)
        messages = capture.messages([public, unsent])
        self.assertTrue(messages[0]["customer_visible"])
        self.assertFalse(messages[1]["customer_visible"])

    def test_unexecuted_rejection_is_disclosed(self):
        orch = FakeOrchestrator()
        capture = EpisodeCapture(orch)
        result = capture.integrity([
            {"role": "assistant", "tool_calls": [{"id": "x", "name": "write", "arguments": {}}]},
            {"role": "tool", "id": "x", "requestor": "assistant", "execution_status": "rejected"},
        ])
        self.assertFalse(result["verified"])
        self.assertEqual(result["unexecuted_calls"][0]["reason"], "orchestrator_rejected")

    def test_spec_rejects_secrets_and_implicit_judges(self):
        with self.assertRaises(ValueError):
            NativeRunSpec("retail", ["0"], "agent", "user", llm_args_agent={"api_key": "do-not-log"})
        with self.assertRaises(ValueError):
            NativeRunSpec("retail", ["0"], "agent", "user", evaluation="all")


BACKEND = Path(__file__).resolve().parents[1] / "vendor" / "tau2-bench"
HAS_RUNTIME = sys.version_info >= (3, 12) and importlib.util.find_spec("litellm") is not None


@unittest.skipUnless(HAS_RUNTIME and BACKEND.is_dir(), "optional tau2 Python/dependencies unavailable")
class PublicBackendSmokeTests(unittest.TestCase):
    def test_public_llm_agent_scripted_completion_actual_tools_and_reset(self):
        counters = {"bench-agent": 0, "bench-user": 0}

        def completion(model, messages, tools=None, **kwargs):
            counters[model] += 1
            user_turn = (counters[model] - 1) % 3
            agent_turn = (counters[model] - 1) % 2
            message = {"role": "assistant", "content": None}
            if model == "bench-user" and user_turn == 0:
                message["tool_calls"] = [{"id": "user-read", "type": "function", "function": {
                    "name": "check_notifications", "arguments": "{}"}}]
            elif model == "bench-user":
                message["content"] = ("Create Important Meeting for user_1." if user_turn == 1
                                      else "###STOP###")
            elif agent_turn == 0:
                message["tool_calls"] = [
                    {"id": "agent-write", "type": "function", "function": {
                        "name": "create_task", "arguments": json.dumps({
                            "user_id": "user_1", "title": "Important Meeting"})}},
                    {"id": "agent-failure", "type": "function", "function": {
                        "name": "update_task_status", "arguments": json.dumps({
                            "task_id": "no_such_task", "status": "completed"})}},
                ]
            else:
                message["content"] = "Created the task; the unrelated status update failed."
            return {"id": "scripted-response", "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

        records = run_native(NativeRunSpec(
            "mock", ["create_task_1"], "bench-agent", "bench-user",
            backend_root=BACKEND, trials=2, max_steps=20, evaluation="none",
        ), completion=completion)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["status"], "complete", record.get("error"))
            self.assertEqual(record["agent_class"], "tau2.agent.llm_agent.LLMAgent")
            self.assertIsNone(record["simulation"]["reward_info"])
            self.assertEqual(len(record["tool_events"]), 3)
            self.assertEqual(record["tool_events"][0]["requestor"], "user")
            self.assertTrue(record["tool_events"][1]["state_changes"])
            self.assertEqual(record["tool_events"][2]["status"], "raised")
            self.assertTrue(record["capture_integrity"]["verified"], record["capture_integrity"])
            self.assertEqual({call["role"] for call in record["api_calls"]}, {"agent", "user"})
            self.assertTrue(any(message.get("customer_visible") and "Created the task" in (message.get("content") or "")
                                for message in record["messages"]))
        self.assertEqual(records[0]["initial_state"], records[1]["initial_state"])
        self.assertEqual(records[0]["final_state"], records[1]["final_state"])

        from tau2.agent.llm_agent import LLMAgent
        from tau2.registry import registry

        factory_inputs = []

        def safe_factory(**kwargs):
            self.assertEqual(set(kwargs), {"tools", "domain_policy", "llm", "llm_args"})
            self.assertNotIn("task", kwargs)
            factory_inputs.append(set(kwargs))
            return LLMAgent(**kwargs)

        direct = run_native(NativeRunSpec(
            "mock", ["create_task_1"], "bench-agent", "bench-user", backend_root=BACKEND,
        ), completion=completion, agent_factory=safe_factory)
        self.assertEqual(direct[0]["status"], "complete", direct[0].get("error"))
        registry.register_agent_factory(safe_factory, name="trajectory_bench_unit_test_agent")
        registered = run_native(NativeRunSpec(
            "mock", ["create_task_1"], "bench-agent", "bench-user", backend_root=BACKEND,
            agent="trajectory_bench_unit_test_agent",
        ), completion=completion)
        self.assertEqual(registered[0]["status"], "complete", registered[0].get("error"))
        self.assertEqual(len(factory_inputs), 2)


if __name__ == "__main__":
    unittest.main()
