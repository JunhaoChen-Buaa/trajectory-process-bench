"""The published custom-Agent example must work without delegating to LLMAgent."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class StatefulAgentExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "examples/stateful_agent.py"
        spec = importlib.util.spec_from_file_location("acceptance_stateful_example", path)
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)

    def make_agent(self):
        return self.example.build_agent(tools=[], domain_policy="Follow the policy.",
                                        llm="bench-agent", llm_args={"temperature": 0})

    def test_own_state_is_separate_and_user_input_is_preserved(self):
        from tau2.data_model.message import AssistantMessage, UserMessage
        agent = self.make_agent()
        other = self.make_agent()
        state = agent.get_init_state()
        message = UserMessage(role="user", content="Please help.")
        answer = AssistantMessage(role="assistant", content="How can I help?")
        with patch.object(self.example, "generate", return_value=answer) as generate:
            result, updated = agent.generate_next_message(message, state)
        self.assertEqual(state, [])
        self.assertEqual(other.get_init_state(), [])
        self.assertEqual(updated, [message, answer])
        self.assertIs(result, answer)
        self.assertEqual(generate.call_args.kwargs["model"], "bench-agent")
        self.assertEqual(generate.call_args.kwargs["messages"][1], message)
        self.assertIn("Follow the policy.", generate.call_args.kwargs["messages"][0].content)

    def test_all_parallel_tool_returns_are_added_before_next_reply(self):
        from tau2.data_model.message import AssistantMessage, ToolMessage, MultiToolMessage
        agent = self.make_agent()
        previous = AssistantMessage(role="assistant", content="Checking.")
        a = ToolMessage(role="tool", id="a", content="first")
        b = ToolMessage(role="tool", id="b", content="second")
        combined = MultiToolMessage(role="tool", tool_messages=[a, b])
        reply = AssistantMessage(role="assistant", content="Both returned.")
        with patch.object(self.example, "generate", return_value=reply) as generate:
            _, updated = agent.generate_next_message(combined, [previous])
        self.assertEqual(updated, [previous, a, b, reply])
        self.assertEqual(generate.call_args.kwargs["messages"][1:], [previous, a, b])


if __name__ == "__main__":
    unittest.main()
