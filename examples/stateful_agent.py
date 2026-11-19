"""An independent stateful Agent used for new-user acceptance.

No task IDs, hidden scenarios, reference labels or expected actions are read.
The public generation helper handles the configured model transport only.
"""
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import MultiToolMessage, SystemMessage
from tau2.utils.llm_utils import generate


class MyAgent(HalfDuplexAgent):
    def __init__(self, *, tools, domain_policy, llm, llm_args):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = llm
        self.options = dict(llm_args or {})

    def get_init_state(self, message_history=None):
        return list(message_history or [])

    def generate_next_message(self, message, state):
        history = list(state)
        if isinstance(message, MultiToolMessage):
            history.extend(message.tool_messages)
        else:
            history.append(message)
        instruction = SystemMessage(
            role="system",
            content=("Help the user by following the business policy below. "
                     "In each turn, either speak to the user or request tool calls. "
                     "Use the environment tools for real changes.\n\n" + self.domain_policy),
        )
        reply = generate(model=self.model, messages=[instruction] + history,
                         tools=self.tools, **self.options)
        return reply, history + [reply]


def build_agent(*, tools, domain_policy, llm, llm_args):
    return MyAgent(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
