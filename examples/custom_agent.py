"""Runnable adapter example, not a new Agent method.

Replace the delegate with your own stateful workflow while returning native
AssistantMessage objects. Hidden user scenarios and reference answers are not
passed to this factory.
"""


def build_agent(*, tools, domain_policy, llm, llm_args):
    from tau2.agent.llm_agent import LLMAgent
    return LLMAgent(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
