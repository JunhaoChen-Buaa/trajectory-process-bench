"""Route tau2 completion calls to the explicitly configured role providers."""
from __future__ import annotations

from pathlib import Path

from .llm import ChatClient


def make_completion(cfg: dict, audit_dir: str | Path):
    clients = {role: ChatClient(cfg[role], Path(audit_dir) / role)
               for role in ("agent", "user") if role in cfg}
    counters = {role: 0 for role in clients}

    def completion(**kwargs):
        model = kwargs.pop("model")
        role = {"bench-agent": "agent", "bench-user": "user"}.get(model)
        if role not in clients:
            raise ValueError(f"Unconfigured native model route: {model}")
        messages = kwargs.pop("messages")
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        # Native's LiteLLM-only transport controls must not reach the provider.
        for key in ("num_retries", "caching", "cache", "drop_params", "call_name",
                    "custom_llm_provider", "allowed_openai_params"):
            kwargs.pop(key, None)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if kwargs.get("extra_body"):
            kwargs.update(kwargs.pop("extra_body"))
        counters[role] += 1
        return clients[role].call(messages, request_id=f"turn_{counters[role]:04d}",
                                  tools=tools, tool_choice=tool_choice, options=kwargs)

    return completion
