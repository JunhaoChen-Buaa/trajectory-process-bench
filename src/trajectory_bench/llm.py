"""OpenAI-compatible transport. Persist first answers; retry transport failures only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import ssl
import time
import urllib.error
import urllib.request

from .config import digest, load_json, read_key, save_json


class APIError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatClient:
    def __init__(self, conf: dict, audit_dir: str | Path):
        self.conf = conf
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def call(self, messages: list[dict], *, request_id: str, tools: list | None = None,
             tool_choice: str | dict | None = None, options: dict | None = None) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
            raise ValueError("Unsafe request_id")
        opts = dict(self.conf.get("options", {}))
        opts.update(options or {})
        extra_body = opts.pop("extra_body", None)
        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise ValueError("extra_body must be a JSON object")
            opts.update(extra_body)
        forbidden = {"api_key", "api_key_file", "api_key_env", "headers", "base_url", "messages", "model", "tools"}
        if forbidden.intersection(opts):
            raise ValueError("Credentials and transport fields cannot be options")
        payload = dict(opts, model=self.conf["model"], messages=messages)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        elif tool_choice is not None:
            payload["tool_choice"] = tool_choice
        fingerprint = digest(payload)
        folder = self.audit_dir / request_id
        folder.mkdir(parents=True, exist_ok=True)
        request_path = folder / "request.json"
        saved_request = {"request_sha256": fingerprint, "payload": payload,
                         "provider": self.conf.get("provider", "openai_compatible"),
                         "base_url": self.conf["base_url"], "created_at": now()}
        if request_path.exists():
            if load_json(request_path)["request_sha256"] != fingerprint:
                raise APIError(f"Saved request differs: {request_id}; choose a new output directory")
        else:
            save_json(request_path, saved_request)
        # Cached HTTP200 is terminal even when its model output is malformed.
        attempts = sorted(folder.glob("attempt_*.json"))
        for path in attempts:
            old = load_json(path)
            if old.get("http_status") == 200:
                if old.get("request_sha256") != fingerprint:
                    raise APIError("Cache input hash mismatch")
                return self._parse(old, path)
        max_attempts = min(int(self.conf.get("max_attempts", 3)), 5)
        if len(attempts) >= max_attempts:
            raise APIError(f"Transport attempts exhausted at {folder}; original errors retained")
        key = read_key(self.conf)
        base_url = self.conf["base_url"].rstrip("/")
        endpoint = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
        handlers = []
        if self.conf.get("trust_env", True) is False:
            handlers.append(urllib.request.ProxyHandler({}))
        handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        opener = urllib.request.build_opener(*handlers)
        for attempt in range(len(attempts) + 1, max_attempts + 1):
            path = folder / f"attempt_{attempt:02d}.json"
            start = time.monotonic()
            row = {"request_sha256": fingerprint, "recorded_at": now()}
            req = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            try:
                try:
                    with opener.open(req, timeout=float(self.conf.get("timeout", 240))) as response:
                        row.update(http_status=response.status, response_text=response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    row.update(http_status=exc.code, response_text=exc.read().decode("utf-8", errors="replace"))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                row.update(transport_error=type(exc).__name__, detail=str(exc).replace(key, "[REDACTED]")[:500])
            row["seconds"] = round(time.monotonic() - start, 3)
            if "response_text" in row:
                row["response_text"] = row["response_text"].replace(key, "[REDACTED]")
            save_json(path, row)
            if row.get("http_status") == 200:
                return self._parse(row, path)
            retryable = row.get("http_status") in (408, 429, 500, 502, 503, 504) or "transport_error" in row
            if not retryable or attempt == max_attempts:
                raise APIError(f"API failed ({row.get('http_status', row.get('transport_error'))}); see {path}")
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("Transport loop did not return")

    def _parse(self, row: dict, path: Path) -> dict:
        try:
            result = json.loads(row["response_text"])
        except (ValueError, KeyError) as exc:
            raise APIError(f"HTTP200 response is not JSON; retained at {path}, no resampling") from exc
        if not isinstance(result, dict) or not result.get("choices"):
            raise APIError(f"HTTP200 response lacks choices; retained at {path}, no resampling")
        save_json(path.parent / "receipt.json", {
            "provider": self.conf.get("provider", "openai_compatible"),
            "requested_model": self.conf["model"], "actual_model": result.get("model"),
            "request_sha256": row["request_sha256"], "usage": result.get("usage"),
            "finish_reason": result["choices"][0].get("finish_reason"),
            "raw_path": path.name, "model_identity_verified": result.get("model") == self.conf["model"],
        })
        return result
