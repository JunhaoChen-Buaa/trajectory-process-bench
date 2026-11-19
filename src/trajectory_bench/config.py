"""Portable JSON configuration with secret references, never embedded credentials."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = load_json(path)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a JSON object")
    cfg = dict(cfg)
    cfg["_config_dir"] = str(path.parent)
    for role in ("agent", "user", "judge"):
        if role not in cfg:
            continue
        conf = cfg[role]
        if not isinstance(conf, dict) or not conf.get("model") or not conf.get("base_url"):
            raise ValueError(f"{role} needs model and base_url")
        if "api_key" in conf:
            raise ValueError("Use api_key_env or api_key_file; do not embed api_key")
        if not conf.get("api_key_env") and not conf.get("api_key_file"):
            raise ValueError(f"{role} needs api_key_env or api_key_file")
        if conf.get("api_key_file"):
            keypath = Path(os.path.expandvars(conf["api_key_file"])).expanduser()
            if not keypath.is_absolute():
                keypath = path.parent / keypath
            conf["api_key_file"] = str(keypath.resolve())
        if not conf["base_url"].startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("Use HTTPS or an explicit localhost compatible endpoint")
        parsed = urlsplit(conf["base_url"])
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, query parameters or fragments")
        if not 1 <= int(conf.get("max_attempts", 3)) <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
    native = cfg.get("native", {})
    for field in ("backend_root", "data_dir"):
        if native.get(field):
            val = Path(native[field]).expanduser()
            native[field] = str((path.parent / val).resolve() if not val.is_absolute() else val.resolve())
    return cfg


def safe_config(cfg: dict) -> dict:
    """Do not publish credential paths, values or arbitrary extra secret fields."""
    result = {k: v for k, v in cfg.items() if k not in ("agent", "user", "judge", "_config_dir")}
    for role in ("agent", "user", "judge"):
        if role in cfg:
            conf = cfg[role]
            result[role] = {k: conf[k] for k in ("provider", "base_url", "model", "options", "timeout", "max_attempts") if k in conf}
            result[role]["credentials"] = "environment" if conf.get("api_key_env") else "local_file"
    return result


def read_key(conf: dict) -> str:
    if conf.get("api_key_env"):
        key = os.environ.get(conf["api_key_env"], "").strip()
    else:
        key = Path(conf["api_key_file"]).read_text(encoding="utf-8-sig").strip()
    if not key or any(ch in key for ch in "\r\n"):
        raise ValueError("API credential unavailable or malformed; expected one nonempty line")
    return key
