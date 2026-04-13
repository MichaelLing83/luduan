"""Configuration management for Luduan."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomli_w

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-reattr]


CONFIG_DIR = Path.home() / ".config" / "luduan"
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG: dict[str, Any] = {
    "hotkey": {
        "keys": ["cmd", "shift", "space"],
    },
    "whisper": {
        "model": "base",
        "language": "",  # empty string = auto-detect; set to e.g. "en", "zh", "sv", "fr" to pin
    },
    "ollama": {
        "host": "http://localhost:11434",
        "model": "gemma3:4b",
        "enabled": True,
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
    },
    "context": {
        "enabled": False,
        "max_chars": 500,
    },
}


def load() -> dict[str, Any]:
    """Load config from disk, creating defaults if missing."""
    if not CONFIG_PATH.exists():
        save(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    with open(CONFIG_PATH, "rb") as f:
        on_disk = tomllib.load(f)

    # Deep-merge defaults so new keys are always present
    return _deep_merge(DEFAULT_CONFIG, on_disk)


def save(config: dict[str, Any]) -> None:
    """Persist config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(config, f)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
