"""Configuração persistente do Mnemo (~/.mnemo/config.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG_DIR = Path.home() / ".mnemo"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS: Dict[str, Any] = {
    "theme": "auto",                     # "auto" | "dark" | "light"
    "vault_default": "vault-teste",
    "model_default": "nvidia/nemotron-3-super-120b-a12b",
    "timeout": 60,                       # seconds
    "auto_save": True,
    "max_history": 1000,
    "fallback_models": [
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-ultra",
        "nvidia/nemotron-4-340b-instruct",
    ],
    "health_check_timeout": 5,
}


class Config:
    """Carrega, valida e grava ~/.mnemo/config.json."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = {**DEFAULTS, **(data or {})}

    @classmethod
    def load(cls) -> "Config":
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if CONFIG_FILE.exists():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                return cls(raw)
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key in DEFAULTS:
            self.data[key] = value
        else:
            raise KeyError(f"Chave de config desconhecida: {key}")

    def merge_cli(self, **kwargs: Any) -> None:
        """Sobrescreve chaves vindas da CLI (None = não muda)."""
        for k, v in kwargs.items():
            if v is not None and k in DEFAULTS:
                self.data[k] = v

    def __repr__(self) -> str:
        return f"Config({self.data})"