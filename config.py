"""Configuração persistente do Mnemo (~/.mnemo/config.json)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_DIR = Path.home() / ".mnemo"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS: Dict[str, Any] = {
    "theme": "auto",                     # "auto" | "dark" | "light"
    "vault_default": "vault-teste",
    "model_default": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "timeout": 60,                       # seconds
    "auto_save": True,
    "max_history": 1000,
    "fallback_models": [
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-ultra",
    ],
    "health_check_timeout": 5,
}


class Config:
    """Carrega, valida e grava ~/.mnemo/config.json."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
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

    def validate(self, vault_root: Optional[str] = None) -> List[str]:
        """Valida configuração e retorna lista de erros (vazia se OK)."""
        errors: List[str] = []

        # API key
        if not os.environ.get("NVIDIA_API_KEY"):
            errors.append("NVIDIA_API_KEY não definida no .env ou variáveis de ambiente")

        # Vault
        vault_path = Path(vault_root or self.get("vault_default", ""))
        if not vault_path.exists():
            errors.append(f"Vault não existe: {vault_path}")
        else:
            # Verifica pastas permitidas
            try:
                from mnemo.permissoes import Permissoes
                perms = Permissoes(str(vault_path))
                for p in perms.raizes_permitidas():
                    if not p.exists():
                        errors.append(f"Pasta permitida não existe: {p}")
            except Exception as e:
                errors.append(f"Erro ao validar permissões do vault: {e}")

        # Timeout válido
        timeout = self.get("timeout", 60)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append(f"Timeout inválido: {timeout} (deve ser número > 0)")

        # Fallback models
        fallbacks = self.get("fallback_models", [])
        if not isinstance(fallbacks, list):
            errors.append("fallback_models deve ser uma lista")

        return errors