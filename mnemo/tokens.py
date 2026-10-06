"""Contador de tokens para rastreamento de uso da API."""

from typing import Dict


class TokenCounter:
    """Rastreia tokens de prompt, completion e total por turno e por sessão."""

    def __init__(self) -> None:
        self.session_prompt = 0
        self.session_completion = 0
        self.session_total = 0
        self.turn_prompt = 0
        self.turn_completion = 0
        self.turn_total = 0

    def add(self, usage: Dict[str, int]) -> None:
        """Adiciona tokens do uso atual ao contador."""
        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        t = usage.get("total_tokens", p + c)
        self.turn_prompt += p
        self.turn_completion += c
        self.turn_total += t
        self.session_prompt += p
        self.session_completion += c
        self.session_total += t

    def reset_turn(self) -> None:
        """Reseta contadores do turno atual (mantém sessão)."""
        self.turn_prompt = 0
        self.turn_completion = 0
        self.turn_total = 0

    def reset_session(self) -> None:
        """Reseta todos os contadores."""
        self.session_prompt = 0
        self.session_completion = 0
        self.session_total = 0
        self.reset_turn()

    def get_turn_summary(self) -> str:
        """Retorna resumo formatado do turno atual."""
        return f"Turn: {self.turn_total} tokens (prompt: {self.turn_prompt}, completion: {self.turn_completion})"

    def get_session_summary(self) -> str:
        """Retorna resumo formatado da sessão completa."""
        return f"Session: {self.session_total} tokens (prompt: {self.session_prompt}, completion: {self.session_completion})"

    def get_totals(self) -> Dict[str, int]:
        """Retorna totais brutos para serialização."""
        return {
            "session_prompt": self.session_prompt,
            "session_completion": self.session_completion,
            "session_total": self.session_total,
            "turn_prompt": self.turn_prompt,
            "turn_completion": self.turn_completion,
            "turn_total": self.turn_total,
        }