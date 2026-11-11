"""Estimativa de tokens para textos em português.

Não usa o TokenCounter (que só soma usage da API).
Heurística calibrável: chars_por_token (default 3.5 para PT).
"""

from __future__ import annotations


# Valor default calibrado para português (pode ser ajustado com usage real da API)
CHARS_POR_TOKEN_DEFAULT = 3.5


def estimar_tokens(texto: str, chars_por_token: float = CHARS_POR_TOKEN_DEFAULT) -> int:
    """
    Estima número de tokens num texto.

    Args:
        texto: Texto a estimar
        chars_por_token: Heurística (chars/token). Default 3.5 para português.

    Returns:
        Número estimado de tokens (mínimo 1 se texto não vazio)
    """
    if not texto or not texto.strip():
        return 0
    return max(1, int(len(texto) / chars_por_token))


def calibrar_chars_por_token(textos: list[str], tokens_reais: list[int]) -> float:
    """
    Calibra chars_por_token a partir de amostras com tokens reais (usage da API).

    Args:
        textos: Lista de textos
        tokens_reais: Lista de token counts reais (prompt_tokens da API)

    Returns:
        chars_por_token calibrado
    """
    if len(textos) != len(tokens_reais) or not textos:
        return CHARS_POR_TOKEN_DEFAULT

    total_chars = sum(len(t) for t in textos)
    total_tokens = sum(tokens_reais)
    if total_tokens == 0:
        return CHARS_POR_TOKEN_DEFAULT

    return total_chars / total_tokens