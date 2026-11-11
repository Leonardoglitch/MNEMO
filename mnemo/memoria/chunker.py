"""Chunking de notas Markdown em trechos por cabeçalhos e parágrafos.

Função pura, testável sem rede nem base de dados.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from mnemo.indexador import separar_frontmatter


# Regex para cabeçalhos markdown (# ## ###)
CABECALHO_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Divide por parágrafos (linhas vazias)
PARAGRAFO_RE = re.compile(r"\n\s*\n")

# Limites
TRECHO_TOKENS_ALVO = 300
TRECHO_TOKENS_MAX = 400
SOBREPOSICAO_FRASES = 2


def estimar_tokens(texto: str, chars_por_token: float = 3.5) -> int:
    """Estimativa rápida: ~3.5 chars/token em português."""
    return max(1, int(len(texto) / chars_por_token))


def partir_por_frases(texto: str) -> List[str]:
    """Divide texto em frases (heurística simples)."""
    # Protege abreviações comuns
    protegido = texto.replace("e.g.", "e.g.<PROT>").replace("i.e.", "i.e.<PROT>")
    frases = re.split(r"(?<=[.!?])\s+", protegido)
    return [f.replace("<PROT>", ".") for f in frases if f.strip()]


def chunker(
    titulo_nota: str,
    corpo: str,
    *,
    trecho_tokens_alvo: int = TRECHO_TOKENS_ALVO,
    trecho_tokens_max: int = TRECHO_TOKENS_MAX,
    sobreposicao_frases: int = SOBREPOSICAO_FRASES,
    chars_por_token: float = 3.5,
) -> List[Tuple[str, str, int]]:
    """
    Parte o corpo de uma nota em trechos.

    Returns:
        Lista de (titulo_secao, texto_trecho, tokens_estimados)
    """
    trechos: List[Tuple[str, str, int]] = []

    # Encontra cabeçalhos
    matches = list(CABECALHO_RE.finditer(corpo))
    if not matches:
        # Nota sem cabeçalhos: trata tudo como uma secção
        trechos.extend(_partir_secao("", corpo, trecho_tokens_alvo, trecho_tokens_max,
                                     sobreposicao_frases, chars_por_token))
        return trechos

    # Processa cada secção entre cabeçalhos
    for i, match in enumerate(matches):
        nivel = len(match.group(1))
        titulo_secao = match.group(2).strip()
        inicio = match.end()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(corpo)
        texto_secao = corpo[inicio:fim].strip()

        if texto_secao:
            trechos.extend(_partir_secao(
                titulo_secao, texto_secao, trecho_tokens_alvo, trecho_tokens_max,
                sobreposicao_frases, chars_por_token
            ))

    return trechos


def _partir_secao(
    titulo_secao: str,
    texto: str,
    trecho_tokens_alvo: int,
    trecho_tokens_max: int,
    sobreposicao_frases: int,
    chars_por_token: float,
) -> List[Tuple[str, str, int]]:
    """Parte uma secção em trechos respeitando limite de tokens."""
    if not texto.strip():
        return []

    # Estima tokens do texto todo
    tokens_total = estimar_tokens(texto, chars_por_token)
    if tokens_total <= trecho_tokens_max:
        return [(titulo_secao, texto, tokens_total)]

    # Divide em frases
    frases = partir_por_frases(texto)
    trechos: List[Tuple[str, str, int]] = []
    buffer: List[str] = []
    tokens_buffer = 0

    for frase in frases:
        tokens_frase = estimar_tokens(frase, chars_por_token)

        # Se a frase sozinha excede o max, parte-a forçadamente
        if tokens_frase > trecho_tokens_max:
            if buffer:
                trechos.append((titulo_secao, " ".join(buffer), tokens_buffer))
                buffer = []
                tokens_buffer = 0
            # Parte frase longa em pedaços de ~trecho_tokens_max
            palavras = frase.split()
            chunk_palavras = []
            chunk_tokens = 0
            for palavra in palavras:
                t = estimar_tokens(palavra + " ", chars_por_token)
                if chunk_tokens + t > trecho_tokens_max and chunk_palavras:
                    trechos.append((titulo_secao, " ".join(chunk_palavras), chunk_tokens))
                    chunk_palavras = []
                    chunk_tokens = 0
                chunk_palavras.append(palavra)
                chunk_tokens += t
            if chunk_palavras:
                trechos.append((titulo_secao, " ".join(chunk_palavras), chunk_tokens))
            continue

        # Acumula até atingir alvo
        if tokens_buffer + tokens_frase > trecho_tokens_alvo and buffer:
            trechos.append((titulo_secao, " ".join(buffer), tokens_buffer))
            # Sobreposição: mantém últimas N frases
            if sobreposicao_frases > 0:
                buffer = buffer[-sobreposicao_frases:]
                tokens_buffer = sum(estimar_tokens(f, chars_por_token) for f in buffer)
            else:
                buffer = []
                tokens_buffer = 0

        buffer.append(frase)
        tokens_buffer += tokens_frase

    if buffer:
        trechos.append((titulo_secao, " ".join(buffer), tokens_buffer))

    return trechos