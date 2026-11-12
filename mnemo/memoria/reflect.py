"""Reflect: construção de contexto para LLM + verificação de citações."""

from __future__ import annotations

import re
from typing import List, Optional

from mnemo.memoria import Resultado, ContextoReflect


# Regex para citações no formato [[Nome_da_Nota]] ou [[Nota|alias]] ou [[Nota#secção]]
CITACAO_RE = re.compile(r"\[\[([^\]]+)\]\]")


def normalizar_citacao(citacao_bruta: str) -> str:
    """
    Normaliza uma citação extraída:
    - Remove alias (tudo depois de |)
    - Remove secção (tudo depois de #)
    - Remove .md
    - Lowercase, strip
    """
    # Remove alias e secção
    alvo = citacao_bruta.split("|")[0].split("#")[0]
    # Remove .md
    if alvo.endswith(".md"):
        alvo = alvo[:-3]
    return alvo.strip().lower()


def extrair_citacoes(texto: str) -> List[str]:
    """Extrai todas as citações [[...]] do texto (normalizadas)."""
    matches = CITACAO_RE.findall(texto)
    return [normalizar_citacao(m) for m in matches]


def verificar_citacoes(resposta: str, ctx: ContextoReflect) -> List[str]:
    """
    Verifica se as citações na resposta correspondem a notas do contexto.
    
    Returns:
        Lista de citações inválidas (não estão no contexto fornecido).
    """
    citacoes_resposta = set(extrair_citacoes(resposta))
    if not citacoes_resposta:
        return []

    # Constrói conjunto de caminhos válidos do contexto
    caminhos_validos = set()
    for r in ctx.resultados:
        caminho_norm = normalizar_citacao(r.nota_caminho)
        caminhos_validos.add(caminho_norm)

    # Citações que NÃO estão no contexto
    invalidas = [c for c in citacoes_resposta if c not in caminhos_validos]
    return sorted(invalidas)


class Reflect:
    """Constrói contexto para o LLM e verifica citações."""

    def __init__(
        self,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.prompt_template = prompt_template or self._template_padrao()

    def _template_padrao(self) -> str:
        return """=== INSTRUÇÕES DE MEMÓRIA ===
Tens acesso a uma memória persistente sobre o utilizador. Usa-a para responder com factos, preferências e decisões anteriores.

REGRAS:
1. Usa APENAS a memória fornecida abaixo para factos sobre o utilizador.
2. CITA SEMPRE a fonte usando o formato [[caminho/da/nota]] no final de cada afirmação baseada na memória.
3. Se a memória NÃO tem a informação, diz claramente: "A memória não tem informação sobre X."
4. Não inventes factos que não estejam na memória.
5. Se houver contradição entre memória e conhecimento geral, a memória prevalece para factos pessoais.

{contexto}

=== FIM DA MEMÓRIA ==="""

    def construir_contexto(
        self,
        consulta: str,
        resultados: List[Resultado],
        budget_tokens: int = 2000,
    ) -> ContextoReflect:
        """Constrói ContextoReflect a partir dos resultados do Recall."""
        # Estima tokens do contexto
        tokens_estimados = 0
        for r in resultados:
            tokens_estimados += len(r.texto) // 4  # estimativa grosseira

        ctx = ContextoReflect(
            consulta=consulta,
            resultados=resultados,
            budget_tokens=budget_tokens,
            tokens_estimados=tokens_estimados,
        )
        return ctx

    def formatar_para_prompt(self, ctx: ContextoReflect) -> str:
        """Formata o contexto para injeção no prompt do LLM."""
        contexto_str = ctx.formatar_para_prompt()
        return self.prompt_template.format(contexto=contexto_str)

    def verificar(self, resposta: str, ctx: ContextoReflect) -> List[str]:
        """Verifica citações na resposta contra o contexto."""
        return verificar_citacoes(resposta, ctx)