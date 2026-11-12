"""Modelos de dados para o subsistema de memória (Retain/Recall/Reflect)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class Trecho:
    """Um trecho de texto extraído de uma nota."""
    id: Optional[int] = None
    nota_id: int = 0
    ordem: int = 0
    titulo_secao: str = ""
    texto: str = ""
    tokens: int = 0
    hash: str = ""

    def prefixo_embedding(self, titulo_nota: str) -> str:
        """Texto a embeber: 'Título da Nota › Título da Secção' + corpo."""
        partes = [titulo_nota]
        if self.titulo_secao:
            partes.append(self.titulo_secao)
        return " › ".join(partes) + "\n\n" + self.texto


@dataclass(slots=True)
class Resultado:
    """Resultado de uma pesquisa (lexical, semântica ou híbrida)."""
    nota_id: int
    nota_caminho: str
    nota_titulo: str
    trecho_id: int
    trecho_ordem: int
    titulo_secao: str
    texto: str
    score: float
    origem: dict[str, Any]  # {"lex": rank, "sem": rank, "explicar": {...}}
    citacao: str
    modo: str = "hibrido"  # "hibrido" | "lexical"


@dataclass(slots=True)
class Proposta:
    """Proposta de nota/memória gerada a partir de conversa (LLM)."""
    tipo: str  # facto | experiencia | decisao | preferencia | observacao
    titulo: str
    texto: str
    tags: list[str] = field(default_factory=list)
    nota_existente_id: Optional[int] = None  # se for atualização
    score_duplicado: float = 0.0


@dataclass(slots=True)
class ContextoReflect:
    """Contexto injetado no prompt do LLM no Reflect."""
    consulta: str
    resultados: list[Resultado]
    budget_tokens: int
    tokens_estimados: int

    def formatar_para_prompt(self) -> str:
        if not self.resultados:
            return "(memória sem resultados relevantes)"

        linhas = ["=== MEMÓRIA RELEVANTE ==="]
        for i, r in enumerate(self.resultados, 1):
            linhas.append(f"[{i}] [[{r.nota_caminho}]] {r.titulo_secao or r.nota_titulo}")
            linhas.append(f"    {r.texto[:300]}{'...' if len(r.texto) > 300 else ''}")
        linhas.append(f"\n[Budget: {self.tokens_estimados}/{self.budget_tokens} tokens]")
        return "\n".join(linhas)


@dataclass(slots=True)
class RelatorioRetain:
    """Relatório de uma execução de retain_vault."""
    novas: int = 0
    atualizadas: int = 0
    inalteradas: int = 0
    removidas: int = 0
    trechos_novos: int = 0
    trechos_atualizados: int = 0
    embeddings_novos: int = 0
    embeddings_pendentes: list[str] = field(default_factory=list)  # caminhos com falha
    erros: list[str] = field(default_factory=list)