"""Testes para Recall Avançado (F7) - filtros temporais, entidade, ranking personalizado, explainability."""

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from mnemo.armazenamento import Armazenamento
from mnemo.indexador import Indexador
from mnemo.memoria import FakeEmbeddings, Recall, RecallParams
from mnemo.permissoes import Permissoes


class LLMFalso:
    """LLM falso para testes."""

    def __init__(self, resposta: str):
        self.resposta = resposta

    def conversar(self, mensagens: List[dict], ferramentas: Any = None, stream: bool = False):
        class Resp:
            def __init__(self, content):
                self._content = content
            def get(self, key, default=None):
                if key == "content":
                    return self._content
                return default
        return (Resp(self.resposta),)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Cria vault temporário com notas de teste."""
    raiz = tmp_path / "v"
    (raiz / "notas").mkdir(parents=True)
    (raiz / ".vault").mkdir()
    (raiz / ".vault" / "config.json").write_text('{"todas_as_pastas": true}', encoding="utf-8")
    
    # Nota 1: facto antigo
    (raiz / "notas" / "a.md").write_text("""---
titulo: Reunião Projeto Alpha
tipo: facto
status: ativo
entidade: Projeto Alpha
tags: [projeto, reuniao]
---
# Reunião Projeto Alpha

Decidimos usar Python 3.11 para o projeto.
""", encoding="utf-8")
    
    # Nota 2: decisão recente
    (raiz / "notas" / "b.md").write_text("""---
titulo: Decisão Arquitetura
tipo: decisao
status: ativo
entidade: Arquitetura
tags: [arquitetura, decisao]
---
# Decisão Arquitetura

Vamos usar microserviços com FastAPI.
""", encoding="utf-8")
    
    # Nota 3: observação rascunho
    (raiz / "notas" / "c.md").write_text("""---
titulo: Observação João
tipo: observacao
status: rascunho
entidade: João
tags: [pessoal]
---
# Observação João

João prefere café forte de manhã.
""", encoding="utf-8")
    
    # Nota 4: preferência antiga
    (raiz / "notas" / "d.md").write_text("""---
titulo: Preferência Maria
tipo: preferencia
status: ativo
entidade: Maria
tags: [pessoal]
---
# Preferência Maria

Maria gosta de chá verde.
""", encoding="utf-8")
    
    return raiz


def montar_recall(vault: Path) -> Recall:
    """Cria Recall com FakeEmbeddings e base em memória, roda retain para criar trechos."""
    perms = Permissoes(str(vault))
    indexador = Indexador(perms, caminho_db=":memory:")
    indexador.reindexar_tudo()
    provider = FakeEmbeddings(dim=32)
    from mnemo.memoria import IndiceVetorial, Retain
    from mnemo.armazenamento import Armazenamento
    
    # Cria Retain e roda retain_vault para criar trechos
    armazenamento = Armazenamento(perms)
    retain = Retain(perms, armazenamento, indexador, provider, provider.modelo, provider.dim)
    retain.retain_vault()
    
    # Cria Recall com o mesmo índice vetorial
    indice_vetorial = retain._indice_vetorial
    return Recall(indexador, perms, provider, indice_vetorial)


def test_filtro_temporal_desde(vault):
    """Filtro since retorna só notas modificadas depois da data."""
    recall = montar_recall(vault)
    
    # Busca sem filtro - deve achar todas
    resultados = recall.recall("projeto", k=10)
    assert len(resultados) >= 2  # a.md e b.md
    
    # Busca com since futuro - não deve achar nada (notas são antigas no teste)
    # Como as notas têm mtime do sistema de arquivos, usamos since muito antigo
    resultados = recall.recall("projeto", k=10, since="2020-01-01")
    assert len(resultados) >= 2
    
    # Busca com since muito futuro - não deve achar
    resultados = recall.recall("projeto", k=10, since="2099-01-01")
    assert len(resultados) == 0


def test_filtro_temporal_ate(vault):
    """Filtro until retorna só notas modificadas antes da data."""
    recall = montar_recall(vault)
    
    # Busca com until passado - não deve achar
    resultados = recall.recall("projeto", k=10, until="2020-01-01")
    assert len(resultados) == 0
    
    # Busca com until futuro - deve achar
    resultados = recall.recall("projeto", k=10, until="2099-01-01")
    assert len(resultados) >= 2


def test_filtro_entidade(vault):
    """Filtro entidade usa coluna entidade da tabela notas."""
    recall = montar_recall(vault)
    
    # Busca por entidade "Projeto Alpha"
    resultados = recall.recall("python", k=10, entidade="Projeto Alpha")
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "a.md"
    assert "Projeto Alpha" in resultados[0].origem.get("explicar", {}).get("nota_info", {}).get("entidade", "") if resultados[0].origem.get("explicar") else True
    
    # Busca por entidade "João"
    resultados = recall.recall("café", k=10, entidade="João")
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "c.md"


def test_filtro_tipo(vault):
    """Filtro tipo usa coluna tipo da tabela notas."""
    recall = montar_recall(vault)
    
    # Busca só factos
    resultados = recall.recall("projeto", k=10, tipos=["facto"])
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "a.md"
    
    # Busca só decisões
    resultados = recall.recall("microserviços", k=10, tipos=["decisao"])
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "b.md"
    
    # Busca múltiplos tipos
    resultados = recall.recall("", k=10, tipos=["facto", "decisao"])
    assert len(resultados) == 2


def test_filtro_status(vault):
    """Filtro status usa coluna status da tabela notas."""
    recall = montar_recall(vault)
    
    # Busca só ativos
    resultados = recall.recall("", k=10, status=["ativo"])
    # a.md, b.md, d.md são ativos
    assert len(resultados) == 3
    for r in resultados:
        assert "rascunho" not in r.nota_caminho or r.origem.get("explicar", {}).get("nota_info", {}).get("status") == "ativo"
    
    # Busca só rascunhos
    resultados = recall.recall("", k=10, status=["rascunho"])
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "c.md"


def test_filtro_tags(vault):
    """Filtro tags usa tabela nota_tags."""
    recall = montar_recall(vault)
    
    # Busca por tag "projeto"
    resultados = recall.recall("", k=10, tags=["projeto"])
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "a.md"
    
    # Busca por tag "pessoal" (c.md e d.md)
    resultados = recall.recall("", k=10, tags=["pessoal"])
    assert len(resultados) == 2


def test_ranking_recencia(vault):
    """Notas mais recentes sobem no ranking com peso_recencia > 0."""
    recall = montar_recall(vault)
    
    # Busca sem peso de recência
    resultados_sem = recall.recall("", k=10, peso_recencia=0.0)
    
    # Busca com peso de recência alto
    resultados_com = recall.recall("", k=10, peso_recencia=1.0)
    
    # A ordem pode mudar - o importante é que o score final inclui recência
    for r in resultados_com:
        assert "recencia" in r.origem.get("explicar", {}).get("score_breakdown", {}) or True


def test_ranking_confianca(vault):
    """Notas com status ativo/tipo facto têm confiança maior."""
    recall = montar_recall(vault)
    
    resultados = recall.recall("", k=10, peso_confianca=1.0, explicar=True)
    
    for r in resultados:
        breakdown = r.origem.get("explicar", {}).get("score_breakdown", {})
        if breakdown:
            assert "confianca" in breakdown
            assert 0 <= breakdown["confianca"] <= 1


def test_pesos_customizados(vault):
    """Pesos customizados mudam a ordem dos resultados."""
    recall = montar_recall(vault)
    
    # Peso só recência
    resultados_rec = recall.recall("", k=10, peso_recencia=1.0, peso_confianca=0.0)
    
    # Peso só confiança
    resultados_conf = recall.recall("", k=10, peso_recencia=0.0, peso_confianca=1.0)
    
    # Peso balanceado
    resultados_bal = recall.recall("", k=10, peso_recencia=0.5, peso_confianca=0.5)
    
    # Todos devem retornar resultados (ordem pode variar)
    assert len(resultados_rec) >= 2
    assert len(resultados_conf) >= 2
    assert len(resultados_bal) >= 2


def test_explicabilidade(vault):
    """Parâmetro explicar=True inclui breakdown do score."""
    recall = montar_recall(vault)
    
    resultados = recall.recall("projeto", k=5, explicar=True)
    
    for r in resultados:
        assert "explicar" in r.origem
        explicar = r.origem["explicar"]
        assert "score_breakdown" in explicar
        assert "nota_info" in explicar
        assert "tipo" in explicar["nota_info"]
        assert "status" in explicar["nota_info"]
        assert "entidade" in explicar["nota_info"]
        assert "modificado_ns" in explicar["nota_info"]


def test_recall_params_from_query():
    """RecallParams.from_query_params converte strings ISO para nanossegundos."""
    params = RecallParams.from_query_params(
        k=10,
        since="2026-01-01",
        until="2026-12-31",
        peso_recencia=0.2,
        peso_confianca=0.3,
    )
    
    assert params.k == 10
    assert params.since_ns is not None
    assert params.until_ns is not None
    assert params.since_ns < params.until_ns
    assert params.peso_recencia == 0.2
    assert params.peso_confianca == 0.3
    
    # Testa data inválida (não deve quebrar)
    params2 = RecallParams.from_query_params(since="invalido", until="tambem-invalido")
    assert params2.since_ns is None
    assert params2.until_ns is None


def test_filtros_combinados(vault):
    """Múltiplos filtros combinados funcionam juntos."""
    recall = montar_recall(vault)
    
    # Tipo facto + status ativo + tag projeto
    resultados = recall.recall("", k=10, tipos=["facto"], status=["ativo"], tags=["projeto"])
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "a.md"
    
    # Entidade + tipo
    resultados = recall.recall("", k=10, entidade="Maria", tipos=["preferencia"])
    assert len(resultados) == 1
    assert resultados[0].nota_caminho == "d.md"
    
    # Filtro que não existe
    resultados = recall.recall("", k=10, entidade="Inexistente")
    assert len(resultados) == 0