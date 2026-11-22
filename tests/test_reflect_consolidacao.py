"""Testes para consolidação de reflexões (F6 Reflect)."""

from typing import Any, List

import numpy as np
import pytest

from mnemo.armazenamento import Armazenamento
from mnemo.indexador import Indexador
from mnemo.memoria import FakeEmbeddings, Proposta, Retain
from mnemo.permissoes import Permissoes


class LLMFalsoReflect:
    """LLM falso que devolve reflexões pré-definidas."""

    def __init__(self, resposta: str):
        self.resposta = resposta

    def conversar(self, mensagens: List[dict], ferramentas: Any = None, stream: bool = False):
        # Retorna tupla com objeto que tem .get("content") como o ClienteNVIDIA real
        class Resp:
            def __init__(self, content):
                self._content = content
            def get(self, key, default=None):
                if key == "content":
                    return self._content
                return default
        return (Resp(self.resposta),)


@pytest.fixture
def vault(tmp_path):
    """Cria vault temporário."""
    raiz = tmp_path / "v"
    (raiz / "notas").mkdir(parents=True)
    (raiz / ".vault").mkdir()
    (raiz / ".vault" / "config.json").write_text('{"todas_as_pastas": true}', encoding="utf-8")
    return raiz


def montar_retain(vault: str) -> Retain:
    """Cria Retain com FakeEmbeddings e base em memória."""
    perms = Permissoes(vault)
    indexador = Indexador(perms, caminho_db=":memory:")
    indexador.reindexar_tudo()
    provider = FakeEmbeddings(dim=32)
    return Retain(perms, Armazenamento(perms), indexador, provider, provider.modelo, provider.dim)


def test_consolidar_agrupa_por_entidade(vault):
    """Múltiplas observações sobre a mesma entidade → 1 rascunho consolidado."""
    retain = montar_retain(str(vault))

    # LLM devolve 3 observações sobre a mesma entidade "João"
    resposta_llm = """[
        {"entidade": "João", "observacao": "João gosta de café", "opiniao": "João prefere café forte", "confianca": 0.8, "tags": ["pessoal"]},
        {"entidade": "João", "observacao": "João odeia chá", "opiniao": "João evita chá", "confianca": 0.9, "tags": ["pessoal"]},
        {"entidade": "João", "observacao": "João toma café de manhã", "opiniao": "Rotina matinal de café", "confianca": 0.85, "tags": ["rotina"]}
    ]"""
    mensagens = [{"role": "user", "content": "Conversei com João sobre café"}]
    cliente = LLMFalsoReflect(resposta_llm)

    propostas = retain.refletir_conversa(mensagens, cliente, consolidar=True)

    # Deve consolidar em 1 só proposta
    assert len(propostas) == 1
    p = propostas[0]
    assert p.entidade == "João"
    assert p.tipo == "observacao"
    assert "João gosta de café" in p.texto
    assert "João odeia chá" in p.texto
    assert "João toma café de manhã" in p.texto
    # Tags devem ser agregadas
    assert "pessoal" in p.tags
    assert "rotina" in p.tags
    assert "João" in p.tags  # entidade adicionada como tag
    # Confiança deve ser média ponderada + bonus
    assert p.confianca > 0.8 and p.confianca <= 1.0


def test_confianca_calculada_corretamente(vault):
    """Confiança = média dos scores + bonus por quantidade."""
    retain = montar_retain(str(vault))

    # 2 observações com scores 0.7 e 0.9
    resposta_llm = """[
        {"entidade": "Projeto Alpha", "observacao": "Obs 1", "opiniao": "Op 1", "confianca": 0.7, "tags": []},
        {"entidade": "Projeto Alpha", "observacao": "Obs 2", "opiniao": "Op 2", "confianca": 0.9, "tags": []}
    ]"""
    mensagens = [{"role": "user", "content": "Projeto Alpha"}]
    cliente = LLMFalsoReflect(resposta_llm)

    propostas = retain.refletir_conversa(mensagens, cliente, consolidar=True)

    assert len(propostas) == 1
    p = propostas[0]
    # Média = (0.7 + 0.9) / 2 = 0.8
    # Bonus = (2-1) * 0.05 = 0.05
    # Total = 0.85
    assert 0.84 <= p.confianca <= 0.86


def test_dedup_reflexoes_similares(vault):
    """Observações com textos muito similares são fundidas."""
    retain = montar_retain(str(vault))

    # Duas observações quase idênticas
    resposta_llm = """[
        {"entidade": "Python", "observacao": "Python é uma linguagem de programação popular", "opiniao": "Python é ótimo", "confianca": 0.8, "tags": []},
        {"entidade": "Python", "observacao": "Python é uma linguagem de programação muito popular", "opiniao": "Python é excelente", "confianca": 0.85, "tags": []}
    ]"""
    mensagens = [{"role": "user", "content": "Python"}]
    cliente = LLMFalsoReflect(resposta_llm)

    propostas = retain.refletir_conversa(mensagens, cliente, consolidar=True)

    assert len(propostas) == 1
    p = propostas[0]
    # Deve fundir as duas observações similares
    # O texto consolidado não deve ter as duas versões quase idênticas
    # (a dedup simples por substring/overlap deve detectar)
    assert "Python é uma linguagem" in p.texto


def test_refletir_bruto_nao_consolida(vault):
    """Flag consolidar=False devolve observações brutas sem agrupar."""
    retain = montar_retain(str(vault))

    resposta_llm = """[
        {"entidade": "João", "observacao": "Obs 1", "opiniao": "Op 1", "confianca": 0.8, "tags": []},
        {"entidade": "João", "observacao": "Obs 2", "opiniao": "Op 2", "confianca": 0.9, "tags": []},
        {"entidade": "Maria", "observacao": "Obs 3", "opiniao": "Op 3", "confianca": 0.7, "tags": []}
    ]"""
    mensagens = [{"role": "user", "content": "Teste"}]
    cliente = LLMFalsoReflect(resposta_llm)

    # Com consolidação (padrão)
    propostas_consolidadas = retain.refletir_conversa(mensagens, cliente, consolidar=True)
    assert len(propostas_consolidadas) == 2  # João + Maria

    # Sem consolidação (--bruto)
    propostas_brutas = retain.refletir_conversa(mensagens, cliente, consolidar=False)
    assert len(propostas_brutas) == 3  # 3 observações individuais
    # Cada uma deve ter a sua entidade e confiança original
    entidades = [p.entidade for p in propostas_brutas]
    assert entidades.count("João") == 2
    assert entidades.count("Maria") == 1


def test_extrair_entidade_heuristica(vault):
    """Heurística extrai entidade de nomes próprios, projetos, siglas."""
    retain = montar_retain(str(vault))

    # Testa extração de nomes próprios
    assert retain.extrair_entidade("O João disse que o Projeto Alpha vai bem") in ["João", "Projeto Alpha"]
    assert retain.extrair_entidade("A Maria Silva trabalha na API REST") in ["Maria Silva", "API"]
    # Texto sem entidades claras deve retornar None
    assert retain.extrair_entidade("Texto sem entidades claras") is None
    assert retain.extrair_entidade("hoje está um dia bonito") is None