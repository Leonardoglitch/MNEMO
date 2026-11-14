"""Testes de integração da API FastAPI do subsistema de memória.

Tudo é injetado: base de dados em memória, FakeEmbeddings e um LLM falso. Nada toca no disco
(fora do tmp_path), na rede ou no .env.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest
from fastapi.testclient import TestClient

from api_memoria import Servicos, construir_servicos, criar_app
from mnemo.memoria import FakeEmbeddings


class LLMFalso:
    """Imita ClienteNVIDIA.conversar(); devolve sempre a resposta JSON configurada."""

    def __init__(self, resposta: Any) -> None:
        self.resposta = resposta
        self.chamadas = 0

    def conversar(self, mensagens: List[Dict[str, Any]], ferramentas: Any = None, stream: bool = False) -> Tuple[Dict[str, Any], Dict[str, int]]:
        self.chamadas += 1
        return {"role": "assistant", "content": json.dumps(self.resposta)}, {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }


MEMORIA_EXTRAIDA = [
    {"tipo": "preferencia", "titulo": "Preferência de café", "texto": "O utilizador prefere café sem açúcar.", "tags": ["bebidas"]}
]
REFLEXAO = [
    {
        "entidade": "Projeto X",
        "observacao": "Tem poucos recursos.",
        "opiniao": "Pode falhar por falta de recursos.",
        "confianca": 0.7,
        "tags": ["projetos"],
        "fonte": "conversa",
    }
]


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    raiz = tmp_path / "vault"
    (raiz / "notas").mkdir(parents=True)
    (raiz / ".vault").mkdir()
    (raiz / ".vault" / "config.json").write_text('{"todas_as_pastas": true}', encoding="utf-8")
    (raiz / "notas" / "cafe.md").write_text(
        "---\ntags: [bebidas]\n---\n# Café\n\nPrefiro café sem açúcar de manhã.\n", encoding="utf-8"
    )
    (raiz / "notas" / "projeto.md").write_text(
        "# Projeto\n\nO projeto usa SQLite e FTS5 para a pesquisa lexical.\n", encoding="utf-8"
    )
    return raiz


@pytest.fixture
def llm() -> LLMFalso:
    return LLMFalso(MEMORIA_EXTRAIDA)


@pytest.fixture
def servicos(vault: Path, llm: LLMFalso) -> Iterator[Servicos]:
    s = construir_servicos(
        str(vault), provider=FakeEmbeddings(dim=64), cliente_llm=llm, caminho_db=":memory:"
    )
    yield s
    s.fechar()


@pytest.fixture
def client(servicos: Servicos) -> Iterator[TestClient]:
    # `with` corre o lifespan; como os serviços foram injetados, não lê .env nem abre bases em disco.
    with TestClient(criar_app(servicos)) as c:
        yield c


@pytest.fixture
def client_sem_llm(vault: Path) -> Iterator[TestClient]:
    s = construir_servicos(str(vault), provider=FakeEmbeddings(dim=64), cliente_llm=None, caminho_db=":memory:")
    with TestClient(criar_app(s)) as c:
        yield c
    s.fechar()


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "memoria_inicializada": True}


def test_sem_servicos_devolve_503() -> None:
    # Sem servicos e sem lifespan (TestClient sem `with`): o endpoint recusa em vez de rebentar.
    c = TestClient(criar_app(None))
    assert c.get("/memoria/stats").status_code == 503


def test_api_nao_cria_base_em_disco(client: TestClient, vault: Path) -> None:
    client.post("/memoria/retain/vault")
    assert not (vault / ".vault" / "index.db").exists()


class TestRetainVault:
    def test_primeira_sincronizacao(self, client: TestClient) -> None:
        d = client.post("/memoria/retain/vault").json()
        assert d["novas"] == 2
        assert d["trechos_novos"] >= 2
        assert d["embeddings_novos"] >= 2
        assert d["erros"] == [] and d["embeddings_pendentes"] == []

    def test_segunda_sincronizacao_e_idempotente(self, client: TestClient) -> None:
        client.post("/memoria/retain/vault")
        d = client.post("/memoria/retain/vault").json()
        assert d["novas"] == 0 and d["inalteradas"] == 2 and d["embeddings_novos"] == 0

    def test_reparar(self, client: TestClient) -> None:
        assert client.post("/memoria/retain/vault", params={"reparar": True}).status_code == 200


class TestRecall:
    def test_encontra_nota_depois_do_retain(self, client: TestClient) -> None:
        client.post("/memoria/retain/vault")
        d = client.get("/memoria/recall", params={"consulta": "café açúcar"}).json()
        assert d["resultados"], "devia encontrar a nota do café"
        assert d["resultados"][0]["nota_caminho"] == "cafe.md"
        campos = {"nota_id", "nota_caminho", "nota_titulo", "trecho_id", "trecho_ordem", "titulo_secao", "texto", "score", "origem", "citacao", "modo"}
        assert campos <= set(d["resultados"][0])

    def test_usa_vetores_novos_sem_reiniciar(self, client: TestClient) -> None:
        """O Recall partilha o índice vetorial do Retain: depois do retain fica híbrido."""
        client.post("/memoria/retain/vault")
        d = client.get("/memoria/recall", params={"consulta": "SQLite FTS5"}).json()
        assert d["modo"] == "hibrido"

    def test_sem_dados_devolve_vazio(self, client: TestClient) -> None:
        d = client.get("/memoria/recall", params={"consulta": "café"}).json()
        assert d["resultados"] == [] and d["modo"] == "vazio"

    @pytest.mark.parametrize(
        "params",
        [{}, {"consulta": "x", "k": 0}, {"consulta": "x", "k": 51}, {"consulta": "x", "k": -1},
         {"consulta": "x", "budget_tokens": -1}, {"consulta": "x", "budget_tokens": 20000}],
    )
    def test_parametros_invalidos(self, client: TestClient, params: Dict[str, Any]) -> None:
        assert client.get("/memoria/recall", params=params).status_code == 422


class TestRetainConversa:
    MSGS = {"mensagens": [{"role": "user", "content": "Prefiro café sem açúcar."}]}

    def test_cria_rascunho(self, client: TestClient, vault: Path) -> None:
        r = client.post("/memoria/retain/conversa", json=self.MSGS)
        assert r.status_code == 200
        d = r.json()
        assert [p["titulo"] for p in d] == ["Preferência de café"]
        rascunho = vault / "notas" / "memoria" / "preferência-de-café.md"
        assert rascunho.exists()
        assert "status: rascunho" in rascunho.read_text(encoding="utf-8")

    def test_payload_invalido(self, client: TestClient) -> None:
        assert client.post("/memoria/retain/conversa", json={}).status_code == 422

    def test_sem_llm_devolve_503(self, client_sem_llm: TestClient) -> None:
        assert client_sem_llm.post("/memoria/retain/conversa", json=self.MSGS).status_code == 503


class TestRefletir:
    MSGS = {"mensagens": [{"role": "user", "content": "Acho que o projeto X vai falhar."}]}

    def test_consolida_com_uma_unica_chamada_ao_llm(self, servicos: Servicos, client: TestClient) -> None:
        servicos.cliente_llm = LLMFalso(REFLEXAO)
        r = client.post("/memoria/refletir", json=self.MSGS)
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert servicos.cliente_llm.chamadas == 1  # antes o endpoint chamava o LLM duas vezes

    def test_payload_invalido(self, client: TestClient) -> None:
        assert client.post("/memoria/refletir", json={}).status_code == 422

    def test_sem_llm_devolve_503(self, client_sem_llm: TestClient) -> None:
        assert client_sem_llm.post("/memoria/refletir", json=self.MSGS).status_code == 503


class TestStatsELimpar:
    def test_stats_antes_e_depois_do_retain(self, client: TestClient) -> None:
        antes = client.get("/memoria/stats").json()
        assert antes["total_trechos"] == 0 and antes["total_vetores"] == 0
        client.post("/memoria/retain/vault")
        depois = client.get("/memoria/stats").json()
        assert depois["total_notas"] == 2
        assert depois["total_trechos"] >= 2 and depois["total_vetores"] >= 2

    def test_limpar_apaga_e_o_retain_reconstroi(self, client: TestClient) -> None:
        client.post("/memoria/retain/vault")
        assert client.post("/memoria/limpar").json()["sucesso"] is True
        s = client.get("/memoria/stats").json()
        assert s["total_trechos"] == 0 and s["total_vetores"] == 0
        # Depois de limpar, um novo retain tem de voltar a construir tudo.
        client.post("/memoria/retain/vault")
        s = client.get("/memoria/stats").json()
        assert s["total_trechos"] >= 2 and s["total_vetores"] >= 2
        assert client.get("/memoria/recall", params={"consulta": "café"}).json()["resultados"]
