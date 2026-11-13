#!/usr/bin/env python3
"""Testes de integração para a API FastAPI do subsistema Memória."""

import sys
import os
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configura variáveis de ambiente ANTES de importar a API
os.environ.setdefault("NVIDIA_API_KEY", "test-fake-key")
os.environ.setdefault("MNEMO_VAULT", "vault-teste-api")

# Remove vault de teste anterior se existir
test_vault_path = Path("vault-teste-api")
if test_vault_path.exists():
    shutil.rmtree(test_vault_path)
test_vault_path.mkdir()
(test_vault_path / "notas").mkdir()

# Carrega .env se existir
def carregar_env(caminho: str = ".env") -> None:
    from pathlib import Path
    if not Path(caminho).is_absolute():
        caminho = str(Path(__file__).parent.parent / caminho)
    try:
        with open(caminho, encoding="utf-8-sig") as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                chave, valor = linha.split("=", 1)
                os.environ[chave.strip()] = valor.strip().strip("'\"")
    except FileNotFoundError:
        pass

import os
import shutil
from pathlib import Path

carregar_env()

from api_memoria import app, state
from mnemo import Permissoes
from mnemo.memoria import (
    Recall, Retain, Reflect, FakeEmbeddings, IndiceVetorial
)
from mnemo.indexador import Indexador
from mnemo.armazenamento import Armazenamento
from mnemo import Permissoes
from mnemo.memoria import FakeEmbeddings, IndiceVetorial

import pytest
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def setup_test_env():
    """Configura ambiente de teste antes de cada teste."""
    os.environ["NVIDIA_API_KEY"] = "test-fake-key"
    os.environ["MNEMO_VAULT"] = "vault-teste-api"
    yield


@pytest.fixture
def client():
    """Cliente de teste FastAPI com estado mockado usando FakeEmbeddings."""
    from api_memoria import app, state
    from mnemo import Permissoes
    from mnemo.memoria import (
        Recall, Retain, Reflect, FakeEmbeddings, IndiceVetorial
    )
    from mnemo.indexador import Indexador
    from mnemo.armazenamento import Armazenamento
    from mnemo import Permissoes
    from mnemo.memoria import FakeEmbeddings, IndiceVetorial
    import os

    # Configura estado global para testes
    from api_memoria import state

    # Configura vault de teste
    os.environ.setdefault("NVIDIA_API_KEY", "test-fake-key")
    os.environ.setdefault("MNEMO_VAULT", "vault-teste-api")

    # Remove banco de dados de teste anterior se existir
    test_db_path = Path("vault-teste-api/.vault/index.db")
    if test_db_path.exists():
        test_db_path.unlink(missing_ok=True)

    state.perms = Permissoes("vault-teste-api")
    state.indexador = Indexador(state.perms, caminho_db=":memory:")
    state.indexador.reindexar_tudo()
    state.armazenamento = Armazenamento(state.perms)

    # Usa FakeEmbeddings para testes
    state.provider = FakeEmbeddings(dim=512)
    state.indice = IndiceVetorial(512)

    state.retain = Retain(
        perms=Permissoes("vault-teste-api"),
        armazenamento=Armazenamento(Permissoes("vault-teste-api")),
        indexador=Indexador(Permissoes("vault-teste-api"), caminho_db=":memory:"),
        provider=FakeEmbeddings(dim=512),
        modelo_emb="fake",
        dim_emb=512,
    )

    from mnemo.memoria import Recall, Reflect, IndiceVetorial
    state.recall = Recall(
        Indexador(Permissoes("vault-teste-api"), caminho_db=":memory:"),
        Permissoes("vault-teste-api"),
        FakeEmbeddings(dim=512),
        IndiceVetorial(512),
    )
        state.reflect = Reflect()
        state.provider = FakeEmbeddings(dim=512)
        state.cliente_llm = None
    state.reflect = Reflect()
    state.provider = FakeEmbeddings(dim=512)
    state.cliente_llm = None

    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    yield client
    
    # Cleanup: fecha conexões do banco de dados
    if state.indexador:
        state.indexador.fechar()


class TestHealthEndpoint:
    """Testes para o endpoint /health."""

    def test_health_ok(self, client):
        """Health check deve retornar status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "memoria_inicializada" in data


class TestRecallEndpoint:
    """Testes para o endpoint /memoria/recall."""

    def test_recall_basico(self, client):
        """Recall básico deve funcionar."""
        response = client.get("/memoria/recall", params={"consulta": "teste"})
        assert response.status_code == 200
        data = response.json()
        assert "resultados" in data
        assert "modo" in data
        assert data["modo"] in ("hibrido", "lexical", "semantica")
        assert isinstance(data["resultados"], list)

    def test_recall_com_parametros(self, client):
        """Recall com parâmetros opcionais."""
        response = client.get(
            "/memoria/recall",
            params={
                "consulta": "RAG",
                "k": 5,
                "budget_tokens": 1000,
                "explicar": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "resultados" in data
        assert "modo" in data
        if data["resultados"]:
            r = data["resultados"][0]
            assert "nota_id" in r
            assert "nota_caminho" in r
            assert "score" in r
            assert "origem" in r
            assert "citacao" in r
            assert "modo" in r

    def test_recall_sem_consulta(self, client):
        """Recall sem consulta deve retornar 422."""
        response = client.get("/memoria/recall")
        assert response.status_code == 422

    def test_recall_k_invalido(self, client):
        """Recall com k inválido deve retornar 422."""
        response = client.get("/memoria/recall", params={"consulta": "teste", "k": 0})
        assert response.status_code == 422
        response = client.get("/memoria/recall", params={"consulta": "teste", "k": 100})
        assert response.status_code == 422


class TestRetainVaultEndpoint:
    """Testes para o endpoint /memoria/retain/vault."""

    def test_retain_vault_basico(self, client):
        """Retain vault básico deve funcionar."""
        response = client.post("/memoria/retain/vault")
        assert response.status_code == 200
        data = response.json()
        assert "novas" in data
        assert "atualizadas" in data
        assert "inalteradas" in data
        assert "removidas" in data
        assert "trechos_novos" in data
        assert "trechos_atualizados" in data
        assert "embeddings_novos" in data
        assert "embeddings_pendentes" in data
        assert "erros" in data
        assert all(isinstance(v, int) for k, v in data.items() if k not in ("embeddings_pendentes", "erros"))
        assert isinstance(data["embeddings_pendentes"], list)
        assert isinstance(data["erros"], list)

    def test_retain_vault_reparar(self, client):
        """Retain vault com reparar=True."""
        response = client.post("/memoria/retain/vault", params={"reparar": True})
        assert response.status_code == 200
        data = response.json()
        assert "novas" in data


class TestRetainConversaEndpoint:
    """Testes para o endpoint /memoria/retain/conversa."""

    def test_retain_conversa_sem_mensagens(self, client):
        """Retain conversa com lista vazia."""
        response = client.post("/memoria/retain/conversa", json={"mensagens": []})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_retain_conversa_com_mensagens(self, client):
        """Retain conversa com mensagens."""
        response = client.post(
            "/memoria/retain/conversa",
            json={
                "mensagens": [
                    {"role": "user", "content": "Olá, quero guardar que prefiro café."},
                    {"role": "assistant", "content": "Entendido, prefere café."},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_retain_conversa_invalido(self, client):
        """Retain conversa com payload inválido."""
        response = client.post("/memoria/retain/conversa", json={})
        assert response.status_code == 422


class TestRefletirEndpoint:
    """Testes para o endpoint /memoria/refletir."""

    def test_refletir_sem_mensagens(self, client):
        """Refletir com lista vazia."""
        response = client.post("/memoria/refletir", json={"mensagens": []})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_refletir_com_mensagens(self, client):
        """Refletir com mensagens."""
        response = client.post(
            "/memoria/refletir",
            json={
                "mensagens": [
                    {"role": "user", "content": "Acho que o projeto X vai falhar."},
                    {"role": "assistant", "content": "Por que acha isso?"},
                    {"role": "user", "content": "Porque não tem recursos suficientes."},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_refletir_invalido(self, client):
        """Refletir com payload inválido."""
        response = client.post("/memoria/refletir", json={})
        assert response.status_code == 422


class TestStatsEndpoint:
    """Testes para o endpoint /memoria/stats."""

    def test_stats(self, client):
        """Stats deve retornar contadores."""
        response = client.get("/memoria/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_notas" in data
        assert "total_trechos" in data
        assert "total_vetores" in data
        assert all(isinstance(v, int) for v in data.values())
        assert all(v >= 0 for v in data.values())


class TestLimparEndpoint:
    """Testes para o endpoint /memoria/limpar."""

    def test_limpar(self, client):
        """Limpar memória."""
        response = client.post("/memoria/limpar")
        assert response.status_code == 200
        data = response.json()
        assert "sucesso" in data
        assert "mensagem" in data
        assert data["sucesso"] is True


class TestValidationErrors:
    """Testes de erros de validação."""

    def test_recall_k_negativo(self, client):
        response = client.get("/memoria/recall", params={"consulta": "teste", "k": -1})
        assert response.status_code == 422

    def test_recall_budget_negativo(self, client):
        response = client.get("/memoria/recall", params={"consulta": "teste", "budget_tokens": -1})
        assert response.status_code == 422

    def test_recall_budget_alto(self, client):
        response = client.get("/memoria/recall", params={"consulta": "teste", "budget_tokens": 20000})
        assert response.status_code == 422


class TestResponseStructure:
    """Verifica estrutura das respostas."""

    def test_recall_resultado_estrutura(self, client):
        """Resultado de recall tem todos os campos esperados."""
        response = client.get("/memoria/recall", params={"consulta": "teste"})
        assert response.status_code == 200
        data = response.json()
        if data["resultados"]:
            r = data["resultados"][0]
            campos_obrigatorios = [
                "nota_id", "nota_caminho", "nota_titulo",
                "trecho_id", "trecho_ordem", "titulo_secao",
                "texto", "score", "origem", "citacao", "modo"
            ]
            for campo in campos_obrigatorios:
                assert campo in data["resultados"][0], f"Campo faltando: {campo}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])