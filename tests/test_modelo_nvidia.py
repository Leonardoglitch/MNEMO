"""Testes do ClienteNVIDIA. A rede é sempre simulada — nunca contacta a
NVIDIA de verdade, por isso corre em qualquer máquina, sem chave de API real.
"""

import io
import json

import pytest

import mnemo.modelo_nvidia as mod
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA, esquema_openai


class RespostaFalsa:
    """Imita o objeto devolvido por urlopen (suficiente para .read() e 'with')."""

    def __init__(self, dados):
        self._bytes = json.dumps(dados).encode("utf-8")

    def read(self):
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_esquema_openai_converte_formato():
    descricao = {
        "name": "search",
        "description": "pesquisa notas",
        "input_schema": {"type": "object", "properties": {"consulta": {"type": "string"}}},
    }
    assert esquema_openai(descricao) == {
        "type": "function",
        "function": {
            "name": "search",
            "description": "pesquisa notas",
            "parameters": descricao["input_schema"],
        },
    }


def test_falta_chave_api(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ErroModeloNVIDIA, match="NVIDIA_API_KEY"):
        ClienteNVIDIA()


def test_conversar_envia_pedido_correto_e_devolve_mensagem(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    dados_resposta = {"choices": [{"message": {"role": "assistant", "content": "olá"}}]}
    capturado = {}

    def urlopen_falso(pedido, timeout=60):
        capturado["pedido"] = pedido
        return RespostaFalsa(dados_resposta)

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_falso)

    cliente = ClienteNVIDIA()
    resposta = cliente.conversar([{"role": "user", "content": "oi"}])

    assert resposta["content"] == "olá"
    pedido = capturado["pedido"]
    assert pedido.headers["Authorization"] == "Bearer chave-de-teste"
    corpo = json.loads(pedido.data.decode("utf-8"))
    assert corpo["model"] == mod.MODELO_PADRAO
    assert corpo["messages"] == [{"role": "user", "content": "oi"}]
    assert "tools" not in corpo  # sem ferramentas, não deve enviar "tools"


def test_conversar_inclui_ferramentas_no_formato_openai(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    dados_resposta = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    capturado = {}

    def urlopen_falso(pedido, timeout=60):
        capturado["corpo"] = json.loads(pedido.data.decode("utf-8"))
        return RespostaFalsa(dados_resposta)

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_falso)

    ferramentas = [
        {"name": "search", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    ]
    ClienteNVIDIA().conversar([{"role": "user", "content": "oi"}], ferramentas=ferramentas)

    corpo = capturado["corpo"]
    assert corpo["tool_choice"] == "auto"
    assert corpo["tools"][0]["function"]["name"] == "search"


def test_erro_http_e_transformado_em_erro_proprio(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")

    def urlopen_falha(pedido, timeout=60):
        raise mod.urllib.error.HTTPError(
            mod.URL_BASE, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}')
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_falha)
    with pytest.raises(ErroModeloNVIDIA, match="401"):
        ClienteNVIDIA().conversar([{"role": "user", "content": "oi"}])


def test_erro_de_rede_e_transformado_em_erro_proprio(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")

    def urlopen_falha(pedido, timeout=60):
        raise mod.urllib.error.URLError("sem ligação")

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_falha)
    with pytest.raises(ErroModeloNVIDIA):
        ClienteNVIDIA().conversar([{"role": "user", "content": "oi"}])


def test_resposta_sem_choices_da_erro_claro(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    monkeypatch.setattr(
        mod.urllib.request, "urlopen", lambda pedido, timeout=60: RespostaFalsa({"erro": "quota excedida"})
    )
    with pytest.raises(ErroModeloNVIDIA, match="Resposta inesperada"):
        ClienteNVIDIA().conversar([{"role": "user", "content": "oi"}])


# --- Novos testes para retry, fallback, streaming, health_check ---

class RespostaFalsaStream:
    """Simula resposta de streaming (Server-Sent Events)."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.index]
        self.index += 1
        # Formato SSE: "data: {json}\n\n"
        return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_retry_em_erro_5xx(monkeypatch):
    """Deve tentar 3 vezes antes de falhar em erro 500."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    chamadas = []

    def urlopen_falha_500(pedido, timeout=60):
        chamadas.append(1)
        raise mod.urllib.error.HTTPError(
            mod.URL_BASE, 500, "Internal Server Error", {}, io.BytesIO(b'{"error":"server error"}')
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_falha_500)

    # Apenas um modelo para testar retry sem fallback
    cliente = ClienteNVIDIA(max_retries=3, base_delay=0.01, fallback_models=[])
    with pytest.raises(ErroModeloNVIDIA):
        cliente.conversar([{"role": "user", "content": "oi"}])

    assert len(chamadas) == 3  # 3 tentativas


def test_retry_em_erro_429(monkeypatch):
    """Deve tentar novamente em erro 429 (rate limit)."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    chamadas = []

    def urlopen_429_duas_vezes(pedido, timeout=60):
        chamadas.append(1)
        if len(chamadas) < 3:
            raise mod.urllib.error.HTTPError(
                mod.URL_BASE, 429, "Rate Limited", {}, io.BytesIO(b'{"error":"rate limit"}')
            )
        return RespostaFalsa({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_429_duas_vezes)

    cliente = ClienteNVIDIA(max_retries=3, base_delay=0.01)
    resposta = cliente.conversar([{"role": "user", "content": "oi"}])
    assert resposta["content"] == "ok"
    assert len(chamadas) == 3  # falha 2x, sucesso na 3ª


def test_fallback_para_proximo_modelo(monkeypatch):
    """Deve tentar modelo fallback quando o principal falha consistentemente."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    modelos_tentados = []

    def urlopen_mock(pedido, timeout=60):
        corpo = json.loads(pedido.data.decode("utf-8"))
        modelos_tentados.append(corpo["model"])
        if corpo["model"] == "modelo-principal":
            raise mod.urllib.error.HTTPError(
                mod.URL_BASE, 500, "Internal Server Error", {}, io.BytesIO(b'{"error":"server error"}')
            )
        return RespostaFalsa({"choices": [{"message": {"role": "assistant", "content": "ok fallback"}}]})

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_mock)

    cliente = ClienteNVIDIA(
        modelo="modelo-principal",
        fallback_models=["modelo-fallback"],
        max_retries=1,
        base_delay=0.01
    )
    resposta = cliente.conversar([{"role": "user", "content": "oi"}])

    assert resposta["content"] == "ok fallback"
    assert "modelo-principal" in modelos_tentados
    assert "modelo-fallback" in modelos_tentados


def test_streaming_retorna_generator(monkeypatch):
    """Deve retornar generator quando stream=True."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")

    chunks = [
        {"choices": [{"delta": {"content": "Olá"}}]},
        {"choices": [{"delta": {"content": " mundo"}}]},
        {"choices": [{"delta": {}}]},
    ]

    def urlopen_stream(pedido, timeout=60):
        corpo = json.loads(pedido.data.decode("utf-8"))
        assert corpo.get("stream") is True
        return RespostaFalsaStream(chunks)

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_stream)

    cliente = ClienteNVIDIA()
    generator = cliente.conversar([{"role": "user", "content": "oi"}], stream=True)

    # Verifica que é um generator
    assert hasattr(generator, "__iter__")
    assert hasattr(generator, "__next__")

    # Consome o generator
    chunks_recebidos = list(generator)
    assert len(chunks_recebidos) == 2  # duas deltas com conteúdo
    assert chunks_recebidos[0]["delta"] == {"content": "Olá"}
    assert chunks_recebidos[1]["delta"] == {"content": " mundo"}


def test_health_check_sucesso(monkeypatch):
    """Health check deve retornar True se API responder."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")

    def urlopen_ok(pedido, timeout=60):
        return RespostaFalsa({"choices": [{"message": {"role": "assistant", "content": "pong"}}]})

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_ok)

    cliente = ClienteNVIDIA()
    assert cliente.health_check() is True


def test_health_check_falha(monkeypatch):
    """Health check deve retornar False se API falhar."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")

    def urlopen_falha(pedido, timeout=60):
        raise mod.urllib.error.URLError("conexão recusada")

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_falha)

    cliente = ClienteNVIDIA()
    assert cliente.health_check() is False


def test_timeout_configuravel(monkeypatch):
    """Timeout deve ser passado para urlopen."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    timeouts_capturados = []

    def urlopen_captura_timeout(pedido, timeout=60):
        timeouts_capturados.append(timeout)
        return RespostaFalsa({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_captura_timeout)

    cliente = ClienteNVIDIA(timeout=30)
    cliente.conversar([{"role": "user", "content": "oi"}])

    assert timeouts_capturados[0] == 30


def test_nao_retry_em_erro_4xx(monkeypatch):
    """Não deve tentar novamente em erros 4xx (exceto 429)."""
    monkeypatch.setenv("NVIDIA_API_KEY", "chave-de-teste")
    chamadas = []

    def urlopen_400(pedido, timeout=60):
        chamadas.append(1)
        raise mod.urllib.error.HTTPError(
            mod.URL_BASE, 400, "Bad Request", {}, io.BytesIO(b'{"error":"bad request"}')
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_400)

    cliente = ClienteNVIDIA(max_retries=3, base_delay=0.01)
    with pytest.raises(ErroModeloNVIDIA):
        cliente.conversar([{"role": "user", "content": "oi"}])

    assert len(chamadas) == 1  # apenas uma tentativa, sem retry
