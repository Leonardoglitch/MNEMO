"""Cliente para a API da NVIDIA (compatível com OpenAI), usado para falar
com o Nemotron via tool-calling.

Suporta retry com backoff exponencial, fallback de modelos, timeout configurável
e streaming de tokens (se a API suportar).

Documentação da NVIDIA: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Union

URL_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELO_PADRAO = "nvidia/nemotron-3-super-120b-a12b"


class ErroModeloNVIDIA(Exception):
    """A chamada à API da NVIDIA falhou: rede, autenticação ou resposta inesperada."""


def esquema_openai(descricao_ferramenta: Dict[str, Any]) -> Dict[str, Any]:
    """Converte {name, description, input_schema} (formato usado no resto do
    Mnemo) para o formato "tools" da API OpenAI/NVIDIA:
    {"type": "function", "function": {name, description, parameters}}.
    """
    return {
        "type": "function",
        "function": {
            "name": descricao_ferramenta["name"],
            "description": descricao_ferramenta["description"],
            "parameters": descricao_ferramenta["input_schema"],
        },
    }


class ClienteNVIDIA:
    def __init__(
        self,
        chave_api: Optional[str] = None,
        modelo: Optional[str] = None,
        url_base: str = URL_BASE,
        ativar_pensamento: bool = False,
        timeout: int = 60,
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        self.chave_api = chave_api or os.environ.get("NVIDIA_API_KEY")
        if not self.chave_api:
            raise ErroModeloNVIDIA(
                "Falta a variável de ambiente NVIDIA_API_KEY com a chave da API da NVIDIA."
            )
        self.modelo = modelo or os.environ.get("NVIDIA_MODEL") or MODELO_PADRAO
        self.url_base = url_base
        self.ativar_pensamento = ativar_pensamento
        self.timeout = timeout
        self.fallback_models = fallback_models if fallback_models is not None else [
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-ultra",
            "nvidia/nemotron-4-340b-instruct",
        ]
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _fazer_pedido_bruto(
        self, corpo: Dict[str, Any], stream: bool = False
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Faz o pedido HTTP à API. Se stream=True, retorna generator de chunks.
        Levanta exceções originais para erros HTTP/URL (não embrulha)."""
        pedido = urllib.request.Request(
            self.url_base,
            data=json.dumps(corpo).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.chave_api}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(pedido, timeout=self.timeout) as resposta:
            if stream:
                return self._iter_stream(resposta)
            dados = json.loads(resposta.read().decode("utf-8"))
            return dados

    def _iter_stream(self, resposta) -> Generator[Dict[str, Any], None, None]:
        """Itera sobre resposta de streaming (Server-Sent Events)."""
        for linha in resposta:
            linha = linha.decode("utf-8").strip()
            if not linha or linha == "data: [DONE]":
                continue
            if linha.startswith("data: "):
                try:
                    yield json.loads(linha[6:])
                except json.JSONDecodeError:
                    continue

    def _extrair_mensagem(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return dados["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise ErroModeloNVIDIA(f"Resposta inesperada da API: {dados}") from e

    def _deve_tentar_novamente(self, erro: Exception) -> bool:
        """Decide se vale a pena tentar novamente baseado no tipo de erro."""
        if isinstance(erro, urllib.error.HTTPError):
            # Tentar novamente em erros 5xx (servidor) e 429 (rate limit)
            return erro.code >= 500 or erro.code == 429
        if isinstance(erro, (urllib.error.URLError, TimeoutError)):
            return True
        return False

    def _delay_com_backoff(self, attempt: int) -> None:
        delay = self.base_delay * (2 ** attempt)
        time.sleep(delay)

    def conversar(
        self,
        mensagens: List[Dict[str, Any]],
        ferramentas: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperatura: float = 0.7,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Envia o histórico de mensagens ao modelo.
        Se stream=True, retorna generator de chunks (dicts com 'delta').
        Caso contrário, retorna a mensagem completa (dict com role, content, tool_calls).
        """
        modelos_tentados = set()
        modelo_atual = self.modelo

        # Lista de modelos a tentar: primeiro o configurado, depois os fallbacks
        modelos_para_tentar = [self.modelo] + [m for m in self.fallback_models if m != self.modelo]

        for modelo in modelos_para_tentar:
            if modelo in modelos_tentados:
                continue
            modelos_tentados.add(modelo)

            for attempt in range(self.max_retries):
                try:
                    corpo = {
                        "model": modelo,
                        "messages": mensagens,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    if ferramentas and not stream:
                        corpo["tools"] = [esquema_openai(f) for f in ferramentas]
                        corpo["tool_choice"] = "auto"
                    if stream:
                        corpo["stream"] = True

                    # Tenta fazer o pedido
                    resultado = self._fazer_pedido_bruto(corpo, stream=stream)

                    # Sucesso!
                    if stream:
                        return self._processar_stream(resultado)
                    else:
                        return self._extrair_mensagem(resultado)

                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                    if not self._deve_tentar_novamente(e):
                        raise ErroModeloNVIDIA(f"Erro não recuperável: {e}") from e
                    # Se não é a última tentativa, espera e tenta novamente
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    # Se esgotou tentativas para este modelo, tenta próximo modelo
                    break
                except ErroModeloNVIDIA:
                    raise
                except Exception as e:
                    raise ErroModeloNVIDIA(f"Erro inesperado: {e}") from e

        raise ErroModeloNVIDIA(f"Todos os modelos falharam após {self.max_retries} tentativas cada.")

    def _processar_stream(self, generator: Generator[Dict[str, Any], None, None]) -> Generator[Dict[str, Any], None, None]:
        """Processa stream de chunks e yield mensagens incrementais."""
        for chunk in generator:
            try:
                delta = chunk["choices"][0].get("delta", {})
                if delta:
                    yield {"delta": delta}
            except (KeyError, IndexError):
                continue

    def health_check(self) -> bool:
        """Verifica se a API está acessível com um pedido mínimo."""
        try:
            self._fazer_pedido_bruto({
                "model": self.modelo,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            })
            return True
        except Exception:
            return False