"""Cliente para a API da NVIDIA (compatível com OpenAI), usado para falar
com o Nemotron via tool-calling.

Não depende de bibliotecas externas: usa só urllib da biblioteca padrão, para
manter o projeto sem dependências extra nesta fase.

Documentação da NVIDIA: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b
"""

import json
import os
import urllib.error
import urllib.request

URL_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELO_PADRAO = "nvidia/nemotron-3-super-120b-a12b"


class ErroModeloNVIDIA(Exception):
    """A chamada à API da NVIDIA falhou: rede, autenticação ou resposta inesperada."""


def esquema_openai(descricao_ferramenta):
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
    def __init__(self, chave_api=None, modelo=None, url_base=URL_BASE, ativar_pensamento=False):
        self.chave_api = chave_api or os.environ.get("NVIDIA_API_KEY")
        if not self.chave_api:
            raise ErroModeloNVIDIA(
                "Falta a variável de ambiente NVIDIA_API_KEY com a chave da API da NVIDIA."
            )
        self.modelo = modelo or os.environ.get("NVIDIA_MODEL") or MODELO_PADRAO
        self.url_base = url_base
        self.ativar_pensamento = ativar_pensamento

    def conversar(self, mensagens, ferramentas=None, max_tokens=2048, temperatura=0.7):
        """Envia o histórico de mensagens (formato OpenAI) ao modelo e devolve
        a mensagem de resposta (um dicionário com "role", "content" e,
        possivelmente, "tool_calls")."""
        corpo = {
            "model": self.modelo,
            "messages": mensagens,
            "max_tokens": max_tokens,
            "temperature": temperatura,
            "chat_template_kwargs": {"enable_thinking": self.ativar_pensamento},
        }
        if ferramentas:
            corpo["tools"] = [esquema_openai(f) for f in ferramentas]
            corpo["tool_choice"] = "auto"

        pedido = urllib.request.Request(
            self.url_base,
            data=json.dumps(corpo).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.chave_api}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(pedido, timeout=60) as resposta:
                dados = json.loads(resposta.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            corpo_erro = e.read().decode("utf-8", errors="replace")
            raise ErroModeloNVIDIA(f"A API respondeu com erro {e.code}: {corpo_erro}") from e
        except urllib.error.URLError as e:
            raise ErroModeloNVIDIA(
                f"Não foi possível contactar a API da NVIDIA: {e.reason}"
            ) from e

        try:
            return dados["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise ErroModeloNVIDIA(f"Resposta inesperada da API: {dados}") from e
