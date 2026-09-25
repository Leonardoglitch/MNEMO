"""Testes do laço de tool-calling em chat.py. Usa um cliente falso — nunca
contacta a NVIDIA — mas usa um FerramentasVault real, para confirmar que a
ligação entre o modelo e o vault funciona de ponta a ponta.
"""

import json

import pytest

import chat
from mnemo import FerramentasVault


class ClienteFalso:
    """Devolve, em sequência, as respostas passadas ao construtor — como se
    fossem vindas da API — sem qualquer chamada de rede."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = 0

    def conversar(self, mensagens, ferramentas=None, **_):
        self.chamadas += 1
        return self.respostas.pop(0)


@pytest.fixture
def vault(tmp_path):
    nota = tmp_path / "notas" / "projetos" / "orcamento.md"
    nota.parent.mkdir(parents=True)
    nota.write_text("O orçamento é de mil euros.\n", encoding="utf-8")
    (tmp_path / ".vault").mkdir()
    (tmp_path / ".vault" / "config.json").write_text(
        json.dumps({"versao": 1, "todas_as_pastas": False, "pastas_permitidas": ["projetos"]}),
        encoding="utf-8",
    )
    with FerramentasVault(tmp_path) as v:
        yield v


def chamada_ferramenta(id_, nome, argumentos):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": id_, "type": "function", "function": {"name": nome, "arguments": json.dumps(argumentos)}}
        ],
    }


def test_resposta_direta_sem_ferramentas(vault):
    cliente = ClienteFalso([{"role": "assistant", "content": "Olá!"}])
    mensagens = [{"role": "user", "content": "oi"}]

    resposta = chat.executar_ciclo_ferramentas(cliente, vault, mensagens)

    assert resposta == "Olá!"
    assert cliente.chamadas == 1
    assert mensagens[-1]["content"] == "Olá!"


def test_uma_chamada_de_ferramenta_ate_resposta_final(vault):
    cliente = ClienteFalso(
        [
            chamada_ferramenta("call_1", "search", {"consulta": "orçamento"}),
            {"role": "assistant", "content": "Encontrei a nota do orçamento."},
        ]
    )
    mensagens = [{"role": "user", "content": "procura orçamento"}]

    resposta = chat.executar_ciclo_ferramentas(cliente, vault, mensagens)

    assert resposta == "Encontrei a nota do orçamento."
    assert cliente.chamadas == 2
    msg_ferramenta = mensagens[-2]
    assert msg_ferramenta["role"] == "tool" and msg_ferramenta["tool_call_id"] == "call_1"
    resultado = json.loads(msg_ferramenta["content"])
    assert resultado["ok"] is True
    assert resultado["resultados"][0]["caminho"] == "projetos/orcamento.md"


def test_ferramenta_bloqueada_devolve_erro_ao_modelo_sem_rebentar(vault):
    cliente = ClienteFalso(
        [
            chamada_ferramenta("call_1", "read_note", {"caminho": "pessoal/segredo.md"}),
            {"role": "assistant", "content": "Não tenho acesso a essa nota."},
        ]
    )
    mensagens = [{"role": "user", "content": "lê a nota privada"}]

    resposta = chat.executar_ciclo_ferramentas(cliente, vault, mensagens)

    assert resposta == "Não tenho acesso a essa nota."
    resultado = json.loads(mensagens[-2]["content"])
    assert resultado["ok"] is False


def test_argumentos_invalidos_nao_rebentam_o_ciclo(vault):
    pedido_invalido = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "search", "arguments": "não é json"}}],
    }
    cliente = ClienteFalso([pedido_invalido, {"role": "assistant", "content": "ok"}])
    mensagens = [{"role": "user", "content": "..."}]

    resposta = chat.executar_ciclo_ferramentas(cliente, vault, mensagens)

    assert resposta == "ok"
    resultado = json.loads(mensagens[-2]["content"])
    assert resultado["ok"] is False and "JSON" in resultado["erro"]


def test_limite_de_ciclos_evita_loop_infinito(vault):
    pedido_repetido = chamada_ferramenta("x", "search", {"consulta": "a"})
    cliente = ClienteFalso([pedido_repetido] * chat.MAX_CICLOS_FERRAMENTAS)
    mensagens = [{"role": "user", "content": "..."}]

    resposta = chat.executar_ciclo_ferramentas(cliente, vault, mensagens)

    assert cliente.chamadas == chat.MAX_CICLOS_FERRAMENTAS
    assert "ciclo" in resposta


def test_varias_chamadas_de_ferramentas_na_mesma_resposta(vault):
    pedido_duplo = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "search", "arguments": json.dumps({"consulta": "orçamento"})}},
            {"id": "b", "type": "function", "function": {"name": "create_note", "arguments": json.dumps({"caminho": "projetos/nova.md", "conteudo": "x"})}},
        ],
    }
    cliente = ClienteFalso([pedido_duplo, {"role": "assistant", "content": "feito"}])
    mensagens = [{"role": "user", "content": "..."}]

    chat.executar_ciclo_ferramentas(cliente, vault, mensagens)

    ids_respondidos = [m["tool_call_id"] for m in mensagens if m.get("role") == "tool"]
    assert ids_respondidos == ["a", "b"]
