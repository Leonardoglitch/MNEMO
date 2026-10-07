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
        resp = self.respostas.pop(0)
        # Retorna tupla (mensagem, usage) para compatibilidade com nova API
        if isinstance(resp, tuple):
            return resp
        return resp, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


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

    resposta, usage = chat.executar_ciclo_ferramentas(cliente, vault, mensagens, shutdown_check=lambda: False)

    assert resposta == "Olá!"
    assert cliente.chamadas == 1
    assert mensagens[-1]["content"] == "Olá!"
    assert usage["total_tokens"] > 0


def test_uma_chamada_de_ferramenta_ate_resposta_final(vault):
    cliente = ClienteFalso(
        [
            chamada_ferramenta("call_1", "search", {"consulta": "orçamento"}),
            {"role": "assistant", "content": "Encontrei a nota do orçamento."},
        ]
    )
    mensagens = [{"role": "user", "content": "procura orçamento"}]

    resposta, usage = chat.executar_ciclo_ferramentas(cliente, vault, mensagens, shutdown_check=lambda: False)

    assert resposta == "Encontrei a nota do orçamento."
    assert cliente.chamadas == 2
    assert usage["total_tokens"] > 0
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

    resposta, usage = chat.executar_ciclo_ferramentas(cliente, vault, mensagens, shutdown_check=lambda: False)

    assert resposta == "Não tenho acesso a essa nota."
    assert usage["total_tokens"] > 0
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

    resposta, usage = chat.executar_ciclo_ferramentas(cliente, vault, mensagens, shutdown_check=lambda: False)

    assert resposta == "ok"
    assert usage["total_tokens"] > 0
    resultado = json.loads(mensagens[-2]["content"])
    assert resultado["ok"] is False and "JSON" in resultado["erro"]


def test_limite_de_ciclos_evita_loop_infinito(vault):
    pedido_repetido = chamada_ferramenta("x", "search", {"consulta": "a"})
    cliente = ClienteFalso([pedido_repetido] * chat.MAX_CICLOS_FERRAMENTAS)
    mensagens = [{"role": "user", "content": "..."}]

    resposta, usage = chat.executar_ciclo_ferramentas(cliente, vault, mensagens, shutdown_check=lambda: False)

    assert cliente.chamadas == chat.MAX_CICLOS_FERRAMENTAS
    assert "ciclo" in resposta
    assert usage["total_tokens"] > 0


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


# --- testes de persistência de histórico ---

@pytest.fixture
def vault_com_historico(tmp_path):
    """Vault temporário com pasta 'historico' permitida."""
    nota = tmp_path / "notas" / "projetos" / "orcamento.md"
    nota.parent.mkdir(parents=True)
    nota.write_text("O orçamento é de mil euros.\n", encoding="utf-8")
    (tmp_path / ".vault").mkdir()
    (tmp_path / ".vault" / "config.json").write_text(
        json.dumps({
            "versao": 1,
            "todas_as_pastas": False,
            "pastas_permitidas": ["projetos", "historico"]
        }),
        encoding="utf-8",
    )
    with FerramentasVault(tmp_path) as v:
        yield v


def test_salvar_historico_cria_nota_com_tool_calls(vault_com_historico):
    mensagens = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "procura orçamento"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": json.dumps({"consulta": "orçamento"})}}
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"ok": True, "resultados": [{"caminho": "projetos/orcamento.md", "titulo": "Orçamento"}]}),
        },
        {"role": "assistant", "content": "Encontrei."}
    ]

    chat._salvar_historico(mensagens, str(vault_com_historico.armazenamento.perms.raiz), "test-model")

    # Verifica que a nota foi criada em historico/
    historico_dir = vault_com_historico.armazenamento.perms.raiz / "notas" / "historico"
    arquivos = list(historico_dir.glob("*.md"))
    assert len(arquivos) == 1
    conteudo = arquivos[0].read_text(encoding="utf-8")
    assert "# Conversa" in conteudo
    assert "procura orçamento" in conteudo
    assert "Tool call" in conteudo
    assert "search" in conteudo
    assert "Tool result" in conteudo
    assert "call_1" in conteudo


def test_historico_folder_permissions(vault_com_historico):
    # A pasta historico deve estar nas permitidas
    perms = vault_com_historico.armazenamento.perms
    assert "historico" in [str(p.relative_to(perms.notas)) for p in perms.raizes_permitidas()]
