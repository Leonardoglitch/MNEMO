"""Testes das ferramentas do agente (FerramentasVault). Usa vault temporário real."""

import json
import pytest

from mnemo import FerramentasVault, Permissoes, PermissaoNegada


NOTAS_INICIAIS = {
    "projetos/orcamento.md": (
        "---\ntitle: Orçamento do projeto\ntags: [projeto]\ncreated: 2026-09-01\n---\n"
        "O orçamento da app é de mil euros.\n"
    ),
    "projetos/tarefas.md": "# Lista de tarefas\nFazer o protótipo da aplicação.\n",
    "estudo/sqlite.md": "Apontamentos sem título sobre SQLite e pesquisa FTS5.\n",
    "pessoal/orcamento-pessoal.md": "Meu orçamento pessoal secreto para presentes.\n",
}


def escrever_config(raiz, pastas, todas=False):
    (raiz / ".vault").mkdir(exist_ok=True)
    (raiz / ".vault" / "config.json").write_text(
        json.dumps(
            {"versao": 1, "todas_as_pastas": todas, "pastas_permitidas": pastas}
        ),
        encoding="utf-8",
    )


@pytest.fixture
def vault(tmp_path):
    for caminho, texto in NOTAS_INICIAIS.items():
        f = tmp_path / "notas" / caminho
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(texto, encoding="utf-8")
    escrever_config(tmp_path, ["projetos", "estudo"])
    with FerramentasVault(tmp_path) as v:
        yield v


def test_search_so_pastas_permitidas(vault):
    res = vault.executar("search", {"consulta": "orçamento"})
    assert res["ok"] is True
    caminhos = [r["caminho"] for r in res["resultados"]]
    assert "projetos/orcamento.md" in caminhos
    # pessoal não deve aparecer
    assert "pessoal/orcamento-pessoal.md" not in caminhos


def test_search_ignora_acentos_e_prefixos(vault):
    res = vault.executar("search", {"consulta": "orcamento"})
    assert res["ok"] is True
    caminhos = [r["caminho"] for r in res["resultados"]]
    assert "projetos/orcamento.md" in caminhos

    res2 = vault.executar("search", {"consulta": "orça"})
    assert res2["ok"] is True
    assert "projetos/orcamento.md" in [r["caminho"] for r in res2["resultados"]]


def test_read_note_permitida(vault):
    res = vault.executar("read_note", {"caminho": "projetos/tarefas.md"})
    assert res["ok"] is True
    assert "protótipo" in res["conteudo"]


def test_read_note_bloqueada_fora_das_permissoes(vault):
    res = vault.executar("read_note", {"caminho": "pessoal/orcamento-pessoal.md"})
    assert res["ok"] is False
    assert "Sem permissão" in res["erro"]


def test_create_note_reindexa(vault):
    res = vault.executar(
        "create_note",
        {"caminho": "projetos/nova.md", "conteudo": "# Nova\nTexto sobre girafas.\n"},
    )
    assert res["ok"] is True
    # search deve encontrar a nova nota
    res2 = vault.executar("search", {"consulta": "girafas"})
    assert res2["ok"] is True
    assert any(r["caminho"] == "projetos/nova.md" for r in res2["resultados"])


def test_append_to_note_atualiza_indice(vault):
    # cria nota
    vault.executar("create_note", {"caminho": "projetos/nota.md", "conteudo": "Início.\n"})
    # acrescenta
    res = vault.executar("append_to_note", {"caminho": "projetos/nota.md", "texto": " Mais zebras."})
    assert res["ok"] is True
    # search deve achar zebras
    res2 = vault.executar("search", {"consulta": "zebras"})
    assert res2["ok"] is True
    assert any(r["caminho"] == "projetos/nota.md" for r in res2["resultados"])


def test_ferramenta_desconhecida(vault):
    res = vault.executar("nao_existe", {})
    assert res["ok"] is False
    assert "desconhecida" in res["erro"]


def test_create_note_caminho_invalido_sem_md(vault):
    res = vault.executar("create_note", {"caminho": "projetos/sem_extensao", "conteudo": "x"})
    assert res["ok"] is False
    assert "Só são permitidas notas .md" in res["erro"]


def test_append_to_note_inexistente(vault):
    res = vault.executar("append_to_note", {"caminho": "projetos/nao_existe.md", "texto": "x"})
    assert res["ok"] is False
    assert "não existe" in res["erro"] or "Nota não existe" in res["erro"]