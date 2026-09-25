"""Testes do Indexador. Correr a partir da raiz do repositório: pytest"""

import json
import os

import pytest

from mnemo import Armazenamento, Indexador, Permissoes, PermissaoNegada

NOTAS = {
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
    for caminho, texto in NOTAS.items():
        f = tmp_path / "notas" / caminho
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(texto, encoding="utf-8")
    escrever_config(tmp_path, ["projetos", "estudo"])
    return tmp_path


@pytest.fixture
def ambiente(vault):
    perms = Permissoes(vault)
    idx = Indexador(perms)
    yield vault, perms, idx
    idx.fechar()


def caminhos(resultados):
    return [r["caminho"] for r in resultados]


def test_so_indexa_pastas_permitidas(ambiente):
    _, _, idx = ambiente
    resumo = idx.reindexar_tudo()
    assert resumo["novas"] == 3
    total = idx.con.execute("SELECT COUNT(*) FROM notas").fetchone()[0]
    assert total == 3


def test_pesquisa_nao_devolve_pasta_privada(ambiente):
    _, _, idx = ambiente
    idx.reindexar_tudo()
    assert caminhos(idx.pesquisar("orçamento")) == ["projetos/orcamento.md"]
    assert idx.pesquisar("secreto") == []


def test_pesquisa_ignora_acentos_e_aceita_prefixos(ambiente):
    _, _, idx = ambiente
    idx.reindexar_tudo()
    assert caminhos(idx.pesquisar("orcamento")) == ["projetos/orcamento.md"]
    assert caminhos(idx.pesquisar("orça")) == ["projetos/orcamento.md"]


def test_titulos(ambiente):
    _, _, idx = ambiente
    idx.reindexar_tudo()
    titulos = dict(idx.con.execute("SELECT caminho, titulo FROM notas"))
    assert titulos["projetos/orcamento.md"] == "Orçamento do projeto"  # frontmatter
    assert titulos["projetos/tarefas.md"] == "Lista de tarefas"  # cabeçalho #
    assert titulos["estudo/sqlite.md"] == "sqlite"  # nome do ficheiro


def test_consulta_com_caracteres_especiais_nao_rebenta(ambiente):
    _, _, idx = ambiente
    idx.reindexar_tudo()
    for consulta in ['orçamento-pessoal', '"aspas', "AND OR NOT", "***", "", "   "]:
        idx.pesquisar(consulta)  # não pode levantar exceção


def test_excerto_e_relevancia(ambiente):
    _, _, idx = ambiente
    idx.reindexar_tudo()
    (r,) = idx.pesquisar("protótipo")
    assert "[protótipo]" in r["excerto"]
    assert r["relevancia"] > 0


def test_excerto_nao_mostra_frontmatter(ambiente):
    _, _, idx = ambiente
    idx.reindexar_tudo()
    (r,) = idx.pesquisar("mil euros")
    assert "title:" not in r["excerto"] and "---" not in r["excerto"]
    assert "[mil] [euros]" in r["excerto"]


def test_reindexar_deteta_alteracoes_externas(ambiente):
    vault, _, idx = ambiente
    idx.reindexar_tudo()
    assert idx.reindexar_tudo()["inalteradas"] == 3

    nota = vault / "notas" / "projetos" / "tarefas.md"
    nota.write_text("# Lista de tarefas\nAgora fala de zebras.\n", encoding="utf-8")
    ns = nota.stat().st_mtime_ns + 5_000_000_000
    os.utime(nota, ns=(ns, ns))  # garante que a data mudou

    resumo = idx.reindexar_tudo()
    assert resumo["atualizadas"] == 1
    assert caminhos(idx.pesquisar("zebras")) == ["projetos/tarefas.md"]

    nota.unlink()
    assert idx.reindexar_tudo()["removidas"] == 1
    assert idx.pesquisar("zebras") == []


def test_armazenamento_mantem_o_indice_atualizado(ambiente):
    _, perms, idx = ambiente
    idx.reindexar_tudo()
    arm = Armazenamento(perms, indexador=idx)

    arm.criar("projetos/nova.md", "# Nova\nUm texto sobre girafas.\n")
    assert caminhos(idx.pesquisar("girafas")) == ["projetos/nova.md"]

    arm.editar("projetos/nova.md", "# Nova\nAgora só há elefantes.\n")
    assert idx.pesquisar("girafas") == []
    assert caminhos(idx.pesquisar("elefantes")) == ["projetos/nova.md"]

    arm.acrescentar("projetos/nova.md", "E também pinguins.")
    assert caminhos(idx.pesquisar("pinguins")) == ["projetos/nova.md"]

    arm.apagar("projetos/nova.md")
    assert idx.pesquisar("elefantes") == []


def test_pasta_deixa_de_ser_permitida(ambiente):
    vault, perms, idx = ambiente
    idx.reindexar_tudo()
    assert caminhos(idx.pesquisar("SQLite")) == ["estudo/sqlite.md"]

    escrever_config(vault, ["projetos"])  # tira "estudo"
    perms.carregar()

    # Mesmo antes de reindexar, a pesquisa já não mostra a pasta retirada...
    assert idx.pesquisar("SQLite") == []
    # ...e depois de reindexar, o conteúdo sai mesmo do índice.
    assert idx.reindexar_tudo()["removidas"] == 1
    total = idx.con.execute("SELECT COUNT(*) FROM notas_fts").fetchone()[0]
    assert total == 2


def test_todas_as_pastas_indexa_tudo(ambiente):
    vault, perms, idx = ambiente
    escrever_config(vault, [], todas=True)
    perms.carregar()
    idx.reindexar_tudo()
    assert "pessoal/orcamento-pessoal.md" in caminhos(idx.pesquisar("secreto"))


def test_ignora_pastas_ocultas(ambiente):
    vault, _, idx = ambiente
    oculta = vault / "notas" / "projetos" / ".trash"
    oculta.mkdir()
    (oculta / "lixo.md").write_text("conteúdo apagado do obsidian", encoding="utf-8")
    idx.reindexar_tudo()
    assert idx.pesquisar("obsidian") == []


def test_atualizar_nota_fora_das_permissoes_falha(ambiente):
    _, _, idx = ambiente
    with pytest.raises(PermissaoNegada):
        idx.atualizar_nota("pessoal/orcamento-pessoal.md")
