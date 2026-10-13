"""Testes para o sistema de tags (Fase 5)."""

import json
import tempfile
from pathlib import Path

import pytest

from mnemo.armazenamento import Armazenamento
from mnemo.ferramentas import FerramentasVault
from mnemo.indexador import Indexador
from mnemo.permissoes import Permissoes


@pytest.fixture
def vault_temp(tmp_path):
    """Vault temporário com estrutura básica."""
    (tmp_path / "notas" / "projetos").mkdir(parents=True)
    (tmp_path / "notas" / "estudo").mkdir(parents=True)
    (tmp_path / ".vault").mkdir()
    (tmp_path / ".vault" / "config.json").write_text(
        json.dumps({
            "versao": 1,
            "todas_as_pastas": False,
            "pastas_permitidas": ["projetos", "estudo"]
        }),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def vault_com_tags(vault_temp):
    """Vault com algumas notas já com tags."""
    # Cria notas com tags no frontmatter
    nota1 = vault_temp / "notas" / "projetos" / "reuniao.md"
    nota1.write_text(
        "---\ntags: [trabalho, reuniao, importante]\n---\n\n# Reunião\nConteúdo da reunião.\n",
        encoding="utf-8",
    )
    
    nota2 = vault_temp / "notas" / "projetos" / "tarefas.md"
    nota2.write_text(
        "---\ntags: [trabalho, tarefa]\n---\n\n# Tarefas\nLista de tarefas.\n",
        encoding="utf-8",
    )
    
    nota3 = vault_temp / "notas" / "estudo" / "python.md"
    nota3.write_text(
        "---\ntags: [estudo, python, programacao]\n---\n\n# Python\nAprendendo Python.\n",
        encoding="utf-8",
    )
    
    nota4 = vault_temp / "notas" / "projetos" / "sem_tags.md"
    nota4.write_text(
        "# Sem Tags\nEsta nota não tem tags.\n",
        encoding="utf-8",
    )
    
    return vault_temp


class TestArmazenamentoTags:
    """Testes para operações de tags no Armazenamento."""
    
    def test_ler_tags_existentes(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        tags = arm.ler_tags("projetos/reuniao.md")
        assert tags == ["importante", "reuniao", "trabalho"]
    
    def test_ler_tags_inexistente(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        tags = arm.ler_tags("projetos/sem_tags.md")
        assert tags == []
    
    def test_ler_tags_sem_frontmatter(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        tags = arm.ler_tags("projetos/sem_tags.md")
        assert tags == []
    
    def test_adicionar_tag_nova(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        novas = arm.adicionar_tag("projetos/tarefas.md", "urgente")
        assert "urgente" in novas
        assert "trabalho" in novas
        assert "tarefa" in novas
        assert len(novas) == 3
    
    def test_adicionar_tag_duplicada(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        novas = arm.adicionar_tag("projetos/reuniao.md", "trabalho")
        # Não deve duplicar
        assert novas.count("trabalho") == 1
    
    def test_remover_tag_existente(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        novas = arm.remover_tag("projetos/reuniao.md", "importante")
        assert "importante" not in novas
        assert "reuniao" in novas
        assert "trabalho" in novas
    
    def test_remover_tag_inexistente(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        novas = arm.remover_tag("projetos/reuniao.md", "inexistente")
        # Não deve alterar
        assert novas == ["importante", "reuniao", "trabalho"]
    
    def test_substituir_tags(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        novas = arm.substituir_tags("projetos/reuniao.md", ["nova", "tags"])
        assert novas == ["nova", "tags"]
    
    def test_escrever_tags_normaliza(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        arm = Armazenamento(perms)
        
        # Tags com maiúsculas, espaços, duplicadas
        arm.escrever_tags("projetos/tarefas.md", ["TRABALHO", "  urgente  ", "trabalho", "URGENTE"])
        tags = arm.ler_tags("projetos/tarefas.md")
        assert tags == ["trabalho", "urgente"]


class TestIndexadorTags:
    """Testes para indexação e busca por tags no Indexador."""
    
    def test_indexar_notas_com_tags(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        idx = Indexador(perms)
        idx.reindexar_tudo()
        
        # Verifica se tags foram indexadas na tabela nota_tags
        cursor = idx.con.execute("SELECT tag, COUNT(*) as cnt FROM nota_tags GROUP BY tag")
        tags_db = {row[0]: row[1] for row in cursor}
        
        assert tags_db.get("trabalho") == 2
        assert tags_db.get("reuniao") == 1
        assert tags_db.get("importante") == 1
        assert tags_db.get("tarefa") == 1
        assert tags_db.get("estudo") == 1
        assert tags_db.get("python") == 1
        assert tags_db.get("programacao") == 1
    
    def test_pesquisar_por_tag_all(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        idx = Indexador(perms)
        idx.reindexar_tudo()
        
        # Busca notas que tenham AMBAS as tags "trabalho" E "reuniao"
        resultados = idx.pesquisar("", tags=["trabalho", "reuniao"], modo_tags="all")
        assert len(resultados) == 1
        assert resultados[0]["caminho"] == "projetos/reuniao.md"
    
    def test_pesquisar_por_tag_any(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        idx = Indexador(perms)
        idx.reindexar_tudo()
        
        # Busca notas que tenham PELO MENOS UMA das tags "reuniao" OU "python"
        resultados = idx.pesquisar("", tags=["reuniao", "python"], modo_tags="any")
        assert len(resultados) == 2
        caminhos = {r["caminho"] for r in resultados}
        assert "projetos/reuniao.md" in caminhos
        assert "estudo/python.md" in caminhos
    
    def test_pesquisar_texto_com_tag(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        idx = Indexador(perms)
        idx.reindexar_tudo()
        
        # Busca textual "reunião" + filtro tag "trabalho"
        resultados = idx.pesquisar("reunião", tags=["trabalho"], modo_tags="all")
        assert len(resultados) == 1
        assert resultados[0]["caminho"] == "projetos/reuniao.md"
    
    def test_pesquisar_tag_inexistente(self, vault_com_tags):
        perms = Permissoes(str(vault_com_tags))
        idx = Indexador(perms)
        idx.reindexar_tudo()
        
        resultados = idx.pesquisar("", tags=["inexistente"], modo_tags="all")
        assert resultados == []


class TestFerramentasTags:
    """Testes para as ferramentas de tags do agente."""
    
    def test_tag_note_add(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("tag_note", {
                "caminho": "projetos/tarefas.md",
                "acao": "add",
                "tags": ["urgente"]
            })
            assert resultado["ok"] is True
            assert "urgente" in resultado["tags"]
    
    def test_tag_note_remove(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("tag_note", {
                "caminho": "projetos/reuniao.md",
                "acao": "remove",
                "tags": ["importante"]
            })
            assert resultado["ok"] is True
            assert "importante" not in resultado["tags"]
    
    def test_tag_note_set(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("tag_note", {
                "caminho": "projetos/tarefas.md",
                "acao": "set",
                "tags": ["nova", "lista"]
            })
            assert resultado["ok"] is True
            assert resultado["tags"] == ["lista", "nova"]
    
    def test_tag_note_acao_invalida(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("tag_note", {
                "caminho": "projetos/tarefas.md",
                "acao": "invalida",
                "tags": ["teste"]
            })
            assert resultado["ok"] is False
            assert "Ação inválida" in resultado["erro"]
    
    def test_search_by_tag_all(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("search_by_tag", {
                "tags": ["trabalho", "reuniao"],
                "modo": "all",
                "limite": 10
            })
            assert resultado["ok"] is True
            assert len(resultado["resultados"]) == 1
            assert resultado["resultados"][0]["caminho"] == "projetos/reuniao.md"
    
    def test_search_by_tag_any(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("search_by_tag", {
                "tags": ["reuniao", "python"],
                "modo": "any",
                "limite": 10
            })
            assert resultado["ok"] is True
            assert len(resultado["resultados"]) == 2
    
    def test_search_by_tag_vazio(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("search_by_tag", {
                "tags": [],
                "modo": "all"
            })
            assert resultado["ok"] is False
            assert "vazia" in resultado["erro"]
    
    def test_list_tags_todas(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("list_tags", {})
            assert resultado["ok"] is True
            assert resultado["total"] == 7  # trabalho, reuniao, importante, tarefa, estudo, python, programacao
            
            tags_dict = {t["tag"]: t["count"] for t in resultado["tags"]}
            assert tags_dict["trabalho"] == 2
            assert tags_dict["reuniao"] == 1
    
    def test_list_tags_com_pasta(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("list_tags", {"pasta": "projetos"})
            assert resultado["ok"] is True
            tags_dict = {t["tag"]: t["count"] for t in resultado["tags"]}
            assert tags_dict["trabalho"] == 2
            assert tags_dict["reuniao"] == 1
            assert tags_dict["importante"] == 1
            assert tags_dict["tarefa"] == 1
            assert "estudo" not in tags_dict
            assert "python" not in tags_dict
    
    def test_list_tags_com_prefixo(self, vault_com_tags):
        with FerramentasVault(str(vault_com_tags)) as ferramentas:
            resultado = ferramentas.executar("list_tags", {"prefixo": "tra"})
            assert resultado["ok"] is True
            tags = [t["tag"] for t in resultado["tags"]]
            assert "trabalho" in tags
            # "tarefa" starts with "tar", not "tra"
            # assert "tarefa" in tags
            assert "programacao" not in tags


class TestIntegracaoTags:
    """Testes de integração completa do fluxo de tags."""
    
    def test_fluxo_completo_tags(self, vault_temp):
        """Testa fluxo: criar nota -> adicionar tags -> buscar por tag -> listar tags."""
        with FerramentasVault(str(vault_temp)) as ferramentas:
            # 1. Cria nota
            res = ferramentas.executar("create_note", {
                "caminho": "projetos/minha-nota.md",
                "conteudo": "# Minha Nota\nConteúdo de teste."
            })
            assert res["ok"] is True
            
            # 2. Adiciona tags
            res = ferramentas.executar("tag_note", {
                "caminho": "projetos/minha-nota.md",
                "acao": "add",
                "tags": ["importante", "projeto"]
            })
            assert res["ok"] is True
            assert "importante" in res["tags"]
            assert "projeto" in res["tags"]
            
            # 3. Busca por tag
            res = ferramentas.executar("search_by_tag", {
                "tags": ["importante"],
                "modo": "all"
            })
            assert res["ok"] is True
            assert len(res["resultados"]) == 1
            assert res["resultados"][0]["caminho"] == "projetos/minha-nota.md"
            
            # 4. Lista tags
            res = ferramentas.executar("list_tags", {})
            assert res["ok"] is True
            tags_dict = {t["tag"]: t["count"] for t in res["tags"]}
            assert tags_dict["importante"] == 1
            assert tags_dict["projeto"] == 1
    
    def test_tags_persistiram_apos_reindexar(self, vault_com_tags):
        """Verifica que tags persistem após reindexar_tudo()."""
        perms = Permissoes(str(vault_com_tags))
        idx = Indexador(perms)
        idx.reindexar_tudo()
        
        # Adiciona tag via armazenamento
        arm = Armazenamento(perms, indexador=idx)
        arm.adicionar_tag("projetos/tarefas.md", "nova-tag")
        
        # Reindexa
        idx.reindexar_tudo()
        
        # Verifica se tag persistiu
        cursor = idx.con.execute(
            "SELECT tag FROM nota_tags nt JOIN notas n ON n.id = nt.nota_id WHERE n.caminho = ?",
            ("projetos/tarefas.md",)
        )
        tags = [row[0] for row in cursor]
        assert "nova-tag" in tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])