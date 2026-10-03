"""Testes das funcionalidades de histórico (chat_ui.py)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from chat_ui import ChatUI
from config import Config


@pytest.fixture
def vault_com_historico(tmp_path):
    """Vault temporário com pasta 'historico' permitida e alguns arquivos de histórico."""
    # Cria estrutura do vault
    (tmp_path / "notas" / "projetos").mkdir(parents=True)
    (tmp_path / "notas" / "historico").mkdir(parents=True)
    (tmp_path / ".vault").mkdir()
    (tmp_path / ".vault" / "config.json").write_text(
        json.dumps({
            "versao": 1,
            "todas_as_pastas": False,
            "pastas_permitidas": ["projetos", "historico"]
        }),
        encoding="utf-8",
    )
    # Cria arquivo de teste em projetos/
    (tmp_path / "notas" / "projetos" / "orcamento.md").write_text(
        "# Orçamento\n\nO orçamento é de mil euros.\n", encoding="utf-8"
    )
    # Cria alguns arquivos de histórico de teste
    hist_dir = tmp_path / "notas" / "historico"
    
    # Sessão 1
    s1 = hist_dir / "2026-09-27_10-30.md"
    s1.write_text("""# Conversa 2026-09-27_10-30

**Modelo:** nvidia/nemotron-3-ultra

## 🧑 Tu
Olá, tudo bem?

## 🤖 Mnemo
Olá! Como posso ajudar?

## 🧑 Tu
Procura orçamento

## 🤖 Mnemo
> **Tool call:** `search`({"consulta": "orçamento"})

## 🔧 Tool (call_1)
> **Tool result** (`call_1`):
```json
{"ok": true, "resultados": [{"caminho": "projetos/orcamento.md", "titulo": "Orçamento"}]}
```

## 🤖 Mnemo
Encontrei o orçamento.
""", encoding="utf-8")
    
    # Sessão 2
    s2 = hist_dir / "2026-09-26_15-45.md"
    s2.write_text("""# Conversa 2026-09-26_15-45

**Modelo:** nvidia/nemotron-3-super-120b-a12b

## 🧑 Tu
Cria uma nota

## 🤖 Mnemo
> **Tool call:** `create_note`({"caminho": "projetos/nova.md", "conteudo": "Nova nota"})

## 🔧 Tool (call_2)
> **Tool result** (`call_2`):
```json
{"ok": true, "caminho": "projetos/nova.md"}
```

## 🤖 Mnemo
Nota criada.
""", encoding="utf-8")
    
    # Sessão 3 (sem modelo no cabeçalho)
    s3 = hist_dir / "2026-09-25_08-00.md"
    s3.write_text("""# Conversa 2026-09-25_08-00

## 🧑 Tu
Teste antigo

## 🤖 Mnemo
Resposta antiga.
""", encoding="utf-8")
    
    return tmp_path


@pytest.fixture
def chat_ui():
    """Instância de ChatUI para testes."""
    config = Config()
    return ChatUI(config)


class TestSearchHistory:
    """Testes para search_history."""
    
    def test_busca_termo_existente(self, chat_ui, vault_com_historico):
        chat_ui.search_history(str(vault_com_historico), "orçamento")
        # Não levanta exceção, apenas imprime
    
    def test_busca_termo_inexistente(self, chat_ui, vault_com_historico, capsys):
        chat_ui.search_history(str(vault_com_historico), "xyzinexistente")
        captured = capsys.readouterr()
        assert "Nenhum resultado" in captured.out
    
    def test_busca_sem_historico(self, chat_ui, tmp_path, capsys):
        # Vault sem pasta historico
        (tmp_path / "notas" / "projetos").mkdir(parents=True)
        (tmp_path / ".vault").mkdir()
        (tmp_path / ".vault" / "config.json").write_text(
            json.dumps({"versao": 1, "todas_as_pastas": False, "pastas_permitidas": ["projetos"]}),
            encoding="utf-8",
        )
        chat_ui.search_history(str(tmp_path), "teste")
        captured = capsys.readouterr()
        assert "Nenhum histórico ainda" in captured.out


class TestLoadHistorySession:
    """Testes para load_history_session."""
    
    def test_carrega_sessao_por_nome_completo(self, chat_ui, vault_com_historico):
        mensagens = chat_ui.load_history_session(str(vault_com_historico), "2026-09-27_10-30.md")
        assert len(mensagens) > 0
        # Verifica estrutura das mensagens
        for msg in mensagens:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant", "tool")
    
    def test_carrega_sessao_por_prefixo(self, chat_ui, vault_com_historico):
        mensagens = chat_ui.load_history_session(str(vault_com_historico), "2026-09-27")
        assert len(mensagens) > 0
    
    def test_carrega_sessao_inexistente(self, chat_ui, vault_com_historico, capsys):
        mensagens = chat_ui.load_history_session(str(vault_com_historico), "2026-01-01.md")
        assert mensagens == []
        captured = capsys.readouterr()
        assert "não encontrada" in captured.out
    
    def test_carrega_sessao_ambigua(self, chat_ui, vault_com_historico, capsys):
        # Cria dois arquivos com prefixo similar
        hist_dir = vault_com_historico / "notas" / "historico"
        (hist_dir / "2026-09-27_11-00.md").write_text("# Conversa 2026-09-27_11-00\n\n## 🧑 Tu\nteste\n", encoding="utf-8")
        
        mensagens = chat_ui.load_history_session(str(vault_com_historico), "2026-09-27")
        assert mensagens == []
        captured = capsys.readouterr()
        assert "Múltiplas sessões" in captured.out
    
    def test_filtra_mensagens_sistema(self, chat_ui, vault_com_historico):
        mensagens = chat_ui.load_history_session(str(vault_com_historico), "2026-09-27_10-30.md")
        # Não deve haver mensagens com role="system"
        assert all(m["role"] != "system" for m in mensagens)


class TestShowHistoryList:
    """Testes para show_history_list com filtros."""
    
    def test_lista_sem_filtros(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico))
        captured = capsys.readouterr()
        assert "2026-09-27_10-30.md" in captured.out
        assert "2026-09-26_15-45.md" in captured.out
        assert "2026-09-25_08-00.md" in captured.out
    
    def test_filtro_since(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico), since="2026-09-27")
        captured = capsys.readouterr()
        assert "2026-09-27_10-30.md" in captured.out
        assert "2026-09-26_15-45.md" not in captured.out
    
    def test_filtro_until(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico), until="2026-09-26")
        captured = capsys.readouterr()
        assert "2026-09-26_15-45.md" in captured.out
        assert "2026-09-27_10-30.md" not in captured.out
    
    def test_filtro_model(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico), model="nemotron-3-ultra")
        captured = capsys.readouterr()
        assert "2026-09-27_10-30.md" in captured.out
        assert "2026-09-26_15-45.md" not in captured.out
    
    def test_filtro_model_case_insensitive(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico), model="NEMOTRON-3-ULTRA")
        captured = capsys.readouterr()
        assert "2026-09-27_10-30.md" in captured.out
    
    def test_limite_resultados(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico), limit=2)
        captured = capsys.readouterr()
        # Deve mostrar apenas 2 resultados
        linhas = [l for l in captured.out.split("\n") if ".md" in l]
        assert len(linhas) <= 2
    
    def test_data_invalida_since(self, chat_ui, vault_com_historico, capsys):
        chat_ui.show_history_list(str(vault_com_historico), since="data-invalida")
        captured = capsys.readouterr()
        assert "Formato de data inválido" in captured.out


class TestExportHistory:
    """Testes para export_history."""
    
    def test_exporta_tudo(self, chat_ui, vault_com_historico, tmp_path):
        output = tmp_path / "export.json"
        chat_ui.export_history(str(vault_com_historico), str(output))
        
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "exported_at" in data
        assert "vault_root" in data
        assert "filters" in data
        assert "sessions" in data
        assert len(data["sessions"]) == 3
    
    def test_exporta_com_filtros(self, chat_ui, vault_com_historico, tmp_path):
        output = tmp_path / "export_filtrado.json"
        chat_ui.export_history(str(vault_com_historico), str(output), since="2026-09-27")
        
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["filename"] == "2026-09-27_10-30.md"
    
    def test_exporta_sem_historico(self, chat_ui, tmp_path, capsys):
        (tmp_path / "notas" / "projetos").mkdir(parents=True)
        (tmp_path / ".vault").mkdir()
        (tmp_path / ".vault" / "config.json").write_text(
            json.dumps({"versao": 1, "todas_as_pastas": False, "pastas_permitidas": ["projetos"]}),
            encoding="utf-8",
        )
        output = tmp_path / "export_vazio.json"
        chat_ui.export_history(str(tmp_path), str(output))
        captured = capsys.readouterr()
        assert "Diretório de histórico não existe" in captured.out


class TestImportHistory:
    """Testes para import_history."""
    
    def test_importa_json_valido(self, chat_ui, vault_com_historico, tmp_path):
        # Cria JSON de exportação
        export_data = {
            "exported_at": "2026-09-27T10:00:00",
            "vault_root": str(vault_com_historico),
            "filters": {},
            "sessions": [
                {
                    "filename": "2026-09-28_12-00.md",
                    "model": "nvidia/nemotron-3-ultra",
                    "messages": [
                        {"role": "user", "content": "Importado"},
                        {"role": "assistant", "content": "OK"}
                    ]
                }
            ]
        }
        import_file = tmp_path / "import.json"
        import_file.write_text(json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8")
        
        count = chat_ui.import_history(str(vault_com_historico), str(import_file))
        assert count == 1
        
        # Verifica se arquivo foi criado
        hist_dir = vault_com_historico / "notas" / "historico"
        assert (hist_dir / "2026-09-28_12-00.md").exists()
    
    def test_importa_pula_duplicados(self, chat_ui, vault_com_historico, tmp_path, capsys):
        # Tenta importar sessão que já existe
        export_data = {
            "exported_at": "2026-09-27T10:00:00",
            "vault_root": str(vault_com_historico),
            "filters": {},
            "sessions": [
                {
                    "filename": "2026-09-27_10-30.md",  # Já existe
                    "model": "nvidia/nemotron-3-ultra",
                    "messages": [{"role": "user", "content": "Importado"}]
                }
            ]
        }
        import_file = tmp_path / "import_dup.json"
        import_file.write_text(json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8")
        
        count = chat_ui.import_history(str(vault_com_historico), str(import_file))
        assert count == 0
        captured = capsys.readouterr()
        assert "Já existe" in captured.out
    
    def test_importa_arquivo_inexistente(self, chat_ui, vault_com_historico, capsys):
        count = chat_ui.import_history(str(vault_com_historico), "/caminho/inexistente.json")
        assert count == 0
        captured = capsys.readouterr()
        assert "não encontrado" in captured.out
    
    def test_importa_json_invalido(self, chat_ui, vault_com_historico, tmp_path, capsys):
        import_file = tmp_path / "invalido.json"
        import_file.write_text("não é json", encoding="utf-8")
        
        count = chat_ui.import_history(str(vault_com_historico), str(import_file))
        assert count == 0
        captured = capsys.readouterr()
        assert "Erro ao ler JSON" in captured.out
    
    def test_importa_formato_invalido(self, chat_ui, vault_com_historico, tmp_path, capsys):
        import_file = tmp_path / "formato_invalido.json"
        import_file.write_text(json.dumps({"outra_chave": "valor"}), encoding="utf-8")
        
        count = chat_ui.import_history(str(vault_com_historico), str(import_file))
        assert count == 0
        captured = capsys.readouterr()
        assert "chave 'sessions' não encontrada" in captured.out


class TestShowStatus:
    """Testes para show_status."""
    
    def test_mostra_status(self, chat_ui, vault_com_historico, capsys):
        # Cria um cliente mock
        class ClienteMock:
            modelo = "nvidia/nemotron-3-ultra"
        
        cliente = ClienteMock()
        config = Config()
        
        chat_ui.show_status(str(vault_com_historico), cliente, config)
        captured = capsys.readouterr()
        assert "STATUS" in captured.out
        assert "Vault" in captured.out
        assert "Modelo" in captured.out
        assert "Pastas permitidas" in captured.out
        assert "Índice FTS5" in captured.out


class TestFilterHistoryFiles:
    """Testes internos para _filter_history_files."""
    
    def test_filtro_data_parse_nome_arquivo(self, chat_ui, vault_com_historico):
        hist_dir = vault_com_historico / "notas" / "historico"
        files = chat_ui._filter_history_files(hist_dir, since="2026-09-27")
        assert len(files) == 1
        assert files[0].name == "2026-09-27_10-30.md"
    
    def test_filtro_data_parse_mtime_fallback(self, chat_ui, tmp_path):
        # Arquivo sem data no nome
        hist_dir = tmp_path / "historico"
        hist_dir.mkdir(parents=True)
        f = hist_dir / "sem_data.md"
        f.write_text("# Conversa\n\n## 🧑 Tu\nteste\n", encoding="utf-8")
        
        # Deve usar mtime do arquivo
        files = chat_ui._filter_history_files(hist_dir)
        assert len(files) == 1
    
    def test_filtro_modelo_no_conteudo(self, chat_ui, vault_com_historico):
        hist_dir = vault_com_historico / "notas" / "historico"
        files = chat_ui._filter_history_files(hist_dir, model="nemotron-3-ultra")
        assert len(files) == 1
        assert files[0].name == "2026-09-27_10-30.md"


class TestPathCompleter:
    """Testes para o PathCompleter."""
    
    def test_completa_pastas_raiz(self, vault_com_historico):
        from chat_ui import PathCompleter
        completer = PathCompleter(str(vault_com_historico), ["/teste"])
        
        # Simula documento vazio
        class MockDoc:
            text_before_cursor = ""
        class MockEvent:
            pass
        
        completions = list(completer.get_completions(MockDoc(), MockEvent()))
        textos = [c.text for c in completions]
        assert "projetos/" in textos
        assert "historico/" in textos
    
    def test_completa_arquivos_dentro_pasta(self, vault_com_historico):
        from chat_ui import PathCompleter
        completer = PathCompleter(str(vault_com_historico), ["/teste"])
        
        class MockDoc:
            text_before_cursor = "create_note projetos/"
        class MockEvent:
            pass
        
        completions = list(completer.get_completions(MockDoc(), MockEvent()))
        textos = [c.text for c in completions]
        # Deve sugerir arquivos .md dentro de projetos/
        assert any("projetos/" in t for t in textos)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])