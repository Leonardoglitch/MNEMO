# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não Lançado]

### Adicionado
- Sistema de histórico navegável com TUI interativa (`/historico`)
- Filtros de histórico por data (`--since`, `--until`) e modelo (`--model`)
- Busca textual no histórico (`/historico search <termo>`)
- Carregamento de sessão anterior (`/historico load <id>`)
- Exportação de histórico para JSON (`/historico export <arquivo.json>`)
- Importação de histórico de JSON (`/historico import <arquivo.json>`)
- Comando `/status` com painel de informações do vault
- Autocomplete contextual de paths (Tab-complete para create_note, read_note, etc.)
- Validação de configuração na inicialização (vault, API key, pastas permitidas)
- Painel de ajuda reestruturado (`/ajuda`) com exemplos copy-paste
- 29 novos testes de integração para histórico (`tests/test_historico.py`)
- Type hints completos com `mypy --strict` nos módulos core (`mnemo/`, `config.py`)

### Alterado
- `chat_ui.py`: Adicionadas classes `HistoryBrowser`, `PathCompleter`, métodos de export/import
- `chat.py`: Novos handlers para comandos de histórico, status, autocomplete
- `config.py`: Validação de configuração na inicialização
- `README.md`: Documentação dos novos comandos e funcionalidades

### Corrigido
- Tratamento de erros em importação de histórico (formato inválido, duplicados)
- Filtros de data no histórico com fallback para mtime
- Autocomplete respeita apenas pastas permitidas

## [1.0.0] - 2026-09-20

### Adicionado
- Vault pessoal com notas Markdown
- Pesquisa full-text com SQLite FTS5
- Indexação incremental com remoção de diacríticos
- Cliente NVIDIA Nemotron (OpenAI-compatível)
- Loop de tool-calling com 5 ferramentas: search, read_note, create_note, append_to_note, list_files
- Interface REPL com Rich + prompt_toolkit
- Temas de cores (auto, dark, light)
- Configuração persistente em `~/.mnemo/config.json`
- Sistema de permissões com whitelist de pastas
- Backups automáticos (`.backups/`) e lixeira (`.lixo/`)
- 56 testes unitários e de integração

### Componentes Core
- `mnemo/permissoes.py` — Whitelist, path traversal protection
- `mnemo/armazenamento.py` — CRUD atômico, backups, lixo
- `mnemo/indexador.py` — SQLite FTS5, prefix search, diacríticos
- `mnemo/ferramentas.py` — 5 ferramentas do agente
- `mnemo/modelo_nvidia.py` — Cliente HTTP streaming + non-streaming
- `chat.py` — Loop principal, histórico de mensagens
- `chat_ui.py` — UI rica, spinners, formatação
- `config.py` — Configuração persistente