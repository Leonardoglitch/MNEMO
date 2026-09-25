# MNEMO — Vault Pessoal com IA

Plataforma local-first para notas Markdown com pesquisa semântica e assistente conversacional (Nemotron/NVIDIA).

## Arquitetura (Fase 1)

```mermaid
graph TD
    A[chat.py] --> B[FerramentasVault]
    B --> C[Permissoes]
    B --> D[Armazenamento]
    B --> E[Indexador]
    D --> F[(.backups/ .lixo/)]
    E --> G[(.vault/index.db)]
    C --> H[.vault/config.json]
```

## Estrutura do Projeto

```
MNEMO/
├── mnemo/              # pacote Python
│   ├── __init__.py
│   ├── permissoes.py
│   ├── armazenamento.py
│   ├── indexador.py
│   ├── ferramentas.py
│   └── modelo_nvidia.py
├── tests/              # 35 testes pytest
├── vault-teste/        # vault de demonstração (commit-safe)
├── chat.py             # CLI interativo
├── .env.example        # template de variáveis
└── .gitignore
```

## Instalação

```bash
git clone https://github.com/Leonardoglitch/MNEMO.git
cd MNEMO
# Python 3.9+, sem dependências externas obrigatórias
```

## Configuração

```bash
cp .env.example .env
# edite .env e insira sua chave:
# NVIDIA_API_KEY=nvapi-...
```

## Testes

```bash
python -m pytest tests/ -v
# 35 testes: permissões, índice, modelo NVIDIA, chat, ferramentas
```

## Uso Rápido

```bash
# Indexar vault de teste
python -c "from mnemo import Permissoes, Indexador; i=Indexador(Permissoes('vault-teste')); print(i.reindexar_tudo())"

# Chat interativo
python chat.py --vault vault-teste
# Tu: procura orçamento
# Mnemo: [resultados da pesquisa] ...
```

## Componentes Principais

| Módulo | Responsabilidade |
|--------|------------------|
| `permissoes.py` | Whitelist de pastas, bloqueio de path traversal |
| `armazenamento.py` | CRUD atómico + backups + lixo (.lixo/) |
| `indexador.py` | SQLite FTS5, prefixo + remove_diacritics |
| `ferramentas.py` | search, read_note, create_note, append_to_note |
| `modelo_nvidia.py` | Cliente HTTP OpenAI-compatível p/ Nemotron |
| `chat.py` | Loop tool-calling com histórico de mensagens |

## Convenções Git

- `git add -A` (apanha dotfiles)
- Commits atómicos, mensagens imperativas ("feat:", "fix:", "docs:")
- Sem `commit --amend` no histórico partilhado
- Branch `main` protegida; PRs para revisão