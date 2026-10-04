# Guia de Contribuição

Obrigado por contribuir com o MNEMO! Este documento orienta como configurar o ambiente, rodar testes, fazer linting e enviar PRs.

---

## Setup do Ambiente

```bash
# Clone o repositório
git clone https://github.com/Leonardoglitch/MNEMO.git
cd MNEMO

# Crie virtualenv (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instale em modo desenvolvimento
pip install -e .

# Instale dependências de desenvolvimento
pip install pytest mypy black isort ruff
```

---

## Comandos Essenciais

### Testes
```bash
# Todos os testes (85)
python -m pytest tests/ -v

# Testes específicos
python -m pytest tests/test_historico.py -v
python -m pytest tests/test_ferramentas.py::TestSearch -v

# Com coverage
python -m pytest tests/ --cov=mnemo --cov=chat --cov=chat_ui --cov=config
```

### Type Checking
```bash
# Módulos core (deve passar sem erros)
python -m mypy --strict mnemo/ config.py

# Projeto completo (37 erros conhecidos, não bloqueantes)
python -m mypy --strict mnemo/ chat.py chat_ui.py config.py
```

### Formatação e Linting
```bash
# Black (formatação)
black --line-length 100 mnemo/ chat.py chat_ui.py config.py tests/

# isort (imports)
isort --profile black mnemo/ chat.py chat_ui.py config.py tests/

# Ruff (linting rápido)
ruff check mnemo/ chat.py chat_ui.py config.py tests/
ruff check --fix mnemo/ chat.py chat_ui.py config.py tests/
```

### Pré-commit (recomendado)
```bash
pip install pre-commit
pre-commit install
# Roda black, isort, ruff, mypy (core) em cada commit
```

---

## Estrutura de Branches

```
main                    # Protegida, apenas via PR
├── feat/nova-feature   # Nova funcionalidade
├── fix/bug-description # Correção de bug
├── docs/atualizacao    # Documentação
├── refactor/...        # Refatoração sem mudança de comportamento
└── test/...            # Adição/ajuste de testes
```

---

## Convenções de Commit

Use **Conventional Commits** (imperativo, minúsculas):

| Tipo | Exemplo |
|------|---------|
| `feat:` | `feat: adiciona busca semântica com FAISS` |
| `fix:` | `fix: corrige path traversal em armazenamento` |
| `docs:` | `docs: atualiza README com novos comandos` |
| `refactor:` | `refactor: extrai validação para Permissoes.validate()` |
| `test:` | `test: adiciona testes para export/import histórico` |
| `chore:` | `chore: atualiza dependências no pyproject.toml` |
| `ci:` | `ci: adiciona workflow GitHub Actions` |

**Exemplo completo:**
```
feat: adiciona comando /status com painel de informações do vault

- Novo método show_status() em chat_ui.py
- Handler em chat.py para comando /status
- Teste em tests/test_historico.py::TestShowStatus
- Atualiza README com documentação do comando
```

---

## Adicionando Nova Funcionalidade

### 1. Nova Ferramenta do Agente

```python
# 1. Em mnemo/ferramentas.py
def nova_ferramenta(self, arg1: str, arg2: int = 10) -> dict:
    """Docstring descritiva."""
    # Validação via self.permissoes
    # Operação via self.armazenamento / self.indexador
    return {"resultado": "..."}

# 2. Registrar schema OpenAI
def get_tool_definitions(self) -> List[dict]:
    return [
        # ... existentes ...
        {
            "type": "function",
            "function": {
                "name": "nova_ferramenta",
                "description": "Descrição para o modelo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"},
                        "arg2": {"type": "integer", "default": 10}
                    },
                    "required": ["arg1"]
                }
            }
        }
    ]

# 3. Handler em chat.py (loop tool-calling)
elif tool_name == "nova_ferramenta":
    resultado = self.ferramentas.executar("nova_ferramenta", arguments)
    mensagens.append({"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(resultado)})

# 4. Testes em tests/test_ferramentas.py
class TestNovaFerramenta:
    def test_nova_ferramenta_basico(self, vault_com_historico):
        ...

# 5. Validação
python -m pytest tests/test_ferramentas.py::TestNovaFerramenta -v
python -m mypy --strict mnemo/ferramentas.py
```

### 2. Novo Comando REPL

```python
# Em chat.py, método processar_comando()
elif texto.startswith("/novo_comando"):
    args = texto.split()[1:]
    # Parse args (argparse ou manual)
    self.ui.novo_metodo(args)
    return True

# Em chat_ui.py
def novo_metodo(self, args: List[str]) -> None:
    # Implementação
    self.console.print("[green]Feito![/green]")

# Documentar em README.md (tabela de comandos)
```

### 3. Novo Módulo Core

```
mnemo/
├── novo_modulo.py      # Implementação com type hints completos
├── __init__.py         # Exportar: from .novo_modulo import NovaClasse
```

- Type hints **obrigatórios** (`mypy --strict` deve passar)
- Testes em `tests/test_novo_modulo.py`
- Documentar em `docs/ARCHITECTURE.md` e `docs/API.md`

---

## Padrões de Código

### Type Hints
```python
# Bom: tipos explícitos
def buscar(self, termo: str, limite: int = 10) -> List[Dict[str, Any]]:
    ...

# Evite: Any desnecessário
def processar(dados: Any) -> Any:  # ❌
    ...

# Use: Union, Optional, Literal
from typing import Union, Optional, Literal
def configurar(opcao: Literal["auto", "dark", "light"] = "auto") -> None:
    ...
```

### Testes
```python
# Fixtures em conftest.py (futuro) ou no arquivo de teste
@pytest.fixture
def vault_temp(tmp_path):
    # Setup isolado, sem side effects
    ...

# Classes de teste organizadas por feature
class TestBusca:
    def test_busca_term_simples(self, vault_temp): ...
    def test_busca_com_filtro_data(self, vault_temp): ...
    def test_busca_sem_resultados(self, vault_temp): ...

# Nomes descritivos: test_<o_que>_<condicao>_<resultado_esperado>
def test_criar_nota_em_pasta_permitida_sucesso(self, vault_temp): ...
def test_criar_nota_fora_pasta_permitida_levanta_erro(self, vault_temp): ...
```

### Erros e Exceções
```python
# Exceções customizadas em mnemo/excecoes.py (futuro)
class VaultError(Exception): ...
class PermissaoError(VaultError): ...
class IndiceError(VaultError): ...

# Levantar com contexto
raise PermissaoError(f"Pasta '{pasta}' não permitida. Permitidas: {permitidas}")
```

---

## Checklist de PR

Antes de abrir PR, confirme:

- [ ] `python -m pytest tests/ -v` → **85 passed**
- [ ] `python -m mypy --strict mnemo/ config.py` → **0 erros**
- [ ] `black --check --line-length 100 mnemo/ chat.py chat_ui.py config.py tests/`
- [ ] `isort --check --profile black mnemo/ chat.py chat_ui.py config.py tests/`
- [ ] `ruff check mnemo/ chat.py chat_ui.py config.py tests/`
- [ ] Testes novos para funcionalidade nova
- [ ] Documentação atualizada (README.md, docs/, docstrings)
- [ ] Commit messages seguem Conventional Commits
- [ ] Branch atualizada com `main` (rebase ou merge)

---

## Reportando Bugs

Use **GitHub Issues** com template:

```markdown
## Descrição
Breve descrição do bug.

## Passos para Reproduzir
1. ...
2. ...
3. ...

## Comportamento Esperado
O que deveria acontecer.

## Comportamento Atual
O que acontece (erro, stack trace, logs).

## Ambiente
- OS: Windows 11 / Linux / macOS
- Python: 3.11.x
- MNEMO: commit hash ou versão

## Contexto Adicional
Screenshots, configuração relevante, etc.
```

---

## Dúvidas?

- Abra uma **Discussion** no GitHub para perguntas gerais
- Para bugs: **Issue** com template acima
- Para features: **Issue** com label `enhancement` + descrição do caso de uso