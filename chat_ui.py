"""Interface de utilizador do chat (rich + prompt_toolkit)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Generator, Optional, Union, Tuple, Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent

from config import Config
from rich.table import Table

if TYPE_CHECKING:
    from mnemo.modelo_nvidia import ClienteNVIDIA


class PathCompleter(Completer):
    """Completer contextual: completa paths para comandos de nota, comandos gerais caso contrário."""

    def __init__(self, vault_root: str, base_commands: List[str]) -> None:
        self.vault_root = vault_root
        self.base_commands = base_commands
        from mnemo.permissoes import Permissoes
        self.perms = Permissoes(vault_root)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Generator[Completion, None, None]:
        text = document.text_before_cursor

        # Comandos de nota que aceitam paths
        for cmd in ["create_note", "read_note", "append_to_note", "list_files"]:
            if text.startswith(cmd + " "):
                prefix = text[len(cmd) + 1:]
                yield from self._complete_path(prefix)
                return

        # Completer padrão: comandos + pastas raiz
        for cmd in self.base_commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))
        for p in self.perms.raizes_permitidas():
            folder = str(p.relative_to(self.perms.notas)) + "/"
            if folder.startswith(text):
                yield Completion(folder, start_position=-len(text))

    def _complete_path(self, prefix: str) -> Generator[Completion, None, None]:
        """Completa caminhos relativos a notas/ dentro de pastas permitidas."""
        base = self.perms.notas

        # Se prefix tem "/", completa dentro da pasta
        if "/" in prefix:
            folder, partial = prefix.rsplit("/", 1)
            folder_path = base / folder
            if folder_path.exists() and folder_path.is_dir():
                for f in folder_path.glob(f"{partial}*.md"):
                    yield Completion(f"{folder}/{f.name}", start_position=-len(partial))
        else:
            # Senão, completa pastas permitidas
            for p in self.perms.raizes_permitidas():
                rel = str(p.relative_to(base)) + "/"
                if rel.startswith(prefix):
                    yield Completion(rel, start_position=-len(prefix))


CompleterType = Union[WordCompleter, PathCompleter]


class ChatUI:
    """Camada de apresentação do chat (cores, painéis, spinners, input)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.console = Console(theme=self._build_theme())
        self._base_commands: List[str] = []
        self._completer: Optional[CompleterType] = None
        self._session: Optional[PromptSession[Any]] = None
        # NÃO chama _setup_prompt_session aqui - lazy init

    @property
    def session(self) -> PromptSession[Any]:
        """Lazy initialization do PromptSession para evitar erro em testes sem console."""
        if self._session is None:
            self._setup_prompt_session()
        return self._session

    def _ensure_session(self) -> None:
        """Garante que a sessão foi criada (para métodos que precisam dela)."""
        if self._session is None:
            self._setup_prompt_session()

    # ------------------------------------------------------------------ theme
    def _build_theme(self) -> Theme:
        theme_name = self.config.get("theme", "auto")
        if theme_name == "auto":
            # detecta terminal escuro/claro via env
            import os
            dark = os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit") or \
                   os.environ.get("TERM", "").lower() in ("xterm-256color", "screen-256color")
            theme_name = "dark" if dark else "light"

        if theme_name == "dark":
            return Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "green",
                "muted": "dim white",
                "prompt": "bold cyan",
            })
        else:
            return Theme({
                "info": "blue",
                "warning": "orange3",
                "error": "bold red",
                "success": "green",
                "muted": "dim black",
                "prompt": "bold blue",
            })

    # ------------------------------------------------------------------ prompt_toolkit
    def _setup_prompt_session(self, vault_root: Optional[str] = None) -> None:
        history_file = Path.home() / ".mnemo" / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # completer com comandos + pastas permitidas (será atualizado depois)
        base_commands = [
            "/salvar", "/historico", "/vault", "/modelo",
            "/limpar", "/config", "/ajuda", "/status", "sair", "exit", "quit"
        ]
        self._base_commands = base_commands

        if vault_root:
            completer: CompleterType = PathCompleter(vault_root, base_commands)
        else:
            completer = WordCompleter(base_commands, ignore_case=True, sentence=True)

        kb = KeyBindings()

        @kb.add("c-l")
        def _clear(event) -> None:
            event.app.renderer.clear()

        @kb.add("c-c")
        def _cancel(event) -> None:
            # cancela input atual, não sai do programa
            event.app.current_buffer.reset()

        @kb.add("c-d")
        def _eof(event) -> None:
            # EOF - sair graciosamente
            event.app.exit(result=EOFError)

        self._session = PromptSession(
            history=FileHistory(str(Path.home() / ".mnemo" / "chat_history")),
            completer=completer,
            key_bindings=kb,
            complete_while_typing=True,
        )
        self._completer = completer

    def update_completer(self, vault_root: str) -> None:
        """Atualiza auto-complete com pastas permitidas do vault (com path completion)."""
        self._completer = PathCompleter(vault_root, self._base_commands)
        self._session.completer = self._completer

    # ------------------------------------------------------------------ public API
    def welcome(self, vault: str, model: str) -> None:
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[prompt]Mnemo[/prompt] — vault: [info]{vault}[/info]  modelo: [info]{model}[/info]\n"
                "Comandos: [prompt]/salvar[/prompt]  [prompt]/historico[/prompt]  "
                "[prompt]/vault[/prompt]  [prompt]/modelo[/prompt]  "
                "[prompt]/limpar[/prompt]  [prompt]/config[/prompt]  [prompt]/ajuda[/prompt]  "
                "[prompt]sair[/prompt]",
                title="Bem-vindo",
                border_style="info",
            )
        )

    @contextmanager
    def thinking(self) -> Generator[None, None, None]:
        with Status("[info]A pensar...[/info]", console=self.console, spinner="dots"):
            yield

    def print_streaming(self, generator: Generator[Dict[str, Any], None, None]) -> str:
        """Exibe resposta em streaming com Live e retorna texto completo.
        Simula efeito de streaming palavra-a-palavra (sem chamada extra à API)."""
        import time
        full_text = ""
        md = Markdown("")
        with Live(md, console=self.console, refresh_per_second=15, transient=False) as live:
            for chunk in generator:
                if "delta" in chunk:
                    delta = chunk["delta"]
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        # Re-render markdown with accumulated text
                        md = Markdown(full_text)
                        live.update(md)
                        # Simula streaming visual: ~100 chars/seg
                        time.sleep(0.01)
        return full_text

    def print_response(self, text: str) -> None:
        if not text:
            return
        self.console.print(Markdown(text))

    def print_error(self, msg: str) -> None:
        self.console.print(f"[error]Erro:[/error] {msg}")

    def show_status(self, vault_root: str, cliente: "ClienteNVIDIA", config: Config) -> None:
        """Mostra painel com estado completo do sistema."""
        from mnemo.permissoes import Permissoes

        perms = Permissoes(vault_root)
        pastas = [str(p.relative_to(perms.notas)) for p in perms.raizes_permitidas()]

        notas_por_pasta: Dict[str, int] = {}
        total = 0
        for p in pastas:
            pasta_path = perms.notas / p
            if pasta_path.exists():
                count = len(list(pasta_path.rglob("*.md")))
            else:
                count = 0
            notas_por_pasta[p] = count
            total += count

        index_path = perms.raiz / ".vault" / "index.db"
        index_size = index_path.stat().st_size if index_path.exists() else 0

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="info")
        table.add_column("Value")
        table.add_row("Vault", f"{perms.raiz.name} ({vault_root})")
        table.add_row("Modelo", cliente.modelo)
        table.add_row("Tema", config.get("theme", "auto"))
        table.add_row("", "")
        table.add_row("Pastas permitidas:", "")
        for pasta, count in notas_por_pasta.items():
            table.add_row(f"  {pasta}/", f"{count} notas")
        table.add_row("  TOTAL", f"{total} notas")
        table.add_row("", "")
        table.add_row("Índice FTS5", f"{index_size / 1024:.0f} KB")
        table.add_row("Timeout API", f"{config.get('timeout', 30)}s")
        fallbacks = config.get("fallback_models", [])
        table.add_row("Fallback", ", ".join(fallbacks) if fallbacks else "(nenhum)")

        self.console.print(Panel(table, title="STATUS", border_style="info"))

    def print_panel(self, title: str, content: str) -> None:
        self.console.print(Panel(content, title=title, border_style="info"))

    def prompt(self, prompt_text: str = "Tu: ") -> str:
        return self.session.prompt(prompt_text).strip()

    def clear(self) -> None:
        self.console.clear()
        self.welcome(self.config.get("vault_default", "vault-teste"),
                     self.config.get("model_default", "nvidia/nemotron-3-super-120b-a12b"))

    # ------------------------------------------------------------------ helpers para comandos
    def show_history_list(self, vault_root: str, since: Optional[str] = None, until: Optional[str] = None, model: Optional[str] = None, limit: int = 10) -> None:
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.console.print("[muted]Nenhum histórico ainda.[/muted]")
            return
        files = self._filter_history_files(hist_dir, since, until, model)
        if not files:
            self.console.print("[muted]Nenhum histórico com esses filtros.[/muted]")
            return
        files = files[:limit]
        lines = []
        for f in files:
            try:
                txt = f.read_text(encoding="utf-8")
                preview = txt.split("\n")[1:4]  # pula título
                preview = " ".join(p.strip() for p in preview if p.strip())
                if len(preview) > 120:
                    preview = preview[:117] + "..."
            except Exception:
                preview = "(erro ao ler)"
            lines.append(f"  [info]{f.name}[/info] — {preview}")
        self.console.print("\n".join(lines))

    def _filter_history_files(self, hist_dir: Path, since: Optional[str] = None, until: Optional[str] = None, model: Optional[str] = None) -> List[Path]:
        """Filtra arquivos de histórico por data e modelo."""
        # Parse datas
        since_dt: Optional[datetime] = None
        until_dt: Optional[datetime] = None
        if since:
            try:
                since_dt = datetime.strptime(since, "%Y-%m-%d")
            except ValueError:
                self.print_error(f"Formato de data inválido para --since: {since} (use YYYY-MM-DD)")
                return []
        if until:
            try:
                until_dt = datetime.strptime(until, "%Y-%m-%d")
                # Inclui o dia todo
                until_dt = until_dt.replace(hour=23, minute=59, second=59)
            except ValueError:
                self.print_error(f"Formato de data inválido para --until: {until} (use YYYY-MM-DD)")
                return []

        files = sorted(hist_dir.glob("*.md"), reverse=True)
        filtered: List[Path] = []

        for f in files:
            # Extrai data do nome do arquivo (formato: YYYY-MM-DD_HH-MM.md)
            try:
                date_str = f.stem.split("_")[0] + "_" + f.stem.split("_")[1]
                file_dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")
            except (ValueError, IndexError):
                # Se não conseguir parsear a data, usa mtime
                file_dt = datetime.fromtimestamp(f.stat().st_mtime)

            # Filtro por data
            if since_dt and file_dt < since_dt:
                continue
            if until_dt and file_dt > until_dt:
                continue

            # Filtro por modelo (busca no conteúdo)
            if model:
                try:
                    txt = f.read_text(encoding="utf-8")
                    if model.lower() not in txt.lower():
                        continue
                except Exception:
                    continue

            filtered.append(f)

        return filtered

    def show_config(self) -> None:
        data = self.config.data
        lines = [f"  [prompt]{k}[/prompt]: [info]{v}[/info]" for k, v in data.items()]
        self.console.print(Panel("\n".join(lines), title="Configuração Atual", border_style="info"))

    def set_config(self, key: str, value: str) -> None:
        try:
            # tenta converter para tipo original
            default = self.config.data.get(key)
            if isinstance(default, bool):
                value = value.lower() in ("true", "1", "sim", "yes")
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, list):
                value = [v.strip() for v in value.split(",")]
            self.config.set(key, value)
            self.config.save()
            self.console.print(f"[success]Config[/success] {key} = [info]{value}[/info]")
        except Exception as e:
            self.print_error(str(e))

    def get_model_completions(self) -> List[str]:
        """Retorna modelos disponíveis para auto-complete."""
        fallback = self.config.get("fallback_models", [])
        return [m for m in fallback if m not in self._base_commands]

    # ------------------------------------------------------------------ histórico
    def search_history(self, vault_root: str, termo: str) -> None:
        """Busca termo no histórico e mostra resultados com preview."""
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.console.print("[muted]Nenhum histórico ainda.[/muted]")
            return
        files = sorted(hist_dir.glob("*.md"), reverse=True)
        if not files:
            self.console.print("[muted]Nenhum histórico ainda.[/muted]")
            return

        termo_lower = termo.lower()
        resultados: List[Tuple[str, str]] = []
        for f in files:
            try:
                txt = f.read_text(encoding="utf-8")
                if termo_lower in txt.lower():
                    # Encontra linha com o termo para preview
                    linhas = txt.split("\n")
                    for i, linha in enumerate(linhas):
                        if termo_lower in linha.lower():
                            preview = " ".join(linhas[max(0, i-1):i+2]).strip()
                            if len(preview) > 150:
                                preview = preview[:147] + "..."
                            resultados.append((f.name, preview))
                            break
            except Exception:
                continue

        if not resultados:
            self.console.print(f"[muted]Nenhum resultado para '[info]{termo}[/info]'.[/muted]")
            return

        lines = [f"  [info]{nome}[/info] — {preview}" for nome, preview in resultados[:20]]
        self.console.print(f"\n[success]{len(resultados)} resultado(s) para '[info]{termo}[/info]':[/success]\n")
        self.console.print("\n".join(lines))

    def load_history_session(self, vault_root: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Carrega uma sessão do histórico e retorna lista de mensagens.
        session_id pode ser o nome do arquivo (ex: 2026-09-27_11-11.md) ou prefixo (ex: 2026-09-27).
        """
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.print_error("Diretório de histórico não existe.")
            return []

        # Encontra arquivo
        # Se não tem .md, adiciona
        if not session_id.endswith(".md"):
            session_id += ".md"
        
        # Se parece com prefixo de data (YYYY-MM-DD.md), busca por prefixo
        if session_id.count("-") == 2 and session_id.endswith(".md"):
            # Ex: "2026-09-27.md" -> busca "2026-09-27_*.md"
            prefix = session_id[:-3]  # remove .md
            matches = list(hist_dir.glob(f"{prefix}_*.md"))
        else:
            # Busca exata ou com wildcard
            matches = list(hist_dir.glob(f"*{session_id}*"))
        
        if not matches:
            self.print_error(f"Sessão '{session_id}' não encontrada.")
            return []
        if len(matches) > 1:
            self.print_error(f"Múltiplas sessões correspondem a '{session_id}':")
            for m in matches:
                self.console.print(f"  [info]{m.name}[/info]")
            return []

        arquivo = matches[0]
        try:
            txt = arquivo.read_text(encoding="utf-8")
        except Exception as e:
            self.print_error(f"Erro ao ler arquivo: {e}")
            return []

        # Parse do formato markdown do histórico
        mensagens: List[Dict[str, Any]] = []
        current_role: Optional[str] = None
        current_content: List[str] = []

        for line in txt.split("\n"):
            if line.startswith("## "):
                # Nova mensagem
                if current_role and current_content:
                    mensagens.append({"role": current_role, "content": "\n".join(current_content).strip()})
                role_str = line[3:].strip()
                # Detecta role pelos emojis/labels usados no _salvar_historico
                if "🧑" in role_str or "Tu" in role_str or "user" in role_str.lower() or "utilizador" in role_str.lower():
                    current_role = "user"
                elif "🤖" in role_str or "Mnemo" in role_str or "assistant" in role_str.lower():
                    current_role = "assistant"
                elif "tool" in role_str.lower():
                    current_role = "tool"
                else:
                    current_role = "user"
                current_content = []
            elif line.startswith("---"):
                continue
            else:
                current_content.append(line)

        if current_role and current_content:
            mensagens.append({"role": current_role, "content": "\n".join(current_content).strip()})

        # Filtra mensagens de sistema (não devem ser recarregadas)
        mensagens = [m for m in mensagens if m["role"] != "system"]

        self.console.print(f"[success]Sessão carregada:[/success] [info]{arquivo.name}[/info] ({len(mensagens)} mensagens)")
        return mensagens

    # ------------------------------------------------------------------ histórico interativo (TUI)
    def show_history_interactive(self, vault_root: str, since: Optional[str] = None, until: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
        """
        Abre TUI interativa para navegar no histórico.
        Retorna o nome do arquivo selecionado ou None se cancelado.
        """
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.console.print("[muted]Nenhum histórico ainda.[/muted]")
            return None

        files = self._filter_history_files(hist_dir, since, until, model)
        if not files:
            self.console.print("[muted]Nenhum histórico com esses filtros.[/muted]")
            return None

        # Prepara dados para a lista
        items: List[Tuple[str, str]] = []
        for f in files:
            try:
                txt = f.read_text(encoding="utf-8")
                preview = txt.split("\n")[1:4]
                preview = " ".join(p.strip() for p in preview if p.strip())
                if len(preview) > 100:
                    preview = preview[:97] + "..."
            except Exception:
                preview = "(erro ao ler)"
            items.append((f.name, preview))

        selected_index = [0]  # mutable para closure
        result: List[Optional[str]] = [None]  # para capturar resultado

        def get_formatted_text() -> List[Tuple[str, str]]:
            """Gera texto formatado para a lista."""
            result_text: List[Tuple[str, str]] = []
            for i, (nome, preview) in enumerate(items):
                if i == selected_index[0]:
                    result_text.append(("reverse", f"▶ {nome} — {preview}\n"))
                else:
                    result_text.append(("", f"  {nome} — {preview}\n"))
            return result_text

        control = FormattedTextControl(get_formatted_text, focusable=True)

        kb = KeyBindings()

        @kb.add("up")
        def _up(event) -> None:
            if selected_index[0] > 0:
                selected_index[0] -= 1

        @kb.add("down")
        def _down(event) -> None:
            if selected_index[0] < len(items) - 1:
                selected_index[0] += 1

        @kb.add("enter")
        def _enter(event) -> None:
            result[0] = items[selected_index[0]][0]
            event.app.exit()

        @kb.add("c-c")
        @kb.add("q")
        @kb.add("escape")
        def _quit(event) -> None:
            result[0] = None
            event.app.exit()

        # Style para a lista
        style = Style.from_dict({
            "reverse": "bg:#0055aa #ffffff bold",
        })

        layout = Layout(Window(control, wrap_lines=False, style="class:list"))

        app: Application[Any] = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
        )

        # Executa a aplicação
        try:
            app.run()
        except EOFError:
            pass

        return result[0]

    # ------------------------------------------------------------------ histórico export/import
    def export_history(self, vault_root: str, output_file: str, since: Optional[str] = None, until: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Exporta histórico para JSON.
        """
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.print_error("Diretório de histórico não existe.")
            return

        files = self._filter_history_files(hist_dir, since, until, model)
        if not files:
            self.console.print("[muted]Nenhum histórico com esses filtros.[/muted]")
            return

        export_data: Dict[str, Any] = {
            "exported_at": datetime.now().isoformat(),
            "vault_root": vault_root,
            "filters": {"since": since, "until": until, "model": model},
            "sessions": []
        }

        for f in files:
            try:
                txt = f.read_text(encoding="utf-8")
                # Extrai modelo do cabeçalho se existir
                modelo = ""
                for line in txt.split("\n")[:5]:
                    if line.startswith("**Modelo:**"):
                        modelo = line.replace("**Modelo:**", "").strip()
                        break

                # Parse mensagens
                mensagens: List[Dict[str, Any]] = []
                current_role: Optional[str] = None
                current_content: List[str] = []

                for line in txt.split("\n"):
                    if line.startswith("## "):
                        if current_role and current_content:
                            mensagens.append({"role": current_role, "content": "\n".join(current_content).strip()})
                        role_str = line[3:].strip()
                        if "🧑" in role_str or "Tu" in role_str or "user" in role_str.lower() or "utilizador" in role_str.lower():
                            current_role = "user"
                        elif "🤖" in role_str or "Mnemo" in role_str or "assistant" in role_str.lower():
                            current_role = "assistant"
                        elif "tool" in role_str.lower():
                            current_role = "tool"
                        else:
                            current_role = "user"
                        current_content = []
                    elif line.startswith("---"):
                        continue
                    else:
                        current_content.append(line)

                if current_role and current_content:
                    mensagens.append({"role": current_role, "content": "\n".join(current_content).strip()})

                mensagens = [m for m in mensagens if m["role"] != "system"]

                export_data["sessions"].append({
                    "filename": f.name,
                    "model": modelo,
                    "messages": mensagens
                })
            except Exception as e:
                self.print_error(f"Erro ao processar {f.name}: {e}")
                continue

        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.console.print(f"[success]Histórico exportado:[/success] [info]{output_file}[/info] ({len(export_data['sessions'])} sessões)")
        except Exception as e:
            self.print_error(f"Erro ao escrever arquivo: {e}")

    def import_history(self, vault_root: str, input_file: str) -> int:
        """
        Importa histórico de JSON.
        Retorna número de sessões importadas.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            self.print_error(f"Arquivo não encontrado: {input_file}")
            return 0

        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.print_error(f"Erro ao ler JSON: {e}")
            return 0

        if "sessions" not in data:
            self.print_error("Formato inválido: chave 'sessions' não encontrada")
            return 0

        hist_dir = Path(vault_root) / "notas" / "historico"
        hist_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        for session in data["sessions"]:
            filename = session.get("filename", "")
            if not filename:
                continue

            # Verifica se já existe
            target = hist_dir / filename
            if target.exists():
                self.console.print(f"[warning]Já existe:[/warning] {filename} (pulando)")
                continue

            # Reconstrói markdown
            lines = [f"# Conversa {filename.replace('.md', '')}\n"]
            if session.get("model"):
                lines.append(f"**Modelo:** {session['model']}\n")

            for msg in session.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    role_label = "🧑 Tu"
                elif role == "assistant":
                    role_label = "🤖 Mnemo"
                elif role == "tool":
                    role_label = "🔧 Tool"
                else:
                    role_label = "🧑 Tu"

                if content.strip():
                    lines.append(f"## {role_label}\n{content}\n")

            try:
                target.write_text("\n".join(lines), encoding="utf-8")
                imported += 1
            except Exception as e:
                self.print_error(f"Erro ao gravar {filename}: {e}")

        self.console.print(f"[success]Importadas {imported} sessão(ões) para[/success] [info]{hist_dir}[/info]")
        return imported