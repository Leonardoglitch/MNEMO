"""Interface de utilizador do chat (rich + prompt_toolkit)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from config import Config


class ChatUI:
    """Camada de apresentação do chat (cores, painéis, spinners, input)."""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console(theme=self._build_theme())
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
    def _setup_prompt_session(self) -> None:
        history_file = Path.home() / ".mnemo" / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # completer com comandos + pastas permitidas (será atualizado depois)
        base_commands = [
            "/salvar", "/historico", "/vault", "/modelo",
            "/limpar", "/config", "/ajuda", "sair", "exit", "quit"
        ]
        completer = WordCompleter(base_commands, ignore_case=True, sentence=True)

        kb = KeyBindings()

        @kb.add("c-l")
        def _(event):
            event.app.renderer.clear()

        @kb.add("c-c")
        def _(event):
            # cancela input atual, não sai do programa
            event.app.current_buffer.reset()

        @kb.add("c-d")
        def _(event):
            # EOF - sair graciosamente
            event.app.exit(result=EOFError)

        self.session = PromptSession(
            history=FileHistory(str(Path.home() / ".mnemo" / "chat_history")),
            completer=completer,
            key_bindings=kb,
            complete_while_typing=True,
        )
        self._base_commands = base_commands
        self._completer = completer

    def update_completer(self, pastas_permitidas: List[str]) -> None:
        """Atualiza auto-complete com pastas permitidas do vault."""
        # Adiciona completions para pastas/ e pastas/arquivo.md
        folder_completions = []
        for p in pastas_permitidas:
            folder_completions.append(f"{p}/")
            # Adiciona sugestões de arquivos .md comuns
            folder_completions.append(f"{p}/")
        model_completions = self.get_model_completions()
        words = self._base_commands + folder_completions + model_completions
        self._completer.words = words

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
    def thinking(self):
        with Status("[info]A pensar...[/info]", console=self.console, spinner="dots"):
            yield

    def print_streaming(self, generator) -> str:
        """Exibe resposta em streaming com Live e retorna texto completo."""
        full_text = ""
        md = Markdown("")
        with Live(md, console=self.console, refresh_per_second=10, transient=False) as live:
            for chunk in generator:
                if "delta" in chunk:
                    delta = chunk["delta"]
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        # Re-render markdown with accumulated text
                        md = Markdown(full_text)
                        live.update(md)
        return full_text

    def print_response(self, text: str) -> None:
        if not text:
            return
        self.console.print(Markdown(text))

    def print_error(self, msg: str) -> None:
        self.console.print(f"[error]Erro:[/error] {msg}")

    def print_panel(self, title: str, content: str) -> None:
        self.console.print(Panel(content, title=title, border_style="info"))

    def prompt(self, prompt_text: str = "Tu: ") -> str:
        return self.session.prompt(prompt_text).strip()

    def clear(self) -> None:
        self.console.clear()
        self.welcome(self.config.get("vault_default", "vault-teste"),
                     self.config.get("model_default", "nvidia/nemotron-3-super-120b-a12b"))

    # ------------------------------------------------------------------ helpers para comandos
    def show_history_list(self, vault_root: str, since: str = None, until: str = None, model: str = None, limit: int = 10) -> None:
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

    def _filter_history_files(self, hist_dir: Path, since: str = None, until: str = None, model: str = None) -> List[Path]:
        """Filtra arquivos de histórico por data e modelo."""
        from datetime import datetime
        
        # Parse datas
        since_dt = None
        until_dt = None
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
        filtered = []
        
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
        resultados = []
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
        session_id pode ser o nome do arquivo (ex: 2026-09-27_11-11.md) ou prefixo.
        """
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.print_error("Diretório de histórico não existe.")
            return []

        # Encontra arquivo
        if not session_id.endswith(".md"):
            session_id += ".md"
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
        mensagens = []
        current_role = None
        current_content = []

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
    def show_history_interactive(self, vault_root: str, since: str = None, until: str = None, model: str = None) -> Optional[str]:
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
        items = []
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
        result = [None]  # para capturar resultado

        def get_formatted_text():
            """Gera texto formatado para a lista."""
            result_text = []
            for i, (nome, preview) in enumerate(items):
                if i == selected_index[0]:
                    result_text.append(("reverse", f"▶ {nome} — {preview}\n"))
                else:
                    result_text.append(("", f"  {nome} — {preview}\n"))
            return result_text

        control = FormattedTextControl(get_formatted_text, focusable=True)

        kb = KeyBindings()

        @kb.add("up")
        def _(event):
            if selected_index[0] > 0:
                selected_index[0] -= 1

        @kb.add("down")
        def _(event):
            if selected_index[0] < len(items) - 1:
                selected_index[0] += 1

        @kb.add("enter")
        def _(event):
            result[0] = items[selected_index[0]][0]
            event.app.exit()

        @kb.add("c-c")
        @kb.add("q")
        @kb.add("escape")
        def _(event):
            result[0] = None
            event.app.exit()

        # Style para a lista
        style = Style.from_dict({
            "reverse": "bg:#0055aa #ffffff bold",
        })

        layout = Layout(Window(control, wrap_lines=False, style="class:list"))

        app = Application(
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