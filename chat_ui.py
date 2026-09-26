"""Interface de utilizador do chat (rich + prompt_toolkit)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings

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
        completer = WordCompleter(base_commands, ignore_case=True)

        kb = KeyBindings()

        @kb.add("c-l")
        def _(event):
            event.app.renderer.clear()

        @kb.add("c-c")
        def _(event):
            # cancela input atual, não sai do programa
            event.app.current_buffer.reset()

        self.session = PromptSession(
            history=FileHistory(str(Path.home() / ".mnemo" / "chat_history")),
            completer=completer,
            key_bindings=kb,
        )
        self._base_commands = base_commands
        self._completer = completer

    def update_completer(self, pastas_permitidas: List[str]) -> None:
        """Atualiza auto-complete com pastas permitidas do vault."""
        words = self._base_commands + [f"{p}/" for p in pastas_permitidas]
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
    def show_history_list(self, vault_root: str) -> None:
        hist_dir = Path(vault_root) / "notas" / "historico"
        if not hist_dir.exists():
            self.console.print("[muted]Nenhum histórico ainda.[/muted]")
            return
        files = sorted(hist_dir.glob("*.md"), reverse=True)[:10]
        if not files:
            self.console.print("[muted]Nenhum histórico ainda.[/muted]")
            return
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