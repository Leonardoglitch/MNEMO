#!/usr/bin/env python3
"""CLI de conversa com o Nemotron (API da NVIDIA), ligado ao Vault do Mnemo.

Uso:
    export NVIDIA_API_KEY=nvapi-...      # ou põe isto num ficheiro .env
    python chat.py --vault vault-teste

Fica à espera de perguntas na consola. O modelo pode pedir para usar as
ferramentas do vault (search, read_note, create_note, append_to_note);
este programa executa-as e devolve o resultado ao modelo, até ele dar uma
resposta em texto.
"""

import argparse
import ast
import json
import os
import sys
import time
from typing import List, Dict, Any

from mnemo import FerramentasVault
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA

from config import Config, DEFAULTS
from chat_ui import ChatUI
from rich.panel import Panel

MAX_CICLOS_FERRAMENTAS = 15  # trava de segurança contra um ciclo sem fim (aumentado de 8 para 15)

INSTRUCAO_SISTEMA = (
    "És o assistente do Mnemo, uma plataforma pessoal de IA. Tens acesso a um "
    "conjunto de notas (o Vault) através das ferramentas search, read_note, "
    "create_note, append_to_note e list_files. Usa-as sempre que precisares de "
    "consultar ou guardar informação nas notas do utilizador — nunca inventes o "
    "conteúdo de uma nota que não leste. Responde sempre em português.\n\n"
    "Regras para create_note:\n"
    "- O caminho deve ser relativo a notas/ e incluir uma pasta permitida\n"
    "- Pastas permitidas: projetos/, estudo/, historico/\n"
    "- O nome do ficheiro deve terminar em .md\n"
    "- Exemplo correto: projetos/meu-plano.md\n"
    "- Se o utilizador der só um nome (ex.: 'plano'), pergunta em que pasta ou sugere projetos/\n\n"
    "Para listar ou nomear todas as notas, usa `list_files`; usa `search` só quando tiveres palavras-chave concretas.\n\n"
    "Comandos especiais:\n"
    "- `/salvar` — grava checkpoint da conversa em historico/\n"
    "- `/historico` — lista últimos registos\n"
    "- `/vault` — mostra ou troca vault\n"
    "- `/modelo` — troca modelo\n"
    "- `/limpar` — limpa ecrã\n"
    "- `/config` — mostra/altera configuração\n"
    "- Ao sair, o histórico é gravado automaticamente em historico/\n"
)


def carregar_env(caminho: str = ".env") -> None:
    """Lê pares CHAVE=VALOR de um .env simples, sem depender de bibliotecas externas."""
    try:
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                chave, valor = linha.split("=", 1)
                os.environ.setdefault(chave.strip(), valor.strip().strip("'\""))
    except FileNotFoundError:
        pass


def executar_ciclo_ferramentas(
    cliente: ClienteNVIDIA, vault: FerramentasVault, mensagens: List[Dict[str, Any]]
) -> str:
    """Envia mensagens ao modelo e executa as ferramentas que ele pedir."""
    for _ in range(MAX_CICLOS_FERRAMENTAS):
        # Para chamadas de ferramentas, não usar streaming (precisamos do JSON completo)
        resposta = cliente.conversar(mensagens, ferramentas=FerramentasVault.DESCRICOES, stream=False)
        mensagens.append(resposta)

        pedidos = resposta.get("tool_calls")
        if not pedidos:
            # Resposta final sem tool calls
            return resposta.get("content") or ""

        for pedido in pedidos:
            nome = pedido["function"]["name"]
            args_str = pedido["function"]["arguments"] or "{}"
            try:
                argumentos = json.loads(args_str)
            except json.JSONDecodeError:
                # Tenta corrigir JSON comum (aspas simples, trailing commas, etc.)
                try:
                    import ast
                    argumentos = ast.literal_eval(args_str)
                except Exception:
                    resultado = {"ok": False, "erro": f"Argumentos da ferramenta não são JSON válido: {args_str[:100]}"}
                    mensagens.append(
                        {
                            "role": "tool",
                            "tool_call_id": pedido["id"],
                            "content": json.dumps(resultado, ensure_ascii=False),
                        }
                    )
                    continue
            try:
                resultado = vault.executar(nome, argumentos)
            except Exception as e:
                resultado = {"ok": False, "erro": f"Erro ao executar ferramenta: {e}"}
            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": pedido["id"],
                    "content": json.dumps(resultado, ensure_ascii=False),
                }
            )
    return "(demasiados pedidos de ferramentas seguidos — parei para não entrar em ciclo)"


def _salvar_historico(mensagens: List[Dict[str, Any]], raiz_vault: str, modelo: str = "") -> None:
    """Grava histórico completo (incl. tool calls) em historico/YYYY-MM-DD_HH-MM.md"""
    from datetime import datetime
    from mnemo import FerramentasVault

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    caminho = f"historico/{ts}.md"

    linhas = [f"# Conversa {ts}\n"]
    if modelo:
        linhas.append(f"**Modelo:** {modelo}\n")
    for m in mensagens:
        if m["role"] == "system":
            continue

        role_label = "🧑 Tu" if m["role"] == "user" else "🤖 Mnemo"
        content = m.get("content") or ""

        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]["name"]
                args = tc["function"]["arguments"]
                content += f"\n\n> **Tool call:** `{fn}`({args})"

        if m["role"] == "tool":
            content = f"> **Tool result** (`{m.get('tool_call_id')}`):\n```json\n{content}\n```"

        if content.strip():
            linhas.append(f"## {role_label}\n{content}\n")

    conteudo = "\n".join(linhas)

    with FerramentasVault(raiz_vault) as v:
        v.executar("create_note", {"caminho": caminho, "conteudo": conteudo})


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversa com o Nemotron ligado ao Vault do Mnemo.")
    parser.add_argument("--vault", default=None, help="Pasta raiz do vault (padrão: config ou vault-teste)")
    parser.add_argument("--modelo", default=None, help="ID do modelo NVIDIA")
    parser.add_argument("--theme", default=None, choices=["auto", "dark", "light"], help="Tema de cores")
    parser.add_argument("--save-config", action="store_true", help="Gravar vault/modelo/theme como padrão")
    args = parser.parse_args()

    carregar_env()

    # Configuração
    config = Config.load()
    if args.vault:
        config.set("vault_default", args.vault)
    if args.modelo:
        config.set("model_default", args.modelo)
    if args.theme:
        config.set("theme", args.theme)
    if args.save_config:
        config.save()

    vault_path = config.get("vault_default", "vault-teste")

    # Validação de configuração no startup
    errors = config.validate(vault_path)
    if errors:
        ui.console.print("[error]Configuração inválida:[/error]")
        for e in errors:
            ui.console.print(f"  - {e}")
        sys.exit(1)

    # UI
    config.set("theme", config.get("theme", "auto"))  # garante theme
    ui = ChatUI(config)

    # Health check rápido
    # Só passa modelo se foi explicitamente fornecido via --modelo; caso contrário,
    # ClienteNVIDIA lê NVIDIA_MODEL do .env (que usa o default do config se não definido)
    modelo_cli = args.modelo
    try:
        cliente = ClienteNVIDIA(modelo=modelo_cli) if modelo_cli else ClienteNVIDIA()
    except ErroModeloNVIDIA as e:
        print(f"Erro ao inicializar modelo: {e}", file=sys.stderr)
        sys.exit(1)

    # Inicializa vault e UI
    with FerramentasVault(vault_path) as vault:
        # atualiza completer com pastas permitidas (path completion)
        ui.update_completer(vault_path)

        # Health check rápido com retry
        health_ok = False
        for _ in range(3):
            try:
                if cliente.health_check():
                    health_ok = True
                    break
            except Exception:
                pass
            time.sleep(1)
        
        if not health_ok:
            ui.print_error("Health check falhou — API indisponível.")
            sys.exit(1)

        mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}]
        ui.welcome(vault.armazenamento.perms.raiz.name, cliente.modelo)

        while True:
            try:
                texto = ui.prompt("Tu: ")
            except (EOFError, KeyboardInterrupt):
                ui.console.print("\n[Saindo... a gravar histórico]")
                _salvar_historico(mensagens, vault_path, cliente.modelo)
                break

            if not texto:
                continue

            low = texto.lower()
            if low in {"sair", "exit", "quit"}:
                ui.console.print("[Saindo... a gravar histórico]")
                _salvar_historico(mensagens, vault_path, cliente.modelo)
                break

            if texto == "/salvar":
                _salvar_historico(mensagens, vault_path, cliente.modelo)
                ui.console.print("[success]Conversa gravada em historico/[/success]\n")
                continue

            if texto.startswith("/historico"):
                # Parse flags: --since YYYY-MM-DD --until YYYY-MM-DD --model <modelo>
                import shlex
                parts = shlex.split(texto)
                
                # Extrai flags
                since = None
                until = None
                model = None
                subcommand = None
                subcommand_arg = None
                
                i = 1  # pula "/historico"
                while i < len(parts):
                    if parts[i] == "--since" and i + 1 < len(parts):
                        since = parts[i + 1]
                        i += 2
                    elif parts[i] == "--until" and i + 1 < len(parts):
                        until = parts[i + 1]
                        i += 2
                    elif parts[i] == "--model" and i + 1 < len(parts):
                        model = parts[i + 1]
                        i += 2
                    elif parts[i] in ("search", "load", "export", "import"):
                        subcommand = parts[i]
                        if i + 1 < len(parts):
                            subcommand_arg = parts[i + 1]
                        i += 2
                    else:
                        i += 1
                
                if subcommand == "search" and subcommand_arg:
                    ui.search_history(vault_path, subcommand_arg)
                elif subcommand == "load" and subcommand_arg:
                    loaded = ui.load_history_session(vault_path, subcommand_arg)
                    if loaded:
                        mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}] + loaded
                        ui.console.print("[success]Sessão carregada — pode continuar a conversa.[/success]\n")
                elif subcommand == "export" and subcommand_arg:
                    ui.export_history(vault_path, subcommand_arg, since=since, until=until, model=model)
                elif subcommand == "import" and subcommand_arg:
                    ui.import_history(vault_path, subcommand_arg)
                elif subcommand is None:
                    # Modo interativo (TUI) ou lista simples
                    if since or until or model:
                        # Lista filtrada (não interativa)
                        ui.show_history_list(vault_path, since=since, until=until, model=model, limit=20)
                    else:
                        # Modo interativo (TUI)
                        selected = ui.show_history_interactive(vault_path)
                        if selected:
                            loaded = ui.load_history_session(vault_path, selected)
                            if loaded:
                                mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}] + loaded
                                ui.console.print("[success]Sessão carregada — pode continuar a conversa.[/success]\n")
                else:
                    ui.console.print("[warning]Uso:[/warning] /historico [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--model <modelo]]  |  /historico search <termo>  |  /historico load <id>  |  /historico export <arquivo.json> [--since ...] [--until ...] [--model ...]  |  /historico import <arquivo.json>")
                continue

            if texto == "/limpar":
                ui.clear()
                continue

            if texto == "/status":
                ui.show_status(vault_path, cliente, config)
                continue

            if texto == "/ajuda":
                ui.console.print(Panel(
                    "NAVEGAÇÃO & HISTÓRICO\n"
                    "  /historico                    # TUI interativa (↑↓, Enter=carregar, q=sair)\n"
                    "  /historico --since 2026-01-01 # TUI filtrada por data\n"
                    "  /historico --model nemotron   # TUI filtrada por modelo\n"
                    "  /historico search python      # Busca textual com preview\n"
                    "  /historico load 2026-09-27    # Carrega sessão e continua conversa\n"
                    "  /historico export backup.json # Exporta tudo para JSON\n"
                    "  /historico export b.json --since 2026-01-01  # Exporta filtrado\n"
                    "  /historico import backup.json # Importa (pula duplicados)\n\n"
                    "VAULT & CONFIG\n"
                    "  /vault                        # Mostra vault atual\n"
                    "  /vault ~/meu-vault            # Troca vault (reiniciar)\n"
                    "  /status                       # Estado completo do sistema\n"
                    "  /config                       # Mostra config\n"
                    "  /config theme dark            # Tema escuro\n"
                    "  /config model nvidia/nemotron-3-ultra  # Modelo padrão\n"
                    "  /config vault ~/meu-vault     # Vault padrão\n"
                    "  /config timeout 60            # Timeout 60s\n"
                    "  /config fallback m1,m2        # Modelos fallback\n"
                    "  /config reset                 # Reset para padrões\n\n"
                    "MODELO\n"
                    "  /modelo nvidia/nemotron-3-ultra  # Troca modelo (sessão)\n\n"
                    "NOTAS (usam autocomplete Tab)\n"
                    "  create_note projetos/plano.md     # Cria nota\n"
                    "  read_note projetos/plano.md       # Lê nota\n"
                    "  append_to_note projetos/plano.md  # Anexa conteúdo\n"
                    "  search python                     # Busca semântica\n"
                    "  list_files projetos/              # Lista pasta\n\n"
                    "OUTROS\n"
                    "  /salvar              # Checkpoint em historico/\n"
                    "  /limpar              # Limpa ecrã\n"
                    "  sair / exit / quit   # Sai (auto-save)",
                    title="Ajuda", border_style="info"))
                continue

            if texto.startswith("/modelo "):
                novo = texto.split(" ", 1)[1].strip()
                try:
                    cliente = ClienteNVIDIA(modelo=novo)
                    ui.console.print(f"[success]Modelo alterado para:[/success] [info]{novo}[/info]\n")
                except ErroModeloNVIDIA as e:
                    ui.print_error(f"Erro ao trocar modelo: {e}")
                continue

            if texto.startswith("/vault"):
                parts = texto.split()
                if len(parts) == 1:
                    ui.console.print(f"[info]Vault atual:[/info] {vault_path}")
                else:
                    novo_vault = parts[1]
                    ui.console.print(f"[info]A trocar vault para {novo_vault}...[/info]")
                    # reinicia loop com novo vault (simples: reinicia processo)
                    ui.console.print("[warning]Reinicie o programa com --vault <path>[/warning]")
                continue

            if texto.startswith("/config"):
                parts = texto.split()
                if len(parts) == 1:
                    ui.show_config()
                elif parts[1] == "theme" and len(parts) == 3:
                    ui.set_config("theme", parts[2])
                elif parts[1] == "model" and len(parts) == 3:
                    ui.set_config("model_default", parts[2])
                    ui.console.print("[info]Modelo padrão alterado. Reiniciará na próxima sessão.[/info]")
                elif parts[1] == "vault" and len(parts) == 3:
                    ui.set_config("vault_default", parts[2])
                    ui.console.print("[info]Vault padrão alterado. Reiniciará na próxima sessão.[/info]")
                elif parts[1] == "timeout" and len(parts) == 3:
                    try:
                        ui.set_config("timeout", int(parts[2]))
                    except ValueError:
                        ui.print_error("Timeout deve ser um número inteiro (segundos)")
                elif parts[1] == "fallback" and len(parts) >= 3:
                    models = [m.strip() for m in " ".join(parts[2:]).split(",")]
                    ui.set_config("fallback_models", models)
                elif parts[1] == "reset":
                    ui.config.data = {**DEFAULTS}
                    ui.config.save()
                    ui.console.print("[success]Config resetado para padrões.[/success]")
                else:
                    ui.console.print("[warning]Uso:[/warning] /config  |  /config theme dark|light|auto  |  /config model <id>  |  /config vault <path>  |  /config timeout <seg>  |  /config fallback <model1,model2,...>  |  /config reset")
                continue

            if texto.lower() in {"/ajuda", "/help"}:
                ui.console.print(Panel(
                    "Comandos disponíveis:\n"
                    "  /salvar         Grava checkpoint da conversa em historico/\n"
                    "  /historico      Lista últimos registos em historico/\n"
                    "  /vault          Mostra vault atual\n"
                    "  /vault <path>   Troca vault (reinicia necessário)\n"
                    "  /modelo <id>    Troca modelo NVIDIA\n"
                    "  /limpar         Limpa ecrã\n"
                    "  /config         Mostra configuração\n"
                    "  /config theme dark|light|auto   Altera tema\n"
                    "  /config model <id>              Define modelo padrão\n"
                    "  /config vault <path>            Define vault padrão\n"
                    "  /config timeout <seg>           Define timeout API\n"
                    "  /config fallback <m1,m2,...>    Define modelos fallback\n"
                    "  /config reset                   Reseta configuração\n"
                    "  /ajuda           Mostra esta ajuda\n"
                    "  sair / exit / quit   Termina a conversa",
                    title="Ajuda", border_style="info"))
                continue

            # mensagem normal do utilizador
            mensagens.append({"role": "user", "content": texto})
            try:
                with ui.thinking():
                    resposta_texto = executar_ciclo_ferramentas(cliente, vault, mensagens)
                if resposta_texto.startswith("(demasiados pedidos"):
                    ui.print_error(resposta_texto)
                    mensagens.pop()
                    continue
                ui.print_response(resposta_texto)
                mensagens.append({"role": "assistant", "content": resposta_texto})
            except ErroModeloNVIDIA as e:
                ui.print_error(str(e))
                mensagens.pop()
                continue


if __name__ == "__main__":
    main()