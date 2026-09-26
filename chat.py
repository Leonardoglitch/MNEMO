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
import json
import os
import sys
from typing import List, Dict, Any

from mnemo import FerramentasVault
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA

from config import Config
from chat_ui import ChatUI

MAX_CICLOS_FERRAMENTAS = 8  # trava de segurança contra um ciclo sem fim

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
        resposta = cliente.conversar(mensagens, ferramentas=FerramentasVault.DESCRICOES)
        mensagens.append(resposta)

        pedidos = resposta.get("tool_calls")
        if not pedidos:
            return resposta.get("content") or ""

        for pedido in pedidos:
            nome = pedido["function"]["name"]
            try:
                argumentos = json.loads(pedido["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                resultado = {"ok": False, "erro": "Argumentos da ferramenta não são JSON válido."}
            else:
                resultado = vault.executar(nome, argumentos)
            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": pedido["id"],
                    "content": json.dumps(resultado, ensure_ascii=False),
                }
            )
    return "(demasiados pedidos de ferramentas seguidos — parei para não entrar em ciclo)"


def _salvar_historico(mensagens: List[Dict[str, Any]], raiz_vault: str) -> None:
    """Grava histórico completo (incl. tool calls) em historico/YYYY-MM-DD_HH-MM.md"""
    from datetime import datetime
    from mnemo import FerramentasVault

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    caminho = f"historico/{ts}.md"

    linhas = [f"# Conversa {ts}\n"]
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
    model_id = config.get("model_default")

    # UI
    config.set("theme", config.get("theme", "auto"))  # garante theme
    ui = ChatUI(config)

    # Health check rápido
    try:
        cliente = ClienteNVIDIA(modelo=model_id) if model_id else ClienteNVIDIA()
    except ErroModeloNVIDIA as e:
        print(f"Erro ao inicializar modelo: {e}", file=sys.stderr)
        sys.exit(1)

    # Inicializa vault e UI
    with FerramentasVault(vault_path) as vault:
        # atualiza completer com pastas permitidas
        pastas = [str(p.relative_to(vault.armazenamento.perms.notas)) for p in vault.armazenamento.perms.raizes_permitidas()]
        ui.config = ui.config  # no-op, garante instância
        ui.update_completer([p.rstrip("/") for p in pastas])

        # health check rápido
        try:
            _ = cliente.conversar([{"role": "user", "content": "ping"}], ferramentas=[])
        except Exception:
            ui.print_error("Health check falhou — API indisponível.")
            sys.exit(1)

        mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}]
        ui.welcome(vault.armazenamento.perms.raiz.name, model_id or "default")

        while True:
            try:
                texto = ui.prompt("Tu: ")
            except (EOFError, KeyboardInterrupt):
                ui.console.print("\n[Saindo... a gravar histórico]")
                _salvar_historico(mensagens, vault_path)
                break

            if not texto:
                continue

            low = texto.lower()
            if low in {"sair", "exit", "quit"}:
                ui.console.print("[Saindo... a gravar histórico]")
                _salvar_historico(mensagens, vault_path)
                break

            if texto == "/salvar":
                _salvar_historico(mensagens, vault_path)
                ui.console.print("[success]Conversa gravada em historico/[/success]\n")
                continue

            if texto == "/historico":
                ui.show_history_list(vault_path)
                continue

            if texto == "/limpar":
                ui.clear()
                continue

            if texto == "/ajuda":
                ui.console.print(Panel(
                    "Comandos disponíveis:\n"
                    "  /salvar       Grava checkpoint da conversa em historico/\n"
                    "  /historico    Lista últimos registos em historico/\n"
                    "  /vault        Mostra vault atual\n"
                    "  /vault <path> Troca vault (reinicializa)\n"
                    "  /modelo <id>  Troca modelo NVIDIA\n"
                    "  /limpar       Limpa ecrã\n"
                    "  /config       Mostra configuração\n"
                    "  /config theme dark|light|auto  Altera tema\n"
                    "  /ajuda        Mostra esta ajuda\n"
                    "  sair / exit / quit   Termina a conversa",
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
                else:
                    ui.console.print("[warning]Uso:[/warning] /config  |  /config theme dark|light|auto")
                continue

            if texto.lower() in {"/ajuda", "/help"}:
                ui.console.print(Panel(
                    "Comandos disponíveis:\n"
                    "  /salvar       Grava checkpoint da conversa em historico/\n"
                    "  /historico    Lista últimos registos em historico/\n"
                    "  /vault        Mostra vault atual\n"
                    "  /vault <path> Troca vault (reinicializa)\n"
                    "  /modelo <id>  Troca modelo NVIDIA\n"
                    "  /limpar       Limpa ecrã\n"
                    "  /config       Mostra configuração\n"
                    "  /config theme dark|light|auto  Altera tema\n"
                    "  /ajuda        Mostra esta ajuda\n"
                    "  sair / exit / quit   Termina a conversa",
                    title="Ajuda", border_style="info"))
                continue

            # mensagem normal do utilizador
            mensagens.append({"role": "user", "content": texto})
            try:
                with ui.thinking():
                    resposta = executar_ciclo_ferramentas(cliente, vault, mensagens)
            except ErroModeloNVIDIA as e:
                ui.print_error(str(e))
                mensagens.pop()
                continue
            ui.print_response(resposta)


if __name__ == "__main__":
    main()