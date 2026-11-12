#!/usr/bin/env python3
"""
CLI de conversa com o Nemotron (API da NVIDIA), ligado ao Vault do Mnemo.

Uso:
    export NVIDIA_API_KEY=nvapi-...      # ou põe isto num ficheiro .env
    python chat.py --vault vault-teste

Fica à espera de perguntas na consola. O modelo pode pedir para usar as
ferramentas do vault (search, read_note, create_note, append_to_note, list_files);
este programa executa-as e devolve o resultado ao modelo, até ele dar uma
resposta em texto.

Arquitetura:
- Loop principal em main(): gere ciclo de vida da conversa
- executar_ciclo_ferramentas(): orquestra chamadas ao modelo + execução de tools
- TokenCounter: rastreia tokens (prompt/completion/total) por turno e sessão
- Signal handlers: SIGINT/SIGTERM para saída graciosa com auto-save
"""

import argparse
import ast
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, Optional

from mnemo import FerramentasVault, TokenCounter, PermissaoNegada
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA
from mnemo.memoria import (
    Retain, Recall, Reflect,
    NvidiaEmbeddings, IndiceVetorial,
    estimar_tokens,
)
from config import Config, DEFAULTS
from chat_ui import ChatUI
from rich.panel import Panel

# Máximo de iterações de tool-calling por turno do utilizador
# Previne loops infinitos se o modelo chamar ferramentas repetidamente
MAX_CICLOS_FERRAMENTAS = 15


def generate_instant_title(message: str) -> str | None:
    """Extrai título instantâneo da 1.ª mensagem do utilizador (máx 30 palavras)."""
    if not message or not message.strip():
        return None
    
    # 1.ª linha com conteúdo
    line = next((l.strip() for l in message.splitlines() if l.strip()), "")
    line = " ".join(line.split())
    
    # Trunca a 30 palavras
    words = line.split()
    if len(words) > 30:
        line = " ".join(words[:30])
    
    # Sanitiza para filename: remove / \ : * ? " < > | e substitui espaços por _
    line = re.sub(r'[\\/:*?"<>|]', "", line)
    line = line.replace(" ", "_")
    return line or None


INSTRUCAO_SISTEMA = (
    "És o assistente do Mnemo, uma plataforma pessoal de IA. Tens acesso a um "
    "conjunto de notas (o Vault) através das ferramentas search, read_note, "
    "create_note, append_to_note, list_files, tag_note, search_by_tag, list_tags e delete_note. "
    "Usa-as sempre que precisares de "
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
    "- `/new` — inicia nova conversa (limpa histórico atual)\n"
    "- `/delete <id>` — elimina uma sessão do histórico\n"
    "- Ao sair, o histórico é gravado automaticamente em historico/\n"
)


def carregar_env(caminho: str = ".env") -> None:
    """Lê pares CHAVE=VALOR de um .env simples, sem depender de bibliotecas externas.
    
    Usa caminho absoluto relativo ao script, remove BOM se houver, e sobrescreve
    variáveis de ambiente existentes (não usa setdefault).
    """
    if not Path(caminho).is_absolute():
        caminho = str(Path(__file__).parent / caminho)
    try:
        with open(caminho, encoding="utf-8-sig") as f:  # utf-8-sig remove BOM automaticamente
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                chave, valor = linha.split("=", 1)
                os.environ[chave.strip()] = valor.strip().strip("'\"")
    except FileNotFoundError:
        pass


def executar_ciclo_ferramentas(
    cliente: ClienteNVIDIA, vault: FerramentasVault, mensagens: List[Dict[str, Any]], 
    shutdown_check: Optional[Callable[[], bool]] = None, token_counter: Optional[TokenCounter] = None
) -> tuple[str, Dict[str, int]]:
    """Envia mensagens ao modelo e executa as ferramentas que ele pedir.
    
    Returns:
        tuple: (resposta_texto, usage_acumulado)
    """
    usage_acumulado: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for _ in range(MAX_CICLOS_FERRAMENTAS):
        # Verifica shutdown (Ctrl+C) antes de cada chamada ao modelo
        if shutdown_check and shutdown_check():
            raise KeyboardInterrupt("Shutdown requested")
        
        # Para chamadas de ferramentas, não usar streaming (precisamos do JSON completo)
        try:
            resposta_tuple = cliente.conversar(mensagens, ferramentas=FerramentasVault.DESCRICOES, stream=False)
        except KeyboardInterrupt:
            raise
        
        # resposta_tuple é (mensagem, usage)
        resposta, usage = resposta_tuple
        mensagens.append(resposta)
        
        # Acumula usage
        for k in usage_acumulado:
            usage_acumulado[k] += usage.get(k, 0)
        
        # Atualiza token_counter se fornecido
        if token_counter:
            token_counter.add(usage)
        
        pedidos = resposta.get("tool_calls")
        if not pedidos:
            # Resposta final sem tool calls
            return resposta.get("content") or "", usage_acumulado

        for pedido in pedidos:
            # Verifica shutdown durante execução de ferramentas
            if shutdown_check and shutdown_check():
                raise KeyboardInterrupt("Shutdown requested")
            
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
    return "(demasiados pedidos de ferramentas seguidos — parei para não entrar em ciclo)", usage_acumulado


def _salvar_historico(mensagens: List[Dict[str, Any]], raiz_vault: str, modelo: str = "", titulo: str | None = None) -> None:
    """Grava histórico completo (incl. tool calls) em historico/YYYY-MM-DD_HH-MM[_titulo].md
    
    Formato do ficheiro:
    - Cabeçalho com timestamp, modelo e título (se houver)
    - Mensagens do utilizador e assistente (ignora system)
    - Tool calls e resultados formatados em Markdown
    - Guardado como nota na pasta historico/ via FerramentasVault
    """
    from datetime import datetime
    from mnemo import FerramentasVault

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    sufixo = f"_{titulo}" if titulo else ""
    caminho = f"historico/{ts}{sufixo}.md"

    linhas = [f"# {titulo or f'Conversa {ts}'}\n"]
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
        resultado = v.executar("create_note", {"caminho": caminho, "conteudo": conteudo})
        if not resultado.get("ok"):
            raise RuntimeError(f"Falha ao gravar histórico: {resultado.get('erro', 'erro desconhecido')}")


def main() -> None:
    """Função principal: configura ambiente, valida, inicia loop de conversa."""
    parser = argparse.ArgumentParser(description="Conversa com o Nemotron ligado ao Vault do Mnemo.")
    parser.add_argument("--vault", default=None, help="Pasta raiz do vault (padrão: config ou vault-teste)")
    parser.add_argument("--modelo", default=None, help="ID do modelo NVIDIA")
    parser.add_argument("--theme", default=None, choices=["auto", "dark", "light"], help="Tema de cores")
    parser.add_argument("--save-config", action="store_true", help="Gravar vault/modelo/theme como padrão")
    args = parser.parse_args()

    carregar_env()

    # Configuração: carrega ~/.mnemo/config.json e aplica overrides da CLI
    config = Config.load()
    if args.vault:
        config.set("vault_default", args.vault)
    if args.modelo:
        config.set("model_default", args.modelo)
    if args.theme:
        config.set("theme", args.theme)
    if args.save_config:
        config.save()

    # UI criada ANTES da validação para poder mostrar erros no console
    config.set("theme", config.get("theme", "auto"))  # garante theme
    ui = ChatUI(config)

    vault_path = config.get("vault_default", "vault-teste")

    # Validação de configuração no startup (vault existe, API key, pastas permitidas)
    errors = config.validate(vault_path)
    if errors:
        ui.console.print("[error]Configuração inválida:[/error]")
        for e in errors:
            ui.console.print(f"  - {e}")
        sys.exit(1)

    # Health check rápido
    # Só passa modelo se foi explicitamente fornecido via --modelo; caso contrário,
    # ClienteNVIDIA lê NVIDIA_MODEL do .env (que usa o default do config se não definido)
    modelo_cli = args.modelo
    try:
        cliente = ClienteNVIDIA(modelo=modelo_cli) if modelo_cli else ClienteNVIDIA()
    except ErroModeloNVIDIA as e:
        print(f"Erro ao inicializar modelo: {e}", file=sys.stderr)
        sys.exit(1)

    # Flag para controlo de saída (Ctrl+C duplo força saída)
    shutdown_requested = False

    def _signal_handler(sig: int, frame: Optional[object]) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            # 2º sinal -> saída forçada sem gravar histórico
            ui.console.print("\n[error]Saída forçada — histórico NÃO gravado[/error]")
            sys.exit(1)
        shutdown_requested = True
        # O loop principal vai detectar a flag

    # Registra handlers (SIGINT=Ctrl+C, SIGTERM=kill)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    # Inicializa vault e UI
    with FerramentasVault(vault_path) as vault:
        # atualiza completer com pastas permitidas (path completion)
        ui.update_completer(vault_path)

        # Health check rápido com retry (3 tentativas, 1s intervalo)
        health_ok = False
        health_error: Optional[str] = None
        for _ in range(3):
            try:
                ok, err = cliente.health_check()
                if ok:
                    health_ok = True
                    break
                health_error = err
            except Exception as e:
                health_error = str(e)
            time.sleep(1)
        
        if not health_ok:
            msg = "Health check falhou — API indisponível."
            if health_error:
                msg += f" Detalhe: {health_error}"
            ui.print_error(msg)
            sys.exit(1)

        mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}]
        
        # Contador de tokens (reset a cada conversa)
        token_counter = TokenCounter()
        
        ui.welcome(vault.armazenamento.perms.raiz.name, cliente.modelo)

        # Inicializa subsistema de memória (Retain/Recall/Reflect)
        memoria_auto = config.get("memoria_auto", True)
        if memoria_auto:
            provider = NvidiaEmbeddings(
                modelo=config.get("NVIDIA_EMBED_MODEL"),
                dimensions=config.get("dim_embeddings"),
            )
            indice = IndiceVetorial(config.get("dim_embeddings"))
            # Carrega vetores existentes do índice
            rows = vault.indexador.con.execute(
                "SELECT hash, vetor FROM vetores WHERE modelo = ? AND dim = ?",
                (provider.modelo, provider.dim),
            ).fetchall()
            if rows:
                import numpy as np
                hashes = [r[0] for r in rows]
                vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                indice.carregar(hashes, vecs)

            retain = Retain(
                vault.armazenamento.perms,
                vault.armazenamento,
                vault.indexador,
                provider,
                provider.modelo,
                provider.dim,
                trecho_tokens_alvo=config.get("trecho_tokens"),
                chars_por_token=config.get("chars_por_token"),
            )
            recall = Recall(
                vault.indexador,
                vault.armazenamento.perms,
                provider,
                indice,
                pesos_rrf=tuple(map(float, config.get("pesos_rrf", "1.0,1.0").split(","))),
                candidatos_por_pesquisa=config.get("candidatos"),
                max_trechos_por_nota=config.get("max_trechos_por_nota"),
                chars_por_token=config.get("chars_por_token"),
            )
            reflect = Reflect()
            ui.console.print("[info]Memória ativa (Retain/Recall/Reflect)[/info]")
        else:
            retain = recall = reflect = None
            ui.console.print("[muted]Memória desativada (memoria_auto=false)[/muted]")

        # Título da conversa (extraído da 1.ª pergunta do utilizador)
        conversa_titulo = None
        primeira_msg = True

        while True:
            # Verifica se foi pedido shutdown (signal handler ou except)
            if shutdown_requested:
                if ui.confirm("\nSair e gravar histórico?"):
                    ui.console.print("\n[Saindo... a gravar histórico]")
                    _salvar_historico(mensagens, vault_path, cliente.modelo, conversa_titulo)
                    break
                else:
                    ui.console.print("[info]Continuando...[/info]")
                    shutdown_requested = False
                    continue

            try:
                texto = ui.prompt("Tu: ")
            except (EOFError, KeyboardInterrupt):
                if shutdown_requested:
                    # 2º Ctrl+C durante prompt -> força saída
                    ui.console.print("\n[error]Saída forçada — histórico NÃO gravado[/error]")
                    sys.exit(1)
                shutdown_requested = True
                # Aviso imediato: 1º Ctrl+C
                ui.console.print("\n[warning]Pressione Ctrl+C novamente para sair SEM gravar histórico.[/warning]")
                continue

            if not texto:
                continue

            # Recall automático (memória) antes de processar a mensagem
            contexto_memoria = None
            if memoria_auto and recall and reflect:
                budget = config.get("budget_tokens", 2000)
                resultados = recall.recall(texto, k=8, budget_tokens=budget)
                if resultados:
                    contexto_memoria = reflect.construir_contexto(texto, resultados, budget)
                    # Injeta contexto no system message temporariamente
                    mensagens[0] = {"role": "system", "content": INSTRUCAO_SISTEMA + "\n\n" + reflect.formatar_para_prompt(contexto_memoria)}

            low = texto.lower()
            if low in {"sair", "exit", "quit"}:
                ui.console.print("[Saindo... a gravar histórico]")
                _salvar_historico(mensagens, vault_path, cliente.modelo, conversa_titulo)
                break

            if texto == "/salvar":
                _salvar_historico(mensagens, vault_path, cliente.modelo, conversa_titulo)
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
                    elif parts[i] in ("search", "load", "export", "import", "limpar"):
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
                elif subcommand == "limpar":
                    if not ui.confirm("Apagar TODO o histórico? Esta ação não pode ser desfeita."):
                        continue
                    try:
                        with FerramentasVault(vault_path) as ferramentas:
                            resultado = ferramentas.executar("list_files", {"pasta": "historico"})
                            if resultado["ok"] and resultado["notas"]:
                                for nota in resultado["notas"]:
                                    ferramentas.executar("delete_note", {"caminho": nota["caminho"]})
                                ui.console.print(f"[success]Histórico limpo: {len(resultado['notas'])} sessões eliminadas.[/success]")
                            else:
                                ui.console.print("[muted]Histórico já está vazio.[/muted]")
                    except Exception as e:
                        ui.console.print(f"[error]Erro ao limpar histórico: {e}[/error]")
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

            if texto.startswith("/tag"):
                parts = shlex.split(texto)
                if len(parts) < 3:
                    ui.console.print("[warning]Uso:[/warning] /tag <caminho> add|remove|set <tag1> [tag2] ...")
                    continue
                caminho = parts[1]
                acao = parts[2]
                tags = parts[3:] if len(parts) > 3 else []
                if acao not in ("add", "remove", "set"):
                    ui.console.print("[warning]Ação inválida:[/warning] use add, remove ou set")
                    continue
                try:
                    with FerramentasVault(vault_path) as ferramentas:
                        if acao == "add":
                            for tag in tags:
                                novas = ferramentas.armazenamento.adicionar_tag(caminho, tag)
                            ui.console.print(f"[success]Tags adicionadas:[/success] {', '.join(novas)}")
                        elif acao == "remove":
                            for tag in tags:
                                novas = ferramentas.armazenamento.remover_tag(caminho, tag)
                            ui.console.print(f"[success]Tags removidas:[/success] {', '.join(novas)}")
                        elif acao == "set":
                            novas = ferramentas.armazenamento.substituir_tags(caminho, tags)
                            ui.console.print(f"[success]Tags definidas:[/success] {', '.join(novas)}")
                except FileNotFoundError:
                    ui.console.print("[error]Nota não encontrada.[/error]")
                except PermissaoNegada as e:
                    ui.console.print(f"[error]{e}[/error]")
                except Exception as e:
                    ui.console.print(f"[error]Erro ao manipular tags: {e}[/error]")
                continue

            if texto.startswith("/tags"):
                parts = shlex.split(texto)
                pasta = None
                prefixo = None
                i = 1
                while i < len(parts):
                    if parts[i] == "--pasta" and i + 1 < len(parts):
                        pasta = parts[i + 1]
                        i += 2
                    elif parts[i] == "--prefixo" and i + 1 < len(parts):
                        prefixo = parts[i + 1]
                        i += 2
                    else:
                        i += 1
                try:
                    with FerramentasVault(vault_path) as ferramentas:
                        resultado = ferramentas.executar("list_tags", {"pasta": pasta, "prefixo": prefixo})
                        if resultado["ok"]:
                            if resultado["tags"]:
                                ui.console.print(f"[info]Tags ({resultado['total']} únicas):[/info]")
                                for t in resultado["tags"]:
                                    ui.console.print(f"  {t['tag']} ({t['count']})")
                            else:
                                ui.console.print("[muted]Nenhuma tag encontrada.[/muted]")
                        else:
                            ui.console.print(f"[error]{resultado['erro']}[/error]")
                except Exception as e:
                    ui.console.print(f"[error]Erro ao listar tags: {e}[/error]")
                continue

            if texto == "/limpar":
                ui.clear()
                continue

            if texto == "/new":
                # Nova conversa: mantém system message, limpa o resto
                mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}]
                token_counter = TokenCounter()
                conversa_titulo = None
                primeira_msg = True
                ui.console.print("[success]Nova conversa iniciada.[/success]\n")
                continue

            if texto.startswith("/delete"):
                parts = shlex.split(texto)
                if len(parts) < 2:
                    ui.console.print("[warning]Uso:[/warning] /delete <id_da_sessao>")
                    ui.console.print("[muted]Use /historico para ver IDs disponíveis.[/muted]")
                    continue
                session_id = parts[1]
                try:
                    with FerramentasVault(vault_path) as ferramentas:
                        # Tenta apagar o ficheiro de histórico
                        resultado = ferramentas.executar("delete_note", {"caminho": f"historico/{session_id}.md"})
                        if resultado["ok"]:
                            ui.console.print(f"[success]Sessão {session_id} eliminada.[/success]")
                        else:
                            ui.console.print(f"[error]{resultado['erro']}[/error]")
                except Exception as e:
                    ui.console.print(f"[error]Erro ao eliminar sessão: {e}[/error]")
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
                    "TAGS\n"
                    "  /tag <caminho> add <tag1> [tag2]  # Adiciona tag(s)\n"
                    "  /tag <caminho> remove <tag>       # Remove tag\n"
                    "  /tag <caminho> set <tag1> [tag2]  # Substitui todas\n"
                    "  /tags                           # Lista todas as tags\n"
                    "  /tags --pasta projetos           # Filtra por pasta\n"
                    "  /tags --prefixo trab             # Autocomplete por prefixo\n\n"
                    "MODELO\n"
                    "  /modelo nvidia/nemotron-3-ultra  # Troca modelo (sessão)\n\n"
                    "NOTAS (usam autocomplete Tab)\n"
                    "  create_note projetos/plano.md     # Cria nota\n"
                    "  read_note projetos/plano.md       # Lê nota\n"
                    "  append_to_note projetos/plano.md  # Anexa conteúdo\n"
                    "  search python                     # Busca semântica\n"
                    "  list_files projetos/              # Lista pasta\n\n"
                    "CONVERSA\n"
                    "  /new                  # Nova conversa (limpa histórico atual)\n"
                    "  /delete <id>          # Elimina sessão do histórico\n"
                    "  /historico limpar     # Apaga TODO o histórico (confirmação)\n\n"
                    "OUTROS\n"
                    "  /salvar              # Checkpoint em historico/\n"
                    "  /limpar              # Limpa ecrã\n"
                    "  sair / exit / quit   # Sai (auto-save)\n"
                    "  Ctrl+C               # Pergunta confirmação (2x = força saída)",
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
                        ui.set_config("timeout", parts[2])
                    except ValueError:
                        ui.print_error("Timeout deve ser um número inteiro (segundos)")
                elif parts[1] == "fallback" and len(parts) >= 3:
                    models = [m.strip() for m in " ".join(parts[2:]).split(",")]
                    ui.set_config("fallback_models", ",".join(models))
                elif parts[1] == "tokens" and len(parts) == 3:
                    if parts[2].lower() in ("on", "true", "1", "yes"):
                        ui.set_config("show_tokens", "true")
                        ui.console.print("[success]Contador de tokens ativado.[/success]")
                    elif parts[2].lower() in ("off", "false", "0", "no"):
                        ui.set_config("show_tokens", "false")
                        ui.console.print("[success]Contador de tokens desativado.[/success]")
                    else:
                        ui.print_error("Uso: /config tokens on|off")
                elif parts[1] == "reset":
                    ui.config.data = {**DEFAULTS}
                    ui.config.save()
                    ui.console.print("[success]Config resetado para padrões.[/success]")
                else:
                    ui.console.print("[warning]Uso:[/warning] /config  |  /config theme dark|light|auto  |  /config model <id>  |  /config vault <path>  |  /config timeout <seg>  |  /config fallback <model1,model2,...>  |  /config tokens on|off  |  /config reset")
                continue

            if texto.startswith("/memoria"):
                if not memoria_auto:
                    ui.console.print("[muted]Memória desativada (memoria_auto=false no config)[/muted]")
                    continue
                parts = shlex.split(texto)
                if len(parts) == 1:
                    # Status da memória
                    ui.console.print("[info]Memória:[/info]")
                    ui.console.print(f"  Modelo: {provider.modelo}")
                    ui.console.print(f"  Dimensão: {provider.dim}")
                    ui.console.print(f"  Vetores indexados: {indice.n}")
                    ui.console.print(f"  Budget tokens: {config.get('budget_tokens')}")
                    ui.console.print(f"  RRF pesos: {config.get('pesos_rrf')}")
                    continue
                sub = parts[1]
                if sub == "reindex":
                    ui.console.print("[info]Reindexando vault...[/info]")
                    rel = retain.retain_vault()
                    ui.console.print(f"  Novas: {rel.novas}, Atualizadas: {rel.atualizadas}, Inalteradas: {rel.inalteradas}, Removidas: {rel.removidas}")
                    ui.console.print(f"  Trechos: {rel.trechos_novos}, Embeddings: {rel.embeddings_novos}")
                    if rel.embeddings_pendentes:
                        ui.console.print(f"  [warning]Embeddings pendentes: {len(rel.embeddings_pendentes)}[/warning]")
                    # Recarrega vetores no índice
                    rows = vault.indexador.con.execute(
                        "SELECT hash, vetor FROM vetores WHERE modelo = ? AND dim = ?",
                        (provider.modelo, provider.dim),
                    ).fetchall()
                    if rows:
                        import numpy as np
                        hashes = [r[0] for r in rows]
                        vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                        indice.carregar(hashes, vecs)
                    continue
                elif sub == "reparar":
                    ui.console.print("[info]Reparando embeddings pendentes...[/info]")
                    rel = retain.retain_vault(reparar=True)
                    ui.console.print(f"  Embeddings recuperados: {rel.embeddings_novos}")
                    if rel.embeddings_pendentes:
                        ui.console.print(f"  [warning]Ainda pendentes: {len(rel.embeddings_pendentes)}[/warning]")
                    # Recarrega vetores
                    rows = vault.indexador.con.execute(
                        "SELECT hash, vetor FROM vetores WHERE modelo = ? AND dim = ?",
                        (provider.modelo, provider.dim),
                    ).fetchall()
                    if rows:
                        import numpy as np
                        hashes = [r[0] for r in rows]
                        vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                        indice.carregar(hashes, vecs)
                    continue
                elif sub == "stats":
                    # Estatísticas do índice
                    total_notas = vault.indexador.con.execute("SELECT COUNT(*) FROM notas").fetchone()[0]
                    total_trechos = vault.indexador.con.execute("SELECT COUNT(*) FROM trechos").fetchone()[0]
                    total_vetores = vault.indexador.con.execute("SELECT COUNT(*) FROM vetores WHERE modelo = ? AND dim = ?", (provider.modelo, provider.dim)).fetchone()[0]
                    ui.console.print(f"  Notas: {total_notas}")
                    ui.console.print(f"  Trechos: {total_trechos}")
                    ui.console.print(f"  Vetores ({provider.modelo}/{provider.dim}): {total_vetores}")
                    continue
                else:
                    ui.console.print("[warning]Uso:[/warning] /memoria  |  /memoria reindex  |  /memoria reparar  |  /memoria stats")
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
                    "  /config tokens on|off           Ativa/desativa contador de tokens\n"
                    "  /config reset                   Reseta configuração\n"
                    "  /ajuda           Mostra esta ajuda\n"
                    "  sair / exit / quit   Termina a conversa",
                    title="Ajuda", border_style="info"))
                continue

            # Mensagem normal do utilizador -> adiciona ao histórico e processa
            mensagens.append({"role": "user", "content": texto})
            
            # Captura título da 1.ª mensagem (se não for comando)
            if primeira_msg:
                conversa_titulo = generate_instant_title(texto)
                primeira_msg = False
            
            try:
                with ui.thinking():
                    resposta_texto, usage = executar_ciclo_ferramentas(
                        cliente, vault, mensagens, 
                        shutdown_check=lambda: shutdown_requested,
                        token_counter=token_counter
                    )
                if resposta_texto.startswith("(demasiados pedidos"):
                    ui.print_error(resposta_texto)
                    mensagens.pop()
                    continue
                ui.print_response(resposta_texto)
                # Mostra contador de tokens se habilitado
                if config.get("show_tokens"):
                    ui.print_token_usage(
                        token_counter.get_turn_summary(),
                        token_counter.get_session_summary()
                    )
                    token_counter.reset_turn()
                mensagens.append({"role": "assistant", "content": resposta_texto})
            
            # Restaura system message original (remove contexto de memória injetado)
            if memoria_auto and contexto_memoria:
                mensagens[0] = {"role": "system", "content": INSTRUCAO_SISTEMA}
                contexto_memoria = None

        except KeyboardInterrupt:
            # Ctrl+C durante tool calling
            if shutdown_requested:
                ui.console.print("\n[error]Saída forçada — histórico NÃO gravado[/error]")
                sys.exit(1)
            shutdown_requested = True
            # Aviso imediato: 1º Ctrl+C
            ui.console.print("\n[warning]Pressione Ctrl+C novamente para sair SEM gravar histórico.[/warning]")
            # O loop principal vai tratar na próxima iteração
            continue
        except ErroModeloNVIDIA as e:
            ui.print_error(str(e))
            mensagens.pop()
            continue


if __name__ == "__main__":
    main()