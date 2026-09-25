#!/usr/bin/env python3
"""CLI de conversa com o Nemotron (API da NVIDIA), ligado ao Vault do Mnemo.

Uso:
    export NVIDIA_API_KEY=nvapi-...      # ou põe isto num ficheiro .env
    python chat.py --vault vault-teste

Fica à espera de perguntas na consola. O modelo pode pedir para usar as
ferramentas do vault (search, read_note, create_note, append_to_note); este
programa executa-as e devolve o resultado ao modelo, até ele dar uma resposta
em texto.
"""

import argparse
import json
import os
import sys

from mnemo import FerramentasVault
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA

MAX_CICLOS_FERRAMENTAS = 8  # trava de segurança contra um ciclo sem fim

INSTRUCAO_SISTEMA = (
    "És o assistente do Mnemo, uma plataforma pessoal de IA. Tens acesso a um "
    "conjunto de notas (o Vault) através das ferramentas search, read_note, "
    "create_note e append_to_note. Usa-as sempre que precisares de consultar ou "
    "guardar informação nas notas do utilizador — nunca inventes o conteúdo de "
    "uma nota que não leste. Responde sempre em português."
)


def carregar_env(caminho=".env"):
    """Lê pares CHAVE=VALOR de um .env simples, sem depender de bibliotecas
    externas. Não sobrescreve variáveis já definidas no ambiente."""
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


def executar_ciclo_ferramentas(cliente, vault, mensagens):
    """Envia `mensagens` ao modelo e executa as ferramentas que ele pedir, até
    obter uma resposta em texto (ou atingir o limite de ciclos). `mensagens`
    é alterada no próprio local, com as respostas do modelo e das ferramentas,
    para o histórico da conversa ficar completo."""
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


def main():
    parser = argparse.ArgumentParser(description="Conversa com o Nemotron ligado ao Vault do Mnemo.")
    parser.add_argument(
        "--vault", default="vault-teste", help="Pasta raiz do vault (padrão: vault-teste)"
    )
    args = parser.parse_args()

    carregar_env()

    try:
        cliente = ClienteNVIDIA()
    except ErroModeloNVIDIA as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    with FerramentasVault(args.vault) as vault:
        mensagens = [{"role": "system", "content": INSTRUCAO_SISTEMA}]
        print(f"Mnemo — ligado a '{args.vault}'. Escreve 'sair' para terminar.\n")
        while True:
            try:
                texto = input("Tu: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if texto.lower() in {"sair", "exit", "quit"}:
                break
            if not texto:
                continue

            mensagens.append({"role": "user", "content": texto})
            try:
                resposta = executar_ciclo_ferramentas(cliente, vault, mensagens)
            except ErroModeloNVIDIA as e:
                print(f"Erro: {e}\n")
                mensagens.pop()  # não guardar a pergunta se a chamada falhou
                continue
            print(f"Mnemo: {resposta}\n")


if __name__ == "__main__":
    main()
