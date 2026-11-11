#!/usr/bin/env python3
"""
Teste do endpoint de embeddings da NVIDIA.
Verifica: modelo disponível, suporte a `dimensions`, latência, qualidade básica.
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional

# Carrega .env se existir
def carregar_env():
    try:
        with open(".env", encoding="utf-8-sig") as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                chave, valor = linha.split("=", 1)
                os.environ[chave.strip()] = valor.strip().strip("'\"")
    except FileNotFoundError:
        pass

carregar_env()

CHAVE_API = os.environ.get("NVIDIA_API_KEY")
if not CHAVE_API:
    print("ERRO: NVIDIA_API_KEY não definida")
    exit(1)

URL_BASE = "https://integrate.api.nvidia.com/v1/embeddings"
MODELO = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")

def fazer_pedido(textos: list[str], dimensions: Optional[int] = None) -> dict:
    corpo = {"model": MODELO, "input": textos, "input_type": "passage", "truncate": "END"}
    if dimensions is not None:
        corpo["dimensions"] = dimensions

    dados = json.dumps(corpo).encode("utf-8")
    pedido = urllib.request.Request(
        URL_BASE,
        data=dados,
        headers={
            "Authorization": f"Bearer {CHAVE_API}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(pedido, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def testar_modelo():
    print(f"\n=== Teste de Embeddings: {MODELO} ===\n")

    # 1. Teste basico sem dimensions
    print("1. Pedido basico (sem dimensions)...")
    inicio = time.time()
    try:
        resp = fazer_pedido(["Ola mundo", "Teste de embeddings em portugues"])
        elapsed = time.time() - inicio
        embeddings = resp.get("data", [])
        dim = len(embeddings[0]["embedding"]) if embeddings else 0
        print(f"   OK Sucesso em {elapsed:.2f}s")
        print(f"   Dimensao retornada: {dim}")
        print(f"   Uso: {resp.get('usage', {})}")
    except urllib.error.HTTPError as e:
        print(f"   ERRO HTTP {e.code}: {e.read().decode()}")
        return
    except Exception as e:
        print(f"   ERRO: {e}")
        return

    # 2. Teste com dimensions=512
    print("\n2. Pedido com dimensions=512...")
    inicio = time.time()
    try:
        resp = fazer_pedido(["Teste com dimensions"], dimensions=512)
        elapsed = time.time() - inicio
        embeddings = resp.get("data", [])
        dim = len(embeddings[0]["embedding"]) if embeddings else 0
        print(f"   OK Sucesso em {elapsed:.2f}s")
        print(f"   Dimensao retornada: {dim}")
    except urllib.error.HTTPError as e:
        print(f"   ERRO HTTP {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"   ERRO: {e}")

    # 3. Teste com dimensions=256
    print("\n3. Pedido com dimensions=256...")
    inicio = time.time()
    try:
        resp = fazer_pedido(["Teste com dimensions 256"], dimensions=256)
        elapsed = time.time() - inicio
        embeddings = resp.get("data", [])
        dim = len(embeddings[0]["embedding"]) if embeddings else 0
        print(f"   OK Sucesso em {elapsed:.2f}s")
        print(f"   Dimensao retornada: {dim}")
    except urllib.error.HTTPError as e:
        print(f"   ERRO HTTP {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"   ERRO: {e}")

    # 4. Teste de similaridade (cosseno) entre frases semelhantes vs diferentes
    print("\n4. Teste de similaridade semantica...")
    try:
        textos = [
            "O Python e uma linguagem de programacao",
            "Python e usado para programar",
            "O tempo esta ensolarado hoje",
            "A temperatura hoje e alta",
        ]
        resp = fazer_pedido(textos)
        embeddings = [e["embedding"] for e in resp["data"]]

        import numpy as np
        vecs = np.array(embeddings, dtype=np.float32)
        # Normaliza
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)

        sim_12 = np.dot(vecs[0], vecs[1])  # semelhantes
        sim_34 = np.dot(vecs[2], vecs[3])  # semelhantes
        sim_13 = np.dot(vecs[0], vecs[2])  # diferentes

        print(f"   Similaridade 'programacao' vs 'programar': {sim_12:.3f}")
        print(f"   Similaridade 'ensolarado' vs 'temperatura': {sim_34:.3f}")
        print(f"   Similaridade 'programacao' vs 'tempo': {sim_13:.3f}")
        print(f"   Discriminacao (deve ser > 0): {min(sim_12, sim_34) - sim_13:.3f}")
    except Exception as e:
        print(f"   ERRO no teste de similaridade: {e}")

    # 5. Listar modelos disponíveis
    print("\n5. Modelos disponiveis na conta (GET /v1/models)...")
    try:
        pedido = urllib.request.Request(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {CHAVE_API}"},
        )
        with urllib.request.urlopen(pedido, timeout=15) as resp:
            modelos = json.loads(resp.read().decode())
            embed_models = [m for m in modelos.get("data", []) if "embed" in m.get("id", "").lower()]
            for m in embed_models:
                print(f"   - {m['id']}")
            if not embed_models:
                print("   (nenhum modelo com 'embed' no nome encontrado)")
    except Exception as e:
        print(f"   ERRO ao listar modelos: {e}")

    print("\n=== Fim do teste ===")

if __name__ == "__main__":
    testar_modelo()