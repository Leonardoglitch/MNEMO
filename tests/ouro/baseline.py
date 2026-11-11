#!/usr/bin/env python3
"""
Script de baseline para F0: mede Recall@5 e MRR do BM25 por nota (antes de trechos).
Corre sobre o vault de teste e imprime métricas.
"""

import json
import sys
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemo import Permissoes, Indexador


def carregar_perguntas(caminho: str) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(resultados: list[dict], esperadas: list[str], k: int = 5) -> float:
    """Retorna 1 se pelo menos uma esperada está no top-k, senão 0."""
    top_k = [r["caminho"] for r in resultados[:k]]
    return 1.0 if any(e in top_k for e in esperadas) else 0.0


def mrr(resultados: list[dict], esperadas: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank da primeira esperada encontrada."""
    for i, r in enumerate(resultados, 1):
        if r["caminho"] in esperadas:
            return 1.0 / i
    return 0.0


def main():
    vault_path = "vault-teste"
    perguntas_path = "tests/ouro/perguntas.json"

    print(f"=== Baseline BM25 (por nota) ===")
    print(f"Vault: {vault_path}")
    print(f"Perguntas: {perguntas_path}\n")

    # Inicializa
    perms = Permissoes(vault_path)
    indexador = Indexador(perms)
    indexador.reindexar_tudo()

    perguntas = carregar_perguntas(perguntas_path)

    total = len(perguntas)
    recall5_sum = 0.0
    mrr_sum = 0.0
    por_tipo = {"termo_exato": {"n": 0, "r5": 0.0, "mrr": 0.0},
                "parafrase": {"n": 0, "r5": 0.0, "mrr": 0.0}}

    for p in perguntas:
        consulta = p["pergunta"]
        esperadas = p["esperadas"]
        tipo = p.get("tipo", "parafrase")

        resultados = indexador.pesquisar(consulta, limite=10)

        r5 = recall_at_k(resultados, esperadas, 5)
        m = mrr(resultados, esperadas)

        recall5_sum += r5
        mrr_sum += m
        por_tipo[tipo]["n"] += 1
        por_tipo[tipo]["r5"] += r5
        por_tipo[tipo]["mrr"] += m

        status = "OK" if r5 else "FALHA"
        print(f"  [{status}] {consulta[:60]}... (R@5={r5:.0f}, MRR={m:.3f})")

    print(f"\n=== Resultados ({total} perguntas) ===")
    print(f"Recall@5: {recall5_sum/total:.3f}")
    print(f"MRR:      {mrr_sum/total:.3f}")

    print("\n--- Por tipo ---")
    for tipo, stats in por_tipo.items():
        if stats["n"] > 0:
            print(f"  {tipo} ({stats['n']}): R@5={stats['r5']/stats['n']:.3f}, MRR={stats['mrr']/stats['n']:.3f}")

    indexador.fechar()


if __name__ == "__main__":
    main()