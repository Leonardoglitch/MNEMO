"""Índice vetorial em memória (numpy) com similaridade de cosseno via produto interno."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mnemo.memoria.embeddings import EmbeddingProvider


class IndiceVetorial:
    """
    Índice vetorial simples: matriz (n_trechos, dim) em memória.
    Similaridade = produto interno (vetores já normalizados L2=1).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._matriz: Optional[np.ndarray] = None  # (n, dim) float32
        self._hashes: List[str] = []  # hash de cada linha (ordem = rowid em trechos)
        self._versao_vetores: int = 0

    def carregar(self, hashes: List[str], vetores: np.ndarray) -> None:
        """Carrega vetores do SQLite (ordem = trechos.id)."""
        if len(hashes) != vetores.shape[0]:
            raise ValueError(f"Hashes ({len(hashes)}) != linhas vetores ({vetores.shape[0]})")
        if vetores.shape[1] != self.dim:
            raise ValueError(f"Dim vetores ({vetores.shape[1]}) != índice ({self.dim})")
        self._hashes = list(hashes)
        self._matriz = vetores.astype(np.float32, copy=False)
        self._versao_vetores += 1

    def invalidar(self) -> None:
        """Invalida o índice (força recarregar)."""
        self._matriz = None
        self._hashes = []
        self._versao_vetores += 1

    @property
    def pronto(self) -> bool:
        return self._matriz is not None and self._matriz.shape[0] > 0

    @property
    def n(self) -> int:
        return self._matriz.shape[0] if self._matriz is not None else 0

    @property
    def versao(self) -> int:
        return self._versao_vetores

    def top_k(
        self,
        vetor_consulta: np.ndarray,
        k: int,
        *,
        indices_permitidos: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Retorna top-k (idx_local, score) onde idx_local é índice em self._hashes.
        Se indices_permitidos dado, só considera esses índices.
        """
        if not self.pronto or self._matriz is None:
            return []

        matriz = self._matriz
        if indices_permitidos is not None:
            matriz = matriz[indices_permitidos]

        # Produto interno (cosseno, já normalizados)
        scores = matriz @ vetor_consulta.astype(np.float32)

        # Top-k
        k = min(k, len(scores))
        if k <= 0:
            return []

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_idx_list = top_idx.tolist()

        if indices_permitidos is not None:
            # Mapeia de volta para índices globais
            top_idx_list = [indices_permitidos[i] for i in top_idx_list]

        return [(int(i), float(scores[i])) for i in top_idx_list]