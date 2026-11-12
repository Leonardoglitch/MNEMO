"""Provider de embeddings (NVIDIA + fake para testes)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, List, Optional, Protocol, cast

import numpy as np

from mnemo.modelo_nvidia import ErroModeloNVIDIA


class EmbeddingProvider(Protocol):
    """Protocolo para providers de embeddings."""

    def embed(self, textos: List[str], input_type: str = "passage", dimensions: Optional[int] = None) -> np.ndarray:
        """Retorna matriz (n_textos, dim) float32 normalizada (L2=1)."""
        ...

    @property
    def modelo(self) -> str:
        ...

    @property
    def dim(self) -> int:
        ...


class NvidiaEmbeddings:
    """Cliente para /v1/embeddings da NVIDIA (compatível OpenAI)."""

    def __init__(
        self,
        chave_api: Optional[str] = None,
        modelo: Optional[str] = None,
        url_base: str = "https://integrate.api.nvidia.com/v1/embeddings",
        timeout: int = 30,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> None:
        self.chave_api = chave_api or os.environ.get("NVIDIA_API_KEY")
        if not self.chave_api:
            raise ErroModeloNVIDIA("Falta NVIDIA_API_KEY")
        self.modelo = modelo or os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
        self.url_base = url_base
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._dim_cache: Optional[int] = None

    def _pedido(self, textos: List[str], input_type: str, dimensions: Optional[int]) -> dict[str, Any]:
        corpo: dict[str, Any] = {
            "model": self.modelo,
            "input": textos,
            "input_type": input_type,
            "truncate": "END",
        }
        if dimensions is not None:
            corpo["dimensions"] = dimensions

        dados = json.dumps(corpo).encode("utf-8")
        pedido = urllib.request.Request(
            self.url_base,
            data=dados,
            headers={
                "Authorization": f"Bearer {self.chave_api}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(pedido, timeout=self.timeout) as resp:
            return cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))

    def _deve_tentar_novamente(self, erro: Exception) -> bool:
        if isinstance(erro, urllib.error.HTTPError):
            # 404 = modelo inexistente -> não tenta (fallback no chamador)
            if erro.code == 404:
                return False
            return erro.code >= 500 or erro.code == 429
        if isinstance(erro, (urllib.error.URLError, TimeoutError)):
            return True
        return False

    def embed(
        self,
        textos: List[str],
        input_type: str = "passage",
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """Embede lista de textos. Retorna matriz (n, dim) float32 normalizada."""
        if not textos:
            return np.empty((0, self.dim), dtype=np.float32)

        # Processa em lotes de 32
        LOTE = 32
        todos_vecs: List[np.ndarray] = []

        for i in range(0, len(textos), LOTE):
            lote = textos[i:i + LOTE]

            for attempt in range(self.max_retries):
                try:
                    resp = self._pedido(lote, input_type, dimensions)
                    embeddings = [e["embedding"] for e in resp["data"]]
                    vecs = np.array(embeddings, dtype=np.float32)
                    # Normaliza L2
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vecs = vecs / (norms + 1e-8)
                    todos_vecs.append(vecs)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                    if not self._deve_tentar_novamente(e):
                        raise ErroModeloNVIDIA(f"Erro embeddings não recuperável: {e}") from e
                    if attempt < self.max_retries - 1:
                        time.sleep(self.base_delay * (2 ** attempt))
                        continue
                    raise ErroModeloNVIDIA(f"Falha embeddings após {self.max_retries} tentativas") from e

        if not todos_vecs:
            return np.empty((0, self.dim), dtype=np.float32)

        resultado = np.vstack(todos_vecs)
        self._dim_cache = resultado.shape[1]
        return resultado

    @property
    def dim(self) -> int:
        if self._dim_cache is not None:
            return self._dim_cache
        # Descobre dimensão com uma chamada rápida
        vecs = self.embed(["teste"], input_type="passage")
        return int(vecs.shape[1])


class FakeEmbeddings:
    """Embeddings determinísticos para testes (derivados do hash do texto)."""

    def __init__(self, dim: int = 512, seed: int = 42) -> None:
        self._modelo = "fake-embeddings"
        self._dim = dim
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    def embed(
        self,
        textos: List[str],
        input_type: str = "passage",
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        if not textos:
            return np.empty((0, self._dim), dtype=np.float32)

        vecs = []
        for t in textos:
            if t in self._cache:
                vecs.append(self._cache[t])
            else:
                # Hash determinístico -> vetor
                h = hash(t) % (2**32)
                rng = np.random.default_rng(h)
                v = rng.normal(size=self._dim).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                self._cache[t] = v
                vecs.append(v)

        resultado = np.array(vecs, dtype=np.float32)
        if dimensions is not None and dimensions < self._dim:
            resultado = resultado[:, :dimensions]
        return resultado

    @property
    def modelo(self) -> str:
        return self._modelo

    @property
    def dim(self) -> int:
        return self._dim


# Cache LRU simples para embeddings de consulta
class ConsultaCache:
    """Cache LRU em memória para embeddings de consultas (input_type='query')."""

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.maxsize = maxsize

    def get(self, chave: str) -> Optional[np.ndarray]:
        if chave in self._cache:
            self._cache.move_to_end(chave)
            return self._cache[chave]
        return None

    def set(self, chave: str, valor: np.ndarray) -> None:
        if chave in self._cache:
            self._cache.move_to_end(chave)
        self._cache[chave] = valor
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()