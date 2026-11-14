"""API FastAPI para o subsistema de memória do Mnemo (Retain/Recall/Reflect).

Endpoints:
- GET  /memoria/recall           - Busca híbrida (RRF)
- POST /memoria/retain/vault     - Sincroniza vault (retain_vault)
- POST /memoria/retain/conversa  - Extrai memórias de conversa (retain_conversa)
- POST /memoria/refletir         - Consolida observações/opiniões (refletir_conversa)
- GET  /memoria/stats            - Estatísticas do índice
- POST /memoria/limpar           - Limpa dados da memória
- GET  /health

Desenho: os componentes vivem num objeto `Servicos`, construído por `construir_servicos()`
e injetado em `criar_app()`. Em produção o `lifespan` constrói-o a partir do ambiente
(.env, MNEMO_VAULT); nos testes injeta-se um `Servicos` com FakeEmbeddings, LLM falso e
base de dados em memória, sem tocar no disco nem na rede.

Arranque:  uvicorn api_memoria:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from mnemo.armazenamento import Armazenamento
from mnemo.indexador import Indexador
from mnemo.memoria import NvidiaEmbeddings, Recall, Reflect, Retain
from mnemo.memoria.esquema import incrementar_versao_vetores
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA
from mnemo.permissoes import Permissoes


# --- Modelos de pedido ---


class RetainConversaRequest(BaseModel):
    mensagens: List[Dict[str, Any]]


class ReflectRequest(BaseModel):
    mensagens: List[Dict[str, Any]]


# --- Serviços ---


@dataclass
class Servicos:
    """Componentes partilhados pelos endpoints (uma instância por aplicação)."""

    perms: Permissoes
    indexador: Indexador
    armazenamento: Armazenamento
    provider: Any
    retain: Retain
    recall: Recall
    reflect: Reflect
    cliente_llm: Optional[Any] = None
    # A ligação SQLite é partilhada entre threads (check_same_thread=False), por isso todo o
    # acesso passa por este lock.
    lock: Any = field(default_factory=threading.RLock)

    def fechar(self) -> None:
        self.indexador.fechar()


def construir_servicos(
    vault: str,
    *,
    provider: Any = None,
    cliente_llm: Any = None,
    caminho_db: Optional[str] = None,
) -> Servicos:
    """Constrói todos os componentes sobre UM Indexador (uma só ligação) e UM índice vetorial.

    provider=None usa a NvidiaEmbeddings real; nos testes passa-se um FakeEmbeddings.
    caminho_db=":memory:" dá uma base em memória, partilhada por Retain e Recall.
    """
    perms = Permissoes(vault)
    indexador = Indexador(perms, caminho_db=caminho_db, check_same_thread=False)
    indexador.reindexar_tudo()
    armazenamento = Armazenamento(perms)
    provider = provider if provider is not None else NvidiaEmbeddings()

    retain = Retain(perms, armazenamento, indexador, provider, provider.modelo, provider.dim)
    # O Recall usa o MESMO índice vetorial do Retain, para ver os vetores novos.
    recall = Recall(indexador, perms, provider, retain._indice_vetorial)
    return Servicos(
        perms=perms,
        indexador=indexador,
        armazenamento=armazenamento,
        provider=provider,
        retain=retain,
        recall=recall,
        reflect=Reflect(),
        cliente_llm=cliente_llm,
    )


def _carregar_env(caminho: str = ".env") -> None:
    p = Path(caminho)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    try:
        with open(p, encoding="utf-8-sig") as f:
            for linha in f:
                linha = linha.strip()
                if linha.startswith("export "):
                    linha = linha[7:]
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                chave, valor = linha.split("=", 1)
                os.environ[chave.strip()] = valor.strip().strip("'\"")
    except FileNotFoundError:
        pass


def servicos_do_ambiente() -> Servicos:
    """Configuração de produção: .env, MNEMO_VAULT, API da NVIDIA."""
    _carregar_env()
    vault = os.environ.get("MNEMO_VAULT", "vault-teste")
    try:
        cliente_llm: Optional[ClienteNVIDIA] = ClienteNVIDIA()
    except ErroModeloNVIDIA:
        cliente_llm = None
    return construir_servicos(vault, cliente_llm=cliente_llm)


def obter_servicos(request: Request) -> Servicos:
    servicos: Optional[Servicos] = getattr(request.app.state, "servicos", None)
    if servicos is None:
        raise HTTPException(503, "Memória não inicializada")
    return servicos


def _proposta_dict(p: Any) -> Dict[str, Any]:
    return {
        "tipo": p.tipo,
        "titulo": p.titulo,
        "texto": p.texto,
        "tags": p.tags,
        "nota_existente_id": p.nota_existente_id,
        "score_duplicado": p.score_duplicado,
    }


# --- Aplicação ---


def criar_app(servicos: Optional[Servicos] = None) -> FastAPI:
    """Cria a aplicação. Com `servicos` injetados (testes) o arranque não toca no ambiente."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        proprios = getattr(app.state, "servicos", None) is None
        if proprios:
            app.state.servicos = servicos_do_ambiente()
        try:
            yield
        finally:
            if proprios and app.state.servicos is not None:
                app.state.servicos.fechar()
                app.state.servicos = None

    app = FastAPI(
        title="Mnemo Memoria API",
        description="API para o subsistema de memória do Mnemo (Retain/Recall/Reflect)",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.servicos = servicos

    # Endpoints síncronos (def): o FastAPI corre-os numa threadpool, por isso chamadas
    # bloqueantes (embeddings, LLM) não param o servidor. O lock serializa o acesso ao SQLite.

    @app.get("/memoria/recall")
    def recall_endpoint(
        s: Servicos = Depends(obter_servicos),
        consulta: str = Query(..., description="Consulta de busca"),
        k: int = Query(8, ge=1, le=50),
        budget_tokens: int = Query(2000, ge=100, le=10000),
        tags: Optional[List[str]] = Query(None),
        tipos: Optional[List[str]] = Query(None),
        explicar: bool = Query(False),
    ) -> Dict[str, Any]:
        """Busca híbrida (BM25 + embeddings) com fusão RRF."""
        with s.lock:
            resultados = s.recall.recall(
                consulta=consulta,
                k=k,
                budget_tokens=budget_tokens,
                tags=tags,
                tipos=tipos,
                explicar=explicar,
            )
        itens = [
            {
                "nota_id": r.nota_id,
                "nota_caminho": r.nota_caminho,
                "nota_titulo": r.nota_titulo,
                "trecho_id": r.trecho_id,
                "trecho_ordem": r.trecho_ordem,
                "titulo_secao": r.titulo_secao,
                "texto": r.texto,
                "score": r.score,
                "origem": r.origem,
                "citacao": r.citacao,
                "modo": r.modo,
            }
            for r in resultados
        ]
        modo = itens[0]["modo"] if itens else "vazio"
        return {"resultados": itens, "modo": modo}

    @app.post("/memoria/retain/vault")
    def retain_vault(
        s: Servicos = Depends(obter_servicos), reparar: bool = Query(False)
    ) -> Dict[str, Any]:
        """Sincroniza todo o vault (retain_vault)."""
        with s.lock:
            rel = s.retain.retain_vault(reparar=reparar)
        return {
            "novas": rel.novas,
            "atualizadas": rel.atualizadas,
            "inalteradas": rel.inalteradas,
            "removidas": rel.removidas,
            "trechos_novos": rel.trechos_novos,
            "trechos_atualizados": rel.trechos_atualizados,
            "embeddings_novos": rel.embeddings_novos,
            "embeddings_pendentes": rel.embeddings_pendentes,
            "erros": rel.erros,
        }

    @app.post("/memoria/retain/conversa")
    def retain_conversa(
        pedido: RetainConversaRequest, s: Servicos = Depends(obter_servicos)
    ) -> List[Dict[str, Any]]:
        """Extrai memórias de uma conversa via LLM; grava-as como rascunhos."""
        if s.cliente_llm is None:
            raise HTTPException(503, "LLM não configurado (falta NVIDIA_API_KEY?)")
        with s.lock:
            propostas = s.retain.retain_conversa(pedido.mensagens, s.cliente_llm)
        return [_proposta_dict(p) for p in propostas]

    @app.post("/memoria/refletir")
    def refletir(
        pedido: ReflectRequest, s: Servicos = Depends(obter_servicos)
    ) -> List[Dict[str, Any]]:
        """Consolida observações/opiniões por entidade (refletir_conversa)."""
        if s.cliente_llm is None:
            raise HTTPException(503, "LLM não configurado (falta NVIDIA_API_KEY?)")
        with s.lock:
            propostas = s.retain.refletir_conversa(pedido.mensagens, s.cliente_llm)
        return [_proposta_dict(p) for p in propostas]

    @app.get("/memoria/stats")
    def stats(s: Servicos = Depends(obter_servicos)) -> Dict[str, int]:
        """Estatísticas do índice de memória."""
        with s.lock:
            con = s.indexador.con
            return {
                "total_notas": con.execute("SELECT COUNT(*) FROM notas").fetchone()[0],
                "total_trechos": con.execute("SELECT COUNT(*) FROM trechos").fetchone()[0],
                "total_vetores": con.execute(
                    "SELECT COUNT(*) FROM vetores WHERE modelo = ? AND dim = ?",
                    (s.provider.modelo, s.provider.dim),
                ).fetchone()[0],
            }

    @app.post("/memoria/limpar")
    def limpar_memoria(s: Servicos = Depends(obter_servicos)) -> Dict[str, Any]:
        """Limpa vetores, trechos, FTS por trechos e as notas memoria/ do índice."""
        with s.lock:
            con = s.indexador.con
            try:
                con.execute("DELETE FROM vetores")
                con.execute("DELETE FROM trechos_fts")
                con.execute("DELETE FROM trechos")
                con.execute("DELETE FROM notas WHERE caminho LIKE 'memoria/%'")
                # Sem isto, o próximo retain veria as notas como "inalteradas" e não reconstruía nada.
                con.execute("UPDATE notas SET trechos_md5 = NULL")
                con.commit()
            except Exception as e:
                con.rollback()
                raise HTTPException(500, f"Erro ao limpar memória: {e}")
            # O índice vetorial em memória também tem de ser reposto.
            incrementar_versao_vetores(con)
            s.retain._indice_vetorial.invalidar()
            s.retain._indice_carregado = False
        return {"sucesso": True, "mensagem": "Memória limpa completamente"}

    @app.get("/health")
    def health(request: Request) -> Dict[str, Any]:
        return {
            "status": "ok",
            "memoria_inicializada": getattr(request.app.state, "servicos", None) is not None,
        }

    return app


# Instância para `uvicorn api_memoria:app`. Os serviços só são construídos no arranque
# (lifespan), nunca ao importar o módulo.
app = criar_app()


if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1: a API dá acesso às notas e não tem autenticação, por isso não pode ficar
    # exposta à rede local.
    uvicorn.run(
        app,
        host=os.environ.get("MNEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("MNEMO_PORT", "8000")),
    )
