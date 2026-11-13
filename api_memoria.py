"""API FastAPI para o subsistema de memória do Mnemo (Retain/Recall/Reflect).

Endpoints:
- GET  /memoria/recall           - Busca híbrida (RRF)
- POST /memoria/retain/vault     - Sincroniza vault (retain_vault)
- POST /memoria/retain/conversa  - Extrai memórias de conversa (retain_conversa)
- POST /memoria/refletir         - Consolida observações/opiniões (refletir_conversa)
- GET  /memoria/stats            - Estatísticas do índice
- POST /memoria/limpar           - Limpa dados da memória
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from mnemo import Permissoes
from mnemo.memoria import (
    Recall,
    Retain,
    Reflect,
    NvidiaEmbeddings,
    IndiceVetorial,
)
from mnemo.indexador import Indexador
from mnemo.armazenamento import Armazenamento
from mnemo import Permissoes, FerramentasVault
from mnemo.modelo_nvidia import ClienteNVIDIA, ErroModeloNVIDIA
import os


app = FastAPI(
    title="Mnemo Memoria API",
    description="API para o subsistema de memória do Mnemo (Retain/Recall/Reflect)",
    version="0.1.0",
)


# Modelos Pydantic
class RecallRequest(BaseModel):
    consulta: str
    k: int = 8
    budget_tokens: int = 2000
    tags: Optional[List[str]] = None
    tipos: Optional[List[str]] = None
    explicar: bool = False


class RecallResponse(BaseModel):
    resultados: List[Dict[str, Any]]
    modo: str


class RetainVaultResponse(BaseModel):
    novas: int
    atualizadas: int
    inalteradas: int
    removidas: int
    trechos_novos: int
    trechos_atualizados: int
    embeddings_novos: int
    embeddings_pendentes: List[str]
    erros: List[str]


class RetainConversaRequest(BaseModel):
    mensagens: List[Dict[str, Any]]


class PropostaResponse(BaseModel):
    tipo: str
    titulo: str
    texto: str
    tags: List[str]
    nota_existente_id: Optional[int] = None
    score_duplicado: float


class ReflectRequest(BaseModel):
    mensagens: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    total_notas: int
    total_trechos: int
    total_vetores: int


class LimparResponse(BaseModel):
    sucesso: bool
    mensagem: str


# Dependências globais (inicializadas no startup)
class AppState:
    perms: Optional[Any] = None
    vault_path: str = "vault-teste"
    indexador: Optional[Any] = None
    armazenamento: Optional[Any] = None
    provider: Optional[Any] = None
    indice: Optional[Any] = None
    recall: Optional[Any] = None
    retain: Optional[Any] = None
    reflect: Optional[Any] = None
    cliente_llm: Optional[Any] = None


state = AppState()


@app.on_event("startup")
async def startup() -> None:
    """Inicializa componentes na inicialização da aplicação."""
    import os
    
    # Carrega .env
    def carregar_env(caminho: str = ".env") -> None:
        from pathlib import Path
        if not Path(caminho).is_absolute():
            caminho = str(Path(__file__).parent / caminho)
        try:
            with open(caminho, encoding="utf-8-sig") as f:
                for linha in f:
                    linha = linha.strip()
                    if not linha or linha.startswith("#") or "=" not in linha:
                        continue
                    chave, valor = linha.split("=", 1)
                    os.environ[chave.strip()] = valor.strip().strip("'\"")
        except FileNotFoundError:
            pass
    
    carregar_env()
    
    state.vault_path = os.environ.get("MNEMO_VAULT", "vault-teste")
    state.perms = Permissoes(state.vault_path)
    state.indexador = Indexador(state.perms)
    state.indexador.reindexar_tudo()
    state.armazenamento = Armazenamento(state.perms)
    
    # Cliente LLM
    try:
        from mnemo.modelo_nvidia import ClienteNVIDIA
        state.cliente_llm = ClienteNVIDIA()
    except ErroModeloNVIDIA:
        state.cliente_llm = None
    
    # Provider de embeddings
    from mnemo.memoria import NvidiaEmbeddings
    state.provider = NvidiaEmbeddings()
    
    # Índice vetorial
    from mnemo.memoria import IndiceVetorial
    state.indice = IndiceVetorial(state.provider.dim)
    
    # Carrega vetores existentes
    rows = state.indexador.con.execute(
        "SELECT hash, vetor FROM vetores WHERE modelo = ? AND dim = ?",
        (state.provider.modelo, state.provider.dim),
    ).fetchall()
    if rows:
        import numpy as np
        hashes = [r[0] for r in rows]
        vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        state.indice.carregar(hashes, vecs)
    
    # Instancia componentes
    from mnemo.memoria import Retain, Recall, Reflect
    state.retain = Retain(
        perms=Permissoes(state.vault_path),
        armazenamento=Armazenamento(Permissoes(state.vault_path)),
        indexador=state.indexador,
        provider=state.provider,
        modelo_emb=state.provider.modelo,
        dim_emb=state.provider.dim,
    )
    state.recall = Recall(
        state.indexador,
        Permissoes(state.vault_path),
        state.provider,
        state.indice,
    )
    state.reflect = Reflect()


@app.on_event("shutdown")
async def shutdown() -> None:
    """Limpa recursos no encerramento."""
    if state.indexador:
        state.indexador.fechar()


@app.get("/memoria/recall", response_model=Dict[str, Any])
async def recall_endpoint(
    consulta: str = Query(..., description="Consulta de busca"),
    k: int = Query(8, ge=1, le=50),
    budget_tokens: int = Query(2000, ge=100, le=10000),
    tags: Optional[List[str]] = Query(None),
    tipos: Optional[List[str]] = Query(None),
    explicar: bool = Query(False),
) -> Dict[str, Any]:
    """Busca híbrida (BM25 + embeddings) com fusão RRF."""
    if not state.recall:
        raise HTTPException(503, "Memória não inicializada")
    
    resultados = state.recall.recall(
        consulta=consulta,
        k=k,
        budget_tokens=budget_tokens,
        tags=tags,
        tipos=tipos,
        explicar=explicar,
    )
    
    resultados_dict = []
    for r in resultados:
        d = {
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
        resultados_dict.append(d)
    
    return {"resultados": resultados_dict, "modo": "hibrido"}


@app.post("/memoria/retain/vault", response_model=Dict[str, Any])
async def retain_vault(reparar: bool = Query(False)) -> Dict[str, Any]:
    """Sincroniza todo o vault (retain_vault)."""
    if not state.retain:
        raise HTTPException(503, "Memória não inicializada")
    
    rel = state.retain.retain_vault(reparar=reparar)
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


@app.post("/memoria/retain/conversa", response_model=List[Dict[str, Any]])
async def retain_conversa(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extrai memórias de uma conversa via LLM (retain_conversa)."""
    from mnemo import Permissoes, Armazenamento
    from mnemo.indexador import Indexador
    from mnemo.memoria import Retain, NvidiaEmbeddings
    
    perms = Permissoes("vault-teste")
    armazenamento = Armazenamento(Permissoes("vault-teste"))
    indexador = Indexador(Permissoes("vault-teste"))
    
    from mnemo.memoria import NvidiaEmbeddings
    provider = NvidiaEmbeddings()
    
    retain = Retain(
        perms=perms,
        armazenamento=armazenamento,
        indexador=indexador,
        provider=provider,
        modelo_emb=provider.modelo,
        dim_emb=provider.dim,
    )
    
    propostas = retain.retain_conversa(request["mensagens"], state.cliente_llm)
    
    return [
        {
            "tipo": p.tipo,
            "titulo": p.titulo,
            "texto": p.texto,
            "tags": p.tags,
            "nota_existente_id": p.nota_existente_id,
            "score_duplicado": p.score_duplicado,
        }
        for p in propostas
    ]


@app.post("/memoria/refletir", response_model=List[Dict[str, Any]])
async def refletir(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Consolida observações/opiniões por entidade (refletir_conversa)."""
    assert state.retain is not None
    propostas = state.retain.refletir_conversa(request["mensagens"], state.cliente_llm)
    
    return [
        {
            "tipo": p.tipo,
            "titulo": p.titulo,
            "texto": p.texto,
            "tags": p.tags,
            "nota_existente_id": p.nota_existente_id,
            "score_duplicado": p.score_duplicado,
        }
        for p in state.retain.refletir_conversa(request["mensagens"], state.cliente_llm)
    ]


@app.get("/memoria/stats", response_model=Dict[str, int])
async def stats() -> Dict[str, int]:
    """Estatísticas do índice de memória."""
    assert state.indexador is not None
    assert state.provider is not None
    
    total_notas = state.indexador.con.execute("SELECT COUNT(*) FROM notas").fetchone()[0]
    total_trechos = state.indexador.con.execute("SELECT COUNT(*) FROM trechos").fetchone()[0]
    total_vetores = state.indexador.con.execute(
        "SELECT COUNT(*) FROM vetores WHERE modelo = ? AND dim = ?",
        (state.provider.modelo, state.provider.dim),
    ).fetchone()[0]
    
    return {
        "total_notas": total_notas,
        "total_trechos": total_trechos,
        "total_vetores": total_vetores,
    }


@app.post("/memoria/limpar", response_model=Dict[str, Any])
async def limpar_memoria() -> Dict[str, Any]:
    """Limpa todos os dados da memória (vetores, trechos, FTS, notas memoria/)."""
    if not state.indexador:
        raise HTTPException(503, "Índice não inicializado")
    
    try:
        from mnemo import Permissoes
        perms = Permissoes("vault-teste")
        indexador = Indexador(perms)
        
        # Apaga vetores
        indexador.con.execute("DELETE FROM vetores")
        # Apaga trechos e FTS
        indexador.con.execute("DELETE FROM trechos_fts")
        indexador.con.execute("DELETE FROM trechos")
        indexador.con.execute("DELETE FROM notas WHERE caminho LIKE 'memoria/%'")
        indexador.con.commit()
        
        # Limpa índice em memória
        # (reiniciaria o índice)
        
        return {"sucesso": True, "mensagem": "Memória limpa completamente"}
    except Exception as e:
        raise HTTPException(500, f"Erro ao limpar memória: {e}")


# Health check
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "memoria_inicializada": state.recall is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)