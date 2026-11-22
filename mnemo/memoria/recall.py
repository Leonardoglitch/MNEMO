"""Recall: busca híbrida (BM25 + embeddings), fusão RRF, diversidade, budget de tokens."""

from __future__ import annotations

from typing import List, Optional, Set, Tuple, Any, Dict

import numpy as np

from mnemo import Permissoes
from mnemo.indexador import Indexador
from mnemo.memoria import (
    Resultado,
    IndiceVetorial,
    EmbeddingProvider,
    FakeEmbeddings,
    ConsultaCache,
    estimar_tokens,
)
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class RecallParams:
    """Parâmetros avançados para busca recall."""
    k: int = 8
    budget_tokens: int = 2000
    tags: Optional[List[str]] = None
    tipos: Optional[List[str]] = None
    status: Optional[List[str]] = None
    entidade: Optional[str] = None
    since_ns: Optional[int] = None
    until_ns: Optional[int] = None
    peso_recencia: float = 0.1
    peso_confianca: float = 0.15
    explicar: bool = False

    @staticmethod
    def from_query_params(
        k: int = 8,
        budget_tokens: int = 2000,
        tags: Optional[List[str]] = None,
        tipos: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        entidade: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        peso_recencia: float = 0.1,
        peso_confianca: float = 0.15,
        explicar: bool = False,
    ) -> "RecallParams":
        """Cria RecallParams a partir de parâmetros de query (strings ISO para datas)."""
        since_ns = None
        until_ns = None
        if since:
            try:
                dt = datetime.fromisoformat(since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                since_ns = int(dt.timestamp() * 1_000_000_000)
            except ValueError:
                pass
        if until:
            try:
                dt = datetime.fromisoformat(until)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                until_ns = int(dt.timestamp() * 1_000_000_000)
            except ValueError:
                pass
        return RecallParams(
            k=k,
            budget_tokens=budget_tokens,
            tags=tags,
            tipos=tipos,
            status=status,
            entidade=entidade,
            since_ns=since_ns,
            until_ns=until_ns,
            peso_recencia=peso_recencia,
            peso_confianca=peso_confianca,
            explicar=explicar,
        )


class Recall:
    """Busca híbrida: lexical (BM25) + semântica (cosseno) com fusão RRF."""

    def __init__(
        self,
        indexador: Indexador,
        permissoes: Permissoes,
        provider: EmbeddingProvider,
        indice_vetorial: IndiceVetorial,
        pesos_rrf: Tuple[float, float] = (1.0, 1.0),
        k0_rrf: int = 60,
        candidatos_por_pesquisa: int = 50,
        max_trechos_por_nota: int = 2,
        chars_por_token: float = 3.5,
    ) -> None:
        self.indexador = indexador
        self.permissoes = permissoes
        self.provider = provider
        self.indice_vetorial = indice_vetorial
        self.pesos_rrf = pesos_rrf  # (peso_lex, peso_sem)
        self.k0_rrf = k0_rrf
        self.candidatos_por_pesquisa = candidatos_por_pesquisa
        self.max_trechos_por_nota = max_trechos_por_nota
        self.chars_por_token = chars_por_token
        self._cache_consulta = ConsultaCache()

    def _ids_permitidos(
        self,
        tags: Optional[List[str]] = None,
        tipos: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        entidade: Optional[str] = None,
        since_ns: Optional[int] = None,
        until_ns: Optional[int] = None,
    ) -> List[int]:
        """Devolve IDs de notas permitidas (filtro por pastas + tags + tipos + status + entidade + temporal)."""
        # Começa com todas as notas permitidas por pastas
        validas = self.indexador._notas_validas()
        if not validas:
            return []

        # Obtém IDs das notas válidas com filtros adicionais no SQL
        placeholders = ",".join("?" for _ in validas)
        conditions = ["caminho IN (" + placeholders + ")"]
        params_list: List[Any] = list(validas.keys())

        # Filtro por tags
        if tags:
            tags_norm = [t.lower() for t in tags]
            ph_tags = ",".join("?" for _ in tags_norm)
            conditions.append(f"id IN (SELECT nota_id FROM nota_tags WHERE tag IN ({ph_tags}) GROUP BY nota_id HAVING COUNT(DISTINCT tag) = ?)")
            params_list.extend(tags_norm)
            params_list.append(len(tags_norm))

        # Filtro por tipo
        if tipos:
            ph_tipos = ",".join("?" for _ in tipos)
            conditions.append(f"tipo IN ({ph_tipos})")
            params_list.extend(tipos)

        # Filtro por status
        if status:
            ph_status = ",".join("?" for _ in status)
            conditions.append(f"status IN ({ph_status})")
            params_list.extend(status)

        # Filtro por entidade
        if entidade:
            conditions.append("entidade = ?")
            params_list.append(entidade)

        # Filtro temporal since
        if since_ns is not None:
            conditions.append("modificado_ns >= ?")
            params_list.append(since_ns)

        # Filtro temporal until
        if until_ns is not None:
            conditions.append("modificado_ns <= ?")
            params_list.append(until_ns)

        where_clause = " AND ".join(conditions)
        query = f"SELECT id FROM notas WHERE {where_clause}"
        rows = self.indexador.con.execute(query, params_list).fetchall()
        return [r[0] for r in rows]

    def _bm25(self, consulta: str, ids_permitidos: List[int], n: int) -> List[Tuple[int, int, float]]:
        """Busca lexical no trechos_fts. Devolve [(trecho_id, nota_id, score_bm25), ...]."""
        if not ids_permitidos:
            return []

        termos = self.indexador._limpar_consulta(consulta) if hasattr(self.indexador, '_limpar_consulta') else consulta
        # Reutiliza a limpeza do indexador
        import re
        termos = re.findall(r"\w+", consulta)
        if not termos:
            return []

        expressao = " ".join(f'"{t}"*' for t in termos)
        placeholders = ",".join("?" for _ in ids_permitidos)

        sql = f"""
            SELECT t.id, t.nota_id, bm25(trechos_fts, 1.0, 0.5)
            FROM trechos_fts JOIN trechos t ON t.id = trechos_fts.rowid
            WHERE trechos_fts MATCH ?
              AND t.nota_id IN ({placeholders})
            ORDER BY bm25(trechos_fts, 1.0, 0.5)
            LIMIT ?
        """
        params = (expressao,) + tuple(ids_permitidos) + (n,)
        rows = self.indexador.con.execute(sql, params).fetchall()
        return [(r[0], r[1], -r[2]) for r in rows]  # score positivo (maior = melhor)

    def _semantica(self, consulta: str, ids_permitidos: List[int], n: int) -> List[Tuple[int, int, float]]:
        """Busca semântica via cosseno. Devolve [(trecho_id, nota_id, score_cos), ...].
        
        Degradação graciosa: se embeddings falharem (rede, API, índice vazio), 
        retorna lista vazia para que o recall funcione só com BM25.
        """
        if not self.indice_vetorial.pronto or not ids_permitidos:
            return []

        # Obtém embedding da consulta (com cache)
        cache_key = f"{self.provider.modelo}:{self.provider.dim}:query:{consulta}"
        vetor_consulta = self._cache_consulta.get(cache_key)
        if vetor_consulta is None:
            try:
                vecs = self.provider.embed([consulta], input_type="query")
                vetor_consulta = vecs[0]
                self._cache_consulta.set(cache_key, vetor_consulta)
            except Exception as e:
                # Degradação graciosa: loga e retorna vazio para fallback lexical
                import logging
                logging.warning(f"Falha ao embeber consulta (fallback lexical): {e}")
                return []

        # Mapeia ids_permitidos (nota_ids) para índices locais em _hashes
        # Precisamos de saber quais trechos pertencem a notas permitidas
        if not ids_permitidos:
            return []

        placeholders = ",".join("?" for _ in ids_permitidos)
        rows = self.indexador.con.execute(
            f"SELECT id FROM trechos WHERE nota_id IN ({placeholders}) ORDER BY id",
            list(ids_permitidos),
        ).fetchall()
        trecho_ids_permitidos = [r[0] for r in rows]

        if not trecho_ids_permitidos:
            return []

        # Obtém hashes dos trechos permitidos
        hashes_permitidos = self._hashes_por_trecho_id(trecho_ids_permitidos)

        # Top-k no índice vetorial (sem filtro de índices, filtraremos depois)
        top_k = self.indice_vetorial.top_k(vetor_consulta, n * 3)  # busca mais para compensar filtragem

        # Mapeia de volta para trecho_id, filtrando apenas os permitidos
        hash_list = self.indice_vetorial._hashes
        hashes_permitidos_set = set(hashes_permitidos)
        resultados = []
        for idx_local, score in top_k:
            if 0 <= idx_local < len(hash_list):
                hash_t = hash_list[idx_local]
                if hash_t not in hashes_permitidos_set:
                    continue
                # Obtém trecho_id e nota_id do hash
                row = self.indexador.con.execute(
                    "SELECT id, nota_id FROM trechos WHERE hash = ?", (hash_t,)
                ).fetchone()
                if row:
                    resultados.append((row[0], row[1], score))

        return resultados

    def _hashes_por_trecho_id(self, trecho_ids: List[int]) -> List[str]:
        if not trecho_ids:
            return []
        placeholders = ",".join("?" for _ in trecho_ids)
        rows = self.indexador.con.execute(
            f"SELECT hash FROM trechos WHERE id IN ({placeholders})",
            trecho_ids,
        ).fetchall()
        return [r[0] for r in rows]

    def _rrf_fusion(
        self,
        lex: List[Tuple[int, int, float]],
        sem: List[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float, dict[str, Any]]]:
        """
        Fusão RRF: score = sum(peso / (k0 + rank)).
        Devolve [(trecho_id, nota_id, score_rrf, {"lex": rank, "sem": rank, "score_lex": float, "score_sem": float}), ...]
        """
        peso_lex, peso_sem = self.pesos_rrf

        # Rank de cada resultado em cada lista (1-indexed)
        rank_lex = {tid: i + 1 for i, (tid, _, _) in enumerate(lex)}
        rank_sem = {tid: i + 1 for i, (tid, _, _) in enumerate(sem)}
        
        # Scores originais (BM25 e cosseno)
        score_lex_dict = {tid: score for tid, _, score in lex}
        score_sem_dict = {tid: score for tid, _, score in sem}

        # Todos os trechos únicos
        todos_ids = set(rank_lex.keys()) | set(rank_sem.keys())

        fundidos = []
        for tid in todos_ids:
            score = 0.0
            origem: dict[str, Any] = {}
            if tid in rank_lex:
                score += peso_lex / (self.k0_rrf + rank_lex[tid])
                origem["lex"] = rank_lex[tid]
                origem["score_lex"] = score_lex_dict[tid]
            if tid in rank_sem:
                score += peso_sem / (self.k0_rrf + rank_sem[tid])
                origem["sem"] = rank_sem[tid]
                origem["score_sem"] = score_sem_dict[tid]

            # Obtém nota_id
            row = self.indexador.con.execute("SELECT nota_id FROM trechos WHERE id = ?", (tid,)).fetchone()
            if row:
                fundidos.append((tid, row[0], score, origem))

        # Ordena por score descendente
        fundidos.sort(key=lambda x: x[2], reverse=True)
        return fundidos

    def _diversificar(self, fundidos: List[Tuple[int, int, float, dict[str, int]]]) -> List[Tuple[int, int, float, dict[str, int]]]:
        """Limita a max_trechos_por_nota."""
        if self.max_trechos_por_nota <= 0:
            return fundidos

        contagem_por_nota: dict[int, int] = {}
        resultado = []
        for tid, nid, score, origem in fundidos:
            cnt = contagem_por_nota.get(nid, 0)
            if cnt < self.max_trechos_por_nota:
                resultado.append((tid, nid, score, origem))
                contagem_por_nota[nid] = cnt + 1
        return resultado

    def _calcular_score_final(
        self,
        score_rrf: float,
        nid: int,
        peso_recencia: float,
        peso_confianca: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Calcula score final combinando RRF + recência + confiança.
        
        Returns: (score_final, breakdown_dict)
        """
        if peso_recencia == 0 and peso_confianca == 0:
            return score_rrf, {"rrf": score_rrf}
        
        # Busca modificado_ns e confianca (do frontmatter status/tipo)
        row = self.indexador.con.execute(
            "SELECT modificado_ns, tipo, status, entidade FROM notas WHERE id = ?",
            (nid,),
        ).fetchone()
        
        if not row:
            return score_rrf, {"rrf": score_rrf}
        
        modificado_ns, tipo, status, entidade = row
        breakdown = {"rrf": score_rrf}
        score_final = score_rrf
        
        # Recência: normaliza para 0-1 baseado em janela de 30 dias
        if peso_recencia > 0 and modificado_ns:
            import time
            agora_ns = int(time.time() * 1_000_000_000)
            janela_ns = 30 * 24 * 60 * 60 * 1_000_000_000  # 30 dias
            idade_ns = agora_ns - modificado_ns
            recencia_norm = max(0.0, 1.0 - (idade_ns / janela_ns))
            score_final += peso_recencia * recencia_norm
            breakdown["recencia"] = recencia_norm
            breakdown["recencia_weight"] = peso_recencia
        
        # Confiança: extrai do frontmatter (campo confianca ou score_duplicado)
        # Para notas de memória, pode haver confianca no frontmatter
        # Como não temos coluna confianca, usamos heurística baseada em status/tipo
        if peso_confianca > 0:
            confianca = 0.5  # default
            if status == "ativo":
                confianca = 0.8
            elif status == "rascunho":
                confianca = 0.4
            elif tipo == "facto":
                confianca = 0.9
            elif tipo == "decisao":
                confianca = 0.85
            elif tipo == "observacao":
                confianca = 0.6
            score_final += peso_confianca * confianca
            breakdown["confianca"] = confianca
            breakdown["confianca_weight"] = peso_confianca
        
        return score_final, breakdown

    def _empacotar_resultados(
        self,
        fundidos: List[Tuple[int, int, float, Dict[str, Any]]],
        k: int,
        budget_tokens: int,
        peso_recencia: float = 0.1,
        peso_confianca: float = 0.15,
        explicar: bool = False,
    ) -> List[Resultado]:
        """Converte para objetos Resultado, respeitando budget de tokens e aplicando ranking final."""
        # Primeiro calcula scores finais para todos os candidatos
        candidatos_com_score = []
        for tid, nid, score_rrf, origem in fundidos:
            score_final, breakdown = self._calcular_score_final(score_rrf, nid, peso_recencia, peso_confianca)
            candidatos_com_score.append((tid, nid, score_final, origem, breakdown))
        
        # Reordena por score final
        candidatos_com_score.sort(key=lambda x: x[2], reverse=True)
        
        resultados: List[Resultado] = []
        tokens_acumulados = 0

        for tid, nid, score_final, origem, breakdown in candidatos_com_score[:k]:
            row = self.indexador.con.execute(
                "SELECT nota_id, ordem, titulo_secao, texto, tokens FROM trechos WHERE id = ?",
                (tid,),
            ).fetchone()
            if not row:
                continue

            _, ordem, titulo_secao, texto, tokens = row

            # Nota info
            row_nota = self.indexador.con.execute(
                "SELECT caminho, titulo, tipo, status, entidade, modificado_ns FROM notas WHERE id = ?", (nid,)
            ).fetchone()
            if not row_nota:
                continue
            caminho, titulo, tipo, status, entidade, modificado_ns = row_nota

            # Verifica permissão final (defesa em profundidade)
            try:
                self.permissoes.verificar(caminho)
            except Exception:
                continue

            tokens_trecho = estimar_tokens(texto, self.chars_por_token)
            if tokens_acumulados + tokens_trecho > budget_tokens and resultados:
                break

            # Adiciona info de explainability se pedido
            origem_final = dict(origem)
            if explicar:
                origem_final["explicar"] = {
                    "lex_count": len([1 for o in origem if "lex" in o]),
                    "sem_count": len([1 for o in origem if "sem" in o]),
                    "score_breakdown": breakdown,
                    "nota_info": {
                        "tipo": tipo,
                        "status": status,
                        "entidade": entidade,
                        "modificado_ns": modificado_ns,
                    },
                }

            r = Resultado(
                nota_id=nid,
                nota_caminho=caminho,
                nota_titulo=titulo,
                trecho_id=tid,
                trecho_ordem=ordem,
                titulo_secao=titulo_secao or "",
                texto=texto,
                score=score_final,
                origem=origem_final,
                citacao=f"[[{caminho}]]",
                modo="hibrido" if ("lex" in origem and "sem" in origem) else ("lexical" if "lex" in origem else "semantica"),
            )
            resultados.append(r)
            tokens_acumulados += tokens_trecho

        return resultados

    def recall(
        self,
        consulta: str,
        *,
        k: int = 8,
        budget_tokens: int = 2000,
        tags: Optional[List[str]] = None,
        tipos: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        entidade: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        peso_recencia: float = 0.1,
        peso_confianca: float = 0.15,
        explicar: bool = False,
    ) -> List[Resultado]:
        """Busca híbrida principal com filtros avançados.
        
        Args:
            consulta: Texto da consulta
            k: Número máximo de resultados
            budget_tokens: Orçamento de tokens para resultados
            tags: Filtrar por tags (todas devem estar presentes)
            tipos: Filtrar por tipo (facto, decisao, preferencia, experiencia, observacao)
            status: Filtrar por status (ativo, rascunho, arquivado)
            entidade: Filtrar por entidade (ex: "João", "Projeto Alpha")
            since: Data ISO (ex: "2026-01-01") - só notas modificadas depois
            until: Data ISO (ex: "2026-12-31") - só notas modificadas antes
            peso_recencia: Peso da recência no ranking final (0-1)
            peso_confianca: Peso da confiança no ranking final (0-1)
            explicar: Se True, inclui breakdown do score na origem
        """
        from datetime import datetime, timezone
        
        # Converte since/until strings para nanossegundos
        since_ns = None
        until_ns = None
        if since:
            try:
                dt = datetime.fromisoformat(since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                since_ns = int(dt.timestamp() * 1_000_000_000)
            except ValueError:
                pass
        if until:
            try:
                dt = datetime.fromisoformat(until)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                until_ns = int(dt.timestamp() * 1_000_000_000)
            except ValueError:
                pass
        
        ids_permitidos = self._ids_permitidos(
            tags=tags,
            tipos=tipos,
            status=status,
            entidade=entidade,
            since_ns=since_ns,
            until_ns=until_ns,
        )

        # Busca lexical
        lex = self._bm25(consulta, ids_permitidos, self.candidatos_por_pesquisa)

        # Busca semântica
        sem = self._semantica(consulta, ids_permitidos, self.candidatos_por_pesquisa)

        # Fusão RRF
        fundidos = self._rrf_fusion(lex, sem)

        # Diversidade
        fundidos = self._diversificar(fundidos)

        # Empacota resultados com ranking final
        resultados = self._empacotar_resultados(fundidos, k, budget_tokens, peso_recencia, peso_confianca, explicar)

        return resultados