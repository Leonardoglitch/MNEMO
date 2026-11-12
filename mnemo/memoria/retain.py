"""Retain: sincronização do vault, deteção de alterações, chunking, embeddings incrementais."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mnemo import Armazenamento, Permissoes
from mnemo.indexador import Indexador, separar_frontmatter
from mnemo.memoria import (
    Trecho,
    RelatorioRetain,
    estimar_tokens,
    NvidiaEmbeddings,
    FakeEmbeddings,
    EmbeddingProvider,
    IndiceVetorial,
    migrar,
    inicializar_meta,
    verificar_modelo_emb,
    incrementar_versao_vetores,
    obter_versao_vetores,
)
from mnemo.memoria.chunker import chunker
from mnemo.memoria.esquema import ESQUEMA_VERSAO


class Retain:
    """Orquestra a sincronização do vault: ficheiros -> trechos -> FTS + vetores."""

    def __init__(
        self,
        perms: Permissoes,
        armazenamento: Armazenamento,
        indexador: Indexador,
        provider: EmbeddingProvider,
        modelo_emb: str,
        dim_emb: int,
        trecho_tokens_alvo: int = 300,
        trecho_tokens_max: int = 400,
        lote_embeddings: int = 32,
        chars_por_token: float = 3.5,
    ) -> None:
        self.perms = perms
        self.armazenamento = armazenamento
        self.indexador = indexador
        self.provider = provider
        self.modelo_emb = modelo_emb
        self.dim_emb = dim_emb
        self.trecho_tokens_alvo = trecho_tokens_alvo
        self.trecho_tokens_max = trecho_tokens_max
        self.lote_embeddings = lote_embeddings
        self.chars_por_token = chars_por_token

        # Verifica modelo/dim compatíveis
        if not verificar_modelo_emb(indexador.con, modelo_emb, dim_emb):
            raise ValueError(
                f"Modelo/dimensão incompatíveis com índice existente: "
                f"{modelo_emb}/{dim_emb} vs {meta_get(indexador.con, 'modelo_emb')}/{meta_get(indexador.con, 'dim_emb')}"
            )

        # Garante esquema e meta
        migrar(indexador.con)
        inicializar_meta(indexador.con, modelo_emb, dim_emb)

        # Índice vetorial (carregado preguiçosamente)
        self._indice_vetorial = IndiceVetorial(dim_emb)
        self._indice_carregado = False

    def _carregar_indice_vetorial(self) -> None:
        """Carrega vetores do SQLite para o IndiceVetorial (uma vez)."""
        if self._indice_carregado:
            return

        # Verifica se a versão dos vetores mudou
        versao_db = obter_versao_vetores(self.indexador.con)
        if self._indice_vetorial.versao != versao_db:
            self._indice_vetorial.invalidar()

        if not self._indice_vetorial.pronto:
            # Carrega todos os vetores do modelo/dim atual
            rows = self.indexador.con.execute(
                "SELECT hash, vetor FROM vetores WHERE modelo = ? AND dim = ? ORDER BY hash",
                (self.modelo_emb, self.dim_emb),
            ).fetchall()

            if rows:
                hashes = [r[0] for r in rows]
                # Reconstitui matriz a partir dos blobs
                vetores_lista = []
                for _, blob in rows:
                    vetores_lista.append(np.frombuffer(blob, dtype=np.float32))
                matriz = np.vstack(vetores_lista).astype(np.float32)
                self._indice_vetorial.carregar(hashes, matriz)

        self._indice_carregado = True

    def _hash_trecho(self, titulo_nota: str, trecho: Trecho) -> str:
        """Calcula hash do trecho (com prefixo Título › Secção)."""
        texto_emb = trecho.prefixo_embedding(titulo_nota)
        return hashlib.sha256(texto_emb.encode("utf-8")).hexdigest()[:32]

    def _partir_nota_em_trechos(self, caminho: str, titulo_nota: str, corpo: str) -> List[Trecho]:
        """Chunka o corpo da nota e devolve lista de Trechos (sem ID, sem hash)."""
        raw_trechos = chunker(
            titulo_nota,
            corpo,
            trecho_tokens_alvo=self.trecho_tokens_alvo,
            trecho_tokens_max=self.trecho_tokens_max,
            chars_por_token=self.chars_por_token,
        )
        trechos = []
        for ordem, (titulo_secao, texto, tokens) in enumerate(raw_trechos):
            t = Trecho(
                nota_id=0,  # será preenchido depois
                ordem=ordem,
                titulo_secao=titulo_secao,
                texto=texto,
                tokens=tokens,
                hash="",  # será calculado depois
            )
            trechos.append(t)
        return trechos

    def _gravar_trechos(self, nota_id: int, trechos: List[Trecho]) -> None:
        """Substitui trechos da nota (mesma transação: trechos + trechos_fts)."""
        # Remove antigos
        self.indexador.con.execute("DELETE FROM trechos WHERE nota_id = ?", (nota_id,))
        self.indexador.con.execute("DELETE FROM trechos_fts WHERE rowid IN (SELECT id FROM trechos WHERE nota_id = ?)", (nota_id,))

        for t in trechos:
            cur = self.indexador.con.execute(
                "INSERT INTO trechos (nota_id, ordem, titulo_secao, texto, tokens, hash) VALUES (?, ?, ?, ?, ?, ?)",
                (nota_id, t.ordem, t.titulo_secao, t.texto, t.tokens, t.hash),
            )
            trecho_id = cur.lastrowid
            # FTS5 usa rowid = trecho_id
            self.indexador.con.execute(
                "INSERT INTO trechos_fts (rowid, texto, titulo_secao) VALUES (?, ?, ?)",
                (trecho_id, t.texto, t.titulo_secao or ""),
            )

    def _embeddings_pendentes_para_trechos(self, trechos: List[Trecho]) -> List[Tuple[str, Trecho]]:
        """Filtra trechos que precisam de embedding (hash não existe em vetores)."""
        if not trechos:
            return []

        hashes = [t.hash for t in trechos]
        placeholders = ",".join("?" for _ in hashes)
        rows = self.indexador.con.execute(
            f"SELECT hash FROM vetores WHERE hash IN ({placeholders}) AND modelo = ? AND dim = ?",
            hashes + [self.modelo_emb, self.dim_emb],
        ).fetchall()
        existentes = {r[0] for r in rows}

        return [(t.hash, t) for t in trechos if t.hash not in existentes]

    def _gerar_embeddings_lote(self, trechos_com_hash: List[Tuple[str, Trecho]]) -> None:
        """Gera embeddings para um lote de trechos e grava em vetores."""
        if not trechos_com_hash:
            return

        textos = [t.texto for _, t in trechos_com_hash]
        # Embede com prefixo Título › Secção
        textos_emb = [t.prefixo_embedding("") for _, t in trechos_com_hash]  # prefixo vazio, já está no texto

        vecs = self.provider.embed(textos_emb, input_type="passage")
        # vecs já vem normalizado

        for (hash_t, _), vec in zip(trechos_com_hash, vecs):
            blob = vec.astype(np.float32).tobytes()
            self.indexador.con.execute(
                "INSERT OR REPLACE INTO vetores (hash, modelo, dim, vetor) VALUES (?, ?, ?, ?)",
                (hash_t, self.modelo_emb, self.dim_emb, blob),
            )

    def _processar_nota(
        self,
        caminho: str,
        alvo: Path,
        relatorio: RelatorioRetain,
    ) -> Tuple[bool, Optional[str]]:
        """Processa uma nota: chunking, FTS, embeddings. Devolve (alterada, erro_pendente)."""
        conteudo = alvo.read_text(encoding="utf-8-sig", errors="replace")
        frontmatter, corpo = separar_frontmatter(conteudo)
        titulo_nota = self.indexador._extrair_titulo(conteudo, alvo.stem) if hasattr(self.indexador, '_extrair_titulo') else alvo.stem
        # Fallback se não tiver o método
        try:
            from mnemo.indexador import extrair_titulo
            titulo_nota = extrair_titulo(conteudo, alvo.stem)
        except Exception:
            titulo_nota = alvo.stem

        # Calcula MD5 do conteúdo (para deteção de alterações)
        md5 = hashlib.md5(conteudo.encode("utf-8")).hexdigest()

        # Verifica se nota já existe e se MD5 é igual
        row = self.indexador.con.execute(
            "SELECT id, md5 FROM notas WHERE caminho = ?", (caminho,)
        ).fetchone()

        if row:
            nota_id, md5_guardado = row
            if md5_guardado == md5:
                relatorio.inalteradas += 1
                return False, None  # sem alterações
            # MD5 diferente -> atualiza
            relatorio.atualizadas += 1
        else:
            # Nova nota
            cur = self.indexador.con.execute(
                "INSERT INTO notas (caminho, titulo, modificado_ns, tags, md5) VALUES (?, ?, ?, ?, ?)",
                (caminho, titulo_nota, alvo.stat().st_mtime_ns, "[]", md5),
            )
            nota_id = cur.lastrowid
            relatorio.novas += 1

        # Atualiza mtime e md5
        self.indexador.con.execute(
            "UPDATE notas SET modificado_ns = ?, md5 = ? WHERE id = ?",
            (alvo.stat().st_mtime_ns, md5, nota_id),
        )

        # Chunking
        trechos = self._partir_nota_em_trechos(caminho, titulo_nota, corpo)

        # Calcula hashes
        for t in trechos:
            t.nota_id = nota_id
            t.hash = self._hash_trecho(titulo_nota, t)

        # Grava trechos + FTS (mesma transação)
        self._gravar_trechos(nota_id, trechos)
        relatorio.trechos_novos += len(trechos)

        # Embeddings incrementais
        pendentes = self._embeddings_pendentes_para_trechos(trechos)
        if pendentes:
            try:
                self._gerar_embeddings_lote(pendentes)
                relatorio.embeddings_novos += len(pendentes)
                # Invalida índice vetorial
                self._indice_vetorial.invalidar()
            except Exception as e:
                # Marca como pendente para retry posterior
                relatorio.embeddings_pendentes.append(caminho)
                relatorio.erros.append(f"Embeddings falharam para {caminho}: {e}")
                return True, f"Embeddings pendentes: {e}"

        return True, None

    def retain_vault(self, reparar: bool = False) -> RelatorioRetain:
        """Sincroniza todo o vault: percorre pastas permitidas, processa notas alteradas."""
        relatorio = RelatorioRetain()
        validas = self.indexador._notas_validas()

        # Carrega índice vetorial uma vez
        self._carregar_indice_vetorial()

        for rel, alvo in validas.items():
            try:
                alterada, erro = self._processar_nota(rel, alvo, relatorio)
                if erro and not reparar:
                    # Em modo normal, erros de embeddings vão para relatório
                    pass
            except Exception as e:
                relatorio.erros.append(f"Erro a processar {rel}: {e}")

        # Remove notas que já não existem ou não são permitidas
        existentes = {
            caminho for caminho, in self.indexador.con.execute("SELECT caminho FROM notas")
        }
        for rel in set(existentes) - set(validas):
            self.indexador._remover(rel)
            relatorio.removidas += 1

        # Se modo reparar, tenta embeddings pendentes
        if reparar and relatorio.embeddings_pendentes:
            for caminho in relatorio.embeddings_pendentes[:]:
                try:
                    alvo = self.perms.verificar(caminho)
                    if alvo.is_file():
                        rel2 = RelatorioRetain()
                        _, err = self._processar_nota(caminho, alvo, rel2)
                        if not err:
                            relatorio.embeddings_pendentes.remove(caminho)
                            relatorio.embeddings_novos += 1
                except Exception:
                    pass

        self.indexador.con.commit()
        return relatorio


def meta_get(con: sqlite3.Connection, chave: str) -> Optional[str]:
    row = con.execute("SELECT valor FROM meta WHERE chave = ?", (chave,)).fetchone()
    return row[0] if row else None