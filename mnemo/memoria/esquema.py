"""Migrações de esquema e tabela meta para o subsistema de memória.

Gera as tabelas: trechos, trechos_fts, vetores, meta.
Usa PRAGMA user_version para versionamento idempotente.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# Versão actual do esquema (incrementar a cada migração)
ESQUEMA_VERSAO = 4


ESQUEMA_MEMORIA = """
-- Tabela meta: chave-valor para versão, modelo, dimensões, etc.
CREATE TABLE IF NOT EXISTS meta (
    chave TEXT PRIMARY KEY,
    valor TEXT
);

-- Trechos de notas (chunking)
CREATE TABLE IF NOT EXISTS trechos (
    id            INTEGER PRIMARY KEY,
    nota_id       INTEGER NOT NULL REFERENCES notas(id) ON DELETE CASCADE,
    ordem         INTEGER NOT NULL,
    titulo_secao  TEXT,
    texto         TEXT NOT NULL,
    tokens        INTEGER NOT NULL,
    hash          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trechos_nota ON trechos(nota_id);
CREATE INDEX IF NOT EXISTS idx_trechos_hash ON trechos(hash);

-- FTS5 para busca lexical por trechos
CREATE VIRTUAL TABLE IF NOT EXISTS trechos_fts USING fts5(
    texto, titulo_secao,
    tokenize = 'unicode61 remove_diacritics 2'
);  -- rowid = trechos.id, escrito na mesma transação

-- Vetores de embeddings (chave composta: hash, modelo, dim)
CREATE TABLE IF NOT EXISTS vetores (
    hash      TEXT NOT NULL,
    modelo    TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vetor     BLOB NOT NULL,  -- float32, normalizado L2=1
    PRIMARY KEY (hash, modelo, dim)
);
"""

# Adiciona coluna md5 à tabela notas existente
MIGRACAO_NOTAS_MD5 = """
ALTER TABLE notas ADD COLUMN md5 TEXT;
CREATE INDEX IF NOT EXISTS idx_notas_md5 ON notas(md5);
"""


def obter_versao(con: sqlite3.Connection) -> int:
    """Lê a versão do esquema via PRAGMA user_version."""
    row = con.execute("PRAGMA user_version").fetchone()
    return row[0] if row else 0


def definir_versao(con: sqlite3.Connection, versao: int) -> None:
    """Define a versão do esquema via PRAGMA user_version."""
    con.execute(f"PRAGMA user_version = {versao}")


def meta_get(con: sqlite3.Connection, chave: str) -> Optional[str]:
    row = con.execute("SELECT valor FROM meta WHERE chave = ?", (chave,)).fetchone()
    return row[0] if row else None


def meta_set(con: sqlite3.Connection, chave: str, valor: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO meta (chave, valor) VALUES (?, ?)",
        (chave, valor),
    )


def migrar(con: sqlite3.Connection) -> None:
    """Executa migrações necessárias para chegar à ESQUEMA_VERSAO."""
    versao_atual = obter_versao(con)

    if versao_atual >= ESQUEMA_VERSAO:
        return  # já está atualizado

    # Executa script base (cria tabelas se não existirem)
    con.executescript(ESQUEMA_MEMORIA)

    # Migração v1 -> v2: adicionar coluna md5 a notas
    if versao_atual < 2:
        cols = {row[1] for row in con.execute("PRAGMA table_info(notas)")}
        if "md5" not in cols:
            con.execute("ALTER TABLE notas ADD COLUMN md5 TEXT")
            con.execute("CREATE INDEX IF NOT EXISTS idx_notas_md5 ON notas(md5)")

    # Migração v2 -> v3: garantir índices em trechos
    if versao_atual < 3:
        # Índices já criados no ESQUEMA_MEMORIA (IF NOT EXISTS)
        pass

    # Migração v3 -> v4: md5 do conteúdo a partir do qual os trechos foram construídos.
    # notas.md5 pertence ao Indexador (que o atualiza em reindexar_tudo); se o Retain o usasse
    # para detetar alterações, nunca criaria trechos para notas já indexadas.
    if versao_atual < 4:
        cols = {row[1] for row in con.execute("PRAGMA table_info(notas)")}
        if "trechos_md5" not in cols:
            con.execute("ALTER TABLE notas ADD COLUMN trechos_md5 TEXT")

    # Atualiza versão
    definir_versao(con, ESQUEMA_VERSAO)
    con.commit()


def inicializar_meta(
    con: sqlite3.Connection,
    modelo_emb: str,
    dim_emb: int,
) -> None:
    """Inicializa/atualiza meta com info do modelo de embeddings."""
    meta_set(con, "modelo_emb", modelo_emb)
    meta_set(con, "dim_emb", str(dim_emb))
    meta_set(con, "versao_vetores", "0")
    con.commit()


def verificar_modelo_emb(con: sqlite3.Connection, modelo_emb: str, dim_emb: int) -> bool:
    """Verifica se o modelo/dimensão corresponde ao guardado (recusa misturas)."""
    m = meta_get(con, "modelo_emb")
    d = meta_get(con, "dim_emb")
    if m is None or d is None:
        return True  # primeira vez
    return m == modelo_emb and int(d) == dim_emb


def incrementar_versao_vetores(con: sqlite3.Connection) -> int:
    """Incrementa versao_vetores (invalida cache do IndiceVetorial)."""
    v = int(meta_get(con, "versao_vetores") or "0") + 1
    meta_set(con, "versao_vetores", str(v))
    con.commit()
    return v


def obter_versao_vetores(con: sqlite3.Connection) -> int:
    return int(meta_get(con, "versao_vetores") or "0")