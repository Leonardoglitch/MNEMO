"""Índice de pesquisa do Mnemo (Fase 1).

Guarda num SQLite (.vault/index.db) o título e o texto de cada nota, com
pesquisa de texto (FTS5). O índice é DERIVADO: os ficheiros .md são a fonte de
verdade, por isso podes apagar o index.db e chamar `reindexar_tudo()`.

Privacidade: só entram no índice notas de pastas permitidas, e a pesquisa
volta a verificar as permissões em cada resultado. Assim, uma pasta que deixe
de ser permitida deixa logo de aparecer, mesmo antes de reindexar.
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .permissoes import Permissoes, PermissaoNegada

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS notas (
    id            INTEGER PRIMARY KEY,
    caminho       TEXT NOT NULL UNIQUE,   -- relativo a notas/, com '/'
    titulo        TEXT NOT NULL,
    modificado_ns INTEGER NOT NULL,
    tags          TEXT DEFAULT '[]',       -- JSON array: '["tag1","tag2"]'
    md5           TEXT                    -- hash do conteúdo para deteção de alterações
);
CREATE TABLE IF NOT EXISTS nota_tags (
    nota_id INTEGER REFERENCES notas(id) ON DELETE CASCADE,
    tag       TEXT NOT NULL,
    PRIMARY KEY (nota_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_nota_tags_tag ON nota_tags(tag);
CREATE VIRTUAL TABLE IF NOT EXISTS notas_fts USING fts5(
    titulo, conteudo,
    tokenize = 'unicode61 remove_diacritics 2'   -- "orcamento" acha "orçamento"
);

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


def separar_frontmatter(conteudo: str) -> Tuple[List[str], str]:
    """Separa o bloco '---' do início. Devolve (linhas_do_frontmatter, corpo)."""
    linhas = conteudo.splitlines()
    if linhas and linhas[0].strip() == "---":
        for i in range(1, len(linhas)):
            if linhas[i].strip() == "---":
                return linhas[1:i], "\n".join(linhas[i + 1:])
    return [], conteudo


def extrair_titulo(conteudo: str, nome_ficheiro: str) -> str:
    """Título = 'title:' do frontmatter, senão o 1.º '# título', senão o nome."""
    frontmatter, corpo = separar_frontmatter(conteudo)
    for linha in frontmatter:
        if linha.lower().startswith("title:"):
            titulo = linha.split(":", 1)[1].strip().strip("'\"")
            if titulo:
                return titulo
    for linha in corpo.splitlines():
        if linha.startswith("# "):
            return linha[2:].strip() or nome_ficheiro
    return nome_ficheiro


def extrair_tags(conteudo: str) -> List[str]:
    """Extrai tags do frontmatter YAML.
    
    Suporta:
    - tags: [tag1, tag2, tag3]  (YAML list)
    - tags: tag1, tag2, tag3    (string CSV)
    - tags: "tag1, tag2"        (string quoted)
    
    Normaliza para lowercase, remove duplicatas, ordena.
    """
    frontmatter, _ = separar_frontmatter(conteudo)
    for linha in frontmatter:
        if linha.lower().startswith("tags:"):
            valor = linha.split(":", 1)[1].strip()
            if not valor:
                return []
            # Remove aspas se houver
            valor = valor.strip("'\"")
            # Tenta parsear como YAML list primeiro
            if valor.startswith("[") and valor.endswith("]"):
                try:
                    import json
                    tags = json.loads(valor.replace("'", '"'))
                except json.JSONDecodeError:
                    tags = [t.strip() for t in valor[1:-1].split(",")]
            else:
                # String CSV
                tags = [t.strip() for t in valor.split(",")]
            # Normaliza: lowercase, remove vazios, deduplica, ordena
            tags_norm = sorted({t.lower() for t in tags if t.strip()})
            return tags_norm
    return []


class Indexador:
    def __init__(
        self,
        perms: Permissoes,
        caminho_db: Optional[str] = None,
        *,
        check_same_thread: bool = True,
    ) -> None:
        self.perms = perms
        self.caminho_db = (
            Path(caminho_db) if caminho_db else perms.raiz / ".vault" / "index.db"
        )
        self.caminho_db.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: necessário na API (FastAPI corre handlers noutras threads).
        # Quem partilha a ligação entre threads deve serializar o acesso (ver Servicos.lock).
        self.con = sqlite3.connect(self.caminho_db, check_same_thread=check_same_thread)
        self.con.executescript(_ESQUEMA)
        self._migrar_schema()

    def _migrar_schema(self) -> None:
        """Adiciona colunas/tabelas em falta em bases de dados antigas."""
        # Adiciona coluna tags se não existir (migração v1 -> v2)
        cols = {row[1] for row in self.con.execute("PRAGMA table_info(notas)")}
        if "tags" not in cols:
            self.con.execute("ALTER TABLE notas ADD COLUMN tags TEXT DEFAULT '[]'")
        if "md5" not in cols:
            self.con.execute("ALTER TABLE notas ADD COLUMN md5 TEXT")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_notas_md5 ON notas(md5)")
        self.con.commit()

    # --- helpers internos ---

    def _rel(self, alvo: Path) -> str:
        return alvo.relative_to(self.perms.notas).as_posix()

    def _gravar(self, rel: str, alvo: Path) -> None:
        conteudo = alvo.read_text(encoding="utf-8-sig", errors="replace")
        titulo = extrair_titulo(conteudo, alvo.stem)
        tags = extrair_tags(conteudo)
        tags_json = json.dumps(tags, ensure_ascii=False)
        _, corpo = separar_frontmatter(conteudo)  # o frontmatter não polui os excertos
        mtime = alvo.stat().st_mtime_ns
        md5 = hashlib.md5(conteudo.encode("utf-8")).hexdigest()
        linha = self.con.execute(
            "SELECT id FROM notas WHERE caminho = ?", (rel,)
        ).fetchone()
        if linha:
            nota_id = linha[0]
            self.con.execute(
                "UPDATE notas SET titulo = ?, modificado_ns = ?, tags = ?, md5 = ? WHERE id = ?",
                (titulo, mtime, tags_json, md5, nota_id),
            )
            self.con.execute("DELETE FROM notas_fts WHERE rowid = ?", (nota_id,))
            # Atualiza nota_tags: remove antigas, insere novas
            self.con.execute("DELETE FROM nota_tags WHERE nota_id = ?", (nota_id,))
            for tag in tags:
                self.con.execute(
                    "INSERT INTO nota_tags (nota_id, tag) VALUES (?, ?)",
                    (nota_id, tag),
                )
        else:
            cur = self.con.execute(
                "INSERT INTO notas (caminho, titulo, modificado_ns, tags, md5) VALUES (?, ?, ?, ?, ?)",
                (rel, titulo, mtime, tags_json, md5),
            )
            nota_id = cur.lastrowid
            for tag in tags:
                self.con.execute(
                    "INSERT INTO nota_tags (nota_id, tag) VALUES (?, ?)",
                    (nota_id, tag),
                )
        self.con.execute(
            "INSERT INTO notas_fts (rowid, titulo, conteudo) VALUES (?, ?, ?)",
            (nota_id, titulo, corpo),
        )

    def _remover(self, rel: str) -> None:
        linha = self.con.execute(
            "SELECT id FROM notas WHERE caminho = ?", (rel,)
        ).fetchone()
        if linha:
            nota_id = linha[0]
            # Os trechos da memória não têm cascade efetivo (foreign_keys está desligado): sem isto
            # ficavam órfãos e uma nota nova podia herdar os trechos de outra ao reutilizar o id.
            if self.con.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'trechos'"
            ).fetchone():
                self.con.execute(
                    "DELETE FROM trechos_fts WHERE rowid IN (SELECT id FROM trechos WHERE nota_id = ?)",
                    (nota_id,),
                )
                self.con.execute("DELETE FROM trechos WHERE nota_id = ?", (nota_id,))
            self.con.execute("DELETE FROM nota_tags WHERE nota_id = ?", (nota_id,))
            self.con.execute("DELETE FROM notas_fts WHERE rowid = ?", (linha[0],))
            self.con.execute("DELETE FROM notas WHERE id = ?", (linha[0],))

    def _notas_validas(self) -> Dict[str, Path]:
        """Notas .md nas pastas permitidas: {caminho_relativo: caminho_absoluto}."""
        validas: Dict[str, Path] = {}
        for raiz in self.perms.raizes_permitidas():
            if not raiz.is_dir():
                continue
            for ficheiro in raiz.rglob("*.md"):
                rel = ficheiro.relative_to(self.perms.notas)
                if any(parte.startswith(".") for parte in rel.parts):
                    continue  # ignora pastas ocultas (.obsidian, .trash, ...)
                try:
                    alvo = self.perms.verificar(rel.as_posix())
                except PermissaoNegada:
                    continue  # p. ex. link simbólico que aponta para fora
                if alvo.is_file():
                    validas[self._rel(alvo)] = alvo
        return validas

    def ids_permitidos(self, tags: Optional[List[str]] = None, tipos: Optional[List[str]] = None) -> List[int]:
        """Devolve IDs de notas permitidas (filtro por pastas + tags + tipos).
        
        Usado pelo Recall para filtrar ANTES das pesquisas.
        """
        # Começa com todas as notas permitidas por pastas
        validas = self._notas_validas()
        if not validas:
            return []

        # Obtém IDs das notas válidas
        placeholders = ",".join("?" for _ in validas)
        rows = self.con.execute(
            f"SELECT id FROM notas WHERE caminho IN ({placeholders})",
            list(validas.keys()),
        ).fetchall()
        ids = [r[0] for r in rows]

        # Filtro adicional por tags
        if tags:
            tags_norm = [t.lower() for t in tags]
            placeholders = ",".join("?" for _ in tags_norm)
            query = f"""
                SELECT nota_id FROM nota_tags
                WHERE tag IN ({placeholders})
                GROUP BY nota_id
                HAVING COUNT(DISTINCT tag) = ?
            """
            params = list(tags_norm) + [len(tags_norm)]
            rows = self.con.execute(query, params).fetchall()
            tag_ids = {r[0] for r in rows}
            ids = [i for i in ids if i in tag_ids]

        # Filtro por tipos (frontmatter tipo:) - opcional
        if tipos:
            # TODO: implementar quando coluna tipo existir
            pass

        return ids

    # --- API pública ---

    def reindexar_tudo(self, forcar: bool = False) -> Dict[str, int]:
        """Sincroniza o índice com as pastas permitidas.

        Só relê os ficheiros cuja data de modificação mudou (ou todos, com
        forcar=True). Remove do índice o que foi apagado ou deixou de ser
        permitido. Serve também de verificação no arranque da aplicação.
        """
        validas = self._notas_validas()
        existentes = {
            caminho: mtime
            for caminho, mtime in self.con.execute(
                "SELECT caminho, modificado_ns FROM notas"
            )
        }
        resumo: Dict[str, int] = {"novas": 0, "atualizadas": 0, "removidas": 0, "inalteradas": 0}
        for rel, alvo in validas.items():
            if rel not in existentes:
                self._gravar(rel, alvo)
                resumo["novas"] += 1
            elif forcar or existentes[rel] != alvo.stat().st_mtime_ns:
                self._gravar(rel, alvo)
                resumo["atualizadas"] += 1
            else:
                resumo["inalteradas"] += 1
        for rel in set(existentes) - set(validas):
            self._remover(rel)
            resumo["removidas"] += 1
        self.con.commit()
        return resumo

    def atualizar_nota(self, caminho: str) -> None:
        """Reindexa uma nota (ou remove-a do índice se já não existir)."""
        alvo = self.perms.verificar(caminho)
        rel = self._rel(alvo)
        if alvo.is_file():
            self._gravar(rel, alvo)
        else:
            self._remover(rel)
        self.con.commit()

    def remover_nota(self, caminho: str) -> None:
        alvo = self.perms.verificar(caminho)
        self._remover(self._rel(alvo))
        self.con.commit()

    def pesquisar(self, consulta: str, limite: int = 10, tags: Optional[List[str]] = None, modo_tags: str = "all") -> List[Dict[str, object]]:
        """Devolve [{caminho, titulo, excerto, relevancia}], das mais relevantes.

        A consulta é limpa para só ter palavras (aspas, hífenes ou operadores
        que o modelo escreva não rebentam o FTS5). As palavras têm de aparecer
        todas na nota, e cada uma também casa como prefixo ("orça" acha "orçamento").

        Args:
            consulta: Texto de pesquisa (palavras-chave)
            limite: Máximo de resultados
            tags: Lista de tags para filtrar (opcional)
            modo_tags: "all" = nota deve ter TODAS as tags; "any" = nota deve ter PELO MENOS UMA
        """
        # Filtro por tags: obtém IDs de notas que correspondem
        tag_nota_ids: Optional[set[int]] = None
        if tags:
            tags_norm = [t.lower() for t in tags]
            placeholders = ",".join("?" for _ in tags_norm)
            if modo_tags == "all":
                # Nota deve ter TODAS as tags
                query = f"""
                    SELECT nota_id FROM nota_tags
                    WHERE tag IN ({placeholders})
                    GROUP BY nota_id
                    HAVING COUNT(DISTINCT tag) = ?
                """
                params: list[str | int] = tags_norm + [len(tags_norm)]
            else:  # "any"
                # Nota deve ter PELO MENOS UMA das tags
                query = f"""
                    SELECT DISTINCT nota_id FROM nota_tags
                    WHERE tag IN ({placeholders})
                """
                params = list(tags_norm)
            cursor = self.con.execute(query, params)
            tag_nota_ids = {row[0] for row in cursor}
            if not tag_nota_ids:
                return []  # nenhuma nota tem as tags pedidas

        termos = re.findall(r"\w+", consulta)
        if not termos and not tags:
            return []

        # Constrói a query de forma estruturada
        if termos:
            expressao = " ".join(f'"{t}"*' for t in termos)
            
            # Query base com FTS5
            if tag_nota_ids is not None:
                # Com filtro de tags + texto
                placeholders = ",".join("?" for _ in tag_nota_ids)
                sql = f"""
                    SELECT n.caminho, n.titulo,
                           snippet(notas_fts, 1, '[', ']', '…', 12),
                           bm25(notas_fts, 5.0, 1.0)
                    FROM notas_fts JOIN notas n ON n.id = notas_fts.rowid
                    WHERE notas_fts MATCH ?
                      AND n.id IN ({placeholders})
                    ORDER BY bm25(notas_fts, 5.0, 1.0)
                    LIMIT ?
                """
                sql_params = (expressao,) + tuple(tag_nota_ids) + (limite,)
            else:
                # Apenas busca textual
                sql = """
                    SELECT n.caminho, n.titulo,
                           snippet(notas_fts, 1, '[', ']', '…', 12),
                           bm25(notas_fts, 5.0, 1.0)
                    FROM notas_fts JOIN notas n ON n.id = notas_fts.rowid
                    WHERE notas_fts MATCH ?
                    ORDER BY bm25(notas_fts, 5.0, 1.0)
                    LIMIT ?
                """
                sql_params = (expressao, limite)
        else:
            # Apenas filtro por tags, sem consulta textual
            if tag_nota_ids is not None:
                placeholders = ",".join("?" for _ in tag_nota_ids)
                sql = f"""
                    SELECT n.caminho, n.titulo,
                           '' as excerto,
                           0.0 as pontuacao
                    FROM notas n
                    WHERE n.id IN ({placeholders})
                    LIMIT ?
                """
                sql_params = tuple(tag_nota_ids) + (limite,)
            else:
                return []  # sem termos e sem tags

        cursor = self.con.execute(sql, sql_params)
        resultados: List[Dict[str, object]] = []
        for caminho, titulo, excerto, pontuacao in cursor:
            try:
                self.perms.verificar(caminho)  # defesa em profundidade
            except PermissaoNegada:
                continue
            resultados.append(
                {
                    "caminho": caminho,
                    "titulo": titulo,
                    "excerto": excerto,
                    "relevancia": round(-pontuacao, 3),
                }
            )
            if len(resultados) >= limite:
                break
        return resultados

    def fechar(self) -> None:
        self.con.close()

    def __enter__(self) -> "Indexador":
        return self

    def __exit__(self, *_: object) -> None:
        self.fechar()
