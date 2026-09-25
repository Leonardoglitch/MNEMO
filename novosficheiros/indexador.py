"""Índice de pesquisa do Mnemo (Fase 1).

Guarda num SQLite (.vault/index.db) o título e o texto de cada nota, com
pesquisa de texto (FTS5). O índice é DERIVADO: os ficheiros .md são a fonte de
verdade, por isso podes apagar o index.db e chamar `reindexar_tudo()`.

Privacidade: só entram no índice notas de pastas permitidas, e a pesquisa
volta a verificar as permissões em cada resultado. Assim, uma pasta que deixe
de ser permitida deixa logo de aparecer, mesmo antes de reindexar.
"""

import re
import sqlite3
from pathlib import Path

from .permissoes import Permissoes, PermissaoNegada

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS notas (
    id            INTEGER PRIMARY KEY,
    caminho       TEXT NOT NULL UNIQUE,   -- relativo a notas/, com '/'
    titulo        TEXT NOT NULL,
    modificado_ns INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS notas_fts USING fts5(
    titulo, conteudo,
    tokenize = 'unicode61 remove_diacritics 2'   -- "orcamento" acha "orçamento"
);
"""


def separar_frontmatter(conteudo):
    """Separa o bloco '---' do início. Devolve (linhas_do_frontmatter, corpo)."""
    linhas = conteudo.splitlines()
    if linhas and linhas[0].strip() == "---":
        for i in range(1, len(linhas)):
            if linhas[i].strip() == "---":
                return linhas[1:i], "\n".join(linhas[i + 1:])
    return [], conteudo


def extrair_titulo(conteudo, nome_ficheiro):
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


class Indexador:
    def __init__(self, perms: Permissoes, caminho_db=None):
        self.perms = perms
        self.caminho_db = (
            Path(caminho_db) if caminho_db else perms.raiz / ".vault" / "index.db"
        )
        self.caminho_db.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.caminho_db)
        self.con.executescript(_ESQUEMA)

    # --- helpers internos ---

    def _rel(self, alvo):
        return alvo.relative_to(self.perms.notas).as_posix()

    def _gravar(self, rel, alvo):
        conteudo = alvo.read_text(encoding="utf-8-sig", errors="replace")
        titulo = extrair_titulo(conteudo, alvo.stem)
        _, corpo = separar_frontmatter(conteudo)  # o frontmatter não polui os excertos
        mtime = alvo.stat().st_mtime_ns
        linha = self.con.execute(
            "SELECT id FROM notas WHERE caminho = ?", (rel,)
        ).fetchone()
        if linha:
            nota_id = linha[0]
            self.con.execute(
                "UPDATE notas SET titulo = ?, modificado_ns = ? WHERE id = ?",
                (titulo, mtime, nota_id),
            )
            self.con.execute("DELETE FROM notas_fts WHERE rowid = ?", (nota_id,))
        else:
            cur = self.con.execute(
                "INSERT INTO notas (caminho, titulo, modificado_ns) VALUES (?, ?, ?)",
                (rel, titulo, mtime),
            )
            nota_id = cur.lastrowid
        self.con.execute(
            "INSERT INTO notas_fts (rowid, titulo, conteudo) VALUES (?, ?, ?)",
            (nota_id, titulo, corpo),
        )

    def _remover(self, rel):
        linha = self.con.execute(
            "SELECT id FROM notas WHERE caminho = ?", (rel,)
        ).fetchone()
        if linha:
            self.con.execute("DELETE FROM notas_fts WHERE rowid = ?", (linha[0],))
            self.con.execute("DELETE FROM notas WHERE id = ?", (linha[0],))

    def _notas_validas(self):
        """Notas .md nas pastas permitidas: {caminho_relativo: caminho_absoluto}."""
        validas = {}
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

    # --- API pública ---

    def reindexar_tudo(self, forcar=False):
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
        resumo = {"novas": 0, "atualizadas": 0, "removidas": 0, "inalteradas": 0}
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

    def atualizar_nota(self, caminho):
        """Reindexa uma nota (ou remove-a do índice se já não existir)."""
        alvo = self.perms.verificar(caminho)
        rel = self._rel(alvo)
        if alvo.is_file():
            self._gravar(rel, alvo)
        else:
            self._remover(rel)
        self.con.commit()

    def remover_nota(self, caminho):
        alvo = self.perms.verificar(caminho)
        self._remover(self._rel(alvo))
        self.con.commit()

    def pesquisar(self, consulta, limite=10):
        """Devolve [{caminho, titulo, excerto, relevancia}], das mais relevantes.

        A consulta é limpa para só ter palavras (aspas, hífenes ou operadores
        que o modelo escreva não rebentam o FTS5). As palavras têm de aparecer
        todas na nota, e cada uma também casa como prefixo ("orça" acha "orçamento").
        """
        termos = re.findall(r"\w+", consulta)
        if not termos:
            return []
        expressao = " ".join(f'"{t}"*' for t in termos)
        cursor = self.con.execute(
            """
            SELECT n.caminho, n.titulo,
                   snippet(notas_fts, 1, '[', ']', '…', 12),
                   bm25(notas_fts, 5.0, 1.0)
            FROM notas_fts JOIN notas n ON n.id = notas_fts.rowid
            WHERE notas_fts MATCH ?
            ORDER BY bm25(notas_fts, 5.0, 1.0)
            """,
            (expressao,),
        )
        resultados = []
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

    def fechar(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()
