"""Módulo de armazenamento do Mnemo (Fase 1).

Lê, cria, edita e "apaga" notas .md. Todas as operações passam primeiro por
`Permissoes.verificar()`. Regras de segurança:
  - antes de sobrescrever ou acrescentar, guarda uma cópia em .backups/
  - "apagar" move a nota para .lixo/ (nunca elimina de vez)
  - a escrita é atómica: grava num ficheiro temporário e só depois substitui
  - se receber um Indexador, mantém o índice de pesquisa sempre atualizado

.backups/ e .lixo/ ficam fora de notas/, por isso o agente não lhes acede.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .permissoes import Permissoes, PermissaoNegada  # noqa: F401
from .indexador import Indexador  # noqa: F401


def _carimbo() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class Armazenamento:
    def __init__(self, perms: Permissoes, indexador: Optional[Indexador] = None) -> None:
        self.perms = perms
        self.indexador = indexador
        self.backups = perms.raiz / ".backups"
        self.lixo = perms.raiz / ".lixo"

    # --- helpers ---

    def _nota_existente(self, caminho: str) -> Path:
        alvo = self.perms.verificar(caminho)
        if not alvo.is_file():
            raise FileNotFoundError(f"Nota não existe: {caminho}")
        return alvo

    def _copiar_para(self, alvo: Path, pasta_base: Path, mover: bool = False) -> Path:
        """Copia (ou move) a nota para pasta_base, mantendo a estrutura de pastas."""
        rel = alvo.relative_to(self.perms.notas)
        destino = pasta_base / rel.parent / f"{alvo.stem}.{_carimbo()}{alvo.suffix}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        if mover:
            shutil.move(str(alvo), str(destino))
        else:
            shutil.copy2(alvo, destino)
        return destino

    def _indexar(self, caminho: str) -> None:
        if self.indexador:
            self.indexador.atualizar_nota(caminho)

    @staticmethod
    def _escrever_atomico(alvo: Path, conteudo: str) -> None:
        tmp = alvo.with_name(alvo.name + ".tmp")
        tmp.write_text(conteudo, encoding="utf-8")
        os.replace(tmp, alvo)

    # --- operações ---

    def ler(self, caminho: str) -> str:
        return self._nota_existente(caminho).read_text(encoding="utf-8")

    def criar(self, caminho: str, conteudo: str = "") -> None:
        alvo = self.perms.verificar(caminho)
        if alvo.suffix != ".md":
            raise ValueError("Só são permitidas notas .md")
        if alvo.exists():
            raise FileExistsError(f"A nota já existe: {caminho}")
        alvo.parent.mkdir(parents=True, exist_ok=True)
        self._escrever_atomico(alvo, conteudo)
        self._indexar(caminho)

    def editar(self, caminho: str, novo_conteudo: str) -> None:
        """Substitui o conteúdo (com backup da versão anterior)."""
        alvo = self._nota_existente(caminho)
        self._copiar_para(alvo, self.backups)
        self._escrever_atomico(alvo, novo_conteudo)
        self._indexar(caminho)

    def acrescentar(self, caminho: str, texto: str) -> None:
        """Acrescenta texto ao fim da nota (com backup)."""
        alvo = self._nota_existente(caminho)
        atual = alvo.read_text(encoding="utf-8")
        self._copiar_para(alvo, self.backups)
        separador = "" if atual.endswith("\n") or not atual else "\n"
        self._escrever_atomico(alvo, atual + separador + texto)
        self._indexar(caminho)

    def apagar(self, caminho: str) -> Path:
        """Move a nota para .lixo/ em vez de a eliminar."""
        alvo = self._nota_existente(caminho)
        destino = self._copiar_para(alvo, self.lixo, mover=True)
        if self.indexador:
            self.indexador.remover_nota(caminho)
        return destino

    # --- tags ---

    def _ler_frontmatter(self, alvo: Path) -> tuple[List[str], str]:
        """Lê frontmatter e corpo da nota. Retorna (linhas_frontmatter, corpo)."""
        linhas = alvo.read_text(encoding="utf-8").splitlines()
        if linhas and linhas[0].strip() == "---":
            for i in range(1, len(linhas)):
                if linhas[i].strip() == "---":
                    return linhas[1:i], "\n".join(linhas[i + 1:])
        return [], "\n".join(linhas)

    def _extrair_tags_do_frontmatter(self, frontmatter: List[str]) -> List[str]:
        """Extrai tags do frontmatter. Suporta YAML list e string CSV."""
        for linha in frontmatter:
            if linha.lower().startswith("tags:"):
                valor = linha.split(":", 1)[1].strip()
                if not valor:
                    return []
                valor = valor.strip("'\"")
                if valor.startswith("[") and valor.endswith("]"):
                    try:
                        tags = json.loads(valor.replace("'", '"'))
                    except json.JSONDecodeError:
                        tags = [t.strip() for t in valor[1:-1].split(",")]
                else:
                    tags = [t.strip() for t in valor.split(",")]
                return sorted({t.lower() for t in tags if t.strip()})
        return []

    def ler_tags(self, caminho: str) -> List[str]:
        """Lê tags da nota."""
        alvo = self._nota_existente(caminho)
        frontmatter, _ = self._ler_frontmatter(alvo)
        return self._extrair_tags_do_frontmatter(frontmatter)

    def _escrever_frontmatter(self, alvo: Path, frontmatter: List[str], corpo: str) -> None:
        """Escreve frontmatter + corpo de forma atómica."""
        linhas = ["---"] + frontmatter + ["---", corpo]
        if not corpo.endswith("\n"):
            linhas.append("")
        self._escrever_atomico(alvo, "\n".join(linhas))

    def escrever_tags(self, caminho: str, tags: List[str]) -> None:
        """Substitui todas as tags da nota (com backup)."""
        alvo = self._nota_existente(caminho)
        frontmatter, corpo = self._ler_frontmatter(alvo)
        
        # Normaliza tags
        tags_norm = sorted({t.lower().strip() for t in tags if t.strip()})
        tags_json = json.dumps(tags_norm, ensure_ascii=False)
        
        # Atualiza ou adiciona linha tags:
        tags_linha = f"tags: {tags_json}"
        nova_frontmatter = []
        tags_encontrada = False
        for linha in frontmatter:
            if linha.lower().startswith("tags:"):
                nova_frontmatter.append(tags_linha)
                tags_encontrada = True
            else:
                nova_frontmatter.append(linha)
        if not tags_encontrada:
            nova_frontmatter.append(tags_linha)
        
        # Backup e escrita atómica
        self._copiar_para(alvo, self.backups)
        self._escrever_frontmatter(alvo, nova_frontmatter, corpo)
        self._indexar(caminho)

    def adicionar_tag(self, caminho: str, tag: str) -> List[str]:
        """Adiciona uma tag à nota. Retorna lista atualizada."""
        tags = self.ler_tags(caminho)
        tag_norm = tag.lower().strip()
        if tag_norm and tag_norm not in tags:
            tags.append(tag_norm)
            self.escrever_tags(caminho, tags)
        return tags

    def remover_tag(self, caminho: str, tag: str) -> List[str]:
        """Remove uma tag da nota. Retorna lista atualizada."""
        tags = self.ler_tags(caminho)
        tag_norm = tag.lower().strip()
        if tag_norm in tags:
            tags.remove(tag_norm)
            self.escrever_tags(caminho, tags)
        return tags

    def substituir_tags(self, caminho: str, tags: List[str]) -> List[str]:
        """Substitui todas as tags da nota. Retorna lista aplicada."""
        self.escrever_tags(caminho, tags)
        return self.ler_tags(caminho)
