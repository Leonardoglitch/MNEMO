"""Módulo de armazenamento do Mnemo (Fase 1).

Lê, cria, edita e "apaga" notas .md. Todas as operações passam primeiro por
`Permissoes.verificar()`. Regras de segurança:
  - antes de sobrescrever ou acrescentar, guarda uma cópia em .backups/
  - "apagar" move a nota para .lixo/ (nunca elimina de vez)
  - a escrita é atómica: grava num ficheiro temporário e só depois substitui
  - se receber um Indexador, mantém o índice de pesquisa sempre atualizado

.backups/ e .lixo/ ficam fora de notas/, por isso o agente não lhes acede.
"""

import os
import shutil
from datetime import datetime

from .permissoes import Permissoes, PermissaoNegada  # noqa: F401


def _carimbo():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class Armazenamento:
    def __init__(self, perms: Permissoes, indexador=None):
        self.perms = perms
        self.indexador = indexador
        self.backups = perms.raiz / ".backups"
        self.lixo = perms.raiz / ".lixo"

    # --- helpers ---

    def _nota_existente(self, caminho):
        alvo = self.perms.verificar(caminho)
        if not alvo.is_file():
            raise FileNotFoundError(f"Nota não existe: {caminho}")
        return alvo

    def _copiar_para(self, alvo, pasta_base, mover=False):
        """Copia (ou move) a nota para pasta_base, mantendo a estrutura de pastas."""
        rel = alvo.relative_to(self.perms.notas)
        destino = pasta_base / rel.parent / f"{alvo.stem}.{_carimbo()}{alvo.suffix}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        if mover:
            shutil.move(str(alvo), str(destino))
        else:
            shutil.copy2(alvo, destino)
        return destino

    def _indexar(self, caminho):
        if self.indexador:
            self.indexador.atualizar_nota(caminho)

    @staticmethod
    def _escrever_atomico(alvo, conteudo):
        tmp = alvo.with_name(alvo.name + ".tmp")
        tmp.write_text(conteudo, encoding="utf-8")
        os.replace(tmp, alvo)

    # --- operações ---

    def ler(self, caminho):
        return self._nota_existente(caminho).read_text(encoding="utf-8")

    def criar(self, caminho, conteudo=""):
        alvo = self.perms.verificar(caminho)
        if alvo.suffix != ".md":
            raise ValueError("Só são permitidas notas .md")
        if alvo.exists():
            raise FileExistsError(f"A nota já existe: {caminho}")
        alvo.parent.mkdir(parents=True, exist_ok=True)
        self._escrever_atomico(alvo, conteudo)
        self._indexar(caminho)

    def editar(self, caminho, novo_conteudo):
        """Substitui o conteúdo (com backup da versão anterior)."""
        alvo = self._nota_existente(caminho)
        self._copiar_para(alvo, self.backups)
        self._escrever_atomico(alvo, novo_conteudo)
        self._indexar(caminho)

    def acrescentar(self, caminho, texto):
        """Acrescenta texto ao fim da nota (com backup)."""
        alvo = self._nota_existente(caminho)
        atual = alvo.read_text(encoding="utf-8")
        self._copiar_para(alvo, self.backups)
        separador = "" if atual.endswith("\n") or not atual else "\n"
        self._escrever_atomico(alvo, atual + separador + texto)
        self._indexar(caminho)

    def apagar(self, caminho):
        """Move a nota para .lixo/ em vez de a eliminar."""
        alvo = self._nota_existente(caminho)
        destino = self._copiar_para(alvo, self.lixo, mover=True)
        if self.indexador:
            self.indexador.remover_nota(caminho)
        return destino
