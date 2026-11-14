"""Deteção de alterações do Retain (trechos_md5) e repetição de embeddings pendentes."""

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from mnemo.armazenamento import Armazenamento
from mnemo.indexador import Indexador
from mnemo.memoria import FakeEmbeddings, Retain
from mnemo.permissoes import Permissoes


class ProviderInstavel:
    """FakeEmbeddings que falha nas primeiras `falhas` chamadas a embed()."""

    def __init__(self, falhas: int) -> None:
        self._real = FakeEmbeddings(dim=32)
        self.falhas = falhas
        self.modelo = self._real.modelo
        self.dim = self._real.dim

    def embed(self, textos: List[str], input_type: str = "passage", dimensions: Any = None) -> np.ndarray:
        if self.falhas > 0:
            self.falhas -= 1
            raise RuntimeError("API indisponível")
        return self._real.embed(textos, input_type=input_type, dimensions=dimensions)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    raiz = tmp_path / "v"
    (raiz / "notas").mkdir(parents=True)
    (raiz / ".vault").mkdir()
    (raiz / ".vault" / "config.json").write_text('{"todas_as_pastas": true}', encoding="utf-8")
    (raiz / "notas" / "a.md").write_text("# A\n\nTexto sobre café.\n", encoding="utf-8")
    return raiz


def montar(vault: Path, provider: Any) -> Retain:
    perms = Permissoes(str(vault))
    indexador = Indexador(perms, caminho_db=":memory:")
    indexador.reindexar_tudo()  # o Indexador já grava notas.md5 antes de o Retain correr
    return Retain(perms, Armazenamento(perms), indexador, provider, provider.modelo, provider.dim)


def n_trechos(r: Retain) -> int:
    return int(r.indexador.con.execute("SELECT COUNT(*) FROM trechos").fetchone()[0])


def test_notas_ja_indexadas_pelo_indexador_recebem_trechos(vault: Path) -> None:
    r = montar(vault, FakeEmbeddings(dim=32))
    rel = r.retain_vault()
    assert rel.novas == 1 and rel.inalteradas == 0
    assert n_trechos(r) >= 1


def test_segunda_passagem_nao_faz_nada_e_edicao_e_detetada(vault: Path) -> None:
    r = montar(vault, FakeEmbeddings(dim=32))
    r.retain_vault()
    assert r.retain_vault().inalteradas == 1

    (vault / "notas" / "a.md").write_text("# A\n\nTexto novo sobre chá.\n", encoding="utf-8")
    rel = r.retain_vault()
    assert rel.atualizadas == 1 and rel.embeddings_novos >= 1


def test_embeddings_pendentes_sao_repetidos_no_retain_seguinte(vault: Path) -> None:
    r = montar(vault, ProviderInstavel(falhas=1))
    rel = r.retain_vault()
    assert rel.embeddings_pendentes == ["a.md"] and rel.erros
    assert r.indexador.con.execute("SELECT COUNT(*) FROM vetores").fetchone()[0] == 0

    rel2 = r.retain_vault()  # a API voltou: a nota não pode ficar "inalterada" sem vetores
    assert rel2.embeddings_pendentes == [] and rel2.embeddings_novos >= 1
    assert r.indexador.con.execute("SELECT COUNT(*) FROM vetores").fetchone()[0] >= 1
    assert r.indexador.con.execute("SELECT trechos_md5 FROM notas").fetchone()[0] is not None


def test_apagar_nota_remove_trechos_e_fts_sem_orfaos(vault: Path) -> None:
    r = montar(vault, FakeEmbeddings(dim=32))
    r.retain_vault()
    (vault / "notas" / "a.md").unlink()
    rel = r.retain_vault()
    assert rel.removidas == 1
    con = r.indexador.con
    assert con.execute("SELECT COUNT(*) FROM trechos").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM trechos_fts").fetchone()[0] == 0
