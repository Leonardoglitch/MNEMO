"""Módulo de permissões do Vault (Fase 1).

Regra central: o agente nunca toca em ficheiros diretamente. Todas as
ferramentas (search, read_note, create_note, ...) chamam `verificar()` antes
de qualquer operação. Se o caminho estiver fora das pastas permitidas,
levanta PermissaoNegada.

Requer Python 3.9+ (usa Path.is_relative_to).

Estrutura esperada:
    meu-vault/
    ├── notas/                 <- só aqui o agente pode ler/escrever
    └── .vault/config.json     <- fora de "notas", logo o agente não lhe chega

Exemplo de config.json:
    {
      "todas_as_pastas": false,
      "pastas_permitidas": ["projetos", "diario/2026"]
    }
"""

import json
from pathlib import Path


class PermissaoNegada(Exception):
    """A operação tenta aceder a um caminho que a IA não pode usar."""


class Permissoes:
    def __init__(self, raiz_vault):
        self.raiz = Path(raiz_vault).resolve()
        self.notas = (self.raiz / "notas").resolve()
        self.config_path = self.raiz / ".vault" / "config.json"
        self.todas_as_pastas = False
        self.pastas_permitidas = []
        self.carregar()

    def carregar(self):
        """Lê o config.json. Sem ficheiro, por defeito nada é permitido."""
        if not self.config_path.exists():
            return
        dados = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.todas_as_pastas = bool(dados.get("todas_as_pastas", False))
        self.pastas_permitidas = list(dados.get("pastas_permitidas", []))

    def _resolver(self, caminho_relativo):
        """Converte para caminho absoluto e garante que fica dentro de notas/.

        resolve() elimina '../' e segue links simbólicos, por isso truques
        como '../../etc/passwd' ou um symlink para fora acabam bloqueados.
        """
        alvo = (self.notas / caminho_relativo).resolve()
        if not alvo.is_relative_to(self.notas):
            raise PermissaoNegada(f"Caminho fora do vault: {caminho_relativo}")
        return alvo

    def verificar(self, caminho_relativo):
        """Devolve o caminho absoluto se for permitido; senão levanta erro."""
        alvo = self._resolver(caminho_relativo)
        if self.todas_as_pastas:
            return alvo
        for pasta in self.pastas_permitidas:
            permitida = self._resolver(pasta)
            if alvo == permitida or alvo.is_relative_to(permitida):
                return alvo
        raise PermissaoNegada(f"Sem permissão para: {caminho_relativo}")

    def raizes_permitidas(self):
        """Pastas que o indexador pode percorrer (e enviar à API de embeddings)."""
        if self.todas_as_pastas:
            return [self.notas]
        return [self._resolver(p) for p in self.pastas_permitidas]


# --- Exemplo de uma ferramenta do agente a usar o módulo ---

def read_note(perms, caminho_relativo):
    alvo = perms.verificar(caminho_relativo)
    return alvo.read_text(encoding="utf-8")
