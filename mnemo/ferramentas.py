"""Ferramentas do agente para interagir com o Vault (Fase 1).

Expõe quatro operações seguras: search, read_note, create_note, append_to_note.
Todas respeitam as permissões configuradas no .vault/config.json.
"""

from typing import Any, Dict, List

from .permissoes import Permissoes, PermissaoNegada
from .armazenamento import Armazenamento
from .indexador import Indexador


class FerramentasVault:
    DESCRICOES: List[Dict[str, Any]] = [
        {
            "name": "search",
            "description": "Pesquisa notas no índice por palavras-chave. Devolve até 10 resultados com título, excerto e relevância.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "Termos de pesquisa (ex.: 'orçamento projeto')"}
                },
                "required": ["consulta"],
            },
        },
        {
            "name": "read_note",
            "description": "Lê o conteúdo completo de uma nota .md.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "caminho": {"type": "string", "description": "Caminho relativo dentro de notas/ (ex.: 'projetos/tarefas.md')"}
                },
                "required": ["caminho"],
            },
        },
        {
            "name": "create_note",
            "description": "Cria uma nova nota .md com o conteúdo indicado.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "caminho": {"type": "string", "description": "Caminho relativo dentro de notas/ (ex.: 'projetos/nova.md')"},
                    "conteudo": {"type": "string", "description": "Texto inicial da nota (pode incluir frontmatter)"}
                },
                "required": ["caminho", "conteudo"],
            },
        },
        {
            "name": "append_to_note",
            "description": "Acrescenta texto ao final de uma nota existente.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "caminho": {"type": "string", "description": "Caminho relativo dentro de notas/ (ex.: 'projetos/tarefas.md')"},
                    "texto": {"type": "string", "description": "Texto a adicionar"}
                },
                "required": ["caminho", "texto"],
            },
        },
    ]

    def __init__(self, raiz_vault: str):
        self.perms = Permissoes(raiz_vault)
        self.indexador = Indexador(self.perms)
        self.indexador.reindexar_tudo()
        self.armazenamento = Armazenamento(self.perms, indexador=self.indexador)

    # --- context manager ---
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.indexador.fechar()

    # --- API pública ---
    def executar(self, nome: str, argumentos: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if nome == "search":
                return self._search(argumentos)
            if nome == "read_note":
                return self._read_note(argumentos)
            if nome == "create_note":
                return self._create_note(argumentos)
            if nome == "append_to_note":
                return self._append_to_note(argumentos)
            return {"ok": False, "erro": f"Ferramenta desconhecida: {nome}"}
        except PermissaoNegada as e:
            return {"ok": False, "erro": str(e)}
        except FileExistsError as e:
            return {"ok": False, "erro": "Nota já existe.", "sugestao": "append_to_note"}
        except FileNotFoundError as e:
            return {"ok": False, "erro": "Nota não encontrada.", "sugestao": "create_note"}
        except ValueError as e:
            # only extension check raises ValueError with specific message
            return {"ok": False, "erro": "Só são permitidas notas .md", "sugestao": "use caminho terminado em .md"}
        except Exception as e:  # pylint: disable=broad-except
            return {"ok": False, "erro": f"Erro interno: {e}"}

    # --- implementações ---
    def _search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        consulta = args.get("consulta", "")
        resultados = self.indexador.pesquisar(consulta)
        return {"ok": True, "resultados": resultados}

    def _read_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        caminho = args["caminho"]
        conteudo = self.armazenamento.ler(caminho)
        return {"ok": True, "conteudo": conteudo}

    def _create_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        caminho = args["caminho"]
        conteudo = args.get("conteudo", "")
        self.armazenamento.criar(caminho, conteudo)
        return {"ok": True, "caminho": caminho}

    def _append_to_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        caminho = args["caminho"]
        texto = args.get("texto", "")
        self.armazenamento.acrescentar(caminho, texto)
        return {"ok": True, "caminho": caminho}