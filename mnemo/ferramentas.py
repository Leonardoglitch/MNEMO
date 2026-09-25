"""Ferramentas do agente para interagir com o Vault (Fase 1).

Expõe quatro operações seguras: search, read_note, create_note, append_to_note.
Todas respeitam as permissões configuradas no .vault/config.json.
"""

from typing import Any, Dict, List, Tuple, Optional

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
            "description": "Cria uma nova nota .md com o conteúdo indicado. O caminho deve ser relativo a notas/ e dentro de pastas permitidas (ex.: 'projetos/minha-nota.md').",
            "input_schema": {
                "type": "object",
                "properties": {
                    "caminho": {"type": "string", "description": "Caminho relativo dentro de notas/ (ex.: 'projetos/nova.md'). Deve incluir pasta permitida e terminar em .md"},
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

    # --- validação ---
    def _validar_caminho_create(self, caminho: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Valida caminho para create_note antes de chamar armazenamento.
        Returns: (is_valid, erro_msg, sugestao_dict)
        """
        # 1. Deve ter extensão .md
        if not caminho.endswith(".md"):
            return False, "Só são permitidas notas .md", {
                "sugestao": "use caminho terminado em .md",
                "exemplo": "projetos/meu-arquivo.md"
            }

        # 2. Não pode ser caminho absoluto ou ter ..
        if caminho.startswith("/") or ".." in caminho:
            return False, "Caminho inválido", {
                "sugestao": "use caminho relativo (ex.: projetos/arquivo.md)"
            }

        # 3. Não pode ter espaços no nome do ficheiro
        nome_ficheiro = caminho.rsplit("/", 1)[-1]
        if " " in nome_ficheiro:
            return False, "Nome do ficheiro não pode conter espaços", {
                "sugestao": "use hífens ou underscores (ex.: meu-arquivo.md)",
                "exemplo": "projetos/meu-arquivo.md"
            }

        # 3. Se não tem pasta (ex.: "arquivo.md"), sugerir pastas permitidas
        if "/" not in caminho:
            pastas = [str(p.relative_to(self.perms.notas)) for p in self.perms.raizes_permitidas()]
            return False, "Caminho deve incluir pasta (ex.: projetos/arquivo.md)", {
                "sugestao": f"use uma pasta permitida: {', '.join(pastas)}",
                "pastas_permitidas": pastas,
                "exemplo": f"{pastas[0]}/{caminho}" if pastas else None
            }

        # 4. Verificar se pasta pai é permitida (sem criar arquivo)
        pasta_pai = caminho.rsplit("/", 1)[0]
        try:
            self.perms.verificar(pasta_pai + "/")
        except PermissaoNegada:
            pastas = [str(p.relative_to(self.perms.notas)) for p in self.perms.raizes_permitidas()]
            return False, f"Pasta '{pasta_pai}' não permitida", {
                "sugestao": f"use uma pasta permitida: {', '.join(pastas)}",
                "pastas_permitidas": pastas
            }

        return True, None, None

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

        # Validação prévia com sugestões
        valido, erro, sugestao = self._validar_caminho_create(caminho)
        if not valido:
            return {"ok": False, "erro": erro, **(sugestao or {})}

        self.armazenamento.criar(caminho, conteudo)
        return {"ok": True, "caminho": caminho}

    def _append_to_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        caminho = args["caminho"]
        texto = args.get("texto", "")
        self.armazenamento.acrescentar(caminho, texto)
        return {"ok": True, "caminho": caminho}