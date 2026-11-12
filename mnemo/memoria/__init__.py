"""Subsistema de memória do Mnemo (Retain / Recall / Reflect).

Componentes:
- modelos: dataclasses (Trecho, Resultado, Proposta, ContextoReflect, RelatorioRetain)
- chunker: função pura para partir notas em trechos
- tokens_texto: estimativa de tokens (heurística chars/token)
- embeddings: providers (NvidiaEmbeddings, FakeEmbeddings) + cache de consultas
- vetores: IndiceVetorial (numpy, cosseno via produto interno)
- esquema: migrações, tabela meta, versionamento
- retain: sincronização vault, deteção alterações, embeddings incrementais
- recall: busca híbrida (BM25 + embeddings), RRF, diversidade, budget
- reflect: construção de contexto, verificação de citações
"""

from mnemo.memoria.modelos import (
    Trecho,
    Resultado,
    Proposta,
    ContextoReflect,
    RelatorioRetain,
)
from mnemo.memoria.chunker import chunker, estimar_tokens as chunker_estimar_tokens
from mnemo.memoria.tokens_texto import estimar_tokens, calibrar_chars_por_token
from mnemo.memoria.embeddings import (
    EmbeddingProvider,
    NvidiaEmbeddings,
    FakeEmbeddings,
    ConsultaCache,
)
from mnemo.memoria.vetores import IndiceVetorial
from mnemo.memoria.esquema import (
    ESQUEMA_VERSAO,
    migrar,
    inicializar_meta,
    verificar_modelo_emb,
    incrementar_versao_vetores,
    obter_versao_vetores,
    meta_get,
    meta_set,
)
from mnemo.memoria.retain import Retain
from mnemo.memoria.recall import Recall
from mnemo.memoria.reflect import Reflect, verificar_citacoes

__all__ = [
    # modelos
    "Trecho",
    "Resultado",
    "Proposta",
    "ContextoReflect",
    "RelatorioRetain",
    # chunker
    "chunker",
    "chunker_estimar_tokens",
    # tokens_texto
    "estimar_tokens",
    "calibrar_chars_por_token",
    # embeddings
    "EmbeddingProvider",
    "NvidiaEmbeddings",
    "FakeEmbeddings",
    "ConsultaCache",
    # vetores
    "IndiceVetorial",
    # esquema
    "ESQUEMA_VERSAO",
    "migrar",
    "inicializar_meta",
    "verificar_modelo_emb",
    "incrementar_versao_vetores",
    "obter_versao_vetores",
    "meta_get",
    "meta_set",
    # retain/recall/reflect
    "Retain",
    "Recall",
    "Reflect",
    "verificar_citacoes",
]