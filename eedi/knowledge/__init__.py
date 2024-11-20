from eedi import (
    KNOWLEDGE_TYPE_GENK,
    KNOWLEDGE_TYPE_NONE,
    KNOWLEDGE_TYPE_RAG,
    KNOWLEDGE_TYPE_TOT,
)
from eedi.knowledge.genk import enhance_with_knowledge as enhancer_genk
from eedi.knowledge.none import enhance_with_knowledge as enhancer_none
from eedi.knowledge.rag import enhance_with_knowledge as enhancer_rag
from eedi.knowledge.tot import enhance_with_knowledge as enhancer_tot

KNOWLEDGE_ENHANCER_MAP = {
    KNOWLEDGE_TYPE_NONE: enhancer_none,
    KNOWLEDGE_TYPE_GENK: enhancer_genk,
    KNOWLEDGE_TYPE_TOT: enhancer_tot,
    KNOWLEDGE_TYPE_RAG: enhancer_rag,
}
