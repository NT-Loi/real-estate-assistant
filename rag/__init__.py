"""
RAG Query Engine for the Vietnamese Real Estate Assistant.

Modules:
    query_parser – Rule-based intent detection and filter extraction
    retriever    – Semantic search across ChromaDB collections
    prompts      – Vietnamese prompt templates for LLM generation
    llm          – Google Gemini wrapper with graceful fallback
    chain        – End-to-end RAG pipeline (parser → retriever → LLM)
"""
from rag.query_parser import QueryParser, ParsedQuery
from rag.retriever import Retriever, RetrievedDocument
from rag.prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE
from rag.llm import LLMClient
from rag.chain import RAGChain, RAGResponse

__all__ = [
    "QueryParser",
    "ParsedQuery",
    "Retriever",
    "RetrievedDocument",
    "LLMClient",
    "RAGChain",
    "RAGResponse",
    "SYSTEM_PROMPT",
    "QA_PROMPT_TEMPLATE",
]
