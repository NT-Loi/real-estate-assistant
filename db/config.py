"""
Central configuration for the RAG database layer.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data-Loi"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_TOKENS = 2048
EMBEDDING_NORMALIZE = True
QDRANT_DISTANCE = "DOT"

# ---------------------------------------------------------------------------
# Qdrant connection
# ---------------------------------------------------------------------------
# For local Docker, keep QDRANT_URL empty and use host/port.
# For Qdrant Cloud, set QDRANT_URL=https://... and QDRANT_API_KEY=...
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost").strip()
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in {"1", "true", "yes", "on"}
RERANKER_MAX_LENGTH = 2304
RERANKER_CANDIDATES = 60
RERANKER_KEEP = 10

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Keep chunk-level evidence by default. Turning this on keeps only the best
# chunk per source record, which can hide useful context from long listings,
# articles, or social threads.
RETRIEVAL_COLLAPSE_BY_SOURCE = os.getenv("RETRIEVAL_COLLAPSE_BY_SOURCE", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RETRIEVAL_MIN_SEMANTIC_LIMIT = int(os.getenv("RETRIEVAL_MIN_SEMANTIC_LIMIT", "10"))
RETRIEVAL_MIN_HYBRID_LIMIT = int(os.getenv("RETRIEVAL_MIN_HYBRID_LIMIT", "12"))

# ---------------------------------------------------------------------------
# Qdrant collection names
# ---------------------------------------------------------------------------
COLLECTION_LISTINGS = "listings"
COLLECTION_PROJECTS = "projects"
COLLECTION_ARTICLES = "articles"
COLLECTION_SOCIAL    = "social_neighborhood"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Vietnamese_Embedding_v2 supports 2048 tokens. We stay below that so the title,
# source prefix, and prompt-side context do not push chunks into truncation.
CHUNK_TARGET_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 120
ARTICLE_CHUNK_SIZE = CHUNK_TARGET_TOKENS
ARTICLE_CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS
SOCIAL_COMMENT_BATCH_TOKENS = 700

# ---------------------------------------------------------------------------
# Gemini / LLM
# ---------------------------------------------------------------------------
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# gemini-2.5-flash — stronger tool planning for multi-step ReAct queries
# gemini-2.0-flash — fast, cheap fallback for structured extraction + RAG answers
