"""
Central configuration for the RAG database layer.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# ChromaDB collection names
# ---------------------------------------------------------------------------
COLLECTION_LISTINGS = "listings"
COLLECTION_PROJECTS = "projects"
COLLECTION_ARTICLES = "articles"
COLLECTION_SOCIAL    = "social_neighborhood"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Articles: 800 chars per chunk (up from 500) — Vietnamese BDS analysis
# articles average 1500-3000 chars per topic section. Larger chunks keep
# more context per embedding, reducing retrieval fragmentation.
ARTICLE_CHUNK_SIZE    = 800
ARTICLE_CHUNK_OVERLAP = 100
MAX_LISTING_TEXT_LENGTH = 2000  # chars — listings are single-chunk
MAX_PROJECT_TEXT_LENGTH = 3000  # chars — projects may need 1-2 chunks

# ---------------------------------------------------------------------------
# Gemini / LLM
# ---------------------------------------------------------------------------
import os
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# gemini-2.0-flash  — fast, cheap, great for structured extraction + RAG answers
# gemini-1.5-pro    — higher quality for complex reasoning (higher cost)
