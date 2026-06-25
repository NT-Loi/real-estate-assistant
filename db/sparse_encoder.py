from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

from qdrant_client.models import SparseVector


_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_HASH_MOD = 2_147_483_647
BM25_K1 = 1.2
BM25_B = 0.75


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _HASH_MOD


def tokenize_sparse(text: str) -> list[str]:
    """Tokenize Vietnamese text for Qdrant sparse lexical retrieval.

    This is a lightweight, dependency-free lexical encoder. It keeps accented
    tokens and adds unaccented variants so user queries with or without
    Vietnamese diacritics can still match the same documents.
    """
    text = (text or "").lower()
    base_tokens = [tok for tok in _TOKEN_RE.findall(text) if len(tok) >= 2]
    tokens: list[str] = []

    for tok in base_tokens:
        tokens.append(tok)
        plain = _strip_accents(tok)
        if plain != tok:
            tokens.append(plain)

    for left, right in zip(base_tokens, base_tokens[1:]):
        if len(left) >= 2 and len(right) >= 2:
            bigram = f"{left}_{right}"
            tokens.append(bigram)
            plain = _strip_accents(bigram)
            if plain != bigram:
                tokens.append(plain)

    return tokens


def sparse_doc_len(text: str) -> int:
    return len(tokenize_sparse(text))


def encode_sparse(
    text: str,
    avg_doc_len: float | None = None,
    is_query: bool = False,
) -> SparseVector:
    """Encode text for Qdrant sparse BM25-style retrieval.

    Qdrant applies collection-level IDF when the sparse vector is configured
    with Modifier.IDF. We provide BM25-style term-frequency saturation here.
    Documents use length-normalized BM25 TF; queries use binary term weights.
    """
    tokens = tokenize_sparse(text)
    counts = Counter(tokens)
    if not counts:
        return SparseVector(indices=[], values=[])

    doc_len = max(len(tokens), 1)
    avg_doc_len = max(float(avg_doc_len or doc_len), 1.0)
    norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len))

    weighted: dict[int, float] = {}
    for token, count in counts.items():
        idx = _hash_token(token)
        if is_query:
            value = 1.0
        else:
            tf = float(count)
            value = (tf * (BM25_K1 + 1.0)) / (tf + norm)
        weighted[idx] = weighted.get(idx, 0.0) + value

    indices = sorted(weighted)
    values = [weighted[idx] for idx in indices]

    return SparseVector(indices=indices, values=values)
