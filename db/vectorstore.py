"""
VectorStore — Unified Qdrant + PostgreSQL manager for the real estate RAG system.

Manages four collections/tables:
    - listings              (nhà đất bán / cho thuê)
    - projects              (dự án BĐS)
    - articles              (tin tức + wiki)
    - social_neighborhood   (YouTube + TikTok + VOZ forums)
"""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from db.postgres_client import PostgresClient
from db.qdrant_client import QdrantClientWrapper
from db.embedder import Embedder

log = logging.getLogger("bds_vectorstore")


class VectorStore:
    """Manages transactional relational and semantic vector storage layers."""

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder or Embedder()
        
        # Connect to our new production database services
        self.pg = PostgresClient()
        self.qdrant = QdrantClientWrapper()

    # ----- Add documents -----
    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        records: list[dict] | None = None,
        batch_size: int = 100,
    ):
        """
        Ingest records into both PostgreSQL and Qdrant in parallel.

        Args:
            collection_name: 'listings', 'projects', 'articles', 'social_neighborhood'
            ids: unique document chunk IDs
            documents: text content to embed
            metadatas: metadata dicts for vector filtering
            records: list of original raw crawled dictionaries (for PostgreSQL)
            batch_size: number of documents to process at once
        """
        total = len(documents)
        log.info(f"Adding {total} items to Qdrant + PostgreSQL in collection '{collection_name}'")

        # 1. Store structured raw payloads in PostgreSQL. A source record may
        # produce multiple vector chunks, so de-duplicate by relational ID.
        if records:
            log.info(f"Committing {len(records)} structured records to PostgreSQL '{collection_name}'")
            try:
                seen_record_ids: set[str] = set()
                for idx, r in enumerate(records):
                    # Attach ID dynamically so relational rows match vector points
                    if "id" not in r:
                        r["id"] = ids[idx] if idx < len(ids) else ids[0]

                    record_id = str(r.get("id") or "")
                    if record_id and record_id in seen_record_ids:
                        continue
                    seen_record_ids.add(record_id)

                    if collection_name == "listings":
                        self.pg.upsert_listing(r)
                    elif collection_name == "projects":
                        self.pg.upsert_project(r)
                    elif collection_name == "articles":
                        self.pg.upsert_article(r)
                    elif collection_name == "social_neighborhood":
                        self.pg.upsert_social_neighborhood(r)
            except Exception as e:
                log.error(f"PostgreSQL commit failed: {e}")
                # We still proceed to write vectors so that system is partially functional

        # 2. Store dense vectors in Qdrant
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_ids = ids[start:end]
            batch_meta = metadatas[start:end]

            embeddings = self._embedder.embed(batch_docs)

            self.qdrant.add_documents(
                collection_name=collection_name,
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=embeddings,
            )
            log.info(f"  Qdrant Ingested: {end}/{total}")

    # ----- Search -----
    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Semantic search backed by high-performance Qdrant similarity search.

        Args:
            collection_name: listings / projects / articles / social_neighborhood
            query: natural language query
            n_results: limit
            where: payload filters matching Qdrant schema

        Returns:
            Formatted dict compatible with original retriever expectation:
            {
                "ids": [[id1, id2...]],
                "documents": [[doc1, doc2...]],
                "metadatas": [[meta1, meta2...]],
                "distances": [[score1, score2...]]
            }
        """
        raw = self._embedder.embed(query)
        # embed() always returns list-of-lists; unwrap to flat vector for single query
        query_vector = raw[0] if raw and isinstance(raw[0], list) else raw
        hits = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            n_results=n_results,
            payload_filter=where
        )
        
        # Convert Qdrant format to match the ChromaDB client expectation for backward compatibility
        res = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        for hit in hits:
            res["ids"][0].append(hit["id"])
            res["documents"][0].append(hit["document"])
            res["metadatas"][0].append(hit["metadata"])
            res["distances"][0].append(1.0 - hit["score"])  # convert similarity score to distance metric
            
        return res

    def search_multi(
        self,
        query: str,
        collection_names: Optional[list[str]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict[str, dict]:
        """Search across multiple Qdrant collections."""
        if collection_names is None:
            collection_names = ["listings", "projects", "articles", "social_neighborhood"]

        results = {}
        for name in collection_names:
            try:
                results[name] = self.search(
                    collection_name=name,
                    query=query,
                    n_results=n_results,
                    where=where,
                )
            except Exception as e:
                log.warning(f"Search failed on collection '{name}': {e}")
        return results

    # ----- Stats -----
    def stats(self) -> dict[str, int]:
        """Return point count stats from Qdrant."""
        return self.qdrant.stats()

    # ----- Reset -----
    def reset_collection(self, collection_name: str):
        """Reset Qdrant collection and clean PostgreSQL table."""
        self.qdrant.reset_collection(collection_name)
        
        valid_tables = {
            "listings": "listings",
            "projects": "projects",
            "articles": "articles",
            "social_neighborhood": "social_neighborhood"
        }
        if collection_name in valid_tables:
            t = valid_tables[collection_name]
            with self.pg.get_cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {t} CASCADE;")
            log.info(f"Truncated PostgreSQL table '{t}'")
