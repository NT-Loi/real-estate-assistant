import logging
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient as RealQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range

log = logging.getLogger("bds_database.qdrant")

class QdrantClientWrapper:
    """Manages high-performance Qdrant vector database collections and similarity searches."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, vector_dim: int = 384):
        self.host = host
        self.port = port
        self.vector_dim = vector_dim
        
        log.info(f"Initializing Qdrant client at http://{self.host}:{self.port}")
        self.client = RealQdrantClient(url=f"http://{self.host}:{self.port}", check_compatibility=False)
        self.init_collections()

    def init_collections(self):
        """Ensure listings, projects, articles, and social_neighborhood collections exist with cosine indices."""
        collections = ["listings", "projects", "articles", "social_neighborhood"]

        
        try:
            existing_colls = [c.name for c in self.client.get_collections().collections]
            for c in collections:
                if c not in existing_colls:
                    log.info(f"Creating Qdrant collection: '{c}'")
                    self.client.create_collection(
                        collection_name=c,
                        vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                    )
            log.info("Qdrant collections verified/initialized successfully.")
        except Exception as e:
            log.error(f"Failed to connect or initialize Qdrant: {e}")
            raise e

    def reset_collection(self, collection_name: str):
        """Recreate and drop a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            log.info(f"Reset Qdrant collection '{collection_name}' successfully.")
        except Exception as e:
            log.warning(f"Error resetting collection {collection_name}: {e}")

    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """
        Upsert a batch of vectors with payload to Qdrant.
        
        Args:
            collection_name:listings / projects / articles / social_neighborhood
            ids: unique hashes
            documents: original raw text representation
            metadatas: payload fields
            embeddings: 384-dimensional dense vectors
        """
        points = []
        for i in range(len(ids)):
            # Combine the original text document inside the payload
            payload = {**metadatas[i], "document_text": documents[i]}
            
            # Qdrant requires string IDs to be valid UUIDs or integers.
            # However, Qdrant allows string UUID values, or we can use our 16-char hashes
            # if we pad them, or Qdrant now supports standard string IDs seamlessly in newer versions!
            # Since Qdrant natively accepts any unique string, our sha256 16-char hash is fully valid.
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload=payload
                )
            )
            
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            log.info(f"Successfully upserted {len(ids)} points to Qdrant '{collection_name}'")
        except Exception as e:
            log.error(f"Error upserting to Qdrant: {e}")
            raise e

    def _build_filter(self, payload_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Convert standard key-value filter dicts into formal Qdrant Filters."""
        if not payload_filter:
            return None
            
        must_conditions = self._build_conditions(payload_filter)
        if must_conditions:
            return Filter(must=must_conditions)
        return None

    def _build_conditions(self, payload_filter: Dict[str, Any]) -> List[FieldCondition]:
        """Build Qdrant must conditions from simple Chroma-like filters."""
        conditions = []
        for k, v in payload_filter.items():
            if k == "$and" and isinstance(v, list):
                for clause in v:
                    if isinstance(clause, dict):
                        conditions.extend(self._build_conditions(clause))
                continue

            if isinstance(v, dict):
                # Handle range filters (e.g. {"$lte": 3000, "$gte": 1000})
                gte = v["$gte"] if "$gte" in v else v.get("$gt")
                lte = v["$lte"] if "$lte" in v else v.get("$lt")
                eq = v.get("$eq")
                if eq is not None:
                    conditions.append(
                        FieldCondition(key=k, match=MatchValue(value=eq))
                    )
                    continue
                conditions.append(
                    FieldCondition(
                        key=k,
                        range=Range(
                            gte=float(gte) if gte is not None else None,
                            lte=float(lte) if lte is not None else None,
                        ),
                    )
                )
            else:
                # Standard value match
                conditions.append(
                    FieldCondition(
                        key=k,
                        match=MatchValue(value=v)
                    )
                )
        return conditions

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        n_results: int = 10,
        payload_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        Uses query_points() (qdrant-client ≥1.8) with fallback to legacy search().

        Returns:
            List of parsed dicts containing id, document, metadata, and score
        """
        qdrant_filter = self._build_filter(payload_filter)

        try:
            # qdrant-client ≥1.8: query_points replaces the deprecated search()
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=n_results,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            # Fallback for older qdrant-client builds that still have .search()
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=n_results,
                with_payload=True,
            )
        except Exception as e:
            log.warning(f"Qdrant search failed on collection '{collection_name}': {e}")
            return []

        output = []
        for hit in hits:
            payload = dict(hit.payload or {})
            doc = payload.pop("document_text", "")
            output.append({
                "id": hit.id,
                "document": doc,
                "metadata": payload,
                "score": hit.score,
            })
        return output

    def stats(self) -> Dict[str, int]:
        """Query count statistics from each collection."""
        res = {}
        for c in ["listings", "projects", "articles", "social_neighborhood"]:

            try:
                res[c] = self.client.get_collection(collection_name=c).points_count
            except Exception:
                res[c] = 0
        return res
