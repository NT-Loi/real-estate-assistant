"""
Retriever — Semantic search across ChromaDB collections with metadata filtering.

Searches relevant collections based on parsed query intent, applies structured
filters, and returns ranked results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from db.vectorstore import VectorStore

log = logging.getLogger("bds_retriever")


def _format_market_row(row: dict) -> str:
    """Format a market_snapshots row as readable Vietnamese text for LLM context."""
    listing_label = {
        "ban": "Nhà bán",
        "cho_thue": "Nhà cho thuê",
    }.get(row.get("listing_type", ""), row.get("listing_type", ""))

    loc_parts = [p for p in [row.get("district"), row.get("province")] if p]
    location = ", ".join(loc_parts) or "Toàn quốc"
    prop_type = row.get("property_type") or "Tất cả loại"
    period = row.get("period", "")

    def fmt_price(vnd):
        if vnd is None:
            return "N/A"
        b = vnd / 1_000_000_000
        return f"{b:.1f} tỷ" if b >= 1 else f"{vnd / 1_000_000:.0f} triệu"

    def fmt_per_m2(vnd):
        if vnd is None:
            return "N/A"
        return f"{vnd / 1_000_000:.1f} triệu/m²"

    return (
        f"[{period}] {listing_label} — {prop_type} tại {location}\n"
        f"  Số tin: {row.get('listing_count', 0):,}\n"
        f"  Giá trung vị: {fmt_price(row.get('median_price_vnd'))}\n"
        f"  Giá trung bình: {fmt_price(row.get('avg_price_vnd'))}\n"
        f"  Giá/m² trung vị: {fmt_per_m2(row.get('median_price_per_m2_vnd'))}\n"
        f"  Diện tích TB: {row.get('avg_area_m2') or 'N/A'} m²\n"
        f"  Khoảng giá: {fmt_price(row.get('min_price_vnd'))} – {fmt_price(row.get('max_price_vnd'))}"
    )


@dataclass
class RetrievedDocument:
    """A single retrieved document with metadata and relevance score."""
    text: str
    metadata: dict
    score: float          # 1 - distance (higher = more relevant)
    collection: str       # which collection it came from
    record: Optional[dict] = None  # hydrated raw PostgreSQL record


class Retriever:
    """
    Multi-collection retriever with metadata filtering and re-ranking.

    Searches across listings, projects, articles, and social_neighborhood
    collections based on query intent.

    For market_report intent, bypasses Qdrant and reads pre-computed
    aggregates from market_snapshots (SQL) — giving accurate median/avg
    price trends from actual listing data, not news articles.
    """

    def __init__(self, store: Optional[VectorStore] = None):
        self._store = store or VectorStore()

    def retrieve(
        self,
        query_text: str,
        collections: list[str],
        filters: Optional[dict] = None,
        top_k: int = 5,
        per_collection_k: int = 8,
        lifestyle_signals: Optional[list[str]] = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents from specified collections.

        Args:
            query_text: Natural language query for semantic search
            collections: List of collection names to search
            filters: Metadata filters (Qdrant payload filter)
            top_k: Total number of results to return
            per_collection_k: Number of results per collection before merging
            lifestyle_signals: Detected lifestyle signals (e.g. ["metro", "school"]).
                               When provided, matched listing/project docs are enriched
                               with nearby POI context from Postgres (offline, no API).

        Returns:
            List of RetrievedDocument sorted by relevance score
        """
        all_docs: list[RetrievedDocument] = []

        for coll_name in collections:
            try:
                docs = self._search_collection(
                    coll_name, query_text, filters, per_collection_k
                )
                all_docs.extend(docs)
            except Exception as e:
                log.warning(f"Search failed on '{coll_name}': {e}")

        # Sort by score (highest first) and deduplicate
        all_docs.sort(key=lambda d: d.score, reverse=True)
        all_docs = self._deduplicate(all_docs)
        result = all_docs[:top_k]

        # Enrich with offline POI context if lifestyle signals are present
        if lifestyle_signals:
            result = self._enrich_with_pois(result, lifestyle_signals)

        return result

    def retrieve_market_report(
        self,
        filters: Optional[dict] = None,
        months: int = 12,
    ) -> list[RetrievedDocument]:
        """
        Retrieve market statistics from pre-computed SQL aggregates.

        Reads market_snapshots which is built from actual listing prices
        (nhà bán / nhà cho thuê / dự án) grouped by region + period.
        Does NOT use Qdrant or news articles — this is pure SQL aggregation.

        Args:
            filters: ParsedQuery.filters dict, used to extract province/district.
            months: Number of past months to include (default 12).

        Returns:
            List of RetrievedDocument, one per (period, district, listing_type)
            row, formatted as human-readable Vietnamese text for the LLM.
        """
        f = filters or {}
        province = f.get("tinh_thanh")
        district = f.get("quan_huyen")
        property_type = f.get("loai_nha_dat")

        try:
            rows = self._store.pg.fetch_market_stats(
                province=province,
                district=district,
                property_type=property_type,
                months=months,
            )
        except Exception as e:
            log.warning(f"fetch_market_stats failed: {e}")
            rows = []

        docs: list[RetrievedDocument] = []
        for row in rows:
            text = _format_market_row(row)
            docs.append(RetrievedDocument(
                text=text,
                metadata=row,
                score=1.0,
                collection="market_snapshots",
                record=row,
            ))
        return docs

    def _search_collection(
        self,
        collection_name: str,
        query_text: str,
        filters: Optional[dict],
        n_results: int,
    ) -> list[RetrievedDocument]:
        """Search a single collection and return RetrievedDocument list."""
        # Build Qdrant filter condition from filters
        where = self._build_where_clause(filters, collection_name)

        try:
            results = self._store.search(
                collection_name=collection_name,
                query=query_text,
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            # If filter causes error, retry without
            if where:
                log.warning(
                    f"Filtered search failed on '{collection_name}', "
                    f"retrying without filters: {e}"
                )
                results = self._store.search(
                    collection_name=collection_name,
                    query=query_text,
                    n_results=n_results,
                )
            else:
                raise

        docs = []
        if not results or not results.get("ids"):
            return docs

        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        # Hydrate rich contexts from PostgreSQL by point IDs
        hydrated_records = {}
        try:
            pg_table = "social_neighborhood" if collection_name == "social_neighborhood" else collection_name
            records = self._store.pg.fetch_by_ids(pg_table, ids)
            hydrated_records = {r["id"]: r for r in records if "id" in r}
        except Exception as pg_err:
            log.warning(f"Failed to hydrate from PostgreSQL: {pg_err}")

        for i in range(len(ids)):
            doc_id = ids[i]
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0

            # Convert cosine distance to similarity score
            score = max(0.0, 1.0 - dist)
            
            # Hydrate text using rich metadata fields
            record = hydrated_records.get(doc_id)
            if record:
                if collection_name == "listings" and record.get("mo_ta_chi_tiet"):
                    text = f"{text}. Chi tiết tin đăng: {record['mo_ta_chi_tiet']}"
                elif collection_name == "projects" and record.get("mo_ta_chi_tiet"):
                    text = f"{text}. Chi tiết dự án: {record['mo_ta_chi_tiet']}"
                elif collection_name == "social_neighborhood":
                    title = record.get("title") or ""
                    source = record.get("source_type") or "mạng xã hội"
                    description = text.strip()           # Qdrant chunk = short description
                    body = (record.get("text_content") or "").strip()

                    import json as _json
                    raw_comments = record.get("comments_json") or []
                    if isinstance(raw_comments, str):
                        try:
                            raw_comments = _json.loads(raw_comments)
                        except Exception:
                            raw_comments = []

                    # Build structured readable context
                    lines = []
                    lines.append(
                        f"## Nội dung về \"{title}\" trên {source} ##"
                    )

                    if description and description != body:
                        lines.append(f"\n### Mô tả ###\n{description}")

                    if body:
                        lines.append(f"\n### Nội dung ###\n{body}")

                    if raw_comments:
                        lines.append("\n### Bình luận ###")
                        for idx, c in enumerate(raw_comments[:15], 1):
                            author = (c.get("author") or "Ẩn danh").strip()
                            content = (
                                c.get("comment_raw")
                                or c.get("content")
                                or c.get("text")
                                or ""
                            ).strip()
                            if content:
                                lines.append(f"{idx}. [{author}]: {content}")

                    text = "\n".join(lines)


            docs.append(RetrievedDocument(
                text=text,
                metadata=meta,
                score=score,
                collection=collection_name,
                record=record
            ))

        return docs

    def _build_where_clause(
        self, filters: Optional[dict], collection_name: str
    ) -> Optional[dict]:
        """Build a ChromaDB where clause from parsed filters."""
        if not filters:
            if collection_name == "social_neighborhood":
                return {"relevance_score": {"$gte": 0.15}}
            return None

        conditions = []

        for key, value in filters.items():
            if key in ("gia_trieu", "price_vnd", "price_per_m2_vnd", "dien_tich_m2", "so_phong_ngu", "so_phong_tam"):
                # Only apply numeric filters to listings collection
                if collection_name != "listings":
                    continue
                if isinstance(value, dict):
                    # Range filter: {"$gte": X, "$lte": Y}
                    for op, val in value.items():
                        conditions.append({key: {op: val}})
                else:
                    conditions.append({key: {"$eq": value}})

            elif key in ("tinh_thanh", "quan_huyen", "province", "district", "ward"):
                # Location filters apply to listings and projects
                if collection_name not in ("listings", "projects"):
                    continue
                conditions.append({key: {"$eq": value}})

            elif key == "loai_nha_dat":
                if collection_name != "listings":
                    continue
                conditions.append({key: {"$eq": value}})

        if collection_name == "social_neighborhood":
            conditions.append({"relevance_score": {"$gte": 0.15}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _deduplicate(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Remove duplicate documents based on text similarity."""
        seen_texts: set[str] = set()
        unique: list[RetrievedDocument] = []

        for doc in docs:
            # Use first 200 chars as dedup key
            key = doc.text[:200].strip()
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(doc)

        return unique

    # ---------------------------------------------------------------------------
    # Lifestyle / POI enrichment
    # ---------------------------------------------------------------------------

    # Signal key → OSM POI categories to fetch from Postgres
    _SIGNAL_TO_CATEGORIES: dict[str, list[str]] = {
        "metro":          ["transit_station"],
        "school":         ["school"],
        "hospital":       ["hospital"],
        "park":           ["park"],
        "shopping":       ["shopping_mall", "supermarket"],
        # flood / livability / safety / appreciation / infrastructure:
        # no POI category — handled via social_neighborhood retrieval
    }

    # Human-readable Vietnamese category labels for the context string
    _CATEGORY_LABELS: dict[str, str] = {
        "transit_station": "Ga/Trạm metro",
        "school":           "Trường học",
        "hospital":         "Bệnh viện",
        "park":             "Công viên",
        "shopping_mall":    "Trung tâm thương mại",
        "supermarket":      "Siêu thị",
    }

    def _enrich_with_pois(
        self,
        docs: list[RetrievedDocument],
        lifestyle_signals: list[str],
    ) -> list[RetrievedDocument]:
        """
        Append offline POI amenity summaries to listing/project documents.

        Executes a single batched Postgres JOIN across all doc IDs — no per-doc
        query, no external API calls. Silently skips docs with no cached POI rows.
        """
        # Collect relevant categories from signals
        categories: list[str] = []
        for sig in lifestyle_signals:
            categories.extend(self._SIGNAL_TO_CATEGORIES.get(sig, []))
        categories = list(dict.fromkeys(categories))  # dedupe, preserve order

        if not categories:
            return docs  # no POI-backed signals; social retrieval handles the rest

        # Gather IDs by entity type from listing/project docs only
        listing_ids = [
            d.record["id"] for d in docs
            if d.collection == "listings" and d.record and d.record.get("id")
        ]
        project_ids = [
            d.record["id"] for d in docs
            if d.collection == "projects" and d.record and d.record.get("id")
        ]

        # Batch fetch: one query per entity type
        poi_map: dict[str, list[dict]] = {}
        if listing_ids:
            try:
                poi_map.update(
                    self._store.pg.fetch_nearby_pois(
                        listing_ids, entity_type="listing",
                        categories=categories, top_n_per_category=2,
                    )
                )
            except Exception as e:
                log.warning(f"POI enrichment failed for listings: {e}")
        if project_ids:
            try:
                poi_map.update(
                    self._store.pg.fetch_nearby_pois(
                        project_ids, entity_type="project",
                        categories=categories, top_n_per_category=2,
                    )
                )
            except Exception as e:
                log.warning(f"POI enrichment failed for projects: {e}")

        # Append amenity summary to each matching doc
        for doc in docs:
            if doc.collection not in ("listings", "projects") or not doc.record:
                continue
            entity_id = doc.record.get("id")
            if not entity_id:
                continue
            pois = poi_map.get(entity_id, [])
            if not pois:
                continue

            # Build a concise amenity line grouped by category
            by_cat: dict[str, list[str]] = {}
            for p in pois:
                label = self._CATEGORY_LABELS.get(p["category"], p["category"])
                dist = f"{p['distance_m']}m" if p.get("distance_m") else ""
                entry = f"{p['name']} ({dist})".strip(" ()")
                by_cat.setdefault(label, []).append(entry)

            lines = [f"{label}: {', '.join(names)}" for label, names in by_cat.items()]
            doc.text += "\nTiện ích lân cận: " + " | ".join(lines)

        return docs
