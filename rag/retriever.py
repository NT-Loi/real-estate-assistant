"""
Retriever — dense, keyword, and hybrid search across Qdrant/PostgreSQL.

Searches relevant collections based on parsed query intent, applies structured
filters, and returns ranked results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from db.config import ENABLE_RERANKER, RERANKER_CANDIDATES
from db.vectorstore import VectorStore
from rag.reranker import VietnameseReranker

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
        self._reranker = VietnameseReranker() if ENABLE_RERANKER else None

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
        if not collections:
            collections = ["articles", "social_neighborhood"]

        for coll_name in collections:
            try:
                docs = self._search_collection(
                    coll_name, query_text, filters, per_collection_k
                )
                all_docs.extend(docs)
            except Exception as e:
                log.warning(f"Search failed on '{coll_name}': {e}")

        all_docs.sort(key=lambda d: d.score, reverse=True)
        all_docs = self._deduplicate(all_docs)
        if self._reranker and all_docs:
            all_docs = self._reranker.rerank(query_text, all_docs[:RERANKER_CANDIDATES], top_k)
        result = all_docs[:top_k]

        return result

    def hybrid_retrieve(
        self,
        query_text: str,
        collections: list[str],
        filters: Optional[dict] = None,
        top_k: int = 5,
        per_collection_k: int = 20,
        lifestyle_signals: Optional[list[str]] = None,
    ) -> list[RetrievedDocument]:
        """Combine dense Qdrant candidates with PostgreSQL keyword candidates."""
        dense_docs = self.retrieve(
            query_text=query_text,
            collections=collections,
            filters=filters,
            top_k=RERANKER_CANDIDATES,
            per_collection_k=per_collection_k,
            lifestyle_signals=lifestyle_signals,
        )
        keyword_docs = self.keyword_retrieve(
            query_text=query_text,
            collections=collections,
            top_k=RERANKER_CANDIDATES,
        )
        merged = self._merge_candidates(dense_docs + keyword_docs)
        if self._reranker and merged:
            return self._reranker.rerank(query_text, merged[:RERANKER_CANDIDATES], top_k)
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[:top_k]

    def keyword_retrieve(
        self,
        query_text: str,
        collections: list[str],
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        """Keyword fallback over PostgreSQL source tables."""
        if not collections:
            collections = ["articles", "social_neighborhood"]

        docs: list[RetrievedDocument] = []
        query_like = f"%{query_text}%"

        try:
            with self._store.pg.get_cursor() as cur:
                for coll in collections:
                    if coll == "listings":
                        cur.execute(
                            """
                            SELECT id, raw_json FROM listings
                            WHERE tieu_de ILIKE %s OR mo_ta ILIKE %s OR mo_ta_chi_tiet ILIKE %s
                               OR dia_chi ILIKE %s OR khu_vuc ILIKE %s OR du_an ILIKE %s
                            LIMIT %s
                            """,
                            (query_like, query_like, query_like, query_like, query_like, query_like, top_k),
                        )
                    elif coll == "projects":
                        cur.execute(
                            """
                            SELECT id, raw_json FROM projects
                            WHERE ten_du_an ILIKE %s OR mo_ta_chi_tiet ILIKE %s OR tien_ich::text ILIKE %s
                               OR dia_chi ILIKE %s OR khu_vuc ILIKE %s OR chu_dau_tu ILIKE %s
                            LIMIT %s
                            """,
                            (query_like, query_like, query_like, query_like, query_like, query_like, top_k),
                        )
                    elif coll == "articles":
                        cur.execute(
                            """
                            SELECT id, raw_json FROM articles
                            WHERE tieu_de ILIKE %s OR mo_ta ILIKE %s OR mo_ta_chi_tiet ILIKE %s
                               OR danh_muc ILIKE %s
                            LIMIT %s
                            """,
                            (query_like, query_like, query_like, query_like, top_k),
                        )
                    elif coll == "social_neighborhood":
                        cur.execute(
                            """
                            SELECT id, raw_json FROM social_neighborhood
                            WHERE title ILIKE %s OR text_content ILIKE %s OR comments_json::text ILIKE %s
                               OR keyword ILIKE %s
                            LIMIT %s
                            """,
                            (query_like, query_like, query_like, query_like, top_k),
                        )
                    else:
                        continue

                    for row_id, raw in cur.fetchall():
                        record = self._decode_record(raw)
                        if record is None:
                            continue
                        record["id"] = record.get("id") or row_id
                        docs.append(self._record_to_doc(coll, record, score=0.65))
        except Exception as exc:
            log.warning("keyword_retrieve failed: %s", exc)

        return self._deduplicate(docs)[:top_k]

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

        source_ids = [
            (m or {}).get("source_record_id") or ids[idx]
            for idx, m in enumerate(metadatas)
        ]

        # Hydrate rich contexts from PostgreSQL by source record IDs. Qdrant point
        # IDs are chunk IDs, while PostgreSQL stores one row per source record.
        hydrated_records = {}
        try:
            pg_table = "social_neighborhood" if collection_name == "social_neighborhood" else collection_name
            records = self._store.pg.fetch_by_ids(pg_table, source_ids)
            hydrated_records = {r["id"]: r for r in records if "id" in r}
        except Exception as pg_err:
            log.warning(f"Failed to hydrate from PostgreSQL: {pg_err}")

        for i in range(len(ids)):
            doc_id = ids[i]
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0

            score = max(0.0, 1.0 - dist)
            source_id = meta.get("source_record_id") or doc_id
            
            # Hydrate text using rich metadata fields
            record = hydrated_records.get(source_id)
            if record:
                if collection_name == "social_neighborhood":
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
                else:
                    text = self._attach_record_summary(collection_name, text, record)


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
        """Remove exact duplicate chunks while allowing multiple chunk types per record."""
        seen: set[tuple[str, str, str, str]] = set()
        unique: list[RetrievedDocument] = []

        for doc in docs:
            meta = doc.metadata or {}
            key = (
                doc.collection,
                str(meta.get("source_record_id") or meta.get("url") or ""),
                str(meta.get("chunk_type") or ""),
                doc.text[:160].strip(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique

    def _merge_candidates(self, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        best: dict[tuple[str, str, str], RetrievedDocument] = {}
        for doc in docs:
            meta = doc.metadata or {}
            key = (
                doc.collection,
                str(meta.get("source_record_id") or meta.get("url") or doc.text[:80]),
                str(meta.get("chunk_type") or "record"),
            )
            existing = best.get(key)
            if existing is None or doc.score > existing.score:
                best[key] = doc
        merged = list(best.values())
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged

    def _decode_record(self, raw):
        import json

        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        return None

    def _record_to_doc(self, collection: str, record: dict, score: float) -> RetrievedDocument:
        if collection == "listings":
            title = record.get("tieu_de") or "Tin đăng"
            text = (
                f"{title}. Giá: {record.get('gia') or 'Thỏa thuận'}, "
                f"Diện tích: {record.get('dien_tich') or 'Chưa rõ'}, "
                f"Địa chỉ: {record.get('dia_chi') or record.get('khu_vuc') or 'Chưa rõ'}. "
                f"{record.get('mo_ta_chi_tiet') or record.get('mo_ta') or ''}"
            )
        elif collection == "projects":
            text = (
                f"Dự án {record.get('ten_du_an') or ''}. "
                f"Địa chỉ: {record.get('dia_chi') or record.get('khu_vuc') or 'Chưa rõ'}. "
                f"Chủ đầu tư: {record.get('chu_dau_tu') or 'Chưa rõ'}. "
                f"Tiện ích: {', '.join(record.get('tien_ich') or []) if isinstance(record.get('tien_ich'), list) else record.get('tien_ich') or ''}. "
                f"{record.get('mo_ta_chi_tiet') or ''}"
            )
        elif collection == "social_neighborhood":
            text = (
                f"{record.get('source_type') or 'social'} - {record.get('title') or record.get('thread_title') or ''}. "
                f"{record.get('text_content') or record.get('description') or ''}"
            )
        else:
            text = f"{record.get('tieu_de') or record.get('title') or ''}. {record.get('mo_ta_chi_tiet') or record.get('mo_ta') or ''}"

        metadata = {
            "url": record.get("url") or record.get("thread_url") or "",
            "source_record_id": record.get("id") or "",
            "chunk_type": "keyword_record",
        }
        return RetrievedDocument(text=text.strip(), metadata=metadata, score=score, collection=collection, record=record)

    def _attach_record_summary(self, collection_name: str, text: str, record: dict) -> str:
        if collection_name == "listings":
            summary = (
                f"Thông tin gốc: {record.get('tieu_de') or ''}. "
                f"Giá: {record.get('gia') or 'Chưa rõ'}, "
                f"Diện tích: {record.get('dien_tich') or 'Chưa rõ'}, "
                f"Địa chỉ: {record.get('dia_chi') or record.get('khu_vuc') or 'Chưa rõ'}."
            )
        elif collection_name == "projects":
            summary = (
                f"Thông tin gốc: Dự án {record.get('ten_du_an') or ''}. "
                f"Chủ đầu tư: {record.get('chu_dau_tu') or 'Chưa rõ'}, "
                f"Địa chỉ: {record.get('dia_chi') or record.get('khu_vuc') or 'Chưa rõ'}."
            )
        else:
            summary = ""
        return f"{text}\n{summary}".strip()

    # ---------------------------------------------------------------------------
