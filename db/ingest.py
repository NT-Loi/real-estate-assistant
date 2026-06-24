"""
Ingest — end-to-end pipeline that reads crawled JSON files and loads them
into PostgreSQL (structured storage) and Qdrant (vector search).

Usage:
    python -m db.ingest                     # Ingest all available data
    python -m db.ingest --source listings   # Ingest only listings
    python -m db.ingest --source projects
    python -m db.ingest --source articles
    python -m db.ingest --source social
    python -m db.ingest --source pois
    python -m db.ingest --reset             # Clear collections before ingesting
    python -m db.ingest --batch-size 64     # Override embedding batch size

Progress is tracked with tqdm bars at the record level, one bar per source.
Each bar shows: records processed, skipped, and embedding throughput.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from tqdm import tqdm

from db.config import DATA_DIR
from db.chunker import (
    listing_to_chunks,
    project_to_chunks,
    article_to_chunks,
    social_to_chunks,
)
from db.normalizer import (
    normalize_listing_metadata,
    normalize_project_metadata,
    normalize_article_metadata,
)
from db.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bds_ingest")


HF_DEFAULT_DATASET = "tinixai/vietnam-real-estates"
HF_DEFAULT_SPLIT = "train"
HF_CACHE_FILE = DATA_DIR / "hf_vietnam_real_estates.jsonl"

# Silence noisy sub-loggers during ingestion so tqdm bars are readable
for _noisy in ("bds_vectorstore", "bds_embedder", "qdrant_client", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str, key: str, chunk_idx: int = 0) -> str:
    """Generate a deterministic UUID from source prefix + key + chunk index."""
    raw = f"{prefix}:{key}:{chunk_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _stream_json(path: Path) -> Iterator[dict]:
    """Yield records one by one from a JSON file (array at root level)."""
    if not path.exists():
        log.warning(f"File not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        log.warning(f"Expected a JSON array in {path.name}, got {type(data).__name__}")
        return
    yield from data


def _count_records(path: Path) -> int:
    """Count records in a JSON file without loading full content into memory."""
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0
    
def _count_jsonl(path: Path) -> int:
    """Count newline-delimited JSON records."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
    
def _stream_jsonl(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL cache file."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _first_non_empty(record: dict, *keys: str) -> Any:
    """Return the first present, non-empty value from several possible field names."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _to_float(value: Any) -> float | None:
    """Best-effort numeric coercion for HuggingFace tabular fields."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        text = str(value).replace(",", ".")
        import re

        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None


def _to_int(value: Any) -> int | None:
    """Best-effort integer coercion."""
    number = _to_float(value)
    return int(number) if number is not None else None


def _safe_int(value: Any, default: int = 0) -> int:
    number = _to_int(value)
    return number if number is not None else default

def _price_to_vnd(value: Any) -> int | None:
    """
    Normalize common real-estate dataset price scales into VND.

    The HuggingFace dataset has used numeric price fields in different exports.
    This heuristic accepts:
    - large values as already-VND,
    - 1,000..100,000,000 as million-VND,
    - smaller values as billion-VND.
    """
    number = _to_float(value)
    if number is None or number <= 0:
        return None
    if number >= 100_000_000:
        return int(number)
    if number >= 1_000:
        return int(number * 1_000_000)
    return int(number * 1_000_000_000)


def _vnd_to_label(price_vnd: int | None, listing_type: str = "ban") -> str | None:
    """Format VND into the crawler's compact Vietnamese price label."""
    if price_vnd is None:
        return None
    suffix = "/tháng" if listing_type == "cho_thue" else ""
    if price_vnd >= 1_000_000_000:
        value = price_vnd / 1_000_000_000
        return f"{value:.2f} tỷ{suffix}".replace(".", ",")
    value = price_vnd / 1_000_000
    return f"{value:.0f} triệu{suffix}"


def _detect_listing_type(raw: dict) -> str:
    """Infer sale/rent class from the available dataset fields."""
    text = " ".join(
        str(_first_non_empty(raw, key) or "")
        for key in (
            "listing_type",
            "listing_type_name",
            "transaction_type",
            "transaction_type_name",
            "category",
            "category_name",
            "type",
            "url",
        )
    ).lower()
    if any(token in text for token in ("thuê", "thue", "rent", "rental", "lease")):
        return "cho_thue"
    return "ban"


def _province_address(raw: dict) -> str:
    """Build a readable address from tabular HuggingFace location columns."""
    parts = []
    for key in (
        "address",
        "street_name",
        "ward_name",
        "district_name",
        "province_name",
    ):
        value = _first_non_empty(raw, key)
        if value and str(value) not in parts:
            parts.append(str(value))
    return ", ".join(parts)

def normalize_hf_listing(raw: dict, row_index: int, dataset_name: str) -> dict:
    """
    Convert tinixai/vietnam-real-estates rows into the internal listing schema.

    The output intentionally mirrors crawler fields so existing PostgreSQL and
    Qdrant ingestion paths can index this external dataset without a new table.
    """
    source_id = str(
        _first_non_empty(raw, "id", "listing_id", "post_id", "url")
        or row_index
    )
    listing_type = _detect_listing_type(raw)
    price_vnd = _price_to_vnd(_first_non_empty(raw, "price", "price_vnd", "price_value"))
    area = _to_float(_first_non_empty(raw, "area", "area_m2", "acreage"))
    price_per_m2_vnd = int(price_vnd / area) if price_vnd and area and area > 0 else None

    title = _first_non_empty(raw, "name", "title", "tieu_de") or "Tin bất động sản"
    property_type = _first_non_empty(
        raw,
        "property_type_name",
        "property_type",
        "house_type_name",
        "realestate_type_name",
        "category_name",
    )
    province = _first_non_empty(raw, "province_name", "province")
    district = _first_non_empty(raw, "district_name", "district")
    ward = _first_non_empty(raw, "ward_name", "ward")
    address = _province_address(raw) or ", ".join(
        str(v) for v in (ward, district, province) if v
    )
    url = _first_non_empty(raw, "url", "link", "source_url") or f"hf://{dataset_name}/{source_id}"

    record = {
        "loai_hinh": listing_type,
        "loai_nha_dat": property_type,
        "khu_vuc": province,
        "dia_chi": address,
        "gia": _vnd_to_label(price_vnd, listing_type),
        "price_vnd": price_vnd,
        "price_per_m2_vnd": price_per_m2_vnd,
        "gia_per_m2": _vnd_to_label(price_per_m2_vnd) if price_per_m2_vnd else None,
        "dien_tich": f"{area:g} m2" if area else None,
        "dien_tich_m2": area,
        "so_phong_ngu": _to_int(_first_non_empty(raw, "bedroom_count", "bedrooms", "bedroom")),
        "so_phong_tam": _to_int(_first_non_empty(raw, "bathroom_count", "bathrooms", "toilet_count")),
        "huong_nha": _first_non_empty(raw, "house_direction", "direction"),
        "huong_ban_cong": _first_non_empty(raw, "balcony_direction"),
        "phap_ly": _first_non_empty(raw, "legal_status", "legal"),
        "noi_that": _first_non_empty(raw, "furniture", "interior"),
        "tieu_de": title,
        "mo_ta": _first_non_empty(raw, "description", "desc", "summary"),
        "mo_ta_chi_tiet": _first_non_empty(raw, "description", "content", "detail"),
        "du_an": _first_non_empty(raw, "project_name", "project", "du_an"),
        "latitude": _to_float(_first_non_empty(raw, "latitude", "lat")),
        "longitude": _to_float(_first_non_empty(raw, "longitude", "lng", "lon")),
        "url": url,
        "ngay_dang": _first_non_empty(raw, "created_at", "posted_at", "published_at", "date"),
        "nguoi_dang": _first_non_empty(raw, "contact_name", "seller_name", "broker_name"),
        "source_dataset": dataset_name,
        "source_record_id": source_id,
        "source_row_index": row_index,
        "raw_hf_json": raw,
    }
    return record

def _ensure_hf_cache(
    dataset_name: str,
    split: str,
    cache_path: Path,
    refresh: bool = False,
    limit: int | None = None,
) -> Path:
    """
    Download/cache a HuggingFace dataset split as JSONL, unless a cache exists.

    Keeping a local JSONL cache makes the PostgreSQL/Qdrant ingest resumable and
    reproducible for demo day without repeatedly hitting HuggingFace.
    """
    if cache_path.exists() and not refresh:
        return cache_path

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The 'datasets' package is required for HuggingFace ingestion. "
            "Install requirements.txt first."
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Loading HuggingFace dataset %s split=%s", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    written = 0
    with open(cache_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            if limit is not None and written >= limit:
                break
            payload = dict(row)
            payload["_hf_row_index"] = idx
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            written += 1

    log.info("Cached %s HuggingFace rows to %s", written, cache_path)
    return cache_path

def _social_relevance_score(record: dict) -> float:
    """
    Cheap guardrail against off-topic social search results.

    Keyword crawlers can return viral but unrelated videos. This score is not a
    sentiment model; it only estimates whether the retrieved post/comments are
    about the requested project/location.
    """
    keyword = (record.get("keyword") or "").lower()
    if not keyword:
        return 0.5

    keyword_terms = [
        t for t in keyword.replace(",", " ").split()
        if len(t) >= 3 and t not in {"review", "chung", "cư", "khu", "dân", "nhà", "đất"}
    ]
    if not keyword_terms:
        return 0.5

    text_parts = [
        record.get("title") or record.get("thread_title") or "",
        record.get("description") or record.get("snippet") or "",
        record.get("text_content") or "",
    ]
    comments = record.get("comments") or record.get("posts") or []
    for c in comments[:20]:
        text_parts.append(c.get("comment_raw") or c.get("content") or "")

    haystack = " ".join(text_parts).lower()
    matches = sum(1 for term in set(keyword_terms) if term in haystack)
    return min(1.0, matches / max(1, min(len(set(keyword_terms)), 5)))


@dataclass
class IngestStats:
    """Per-source ingestion counters."""
    source: str
    total: int = 0
    ingested: int = 0
    skipped: int = 0
    errors: int = 0
    chunks: int = 0

    def summary(self) -> str:
        parts = [
            f"  ✅ {self.source}: {self.ingested} ingested",
            f"{self.skipped} skipped",
            f"{self.errors} errors",
        ]
        if self.chunks:
            parts.append(f"{self.chunks} chunks")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Flush helper — sends a batch to VectorStore
# ---------------------------------------------------------------------------

def _flush(
    store: VectorStore,
    collection: str,
    ids: list[str],
    docs: list[str],
    metas: list[dict],
    records: list[dict],
    batch_size: int,
):
    """Flush a batch of documents to Qdrant + PostgreSQL."""
    if not ids:
        return
    store.add_documents(
        collection_name=collection,
        ids=ids,
        documents=docs,
        metadatas=metas,
        records=records,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Per-source ingest functions
# ---------------------------------------------------------------------------

def ingest_listings(
    store: VectorStore,
    reset: bool = False,
    batch_size: int = 64,
) -> IngestStats:
    """Ingest sale + rental listings into the 'listings' collection."""
    if reset:
        store.reset_collection("listings")

    stats = IngestStats(source="listings")
    files = ["listings_ban.json", "listings_cho_thue.json"]

    # Count total for progress bar
    total = sum(_count_records(DATA_DIR / f) for f in files)
    stats.total = total

    ids, docs, metas, recs = [], [], [], []

    with tqdm(
        total=total,
        desc="📦 Listings",
        unit="rec",
        colour="cyan",
        dynamic_ncols=True,
    ) as pbar:
        for filename in files:
            for record in _stream_json(DATA_DIR / filename):
                pbar.update(1)
                url = record.get("url", "")
                if not url:
                    stats.skipped += 1
                    pbar.set_postfix(skip=stats.skipped, err=stats.errors)
                    continue

                try:
                    source_id = _make_id("listing_record", url)
                    record["id"] = source_id
                    chunks = listing_to_chunks(record)
                    base_meta = normalize_listing_metadata(record)

                    if not chunks:
                        stats.skipped += 1
                        pbar.set_postfix(skip=stats.skipped, err=stats.errors)
                        continue

                    for chunk in chunks:
                        chunk_idx = int(chunk["chunk_index"])
                        ids.append(_make_id("listing", url, chunk_idx))
                        docs.append(chunk["text"])
                        metas.append(
                            {
                                **base_meta,
                                "source_record_id": source_id,
                                "chunk_type": chunk["chunk_type"],
                                "chunk_index": chunk_idx,
                                "total_chunks": chunk["total_chunks"],
                            }
                        )
                        recs.append(record)
                        stats.chunks += 1

                    stats.ingested += 1

                    if len(ids) >= batch_size:
                        _flush(store, "listings", ids, docs, metas, recs, batch_size)
                        ids, docs, metas, recs = [], [], [], []

                except Exception as e:
                    stats.errors += 1
                    log.debug(f"Listing error {url}: {e}")

                pbar.set_postfix(ok=stats.ingested, skip=stats.skipped, err=stats.errors)

        # Flush remainder
        _flush(store, "listings", ids, docs, metas, recs, batch_size)

    # Refresh market stats after listing ingestion
    try:
        tqdm.write("  ↻ Refreshing market_snapshots...")
        store.pg.refresh_market_snapshots()
    except Exception as e:
        tqdm.write(f"  ⚠ refresh_market_snapshots failed: {e}")

    tqdm.write(stats.summary())
    return stats

def ingest_hf_listings(
    store: VectorStore,
    reset: bool = False,
    batch_size: int = 64,
    dataset_name: str = HF_DEFAULT_DATASET,
    split: str = HF_DEFAULT_SPLIT,
    cache_path: Path = HF_CACHE_FILE,
    refresh_cache: bool = False,
    limit: int | None = None,
) -> IngestStats:
    """Ingest HuggingFace real-estate listings into the shared listings store."""
    if reset:
        store.reset_collection("listings")

    stats = IngestStats(source="hf-listings")
    try:
        path = _ensure_hf_cache(
            dataset_name=dataset_name,
            split=split,
            cache_path=cache_path,
            refresh=refresh_cache,
            limit=limit,
        )
    except Exception as exc:
        stats.errors += 1
        log.error("Failed to prepare HuggingFace dataset %s: %s", dataset_name, exc)
        tqdm.write(f"  ⚠ HuggingFace dataset skipped: {exc}")
        return stats

    total = _count_jsonl(path)
    if limit is not None:
        total = min(total, limit)
    stats.total = total

    ids, docs, metas, recs = [], [], [], []

    with tqdm(
        total=total,
        desc="HF Listings",
        unit="rec",
        colour="cyan",
        dynamic_ncols=True,
    ) as pbar:
        for idx, raw in enumerate(_stream_jsonl(path)):
            if limit is not None and idx >= limit:
                break
            pbar.update(1)

            try:
                row_index = int(raw.get("_hf_row_index", idx))
                record = normalize_hf_listing(raw, row_index=row_index, dataset_name=dataset_name)
                url = record.get("url")
                if not url:
                    stats.skipped += 1
                    continue

                source_id = _make_id("hf_listing_record", url)
                record["id"] = source_id
                chunks = listing_to_chunks(record)
                base_meta = normalize_listing_metadata(record)
                base_meta.update(
                    {
                        "source_dataset": dataset_name,
                        "external_source_record_id": str(record.get("source_record_id") or ""),
                        "external_source": "huggingface",
                    }
                )

                if not chunks:
                    stats.skipped += 1
                    continue

                for chunk in chunks:
                    chunk_idx = int(chunk["chunk_index"])
                    ids.append(_make_id("hf_listing", url, chunk_idx))
                    docs.append(chunk["text"])
                    metas.append(
                        {
                            **base_meta,
                            "source_record_id": source_id,
                            "chunk_type": chunk["chunk_type"],
                            "chunk_index": chunk_idx,
                            "total_chunks": chunk["total_chunks"],
                        }
                    )
                    recs.append(record)
                    stats.chunks += 1

                stats.ingested += 1

                if len(ids) >= batch_size:
                    _flush(store, "listings", ids, docs, metas, recs, batch_size)
                    ids, docs, metas, recs = [], [], [], []

            except Exception as e:
                stats.errors += 1
                log.debug("HF listing error row=%s: %s", idx, e)

            pbar.set_postfix(ok=stats.ingested, skip=stats.skipped, err=stats.errors)

        _flush(store, "listings", ids, docs, metas, recs, batch_size)

    try:
        tqdm.write("  ↻ Refreshing market_snapshots...")
        store.pg.refresh_market_snapshots()
    except Exception as e:
        tqdm.write(f"  ⚠ refresh_market_snapshots failed: {e}")

    tqdm.write(stats.summary())
    return stats

def ingest_projects(
    store: VectorStore,
    reset: bool = False,
    batch_size: int = 64,
) -> IngestStats:
    """Ingest project records into the 'projects' collection."""
    if reset:
        store.reset_collection("projects")

    stats = IngestStats(source="projects")
    path = DATA_DIR / "projects.json"
    stats.total = _count_records(path)

    ids, docs, metas, recs = [], [], [], []

    with tqdm(
        total=stats.total,
        desc="🏗  Projects",
        unit="rec",
        colour="blue",
        dynamic_ncols=True,
    ) as pbar:
        for record in _stream_json(path):
            pbar.update(1)
            url = record.get("url", "")
            if not url:
                stats.skipped += 1
                pbar.set_postfix(skip=stats.skipped)
                continue

            try:
                source_id = _make_id("project_record", url)
                record["id"] = source_id
                chunks = project_to_chunks(record)
                base_meta = normalize_project_metadata(record)

                if not chunks:
                    stats.skipped += 1
                    continue

                for chunk in chunks:
                    chunk_idx = int(chunk["chunk_index"])
                    ids.append(_make_id("project", url, chunk_idx))
                    docs.append(chunk["text"])
                    metas.append(
                        {
                            **base_meta,
                            "source_record_id": source_id,
                            "chunk_type": chunk["chunk_type"],
                            "chunk_index": chunk_idx,
                            "total_chunks": chunk["total_chunks"],
                        }
                    )
                    recs.append(record)
                    stats.chunks += 1

                stats.ingested += 1

                if len(ids) >= batch_size:
                    _flush(store, "projects", ids, docs, metas, recs, batch_size)
                    ids, docs, metas, recs = [], [], [], []

            except Exception as e:
                stats.errors += 1
                log.debug(f"Project error {url}: {e}")

            pbar.set_postfix(ok=stats.ingested, skip=stats.skipped, err=stats.errors)

        _flush(store, "projects", ids, docs, metas, recs, batch_size)

    tqdm.write(stats.summary())
    return stats


def ingest_articles(
    store: VectorStore,
    reset: bool = False,
    batch_size: int = 64,
) -> IngestStats:
    """Ingest news + wiki articles into the 'articles' collection."""
    if reset:
        store.reset_collection("articles")

    # Collect all article source files
    article_files: list[Path] = []
    if (DATA_DIR / "news.json").exists():
        article_files.append(DATA_DIR / "news.json")
    wiki_files = sorted(f for f in DATA_DIR.glob("wiki_*.json") if f.name != "wiki_all.json")
    if wiki_files:
        article_files.extend(wiki_files)
    elif (DATA_DIR / "wiki_all.json").exists():
        article_files.append(DATA_DIR / "wiki_all.json")

    stats = IngestStats(source="articles")
    total = sum(_count_records(f) for f in article_files)
    stats.total = total

    seen_urls: set[str] = set()
    ids, docs, metas, recs = [], [], [], []

    with tqdm(
        total=total,
        desc="📰 Articles",
        unit="art",
        colour="yellow",
        dynamic_ncols=True,
    ) as pbar:
        for filepath in article_files:
            pbar.set_description(f"📰 Articles [{filepath.name}]")
            for record in _stream_json(filepath):
                pbar.update(1)
                url = record.get("url", "")
                if not url or url in seen_urls:
                    stats.skipped += 1
                    continue
                seen_urls.add(url)

                try:
                    source_id = _make_id("article_record", url)
                    record["id"] = source_id
                    base_meta = normalize_article_metadata(record)
                    chunks = article_to_chunks(record)

                    for chunk in chunks:
                        chunk_text = chunk["text"]
                        if not chunk_text.strip():
                            continue
                        chunk_idx = int(chunk["chunk_index"])
                        doc_id = _make_id("article", url, chunk_idx)
                        meta = {
                            **base_meta,
                            "source_record_id": source_id,
                            "chunk_type": chunk["chunk_type"],
                            "chunk_index": chunk_idx,
                            "total_chunks": chunk["total_chunks"],
                        }
                        ids.append(doc_id)
                        docs.append(chunk_text)
                        metas.append(meta)
                        recs.append(record)
                        stats.chunks += 1

                    stats.ingested += 1

                    if len(ids) >= batch_size:
                        _flush(store, "articles", ids, docs, metas, recs, batch_size)
                        ids, docs, metas, recs = [], [], [], []

                except Exception as e:
                    stats.errors += 1
                    log.debug(f"Article error {url}: {e}")

                pbar.set_postfix(
                    ok=stats.ingested,
                    chunks=stats.chunks,
                    skip=stats.skipped,
                    err=stats.errors,
                )

        _flush(store, "articles", ids, docs, metas, recs, batch_size)

    tqdm.write(stats.summary())
    return stats


def ingest_social(
    store: VectorStore,
    reset: bool = False,
    batch_size: int = 64,
) -> IngestStats:
    """Ingest social discussions (VOZ, YouTube, TikTok) into PostgreSQL + Qdrant."""
    if reset:
        store.reset_collection("social_neighborhood")

    social_sources = [
        ("youtube_comments.json", "youtube",
         lambda r: r.get("url") or f"https://www.youtube.com/watch?v={r.get('video_id')}"),
        ("tiktok_comments.json", "tiktok",
         lambda r: r.get("url") or f"https://www.tiktok.com/video/{r.get('video_id')}"),
        ("voz_discussions.json", "voz",
         lambda r: r.get("thread_url") or r.get("url") or ""),
    ]

    stats = IngestStats(source="social")
    total = sum(_count_records(DATA_DIR / fname) for fname, _, _ in social_sources)
    stats.total = total

    ids, docs, metas, recs = [], [], [], []

    with tqdm(
        total=total,
        desc="💬 Social  ",
        unit="post",
        colour="magenta",
        dynamic_ncols=True,
    ) as pbar:
        for filename, source_type, url_fn in social_sources:
            pbar.set_description(f"💬 Social [{source_type}]")
            for record in _stream_json(DATA_DIR / filename):
                pbar.update(1)
                record["source_type"] = source_type
                url = url_fn(record)
                if not url:
                    stats.skipped += 1
                    continue

                try:
                    stats_raw = record.get("stats") or {}
                    relevance = _social_relevance_score(record)
                    record["relevance_score"] = relevance
                    source_id = _make_id(f"{source_type}_record", url)
                    record["id"] = source_id
                    chunks = social_to_chunks(record)

                    if not chunks:
                        stats.skipped += 1
                        continue

                    base_meta = {
                        "source_type": source_type,
                        "source_record_id": source_id,
                        "keyword": record.get("keyword") or "",
                        "url": url,
                        "stats_views": _safe_int(stats_raw.get("views") or stats_raw.get("view_count")),
                        "stats_likes": _safe_int(stats_raw.get("likes") or stats_raw.get("like_count")),
                        "reactions": _safe_int(record.get("reactions")),
                        "relevance_score": relevance,
                    }

                    for chunk in chunks:
                        chunk_idx = int(chunk["chunk_index"])
                        ids.append(_make_id(source_type, url, chunk_idx))
                        docs.append(chunk["text"])
                        metas.append(
                            {
                                **base_meta,
                                "chunk_type": chunk["chunk_type"],
                                "chunk_index": chunk_idx,
                                "total_chunks": chunk["total_chunks"],
                            }
                        )
                        recs.append(record)
                        stats.chunks += 1

                    stats.ingested += 1

                    if len(ids) >= batch_size:
                        _flush(store, "social_neighborhood", ids, docs, metas, recs, batch_size)
                        ids, docs, metas, recs = [], [], [], []

                except Exception as e:
                    stats.errors += 1
                    log.debug(f"Social error {url}: {e}")

                pbar.set_postfix(ok=stats.ingested, skip=stats.skipped, err=stats.errors)

        _flush(store, "social_neighborhood", ids, docs, metas, recs, batch_size)

    tqdm.write(stats.summary())
    return stats


def ingest_pois(
    store: VectorStore,
    reset: bool = False,
    batch_size: int = 256,
) -> IngestStats:
    """
    Ingest structured POIs into PostgreSQL only.
    POIs are used for geo distance queries, not semantic search — no Qdrant.
    """
    if reset:
        with store.pg.get_cursor() as cur:
            cur.execute("TRUNCATE TABLE pois CASCADE;")

    path = DATA_DIR / "pois.json"
    stats = IngestStats(source="pois")
    stats.total = _count_records(path)
    if not path.exists():
        tqdm.write(f"  ⏭ POIs skipped: {path} not found")
        return stats

    with tqdm(
        total=stats.total,
        desc="📍 POIs    ",
        unit="poi",
        colour="green",
        dynamic_ncols=True,
    ) as pbar:
        for record in _stream_json(path):
            pbar.update(1)
            if not record.get("name") or not record.get("category"):
                stats.skipped += 1
                pbar.set_postfix(skip=stats.skipped)
                continue

            try:
                store.pg.upsert_poi(record)
                stats.ingested += 1
            except Exception as e:
                stats.errors += 1
                log.debug(f"POI error {record.get('name')}: {e}")

            pbar.set_postfix(ok=stats.ingested, skip=stats.skipped, err=stats.errors)

    tqdm.write(stats.summary())
    return stats


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def ingest_all(reset: bool = False, batch_size: int = 64) -> dict[str, IngestStats]:
    """Run the full ingestion pipeline across all sources."""
    store = VectorStore()

    print()
    print("=" * 62)
    print("  🏠  Real Estate Assistant — Ingestion Pipeline")
    print(f"  Data:  {DATA_DIR}")
    print(f"  Reset: {reset}  |  Batch: {batch_size}")
    print("=" * 62)
    print()

    all_stats: dict[str, IngestStats] = {}

    all_stats["listings"] = ingest_listings(store, reset=reset, batch_size=batch_size)
    all_stats["projects"] = ingest_projects(store, reset=reset, batch_size=batch_size)
    all_stats["articles"] = ingest_articles(store, reset=reset, batch_size=batch_size)
    all_stats["social"]   = ingest_social(store, reset=reset, batch_size=batch_size)
    all_stats["pois"]     = ingest_pois(store, reset=reset)

    # Refresh map pins (requires PostGIS; silently skips if not available)
    tqdm.write("\n  ↻ Refreshing map_pins view...")
    try:
        store.pg.refresh_map_pins()
        tqdm.write("  ✅ map_pins refreshed")
    except Exception as e:
        tqdm.write(f"  ⚠ map_pins skipped: {e}")

    # Summary
    qdrant_stats = store.stats()
    total_ingested = sum(s.ingested for s in all_stats.values())
    total_skipped  = sum(s.skipped  for s in all_stats.values())
    total_errors   = sum(s.errors   for s in all_stats.values())

    print()
    print("=" * 62)
    print("  ✅  Ingestion Complete")
    print("=" * 62)
    print(f"  {'Source':<14} {'Ingested':>9} {'Skipped':>8} {'Errors':>7}")
    print(f"  {'-'*14} {'-'*9} {'-'*8} {'-'*7}")
    for name, s in all_stats.items():
        chunk_note = f" ({s.chunks} chunks)" if s.chunks else ""
        print(f"  {name:<14} {s.ingested:>9,}{chunk_note:<14} {s.skipped:>8,} {s.errors:>7,}")
    print(f"  {'-'*14} {'-'*9} {'-'*8} {'-'*7}")
    print(f"  {'TOTAL':<14} {total_ingested:>9,} {'':>14} {total_skipped:>8,} {total_errors:>7,}")
    print()
    print("  Qdrant collections:")
    for coll, count in qdrant_stats.items():
        print(f"    {coll:<26} {count:>6,} vectors")
    print("=" * 62)
    print()

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest crawled data into PostgreSQL + Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m db.ingest                          # ingest everything
  python -m db.ingest --source listings        # listings only
  python -m db.ingest --source pois --reset    # clear + reload POIs
  python -m db.ingest --batch-size 128         # larger embedding batches
        """,
    )
    parser.add_argument(
        "--source",
        choices=["listings", "projects", "articles", "social", "pois", "all"],
        default="all",
        help="Which data source to ingest (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear target collections/tables before ingesting",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of records per embedding batch (default: 64)",
    )
    args = parser.parse_args()
    store = VectorStore()

    if args.source == "all":
        ingest_all(reset=args.reset, batch_size=args.batch_size)
    elif args.source == "listings":
        ingest_listings(store, reset=args.reset, batch_size=args.batch_size)
    elif args.source == "projects":
        ingest_projects(store, reset=args.reset, batch_size=args.batch_size)
    elif args.source == "articles":
        ingest_articles(store, reset=args.reset, batch_size=args.batch_size)
    elif args.source == "social":
        ingest_social(store, reset=args.reset, batch_size=args.batch_size)
    elif args.source == "pois":
        ingest_pois(store, reset=args.reset)
