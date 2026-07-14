from __future__ import annotations

import asyncio
import json
import math
import mimetypes
import os
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Dict, List
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from db.config import DATA_DIR as CONFIG_DATA_DIR
except Exception:
    CONFIG_DATA_DIR = ROOT / "data"

DATA_DIR = Path(CONFIG_DATA_DIR)

RAG_CHAIN = None
RAG_INIT_ERROR = None

from utils.logging_config import configure_system_logging

# Configure logging to console and logs/system.log.
configure_system_logging()
log = logging.getLogger("bds_server")

CITY_COORDS = {
    "Hà Nội": (21.0285, 105.8542),
    "Hồ Chí Minh": (10.7769, 106.7009),
    "TP Hồ Chí Minh": (10.7769, 106.7009),
    "Đà Nẵng": (16.0544, 108.2022),
    "Hải Phòng": (20.8449, 106.6881),
    "Cần Thơ": (10.0452, 105.7469),
    "Bình Dương": (10.9804, 106.6519),
    "Đồng Nai": (10.9453, 106.8246),
    "Hưng Yên": (20.6464, 106.0511),
    "Quảng Ninh": (20.9712, 107.0448),
    "Khánh Hòa": (12.2388, 109.1967),
    "Nghệ An": (18.6796, 105.6813),
    "Ninh Bình": (20.2506, 105.9745),
    "Hà Nam": (20.5411, 105.9139),
    "Nam Định": (20.4388, 106.1621),
}

DISTRICT_COORDS = {
    "Quận 1": (10.7756, 106.7019),
    "Quận 2": (10.7873, 106.7498),
    "Quận 7": (10.7368, 106.7218),
    "Quận Bình Thạnh": (10.8106, 106.7091),
    "Thành phố Thủ Đức": (10.8494, 106.7537),
    "Quận Cầu Giấy": (21.0362, 105.7906),
    "Quận Long Biên": (21.0549, 105.8885),
    "Quận Ngô Quyền": (20.8561, 106.6881),
    "Thành phố Thuận An": (10.9323, 106.7117),
    "Huyện Văn Giang": (20.9436, 105.9341),
    "Thành phố Nha Trang": (12.2388, 109.1967),
    "Thành phố Hạ Long": (20.9599, 107.0425),
}

def parse_number(raw: object) -> float | None:
    if raw is None:
        return None
    text = str(raw).replace(",", ".")
    number = ""
    seen_dot = False
    for ch in text:
        if ch.isdigit():
            number += ch
        elif ch == "." and not seen_dot:
            number += ch
            seen_dot = True
        elif number:
            break
    if not number:
        return None
    try:
        return float(number)
    except ValueError:
        return None

def price_vnd(raw: object) -> int | None:
    if not raw:
        return None
    text = str(raw).lower()
    if "thỏa thuận" in text or "thoả thuận" in text or "liên hệ" in text:
        return None
    if "/m" in text or "tr/m" in text:
        return None
    num = parse_number(text)
    if num is None:
        return None
    if "tỷ" in text or "ty" in text:
        return int(num * 1_000_000_000)
    if "triệu" in text or "trieu" in text or "tr" in text:
        return int(num * 1_000_000)
    return int(num * 1_000_000) if num > 100 else None

def first_line(raw: object) -> str:
    lines = str(raw or "").splitlines()
    if not lines:
        return ""
    return lines[0].replace(". Xem bản đồ", "").strip()

def location_parts(address: str, fallback_area: str = "") -> dict[str, str]:
    fallback = first_line(fallback_area)
    parts = [p.strip() for p in first_line(address or fallback).split(",") if p.strip()]
    province = parts[-1] if parts else fallback
    district = ""
    ward = ""
    for part in parts:
        lower = part.lower()
        if lower.startswith(("quận ", "huyện ", "thành phố ", "tp ")):
            district = part
        if lower.startswith(("phường ", "xã ", "thị trấn ")):
            ward = part
    province = str(province or "").replace("Hồ Chí Minh mới", "Hồ Chí Minh").replace("Ninh Bình mới", "Ninh Bình")
    return {"province": province, "district": district, "ward": ward}

def stable_offset(seed: str, scale: float = 0.035) -> tuple[float, float]:
    value = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed))
    angle = (value % 360) * math.pi / 180
    radius = ((value % 100) / 100) * scale
    return math.sin(angle) * radius, math.cos(angle) * radius

def approximate_coords(record: dict) -> tuple[float, float, str]:
    lat = record.get("latitude") or record.get("lat")
    lng = record.get("longitude") or record.get("lng") or record.get("lon")
    if lat and lng:
        return float(lat), float(lng), "exact"

    loc = location_parts(record.get("dia_chi", ""), record.get("khu_vuc", ""))
    base = DISTRICT_COORDS.get(loc["district"]) or CITY_COORDS.get(loc["province"])
    if not base:
        base = CITY_COORDS.get(record.get("khu_vuc", ""), (16.0, 106.0))
    dlat, dlng = stable_offset(record.get("url") or record.get("tieu_de") or "")
    return base[0] + dlat, base[1] + dlng, "approximate"

def normalize_listing_type(value: object, filename: str = "") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"cho-thue", "thue", "rental", "rent"}:
        return "cho-thue"
    if text in {"ban", "bán", "sale"}:
        return "ban"
    if "cho_thue" in filename or "cho-thue" in filename:
        return "cho-thue"
    if "ban" in filename:
        return "ban"
    return text

def load_listings() -> list[dict]:
    rows: list[dict] = []
    for filename in ("listings_ban.json", "listings_cho_thue.json"):
        path = DATA_DIR / filename
        if not path.exists():
            log.warning(f"Listing source not found: {path}")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for idx, record in enumerate(records):
            loc = location_parts(record.get("dia_chi", ""), record.get("khu_vuc", ""))
            lat, lng, precision = approximate_coords(record)
            price = price_vnd(record.get("gia"))
            area = parse_number(record.get("dien_tich"))
            listing_type = normalize_listing_type(record.get("loai_hinh"), filename)
            rows.append(
                {
                    "id": f"{listing_type or 'listing'}-{idx}",
                    "listing_type": listing_type,
                    "property_type": record.get("loai_nha_dat") or "Chưa phân loại",
                    "title": record.get("tieu_de") or "",
                    "address": first_line(record.get("dia_chi") or record.get("khu_vuc")),
                    "province": loc["province"],
                    "district": loc["district"],
                    "ward": loc["ward"],
                    "project": record.get("du_an") or "",
                    "price_label": record.get("gia") or "Thỏa thuận",
                    "price_vnd": price,
                    "area_m2": area,
                    "bedrooms": parse_number(record.get("so_phong_ngu")),
                    "bathrooms": parse_number(record.get("so_phong_tam")),
                    "legal": record.get("phap_ly") or "",
                    "furniture": record.get("noi_that") or "",
                    "posted": record.get("ngay_dang") or "",
                    "url": record.get("url") or "",
                    "image": (record.get("hinh_anh") or [""])[0],
                    "lat": lat,
                    "lng": lng,
                    "geo_precision": precision,
                }
            )
    return rows

LISTINGS = load_listings()

def listing_stats(rows: list[dict]) -> dict:
    priced = [r["price_vnd"] for r in rows if r.get("price_vnd")]
    areas = [r["area_m2"] for r in rows if r.get("area_m2")]
    return {
        "count": len(rows),
        "sale_count": sum(1 for r in rows if r["listing_type"] == "ban"),
        "rent_count": sum(1 for r in rows if r["listing_type"] == "cho-thue"),
        "median_price_vnd": sorted(priced)[len(priced) // 2] if priced else None,
        "avg_area_m2": round(sum(areas) / len(areas), 1) if areas else None,
        "approximate_geo_count": sum(1 for r in rows if r["geo_precision"] == "approximate"),
    }

def get_rag_chain():
    """Return the already-initialized RAG chain, or raise if startup failed."""
    if RAG_CHAIN is not None:
        return RAG_CHAIN
    if RAG_INIT_ERROR is not None:
        raise RAG_INIT_ERROR
    # Should not reach here after lifespan runs, but guard anyway
    raise RuntimeError("RAG chain not yet initialized")


# ---------------------------------------------------------------------------
# Startup health status tracker
# ---------------------------------------------------------------------------
STARTUP_STATUS: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly initialize all heavy resources at server startup:
      1. RAGChain (triggers VectorStore → Qdrant + PostgreSQL connections)
      2. Embedding model (SentenceTransformer warm-up)
      3. Reranker model warm-up
      4. LLM client (Ollama/Gemini availability check)
    """
    global RAG_CHAIN, RAG_INIT_ERROR
    t0 = time.monotonic()
    log.info("="*60)
    log.info("  Starting RAG Real-Estate Assistant — loading all components…")
    log.info("="*60)

    # ── 1. RAGChain (includes VectorStore + Qdrant + PostgreSQL) ───────────
    try:
        log.info("[1/4] Initializing RAGChain (Qdrant + PostgreSQL)…")
        loop = asyncio.get_event_loop()
        from rag.chain import RAGChain
        RAG_CHAIN = await loop.run_in_executor(None, RAGChain)
        STARTUP_STATUS["rag_chain"] = "ok"
        log.info("[1/4] ✓ RAGChain ready")
    except Exception as exc:
        RAG_INIT_ERROR = exc
        STARTUP_STATUS["rag_chain"] = f"error: {exc}"
        log.error(f"[1/4] ✗ RAGChain failed: {exc}")
        log.warning("      Chat will fall back to local listing search.")

    # ── 2. Embedding model warm-up ──────────────────────────────────────────
    try:
        log.info("[2/4] Loading embedding model…")
        if RAG_CHAIN is not None:
            embedder = getattr(getattr(RAG_CHAIN, "_store", None), "_embedder", None)
            if embedder is not None and hasattr(embedder, "_load"):
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(None, embedder._load)
                dimension = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else "unknown"
                STARTUP_STATUS["embedding"] = f"ok (dim={dimension})"
                log.info(f"[2/4] ✓ Embedding model ready (dim={dimension})")
            else:
                STARTUP_STATUS["embedding"] = "skipped (embedder not found)"
                log.info("[2/4] - Embedding check skipped (embedder not found)")
        else:
            STARTUP_STATUS["embedding"] = "skipped (RAGChain not loaded)"
            log.info("[2/4] - Embedding check skipped (RAGChain not loaded)")
    except Exception as exc:
        STARTUP_STATUS["embedding"] = f"error: {exc}"
        log.error(f"[2/4] ✗ Embedding warm-up error: {exc}")

    # ── 3. Reranker model warm-up ───────────────────────────────────────────
    try:
        log.info("[3/4] Loading reranker model…")
        if RAG_CHAIN is not None:
            reranker = getattr(getattr(RAG_CHAIN, "_retriever", None), "_reranker", None)
            if reranker is None:
                STARTUP_STATUS["reranker"] = "disabled"
                log.info("[3/4] - Reranker disabled")
            elif hasattr(reranker, "_load"):
                loop = asyncio.get_event_loop()
                loaded = await loop.run_in_executor(None, reranker._load)
                model_name = getattr(reranker, "_model_name", "unknown")
                device = getattr(reranker, "_device", "unknown")
                if loaded:
                    STARTUP_STATUS["reranker"] = f"ok ({model_name}, device={device})"
                    log.info(f"[3/4] ✓ Reranker ready ({model_name}, device={device})")
                else:
                    STARTUP_STATUS["reranker"] = "disabled (load failed)"
                    log.warning("[3/4] ⚠ Reranker disabled (load failed)")
            else:
                STARTUP_STATUS["reranker"] = "skipped (reranker has no loader)"
                log.info("[3/4] - Reranker check skipped (no loader)")
        else:
            STARTUP_STATUS["reranker"] = "skipped (RAGChain not loaded)"
            log.info("[3/4] - Reranker check skipped (RAGChain not loaded)")
    except Exception as exc:
        STARTUP_STATUS["reranker"] = f"error: {exc}"
        log.error(f"[3/4] ✗ Reranker warm-up error: {exc}")

    # ── 4. LLM warm-up ping ─────────────────────────────────────────────────
    try:
        log.info("[4/4] Warming up LLM client…")
        if RAG_CHAIN is not None:
            llm = RAG_CHAIN._llm
            provider = getattr(llm, "_provider", "unknown")
            model = getattr(llm, "_ollama_model") or getattr(llm, "_model_name")
            available = getattr(llm, "is_available", False)
            if available:
                # Send a trivial generation to warm up the model cache
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: llm.generate("hi", max_tokens=1, temperature=0.0))
                STARTUP_STATUS["llm"] = f"ok ({provider}: {model})"
                log.info(f"[4/4] ✓ LLM ready ({provider}: {model})")
            else:
                STARTUP_STATUS["llm"] = "unavailable (no API key or connection)"
                log.warning(f"[4/4] ⚠ LLM unavailable — using formatted fallback")
        else:
            STARTUP_STATUS["llm"] = "skipped (RAGChain not loaded)"
            log.info("[4/4] - LLM check skipped (RAGChain not loaded)")
    except Exception as exc:
        STARTUP_STATUS["llm"] = f"error: {exc}"
        log.error(f"[4/4] ✗ LLM warm-up error: {exc}")

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    STARTUP_STATUS["startup_seconds"] = round(elapsed, 2)
    log.info("="*60)
    log.info(f"  All components loaded in {elapsed:.1f}s — server ready!")
    log.info(f"  RAG:       {STARTUP_STATUS.get('rag_chain', '?')}")
    log.info(f"  Embedding: {STARTUP_STATUS.get('embedding', '?')}")
    log.info(f"  Reranker:  {STARTUP_STATUS.get('reranker', '?')}")
    log.info(f"  LLM:       {STARTUP_STATUS.get('llm', '?')}")
    log.info("="*60)

    yield  # ← server runs here

    log.info("Shutting down RAG Real-Estate Assistant…")

def local_listing_search(message: str, limit: int = 12) -> list[dict]:
    """Small local fallback when RAG infrastructure is unavailable."""
    text = message.lower()
    max_price = None
    price_match = None
    for marker in ("dưới", "<", "tối đa", "không quá"):
        idx = text.find(marker)
        if idx >= 0:
            price_match = text[idx:]
            break
    if price_match:
        num = parse_number(price_match)
        if num is not None:
            max_price = int(num * 1_000_000_000) if "tỷ" in price_match or "ty" in price_match else int(num * 1_000_000)

    bedrooms = None
    for suffix in ("pn", "phòng ngủ"):
        idx = text.find(suffix)
        if idx > 0:
            bedrooms = parse_number(text[max(0, idx - 4):idx + len(suffix)])
            break

    province = ""
    if any(alias in text for alias in ("tp.hcm", "tp hcm", "tphcm", "hồ chí minh", "sài gòn", "saigon")):
        province = "hồ chí minh"
    elif "hà nội" in text or "ha noi" in text:
        province = "hà nội"

    wants_apartment = any(term in text for term in ("căn hộ", "chung cư", "apartment"))
    wants_sale = ("tỷ" in text or "ty" in text) and "thuê" not in text

    def hard_filter(item: dict, use_price: bool = True, use_bedrooms: bool = True) -> bool:
        if wants_sale and item.get("listing_type") != "ban":
            return False
        if province and province not in item.get("province", "").lower():
            return False
        if wants_apartment and "căn hộ" not in item.get("property_type", "").lower() and "chung cư" not in item.get("property_type", "").lower():
            return False
        if use_price and max_price is not None:
            if not item.get("price_vnd") or item["price_vnd"] > max_price:
                return False
        if use_bedrooms and bedrooms is not None and item.get("bedrooms") and int(item["bedrooms"]) != int(bedrooms):
            return False
        return True

    for use_price, use_bedrooms in ((True, True), (False, True), (False, False)):
        filtered = [item for item in LISTINGS if hard_filter(item, use_price=use_price, use_bedrooms=use_bedrooms)]
        if filtered:
            return filtered[:limit]

    tokens = [
        token.strip().lower()
        for token in message.replace(",", " ").replace(".", " ").split()
        if len(token.strip()) >= 2
    ]
    if not tokens:
        return LISTINGS[:limit]

    scored = []
    for item in LISTINGS:
        haystack = " ".join(
            [
                item.get("title", ""),
                item.get("address", ""),
                item.get("project", ""),
                item.get("property_type", ""),
                item.get("province", ""),
                item.get("district", ""),
            ]
        ).lower()
        score = sum(1 for token in tokens if token in haystack)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]] or LISTINGS[:limit]

def serialize_source(doc) -> dict:
    meta = getattr(doc, "metadata", {}) or {}
    record = getattr(doc, "record", None) or {}
    url = meta.get("url") or record.get("url") or ""
    return {
        "collection": getattr(doc, "collection", ""),
        "score": getattr(doc, "score", 0),
        "text": (getattr(doc, "text", "") or "")[:1200],
        "metadata": meta,
        "url": url,
    }

def listings_from_sources(sources: list[dict], fallback_query: str, allow_fallback: bool = True) -> list[dict]:
    source_urls = {source.get("url") for source in sources if source.get("url")}
    if source_urls:
        matched = [item for item in LISTINGS if item.get("url") in source_urls]
        if matched:
            return matched
    if not allow_fallback:
        return []
    return local_listing_search(fallback_query)

# Setup FastAPI App
app = FastAPI(title="Real Estate Assistant RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.get("/api/status")
def get_status():
    """Health check — shows startup status of all RAG components."""
    return {
        "status": "ok" if RAG_CHAIN is not None else "degraded",
        "rag_chain": STARTUP_STATUS.get("rag_chain", "not started"),
        "embedding": STARTUP_STATUS.get("embedding", "not started"),
        "reranker": STARTUP_STATUS.get("reranker", "not started"),
        "llm": STARTUP_STATUS.get("llm", "not started"),
        "startup_seconds": STARTUP_STATUS.get("startup_seconds"),
        "listings_loaded": len(LISTINGS),
    }

# Legacy static routes removed, served via StaticFiles mount below.

@app.get("/api/listings")
def get_listings(
    q: str = None,
    listing_type: str = None,
    province: str = None,
    min_price: int = None,
    max_price: int = None,
    min_area: float = None,
    max_area: float = None,
    bedrooms: int = None,
    bathrooms: int = None,
    sort_by: str = None,
    limit: Optional[int] = 500
):
    rows = LISTINGS
    if listing_type:
        rows = [r for r in rows if r.get("listing_type") == listing_type]
    if province:
        rows = [r for r in rows if province.lower() in r.get("province", "").lower()]
    if q:
        q_lower = q.lower()
        rows = [
            r for r in rows
            if q_lower in " ".join([r.get("title", ""), r.get("address", ""), r.get("project", ""), r.get("property_type", "")]).lower()
        ]
    if min_price is not None:
        rows = [r for r in rows if r.get("price_vnd") and r["price_vnd"] >= min_price]
    if max_price is not None:
        rows = [r for r in rows if r.get("price_vnd") and r["price_vnd"] <= max_price]
    if min_area is not None:
        rows = [r for r in rows if r.get("area_m2") and r["area_m2"] >= min_area]
    if max_area is not None:
        rows = [r for r in rows if r.get("area_m2") and r["area_m2"] <= max_area]
    if bedrooms is not None:
        rows = [r for r in rows if r.get("bedrooms") == bedrooms]
    if bathrooms is not None:
        rows = [r for r in rows if r.get("bathrooms") == bathrooms]
        
    if sort_by == "price_asc":
        rows = sorted(rows, key=lambda x: x.get("price_vnd") or float('inf'))
    elif sort_by == "price_desc":
        rows = sorted(rows, key=lambda x: x.get("price_vnd") or -float('inf'), reverse=True)
    elif sort_by == "area_asc":
        rows = sorted(rows, key=lambda x: x.get("area_m2") or float('inf'))
    elif sort_by == "area_desc":
        rows = sorted(rows, key=lambda x: x.get("area_m2") or -float('inf'), reverse=True)
        
    if limit is None or limit <= 0:
        items = rows
    else:
        items = rows[:limit]

    return {
        "items": items,
        "stats": listing_stats(rows),
        "total": len(rows),
        "returned": len(items),
    }

@app.get("/api/stats")
def get_stats():
    return listing_stats(LISTINGS)

@app.get("/api/listings/{listing_id}/pois")
def get_listing_pois(listing_id: str):
    # Find the listing in LISTINGS to get its url
    listing = None
    for item in LISTINGS:
        if item.get("id") == listing_id:
            listing = item
            break
            
    if not listing:
        return {"pois": [], "error": f"Listing {listing_id} not found locally"}
        
    url = listing.get("url")
    if not url:
        return {"pois": [], "error": "No URL found for this listing"}
        
    # Generate the DB ID using uuid.uuid5
    try:
        raw_key = f"listing_record:{url}:0"
        db_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_key))
    except Exception as e:
        return {"pois": [], "error": f"Failed to generate DB ID: {e}"}
        
    # Query PostgreSQL for nearby POIs
    try:
        from db.postgres_client import PostgresClient
        pg = PostgresClient()
        categories = ["transit_station", "school", "hospital", "park"]
        poi_map = pg.fetch_nearby_pois(
            [db_id], entity_type="listing",
            categories=categories, top_n_per_category=5
        )
        pois = poi_map.get(db_id, [])
        return {"pois": pois}
    except Exception as e:
        log.warning(f"Failed to fetch POIs for listing {listing_id} from PostgreSQL: {e}")
        return {"pois": [], "error": str(e)}

@app.post("/api/chat")
def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    session_id = payload.session_id or "default"

    def sse_stream():
        """Generator that yields SSE-formatted events."""
        try:
            chain = get_rag_chain()
            mapped_listings = None

            for event in chain.query_stream(message, session_id=session_id):
                etype = event.get("type")

                if etype == "metadata":
                    sources = event.get("sources", [])
                    # Keep retrieved candidates for diagnostics/evaluation, but
                    # do not render them as citations before the final answer
                    # declares which URLs were actually used.
                    mapped_listings = listings_from_sources(sources, message)
                    meta_payload = {
                        "intent": event.get("intent", ""),
                        "filters_applied": event.get("filters", {}),
                        "sources": [],
                        "retrieved_sources": event.get("retrieved_sources", sources),
                        "cited_sources": event.get("cited_sources", []),
                        "listings": [],
                        "retrieved_listings": mapped_listings,
                        "llm_used": True,
                    }
                    yield f"event: metadata\ndata: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"

                elif etype == "final_metadata":
                    cited_sources = event.get("cited_sources", [])
                    cited_listings = listings_from_sources(cited_sources, message, allow_fallback=False) if cited_sources else []
                    final_payload = {
                        "intent": event.get("intent", ""),
                        "filters_applied": event.get("filters", {}),
                        "sources": cited_sources,
                        "retrieved_sources": event.get("retrieved_sources", []),
                        "cited_sources": cited_sources,
                        "listings": cited_listings,
                        "llm_used": True,
                    }
                    yield f"event: final_metadata\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

                elif etype in ["thought", "tool_call", "observation"]:
                    text = event.get("text", "")
                    if text:
                        yield f"event: {etype}\ndata: {json.dumps(text, ensure_ascii=False)}\n\n"

                elif etype == "status":
                    text = event.get("text", "")
                    if text:
                        yield f"event: status\ndata: {json.dumps(text, ensure_ascii=False)}\n\n"

                elif etype == "chunk":
                    text = event.get("text", "")
                    if text:
                        # Escape newlines so SSE data field stays on one line
                        yield f"event: chunk\ndata: {json.dumps(text, ensure_ascii=False)}\n\n"

                elif etype == "done":
                    yield "event: done\ndata: {}\n\n"
                    return

        except Exception as exc:
            log.warning(f"RAG streaming failed, sending fallback: {exc}")
            listings = local_listing_search(message)
            meta_payload = {
                "intent": "local_fallback",
                "filters_applied": {},
                "sources": [],
                "retrieved_sources": [],
                "cited_sources": [],
                "listings": listings,
                "llm_used": False,
                "error": str(exc),
            }
            yield f"event: metadata\ndata: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"
            fallback_text = (
                "Tôi chưa kết nối được RAG backend trong phiên này, "
                "nên tạm thời hiển thị các tin đăng khớp từ dữ liệu local."
            )
            yield f"event: chunk\ndata: {json.dumps(fallback_text, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

SERVE_FRONTEND = os.environ.get("SERVE_FRONTEND", "true").lower() not in ("false", "0", "no")
if SERVE_FRONTEND:
    _dist = ROOT / "frontend" / "dist"
    if _dist.exists():
        app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
        log.info(f"Serving frontend from {_dist}")
    else:
        log.warning(f"Frontend dist not found at {_dist}. Run 'npm run build' in frontend/ first, or set SERVE_FRONTEND=false.")
else:
    log.info("SERVE_FRONTEND=false — skipping static files mount (API-only mode)")

def main():
    print("Starting FastAPI web server on http://127.0.0.1:8000")
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
