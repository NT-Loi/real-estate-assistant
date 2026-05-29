from __future__ import annotations

import json
import math
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAG_CHAIN = None
RAG_INIT_ERROR = None


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
    return str(raw or "").splitlines()[0].replace(". Xem bản đồ", "").strip()


def location_parts(address: str, fallback_area: str = "") -> dict[str, str]:
    parts = [p.strip() for p in first_line(address or fallback_area).split(",") if p.strip()]
    province = parts[-1] if parts else fallback_area
    district = ""
    ward = ""
    for part in parts:
        lower = part.lower()
        if lower.startswith(("quận ", "huyện ", "thành phố ", "tp ")):
            district = part
        if lower.startswith(("phường ", "xã ", "thị trấn ")):
            ward = part
    province = province.replace("Hồ Chí Minh mới", "Hồ Chí Minh").replace("Ninh Bình mới", "Ninh Bình")
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


def load_listings() -> list[dict]:
    rows: list[dict] = []
    for filename in ("listings_ban.json", "listings_cho_thue.json"):
        path = DATA_DIR / filename
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for idx, record in enumerate(records):
            loc = location_parts(record.get("dia_chi", ""), record.get("khu_vuc", ""))
            lat, lng, precision = approximate_coords(record)
            price = price_vnd(record.get("gia"))
            area = parse_number(record.get("dien_tich"))
            rows.append(
                {
                    "id": f"{record.get('loai_hinh', 'listing')}-{idx}",
                    "listing_type": record.get("loai_hinh") or "",
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
    """Initialize RAG lazily so static map/listing UI survives DB outages."""
    global RAG_CHAIN, RAG_INIT_ERROR
    if RAG_CHAIN is not None:
        return RAG_CHAIN
    if RAG_INIT_ERROR is not None:
        raise RAG_INIT_ERROR
    try:
        from rag.chain import RAGChain

        RAG_CHAIN = RAGChain()
        return RAG_CHAIN
    except Exception as exc:
        RAG_INIT_ERROR = exc
        raise


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


def listings_from_sources(sources: list[dict], fallback_query: str) -> list[dict]:
    source_urls = {source.get("url") for source in sources if source.get("url")}
    if source_urls:
        matched = [item for item in LISTINGS if item.get("url") in source_urls]
        if matched:
            return matched
    return local_listing_search(fallback_query)


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = WEB_DIR / ("index.html" if parsed.path == "/" else parsed.path.lstrip("/"))
        if not path.resolve().is_relative_to(WEB_DIR.resolve()) or not path.exists():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/listings":
            self.send_json(self.filter_listings(parse_qs(parsed.query)))
            return
        if parsed.path == "/api/stats":
            self.send_json(listing_stats(LISTINGS))
            return
        path = WEB_DIR / ("index.html" if parsed.path == "/" else parsed.path.lstrip("/"))
        if not path.resolve().is_relative_to(WEB_DIR.resolve()) or not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw_body)
            message = str(payload.get("message") or "").strip()
        except Exception:
            self.send_json({"error": "Invalid JSON body", "answer": "", "sources": [], "listings": []}, status=400)
            return

        if not message:
            self.send_json({"error": "Message is required", "answer": "", "sources": [], "listings": []}, status=400)
            return

        try:
            result = get_rag_chain().query(message)
            sources = [serialize_source(doc) for doc in result.sources]
            listings = listings_from_sources(sources, message)
            self.send_json(
                {
                    "answer": result.answer,
                    "intent": result.intent,
                    "filters_applied": result.filters_applied,
                    "sources": sources,
                    "listings": listings,
                    "llm_used": result.llm_used,
                }
            )
        except Exception as exc:
            listings = local_listing_search(message)
            self.send_json(
                {
                    "answer": (
                        "Tôi chưa kết nối được RAG backend trong phiên này, "
                        "nên tạm thời hiển thị các tin đăng khớp từ dữ liệu local."
                    ),
                    "intent": "local_fallback",
                    "filters_applied": {},
                    "sources": [],
                    "listings": listings,
                    "error": str(exc),
                }
            )

    def filter_listings(self, query: dict[str, list[str]]) -> dict:
        rows = LISTINGS
        listing_type = query.get("listing_type", [""])[0]
        province = query.get("province", [""])[0].lower()
        q = query.get("q", [""])[0].lower()
        if listing_type:
            rows = [r for r in rows if r["listing_type"] == listing_type]
        if province:
            rows = [r for r in rows if province in r["province"].lower()]
        if q:
            rows = [
                r for r in rows
                if q in " ".join([r["title"], r["address"], r["project"], r["property_type"]]).lower()
            ]
        return {"items": rows[:500], "stats": listing_stats(rows)}

    def send_json(self, payload: object, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Web frontend: http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
