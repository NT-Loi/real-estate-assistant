"""
Normalizer — parse Vietnamese real estate price, area, and location strings
into structured, filterable values.

Examples:
    parse_price("3,5 tỷ")           → 3500.0   (triệu VND)
    parse_price("850 triệu")        → 850.0
    parse_price("15 triệu/tháng")   → 15.0
    parse_price("Thỏa thuận")       → None

    parse_area("120 m²")            → 120.0
    parse_area("85.5 m²")           → 85.5

    split_location("Quận 7, TP Hồ Chí Minh")
        → ("TP Hồ Chí Minh", "Quận 7")
"""
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------
def parse_price(raw: Optional[str]) -> Optional[float]:
    """
    Parse a Vietnamese price string into millions of VND.
    Returns None for negotiable / unparseable prices.
    """
    if not raw:
        return None

    text = raw.strip().lower()

    # Negotiable
    if any(kw in text for kw in ("thỏa thuận", "thoả thuận", "liên hệ")):
        return None

    # Remove "/ tháng", "/ m²" suffixes (keep the number)
    text = re.sub(r"\s*/\s*(tháng|thang|m²|m2)", "", text)

    # Try to find number + unit
    # Pattern: optional digits.digits or digits,digits  + unit
    match = re.search(
        r"([\d]+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|trieu|tr|nghìn|nghin|ng)",
        text,
    )
    if not match:
        # Try bare number (assume triệu if < 1000, tỷ if >= 1)
        num_match = re.search(r"([\d]+(?:[.,]\d+)?)", text)
        if num_match:
            val = float(num_match.group(1).replace(",", "."))
            # Heuristic: values > 100 are likely triệu
            return val if val > 100 else None
        return None

    num_str = match.group(1).replace(",", ".")
    unit = match.group(2).lower()
    value = float(num_str)

    if unit in ("tỷ", "ty"):
        return value * 1000  # 1 tỷ = 1000 triệu
    elif unit in ("triệu", "trieu", "tr"):
        return value
    elif unit in ("nghìn", "nghin", "ng"):
        return value / 1000  # 1000 nghìn = 1 triệu
    return None


def parse_price_vnd(raw: Optional[str]) -> Optional[int]:
    """Parse a Vietnamese price string into VND."""
    price_million = parse_price(raw)
    if price_million is None:
        return None
    return int(price_million * 1_000_000)


def parse_price_per_m2_vnd(raw: Optional[str]) -> Optional[int]:
    """Parse strings like '225,64 tr/m²' into VND per m²."""
    price_million = parse_price(raw)
    if price_million is None:
        return None
    return int(price_million * 1_000_000)


# ---------------------------------------------------------------------------
# Area parsing
# ---------------------------------------------------------------------------
def parse_area(raw: Optional[str]) -> Optional[float]:
    """Parse area string like '120 m²' into float m²."""
    if not raw:
        return None
    match = re.search(r"([\d]+(?:[.,]\d+)?)\s*m", raw)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


# ---------------------------------------------------------------------------
# Bedroom / bathroom count
# ---------------------------------------------------------------------------
def parse_int_field(raw: Optional[str]) -> Optional[int]:
    """Extract first integer from a string like '3 PN' or '2'."""
    if not raw:
        return None
    match = re.search(r"(\d+)", str(raw))
    return int(match.group(1)) if match else None


def parse_float_field(raw: Optional[str]) -> Optional[float]:
    """Extract first decimal number from strings like '6 m' or '13,5 m'."""
    if raw is None:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)", str(raw))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


# ---------------------------------------------------------------------------
# Location splitting
# ---------------------------------------------------------------------------
# Common city names and their normalized forms
CITY_ALIASES = {
    "hà nội": "Hà Nội",
    "ha noi": "Hà Nội",
    "hn": "Hà Nội",
    "tp hồ chí minh": "TP Hồ Chí Minh",
    "tp. hồ chí minh": "TP Hồ Chí Minh",
    "hồ chí minh": "TP Hồ Chí Minh",
    "ho chi minh": "TP Hồ Chí Minh",
    "hcm": "TP Hồ Chí Minh",
    "tp.hcm": "TP Hồ Chí Minh",
    "tp hcm": "TP Hồ Chí Minh",
    "đà nẵng": "Đà Nẵng",
    "da nang": "Đà Nẵng",
    "hải phòng": "Hải Phòng",
    "hai phong": "Hải Phòng",
    "cần thơ": "Cần Thơ",
    "can tho": "Cần Thơ",
}


def split_location(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Split a Vietnamese location string into (tinh_thanh, quan_huyen).

    Input examples:
        "Quận 7, TP Hồ Chí Minh"
        "Cầu Giấy, Hà Nội"
        "Bình Thạnh, Hồ Chí Minh"
        "TP Hồ Chí Minh"

    Returns:
        (tinh_thanh, quan_huyen) — either may be None
    """
    if not raw:
        return (None, None)

    # Clean up. Batdongsan often appends a second line like
    # "(Phường X, Tỉnh Y mới)", which should not drive canonical filters.
    text = str(raw).splitlines()[0]
    text = re.sub(r"\.\s*Xem bản đồ\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[·\s,]+", "", text).strip()
    text = re.sub(r"[·,]+$", "", text).strip()

    if not text:
        return (None, None)

    # Split by comma
    parts = [p.strip() for p in text.split(",") if p.strip()]

    if len(parts) >= 2:
        # Last part is usually the city/province
        city_candidate = parts[-1].strip()
        district_candidate = parts[-2].strip()

        # Normalize city
        city_lower = city_candidate.lower().strip()
        tinh_thanh = CITY_ALIASES.get(city_lower, city_candidate)

        return (tinh_thanh, district_candidate)

    elif len(parts) == 1:
        # Could be just a city
        city_lower = parts[0].lower().strip()
        if city_lower in CITY_ALIASES:
            return (CITY_ALIASES[city_lower], None)
        # Or just a district (no city info)
        return (None, parts[0])

    return (None, None)


def extract_location_parts(raw: Optional[str]) -> dict:
    """
    Best-effort extraction of Vietnamese address parts.

    Returns province/district/ward text for filtering/reporting. Geocoding is
    intentionally separate because crawled records currently do not include lat/lng.
    """
    if not raw:
        return {"province": None, "district": None, "ward": None}

    text = re.sub(r"\n.*$", "", raw).strip()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    province, district = split_location(text)
    ward = None

    for part in parts:
        lower = part.lower()
        if lower.startswith(("phường ", "xã ", "thị trấn ")):
            ward = part
            break

    return {
        "province": province,
        "district": district,
        "ward": ward,
    }


def normalize_label(raw: Optional[str]) -> Optional[str]:
    """Normalize noisy categorical labels without inventing taxonomy."""
    if raw is None:
        return None
    text = re.sub(r"\s+", " ", str(raw)).strip()
    text = re.sub(r"[.]+$", "", text).strip()
    return text or None


def normalize_listing_metadata(record: dict) -> dict:
    """
    Extract normalized metadata from a merged listing record.
    Returns a flat dict suitable for ChromaDB metadata.
    """
    tinh_thanh, quan_huyen = split_location(
        record.get("dia_chi") or record.get("khu_vuc")
    )
    loc = extract_location_parts(record.get("dia_chi") or record.get("khu_vuc"))

    meta = {
        "listing_id": record.get("id") or "",
        "source_type": record.get("loai_hinh", "ban"),
        "loai_nha_dat": normalize_label(record.get("loai_nha_dat")) or "unknown",
        "tinh_thanh": tinh_thanh or "unknown",
        "quan_huyen": quan_huyen or "unknown",
        "province": loc["province"] or "unknown",
        "district": loc["district"] or "unknown",
        "ward": loc["ward"] or "unknown",
        "gia_raw": record.get("gia") or "",
        "dien_tich_raw": record.get("dien_tich") or "",
        "huong_nha": normalize_label(record.get("huong_nha")) or "unknown",
        "phap_ly": normalize_label(record.get("phap_ly")) or "unknown",
        "noi_that": normalize_label(record.get("noi_that")) or "unknown",
        "url": record.get("url") or "",
        "ngay_dang": record.get("ngay_dang") or "",
    }

    # Numeric fields — ChromaDB supports numeric filtering
    gia = parse_price(record.get("gia"))
    if gia is not None:
        meta["gia_trieu"] = gia
        meta["price_vnd"] = int(gia * 1_000_000)

    price_per_m2 = parse_price_per_m2_vnd(record.get("gia_per_m2"))
    if price_per_m2 is not None:
        meta["price_per_m2_vnd"] = price_per_m2

    area = parse_area(record.get("dien_tich"))
    if area is not None:
        meta["dien_tich_m2"] = area

    bedrooms = parse_int_field(record.get("so_phong_ngu"))
    if bedrooms is not None:
        meta["so_phong_ngu"] = bedrooms

    bathrooms = parse_int_field(record.get("so_phong_tam"))
    if bathrooms is not None:
        meta["so_phong_tam"] = bathrooms

    return meta


def normalize_project_metadata(record: dict) -> dict:
    """Extract normalized metadata from a project record."""
    tinh_thanh, quan_huyen = split_location(
        record.get("dia_chi") or record.get("khu_vuc")
    )
    loc = extract_location_parts(record.get("dia_chi") or record.get("khu_vuc"))
    meta = {
        "project_id": record.get("id") or "",
        "source_type": "du-an",
        "loai_du_an": normalize_label(record.get("loai_du_an")) or "unknown",
        "chu_dau_tu": normalize_label(record.get("chu_dau_tu")) or "unknown",
        "tinh_thanh": tinh_thanh or "unknown",
        "quan_huyen": quan_huyen or "unknown",
        "province": loc["province"] or "unknown",
        "district": loc["district"] or "unknown",
        "ward": loc["ward"] or "unknown",
        "trang_thai": normalize_label(record.get("trang_thai")) or "unknown",
        "gia_raw": record.get("gia") or "",
        "url": record.get("url") or "",
    }
    return meta


def normalize_article_metadata(record: dict) -> dict:
    """Extract normalized metadata from an article (news/wiki) record."""
    meta = {
        "source_type": record.get("loai", "tin-tuc"),
        "danh_muc": record.get("danh_muc") or "unknown",
        "tieu_de": record.get("tieu_de") or "",
        "tac_gia": record.get("tac_gia") or "unknown",
        "ngay_dang": record.get("ngay_dang") or "",
        "url": record.get("url") or "",
    }
    return meta


def normalize_law_metadata(
    dieu: dict,
    ten_luat: str,
    so_hieu: str,
    chuong: str,
    url: str,
) -> dict:
    """Extract normalized metadata for a law article (Điều)."""
    meta = {
        "source_type": "law",
        "ten_luat": ten_luat or "unknown",
        "so_hieu": so_hieu or "unknown",
        "dieu_so": str(dieu.get("dieu_so", "")),
        "tieu_de_dieu": dieu.get("tieu_de", "") or "",
        "chuong": chuong or "unknown",
        "url": url or "",
    }
    return meta
