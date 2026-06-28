"""
Query Intent Parser — LLM-powered structured extraction of intent, filters,
and lifestyle signals from Vietnamese natural language real estate queries.

When the LLM (Gemini) is available it handles all classification.
When it is not, a minimal regex fallback covers basic price/location/bedroom
extraction with a conservative default intent.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from db.normalizer import parse_price, CITY_ALIASES

log = logging.getLogger("bds_query_parser")


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


# ---------------------------------------------------------------------------
# Parsed query result
# ---------------------------------------------------------------------------
@dataclass
class ParsedQuery:
    """Result of parsing a user query."""
    intents: list[str]                  # list of intents: e.g. ["calculate_finance", "lifestyle_search"]
    intent: str                         # comma-separated string for backward compatibility
    query_text: str                    # cleaned query for semantic search
    collections: list[str]             # which collections to search
    filters: dict = field(default_factory=dict)  # Qdrant payload filter
    raw_query: str = ""                # original user query
    lifestyle_signals: list[str] = field(default_factory=list)  # e.g. ["metro", "school", "flood"]


# ---------------------------------------------------------------------------
# Intent → collection mapping (shared by LLM and fallback parsers)
# ---------------------------------------------------------------------------
_INTENT_TO_COLLECTIONS: dict[str, list[str]] = {
    "search_listing":    ["listings"],
    "compare_project":   ["projects", "listings"],
    "ask_knowledge":     ["articles", "listings", "projects"],
    "lifestyle_search":  ["listings", "social_neighborhood", "articles", "projects"],
    "calculate_finance": [],                  # pure calculator intent; combine with search_listing when property search is requested
    "market_report":     [],                  # handled by SQL in chain.py, not Qdrant
}

_VALID_INTENTS = set(_INTENT_TO_COLLECTIONS.keys())

_VALID_SIGNALS = {
    "metro", "school", "hospital", "park", "shopping",
    "flood", "livability", "safety", "appreciation", "infrastructure",
}


# ---------------------------------------------------------------------------
# LLM Parser
# ---------------------------------------------------------------------------

_PARSE_SYSTEM_PROMPT = """\
You are a Vietnamese real estate query analyzer. Output ONLY valid JSON, no explanation.

Output format:
{"intents": ["<value1>", "<value2>"], "lifestyle_signals": [], "filters": {}}

intent values: search_listing | lifestyle_search | compare_project | ask_knowledge | calculate_finance | market_report

Definitions:
- search_listing: buying/selling/renting a specific property
- lifestyle_search: choosing location by lifestyle (metro, school, flood, environment, safety, investment)
- compare_project: comparing projects or areas
- ask_knowledge: general real estate knowledge, legal, tax, process
- calculate_finance: loan, interest rate, affordability calculation
- market_report: market stats, trends, reports

lifestyle_signals (use only these): metro, school, hospital, park, shopping, flood, livability, safety, appreciation, infrastructure

filters keys (only include if clearly mentioned):
- gia_trieu: price in million VND. "3 ty" = {"$lte": 3000}, "2-4 ty" = {"$gte": 2000, "$lte": 4000}
- dien_tich_m2: area in square meters. "460 m2" = {"$gte": 414, "$lte": 506}, "400-500 m2" = {"$gte": 400, "$lte": 500}
- so_phong_ngu: integer number of bedrooms
- tinh_thanh: city/province name in Vietnamese
- quan_huyen: district name
- loai_nha_dat: "Căn hộ chung cư" | "Nhà riêng" | "Đất" | "Nhà biệt thự, liền kề" | "Shophouse"
- loai_hinh: "ban" for buying/sale queries, "cho_thue" for rental queries

Rules: output ONLY the JSON object. No markdown. No explanation.\
"""

_PARSE_PROMPT_TEMPLATE = "Câu hỏi: {query}"

_PARSE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intents": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": sorted(_VALID_INTENTS),
            },
        },
        "lifestyle_signals": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": sorted(_VALID_SIGNALS),
            },
        },
        "filters": {
            "type": "OBJECT",
            "properties": {
                "gia_trieu": {
                    "type": "OBJECT",
                    "properties": {
                        "$gte": {"type": "NUMBER"},
                        "$lte": {"type": "NUMBER"},
                    },
                },
                "dien_tich_m2": {
                    "type": "OBJECT",
                    "properties": {
                        "$gte": {"type": "NUMBER"},
                        "$lte": {"type": "NUMBER"},
                    },
                },
                "so_phong_ngu": {"type": "INTEGER"},
                "tinh_thanh": {"type": "STRING"},
                "quan_huyen": {"type": "STRING"},
                "loai_nha_dat": {
                    "type": "STRING",
                    "enum": ["Căn hộ chung cư", "Nhà riêng", "Đất", "Nhà biệt thự, liền kề", "Shophouse"],
                },
                "loai_hinh": {
                    "type": "STRING",
                    "enum": ["ban", "cho_thue"],
                },
            },
        },
    },
    "required": ["intents", "lifestyle_signals", "filters"],
}


def _repair_json(raw: str) -> Optional[str]:
    """
    Apply a series of repair heuristics to coerce malformed LLM output into
    valid JSON. Returns the repaired string, or None if all attempts fail.
    """
    # 1. Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    raw = raw.strip()

    # 2. Try to extract the outermost {...} block
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        raw = brace_match.group(0)

    # 3. Try as-is first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # 4. Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # 5. Replace Python-style None/True/False
    raw = re.sub(r"\bNone\b", "null", raw)
    raw = re.sub(r"\bTrue\b", "true", raw)
    raw = re.sub(r"\bFalse\b", "false", raw)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # 6. Truncate to last valid closing brace (handles premature stream endings)
    last_brace = raw.rfind("}")
    if last_brace != -1:
        candidate = raw[: last_brace + 1]
        # Count and balance braces
        opens = candidate.count("{")
        closes = candidate.count("}")
        if opens > closes:
            candidate += "}" * (opens - closes)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def _llm_parse(query: str, llm) -> Optional[dict]:
    """Call the LLM to parse the query into structured fields. Returns dict or None."""
    prompt = _PARSE_PROMPT_TEMPLATE.format(query=query)
    try:
        if hasattr(llm, "generate_json"):
            parsed = llm.generate_json(
                prompt=prompt,
                system_prompt=_PARSE_SYSTEM_PROMPT,
                response_schema=_PARSE_RESPONSE_SCHEMA,
                max_tokens=1024,
                temperature=0.0,
            )
            if isinstance(parsed, dict):
                return parsed

        raw = llm.generate(
            prompt=prompt,
            system_prompt=_PARSE_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.0,
        )
        if not raw:
            return None

        repaired = _repair_json(raw)
        if repaired is None:
            log.warning(f"LLM parse failed: could not repair JSON. Raw: {raw[:120]!r}")
            return None

        return json.loads(repaired)
    except Exception as e:
        log.warning(f"LLM parse failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Regex fallback helpers (used when LLM unavailable)
# ---------------------------------------------------------------------------

_CITY_KEYWORDS = {
    "hà nội": "Hà Nội", "ha noi": "Hà Nội",
    "hồ chí minh": "TP Hồ Chí Minh", "sài gòn": "TP Hồ Chí Minh",
    "tp hcm": "TP Hồ Chí Minh", "tphcm": "TP Hồ Chí Minh",
    "đà nẵng": "Đà Nẵng", "hải phòng": "Hải Phòng",
    "cần thơ": "Cần Thơ", "bình dương": "Bình Dương",
    "đồng nai": "Đồng Nai", "long an": "Long An",
    "vũng tàu": "Bà Rịa - Vũng Tàu", "nha trang": "Khánh Hòa",
    "thủ đức": "TP Hồ Chí Minh",
}

_DISTRICT_PATTERNS = [
    r"(?:\bquận\b|\bquan\b|\bq\.?)\s*(\d+|[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+)",
    r"(?:\bhuyện\b|\bh\.|\bh\s+)\s*([a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+)",
    r"(?:\bthành phố\b|\btp\.)\s+([a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+)",
]

_PRICE_PATTERNS = [
    (r"từ\s+(\d+(?:[.,]\d+)?)\s*(?:đến|tới)\s+(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|tr)", "range"),
    (r"dưới\s+(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|tr)", "lte"),
    (r"trên\s+(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|tr)", "gte"),
    (r"khoảng\s+(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|tr)", "approx"),
    (r"(\d+(?:[.,]\d+)?)\s*(tỷ|ty)(?:\s|$|,)", "lte"),
]

_AREA_PATTERNS = [
    (r"(?:diện tích|dien tich|dt|rộng|rong)\s*(?:từ\s*)?(\d+(?:[.,]\d+)?)\s*(?:-|đến|toi|tới)\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét vuông|met vuong)", "range"),
    (r"(?:diện tích|dien tich|dt|rộng|rong)\s*(?:dưới|duoi|tối đa|toi da)\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét vuông|met vuong)", "lte"),
    (r"(?:diện tích|dien tich|dt|rộng|rong)\s*(?:trên|tren|từ|tu|tối thiểu|toi thieu)\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét vuông|met vuong)", "gte"),
    (r"(?:diện tích|dien tich|dt|rộng|rong)\s*(?:khoảng|khoang|tầm|tam)?\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét vuông|met vuong)", "approx"),
]

_BEDROOM_PATTERNS = [
    r"(\d+)\s*(?:phòng ngủ|pn|phong ngu|bedroom)",
    r"(\d+)\s*pn\b",
]

_PROPERTY_TYPE_MAP = [
    ("Căn hộ chung cư", ["căn hộ", "chung cư", "apartment"]),
    ("Nhà riêng", ["nhà riêng", "nhà phố"]),
    ("Nhà biệt thự, liên kề", ["biệt thự", "liên kề", "villa"]),
    ("Nhà mặt phố", ["nhà mặt phố", "mặt tiền"]),
    ("Đất", ["đất bán", "bán đất", "tìm đất", "đất nền", "đất thổ cư", "lô đất"]),
    ("Phòng trọ", ["phòng trọ", "trọ"]),
    ("Shophouse", ["shophouse"]),
]

_PROPERTY_TYPE_ALIASES = {
    "can ho chung cu": "Căn hộ chung cư",
    "căn hộ chung cư": "Căn hộ chung cư",
    "chung cu": "Căn hộ chung cư",
    "chung cư": "Căn hộ chung cư",
    "nha rieng": "Nhà riêng",
    "nhà riêng": "Nhà riêng",
    "nha pho": "Nhà riêng",
    "nhà phố": "Nhà riêng",
    "dat": "Đất",
    "đất": "Đất",
    "biet thu": "Nhà biệt thự, liền kề",
    "biệt thự": "Nhà biệt thự, liền kề",
    "shophouse": "Shophouse",
}


def _regex_extract_filters(text: str) -> dict:
    """Extract structured filters using regex (fallback only)."""
    filters: dict = {}

    # Price
    for pattern, ptype in _PRICE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        if ptype == "range":
            low = float(m.group(1).replace(",", "."))
            high = float(m.group(2).replace(",", "."))
            unit = m.group(3).lower()
            mul = 1000 if unit in ("tỷ", "ty") else 1
            filters["gia_trieu"] = {"$gte": low * mul, "$lte": high * mul}
        elif ptype in ("lte", "approx"):
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2).lower()
            mul = 1000 if unit in ("tỷ", "ty") else 1
            val *= mul
            filters["gia_trieu"] = {"$gte": val * 0.8, "$lte": val * 1.2} if ptype == "approx" else {"$lte": val}
        elif ptype == "gte":
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2).lower()
            mul = 1000 if unit in ("tỷ", "ty") else 1
            filters["gia_trieu"] = {"$gte": val * mul}
        break

    # Area
    for pattern, atype in _AREA_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        if atype == "range":
            low = float(m.group(1).replace(",", "."))
            high = float(m.group(2).replace(",", "."))
            filters["dien_tich_m2"] = {"$gte": low, "$lte": high}
        elif atype == "lte":
            val = float(m.group(1).replace(",", "."))
            filters["dien_tich_m2"] = {"$lte": val}
        elif atype == "gte":
            val = float(m.group(1).replace(",", "."))
            filters["dien_tich_m2"] = {"$gte": val}
        elif atype == "approx":
            val = float(m.group(1).replace(",", "."))
            filters["dien_tich_m2"] = {"$gte": val * 0.9, "$lte": val * 1.1}
        break

    # City
    for kw, city in {**_CITY_KEYWORDS, **CITY_ALIASES}.items():
        if kw in text:
            filters["tinh_thanh"] = city
            break

    # District
    for pat in _DISTRICT_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            d = re.sub(r"[,.\s]+$", "", m.group(1).strip().title())
            d_norm = _strip_accents(d.lower())
            if len(d) > 1 and d_norm not in {"ho chi minh", "ha noi", "da nang", "hai phong", "can tho"}:
                if "thành phố" in pat:
                    filters["quan_huyen"] = f"Thành phố {d}"
                else:
                    filters["quan_huyen"] = f"Quận {d}" if d[0].isdigit() else d
            break

    # Bedrooms
    for pat in _BEDROOM_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            filters["so_phong_ngu"] = int(m.group(1))
            break

    # Property type
    for prop_type, keywords in _PROPERTY_TYPE_MAP:
        if any(kw in text for kw in keywords):
            filters["loai_nha_dat"] = prop_type
            break

    return filters


_LIFESTYLE_SIGNAL_MAP: list[tuple[str, list[str]]] = [
    ("metro",          ["metro", "tàu điện", "tàu metro", "gần metro", "ga metro"]),
    ("school",         ["trường học", "gần trường", "trường tiểu học", "có trường"]),
    ("hospital",       ["bệnh viện", "gần bệnh viện", "y tế"]),
    ("park",           ["công viên", "cây xanh", "không gian xanh"]),
    ("shopping",       ["siêu thị", "mua sắm", "trung tâm thương mại"]),
    ("flood",          ["ngập", "ít ngập", "không ngập", "ngập lụt"]),
    ("livability",     ["môi trường sống", "môi trường", "chất lượng sống"]),
    ("safety",         ["an ninh", "an toàn", "yên tĩnh"]),
    ("appreciation",   ["tiềm năng", "tăng giá", "sinh lời", "đầu tư dài hạn"]),
    ("infrastructure", ["hạ tầng", "quy hoạch", "phát triển hạ tầng"]),
]


def _extract_lifestyle_signals(text: str) -> list[str]:
    return [sig for sig, phrases in _LIFESTYLE_SIGNAL_MAP if any(p in text for p in phrases)]


def _infer_listing_type(text: str) -> Optional[str]:
    """Infer sale/rental intent from the original Vietnamese query."""
    rent_terms = [
        "thuê",
        "thue",
        "cho thuê",
        "cho thue",
        "rent",
        "rental",
    ]
    sale_terms = [
        "mua",
        "bán",
        "ban",
        "cần mua",
        "can mua",
        "tìm mua",
        "tim mua",
    ]
    if any(term in text for term in rent_terms):
        return "cho_thue"
    if any(term in text for term in sale_terms):
        return "ban"
    return None


def _canonicalize_filters(filters: dict) -> dict:
    filters = dict(filters or {})

    for key in ("gia_trieu", "dien_tich_m2"):
        value = filters.get(key)
        if isinstance(value, dict):
            normalized = {}
            if "$gte" in value or "gte" in value or "min" in value:
                normalized["$gte"] = value.get("$gte", value.get("gte", value.get("min")))
            if "$lte" in value or "lte" in value or "max" in value:
                normalized["$lte"] = value.get("$lte", value.get("lte", value.get("max")))
            filters[key] = {k: v for k, v in normalized.items() if v is not None}

    prop_type = filters.get("loai_nha_dat")
    if prop_type:
        key = _strip_accents(str(prop_type).strip().lower())
        filters["loai_nha_dat"] = _PROPERTY_TYPE_ALIASES.get(key, prop_type)
    return filters


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Parse Vietnamese real estate queries into structured ParsedQuery.

    Uses the LLM (Gemini) for intent classification and field extraction
    when available. Falls back to regex-based extraction otherwise.
    The LLM path is more accurate for ambiguous queries.
    """

    def __init__(self, llm=None):
        self._llm = llm  # injected or lazy-initialized on first parse

    def _get_llm(self):
        """Lazy-initialize the LLM client."""
        if self._llm is None:
            try:
                from rag.llm import LLMClient
                self._llm = LLMClient()
            except Exception:
                self._llm = False  # mark unavailable
        return self._llm if self._llm else None

    def parse(self, query: str) -> ParsedQuery:
        """Parse a user query into a structured ParsedQuery."""
        raw = query.strip()
        text = raw.lower()

        llm = self._get_llm()
        parsed_data = None

        if llm and getattr(llm, "is_available", False):
            parsed_data = _llm_parse(raw, llm)

        if parsed_data:
            return self._from_llm_output(raw, parsed_data)
        else:
            log.info("LLM unavailable or parse failed — using regex fallback")
            return self._regex_fallback(raw, text)

    def _from_llm_output(self, raw: str, data: dict) -> ParsedQuery:
        """Build ParsedQuery from validated LLM JSON output."""
        raw_intents = data.get("intents") or []
        if not raw_intents and "intent" in data:
            raw_intents = [data["intent"]]
        intents = [i for i in raw_intents if i in _VALID_INTENTS]
        if not intents:
            intents = ["ask_knowledge"]

        raw_signals = data.get("lifestyle_signals") or []
        signals = [s for s in raw_signals if s in _VALID_SIGNALS]

        # If any intent is lifestyle but no signals extracted, run regex signal extractor
        if "lifestyle_search" in intents and not signals:
            signals = _extract_lifestyle_signals(raw.lower())

        raw_filters = data.get("filters") or {}
        filters = {k: v for k, v in raw_filters.items() if v is not None and v != ""}
        inferred_listing_type = _infer_listing_type(raw.lower())
        if inferred_listing_type and "loai_hinh" not in filters:
            filters["loai_hinh"] = inferred_listing_type
        filters = _canonicalize_filters(filters)

        # Determine union of all collections for all detected intents
        collections = []
        for i in intents:
            for coll in _INTENT_TO_COLLECTIONS.get(i, []):
                if coll not in collections:
                    collections.append(coll)
        # Default collection fallback only for open-ended knowledge queries.
        # Calculator/market intents intentionally use tools or deterministic
        # code paths and should not automatically search Qdrant collections.
        if not collections and "ask_knowledge" in intents:
            collections = ["articles", "listings", "projects"]

        return ParsedQuery(
            intents=intents,
            intent=", ".join(intents),
            query_text=raw,
            collections=collections,
            filters=filters,
            raw_query=raw,
            lifestyle_signals=signals,
        )

    def _regex_fallback(self, raw: str, text: str) -> ParsedQuery:
        """Conservative regex-based fallback when LLM is unavailable."""
        intents: list[str] = []
        lifestyle_signals: list[str] = []

        if any(kw in text for kw in [
            "lãi suất", "trả góp", "khoản vay", "vay ngân hàng", "lãi vay",
            "vay ", "vay mua", "thu nhập", "khả năng tài chính",
            "mỗi tháng trả", "trả bao nhiêu", "dư nợ",
        ]):
            intents.append("calculate_finance")
        if any(kw in text for kw in ["so sánh", "khác gì", "tốt hơn", "nên chọn"]):
            intents.append("compare_project")
        if any(kw in text for kw in ["báo cáo", "thị trường", "xu hướng", "thống kê"]):
            intents.append("market_report")
        if any(kw in text for kw in [
            "metro", "ngập", "trường học", "môi trường sống", "an ninh",
            "tiềm năng", "tăng giá", "hạ tầng", "muốn ở", "phù hợp gia đình",
        ]):
            intents.append("lifestyle_search")
            lifestyle_signals = _extract_lifestyle_signals(text)
        if any(kw in text for kw in [
            "bán nhà", "mua nhà", "thuê nhà", "cho thuê", "căn hộ",
            "đất nền", "đất bán", "bán đất", "dat ban", "ban dat",
            "biệt thự", "nhà riêng", "tìm nhà", "tìm đất", "tim dat",
        ]):
            intents.append("search_listing")

        if not intents:
            intents.append("ask_knowledge")

        filters = _regex_extract_filters(text)
        inferred_listing_type = _infer_listing_type(text)
        if inferred_listing_type:
            filters["loai_hinh"] = inferred_listing_type
        filters = _canonicalize_filters(filters)

        collections = []
        for i in intents:
            for coll in _INTENT_TO_COLLECTIONS.get(i, []):
                if coll not in collections:
                    collections.append(coll)
        if not collections and "ask_knowledge" in intents:
            collections = ["articles", "listings", "projects"]

        return ParsedQuery(
            intents=intents,
            intent=", ".join(intents),
            query_text=raw,
            collections=collections,
            filters=filters,
            raw_query=raw,
            lifestyle_signals=lifestyle_signals,
        )
