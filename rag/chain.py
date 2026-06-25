"""
RAG Chain — End-to-end query pipeline for the real estate assistant.

Pipeline: User Query → Parse → Retrieve (+ POI enrich) → Generate/Format → Response

Usage:
    from rag.chain import RAGChain
    chain = RAGChain()
    response = chain.query("Tôi có 3 tỷ, muốn ở gần metro, ít ngập, có trường học")
    print(response.answer)
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from rag.query_parser import QueryParser, ParsedQuery
from rag.retriever import Retriever, RetrievedDocument
from rag.prompts import (
    SYSTEM_PROMPT,
    FINANCE_PROMPT,
    get_prompt_template,
    format_context,
)
from rag.llm import LLMClient
from rag.finance_calculator import FinanceCalculator
from db.vectorstore import VectorStore

log = logging.getLogger("bds_chain")
trace_log = logging.getLogger("bds_agent_trace")


def _setup_trace_logger() -> logging.Logger:
    """Configure JSONL trace logging for LLM/tool debugging."""
    if trace_log.handlers:
        return trace_log

    trace_path = os.getenv("AGENT_TRACE_LOG", "logs/agent_trace.log")
    path = Path(trace_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    trace_log.addHandler(handler)
    trace_log.setLevel(logging.INFO)
    trace_log.propagate = False
    return trace_log


def _clip_for_trace(value: Any, max_chars: int = 12000) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "\n...[trace truncated]..."
    if isinstance(value, dict):
        return {k: _clip_for_trace(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_for_trace(v, max_chars=max_chars) for v in value[:50]]
    return value


TRACE_TEXT_FIELDS = {
    "answer",
    "candidate_answer",
    "draft",
    "observation",
    "raw_final",
    "response",
}


def _trace_text_summary(value: Any) -> dict[str, Any]:
    text = str(value or "")
    return {
        "redacted": True,
        "chars": len(text),
        "sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else "",
    }


def _sanitize_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove generated/tool text from agent_trace unless explicitly enabled."""
    if _env_flag("AGENT_TRACE_INCLUDE_TEXT", False):
        return payload

    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if key in TRACE_TEXT_FIELDS:
            sanitized[f"{key}_summary"] = _trace_text_summary(value)
        elif key == "function_calls" and isinstance(value, list):
            sanitized[key] = [
                {
                    "name": item.get("name", "") if isinstance(item, dict) else "",
                    "args": item.get("args", {}) if isinstance(item, dict) else {},
                }
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid float for %s=%r; using %s", name, raw, default)
        return default


def _chat_tool_temperature() -> float:
    """Low temperature for tool planning/function calls."""
    return _env_float("CHAT_TOOL_TEMPERATURE", _env_float("LLM_TOOL_TEMPERATURE", 0.1))


def _chat_answer_temperature() -> float:
    """Low-but-readable temperature for the final user-facing answer."""
    return _env_float("CHAT_ANSWER_TEMPERATURE", _env_float("LLM_TEMPERATURE", 0.3))


def _is_llm_failure_text(text: str | None) -> bool:
    normalized = _normalize_text(text or "")
    return "khong the tao cau tra loi" in normalized and "vui long thu lai" in normalized


def _extract_balanced_json_object(text: str, start: int = 0) -> tuple[str | None, int]:
    """Extract the first balanced JSON object in text starting at or after index."""
    brace_start = text.find("{", start)
    if brace_start == -1:
        return None, -1

    depth = 0
    in_string = False
    escape = False
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : idx + 1], idx + 1

    return None, -1


def _parse_action_call(response: str) -> tuple[str | None, str | None]:
    """Find the first valid ReAct action call in a model response."""
    for match in re.finditer(r"Action:\s*`?(\w+)`?", response, re.IGNORECASE):
        tool_name = match.group(1).strip()
        args_str, _ = _extract_balanced_json_object(response, match.end())
        if not args_str:
            return tool_name, None
        try:
            json.loads(args_str)
            return tool_name, args_str
        except Exception:
            # Try later Action blocks if the model emitted more than one.
            continue
    return None, None


def _db_listing_type(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"cho_thue", "thue", "rent", "rental"}:
        return "cho-thue"
    if normalized in {"ban", "mua", "sale", "sell"}:
        return "ban"
    return str(value or "").strip()


HCMC_METRO_STATIONS: list[dict[str, Any]] = [
    {"name": "Ga Metro Bến Thành", "line": "Metro số 1", "lat": 10.7721, "lon": 106.6983, "district": "Quận 1"},
    {"name": "Ga Nhà hát Thành phố", "line": "Metro số 1", "lat": 10.7767, "lon": 106.7033, "district": "Quận 1"},
    {"name": "Ga Ba Son", "line": "Metro số 1", "lat": 10.7807, "lon": 106.7082, "district": "Quận 1"},
    {"name": "Ga Văn Thánh", "line": "Metro số 1", "lat": 10.7959, "lon": 106.7158, "district": "Bình Thạnh"},
    {"name": "Ga Tân Cảng", "line": "Metro số 1", "lat": 10.7984, "lon": 106.7231, "district": "Bình Thạnh"},
    {"name": "Ga Thảo Điền", "line": "Metro số 1", "lat": 10.8023, "lon": 106.7338, "district": "Thủ Đức"},
    {"name": "Ga An Phú", "line": "Metro số 1", "lat": 10.8025, "lon": 106.7427, "district": "Thủ Đức"},
    {"name": "Ga Rạch Chiếc", "line": "Metro số 1", "lat": 10.8114, "lon": 106.7567, "district": "Thủ Đức"},
    {"name": "Ga Phước Long", "line": "Metro số 1", "lat": 10.8215, "lon": 106.7649, "district": "Thủ Đức"},
    {"name": "Ga Bình Thái", "line": "Metro số 1", "lat": 10.8311, "lon": 106.7642, "district": "Thủ Đức"},
    {"name": "Ga Thủ Đức", "line": "Metro số 1", "lat": 10.8465, "lon": 106.7714, "district": "Thủ Đức"},
    {"name": "Ga Khu Công nghệ cao", "line": "Metro số 1", "lat": 10.8630, "lon": 106.7870, "district": "Thủ Đức"},
    {"name": "Ga Đại học Quốc gia", "line": "Metro số 1", "lat": 10.8780, "lon": 106.8020, "district": "Thủ Đức"},
    {"name": "Ga Suối Tiên", "line": "Metro số 1", "lat": 10.8790, "lon": 106.8140, "district": "Thủ Đức"},
]

POI_CATEGORY_ALIASES: dict[str, list[str]] = {
    "transit_station": ["transit_station", "metro", "ga metro", "tàu điện", "tau dien", "bến xe", "ben xe", "bus"],
    "school": ["school", "trường", "truong", "đại học", "dai hoc", "mầm non", "mam non"],
    "hospital": ["hospital", "bệnh viện", "benh vien", "phòng khám", "phong kham", "y tế", "y te"],
    "park": ["park", "công viên", "cong vien", "cây xanh", "cay xanh"],
    "shopping": ["shopping", "mall", "siêu thị", "sieu thi", "trung tâm thương mại", "tttm", "chợ", "cho"],
    "airport": ["airport", "sân bay", "san bay"],
    "landmark": ["landmark", "địa danh", "dia danh", "landmark"],
}

POI_CATEGORY_LABELS = {
    "transit_station": "giao thông công cộng",
    "school": "trường học",
    "hospital": "bệnh viện/y tế",
    "park": "công viên",
    "shopping": "mua sắm",
    "airport": "sân bay",
    "landmark": "địa danh",
}


def _normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _metro_station_matches(station_name: str | None, city: str | None = None) -> list[dict[str, Any]]:
    city_norm = _normalize_text(city)
    if city_norm and not any(token in city_norm for token in ("ho chi minh", "hcm", "sai gon", "thu duc")):
        return []

    query_norm = _normalize_text(station_name)
    if not query_norm or query_norm in {"metro", "ga metro", "tau dien", "tau dien ngam"}:
        return HCMC_METRO_STATIONS

    if "ben thanh" in query_norm:
        return [station for station in HCMC_METRO_STATIONS if "ben thanh" in _normalize_text(station["name"])]

    stopwords = {
        "ga", "metro", "tau", "dien", "ngam", "gan", "near", "ben", "thanh",
        "pho", "tp", "hcm", "ho", "chi", "minh", "quan", "q",
    }
    query_tokens = {tok for tok in query_norm.split() if tok not in stopwords}
    if "metro" in query_norm and not query_tokens:
        return HCMC_METRO_STATIONS
    matches: list[dict[str, Any]] = []
    for station in HCMC_METRO_STATIONS:
        station_norm = _normalize_text(station["name"])
        station_tokens = {tok for tok in station_norm.split() if tok not in stopwords}
        if query_norm in station_norm or station_norm in query_norm or query_tokens.intersection(station_tokens):
            matches.append(station)
    return matches


def _infer_poi_category(query: str | None, category: str | None = None) -> str | None:
    if category:
        cat_norm = _normalize_text(category)
        for canonical, aliases in POI_CATEGORY_ALIASES.items():
            if canonical == category or cat_norm in {_normalize_text(a) for a in aliases}:
                return canonical
        return category

    query_norm = _normalize_text(query)
    for canonical, aliases in POI_CATEGORY_ALIASES.items():
        if any(_normalize_text(alias) in query_norm for alias in aliases):
            return canonical
    return None


def _curated_pois_for_query(
    query: str | None,
    category: str | None = None,
    city: str | None = None,
) -> list[dict[str, Any]]:
    inferred = _infer_poi_category(query, category)
    if inferred not in {None, "transit_station"}:
        return []

    query_norm = _normalize_text(query)
    if inferred == "transit_station" or any(token in query_norm for token in ("metro", "tau dien", "ga metro")):
        return [
            {
                "id": f"curated_hcm_metro_{idx}",
                "name": station["name"],
                "category": "transit_station",
                "address": f'{station["district"]}, TP Hồ Chí Minh',
                "latitude": station["lat"],
                "longitude": station["lon"],
                "source": "curated_hcm_metro_station_table",
                "district": station["district"],
            }
            for idx, station in enumerate(_metro_station_matches(query, city), 1)
        ]
    return []


# ---------------------------------------------------------------------------
# Finance parameter extraction
# ---------------------------------------------------------------------------

def _compute_finance_summary(parsed: ParsedQuery) -> str:
    """
    Extract loan parameters from ParsedQuery and run FinanceCalculator.
    Returns exact pre-computed numbers as a string for LLM context.
    No LLM arithmetic — the formulas are deterministic.
    """
    q = parsed.raw_query.lower()
    f = parsed.filters

    # --- Principal from filters (user said "tôi có X tỷ") ---
    budget_vnd: Optional[float] = None
    gia = f.get("gia_trieu")
    if isinstance(gia, dict):
        lte = gia.get("$lte")
        if lte:
            budget_vnd = lte * 1_000_000  # triệu → VND

    # --- Annual interest rate from query ---
    rate_pct = FinanceCalculator.__class__  # just alias for defaults
    annual_rate = 9.0  # default
    rate_match = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:/\s*năm|năm)?", q)
    if rate_match:
        annual_rate = float(rate_match.group(1).replace(",", "."))

    # --- Term from query ---
    term_years = 20  # default
    term_match = re.search(r"(\d+)\s*năm", q)
    if term_match:
        term_years = int(term_match.group(1))
        term_years = max(1, min(term_years, 35))  # sanity check

    # --- Down payment % from query ---
    down_pct = 0.30  # default 30%
    down_match = re.search(r"(?:trả trước|down payment)\s*(\d+)\s*%", q)
    if down_match:
        down_pct = int(down_match.group(1)) / 100

    # --- Monthly income (for affordability) ---
    income_vnd: Optional[float] = None
    income_match = re.search(r"(?:thu nhập|lương|kiếm)\s*(?:khoảng\s*)?(\d+(?:[.,]\d+)?)\s*(triệu|tr|tỷ)", q)
    if income_match:
        v = float(income_match.group(1).replace(",", "."))
        unit = income_match.group(2).lower()
        income_vnd = v * 1_000_000_000 if unit == "tỷ" else v * 1_000_000

    parts = []

    if budget_vnd:
        # Full loan plan from budget
        plan = FinanceCalculator.loan_from_property_price(
            property_price_vnd=budget_vnd,
            down_payment_pct=down_pct,
            annual_rate_pct=annual_rate,
            term_years=term_years,
        )
        parts.append(plan.summary_text())

        # Multi-rate scenario
        scenarios = FinanceCalculator.multi_scenario(plan.principal_vnd)
        parts.append("\n--- So sánh theo lãi suất ---")
        for s in scenarios:
            def fmt(v):
                return f"{v/1_000_000:.0f} triệu" if v < 1_000_000_000 else f"{v/1_000_000_000:.2f} tỷ"
            parts.append(
                f"  {s.annual_rate_pct:.1f}%/năm → {fmt(s.monthly_payment_vnd)}/tháng "
                f"(tổng lãi: {fmt(s.total_interest_vnd)})"
            )

    if income_vnd:
        afford = FinanceCalculator.max_affordable_loan(
            monthly_income_vnd=income_vnd,
            annual_rate_pct=annual_rate,
            term_years=term_years,
        )
        parts.append("\n" + afford.summary_text())

    if not parts:
        # Generic example with defaults
        example = FinanceCalculator.monthly_payment(
            principal_vnd=2_000_000_000,  # 2 tỷ example
            annual_rate_pct=annual_rate,
            term_years=term_years,
        )
        parts.append("(Ví dụ với khoản vay 2 tỷ)\n" + example.summary_text())

    return "\n".join(parts)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str                              # Generated or formatted answer
    sources: list[RetrievedDocument]         # Retrieved source documents
    intent: str                              # Detected query intent
    filters_applied: dict                    # Metadata filters that were applied
    parsed_query: Optional[ParsedQuery] = None  # Full parsed query details
    llm_used: bool = False                   # Whether LLM was used for generation
    effective_query: str = ""                # Query after multi-turn merge, if any


@dataclass
class ConversationState:
    """Small in-memory state for pending clarification follow-ups."""
    pending_query: str = ""
    pending_questions: list[str] = field(default_factory=list)
    last_answer: str = ""


class RAGChain:
    """
    Agentic RAG pipeline for Vietnamese real estate queries using a ReAct loop.
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        llm: Optional[LLMClient] = None,
    ):
        self._store = store or VectorStore()
        self._retriever = Retriever(self._store)
        self._llm = llm or LLMClient()
        self._parser = QueryParser(llm=self._llm)
        self._agent_sources = []
        self._current_parsed: Optional[ParsedQuery] = None
        self._conversations: dict[str, ConversationState] = {}

        # Optional warm-up. Keep disabled by default so keyword/tools/UI can
        # start even when the embedding model is not cached locally yet.
        eager_load_embedder = os.getenv("RAG_EAGER_LOAD_EMBEDDER", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if eager_load_embedder and hasattr(self._store, "_embedder") and hasattr(self._store._embedder, "_load"):
            self._store._embedder._load()

        log.info("RAG Chain initialized in Agentic mode")
        log.info(f"  LLM available: {self._llm.is_available}")
        log.info(f"  Collections: {self._store.stats()}")

    def _trace(self, event: str, **fields: Any) -> None:
        """Write a structured JSONL trace entry for agent debugging."""
        try:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **fields,
            }
            _setup_trace_logger().info(
                json.dumps(_clip_for_trace(_sanitize_trace_payload(payload)), ensure_ascii=False, default=str)
            )
        except Exception as exc:
            log.debug("Agent trace logging failed: %s", exc)

    def _get_conversation(self, session_id: Optional[str]) -> Optional[ConversationState]:
        if not session_id:
            return None
        return self._conversations.setdefault(session_id, ConversationState())

    def _is_reset_message(self, message: str) -> bool:
        norm = _normalize_text(message)
        return norm in {
            "reset",
            "clear",
            "xoa",
            "xoa bo nho",
            "bat dau lai",
            "hoi moi",
            "new search",
            "tim kiem moi",
        }

    def _prepare_query_for_session(self, user_query: str, session_id: Optional[str]) -> str:
        """Merge a clarification follow-up with the previous pending query."""
        state = self._get_conversation(session_id)
        if state is None:
            return user_query

        if self._is_reset_message(user_query):
            self._conversations.pop(session_id or "", None)
            return user_query

        if state.pending_query:
            merged = (
                f"{state.pending_query}\n\n"
                f"Thông tin bổ sung từ người dùng trong lượt sau: {user_query}"
            )
            state.pending_query = ""
            state.pending_questions = []
            return merged

        return user_query

    def _remember_or_clear_conversation(
        self,
        session_id: Optional[str],
        effective_query: str,
        parsed: ParsedQuery,
        answer: str,
    ) -> None:
        state = self._get_conversation(session_id)
        if state is None:
            return

        if self._should_ask_clarification_only(parsed, answer):
            state.pending_query = effective_query
            state.pending_questions = self._clarification_questions(parsed)
            state.last_answer = answer
        else:
            state.pending_query = ""
            state.pending_questions = []
            state.last_answer = answer

    # ---------------------------------------------------------------------------
    # Tools definition
    # ---------------------------------------------------------------------------
    def _default_collections(self, collections: Optional[list[str]]) -> list[str]:
        if collections:
            return collections
        if self._current_parsed and self._current_parsed.collections:
            return self._current_parsed.collections
        return ["articles", "social_neighborhood"]

    def _current_filters(self) -> dict:
        return self._current_parsed.filters if self._current_parsed else {}

    def _current_lifestyle_signals(self) -> list[str]:
        return self._current_parsed.lifestyle_signals if self._current_parsed else []

    def _tool_semantic_search(self, query_text: str, collections: Optional[list[str]] = None, limit: int = 5) -> str:
        """Run semantic search on Qdrant."""
        try:
            collections = self._default_collections(collections)
            log.info(f"Tool called: semantic_search({query_text=}, {collections=}, {limit=})")
            docs = self._retriever.retrieve(
                query_text=query_text,
                collections=collections,
                filters=self._current_filters(),
                top_k=limit,
                per_collection_k=max(12, limit * 4),
                lifestyle_signals=self._current_lifestyle_signals(),
            )
            if not docs:
                return "Không tìm thấy kết quả tìm kiếm ngữ nghĩa nào."
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi tìm kiếm ngữ nghĩa: {e}"

    def _tool_hybrid_search(self, query_text: str, collections: Optional[list[str]] = None, limit: int = 5) -> str:
        """Run dense + keyword retrieval, followed by reranking when available."""
        try:
            collections = self._default_collections(collections)
            log.info(f"Tool called: hybrid_search({query_text=}, {collections=}, {limit=})")
            docs = self._retriever.hybrid_retrieve(
                query_text=query_text,
                collections=collections,
                filters=self._current_filters(),
                top_k=limit,
                per_collection_k=max(20, limit * 6),
                lifestyle_signals=self._current_lifestyle_signals(),
            )
            if not docs:
                return "Không tìm thấy kết quả phù hợp từ tìm kiếm hybrid."
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi tìm kiếm hybrid: {e}"

    def _tool_keyword_search(self, query_text: str, collections: Optional[list[str]] = None, limit: int = 5) -> str:
        """Run exact keyword search using ILIKE on PostgreSQL."""
        try:
            collections = self._default_collections(collections)
            log.info(f"Tool called: keyword_search({query_text=}, {collections=}, {limit=})")
            docs = self._retriever.keyword_retrieve(
                query_text=query_text,
                collections=collections,
                top_k=limit,
            )
            if not docs:
                return "Không tìm thấy kết quả từ khóa chính xác nào."
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi tìm kiếm từ khóa: {e}"

    def _tool_filter_listings(self, **kwargs) -> str:
        """Run structured filtering of listings on PostgreSQL."""
        try:
            log.info(f"Tool called: filter_listings({kwargs=})")
            clauses = ["is_active = TRUE"]
            params = []
            
            price_max = kwargs.get("price_max_trieu")
            if price_max is not None:
                clauses.append("gia_trieu <= %s")
                params.append(float(price_max))
                
            price_min = kwargs.get("price_min_trieu")
            if price_min is not None:
                clauses.append("gia_trieu >= %s")
                params.append(float(price_min))
                
            bedrooms = kwargs.get("bedrooms")
            if bedrooms is not None:
                clauses.append("so_phong_ngu = %s")
                params.append(int(bedrooms))

            listing_type = kwargs.get("loai_hinh") or kwargs.get("listing_type")
            if listing_type:
                clauses.append("loai_hinh = %s")
                params.append(_db_listing_type(listing_type))
                
            tinh_thanh = kwargs.get("tinh_thanh")
            if tinh_thanh:
                clauses.append("province ILIKE %s")
                params.append(f"%{tinh_thanh}%")
                
            quan_huyen = kwargs.get("quan_huyen")
            if quan_huyen:
                clauses.append("(district ILIKE %s OR province ILIKE %s OR khu_vuc ILIKE %s)")
                params.append(f"%{quan_huyen}%")
                params.append(f"%{quan_huyen}%")
                params.append(f"%{quan_huyen}%")
                
            prop_type = kwargs.get("property_type")
            if prop_type:
                clauses.append("loai_nha_dat ILIKE %s")
                params.append(f"%{prop_type}%")
                
            lat = kwargs.get("lat")
            lon = kwargs.get("lon")
            order_by = ""
            if lat is not None and lon is not None:
                lat = float(lat)
                lon = float(lon)
                radius_km = float(kwargs.get("radius_km", 2.0))
                # Haversine distance formula in km
                haversine_expr = """
                    (6371 * acos(least(1.0, 
                        cos(radians(%s)) * cos(radians(latitude)) * 
                        cos(radians(longitude) - radians(%s)) + 
                        sin(radians(%s)) * sin(radians(latitude))
                    )))
                """
                clauses.append(f"{haversine_expr} <= %s")
                params.extend([lat, lon, lat, radius_km])
                order_by = f"ORDER BY {haversine_expr} ASC"
                params.extend([lat, lon, lat])
                
            limit = int(kwargs.get("limit", 5))
            
            where_sql = " AND ".join(clauses)
            sql = f"SELECT id, raw_json FROM listings WHERE {where_sql} {order_by} LIMIT %s"
            params.append(limit)
            
            results = []
            with self._store.pg.get_cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
                for row_id, raw_payload in rows:
                    payload = raw_payload
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    if isinstance(payload, dict):
                        payload["id"] = payload.get("id") or row_id
                    results.append(payload)
                    
            if not results:
                return "Không tìm thấy bất động sản nào khớp với bộ lọc."
                
            from rag.retriever import RetrievedDocument
            docs = []
            for r in results:
                desc = r.get("mo_ta_chi_tiet") or r.get("mo_ta") or r.get("tieu_de") or ""
                title = r.get("tieu_de", "Tin đăng")
                price = r.get("gia", "Thỏa thuận")
                area = r.get("dien_tich", "Chưa rõ")
                addr = r.get("dia_chi") or r.get("khu_vuc") or ""
                url = r.get("url", "")
                text_content = f"{title}. Giá: {price}, Diện tích: {area}, Địa chỉ: {addr}. Chi tiết: {desc[:500]}"
                
                doc = RetrievedDocument(
                    text=text_content,
                    metadata={
                        "url": url,
                        "source_record_id": r.get("id") or "",
                        "chunk_type": "structured_filter",
                    },
                    score=1.0,
                    collection="listings",
                    record=r
                )
                docs.append(doc)
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi lọc bất động sản: {e}"

    def _tool_find_nearby_pois(
        self,
        entity_ids: list[str],
        entity_type: str = "listing",
        categories: Optional[list[str]] = None,
        radius_m: float = 1500,
        top_n_per_category: int = 5,
    ) -> str:
        """Find nearby amenities for listing/project source IDs."""
        try:
            if not entity_ids:
                return "Thiếu entity_ids để tìm tiện ích lân cận."
            categories = categories or ["transit_station", "school", "hospital", "park", "shopping"]
            poi_map = self._store.pg.fetch_nearby_pois(
                entity_ids=entity_ids,
                entity_type=entity_type,
                categories=categories,
                radius_m=radius_m,
                top_n_per_category=top_n_per_category,
            )
            lines = []
            for entity_id, pois in poi_map.items():
                lines.append(f"Tiện ích quanh {entity_type} {entity_id}:")
                if not pois:
                    lines.append("- Chưa có POI trong bán kính yêu cầu.")
                    continue
                for poi in pois:
                    dist = poi.get("distance_m")
                    dist_text = f"{dist:.0f}m" if isinstance(dist, (int, float)) else "chưa rõ khoảng cách"
                    lines.append(
                        f"- {poi.get('category')}: {poi.get('name')} ({dist_text})"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"Lỗi khi tìm tiện ích lân cận: {e}"

    def _tool_find_pois_near_location(
        self,
        location_name: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        categories: Optional[list[str]] = None,
        radius_m: float = 2000,
        top_n_per_category: int = 8,
    ) -> str:
        """Find POIs around an arbitrary place/project name or coordinate."""
        try:
            categories = categories or ["school", "hospital", "park"]
            resolved = None
            if lat is None or lon is None:
                if not location_name:
                    return "Thiếu location_name hoặc lat/lon để tìm POI theo bán kính."
                resolved = self._store.pg.resolve_location_point(location_name)
                if not resolved:
                    return (
                        f"Không tìm thấy tọa độ nội bộ đủ tin cậy cho: {location_name}. "
                        "Hãy dùng tên dự án/tin đăng rõ hơn hoặc cung cấp lat/lon."
                    )
                lat = float(resolved["latitude"])
                lon = float(resolved["longitude"])

            radius_m = max(100, min(float(radius_m or 2000), 10000))
            pois = self._store.pg.fetch_pois_near_point(
                lat=float(lat),
                lon=float(lon),
                categories=categories,
                radius_m=radius_m,
                top_n_per_category=max(1, min(int(top_n_per_category or 8), 20)),
            )

            loc_label = location_name or f"{lat}, {lon}"
            if resolved:
                loc_label = f"{resolved.get('name') or location_name} ({resolved.get('source_type')})"
            lines = [f"Tiện ích trong bán kính {radius_m:.0f}m quanh {loc_label}:"]
            if resolved:
                lines.append(
                    f"Tọa độ đã dùng: {resolved.get('latitude')}, {resolved.get('longitude')} - "
                    f"{resolved.get('address') or 'không rõ địa chỉ'}"
                )
            if not pois:
                lines.append("- Chưa có POI phù hợp trong bán kính yêu cầu.")
                return "\n".join(lines)

            for poi in pois:
                dist = poi.get("distance_m")
                dist_text = f"{dist:.0f}m" if isinstance(dist, (int, float)) else "chưa rõ khoảng cách"
                rating = poi.get("rating")
                rating_text = f", rating {rating}" if rating is not None else ""
                lines.append(
                    f"- {poi.get('category')}: {poi.get('name')} ({dist_text}{rating_text})"
                    + (f" - {poi.get('address')}" if poi.get("address") else "")
                )

            from rag.retriever import RetrievedDocument
            doc = RetrievedDocument(
                text="\n".join(lines),
                metadata={
                    "location_name": location_name or "",
                    "lat": lat,
                    "lon": lon,
                    "radius_m": radius_m,
                    "categories": categories,
                    "resolved": resolved or {},
                },
                score=1.0,
                collection="pois_near_location",
                record={"pois": pois, "resolved": resolved},
            )
            self._agent_sources.append(doc)
            return "\n".join(lines)
        except Exception as e:
            return f"Lỗi khi tìm POI quanh địa điểm: {e}"

    def _find_poi_candidates(
        self,
        poi_query: Optional[str] = None,
        category: Optional[str] = None,
        city: str = "TP Hồ Chí Minh",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find POIs from curated data plus the local PostgreSQL POI cache."""
        inferred_category = _infer_poi_category(poi_query, category)
        limit = max(1, min(int(limit), 50))
        candidates = _curated_pois_for_query(poi_query, inferred_category, city)
        explicit_category = bool(category)

        query_norm = _normalize_text(poi_query)
        category_aliases = {
            _normalize_text(alias)
            for alias in POI_CATEGORY_ALIASES.get(inferred_category or "", [])
        }
        generic_category_query = bool(
            inferred_category
            and (
                not query_norm
                or query_norm in category_aliases
                or any(alias and alias in query_norm for alias in category_aliases)
            )
        )

        clauses = ["latitude IS NOT NULL", "longitude IS NOT NULL"]
        params: list[Any] = []
        if inferred_category:
            aliases = POI_CATEGORY_ALIASES.get(inferred_category, [inferred_category])
            category_parts = ["category ILIKE %s"]
            params.append(f"%{inferred_category}%")
            if not explicit_category:
                for alias in aliases:
                    category_parts.append("name ILIKE %s")
                    params.append(f"%{alias}%")
            clauses.append("(" + " OR ".join(category_parts) + ")")

        if poi_query and not generic_category_query:
            clauses.append("(name ILIKE %s OR address ILIKE %s OR category ILIKE %s)")
            like = f"%{poi_query}%"
            params.extend([like, like, like])

        city_norm = _normalize_text(city)
        if city_norm and any(tok in city_norm for tok in ("ho chi minh", "hcm", "sai gon", "thu duc")):
            clauses.append(
                "((latitude BETWEEN 10.3 AND 11.2 AND longitude BETWEEN 106.2 AND 107.2) "
                "OR address ILIKE %s OR address ILIKE %s OR address ILIKE %s OR name ILIKE %s)"
            )
            params.extend(["%Hồ Chí Minh%", "%HCM%", "%Sài Gòn%", "%Thủ Đức%"])
        elif city:
            clauses.append("(address ILIKE %s OR name ILIKE %s)")
            params.extend([f"%{city}%", f"%{city}%"])

        sql = f"""
            SELECT id, name, category, address, latitude, longitude, source
            FROM pois
            WHERE {" AND ".join(clauses)}
            ORDER BY
                CASE WHEN name ILIKE %s THEN 0 ELSE 1 END,
                name
            LIMIT %s
        """
        params.extend([f"%{poi_query or ''}%", limit])

        try:
            with self._store.pg.get_cursor() as cur:
                cur.execute(sql, tuple(params))
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    candidates.append(dict(zip(cols, row)))
        except Exception as e:
            log.warning(f"POI search failed: {e}")

        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, Any]] = []
        for poi in candidates:
            lat = poi.get("latitude")
            lon = poi.get("longitude")
            if lat is None or lon is None:
                continue
            key = (
                _normalize_text(poi.get("name")),
                f"{float(lat):.5f}",
                f"{float(lon):.5f}",
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(poi)
            if len(unique) >= limit:
                break
        return unique

    def _tool_search_pois(
        self,
        poi_query: Optional[str] = None,
        category: Optional[str] = None,
        city: str = "TP Hồ Chí Minh",
        limit: int = 10,
    ) -> str:
        """Search local/curated POIs by query/category/city."""
        try:
            log.info(f"Tool called: search_pois({poi_query=}, {category=}, {city=}, {limit=})")
            pois = self._find_poi_candidates(poi_query, category, city, limit)
            if not pois:
                return (
                    "Không tìm thấy POI phù hợp trong dữ liệu nội bộ. "
                    "Nếu đây là địa danh cụ thể, hãy dùng `search_location` để geocode."
                )
            lines = ["Các POI phù hợp:"]
            for poi in pois:
                lines.append(
                    "- {name} | category={category} | lat={lat}, lon={lon} | {address}".format(
                        name=poi.get("name") or "",
                        category=poi.get("category") or "",
                        lat=poi.get("latitude"),
                        lon=poi.get("longitude"),
                        address=poi.get("address") or "",
                    )
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Lỗi khi tìm POI: {e}"

    def _tool_find_listings_near_pois(
        self,
        poi_query: Optional[str] = None,
        category: Optional[str] = None,
        city: str = "TP Hồ Chí Minh",
        radius_km: float = 1.5,
        loai_hinh: Optional[str] = None,
        property_type: Optional[str] = None,
        limit: int = 8,
        poi_limit: int = 12,
    ) -> str:
        """Find listings near local/curated POIs for any location-related query."""
        try:
            log.info(
                "Tool called: find_listings_near_pois("
                f"{poi_query=}, {category=}, {city=}, {radius_km=}, {loai_hinh=}, "
                f"{property_type=}, {limit=}, {poi_limit=})"
            )
            pois = self._find_poi_candidates(poi_query, category, city, poi_limit)
            if not pois:
                return (
                    "Không tìm thấy POI nội bộ phù hợp để lọc tin đăng theo bán kính. "
                    "Nếu người dùng nêu địa danh cụ thể, hãy dùng `search_location` rồi `filter_listings`."
                )

            active_filters = self._current_filters()
            loai_hinh = loai_hinh or active_filters.get("loai_hinh")
            if loai_hinh:
                loai_hinh = _db_listing_type(loai_hinh)
            property_type = property_type or active_filters.get("loai_nha_dat")
            price_filter = active_filters.get("gia_trieu")
            bedrooms = active_filters.get("so_phong_ngu")

            radius_km = max(0.2, min(float(radius_km), 15.0))
            limit = max(1, min(int(limit), 40))
            per_poi_limit = max(2, min(limit, 8))
            distance_expr = """
                (6371 * acos(least(1.0,
                    cos(radians(%s)) * cos(radians(latitude)) *
                    cos(radians(longitude) - radians(%s)) +
                    sin(radians(%s)) * sin(radians(latitude))
                )))
            """

            city_norm = _normalize_text(city)
            hcm_city = bool(city_norm and any(tok in city_norm for tok in ("ho chi minh", "hcm", "sai gon", "thu duc")))
            candidates: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            with self._store.pg.get_cursor() as cur:
                for poi in pois:
                    lat = float(poi["latitude"])
                    lon = float(poi["longitude"])
                    clauses = ["is_active = TRUE", "latitude IS NOT NULL", "longitude IS NOT NULL"]
                    params: list[Any] = [lat, lon, lat]

                    if loai_hinh:
                        clauses.append("loai_hinh = %s")
                        params.append(str(loai_hinh))
                    if property_type:
                        clauses.append("loai_nha_dat ILIKE %s")
                        params.append(f"%{property_type}%")
                    if bedrooms is not None:
                        clauses.append("so_phong_ngu = %s")
                        params.append(int(bedrooms))
                    if isinstance(price_filter, dict):
                        if price_filter.get("$lte") is not None:
                            clauses.append("price_vnd <= %s")
                            params.append(float(price_filter["$lte"]) * 1_000_000)
                        if price_filter.get("$gte") is not None:
                            clauses.append("price_vnd >= %s")
                            params.append(float(price_filter["$gte"]) * 1_000_000)
                    if hcm_city:
                        clauses.append("(province ILIKE %s OR dia_chi ILIKE %s OR khu_vuc ILIKE %s)")
                        params.extend(["%Hồ Chí Minh%", "%Hồ Chí Minh%", "%Hồ Chí Minh%"])
                    elif city:
                        clauses.append("(province ILIKE %s OR dia_chi ILIKE %s OR khu_vuc ILIKE %s)")
                        params.extend([f"%{city}%", f"%{city}%", f"%{city}%"])

                    where_sql = " AND ".join(clauses)
                    sql = f"""
                        SELECT id, raw_json, district, distance_km
                        FROM (
                            SELECT id, raw_json, district, {distance_expr} AS distance_km
                            FROM listings
                            WHERE {where_sql}
                        ) ranked
                        WHERE distance_km <= %s
                        ORDER BY distance_km ASC
                        LIMIT %s
                    """
                    params.extend([radius_km, per_poi_limit])
                    cur.execute(sql, tuple(params))
                    for row_id, raw_payload, listing_district, distance_km in cur.fetchall():
                        if row_id in seen_ids:
                            continue
                        payload = raw_payload
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        if not isinstance(payload, dict):
                            continue

                        poi_category = str(poi.get("category") or "")
                        listing_text = _normalize_text(
                            " ".join(
                                str(payload.get(key) or "")
                                for key in ("tieu_de", "dia_chi", "khu_vuc", "mo_ta", "mo_ta_chi_tiet")
                            )
                        )
                        poi_name_norm = _normalize_text(poi.get("name"))
                        poi_area = _normalize_text(poi.get("district") or poi.get("address"))
                        listing_area = _normalize_text(listing_district or payload.get("dia_chi") or payload.get("khu_vuc"))
                        poi_phrase = poi_name_norm.replace("ga metro ", "").replace("ga ", "")
                        mentions_poi = poi_phrase and (poi_phrase in listing_text)
                        suspicious_near_zero = (
                            poi_category == "transit_station"
                            and float(distance_km) < 0.2
                            and poi_area
                            and listing_area
                            and poi_area not in listing_area
                            and listing_area not in poi_area
                            and "metro" not in listing_text
                            and not mentions_poi
                        )
                        if suspicious_near_zero:
                            continue

                        payload["id"] = payload.get("id") or row_id
                        payload["_nearest_poi_name"] = poi.get("name")
                        payload["_nearest_poi_category"] = poi.get("category")
                        payload["_nearest_poi_distance_km"] = float(distance_km)
                        candidates.append(payload)
                        seen_ids.add(row_id)

            if not candidates:
                label = POI_CATEGORY_LABELS.get(_infer_poi_category(poi_query, category) or "", category or poi_query or "POI")
                return f"Không tìm thấy tin đăng phù hợp trong bán kính {radius_km:g} km quanh {label}."

            candidates.sort(key=lambda item: item.get("_nearest_poi_distance_km", 999.0))
            docs: list[RetrievedDocument] = []
            for r in candidates[:limit]:
                title = r.get("tieu_de") or "Tin đăng"
                price = r.get("gia") or "Thỏa thuận"
                area = r.get("dien_tich") or "Chưa rõ"
                addr = r.get("dia_chi") or r.get("khu_vuc") or ""
                poi_name = r.get("_nearest_poi_name") or "POI"
                poi_category = r.get("_nearest_poi_category") or ""
                distance = r.get("_nearest_poi_distance_km")
                distance_text = f"{distance:.2f} km" if isinstance(distance, (int, float)) else "chưa rõ"
                desc = r.get("mo_ta_chi_tiet") or r.get("mo_ta") or ""
                text_content = (
                    f"{title}. Giá: {price}, Diện tích: {area}, Địa chỉ: {addr}. "
                    f"Gần POI: {poi_name} ({poi_category}, {distance_text}). Chi tiết: {desc[:700]}"
                )
                docs.append(
                    RetrievedDocument(
                        text=text_content,
                        metadata={
                            "url": r.get("url") or "",
                            "source_record_id": r.get("id") or "",
                            "chunk_type": "near_poi_filter",
                            "nearest_poi_name": poi_name,
                            "nearest_poi_category": poi_category,
                            "nearest_poi_distance_km": distance,
                        },
                        score=max(0.0, 1.0 - min(float(distance or 0), radius_km) / max(radius_km, 0.1)),
                        collection="listings",
                        record=r,
                    )
                )
            self._agent_sources.extend(docs)
            return format_context(docs, max_chars=9000)
        except Exception as e:
            return f"Lỗi khi tìm bất động sản gần POI: {e}"

    def _tool_search_metro_stations(
        self,
        city: str = "TP Hồ Chí Minh",
        station_name: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Backward-compatible alias for search_pois(category='transit_station')."""
        query = station_name or "metro"
        return self._tool_search_pois(
            poi_query=query,
            category="transit_station",
            city=city,
            limit=limit,
        )

    def _tool_find_listings_near_metro(
        self,
        city: str = "TP Hồ Chí Minh",
        station_name: Optional[str] = None,
        radius_km: float = 1.5,
        loai_hinh: Optional[str] = None,
        property_type: Optional[str] = None,
        limit: int = 8,
    ) -> str:
        """Backward-compatible alias for find_listings_near_pois(category='transit_station')."""
        query = station_name or "metro"
        return self._tool_find_listings_near_pois(
            poi_query=query,
            category="transit_station",
            city=city,
            radius_km=radius_km,
            loai_hinh=loai_hinh,
            property_type=property_type,
            limit=limit,
        )

    def _tool_analyze_market_trend(self, **kwargs) -> str:
        """Analyze historical market trends and compare property price."""
        try:
            log.info(f"Tool called: analyze_market_trend({kwargs=})")
            tinh_thanh = kwargs.get("tinh_thanh")
            quan_huyen = kwargs.get("quan_huyen")
            property_type = kwargs.get("property_type")
            listing_type = kwargs.get("listing_type") or kwargs.get("loai_hinh")
            target_price_vnd = kwargs.get("target_price_vnd")
            target_area_m2 = kwargs.get("target_area_m2")
            
            if not tinh_thanh or not quan_huyen:
                return "Vui lòng cung cấp cả Tỉnh/Thành phố và Quận/Huyện để phân tích thị trường."
            
            # Fetch past 12 months data
            rows = self._store.pg.fetch_market_stats(
                province=tinh_thanh,
                district=quan_huyen,
                listing_type=_db_listing_type(listing_type) if listing_type else None,
                property_type=property_type,
                months=12
            )
            
            if not rows:
                return f"Không tìm thấy dữ liệu thống kê thị trường lịch sử cho khu vực {quan_huyen}, {tinh_thanh}."
                
            # Formatting the result
            output = [f"BÁO CÁO PHÂN TÍCH XU HƯỚNG THỊ TRƯỜNG: {quan_huyen}, {tinh_thanh}\n"]
            
            # Target property analysis
            if target_price_vnd and target_area_m2 and target_area_m2 > 0:
                target_price_per_m2 = float(target_price_vnd) / float(target_area_m2)
                # Find the latest median price from rows
                latest_row = rows[0]
                latest_median_per_m2 = float(latest_row.get("median_price_per_m2_vnd") or 0)
                
                output.append("1. ĐỐI CHIẾU GIÁ BĐS CỤ THỂ:")
                output.append(f"- Giá BĐS đang xét: {target_price_vnd/1e9:.2f} tỷ ({target_price_per_m2/1e6:.1f} triệu/m2)")
                if latest_median_per_m2 > 0:
                    output.append(f"- Mặt bằng giá chung hiện tại: {latest_median_per_m2/1e6:.1f} triệu/m2")
                    diff_pct = ((target_price_per_m2 - latest_median_per_m2) / latest_median_per_m2) * 100
                    if diff_pct > 0:
                        output.append(f"-> ĐÁNH GIÁ: Giá BĐS này cao hơn mặt bằng chung {diff_pct:.1f}%.")
                    else:
                        output.append(f"-> ĐÁNH GIÁ: Giá BĐS này rẻ hơn mặt bằng chung {abs(diff_pct):.1f}%.")
                else:
                    output.append("-> Không có đủ dữ liệu mặt bằng giá chung để đối chiếu.")
                output.append("")
                
            output.append("2. LỊCH SỬ BIẾN ĐỘNG GIÁ (12 THÁNG GẦN NHẤT):")
            for r in rows:
                period = r.get('period', '')
                median_m2 = float(r.get('median_price_per_m2_vnd') or 0)
                count = r.get('listing_count', 0)
                if median_m2 > 0:
                    output.append(f"- Tháng {period}: {median_m2/1e6:.1f} triệu/m2 ({count} tin đăng)")

            from rag.retriever import RetrievedDocument, _format_market_row
            self._agent_sources.extend([
                RetrievedDocument(
                    text=_format_market_row(row),
                    metadata=row,
                    score=1.0,
                    collection="market_snapshots",
                    record=row,
                )
                for row in rows[:20]
            ])
            
            return "\n".join(output)
        except Exception as e:
            return f"Lỗi khi phân tích xu hướng giá: {e}"

    def _tool_get_market_statistics(self, **kwargs) -> str:
        """Get PostgreSQL-based aggregated statistics snapshots."""
        try:
            log.info(f"Tool called: get_market_statistics({kwargs=})")
            tinh_thanh = kwargs.get("tinh_thanh")
            quan_huyen = kwargs.get("quan_huyen")
            property_type = kwargs.get("property_type") or kwargs.get("loai_nha_dat")
            listing_type = kwargs.get("listing_type") or kwargs.get("loai_hinh")
            
            filters = {}
            if tinh_thanh:
                filters["tinh_thanh"] = tinh_thanh
            if quan_huyen:
                filters["quan_huyen"] = quan_huyen
            if property_type:
                filters["loai_nha_dat"] = property_type
            if listing_type:
                filters["loai_hinh"] = _db_listing_type(listing_type)
                
            docs = self._retriever.retrieve_market_report(filters=filters, months=12)
            if not docs:
                return "Không tìm thấy dữ liệu báo cáo thị trường cho khu vực này."
                
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi truy vấn thống kê thị trường: {e}"



    def _tool_web_search(self, query: str, limit: int = 5) -> str:
        """Search the web using Tavily."""
        try:
            log.info(f"Tool called: web_search({query=}, {limit=})")
            import os
            import requests
            
            api_key = os.getenv("TAVILY_API") or os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Lỗi: Không tìm thấy TAVILY_API trong file .env"
                
            limit = max(3, min(int(limit), 8))
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "max_results": limit
            }
            
            r = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
            if r.status_code != 200:
                return f"Lỗi từ Tavily API: {r.text}"
                
            data = r.json()
            results = data.get("results", [])
            
            if not results:
                return f"Không tìm thấy kết quả nào trên web cho: {query}"
            
            output = []
            for i, res in enumerate(results):
                output.append(f"Kết quả {i+1}:\nTiêu đề: {res.get('title')}\nURL: {res.get('url')}\nTrích đoạn: {res.get('content')}\n")
            output.append(
                "Gợi ý sử dụng công cụ: nếu các trích đoạn trên chưa đủ kết luận, "
                "hãy gọi `read_url` với URL phù hợp hoặc `web_research` để trích xuất nội dung chi tiết."
            )
            
            from rag.retriever import RetrievedDocument
            doc = RetrievedDocument(
                text="\n".join(output),
                metadata={"url": "Tavily Search"},
                score=1.0,
                collection="web_search",
                record=None
            )
            self._agent_sources.append(doc)
            return "\n".join(output)
        except Exception as e:
            return f"Lỗi khi tìm kiếm web: {e}"

    def _tool_web_research(self, query: str, limit: int = 5, extract_top: int = 2) -> str:
        """Search Tavily and extract full text from the top URLs in one step."""
        try:
            log.info(f"Tool called: web_research({query=}, {limit=}, {extract_top=})")
            import os
            import requests

            api_key = os.getenv("TAVILY_API") or os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Lỗi: Không tìm thấy TAVILY_API trong file .env"

            limit = max(3, min(int(limit), 8))
            extract_top = max(1, min(int(extract_top), 3, limit))
            search_payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": limit,
            }
            search_resp = requests.post("https://api.tavily.com/search", json=search_payload, timeout=20)
            search_resp.raise_for_status()
            search_data = search_resp.json()
            results = search_data.get("results", [])
            if not results:
                return f"Không tìm thấy kết quả nào trên web cho: {query}"

            urls = [r.get("url") for r in results[:extract_top] if r.get("url")]
            output = [f"Tổng hợp web cho truy vấn: {query}"]
            if search_data.get("answer"):
                output.append(f"\nTóm tắt Tavily: {search_data.get('answer')}")

            for idx, res in enumerate(results, 1):
                output.append(
                    f"\nKết quả {idx}: {res.get('title')}\nURL: {res.get('url')}\nTrích đoạn: {res.get('content')}"
                )

            if urls:
                extract_payload = {"api_key": api_key, "urls": urls}
                extract_resp = requests.post("https://api.tavily.com/extract", json=extract_payload, timeout=25)
                extract_resp.raise_for_status()
                extract_data = extract_resp.json()
                extracted = extract_data.get("results", [])
                if extracted:
                    output.append("\nNội dung trích xuất từ các nguồn hàng đầu:")
                    for item in extracted:
                        content = (item.get("raw_content") or "").strip()
                        if len(content) > 2500:
                            content = content[:2500] + "\n...[Nội dung bị cắt bớt]..."
                        output.append(f"\nNguồn: {item.get('url')}\n{content}")

            from rag.retriever import RetrievedDocument
            doc = RetrievedDocument(
                text="\n".join(output),
                metadata={"url": "Tavily Research"},
                score=1.0,
                collection="web_search",
                record=None
            )
            self._agent_sources.append(doc)
            return "\n".join(output)
        except Exception as e:
            return f"Lỗi khi nghiên cứu web: {e}"

    def _tool_search_location(self, location_name: str) -> str:
        """Search for a location using OpenStreetMap Nominatim API."""
        try:
            log.info(f"Tool called: search_location({location_name=})")
            import requests

            metro_matches = _metro_station_matches(location_name, "TP Hồ Chí Minh")
            if len(metro_matches) == 1:
                station = metro_matches[0]
                return json.dumps({
                    "lat": station["lat"],
                    "lon": station["lon"],
                    "display_name": f'{station["name"]}, {station["district"]}, TP Hồ Chí Minh',
                    "confidence": "curated",
                    "source": "internal_hcm_metro_station_table",
                }, ensure_ascii=False)
            if len(metro_matches) > 1 and "metro" in _normalize_text(location_name):
                return (
                    "Địa danh metro còn mơ hồ. Không dùng tọa độ một ga bất kỳ. "
                    "Hãy gọi `search_metro_stations` hoặc `find_listings_near_metro`."
                )

            headers = {"User-Agent": "RealEstateAssistant/1.0"}
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location_name, "format": "json", "limit": 5, "countrycodes": "vn"},
                headers=headers,
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                return f"Không tìm thấy tọa độ cho địa danh: {location_name}"

            query_tokens = {
                tok for tok in _normalize_text(location_name).split()
                if tok not in {"ga", "metro", "tp", "thanh", "pho", "quan", "q"}
            }
            candidates = []
            for item in data:
                display_name = item.get("display_name", "")
                display_tokens = set(_normalize_text(display_name).split())
                overlap = len(query_tokens.intersection(display_tokens))
                candidates.append({
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "display_name": display_name,
                    "confidence_score": overlap,
                })

            best = max(candidates, key=lambda c: c["confidence_score"])
            if query_tokens and best["confidence_score"] == 0:
                return json.dumps({
                    "error": "Không tìm thấy tọa độ đủ tin cậy; không nên dùng lat/lon này để lọc BĐS.",
                    "query": location_name,
                    "candidates": candidates[:3],
                }, ensure_ascii=False)
            return json.dumps({
                "lat": best["lat"],
                "lon": best["lon"],
                "display_name": best["display_name"],
                "confidence": "nominatim_candidate",
                "candidates": candidates[:3],
            }, ensure_ascii=False)
        except Exception as e:
            return f"Lỗi khi tìm kiếm vị trí bản đồ: {e}"

    def _tool_read_url(self, url: str) -> str:
        """Read full content of a URL using Tavily Extract API."""
        try:
            log.info(f"Tool called: read_url({url=})")
            import os
            import requests
            
            api_key = os.getenv("TAVILY_API") or os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Lỗi: Không tìm thấy TAVILY_API trong file .env"
                
            payload = {
                "api_key": api_key,
                "urls": [url]
            }
            
            r = requests.post("https://api.tavily.com/extract", json=payload, timeout=15)
            r.raise_for_status()
            
            data = r.json()
            results = data.get("results", [])
            failed = data.get("failed_results", [])
            
            if failed and any(f.get("url") == url for f in failed):
                return f"Lỗi từ Tavily Extract API: Không thể trích xuất {url}"
                
            if not results:
                return f"Không có dữ liệu trả về cho {url}"
                
            content = results[0].get("raw_content") or ""
            
            max_len = 6000
            if len(content) > max_len:
                content = content[:max_len] + "\n...[Nội dung bị cắt bớt vì quá dài]..."
                
            from rag.retriever import RetrievedDocument
            doc = RetrievedDocument(
                text=content,
                metadata={"url": url},
                score=1.0,
                collection="web_search",
                record=None
            )
            self._agent_sources.append(doc)
            return content
        except Exception as e:
            return f"Lỗi khi đọc nội dung URL: {e}"

    def _execute_tool(self, tool_name: str, args_str: str) -> str:
        """Parse arguments and route execution to the corresponding tool."""
        try:
            import json
            args = json.loads(args_str)
        except Exception as e:
            try:
                cleaned = args_str.replace("'", '"')
                args = json.loads(cleaned)
            except Exception:
                return f"Lỗi: Không thể phân tích tham số hành động dưới dạng JSON: {args_str}. Lỗi: {e}"

        return self._execute_tool_args(tool_name, args)

    def _call_tool_safely(self, tool_name: str, func: Any, args: dict[str, Any]) -> str:
        """Call a Python tool while ignoring unexpected model-supplied kwargs."""
        try:
            import inspect

            signature = inspect.signature(func)
            parameters = signature.parameters
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in parameters.values()
            )
            if accepts_kwargs:
                filtered_args = args
                ignored_args = {}
            else:
                allowed = {
                    name
                    for name, param in parameters.items()
                    if name != "self"
                    and param.kind in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                }
                filtered_args = {key: value for key, value in args.items() if key in allowed}
                ignored_args = {key: value for key, value in args.items() if key not in allowed}

            if ignored_args:
                log.info("Ignoring unsupported args for tool %s: %s", tool_name, sorted(ignored_args))
            return func(**filtered_args)
        except TypeError as exc:
            return f"Lỗi khi gọi công cụ {tool_name}: tham số không hợp lệ ({exc})"
        except Exception as exc:
            return f"Lỗi khi gọi công cụ {tool_name}: {exc}"

    def _execute_tool_args(self, tool_name: str, args: dict[str, Any]) -> str:
        """Route a structured native function call to the corresponding local tool."""
        if tool_name == "semantic_search":
            return self._tool_semantic_search(
                query_text=args.get("query_text", ""),
                collections=args.get("collections"),
                limit=args.get("limit", 5)
            )
        elif tool_name == "hybrid_search":
            return self._tool_hybrid_search(
                query_text=args.get("query_text", ""),
                collections=args.get("collections"),
                limit=args.get("limit", 5)
            )
        elif tool_name == "keyword_search":
            return self._tool_keyword_search(
                query_text=args.get("query_text", ""),
                collections=args.get("collections"),
                limit=args.get("limit", 5)
            )
        elif tool_name == "filter_listings":
            return self._call_tool_safely(tool_name, self._tool_filter_listings, args)
        elif tool_name == "find_nearby_pois":
            return self._call_tool_safely(tool_name, self._tool_find_nearby_pois, args)
        elif tool_name == "find_pois_near_location":
            return self._call_tool_safely(tool_name, self._tool_find_pois_near_location, args)
        elif tool_name == "search_pois":
            if isinstance(args.get("category"), list):
                observations = []
                for category in args["category"]:
                    per_args = {**args, "category": category}
                    observations.append(self._call_tool_safely(tool_name, self._tool_search_pois, per_args))
                return "\n\n".join(observations)
            return self._call_tool_safely(tool_name, self._tool_search_pois, args)
        elif tool_name == "find_listings_near_pois":
            if isinstance(args.get("category"), list):
                observations = []
                for category in args["category"]:
                    per_args = {**args, "category": category}
                    observations.append(self._call_tool_safely(tool_name, self._tool_find_listings_near_pois, per_args))
                return "\n\n".join(observations)
            return self._call_tool_safely(tool_name, self._tool_find_listings_near_pois, args)
        elif tool_name == "search_metro_stations":
            return self._call_tool_safely(tool_name, self._tool_search_metro_stations, args)
        elif tool_name == "find_listings_near_metro":
            return self._call_tool_safely(tool_name, self._tool_find_listings_near_metro, args)
        elif tool_name == "analyze_market_trend":
            return self._call_tool_safely(tool_name, self._tool_analyze_market_trend, args)
        elif tool_name == "get_market_statistics":
            return self._call_tool_safely(tool_name, self._tool_get_market_statistics, args)
        elif tool_name == "web_search":
            return self._call_tool_safely(tool_name, self._tool_web_search, args)
        elif tool_name == "web_research":
            return self._call_tool_safely(tool_name, self._tool_web_research, args)
        elif tool_name == "read_url":
            return self._call_tool_safely(tool_name, self._tool_read_url, args)
        elif tool_name == "search_location":
            return self._call_tool_safely(tool_name, self._tool_search_location, args)
        else:
            return f"Lỗi: Không tìm thấy công cụ tên là '{tool_name}'."

    def _native_tool_declarations(self) -> list[Any]:
        """Build Gemini native function declarations for the local RAG tools."""
        from google.genai import types

        def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
            schema: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return schema

        string_array = {"type": "array", "items": {"type": "string"}}
        collection_array = {
            **string_array,
            "description": "Collections: listings, projects, articles, social_neighborhood.",
        }

        declarations = [
            types.FunctionDeclaration(
                name="hybrid_search",
                description="Tìm kiếm hybrid Qdrant dense + sparse trên nhiều nguồn nội bộ.",
                parameters_json_schema=obj({
                    "query_text": {"type": "string"},
                    "collections": collection_array,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                }, ["query_text"]),
            ),
            types.FunctionDeclaration(
                name="semantic_search",
                description="Tìm kiếm ngữ nghĩa thuần trên các chunk đã embed.",
                parameters_json_schema=obj({
                    "query_text": {"type": "string"},
                    "collections": collection_array,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                }, ["query_text"]),
            ),
            types.FunctionDeclaration(
                name="keyword_search",
                description="Tìm kiếm từ khóa chính xác trong Postgres.",
                parameters_json_schema=obj({
                    "query_text": {"type": "string"},
                    "collections": collection_array,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                }, ["query_text"]),
            ),
            types.FunctionDeclaration(
                name="filter_listings",
                description="Lọc tin đăng theo giá, phòng ngủ, khu vực, loại nhà đất, hoặc bán kính tọa độ.",
                parameters_json_schema=obj({
                    "price_max_trieu": {"type": "number"},
                    "price_min_trieu": {"type": "number"},
                    "bedrooms": {"type": "integer"},
                    "loai_hinh": {"type": "string", "enum": ["ban", "cho_thue"]},
                    "tinh_thanh": {"type": "string"},
                    "quan_huyen": {"type": "string"},
                    "property_type": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_km": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 40},
                }),
            ),
            types.FunctionDeclaration(
                name="search_location",
                description="Tìm tọa độ cho một địa danh cụ thể khi người dùng nêu tên địa điểm.",
                parameters_json_schema=obj({"location_name": {"type": "string"}}, ["location_name"]),
            ),
            types.FunctionDeclaration(
                name="search_pois",
                description="Tìm POI nội bộ theo tên hoặc loại tiện ích.",
                parameters_json_schema=obj({
                    "poi_query": {"type": "string"},
                    "category": {"type": "string", "enum": ["transit_station", "school", "hospital", "park", "shopping", "airport", "landmark"]},
                    "city": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                }),
            ),
            types.FunctionDeclaration(
                name="find_listings_near_pois",
                description="Tìm tin đăng gần một nhóm POI như metro, trường học, bệnh viện, công viên.",
                parameters_json_schema=obj({
                    "poi_query": {"type": "string"},
                    "category": {"type": "string", "enum": ["transit_station", "school", "hospital", "park", "shopping", "airport", "landmark"]},
                    "city": {"type": "string"},
                    "radius_km": {"type": "number"},
                    "loai_hinh": {"type": "string", "enum": ["ban", "cho_thue"]},
                    "property_type": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 40},
                    "poi_limit": {"type": "integer", "minimum": 1, "maximum": 30},
                }),
            ),
            types.FunctionDeclaration(
                name="find_nearby_pois",
                description="Tìm tiện ích gần các listing/project đã có source_record_id.",
                parameters_json_schema=obj({
                    "entity_ids": string_array,
                    "entity_type": {"type": "string", "enum": ["listing", "project"]},
                    "categories": string_array,
                    "radius_m": {"type": "number"},
                    "top_n_per_category": {"type": "integer", "minimum": 1, "maximum": 10},
                }, ["entity_ids", "entity_type"]),
            ),
            types.FunctionDeclaration(
                name="find_pois_near_location",
                description="Tìm POI trong bán kính quanh một địa điểm/dự án/tọa độ cụ thể, ví dụ quanh Feliz En Vista trong 2km.",
                parameters_json_schema=obj({
                    "location_name": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "categories": string_array,
                    "radius_m": {"type": "number"},
                    "top_n_per_category": {"type": "integer", "minimum": 1, "maximum": 20},
                }),
            ),
            types.FunctionDeclaration(
                name="web_search",
                description="Tìm thông tin web bổ trợ, không dùng để tìm tin đăng BĐS.",
                parameters_json_schema=obj({
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                }, ["query"]),
            ),
            types.FunctionDeclaration(
                name="web_research",
                description="Tìm kiếm web và trích xuất nội dung top URL để kiểm chứng thông tin ngoài.",
                parameters_json_schema=obj({
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    "extract_top": {"type": "integer", "minimum": 1, "maximum": 5},
                }, ["query"]),
            ),
            types.FunctionDeclaration(
                name="read_url",
                description="Đọc nội dung chi tiết của một URL.",
                parameters_json_schema=obj({"url": {"type": "string"}}, ["url"]),
            ),
            types.FunctionDeclaration(
                name="analyze_market_trend",
                description="Phân tích xu hướng giá và đối chiếu giá trị BĐS theo khu vực.",
                parameters_json_schema=obj({
                    "tinh_thanh": {"type": "string"},
                    "quan_huyen": {"type": "string"},
                    "property_type": {"type": "string"},
                    "listing_type": {"type": "string", "enum": ["ban", "cho_thue"]},
                    "target_price_vnd": {"type": "number"},
                    "target_area_m2": {"type": "number"},
                }, ["tinh_thanh", "quan_huyen"]),
            ),
            types.FunctionDeclaration(
                name="get_market_statistics",
                description="Lấy thống kê giá/diện tích/số lượng tin theo khu vực.",
                parameters_json_schema=obj({
                    "tinh_thanh": {"type": "string"},
                    "quan_huyen": {"type": "string"},
                    "property_type": {"type": "string"},
                    "listing_type": {"type": "string", "enum": ["ban", "cho_thue"]},
                }),
            ),
        ]
        return [types.Tool(function_declarations=declarations)]

    def _extract_native_function_calls(self, response: Any) -> list[Any]:
        calls = list(getattr(response, "function_calls", None) or [])
        if calls:
            return calls
        try:
            parts = response.candidates[0].content.parts or []
            return [part.function_call for part in parts if getattr(part, "function_call", None)]
        except Exception:
            return []

    def _native_call_args(self, function_call: Any) -> dict[str, Any]:
        raw_args = getattr(function_call, "args", None) or {}
        try:
            return json.loads(json.dumps(raw_args, ensure_ascii=False, default=str))
        except Exception:
            try:
                return dict(raw_args)
            except Exception:
                return {}

    def _native_response_text(self, response: Any) -> str:
        return LLMClient._extract_non_thought_text(response).strip() if response else ""

    def _native_agent_system_prompt(self) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "Bạn đang chạy ở chế độ native function calling. Không viết `Thought:` hoặc `Action:`. "
            "Khi cần dữ liệu, hãy gọi function phù hợp bằng structured function call. "
            "Không tự bịa Observation; hệ thống sẽ trả kết quả function. "
            "Ưu tiên dữ liệu nội bộ cho tin đăng/dự án/review. Chỉ dùng web_search/web_research "
            "để kiểm chứng thông tin ngoài như ngập nước/quy hoạch/hạ tầng. "
            "Khi đã đủ dữ liệu, trả lời cuối cùng bằng tiếng Việt Markdown, có nguồn URL. "
            "Nếu thiếu thông tin nền tảng khiến truy vấn quá rộng, hãy hỏi 1-3 câu làm rõ."
        )

    def _use_native_function_calling(self) -> bool:
        return _env_flag("USE_GEMINI_FUNCTION_CALLING", True) and self._llm.supports_native_tools

    def _serialize_sources(self, docs: list[RetrievedDocument]) -> list[dict[str, Any]]:
        serialized_sources = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            serialized_sources.append({
                "collection": getattr(doc, "collection", ""),
                "score": getattr(doc, "score", 0),
                "text": (getattr(doc, "text", "") or "")[:1200],
                "metadata": meta,
                "url": meta.get("url", ""),
            })
        return serialized_sources

    def _unique_agent_sources(self) -> list[RetrievedDocument]:
        seen_urls = set()
        unique_sources = []
        for doc in self._agent_sources:
            url = getattr(doc, "metadata", {}).get("url", "")
            if url:
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(doc)
            else:
                unique_sources.append(doc)
        return unique_sources

    def _query_stream_native_gemini(
        self,
        user_query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Run the Gemini/Vertex agent using native function calls instead of text Action parsing."""
        from google.genai import types

        self._agent_sources = []
        effective_query = self._prepare_query_for_session(user_query, session_id)
        parsed = self._parser.parse(effective_query)
        self._current_parsed = parsed
        trace_id = uuid.uuid4().hex[:12]
        stream_react_trace = _env_flag("STREAM_REACT_TRACE", True)
        max_iterations = max(1, _env_int("REACT_MAX_ITERATIONS", 20))
        tool_history: list[dict[str, Any]] = []

        self._trace(
            "query_start",
            trace_id=trace_id,
            mode="native_stream",
            session_id=session_id,
            user_query=user_query,
            effective_query=effective_query,
            used_conversation_memory=effective_query != user_query,
            parsed_intent=parsed.intent,
            parsed_filters=parsed.filters,
            lifestyle_signals=parsed.lifestyle_signals,
        )

        prompt = (
            f"Câu hỏi của người dùng: {effective_query}\n\n"
            f"Intent đã phân tích: {parsed.intent}\n"
            f"Bộ lọc đã phân tích: {json.dumps(parsed.filters, ensure_ascii=False)}\n"
            f"Tín hiệu lifestyle: {', '.join(parsed.lifestyle_signals or []) or 'không có'}\n\n"
            "Hãy dùng function calls để lấy dữ liệu cần thiết trước khi trả lời. "
            "Nếu người dùng hỏi POI quanh một dự án/địa điểm cụ thể theo bán kính, dùng find_pois_near_location. "
            "Nếu cần kiểm tra tiêu chí gần POI khi tìm tin đăng, dùng find_listings_near_pois hoặc find_nearby_pois. "
            "Nếu cần tìm mô tả như ít ngập/yên tĩnh/gần trường học trong dữ liệu, dùng hybrid_search."
        )
        contents: list[Any] = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        tools = self._native_tool_declarations()
        final_answer = ""

        for iteration in range(max_iterations):
            response = self._llm.generate_with_tools(
                contents=contents,
                tools=tools,
                system_prompt=self._native_agent_system_prompt(),
                max_tokens=4096,
                temperature=_chat_tool_temperature(),
                tool_mode="AUTO",
            )
            if response is None:
                self._trace(
                    "native_llm_empty_response",
                    trace_id=trace_id,
                    mode="native_stream",
                    iteration=iteration + 1,
                )
                break

            function_calls = self._extract_native_function_calls(response)
            response_text = self._native_response_text(response)
            call_summaries = [
                {"name": getattr(call, "name", ""), "args": self._native_call_args(call)}
                for call in function_calls
            ]
            self._trace(
                "native_llm_response",
                trace_id=trace_id,
                mode="native_stream",
                iteration=iteration + 1,
                response=response_text,
                function_calls=call_summaries,
            )

            if function_calls:
                try:
                    contents.append(response.candidates[0].content)
                except Exception:
                    model_parts = [
                        types.Part.from_function_call(
                            name=str(getattr(call, "name", "")),
                            args=self._native_call_args(call),
                        )
                        for call in function_calls
                    ]
                    contents.append(types.Content(role="model", parts=model_parts))

                response_parts = []
                for function_call in function_calls:
                    tool_name = str(getattr(function_call, "name", ""))
                    tool_args = self._native_call_args(function_call)
                    args_json = json.dumps(tool_args, ensure_ascii=False)
                    self._trace(
                        "native_tool_call",
                        trace_id=trace_id,
                        mode="native_stream",
                        iteration=iteration + 1,
                        tool=tool_name,
                        args=tool_args,
                    )
                    yield {"type": "status", "text": f"🔧 Đang tra cứu: {tool_name}..."}
                    if stream_react_trace:
                        yield {"type": "tool_call", "text": f"{tool_name}({args_json})"}

                    observation = self._execute_tool_args(tool_name, tool_args)
                    tool_history.append({"name": tool_name, "args": tool_args, "observation": observation[:1200]})
                    self._trace(
                        "native_tool_observation",
                        trace_id=trace_id,
                        mode="native_stream",
                        iteration=iteration + 1,
                        tool=tool_name,
                        args=tool_args,
                        observation=observation,
                        source_count=len(self._agent_sources),
                    )
                    if stream_react_trace:
                        yield {"type": "observation", "text": observation}

                    response_parts.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": observation},
                        )
                    )

                contents.append(types.Content(role="user", parts=response_parts))
                continue

            if response_text and not _is_llm_failure_text(response_text):
                accepted, gate_feedback = self._final_answer_gate(parsed, tool_history, response_text)
                if accepted or iteration >= max_iterations - 1:
                    final_answer = response_text
                    self._trace(
                        "native_final_answer_accepted",
                        trace_id=trace_id,
                        mode="native_stream",
                        iteration=iteration + 1,
                        accepted=accepted,
                        forced_by_max_iteration=not accepted,
                        answer=final_answer,
                        source_count=len(self._agent_sources),
                    )
                    break

                self._trace(
                    "native_final_answer_rejected",
                    trace_id=trace_id,
                    mode="native_stream",
                    iteration=iteration + 1,
                    candidate_answer=response_text,
                    gate_feedback=gate_feedback,
                )
                yield {"type": "status", "text": "🔎 Cần kiểm tra thêm tiêu chí còn thiếu..."}
                try:
                    contents.append(response.candidates[0].content)
                except Exception:
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=response_text)]))
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=gate_feedback)]))
                continue

            break

        unique_sources = self._unique_agent_sources()
        yield {
            "type": "metadata",
            "intent": parsed.intent,
            "filters": parsed.filters,
            "effective_query": effective_query,
            "used_conversation_memory": effective_query != user_query,
            "sources": self._serialize_sources(unique_sources),
        }

        if not final_answer:
            final_answer = self._synthesize_final_answer(
                user_query=effective_query,
                parsed=parsed,
                draft="",
            )
        elif self._agent_sources and not self._should_ask_clarification_only(parsed, final_answer):
            final_answer = self._synthesize_final_answer(
                user_query=effective_query,
                parsed=parsed,
                draft=final_answer,
            )

        if self._should_ask_clarification_only(parsed, final_answer):
            final_answer = self._format_clarification_only_answer(parsed)
            self._trace(
                "native_clarification_only",
                trace_id=trace_id,
                mode="native_stream",
                session_id=session_id,
                answer=final_answer,
                source_count=len(self._agent_sources),
            )

        self._remember_or_clear_conversation(session_id, effective_query, parsed, final_answer)
        self._trace(
            "native_query_done",
            trace_id=trace_id,
            mode="native_stream",
            session_id=session_id,
            answer=final_answer,
            source_count=len(self._agent_sources),
        )
        yield {"type": "status", "text": "✍️ Đang tạo câu trả lời..."}
        yield {"type": "chunk", "text": final_answer}
        yield {"type": "done"}

    def _final_answer_gate(
        self,
        parsed: ParsedQuery,
        tool_history: list[dict[str, Any]],
        candidate_answer: str = "",
    ) -> tuple[bool, str]:
        """Decide whether a proposed Final Answer has enough tool coverage."""
        if not parsed:
            return True, ""

        query_norm = _normalize_text(parsed.raw_query or parsed.query_text)
        if self._is_clarification_answer_allowed(parsed, candidate_answer):
            return True, ""

        signals = set(parsed.lifestyle_signals or [])
        tool_names = [item.get("name", "") for item in tool_history]
        args_text = " ".join(json.dumps(item.get("args", {}), ensure_ascii=False) for item in tool_history)
        args_norm = _normalize_text(args_text)
        source_text = _normalize_text(" ".join((doc.text or "")[:1200] for doc in self._agent_sources))

        missing: list[str] = []
        suggestions: list[str] = []

        has_hybrid = any(name in {"hybrid_search", "semantic_search", "keyword_search"} for name in tool_names)
        has_listing_filter = any(name in {"filter_listings", "find_listings_near_pois", "find_listings_near_metro"} for name in tool_names)
        has_poi_listing = any(name in {"find_listings_near_pois", "find_listings_near_metro"} for name in tool_names)
        has_nearby_pois = any(name in {"find_nearby_pois", "find_pois_near_location"} for name in tool_names)
        has_web_depth = any(name in {"web_research", "read_url"} for name in tool_names)
        has_web = any(name in {"web_search", "web_research", "read_url"} for name in tool_names)
        has_market_stats = any(name in {"get_market_statistics", "analyze_market_trend"} for name in tool_names)

        if parsed.intent == "market_report":
            if has_market_stats and self._agent_sources:
                return True, ""
            missing.append("chưa lấy số liệu thống kê thị trường")
            suggestions.append("gọi `get_market_statistics` với khu vực và loại hình phù hợp")
            detail = "; ".join(missing)
            next_steps = "; ".join(dict.fromkeys(suggestions))
            return False, f"{detail}. Bước tiếp theo: {next_steps}."

        asks_only_pois_around_place = any(
            token in query_norm
            for token in ("xung quanh", "quanh", "ban kinh", "gan day")
        ) and any(
            token in query_norm
            for token in ("truong", "benh vien", "cong vien", "tien ich", "poi")
        )

        if (
            parsed.intent in {"search_listing", "lifestyle_search"}
            and not asks_only_pois_around_place
            and not (has_hybrid or has_listing_filter)
        ):
            missing.append("chưa tìm tin đăng/dự án từ dữ liệu nội bộ")
            suggestions.append("gọi `hybrid_search` hoặc `filter_listings` trước")

        location_words = {
            "gan", "metro", "truong", "benh vien", "cong vien", "sieu thi",
            "tttm", "san bay", "landmark", "dia danh",
        }
        asks_near_location = "gan" in query_norm and any(word in query_norm for word in location_words)
        if asks_near_location and not (has_poi_listing or has_nearby_pois or "search_location" in tool_names):
            missing.append("chưa xử lý tiêu chí gần địa điểm/POI")
            suggestions.append("gọi `find_pois_near_location` nếu hỏi POI quanh địa điểm/dự án, hoặc `find_listings_near_pois` nếu đang tìm tin đăng")

        wants_metro = "metro" in signals or "metro" in query_norm or "tau dien" in query_norm
        if wants_metro and not (
            has_poi_listing
            or "metro" in source_text
            or "transit station" in args_norm
            or "transit_station" in args_text
        ):
            missing.append("chưa có bằng chứng gần metro/giao thông công cộng")
            suggestions.append("gọi `find_listings_near_pois` với category=`transit_station`, poi_query=`metro`")

        wants_school = "school" in signals or "truong" in query_norm
        if wants_school and not (
            ("school" in args_text and (has_nearby_pois or has_poi_listing))
            or "truong" in source_text
        ):
            missing.append("chưa kiểm tra trường học gần ứng viên")
            suggestions.append("gọi `find_pois_near_location` cho địa điểm/dự án, hoặc `find_nearby_pois` với categories=[\"school\"] cho các source_record_id tốt")

        wants_flood = "flood" in signals or "ngap" in query_norm
        if wants_flood and not (
            "ngap" in source_text
            or (has_hybrid and "ngap" in args_norm)
            or has_web_depth
        ):
            missing.append("chưa kiểm chứng tiêu chí ít ngập")
            suggestions.append("gọi `hybrid_search` với từ khóa ngập hoặc `web_research` để kiểm chứng ngập khu vực")
        elif wants_flood and has_web and not has_web_depth and "ngap" not in source_text:
            missing.append("web_search về ngập chỉ có snippet, chưa đủ sâu")
            suggestions.append("gọi `web_research` hoặc `read_url` cho nguồn web phù hợp")

        wants_safety = "safety" in signals or "yen tinh" in query_norm or "an ninh" in query_norm
        if wants_safety and not (has_hybrid or "yen tinh" in source_text or "an ninh" in source_text):
            missing.append("chưa tìm bằng chứng yên tĩnh/an ninh trong mô tả hoặc review")
            suggestions.append("gọi `hybrid_search` với đầy đủ tiêu chí lifestyle")

        if not missing:
            return True, ""

        detail = "; ".join(missing)
        next_steps = "; ".join(dict.fromkeys(suggestions))
        return (
            False,
            "Bộ điều phối chưa chấp nhận Final Answer vì "
            f"{detail}. Bước tiếp theo nên làm: {next_steps}. "
            "Nếu đã thử mà không có dữ liệu, hãy ghi rõ tiêu chí đó là thiếu/độ tin cậy thấp trong Final Answer.",
        )

    def _is_clarification_answer_allowed(
        self,
        parsed: ParsedQuery,
        candidate_answer: str,
    ) -> bool:
        """Allow the agent to stop and ask the user for genuinely missing info."""
        if not candidate_answer:
            return False

        answer_norm = _normalize_text(candidate_answer)
        query_norm = _normalize_text(parsed.raw_query or parsed.query_text)
        has_question_shape = "?" in candidate_answer or any(
            phrase in answer_norm
            for phrase in (
                "ban muon",
                "ban can",
                "vui long cho biet",
                "cho minh biet",
                "can them thong tin",
                "ban dang tim",
                "mua hay thue",
                "o thanh pho nao",
                "khu vuc nao",
                "ngan sach",
            )
        )
        if not has_question_shape:
            return False

        f = parsed.filters or {}
        missing_reasons = self._missing_clarification_reasons(parsed, query_norm, f)
        if not missing_reasons:
            return False

        asks_for_missing = any(reason in answer_norm for reason in missing_reasons)
        broad_clarification = any(
            phrase in answer_norm
            for phrase in (
                "them thong tin",
                "thong tin bo sung",
                "lam ro",
                "cu the hon",
                "nhu cau",
            )
        )
        return asks_for_missing or broad_clarification

    def _missing_clarification_reasons(
        self,
        parsed: ParsedQuery,
        query_norm: str,
        filters: dict,
    ) -> list[str]:
        """Return normalized reason tokens for missing required user constraints."""
        reasons: list[str] = []

        search_intent = parsed.intent in {"search_listing", "lifestyle_search"} or any(
            intent in {"search_listing", "lifestyle_search"} for intent in parsed.intents
        )
        if not search_intent:
            return reasons

        has_city = bool(filters.get("tinh_thanh")) or any(
            token in query_norm
            for token in (
                "ho chi minh", "hcm", "sai gon", "ha noi", "da nang", "binh duong",
                "dong nai", "thu duc", "quan 1", "quan 2", "quan 7",
            )
        )
        asks_location_sensitive = any(
            token in query_norm
            for token in (
                "gan", "metro", "truong", "benh vien", "cong vien", "san bay",
                "it ngap", "ngap", "yen tinh", "an ninh", "khu vuc",
            )
        )
        if asks_location_sensitive and not has_city:
            reasons.extend(["thanh pho", "o dau", "khu vuc", "tinh thanh"])

        has_listing_type = bool(filters.get("loai_hinh")) or any(
            token in query_norm
            for token in ("mua", "ban", "thue", "cho thue")
        )
        if search_intent and not has_listing_type:
            reasons.extend(["mua", "thue", "mua hay thue"])

        has_budget = bool(filters.get("gia_trieu")) or any(
            token in query_norm
            for token in ("ty", "trieu", "ngan sach", "gia", "duoi", "tam")
        )
        broad_search = any(
            token in query_norm
            for token in ("tim", "goi y", "chung cu", "can ho", "nha", "bds")
        )
        if broad_search and not has_budget:
            reasons.extend(["ngan sach", "gia", "tam gia"])

        return reasons

    def _clarification_questions(self, parsed: ParsedQuery) -> list[str]:
        """Build concise user questions for missing foundational constraints."""
        query_norm = _normalize_text(parsed.raw_query or parsed.query_text)
        reasons = set(self._missing_clarification_reasons(parsed, query_norm, parsed.filters or {}))
        questions: list[str] = []

        if reasons.intersection({"mua", "thue", "mua hay thue"}):
            questions.append("Bạn muốn **mua** hay **thuê**?")
        if reasons.intersection({"thanh pho", "o dau", "khu vuc", "tinh thanh"}):
            questions.append("Bạn muốn tìm ở **thành phố/khu vực nào**?")
        if reasons.intersection({"ngan sach", "gia", "tam gia"}):
            questions.append("Ngân sách hoặc tầm giá mong muốn là bao nhiêu?")

        return questions[:3]

    def _should_ask_clarification_only(self, parsed: ParsedQuery, draft: str = "") -> bool:
        """Return True when the best response is clarification, not recommendations."""
        questions = self._clarification_questions(parsed)
        if not questions:
            return False

        query_norm = _normalize_text(parsed.raw_query or parsed.query_text)
        broad_location_sensitive = any(
            token in query_norm
            for token in ("gan", "metro", "truong", "benh vien", "cong vien", "ngap", "yen tinh", "an ninh")
        )
        sparse_constraints = not parsed.filters.get("tinh_thanh") and not parsed.filters.get("loai_hinh")
        answer_is_clarification = self._is_clarification_answer_allowed(parsed, draft)
        has_sourced_evidence = bool(self._agent_sources) or bool(re.search(r"https?://|hf://", draft or ""))
        return bool(
            answer_is_clarification
            or (broad_location_sensitive and sparse_constraints and not has_sourced_evidence)
        )

    def _format_clarification_only_answer(self, parsed: ParsedQuery) -> str:
        questions = self._clarification_questions(parsed)
        if not questions:
            questions = ["Bạn có thể cho biết thêm khu vực, ngân sách và nhu cầu mua/thuê không?"]
        lines = [
            "Mình cần thêm một chút thông tin để tìm đúng hơn, vì yêu cầu hiện tại còn khá rộng:"
        ]
        lines.extend(f"{idx}. {question}" for idx, question in enumerate(questions, 1))
        lines.append("\nSau khi có các thông tin này, mình sẽ lọc lại theo các tiêu chí yên tĩnh, gần tiện ích/metro/trường học và ít ngập nước.")
        return "\n".join(lines)

    def _synthesize_final_answer(
        self,
        user_query: str,
        parsed: ParsedQuery,
        draft: str = "",
    ) -> str:
        """Create a final user-facing answer from accumulated tool sources."""
        if self._should_ask_clarification_only(parsed, draft):
            return self._format_clarification_only_answer(parsed)

        if not self._llm.is_available:
            return draft.strip() or "Tôi chưa có đủ thông tin để trả lời chính xác."

        context = format_context(self._agent_sources, max_chars=12000)
        prompt = (
            f"Câu hỏi của người dùng: {user_query}\n\n"
            f"Intent đã phân tích: {parsed.intent}\n"
            f"Bộ lọc đã áp dụng: {parsed.filters}\n\n"
            f"Thông tin đã tra cứu từ công cụ:\n{context}\n\n"
        )
        if draft.strip():
            prompt += f"Ghi chú/draft từ vòng ReAct:\n{draft.strip()}\n\n"
        prompt += (
            "Hãy viết câu trả lời cuối cùng bằng tiếng Việt, định dạng Markdown. "
            "Không nhắc lại Thought/Action/Observation. "
            "Chỉ dựa trên dữ liệu đã tra cứu; nếu thiếu dữ liệu cho tiêu chí nào "
            "(ví dụ ít ngập/yên tĩnh), ghi rõ mức độ chắc chắn. "
            "Ưu tiên gợi ý 2-3 lựa chọn phù hợp nhất, nêu rõ: giá, diện tích, vị trí, "
            "gần metro/trường học nếu có dữ liệu, nhận xét về yên tĩnh/ngập nước nếu có bằng chứng, "
            "và kèm URL nguồn."
        )
        answer = self._llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=4096,
            temperature=_chat_answer_temperature(),
        )
        return answer.strip() if answer else (draft.strip() or "Tôi chưa có đủ thông tin để trả lời chính xác.")

    # ---------------------------------------------------------------------------
    # ReAct Agent loop implementation
    # ---------------------------------------------------------------------------
    def query(self, user_query: str, top_k: int = 5, session_id: Optional[str] = None) -> RAGResponse:
        """Process a user query through the agentic RAG loop."""
        log.info(f"Processing query: {user_query[:80]}...")

        if self._use_native_function_calling():
            answer_parts: list[str] = []
            metadata: dict[str, Any] = {}
            for event in self._query_stream_native_gemini(user_query, top_k=top_k, session_id=session_id):
                if event.get("type") == "chunk":
                    answer_parts.append(event.get("text", ""))
                elif event.get("type") == "metadata":
                    metadata = event
            parsed = self._current_parsed or self._parser.parse(metadata.get("effective_query") or user_query)
            return RAGResponse(
                answer="".join(answer_parts),
                sources=self._unique_agent_sources(),
                intent=parsed.intent,
                filters_applied=parsed.filters,
                parsed_query=parsed,
                llm_used=True,
                effective_query=metadata.get("effective_query") or user_query,
            )
        
        # Reset retrieved sources cache
        self._agent_sources = []
        effective_query = self._prepare_query_for_session(user_query, session_id)
        trace_id = uuid.uuid4().hex[:12]
        
        # Step 1: Parse query for initial fallback/metadata tracking
        parsed = self._parser.parse(effective_query)
        self._current_parsed = parsed
        self._trace(
            "query_start",
            trace_id=trace_id,
            mode="query",
            session_id=session_id,
            user_query=user_query,
            effective_query=effective_query,
            used_conversation_memory=effective_query != user_query,
            parsed_intent=parsed.intent,
            parsed_filters=parsed.filters,
            lifestyle_signals=parsed.lifestyle_signals,
        )

        # Step 2: System prompt setup
        from rag.prompts import REACT_SYSTEM_PROMPT, SYSTEM_PROMPT
        agent_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\n"
            f"User Query: {effective_query}\n\n"
            "Hãy bắt đầu với Thought: đầu tiên của bạn."
        )

        MAX_ITERATIONS = max(1, _env_int("REACT_MAX_ITERATIONS", 20))
        llm_used = False
        answer = ""
        tool_history: list[dict[str, Any]] = []

        if self._llm.is_available:
            llm_used = True
            for iteration in range(MAX_ITERATIONS):
                log.info(f"--- ReAct Iteration {iteration+1} ---")
                response = self._llm.generate(
                    prompt=agent_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=_chat_tool_temperature()
                )
                self._trace(
                    "llm_response",
                    trace_id=trace_id,
                    mode="query",
                    iteration=iteration + 1,
                    response=response or "",
                )
                if not response:
                    log.warning("Received empty response from LLM")
                    break

                log.info("LLM response received (chars=%s)", len(response))

                if "Final Answer:" in response:
                    parts = response.split("Final Answer:", 1)
                    candidate_answer = parts[1].strip()
                    accepted, gate_feedback = self._final_answer_gate(parsed, tool_history, candidate_answer)
                    if accepted or iteration >= MAX_ITERATIONS - 1:
                        answer = candidate_answer
                        self._trace(
                            "final_answer_accepted",
                            trace_id=trace_id,
                            mode="query",
                            iteration=iteration + 1,
                            accepted=accepted,
                            forced_by_max_iteration=not accepted,
                            answer=answer,
                        )
                        log.info("Found accepted Final Answer. Exiting loop.")
                        break
                    log.info("Final Answer rejected by coverage gate: %s", gate_feedback)
                    self._trace(
                        "final_answer_rejected",
                        trace_id=trace_id,
                        mode="query",
                        iteration=iteration + 1,
                        candidate_answer=candidate_answer,
                        gate_feedback=gate_feedback,
                    )
                    agent_prompt += f"\n{response}\nObservation: {gate_feedback}\n"
                    continue

                # Parse Action
                tool_name, tool_args_str = _parse_action_call(response)
                if tool_name:
                    if tool_args_str:
                        log.info(f"Executing tool {tool_name} with arguments: {tool_args_str}")
                        self._trace(
                            "tool_call",
                            trace_id=trace_id,
                            mode="query",
                            iteration=iteration + 1,
                            tool=tool_name,
                            args_raw=tool_args_str,
                        )
                        observation = self._execute_tool(tool_name, tool_args_str)
                        try:
                            tool_args = json.loads(tool_args_str)
                        except Exception:
                            tool_args = {}
                        tool_history.append({"name": tool_name, "args": tool_args, "observation": observation[:1200]})
                        log.info("Tool observation received from %s (chars=%s)", tool_name, len(observation))
                        self._trace(
                            "tool_observation",
                            trace_id=trace_id,
                            mode="query",
                            iteration=iteration + 1,
                            tool=tool_name,
                            args=tool_args,
                            observation=observation,
                            source_count=len(self._agent_sources),
                        )
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue
                    else:
                        observation = "Lỗi: Tham số không đúng định dạng JSON. Vui lòng định dạng dưới dạng: Action: tool_name({\"key\": \"value\"})"
                        log.warning("Malformed tool call JSON from LLM output (chars=%s)", len(response))
                        self._trace(
                            "tool_parse_error",
                            trace_id=trace_id,
                            mode="query",
                            iteration=iteration + 1,
                            tool=tool_name,
                            response=response,
                            observation=observation,
                        )
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue

                # Fallback if loop gets stuck
                log.warning("No clear Action or Final Answer. Fallback to response text.")
                draft = "" if _is_llm_failure_text(response) else response.replace("Thought:", "").strip()
                answer = self._synthesize_final_answer(
                    user_query=effective_query,
                    parsed=parsed,
                    draft=draft,
                )
                self._trace(
                    "fallback_synthesis",
                    trace_id=trace_id,
                    mode="query",
                    iteration=iteration + 1,
                    draft=draft,
                    answer=answer,
                    source_count=len(self._agent_sources),
                )
                break
            else:
                log.warning("Exceeded max iterations")
                answer = self._synthesize_final_answer(
                    user_query=effective_query,
                    parsed=parsed,
                    draft="Đã vượt quá giới hạn vòng ReAct trước khi có Final Answer.",
                )
                self._trace(
                    "max_iterations_synthesis",
                    trace_id=trace_id,
                    mode="query",
                    max_iterations=MAX_ITERATIONS,
                    answer=answer,
                    source_count=len(self._agent_sources),
                )
        else:
            # Fallback to local format if LLM is offline
            from rag.llm import LLMClient
            answer = LLMClient.format_without_llm(
                query=effective_query,
                documents=[],
                intent=parsed.intent
            )

        self._remember_or_clear_conversation(session_id, effective_query, parsed, answer)
        self._trace(
            "query_done",
            trace_id=trace_id,
            mode="query",
            session_id=session_id,
            answer=answer,
            source_count=len(self._agent_sources),
        )

        # De-duplicate collected sources
        seen_urls = set()
        unique_sources = []
        for doc in self._agent_sources:
            url = getattr(doc, "metadata", {}).get("url", "")
            if url:
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(doc)
            else:
                unique_sources.append(doc)

        return RAGResponse(
            answer=answer,
            sources=unique_sources,
            intent=parsed.intent,
            filters_applied=parsed.filters,
            parsed_query=parsed,
            llm_used=llm_used,
            effective_query=effective_query,
        )

    def query_stream(
        self,
        user_query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Process a user query: run ReAct loop silently, then stream the final answer."""
        log.info(f"Streaming query: {user_query[:80]}...")

        if self._use_native_function_calling():
            yield from self._query_stream_native_gemini(user_query, top_k=top_k, session_id=session_id)
            return

        self._agent_sources = []
        effective_query = self._prepare_query_for_session(user_query, session_id)
        parsed = self._parser.parse(effective_query)
        self._current_parsed = parsed
        trace_id = uuid.uuid4().hex[:12]
        stream_react_trace = _env_flag("STREAM_REACT_TRACE", True)
        self._trace(
            "query_start",
            trace_id=trace_id,
            mode="stream",
            session_id=session_id,
            user_query=user_query,
            effective_query=effective_query,
            used_conversation_memory=effective_query != user_query,
            parsed_intent=parsed.intent,
            parsed_filters=parsed.filters,
            lifestyle_signals=parsed.lifestyle_signals,
        )

        from rag.prompts import REACT_SYSTEM_PROMPT, SYSTEM_PROMPT
        agent_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\n"
            f"User Query: {effective_query}\n\n"
            "Hãy bắt đầu với Thought: đầu tiên của bạn."
        )

        MAX_ITERATIONS = max(1, _env_int("REACT_MAX_ITERATIONS", 20))

        if self._llm.is_available:
            # ── Phase 1: Run the full ReAct loop silently ──────────────────────
            # The user only sees brief tool-call status events during this phase.
            raw_final = ""
            tool_history: list[dict[str, Any]] = []

            for iteration in range(MAX_ITERATIONS):
                response = self._llm.generate(
                    prompt=agent_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=_chat_tool_temperature()
                )
                self._trace(
                    "llm_response",
                    trace_id=trace_id,
                    mode="stream",
                    iteration=iteration + 1,
                    response=response or "",
                )
                if not response:
                    break
                    
                log.info("LLM response received in stream loop (chars=%s)", len(response))
                if stream_react_trace:
                    yield {"type": "thought", "text": response}

                if "Final Answer:" in response:
                    candidate_final = response.split("Final Answer:", 1)[1].strip()
                    accepted, gate_feedback = self._final_answer_gate(parsed, tool_history, candidate_final)
                    if accepted or iteration >= MAX_ITERATIONS - 1:
                        raw_final = candidate_final
                        self._trace(
                            "final_answer_accepted",
                            trace_id=trace_id,
                            mode="stream",
                            iteration=iteration + 1,
                            accepted=accepted,
                            forced_by_max_iteration=not accepted,
                            answer=raw_final,
                        )
                        break
                    self._trace(
                        "final_answer_rejected",
                        trace_id=trace_id,
                        mode="stream",
                        iteration=iteration + 1,
                        candidate_answer=candidate_final,
                        gate_feedback=gate_feedback,
                    )
                    yield {"type": "status", "text": "🔎 Cần kiểm tra thêm tiêu chí còn thiếu..."}
                    agent_prompt += f"\n{response}\nObservation: {gate_feedback}\n"
                    continue

                # Execute tool call
                tool_name, tool_args_str = _parse_action_call(response)
                if tool_name:
                    if tool_args_str:
                        self._trace(
                            "tool_call",
                            trace_id=trace_id,
                            mode="stream",
                            iteration=iteration + 1,
                            tool=tool_name,
                            args_raw=tool_args_str,
                        )

                        # Brief status — user knows the agent is working
                        yield {"type": "status", "text": f"🔧 Đang tra cứu: {tool_name}..."}

                        observation = self._execute_tool(tool_name, tool_args_str)
                        try:
                            tool_args = json.loads(tool_args_str)
                        except Exception:
                            tool_args = {}
                        tool_history.append({"name": tool_name, "args": tool_args, "observation": observation[:1200]})
                        self._trace(
                            "tool_observation",
                            trace_id=trace_id,
                            mode="stream",
                            iteration=iteration + 1,
                            tool=tool_name,
                            args=tool_args,
                            observation=observation,
                            source_count=len(self._agent_sources),
                        )
                        if stream_react_trace:
                            yield {"type": "observation", "text": f"Observation: {observation}"}
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue
                    else:
                        observation = "Lỗi: Tham số không đúng định dạng JSON. Vui lòng định dạng dưới dạng: Action: tool_name({\"key\": \"value\"})"
                        log.warning("Malformed tool call JSON from streamed LLM output (chars=%s)", len(response))
                        self._trace(
                            "tool_parse_error",
                            trace_id=trace_id,
                            mode="stream",
                            iteration=iteration + 1,
                            tool=tool_name,
                            response=response,
                            observation=observation,
                        )
                        if stream_react_trace:
                            yield {"type": "observation", "text": f"Observation: {observation}"}
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue

                # No Action and no Final Answer — treat as the answer
                raw_final = "" if _is_llm_failure_text(response) else response.replace("Thought:", "").strip()
                self._trace(
                    "fallback_raw_final",
                    trace_id=trace_id,
                    mode="stream",
                    iteration=iteration + 1,
                    raw_final=raw_final,
                    source_count=len(self._agent_sources),
                )
                break

            # ── Phase 2: Emit sources metadata ─────────────────────────────────
            seen_urls = set()
            unique_sources = []
            for doc in self._agent_sources:
                url = getattr(doc, "metadata", {}).get("url", "")
                if url:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_sources.append(doc)
                else:
                    unique_sources.append(doc)

            serialized_sources = []
            for doc in unique_sources:
                meta = getattr(doc, "metadata", {}) or {}
                serialized_sources.append({
                    "collection": getattr(doc, "collection", ""),
                    "score": getattr(doc, "score", 0),
                    "text": (getattr(doc, "text", "") or "")[:1200],
                    "metadata": meta,
                    "url": meta.get("url", ""),
                })

            yield {
                "type": "metadata",
                "intent": parsed.intent,
                "filters": parsed.filters,
                "effective_query": effective_query,
                "used_conversation_memory": effective_query != user_query,
                "sources": serialized_sources,
            }

            if self._should_ask_clarification_only(parsed, raw_final):
                clarification_answer = self._format_clarification_only_answer(parsed)
                self._remember_or_clear_conversation(session_id, effective_query, parsed, clarification_answer)
                self._trace(
                    "clarification_only",
                    trace_id=trace_id,
                    mode="stream",
                    session_id=session_id,
                    raw_final=raw_final,
                    answer=clarification_answer,
                    source_count=len(self._agent_sources),
                )
                yield {"type": "status", "text": "✍️ Đang tạo câu hỏi làm rõ..."}
                yield {"type": "chunk", "text": clarification_answer}
                yield {"type": "done"}
                return

            self._remember_or_clear_conversation(session_id, effective_query, parsed, raw_final)

            # ── Phase 3: Stream the final answer ───────────────────────────────
            # Clean prompt — user query + retrieved source context + raw draft.
            retrieved_context = format_context(self._agent_sources, max_chars=12000)
            stream_prompt = (
                f"Câu hỏi của người dùng: {effective_query}\n\n"
                f"Thông tin đã tra cứu từ công cụ:\n{retrieved_context}\n\n"
                f"Ghi chú/draft từ vòng ReAct:\n{raw_final}\n\n"
                "Dựa trên thông tin trên, hãy viết câu trả lời hoàn chỉnh dưới "
                "dạng Markdown rõ ràng, có cấu trúc, dễ đọc. Không nhắc lại "
                "Thought/Action/Observation. Chỉ xuất nội dung câu trả lời."
            )

            yield {"type": "status", "text": "✍️ Đang tạo câu trả lời..."}
            streamed_answer_parts: list[str] = []
            for token in self._llm.generate_stream(
                prompt=stream_prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=4096,
                temperature=_chat_answer_temperature(),
            ):
                streamed_answer_parts.append(token)
                yield {"type": "chunk", "text": token}

            streamed_answer = "".join(streamed_answer_parts)
            self._trace(
                "final_stream_done",
                trace_id=trace_id,
                mode="stream",
                session_id=session_id,
                raw_final=raw_final,
                answer=streamed_answer,
                source_count=len(self._agent_sources),
            )
            yield {"type": "done"}

        else:
            # Fallback for offline LLM
            from rag.llm import LLMClient
            fallback_text = LLMClient.format_without_llm(
                query=effective_query,
                documents=[],
                intent=parsed.intent
            )
            self._trace(
                "offline_fallback",
                trace_id=trace_id,
                mode="stream",
                session_id=session_id,
                answer=fallback_text,
            )
            yield {
                "type": "metadata",
                "intent": parsed.intent,
                "filters": parsed.filters,
                "sources": [],
            }
            yield {"type": "chunk", "text": fallback_text}
            yield {"type": "done"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    chain = RAGChain()

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Tôi muốn tìm chung cư giá khoảng 3 tỷ ở Quận 2 có 2 phòng ngủ và muốn tính lãi suất vay mua nhà trong 15 năm"

    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print(f"{'='*60}\n")

    result = chain.query(q)

    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result.answer)
    print(f"\n{'='*60}")
    print(f"Intents detected: {result.parsed_query.intents}")
    print(f"Filters: {result.filters_applied}")
    print(f"Sources: {len(result.sources)} documents")
    print(f"LLM used: {result.llm_used}")
