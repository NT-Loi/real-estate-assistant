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
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

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


# ---------------------------------------------------------------------------
# Finance parameter extraction
# ---------------------------------------------------------------------------
import re

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
        self._parser = QueryParser()
        self._retriever = Retriever(self._store)
        self._llm = llm or LLMClient()
        self._agent_sources = []
        self._current_parsed: Optional[ParsedQuery] = None

        # Eagerly load the embedding model to avoid cold-start during first chat query
        if hasattr(self._store, "_embedder") and hasattr(self._store._embedder, "_load"):
            self._store._embedder._load()

        log.info("RAG Chain initialized in Agentic mode")
        log.info(f"  LLM available: {self._llm.is_available}")
        log.info(f"  Collections: {self._store.stats()}")

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

    def _tool_analyze_market_trend(self, **kwargs) -> str:
        """Analyze historical market trends and compare property price."""
        try:
            log.info(f"Tool called: analyze_market_trend({kwargs=})")
            tinh_thanh = kwargs.get("tinh_thanh")
            quan_huyen = kwargs.get("quan_huyen")
            property_type = kwargs.get("property_type")
            target_price_vnd = kwargs.get("target_price_vnd")
            target_area_m2 = kwargs.get("target_area_m2")
            
            if not tinh_thanh or not quan_huyen:
                return "Vui lòng cung cấp cả Tỉnh/Thành phố và Quận/Huyện để phân tích thị trường."
            
            # Fetch past 12 months data
            rows = self._store.pg.fetch_market_stats(
                province=tinh_thanh,
                district=quan_huyen,
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
            
            return "\n".join(output)
        except Exception as e:
            return f"Lỗi khi phân tích xu hướng giá: {e}"

    def _tool_get_market_statistics(self, **kwargs) -> str:
        """Get PostgreSQL-based aggregated statistics snapshots."""
        try:
            log.info(f"Tool called: get_market_statistics({kwargs=})")
            tinh_thanh = kwargs.get("tinh_thanh")
            quan_huyen = kwargs.get("quan_huyen")
            
            filters = {}
            if tinh_thanh:
                filters["tinh_thanh"] = tinh_thanh
            if quan_huyen:
                filters["quan_huyen"] = quan_huyen
                
            docs = self._retriever.retrieve_market_report(filters=filters, months=12)
            if not docs:
                return "Không tìm thấy dữ liệu báo cáo thị trường cho khu vực này."
                
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi truy vấn thống kê thị trường: {e}"



    def _tool_web_search(self, query: str, limit: int = 3) -> str:
        """Search the web using Tavily."""
        try:
            log.info(f"Tool called: web_search({query=}, {limit=})")
            import os
            import requests
            
            api_key = os.getenv("TAVILY_API") or os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Lỗi: Không tìm thấy TAVILY_API trong file .env"
                
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
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

    def _tool_search_location(self, location_name: str) -> str:
        """Search for a location using OpenStreetMap Nominatim API."""
        try:
            log.info(f"Tool called: search_location({location_name=})")
            import requests
            headers = {"User-Agent": "RealEstateAssistant/1.0"}
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location_name, "format": "json", "limit": 1},
                headers=headers,
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                return f"Không tìm thấy tọa độ cho địa danh: {location_name}"
            
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            display_name = data[0].get("display_name", "")
            return f'{{"lat": {lat}, "lon": {lon}, "display_name": "{display_name}"}}'
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
            return self._tool_filter_listings(**args)
        elif tool_name == "find_nearby_pois":
            return self._tool_find_nearby_pois(**args)
        elif tool_name == "analyze_market_trend":
            return self._tool_analyze_market_trend(**args)
        elif tool_name == "get_market_statistics":
            return self._tool_get_market_statistics(**args)
        elif tool_name == "web_search":
            return self._tool_web_search(**args)
        elif tool_name == "read_url":
            return self._tool_read_url(**args)
        elif tool_name == "search_location":
            return self._tool_search_location(**args)
        else:
            return f"Lỗi: Không tìm thấy công cụ tên là '{tool_name}'."

    # ---------------------------------------------------------------------------
    # ReAct Agent loop implementation
    # ---------------------------------------------------------------------------
    def query(self, user_query: str, top_k: int = 5) -> RAGResponse:
        """Process a user query through the agentic RAG loop."""
        log.info(f"Processing query: {user_query[:80]}...")
        
        # Reset retrieved sources cache
        self._agent_sources = []
        
        # Step 1: Parse query for initial fallback/metadata tracking
        parsed = self._parser.parse(user_query)
        self._current_parsed = parsed

        # Step 2: System prompt setup
        from rag.prompts import REACT_SYSTEM_PROMPT, SYSTEM_PROMPT
        agent_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\n"
            f"User Query: {user_query}\n\n"
            "Hãy bắt đầu với Thought: đầu tiên của bạn."
        )

        MAX_ITERATIONS = 5
        llm_used = False
        answer = ""

        if self._llm.is_available:
            llm_used = True
            for iteration in range(MAX_ITERATIONS):
                log.info(f"--- ReAct Iteration {iteration+1} ---")
                response = self._llm.generate(
                    prompt=agent_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.1
                )
                if not response:
                    log.warning("Received empty response from LLM")
                    break

                log.info(f"LLM Response:\n{response}")

                if "Final Answer:" in response:
                    parts = response.split("Final Answer:", 1)
                    answer = parts[1].strip()
                    log.info("Found Final Answer. Exiting loop.")
                    break

                # Parse Action
                action_match = re.search(r"Action:\s*`?(\w+)`?", response, re.IGNORECASE)
                if action_match:
                    tool_name = action_match.group(1).strip()
                    remaining = response[action_match.end():].strip()
                    brace_start = remaining.find("{")
                    brace_end = remaining.rfind("}")
                    
                    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                        tool_args_str = remaining[brace_start:brace_end+1]
                        log.info(f"Executing tool {tool_name} with arguments: {tool_args_str}")
                        observation = self._execute_tool(tool_name, tool_args_str)
                        log.info(f"Observation: {observation}")
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue
                    else:
                        observation = "Lỗi: Tham số không đúng định dạng JSON. Vui lòng định dạng dưới dạng: Action: tool_name({\"key\": \"value\"})"
                        log.warning(f"Malformed tool call JSON. LLM output: {response}")
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue

                # Fallback if loop gets stuck
                log.warning("No clear Action or Final Answer. Fallback to response text.")
                answer = response.replace("Thought:", "").strip()
                break
            else:
                log.warning("Exceeded max iterations")
                answer = "Tôi đã vượt quá giới hạn suy nghĩ nhưng chưa đưa được câu trả lời hoàn chỉnh."
        else:
            # Fallback to local format if LLM is offline
            from rag.llm import LLMClient
            answer = LLMClient.format_without_llm(
                query=user_query,
                documents=[],
                intent=parsed.intent
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
        )

    def query_stream(
        self,
        user_query: str,
        top_k: int = 5,
    ) -> Iterator[dict]:
        """Process a user query: run ReAct loop silently, then stream the final answer."""
        log.info(f"Streaming query: {user_query[:80]}...")

        self._agent_sources = []
        parsed = self._parser.parse(user_query)
        self._current_parsed = parsed

        from rag.prompts import REACT_SYSTEM_PROMPT, SYSTEM_PROMPT
        agent_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\n"
            f"User Query: {user_query}\n\n"
            "Hãy bắt đầu với Thought: đầu tiên của bạn."
        )

        MAX_ITERATIONS = 5

        if self._llm.is_available:
            # ── Phase 1: Run the full ReAct loop silently ──────────────────────
            # The user only sees brief tool-call status events during this phase.
            raw_final = ""

            for iteration in range(MAX_ITERATIONS):
                response = self._llm.generate(
                    prompt=agent_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.1
                )
                if not response:
                    break
                    
                log.info(f"LLM Response in Stream:\n{response}")
                yield {"type": "thought", "text": response}

                if "Final Answer:" in response:
                    raw_final = response.split("Final Answer:", 1)[1].strip()
                    break

                # Execute tool call
                action_match = re.search(r"Action:\s*`?(\w+)`?", response, re.IGNORECASE)
                if action_match:
                    tool_name = action_match.group(1).strip()
                    remaining = response[action_match.end():].strip()
                    brace_start = remaining.find("{")
                    brace_end = remaining.rfind("}")

                    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                        tool_args_str = remaining[brace_start:brace_end+1]

                        # Brief status — user knows the agent is working
                        yield {"type": "status", "text": f"🔧 Đang tra cứu: {tool_name}..."}

                        observation = self._execute_tool(tool_name, tool_args_str)
                        yield {"type": "observation", "text": f"Observation: {observation}"}
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue
                    else:
                        observation = "Lỗi: Tham số không đúng định dạng JSON. Vui lòng định dạng dưới dạng: Action: tool_name({\"key\": \"value\"})"
                        log.warning(f"Malformed tool call JSON. LLM output: {response}")
                        yield {"type": "observation", "text": f"Observation: {observation}"}
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue

                # No Action and no Final Answer — treat as the answer
                raw_final = response.replace("Thought:", "").strip()
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
                "sources": serialized_sources,
            }

            # ── Phase 3: Stream the final answer ───────────────────────────────
            # Clean prompt — only user query + raw draft, no ReAct history.
            stream_prompt = (
                f"Câu hỏi của người dùng: {user_query}\n\n"
                f"Thông tin tổng hợp từ tra cứu:\n{raw_final}\n\n"
                "Dựa trên thông tin trên, hãy viết câu trả lời hoàn chỉnh dưới "
                "dạng Markdown rõ ràng, có cấu trúc, dễ đọc. Chỉ xuất nội dung "
                "câu trả lời."
            )

            yield {"type": "status", "text": "✍️ Đang tạo câu trả lời..."}
            for token in self._llm.generate_stream(
                prompt=stream_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3,
            ):
                yield {"type": "chunk", "text": token}

            yield {"type": "done"}

        else:
            # Fallback for offline LLM
            from rag.llm import LLMClient
            fallback_text = LLMClient.format_without_llm(
                query=user_query,
                documents=[],
                intent=parsed.intent
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
