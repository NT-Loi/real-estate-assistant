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

        # Eagerly load the embedding model to avoid cold-start during first chat query
        if hasattr(self._store, "_embedder") and hasattr(self._store._embedder, "_load"):
            self._store._embedder._load()

        log.info("RAG Chain initialized in Agentic mode")
        log.info(f"  LLM available: {self._llm.is_available}")
        log.info(f"  Collections: {self._store.stats()}")

    # ---------------------------------------------------------------------------
    # Tools definition
    # ---------------------------------------------------------------------------
    def _tool_semantic_search(self, query_text: str, collections: list[str], limit: int = 5) -> str:
        """Run semantic search on Qdrant."""
        try:
            log.info(f"Tool called: semantic_search({query_text=}, {collections=}, {limit=})")
            docs = self._retriever.retrieve(
                query_text=query_text,
                collections=collections,
                top_k=limit
            )
            if not docs:
                return "Không tìm thấy kết quả tìm kiếm ngữ nghĩa nào."
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi tìm kiếm ngữ nghĩa: {e}"

    def _tool_keyword_search(self, query_text: str, collections: list[str], limit: int = 5) -> str:
        """Run exact keyword search using ILIKE on PostgreSQL."""
        try:
            log.info(f"Tool called: keyword_search({query_text=}, {collections=}, {limit=})")
            results = []
            with self._store.pg.get_cursor() as cur:
                for coll in collections:
                    if coll == "listings":
                        q = "SELECT raw_json FROM listings WHERE tieu_de ILIKE %s OR mo_ta ILIKE %s LIMIT %s"
                        cur.execute(q, (f"%{query_text}%", f"%{query_text}%", limit))
                    elif coll == "projects":
                        q = "SELECT raw_json FROM projects WHERE ten_du_an ILIKE %s OR khu_vuc ILIKE %s LIMIT %s"
                        cur.execute(q, (f"%{query_text}%", f"%{query_text}%", limit))
                    elif coll == "articles":
                        q = "SELECT raw_json FROM articles WHERE tieu_de ILIKE %s OR mo_ta ILIKE %s LIMIT %s"
                        cur.execute(q, (f"%{query_text}%", f"%{query_text}%", limit))
                    else:
                        continue
                    
                    rows = cur.fetchall()
                    for r in rows:
                        payload = r[0]
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        results.append((coll, payload))
            
            if not results:
                return "Không tìm thấy kết quả từ khóa chính xác nào."
            
            from rag.retriever import RetrievedDocument
            docs = []
            for coll, r in results[:limit]:
                desc = r.get("mo_ta_chi_tiet") or r.get("mo_ta") or r.get("tieu_de") or str(r)
                doc = RetrievedDocument(
                    text=desc,
                    metadata={"url": r.get("url", "")},
                    score=1.0,
                    collection=coll,
                    record=r
                )
                docs.append(doc)
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
                
            limit = int(kwargs.get("limit", 5))
            
            where_sql = " AND ".join(clauses)
            sql = f"SELECT raw_json FROM listings WHERE {where_sql} LIMIT %s"
            params.append(limit)
            
            results = []
            with self._store.pg.get_cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
                for r in rows:
                    payload = r[0]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
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
                    metadata={"url": url},
                    score=1.0,
                    collection="listings",
                    record=r
                )
                docs.append(doc)
            self._agent_sources.extend(docs)
            return format_context(docs)
        except Exception as e:
            return f"Lỗi khi lọc bất động sản: {e}"

    def _tool_calculate_loan(self, **kwargs) -> str:
        """Run finance loan calculations."""
        try:
            log.info(f"Tool called: calculate_loan({kwargs=})")
            price_trieu = float(kwargs.get("property_price_trieu", 0))
            if price_trieu <= 0:
                return "Lỗi: Giá trị bất động sản phải lớn hơn 0."
                
            down_pct = float(kwargs.get("down_payment_pct", 0.3))
            annual_rate = float(kwargs.get("annual_rate_pct", 9.0))
            term_years = int(kwargs.get("term_years", 20))
            
            plan = FinanceCalculator.loan_from_property_price(
                property_price_vnd=price_trieu * 1_000_000,
                down_payment_pct=down_pct,
                annual_rate_pct=annual_rate,
                term_years=term_years
            )
            summary = plan.summary_text()
            
            scenarios = FinanceCalculator.multi_scenario(plan.principal_vnd)
            scenarios_str = "\n--- So sánh theo lãi suất ---"
            for s in scenarios:
                def fmt(v):
                    return f"{v/1_000_000:.0f} triệu" if v < 1_000_000_000 else f"{v/1_000_000_000:.2f} tỷ"
                scenarios_str += f"\n  {s.annual_rate_pct:.1f}%/năm -> {fmt(s.monthly_payment_vnd)}/tháng"
                
            return summary + "\n" + scenarios_str
        except Exception as e:
            return f"Lỗi khi tính toán khoản vay: {e}"

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
                collections=args.get("collections", []),
                limit=args.get("limit", 5)
            )
        elif tool_name == "keyword_search":
            return self._tool_keyword_search(
                query_text=args.get("query_text", ""),
                collections=args.get("collections", []),
                limit=args.get("limit", 5)
            )
        elif tool_name == "filter_listings":
            return self._tool_filter_listings(**args)
        elif tool_name == "calculate_loan":
            return self._tool_calculate_loan(**args)
        elif tool_name == "get_market_statistics":
            return self._tool_get_market_statistics(**args)
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
                action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
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
        """Process a user query and stream thoughts + tool calls + final answers."""
        log.info(f"Streaming query: {user_query[:80]}...")
        
        self._agent_sources = []
        parsed = self._parser.parse(user_query)

        # Setup ReAct system instructions
        from rag.prompts import REACT_SYSTEM_PROMPT, SYSTEM_PROMPT
        agent_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\n"
            f"User Query: {user_query}\n\n"
            "Hãy bắt đầu với Thought: đầu tiên của bạn."
        )

        MAX_ITERATIONS = 5

        if self._llm.is_available:
            for iteration in range(MAX_ITERATIONS):
                response = self._llm.generate(
                    prompt=agent_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.1
                )
                if not response:
                    break

                # Stream intermediate thought to the UI
                thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", response, re.DOTALL | re.IGNORECASE)
                thought_text = thought_match.group(1).strip() if thought_match else ""
                if thought_text:
                    yield {"type": "chunk", "text": f"\n*[Suy nghĩ: {thought_text}]*\n"}

                if "Final Answer:" in response:
                    parts = response.split("Final Answer:", 1)
                    final_answer = parts[1].strip()
                    
                    # Yield metadata before streaming the final answer
                    serialized_sources = []
                    for doc in self._agent_sources:
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

                    yield {"type": "chunk", "text": f"\n\n{final_answer}"}
                    yield {"type": "done"}
                    return

                # Parse Action
                action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
                if action_match:
                    tool_name = action_match.group(1).strip()
                    remaining = response[action_match.end():].strip()
                    brace_start = remaining.find("{")
                    brace_end = remaining.rfind("}")
                    
                    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                        tool_args_str = remaining[brace_start:brace_end+1]
                        
                        yield {"type": "chunk", "text": f"\n*🔧 Thực hiện công cụ: `{tool_name}({tool_args_str})`*\n"}
                        
                        observation = self._execute_tool(tool_name, tool_args_str)
                        agent_prompt += f"\n{response}\nObservation: {observation}\n"
                        continue

                # Fallback
                clean_response = response.replace("Thought:", "").strip()
                
                serialized_sources = []
                yield {
                    "type": "metadata",
                    "intent": parsed.intent,
                    "filters": parsed.filters,
                    "sources": serialized_sources,
                }
                yield {"type": "chunk", "text": f"\n\n{clean_response}"}
                yield {"type": "done"}
                return
            else:
                yield {"type": "chunk", "text": "\nTôi đã vượt quá giới hạn suy nghĩ nhưng chưa đưa được câu trả lời hoàn chỉnh."}
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
