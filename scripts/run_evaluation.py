"""Run the Agentic RAG system on evaluation_dataset_1000.json.

The script stores raw per-case outputs as JSONL so parser/tool/retrieval/Ragas
metrics can be computed later without re-running the agent.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from utils.logging_config import configure_system_logging


configure_system_logging()
log = logging.getLogger("bds_eval")


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def source_url(source: dict[str, Any]) -> str:
    meta = source.get("metadata") or {}
    return str(source.get("url") or meta.get("url") or meta.get("source_url") or meta.get("source") or "").strip()


def source_title(source: dict[str, Any]) -> str:
    meta = source.get("metadata") or {}
    record = source.get("record") or {}
    return str(
        meta.get("title")
        or meta.get("tieu_de")
        or meta.get("ten_du_an")
        or record.get("title")
        or record.get("tieu_de")
        or record.get("ten_du_an")
        or ""
    ).strip()


def expected_source_hit(expected_sources: list[dict[str, Any]], actual_sources: list[dict[str, Any]]) -> bool:
    if not expected_sources:
        return True
    actual_urls = {source_url(source) for source in actual_sources if source_url(source)}
    actual_titles = " ".join(
        filter(None, [source_title(source) or source.get("text", "")[:300] for source in actual_sources])
    ).lower()
    for expected in expected_sources:
        expected_url = str(expected.get("url") or "").strip()
        expected_title = str(expected.get("title") or "").strip().lower()
        if expected_url and expected_url in actual_urls:
            return True
        if expected_title and expected_title in actual_titles:
            return True
    return False


def run_case(chain: Any, case: dict[str, Any], session_prefix: str, include_events: bool) -> dict[str, Any]:
    case_id = str(case.get("id") or "")
    question = str(case.get("question") or "")
    started = time.monotonic()

    answer_parts: list[str] = []
    tool_calls: list[str] = []
    observations: list[str] = []
    statuses: list[str] = []
    events: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    final_metadata: dict[str, Any] = {}
    error = ""

    try:
        for event in chain.query_stream(question, session_id=f"{session_prefix}-{case_id}"):
            etype = event.get("type")
            if include_events:
                events.append(event)
            if etype == "chunk":
                answer_parts.append(event.get("text", ""))
            elif etype == "metadata":
                metadata = event
            elif etype == "final_metadata":
                final_metadata = event
            elif etype == "tool_call":
                tool_text = str(event.get("text") or "")
                tool_calls.append(tool_text.split("(", 1)[0].strip())
            elif etype == "observation":
                observations.append(str(event.get("text") or "")[:2000])
            elif etype == "status":
                statuses.append(str(event.get("text") or ""))
    except Exception as exc:
        error = str(exc)
        log.exception("Evaluation case %s failed", case_id)

    answer = "".join(answer_parts)
    retrieved_sources = (
        final_metadata.get("retrieved_sources")
        or metadata.get("retrieved_sources")
        or metadata.get("sources")
        or []
    )
    cited_sources = final_metadata.get("cited_sources") or metadata.get("cited_sources") or []
    retrieved_contexts = [str(source.get("text") or "") for source in retrieved_sources if source.get("text")]

    elapsed = round(time.monotonic() - started, 3)
    expected_tools = list(case.get("expected_tools_called") or [])
    tool_hit = all(tool in tool_calls for tool in expected_tools)
    source_hit = expected_source_hit(case.get("expected_sources") or [], retrieved_sources)

    return {
        "id": case_id,
        "category": case.get("category", ""),
        "difficulty": case.get("difficulty", ""),
        "eval_type": case.get("eval_type", []),
        "question": question,
        "reference": case.get("reference_answer", ""),
        "expected": {
            "behavior": case.get("expected_behavior"),
            "intents": case.get("expected_intents", []),
            "filters": case.get("expected_filters", {}),
            "signals": case.get("expected_signals", []),
            "entities": case.get("expected_entities", {}),
            "tools": expected_tools,
            "sources": case.get("expected_sources", []),
        },
        "actual": {
            "answer": answer,
            "intent": metadata.get("intent", ""),
            "filters": metadata.get("filters", {}),
            "tools": tool_calls,
            "statuses": statuses,
            "observations": observations[-8:],
            "retrieved_sources": retrieved_sources,
            "cited_sources": cited_sources,
            "retrieved_contexts": retrieved_contexts,
        },
        "metrics_precheck": {
            "tool_hit": tool_hit,
            "expected_source_hit": source_hit,
            "answer_chars": len(answer),
            "retrieved_source_count": len(retrieved_sources),
            "cited_source_count": len(cited_sources),
            "error": error,
        },
        "elapsed_seconds": elapsed,
        "events": events if include_events else [],
    }


def summarize(results_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    by_category: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = row.get("category", "")
        bucket = by_category.setdefault(category, {"total": 0, "tool_hits": 0, "source_hits": 0, "errors": 0})
        bucket["total"] += 1
        bucket["tool_hits"] += int(bool(row.get("metrics_precheck", {}).get("tool_hit")))
        bucket["source_hits"] += int(bool(row.get("metrics_precheck", {}).get("expected_source_hit")))
        bucket["errors"] += int(bool(row.get("metrics_precheck", {}).get("error")))

    return {
        "total": total,
        "tool_hit_rate": (
            sum(int(bool(row.get("metrics_precheck", {}).get("tool_hit"))) for row in rows) / total
            if total else 0
        ),
        "expected_source_hit_rate": (
            sum(int(bool(row.get("metrics_precheck", {}).get("expected_source_hit"))) for row in rows) / total
            if total else 0
        ),
        "error_count": sum(int(bool(row.get("metrics_precheck", {}).get("error"))) for row in rows),
        "by_category": by_category,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation and save raw outputs.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation_dataset_1000.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "eval_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("USE_GEMINI_FUNCTION_CALLING", "true")
    os.environ["REACT_MAX_ITERATIONS"] = str(args.max_iterations)
    os.environ.setdefault("STREAM_REACT_TRACE", "true")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    cases = load_cases(args.dataset)
    if args.category:
        allowed = set(args.category)
        cases = [case for case in cases if case.get("category") in allowed]
    if args.offset:
        cases = cases[args.offset :]
    if args.limit:
        cases = cases[: args.limit]

    completed: set[str] = set()
    if args.resume and results_path.exists():
        completed = {
            json.loads(line).get("id", "")
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif results_path.exists():
        results_path.unlink()

    config_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "run_name": run_name,
                "case_count": len(cases),
                "limit": args.limit,
                "offset": args.offset,
                "category": args.category,
                "max_iterations": args.max_iterations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    from rag.chain import RAGChain

    chain = RAGChain()
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or "")
        if case_id in completed:
            log.info("Skipping completed case %s", case_id)
            continue
        log.info("Running evaluation case %s/%s: %s", index, len(cases), case_id)
        row = run_case(chain, case, session_prefix=f"eval-{run_name}", include_events=args.include_events)
        append_jsonl(results_path, row)
        summary_path.write_text(json.dumps(summarize(results_path), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(results_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
