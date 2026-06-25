from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
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
log = logging.getLogger("bds_smoke_tests")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _extract_tool_name(text: str) -> str:
    return text.split("(", 1)[0].strip()


MARKDOWN_SOURCE_LINK_RE = re.compile(r"\[[^\]]+\]\(((?:https?://|hf://)[^\s\)]+)\)")
SOURCE_REF_RE = re.compile(r"(?:https?://|hf://)[^\s<>\]\)\"']+")


def _normalize_source_ref(value: str) -> str:
    return str(value or "").strip().rstrip(".,;:!?)\\]}")


def _source_url(source: dict[str, Any]) -> str:
    metadata = source.get("metadata") or {}
    return _normalize_source_ref(
        source.get("url")
        or metadata.get("url")
        or metadata.get("source_url")
        or metadata.get("source")
        or ""
    )


def _source_preview(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection": source.get("collection", ""),
        "score": source.get("score", 0),
        "url": _source_url(source),
        "text": source.get("text", "")[:1000],
        "metadata": source.get("metadata", {}),
    }


def _extract_answer_source_refs(answer: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    # Prefer the real target of markdown links. Otherwise a shortened URL label
    # can look like a separate source from the clickable URL in parentheses.
    text = MARKDOWN_SOURCE_LINK_RE.sub(lambda match: match.group(1), answer or "")
    for match in SOURCE_REF_RE.finditer(text):
        ref = _normalize_source_ref(match.group(0))
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs


def _build_source_previews(answer: str, sources: list[dict[str, Any]], limit: int = 10) -> tuple[str, list[dict[str, Any]]]:
    cited_refs = _extract_answer_source_refs(answer)
    if not cited_refs:
        return "retrieved_sources_fallback", [_source_preview(source) for source in sources[:limit]]

    by_url = {_source_url(source): source for source in sources if _source_url(source)}
    previews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in cited_refs:
        if ref in seen:
            continue
        seen.add(ref)
        matched = by_url.get(ref)
        if matched:
            previews.append(_source_preview(matched))
        else:
            previews.append({
                "collection": "",
                "score": 0,
                "url": ref,
                "text": "",
                "metadata": {"note": "Cited in answer but not present in retrieved metadata sources."},
            })
        if len(previews) >= limit:
            break
    return "answer_citations", previews


def _run_case(chain: Any, case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["id"]
    query = case["query"]
    started = time.monotonic()
    log.info("Running smoke case %s", case_id)

    answer_parts: list[str] = []
    tool_calls: list[str] = []
    statuses: list[str] = []
    metadata: dict[str, Any] = {}
    errors: list[str] = []

    for event in chain.query_stream(query, session_id=f"smoke-{case_id}"):
        etype = event.get("type")
        if etype == "chunk":
            answer_parts.append(event.get("text", ""))
        elif etype == "metadata":
            metadata = event
        elif etype == "tool_call":
            tool_calls.append(_extract_tool_name(event.get("text", "")))
        elif etype == "status":
            statuses.append(event.get("text", ""))

    answer = "".join(answer_parts)
    source_count = len(metadata.get("sources") or [])
    min_answer_chars = int(case.get("min_answer_chars", 1))
    min_sources = int(case.get("min_sources", 0))
    if len(answer) < min_answer_chars:
        errors.append(f"answer too short: {len(answer)} < {min_answer_chars}")
    if source_count < min_sources:
        errors.append(f"too few sources: {source_count} < {min_sources}")

    answer_norm = answer.lower()
    keywords = case.get("answer_keywords_any") or []
    if keywords and not any(str(keyword).lower() in answer_norm for keyword in keywords):
        errors.append(f"answer missing any expected keyword: {keywords}")

    expected_tools = case.get("tools_any") or []
    if expected_tools and not tool_calls:
        errors.append(f"no tool calls observed; expected any of {expected_tools}")
    elif expected_tools and not any(tool in tool_calls for tool in expected_tools):
        errors.append(f"unexpected tools used: got {tool_calls}, expected any of {expected_tools}")

    elapsed = round(time.monotonic() - started, 2)
    passed = not errors
    log.info(
        "Smoke case %s %s in %.2fs (answer_chars=%s, sources=%s, tools=%s)",
        case_id,
        "passed" if passed else "failed",
        elapsed,
        len(answer),
        source_count,
        tool_calls,
    )
    result = {
        "id": case_id,
        "query": query,
        "passed": passed,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "answer_chars": len(answer),
        "source_count": source_count,
        "intent": metadata.get("intent", ""),
        "filters": metadata.get("filters", {}),
        "tools": tool_calls,
        "statuses": statuses[-8:],
    }
    if case.get("_include_answers"):
        preview_basis, source_previews = _build_source_previews(answer, metadata.get("sources") or [])
        result["answer"] = answer
        result["source_preview_basis"] = preview_basis
        result["source_previews"] = source_previews
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small RAG smoke-test set.")
    parser.add_argument("--cases", type=Path, default=ROOT / "tests" / "rag_smoke_cases.json")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--max-iterations", type=int, default=8, help="REACT_MAX_ITERATIONS for smoke runs.")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "rag_smoke_results.json")
    parser.add_argument("--include-answers", action="store_true", help="Include full answers and source previews in the output report.")
    parser.add_argument("--include-react-ui-events", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("USE_GEMINI_FUNCTION_CALLING", "true")
    os.environ["REACT_MAX_ITERATIONS"] = str(args.max_iterations)
    if args.include_react_ui_events:
        os.environ["STREAM_REACT_TRACE"] = "true"
    else:
        os.environ.setdefault("STREAM_REACT_TRACE", "true")

    from rag.chain import RAGChain

    cases = _load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    if args.include_answers:
        for case in cases:
            case["_include_answers"] = True

    chain = RAGChain()
    results = [_run_case(chain, case) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    report = {
        "passed": passed,
        "total": len(results),
        "all_passed": passed == len(results),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote smoke-test report to %s", args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
