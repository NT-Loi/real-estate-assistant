"""Compute deterministic non-Ragas metrics from saved evaluation outputs.

Input is the JSONL produced by scripts/run_evaluation.py. This script does not
call the RAG system or any LLM judge; it only aggregates fields already saved in
results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def rate(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    idx = (len(ordered) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    frac = idx - lower
    return round(ordered[lower] * (1 - frac) + ordered[upper] * frac, 3)


def normalize_behavior(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"refuse_out_of_scope", "refuse", "out_of_scope"}:
        return "refuse"
    if raw in {"clarify", "ask_clarification"}:
        return "clarify"
    if raw in {"answer", "respond"}:
        return "answer"
    return raw


def actual_intents(row: dict[str, Any]) -> set[str]:
    raw = str((row.get("actual") or {}).get("intent") or "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def expected_intents(row: dict[str, Any]) -> set[str]:
    return {str(item).strip() for item in (row.get("expected") or {}).get("intents") or [] if str(item).strip()}


def filter_key_set(filters: Any) -> set[str]:
    if not isinstance(filters, dict):
        return set()
    return {str(k) for k, v in filters.items() if v not in (None, "", {}, [])}


def expected_filter_hit(row: dict[str, Any]) -> bool:
    expected = filter_key_set((row.get("expected") or {}).get("filters"))
    if not expected:
        return True
    actual = filter_key_set((row.get("actual") or {}).get("filters"))
    return expected.issubset(actual)


def expected_tool_exact(row: dict[str, Any]) -> bool:
    expected = list((row.get("expected") or {}).get("tools") or [])
    actual = list((row.get("actual") or {}).get("tools") or [])
    return expected == actual


def expected_tool_subset(row: dict[str, Any]) -> bool:
    expected = set((row.get("expected") or {}).get("tools") or [])
    actual = set((row.get("actual") or {}).get("tools") or [])
    return expected.issubset(actual)


def behavior_match(row: dict[str, Any]) -> bool:
    expected = normalize_behavior((row.get("expected") or {}).get("behavior"))
    audit = (row.get("actual") or {}).get("answer_audit") or {}
    actual = normalize_behavior(audit.get("actual_behavior"))
    return bool(expected and actual and expected == actual)


def row_metrics(row: dict[str, Any], min_answer_chars: int) -> dict[str, Any]:
    actual = row.get("actual") or {}
    pre = row.get("metrics_precheck") or {}
    audit = actual.get("answer_audit") or {}
    answer = str(actual.get("answer") or "")
    exp_intents = expected_intents(row)
    act_intents = actual_intents(row)

    return {
        "tool_subset_hit": expected_tool_subset(row),
        "tool_exact_hit": expected_tool_exact(row),
        "source_hit": bool(pre.get("expected_source_hit")),
        "behavior_match": behavior_match(row),
        "intent_hit": exp_intents.issubset(act_intents) if exp_intents else True,
        "filter_key_hit": expected_filter_hit(row),
        "has_error": bool(pre.get("error")),
        "empty_answer": not answer.strip(),
        "short_answer": len(answer.strip()) < min_answer_chars,
        "citation_hallucination": bool(audit.get("hallucinated_citation_urls")),
        "has_citation": bool(audit.get("used_citation_urls") or actual.get("cited_sources")),
        "retrieved_source_count": int(pre.get("retrieved_source_count") or len(actual.get("retrieved_sources") or [])),
        "cited_source_count": int(pre.get("cited_source_count") or len(actual.get("cited_sources") or [])),
        "answer_chars": int(pre.get("answer_chars") or len(answer)),
        "elapsed_seconds": float(row.get("elapsed_seconds") or 0),
    }


def summarize_rows(rows: list[dict[str, Any]], min_answer_chars: int) -> dict[str, Any]:
    metrics = [row_metrics(row, min_answer_chars=min_answer_chars) for row in rows]
    total = len(rows)

    bool_keys = [
        "tool_subset_hit",
        "tool_exact_hit",
        "source_hit",
        "behavior_match",
        "intent_hit",
        "filter_key_hit",
        "has_error",
        "empty_answer",
        "short_answer",
        "citation_hallucination",
        "has_citation",
    ]
    aggregate = {
        f"{key}_rate": rate(sum(bool(m[key]) for m in metrics), total)
        for key in bool_keys
    }
    elapsed = [m["elapsed_seconds"] for m in metrics if m["elapsed_seconds"] > 0]
    answer_chars = [m["answer_chars"] for m in metrics]
    retrieved_counts = [m["retrieved_source_count"] for m in metrics]
    cited_counts = [m["cited_source_count"] for m in metrics]

    aggregate.update({
        "total": total,
        "latency_seconds": {
            "avg": round(statistics.mean(elapsed), 3) if elapsed else 0.0,
            "p50": percentile(elapsed, 0.50),
            "p90": percentile(elapsed, 0.90),
            "p95": percentile(elapsed, 0.95),
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
        "answer_chars": {
            "avg": round(statistics.mean(answer_chars), 1) if answer_chars else 0.0,
            "p50": percentile([float(v) for v in answer_chars], 0.50),
            "p90": percentile([float(v) for v in answer_chars], 0.90),
            "min": min(answer_chars) if answer_chars else 0,
        },
        "retrieved_source_count_avg": round(statistics.mean(retrieved_counts), 2) if retrieved_counts else 0.0,
        "cited_source_count_avg": round(statistics.mean(cited_counts), 2) if cited_counts else 0.0,
    })
    return aggregate


def breakdown(rows: list[dict[str, Any]], field: str, min_answer_chars: int) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            keys = value or ["<empty>"]
        else:
            keys = [str(value or "<empty>")]
        for key in keys:
            buckets[str(key)].append(row)
    return {
        key: summarize_rows(bucket_rows, min_answer_chars=min_answer_chars)
        for key, bucket_rows in sorted(buckets.items())
    }


def worst_cases(rows: list[dict[str, Any]], min_answer_chars: int, limit: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        m = row_metrics(row, min_answer_chars=min_answer_chars)
        issue_score = sum(
            int(not m[key])
            for key in ("tool_subset_hit", "source_hit", "behavior_match", "intent_hit", "filter_key_hit")
        )
        issue_score += int(m["has_error"]) + int(m["empty_answer"]) + int(m["citation_hallucination"])
        if issue_score <= 0:
            continue
        selected.append((issue_score, m["elapsed_seconds"], row, m))
    selected.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("id"))))
    return [
        {
            "id": row.get("id"),
            "category": row.get("category"),
            "question": row.get("question"),
            "issue_score": score,
            "metrics": metrics,
            "expected_tools": (row.get("expected") or {}).get("tools") or [],
            "actual_tools": (row.get("actual") or {}).get("tools") or [],
            "expected_behavior": (row.get("expected") or {}).get("behavior"),
            "actual_behavior": ((row.get("actual") or {}).get("answer_audit") or {}).get("actual_behavior"),
            "error": (row.get("metrics_precheck") or {}).get("error") or "",
        }
        for score, _, row, metrics in selected[:limit]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute deterministic non-Ragas metrics from eval results.")
    parser.add_argument("--input", type=Path, required=True, help="Path to logs/eval_runs/<run>/results.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ragas", type=Path, default=None, help="Optional ragas_results.json to include aggregate scores.")
    parser.add_argument("--min-answer-chars", type=int, default=80)
    parser.add_argument("--worst-limit", type=int, default=30)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    ragas_payload = load_json(args.ragas)
    output = args.output or (args.input.parent / "custom_metrics.json")

    payload = {
        "input": str(args.input),
        "ragas_input": str(args.ragas) if args.ragas else "",
        "aggregate": summarize_rows(rows, min_answer_chars=args.min_answer_chars),
        "by_category": breakdown(rows, "category", min_answer_chars=args.min_answer_chars),
        "by_difficulty": breakdown(rows, "difficulty", min_answer_chars=args.min_answer_chars),
        "by_eval_type": breakdown(rows, "eval_type", min_answer_chars=args.min_answer_chars),
        "expected_tool_distribution": dict(Counter(t for row in rows for t in (row.get("expected") or {}).get("tools") or [])),
        "actual_tool_distribution": dict(Counter(t for row in rows for t in (row.get("actual") or {}).get("tools") or [])),
        "worst_cases": worst_cases(rows, min_answer_chars=args.min_answer_chars, limit=args.worst_limit),
    }
    if ragas_payload:
        payload["ragas_aggregate"] = ragas_payload.get("aggregate", {})
        payload["ragas_sample_count"] = ragas_payload.get("sample_count")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
