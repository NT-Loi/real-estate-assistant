from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DATASET = Path(__file__).with_name("real_estate_queries.testset.jsonl")
REQUIRED_FIELDS = {
    "id",
    "category",
    "user_query",
    "expected_intent",
    "required_filters_or_inputs",
    "expected_behavior",
    "edge_case_type",
    "notes",
}
EXPECTED_CATEGORY_COUNTS = {
    "property_filter_search": 167,
    "nearby_amenities": 167,
    "financial_calculation": 166,
}
VALID_EDGE_TYPES = {
    "normal",
    "missing_info",
    "ambiguous",
    "invalid_value",
    "multi_constraint",
    "comparison",
    "boundary_value",
}
MOJIBAKE_MARKERS = ("Ã", "Ä", "Æ", "áº", "á»", "Â", "�")
SUSPICIOUS_QUESTION_MARK = re.compile(r"\?\?|[A-Za-zÀ-ỹ]\?[A-Za-zÀ-ỹ]")
BAD_REPAIR_PHRASES = (
    "khôngủ",
    "hơnào",
    "nhiưu",
    "Đông bán kính",
    "khơng",
    "tiền sông",
    "đỏ dịch",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_cases() -> list[dict]:
    if not DATASET.exists():
        fail(f"Dataset not found: {DATASET}")

    cases = []
    with DATASET.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                case = json.loads(stripped)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON on line {line_no}: {exc}")
            if not isinstance(case, dict):
                fail(f"Line {line_no} is not a JSON object")
            cases.append(case)
    return cases


def iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def validate(cases: list[dict]) -> None:
    if len(cases) != 500:
        fail(f"Expected 500 test cases, found {len(cases)}")

    ids = [case.get("id") for case in cases]
    duplicates = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        fail(f"Duplicate IDs: {', '.join(sorted(duplicates))}")

    category_counts = Counter(case.get("category") for case in cases)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        fail(f"Expected category counts {EXPECTED_CATEGORY_COUNTS}, found {dict(category_counts)}")

    for index, case in enumerate(cases, start=1):
        missing = REQUIRED_FIELDS - set(case)
        if missing:
            fail(f"{case.get('id', f'line {index}')} missing fields: {', '.join(sorted(missing))}")

        if case["category"] not in EXPECTED_CATEGORY_COUNTS:
            fail(f"{case['id']} has invalid category: {case['category']}")

        if case["edge_case_type"] not in VALID_EDGE_TYPES:
            fail(f"{case['id']} has invalid edge_case_type: {case['edge_case_type']}")

        inputs = case["required_filters_or_inputs"]
        if not isinstance(inputs, dict):
            fail(f"{case['id']} required_filters_or_inputs must be an object")

        if case["edge_case_type"] == "normal" and not inputs:
            fail(f"{case['id']} normal case has empty required_filters_or_inputs")

        for field in REQUIRED_FIELDS - {"required_filters_or_inputs"}:
            if not isinstance(case[field], str) or not case[field].strip():
                fail(f"{case['id']} field {field} must be a non-empty string")

        for text in iter_strings(case):
            if any(marker in text for marker in MOJIBAKE_MARKERS):
                fail(f"{case['id']} appears to contain mojibake text: {text[:80]}")
            if SUSPICIOUS_QUESTION_MARK.search(text):
                fail(f"{case['id']} appears to contain lossy '?' replacement text: {text[:80]}")
            for phrase in BAD_REPAIR_PHRASES:
                if phrase in text:
                    fail(f"{case['id']} appears to contain an over-repaired phrase: {text[:80]}")


def main() -> None:
    cases = load_cases()
    validate(cases)
    counts = Counter(case["category"] for case in cases)
    print("Test set is valid.")
    print(f"Total cases: {len(cases)}")
    for category in EXPECTED_CATEGORY_COUNTS:
        print(f"{category}: {counts[category]}")


if __name__ == "__main__":
    main()
