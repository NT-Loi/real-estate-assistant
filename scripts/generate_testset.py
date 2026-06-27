"""Generate a validated synthetic evaluation dataset for Agentic RAG.

This dataset is meant for parser/tool-routing/retrieval regression tests.
It is not a fully curated human golden-answer set. For end-to-end answer
correctness, build a smaller manually verified set on top of these cases.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data-Loi"
OUTPUT_FILE = PROJECT_ROOT / "evaluation_dataset_1000.json"
RANDOM_SEED = 42

VALID_INTENTS = {
    "search_listing",
    "lifestyle_search",
    "compare_project",
    "ask_knowledge",
    "calculate_finance",
    "market_report",
}

VALID_SIGNALS = {
    "metro",
    "school",
    "hospital",
    "park",
    "shopping",
    "flood",
    "livability",
    "safety",
    "appreciation",
    "infrastructure",
}

VALID_TOOLS = {
    "semantic_search",
    "hybrid_search",
    "keyword_search",
    "filter_listings",
    "find_nearby_pois",
    "find_pois_near_location",
    "search_pois",
    "find_listings_near_pois",
    "search_location",
    "web_search",
    "web_research",
    "read_url",
    "calculate_finance",
    "analyze_market_trend",
    "get_market_statistics",
}

DISTRICT_PREFIXES = (
    "Quận ",
    "Huyện ",
    "Thành phố ",
    "Thị xã ",
    "TP. ",
    "TP ",
)

LIFESTYLE_SURFACES: list[tuple[str, str]] = [
    ("an ninh tốt", "safety"),
    ("yên tĩnh", "safety"),
    ("ít ngập nước", "flood"),
    ("không ngập nước", "flood"),
    ("chất lượng sống tốt", "livability"),
    ("tiện ích nội khu tốt", "livability"),
    ("khả năng tăng giá", "appreciation"),
    ("hạ tầng kết nối tốt", "infrastructure"),
]

POI_SURFACES: list[tuple[str, str, str]] = [
    ("trường học", "school", "find_pois_near_location"),
    ("bệnh viện", "hospital", "find_pois_near_location"),
    ("ga metro", "metro", "find_pois_near_location"),
    ("siêu thị", "shopping", "find_pois_near_location"),
    ("công viên", "park", "find_pois_near_location"),
    ("trung tâm thương mại", "shopping", "find_pois_near_location"),
]


def choose_poi_for_location(loc: dict[str, str]) -> tuple[str, str, str]:
    location_text = f"{loc.get('district', '')} {loc.get('province', '')}"
    allowed = POI_SURFACES
    if "Hà Nội" not in location_text and "Hồ Chí Minh" not in location_text:
        allowed = [item for item in POI_SURFACES if item[1] != "metro"]
    return random.choice(allowed)


def load_json(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def first_line(value: object) -> str:
    lines = str(value or "").splitlines()
    if not lines:
        return ""
    return lines[0].replace(". Xem bản đồ", "").strip()


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_good_project_name(value: object) -> bool:
    name = clean_text(value)
    lowered = name.lower()
    return bool(name) and "chờ một chút" not in lowered and name not in {".", "..."}


def extract_location_parts(*values: object) -> dict[str, str]:
    text = ", ".join(first_line(v) for v in values if first_line(v))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    district = ""
    province = parts[-1] if parts else ""
    for part in parts:
        if part.startswith(DISTRICT_PREFIXES):
            district = part
    return {"district": district, "province": province}


def source_ref(record: dict[str, Any], collection: str) -> dict[str, str]:
    return {
        "collection": collection,
        "url": clean_text(record.get("url")),
        "title": clean_text(
            record.get("tieu_de")
            or record.get("ten_du_an")
            or record.get("title")
            or record.get("du_an")
        ),
    }


def money_label_to_trieu(value: object) -> int | None:
    text = str(value or "").lower()
    if not text or "thỏa thuận" in text or "thoả thuận" in text or "liên hệ" in text:
        return None
    if "/m" in text or "tr/m" in text:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    if "tỷ" in text or "ty" in text:
        return int(number * 1000)
    if "triệu" in text or "tr" in text:
        return int(number)
    return None


def has_total_price(record: dict[str, Any]) -> bool:
    return money_label_to_trieu(record.get("gia")) is not None


def monthly_payment(principal_vnd: float, annual_rate_pct: float, term_years: int) -> float:
    r = annual_rate_pct / 100 / 12
    n = term_years * 12
    if r == 0:
        return principal_vnd / n
    return principal_vnd * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def fmt_money(vnd: float) -> str:
    return f"{vnd / 1_000_000_000:.2f} tỷ" if vnd >= 1_000_000_000 else f"{vnd / 1_000_000:.0f} triệu"


def case(
    *,
    case_id: str,
    category: str,
    question: str,
    expected_intents: list[str],
    expected_tools_called: list[str],
    eval_type: list[str],
    expected_filters: dict[str, Any] | None = None,
    expected_signals: list[str] | None = None,
    expected_entities: dict[str, Any] | None = None,
    expected_sources: list[dict[str, str]] | None = None,
    expected_behavior: str = "answer",
    reference_answer: str = "",
    difficulty: str = "medium",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "difficulty": difficulty,
        "eval_type": eval_type,
        "question": question,
        "expected_behavior": expected_behavior,
        "expected_intents": expected_intents,
        "expected_filters": expected_filters or {},
        "expected_signals": expected_signals or [],
        "expected_entities": expected_entities or {},
        "expected_tools_called": expected_tools_called,
        "expected_sources": expected_sources or [],
        "reference_answer": reference_answer,
        "notes": notes,
    }


def choose_with_district(records: list[dict[str, Any]], *location_keys: str) -> dict[str, Any]:
    candidates = []
    for record in records:
        loc = extract_location_parts(*(record.get(k) for k in location_keys))
        if loc["district"]:
            candidates.append(record)
    return random.choice(candidates or records)


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids detected")
    for idx, row in enumerate(rows):
        prefix = f"{row.get('id', idx)}"
        invalid_intents = [i for i in row.get("expected_intents", []) if i not in VALID_INTENTS]
        invalid_signals = [s for s in row.get("expected_signals", []) if s not in VALID_SIGNALS]
        invalid_tools = [t for t in row.get("expected_tools_called", []) if t not in VALID_TOOLS]
        if invalid_intents:
            errors.append(f"{prefix}: invalid intents {invalid_intents}")
        if invalid_signals:
            errors.append(f"{prefix}: invalid signals {invalid_signals}")
        if invalid_tools:
            errors.append(f"{prefix}: invalid tools {invalid_tools}")
        if not row.get("question"):
            errors.append(f"{prefix}: empty question")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Dataset validation failed with {len(errors)} errors:\n{preview}")


def main() -> None:
    random.seed(RANDOM_SEED)
    print("Loading data...")
    projects = load_json("projects.json")
    listings_rent = load_json("listings_cho_thue.json")
    listings_sale = load_json("listings_ban.json")

    valid_projects = [
        p for p in projects
        if is_good_project_name(p.get("ten_du_an"))
        and p.get("khu_vuc")
        and extract_location_parts(p.get("dia_chi"), p.get("khu_vuc"))["district"]
    ]
    valid_rentals = [
        l for l in listings_rent
        if l.get("loai_nha_dat") and l.get("gia") and l.get("url")
        and has_total_price(l)
        and extract_location_parts(l.get("dia_chi"), l.get("khu_vuc"))["district"]
    ]
    valid_sales = [
        l for l in listings_sale
        if l.get("loai_nha_dat") and l.get("gia") and l.get("url")
        and has_total_price(l)
        and extract_location_parts(l.get("dia_chi"), l.get("khu_vuc"))["district"]
    ]

    if not valid_projects or not valid_rentals or not valid_sales:
        raise RuntimeError("Not enough valid source data to generate the test set.")

    dataset: list[dict[str, Any]] = []

    # 1. Structured Filtering (400)
    for i, listing in enumerate(random.sample(valid_rentals, 200)):
        loc = extract_location_parts(listing.get("dia_chi"), listing.get("khu_vuc"))
        prop_type = clean_text(listing.get("loai_nha_dat")) or "Căn hộ chung cư"
        price_trieu = money_label_to_trieu(listing.get("gia"))
        price_text = clean_text(listing.get("gia")) or "theo ngân sách phù hợp"
        detail_parts = []
        if listing.get("dien_tich"):
            detail_parts.append(f"diện tích {clean_text(listing.get('dien_tich'))}")
        if listing.get("so_phong_ngu"):
            detail_parts.append(f"{clean_text(listing.get('so_phong_ngu'))} ngủ")
        detail = ", " + ", ".join(detail_parts[:2]) if detail_parts else ""
        question = f"Tìm {prop_type.lower()} cho thuê tại {loc['district']}, giá khoảng {price_text}{detail}."
        filters = {"quan_huyen": loc["district"], "loai_hinh": "cho_thue", "loai_nha_dat": prop_type}
        if price_trieu:
            filters["gia_trieu"] = {"$lte": max(price_trieu + 2, math.ceil(price_trieu * 1.15))}
        dataset.append(case(
            case_id=f"structured_rent_{i:03d}",
            category="Structured Filtering",
            difficulty="easy",
            eval_type=["parser", "tool_routing", "retrieval"],
            question=question,
            expected_intents=["search_listing"],
            expected_filters=filters,
            expected_tools_called=["filter_listings"],
            expected_sources=[source_ref(listing, "listings")],
            reference_answer=(
                f"Cần trả về các {prop_type.lower()} cho thuê tại {loc['district']} "
                f"với giá tham khảo khoảng {price_text}, kèm URL tin đăng nếu có."
            ),
        ))

    for i, listing in enumerate(random.sample(valid_sales, 200)):
        loc = extract_location_parts(listing.get("dia_chi"), listing.get("khu_vuc"))
        prop_type = clean_text(listing.get("loai_nha_dat")) or "Bất động sản"
        price_trieu = money_label_to_trieu(listing.get("gia"))
        price_text = clean_text(listing.get("gia")) or "theo ngân sách phù hợp"
        detail_parts = []
        if listing.get("dien_tich"):
            detail_parts.append(f"diện tích {clean_text(listing.get('dien_tich'))}")
        if listing.get("so_phong_ngu"):
            detail_parts.append(f"{clean_text(listing.get('so_phong_ngu'))} ngủ")
        detail = ", " + ", ".join(detail_parts[:2]) if detail_parts else ""
        question = f"Tìm {prop_type.lower()} bán tại {loc['district']}, giá khoảng {price_text}{detail}."
        filters = {"quan_huyen": loc["district"], "loai_hinh": "ban", "loai_nha_dat": prop_type}
        if price_trieu:
            filters["gia_trieu"] = {"$lte": math.ceil(price_trieu * 1.15)}
        dataset.append(case(
            case_id=f"structured_sale_{i:03d}",
            category="Structured Filtering",
            difficulty="easy",
            eval_type=["parser", "tool_routing", "retrieval"],
            question=question,
            expected_intents=["search_listing"],
            expected_filters=filters,
            expected_tools_called=["filter_listings"],
            expected_sources=[source_ref(listing, "listings")],
            reference_answer=(
                f"Cần trả về các {prop_type.lower()} bán tại {loc['district']} "
                f"với giá tham khảo khoảng {price_text}, kèm URL tin đăng nếu có."
            ),
        ))

    # 2. Semantic & Lifestyle (200)
    for i, project in enumerate(random.sample(valid_projects, 200)):
        surface, signal = random.choice(LIFESTYLE_SURFACES)
        name = clean_text(project.get("ten_du_an"))
        question = f"Theo dữ liệu nội bộ và đánh giá cư dân, dự án {name} có {surface} không?"
        dataset.append(case(
            case_id=f"lifestyle_{i:03d}",
            category="Semantic & Lifestyle",
            difficulty="medium",
            eval_type=["parser", "tool_routing", "retrieval"],
            question=question,
            expected_intents=["lifestyle_search"],
            expected_signals=[signal],
            expected_entities={"project_name": name},
            expected_tools_called=["hybrid_search"],
            expected_sources=[source_ref(project, "projects")],
            reference_answer=(
                f"Cần tổng hợp bằng chứng nội bộ về tiêu chí {surface} của dự án {name}; "
                "nếu thiếu dữ liệu, câu trả lời phải nói rõ mức độ chắc chắn thấp."
            ),
        ))

    # 3. POI & Spatial (100)
    for i, project in enumerate(random.sample(valid_projects, 100)):
        loc = extract_location_parts(project.get("dia_chi"), project.get("khu_vuc"))
        poi_surface, signal, expected_tool = choose_poi_for_location(loc)
        name = clean_text(project.get("ten_du_an"))
        question = f"Xung quanh dự án {name} tại {loc['district']} trong bán kính 2km có {poi_surface} nào gần nhất?"
        dataset.append(case(
            case_id=f"poi_{i:03d}",
            category="POI & Spatial",
            difficulty="medium",
            eval_type=["parser", "tool_routing"],
            question=question,
            expected_intents=["lifestyle_search"],
            expected_signals=[signal],
            expected_entities={"project_name": name, "radius_m": 2000, "poi": poi_surface},
            expected_tools_called=[expected_tool],
            expected_sources=[source_ref(project, "projects")],
            reference_answer=(
                f"Cần dùng công cụ không gian để liệt kê {poi_surface} trong bán kính 2km quanh {name}; "
                "không được nói hệ thống không hỗ trợ bán kính."
            ),
        ))

    # 4. Market Analytics (100)
    project_types = ["Căn hộ chung cư", "Nhà riêng", "Đất"]
    market_pairs = []
    seen_market_keys = set()
    for project in random.sample(valid_projects, len(valid_projects)):
        loc = extract_location_parts(project.get("dia_chi"), project.get("khu_vuc"))
        for prop_type in random.sample(project_types, len(project_types)):
            key = (loc["district"], loc["province"], prop_type)
            if key not in seen_market_keys:
                seen_market_keys.add(key)
                market_pairs.append((project, loc, prop_type))
            if len(market_pairs) >= 100:
                break
        if len(market_pairs) >= 100:
            break
    for i, (project, loc, prop_type) in enumerate(market_pairs):
        question = f"Giá trung bình của {prop_type.lower()} tại {loc['district']}, {loc['province']} hiện nay là khoảng bao nhiêu?"
        dataset.append(case(
            case_id=f"market_{i:03d}",
            category="Market Analytics",
            difficulty="medium",
            eval_type=["parser", "tool_routing"],
            question=question,
            expected_intents=["market_report"],
            expected_filters={"quan_huyen": loc["district"], "loai_nha_dat": prop_type},
            expected_tools_called=["get_market_statistics"],
            expected_entities={"district": loc["district"], "property_type": prop_type},
            reference_answer=(
                f"Cần đọc market_snapshots để trả thống kê giá/diện tích/số tin cho {prop_type} tại {loc['district']}."
            ),
        ))

    # 5. Financial Calculation (100)
    finance_params = [
        (price, term, rate)
        for price in range(20, 51)
        for term in [10, 15, 20]
        for rate in [7.5, 8.0, 8.5, 9.0, 9.5]
    ]
    for i, (price_unit, term_years, annual_rate) in enumerate(random.sample(finance_params, 100)):
        price_vnd = price_unit * 100_000_000
        loan_vnd = int(price_vnd * 0.7)
        monthly = monthly_payment(loan_vnd, annual_rate, term_years)
        question = (
            f"Tôi tính mua căn chung cư giá {price_vnd / 1_000_000_000:.1f} tỷ, "
            f"vay ngân hàng {loan_vnd / 1_000_000_000:.1f} tỷ trong {term_years} năm "
            f"với lãi suất {annual_rate}%/năm. Mỗi tháng tôi phải trả bao nhiêu tiền?"
        )
        dataset.append(case(
            case_id=f"finance_{i:03d}",
            category="Financial Calculation",
            difficulty="easy",
            eval_type=["parser", "tool_routing", "deterministic_answer"],
            question=question,
            expected_intents=["calculate_finance"],
            expected_tools_called=["calculate_finance"],
            expected_entities={
                "property_price_vnd": price_vnd,
                "loan_principal_vnd": loan_vnd,
                "term_years": term_years,
                "annual_rate_pct": annual_rate,
                "expected_monthly_payment_vnd": round(monthly),
            },
            reference_answer=(
                f"Với khoản vay {fmt_money(loan_vnd)} trong {term_years} năm ở lãi suất "
                f"{annual_rate}%/năm, khoản trả hàng tháng xấp xỉ {fmt_money(monthly)}."
            ),
        ))

    # 6. Complex Multi-Intent (50)
    for i, project in enumerate(random.sample(valid_projects, 50)):
        loc = extract_location_parts(project.get("dia_chi"), project.get("khu_vuc"))
        lifestyle_surface, lifestyle_signal = random.choice(LIFESTYLE_SURFACES)
        poi_surface, poi_signal, _ = choose_poi_for_location(loc)
        name = clean_text(project.get("ten_du_an"))
        question = (
            f"Tôi đang cân nhắc dự án {name} ở {loc['district']}. "
            f"Nơi này có {lifestyle_surface} không, và trong 2km có gần {poi_surface} nào không?"
        )
        dataset.append(case(
            case_id=f"complex_{i:03d}",
            category="Complex Multi-Intent",
            difficulty="hard",
            eval_type=["parser", "tool_routing", "retrieval"],
            question=question,
            expected_intents=["lifestyle_search"],
            expected_filters={"quan_huyen": loc["district"]},
            expected_signals=sorted({lifestyle_signal, poi_signal}),
            expected_entities={"project_name": name, "radius_m": 2000, "poi": poi_surface},
            expected_tools_called=["hybrid_search", "find_pois_near_location"],
            expected_sources=[source_ref(project, "projects")],
            reference_answer=(
                f"Cần kết hợp hybrid_search cho tiêu chí {lifestyle_surface} và "
                f"find_pois_near_location cho {poi_surface} quanh {name}."
            ),
        ))

    # 7. Out-of-domain / Clarification (50)
    clarification_questions = [
        "Mình có 3 tỷ muốn mua nhà.",
        "Tôi cần thuê mặt bằng.",
        "Tôi muốn tìm nhà gần trường học.",
        "Gia đình tôi muốn chuyển chỗ ở, tư vấn giúp tôi.",
        "Tôi có ngân sách khoảng 5 tỷ, nên bắt đầu tìm nhà thế nào?",
        "Tôi muốn mua căn hộ nhưng chưa biết khu nào phù hợp.",
        "Tôi muốn thuê nhà cho gia đình nhỏ, bạn cần thêm thông tin gì?",
        "Tôi muốn tìm nơi ở yên tĩnh nhưng chưa chốt khu vực.",
        "Tôi muốn mua bất động sản để ở lâu dài.",
        "Tôi cần tìm chỗ ở gần trường cho con.",
    ]
    refusal_questions = [
        "Kinh tế dạo này lạm phát cao quá, có nên mua vàng không?",
        "Thời tiết Sài Gòn cuối tuần này thế nào?",
        "Cách làm món thịt kho tàu ngon?",
        "Tư vấn giúp tôi mua cổ phiếu ngân hàng tuần này.",
        "Lịch thi đấu bóng đá tối nay ra sao?",
        "Viết giúp tôi một bài thơ tình.",
        "Nên mua điện thoại nào để chụp ảnh đẹp?",
        "Cách trị ho tại nhà thế nào?",
        "Tôi muốn học Python bắt đầu từ đâu?",
        "Du lịch Đà Lạt mùa này có đẹp không?",
    ]
    for i in range(25):
        base_question = clarification_questions[i % len(clarification_questions)]
        question = base_question if i < len(clarification_questions) else f"{base_question} Nhu cầu của tôi vẫn còn khá chung."
        dataset.append(case(
            case_id=f"clarify_{i:03d}",
            category="Out-of-Domain / Clarification",
            difficulty="medium",
            eval_type=["conversation_policy"],
            question=question,
            expected_behavior="clarify",
            expected_intents=["search_listing"],
            expected_tools_called=[],
            expected_entities={"required_clarification_slots": ["city_or_area", "buy_or_rent", "budget_or_price"]},
            reference_answer="Cần hỏi làm rõ ngắn gọn về khu vực, mua/thuê và ngân sách trước khi tìm kiếm.",
        ))
    for i in range(25):
        base_question = refusal_questions[i % len(refusal_questions)]
        question = base_question if i < len(refusal_questions) else f"{base_question} Tôi hỏi ngoài phạm vi bất động sản."
        dataset.append(case(
            case_id=f"refuse_{i:03d}",
            category="Out-of-Domain / Clarification",
            difficulty="easy",
            eval_type=["conversation_policy"],
            question=question,
            expected_behavior="refuse_out_of_scope",
            expected_intents=["ask_knowledge"],
            expected_tools_called=[],
            reference_answer="Cần từ chối ngắn gọn hoặc chuyển hướng vì câu hỏi nằm ngoài phạm vi trợ lý bất động sản.",
        ))

    random.shuffle(dataset)
    final_dataset = dataset[:1000]
    validate_dataset(final_dataset)

    OUTPUT_FILE.write_text(json.dumps(final_dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(final_dataset)} validated test cases at {OUTPUT_FILE}")
    print("Category distribution:", dict(Counter(r["category"] for r in final_dataset)))
    print("Eval type distribution:", dict(Counter(t for r in final_dataset for t in r["eval_type"])))


if __name__ == "__main__":
    main()
