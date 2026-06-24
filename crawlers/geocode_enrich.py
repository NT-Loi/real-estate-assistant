from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


log = logging.getLogger("bds_crawler.geocode")

TARGET_FILES = ("listings_ban.json", "listings_cho_thue.json", "projects.json")

REGION_COORDS = {
    "Hà Nội": (21.0278, 105.8342),
    "Hồ Chí Minh": (10.7769, 106.7009),
    "Đà Nẵng": (16.0471, 108.2068),
    "Bình Chánh": (10.6874, 106.5938),
    "Quốc Oai": (20.9912, 105.6409),
    "Sơn Trà": (16.1067, 108.2521),
    "Nam Từ Liêm": (21.0035, 105.7703),
    "Thủ Đức": (10.8494, 106.7537),
    "Bắc Ninh": (21.1861, 106.0763),
    "Việt Yên": (21.2731, 106.1018),
    "Long An": (10.6956, 106.2431),
    "Cần Đước": (10.5404, 106.5968),
}


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def _write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Failed to read geocode cache %s: %s", path, exc)
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def _clean_address(value: object) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace(". Xem bản đồ", "")
    text = text.replace("Xem bản đồ", "")
    # Batdongsan sometimes appends administrative-renaming hints in parentheses.
    # Keep the original address in the record, but query the canonical first part.
    if "(" in text and ")" in text:
        text = text.split("(", 1)[0].strip(" ,.")
    return text.strip(" ,.")


def _address_for_record(record: dict[str, Any]) -> str:
    address = _clean_address(record.get("dia_chi") or record.get("khu_vuc"))
    if not address and record.get("ten_du_an"):
        address = _clean_address(record.get("ten_du_an"))
    if address and "việt nam" not in address.lower() and "viet nam" not in address.lower():
        address = f"{address}, Việt Nam"
    return address


def _query_candidates_for_record(record: dict[str, Any]) -> list[str]:
    """Return progressively broader geocode queries for one record."""
    raw = _clean_address(record.get("dia_chi") or record.get("khu_vuc"))
    candidates: list[str] = []

    def add(value: object) -> None:
        text = _clean_address(value)
        if not text:
            return
        if "việt nam" not in text.lower() and "viet nam" not in text.lower():
            text = f"{text}, Việt Nam"
        if text not in candidates:
            candidates.append(text)

    add(raw)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for width in (4, 3, 2, 1):
        if len(parts) >= width:
            add(", ".join(parts[-width:]))

    project = record.get("ten_du_an") or record.get("du_an")
    if project and raw:
        add(f"{project}, {raw}")
    elif project:
        add(project)

    return candidates


def _has_coordinates(record: dict[str, Any]) -> bool:
    return record.get("latitude") not in (None, "") and record.get("longitude") not in (None, "")


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _stable_offset(seed: str, scale: float = 0.01) -> tuple[float, float]:
    value = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed or ""))
    # Keep this simple and deterministic; no math dependency needed.
    lat_units = ((value % 200) - 100) / 100
    lon_units = (((value // 200) % 200) - 100) / 100
    return lat_units * scale, lon_units * scale


def _apply_region_fallback(record: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(record.get(key) or "")
        for key in ("dia_chi", "khu_vuc", "tieu_de", "ten_du_an", "url")
    )
    for name, (lat, lon) in REGION_COORDS.items():
        if name.lower() in haystack.lower():
            dlat, dlon = _stable_offset(str(record.get("url") or record.get("tieu_de") or ""))
            record["latitude"] = lat + dlat
            record["longitude"] = lon + dlon
            record["geo_source"] = "local_region_fallback"
            record["geo_confidence"] = "region_approximate"
            record["geo_query"] = name
            return True
    return False


def _copy_geo_fields(target: dict[str, Any], source: dict[str, Any]) -> bool:
    lat = _as_float(source.get("latitude") or source.get("lat"))
    lon = _as_float(source.get("longitude") or source.get("lng") or source.get("lon"))
    if lat is None or lon is None:
        return False
    target["latitude"] = lat
    target["longitude"] = lon
    target["geo_source"] = source.get("geo_source") or "seed_data"
    target["geo_confidence"] = source.get("geo_confidence") or "seeded"
    if source.get("nearby_amenities"):
        target["nearby_amenities"] = source["nearby_amenities"]
        for key in (
            "nearby_amenities_radius_m",
            "nearby_amenities_source",
            "nearby_amenities_target_latitude",
            "nearby_amenities_target_longitude",
        ):
            if key in source:
                target[key] = source[key]
    return True


def _seed_index(seed_dir: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_url: dict[str, dict[str, Any]] = {}
    by_addr: dict[str, dict[str, Any]] = {}
    if not seed_dir:
        return by_url, by_addr

    for filename in TARGET_FILES:
        for row in _load_json_array(seed_dir / filename):
            if not _has_coordinates(row):
                continue
            url = row.get("url")
            if url:
                by_url[str(url)] = row
            addr = _address_for_record(row)
            if addr:
                by_addr.setdefault(addr.lower(), row)
    return by_url, by_addr


def _geocode_nominatim(query: str, user_agent: str, timeout: int = 20) -> dict[str, Any] | None:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "vn",
        }
    )
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload:
        return None
    first = payload[0]
    lat = _as_float(first.get("lat"))
    lon = _as_float(first.get("lon"))
    if lat is None or lon is None:
        return None
    return {
        "latitude": lat,
        "longitude": lon,
        "geo_source": "osm_nominatim",
        "geo_confidence": "address_geocode",
        "geo_display_name": first.get("display_name"),
        "geo_raw": first,
    }


def enrich_data_dir(
    data_dir: Path,
    seed_dir: Path | None = None,
    max_requests: int | None = None,
    rate_limit_seconds: float = 1.1,
    user_agent: str = "RealEstateAssistant/1.0 (geocoding enrichment)",
    cache_name: str = ".geocode_cache.json",
    save_every_changes: int = 25,
) -> dict[str, int]:
    data_dir = data_dir.resolve()
    seed_dir = seed_dir.resolve() if seed_dir else None
    cache_path = data_dir / cache_name
    cache = _load_cache(cache_path)
    seed_by_url, seed_by_addr = _seed_index(seed_dir)

    stats = {
        "rows": 0,
        "already_had_coordinates": 0,
        "seeded": 0,
        "cache_hits": 0,
        "geocoded": 0,
        "geocode_misses": 0,
        "errors": 0,
        "remaining_missing": 0,
        "requests": 0,
        "paused_budget_exhausted": 0,
    }

    request_budget = max_requests if max_requests is not None else 10**12
    budget_exhausted = False

    for filename in TARGET_FILES:
        if budget_exhausted:
            break
        path = data_dir / filename
        rows = _load_json_array(path)
        if not rows:
            continue

        changed = False
        changes_since_save = 0

        def mark_changed() -> None:
            nonlocal changed, changes_since_save
            changed = True
            changes_since_save += 1

        def save_partial(force: bool = False) -> None:
            nonlocal changed, changes_since_save
            if changed and (force or changes_since_save >= save_every_changes):
                _write_json_array(path, rows)
                _save_cache(cache_path, cache)
                log.info("Updated %s", path)
                changed = False
                changes_since_save = 0

        for record in rows:
            stats["rows"] += 1
            if _has_coordinates(record):
                stats["already_had_coordinates"] += 1
                continue

            seeded = None
            url = record.get("url")
            if url:
                seeded = seed_by_url.get(str(url))
            candidates = _query_candidates_for_record(record)
            query = candidates[0] if candidates else ""
            if not seeded and candidates:
                for candidate in candidates:
                    seeded = seed_by_addr.get(candidate.lower())
                    if seeded:
                        break
            if seeded and _copy_geo_fields(record, seeded):
                stats["seeded"] += 1
                mark_changed()
                save_partial()
                continue

            if not candidates:
                if _apply_region_fallback(record):
                    mark_changed()
                    save_partial()
                    stats["geocoded"] += 1
                    continue
                stats["remaining_missing"] += 1
                continue

            used_cache = False
            cache_exhausted = True
            for candidate in candidates:
                if candidate not in cache:
                    cache_exhausted = False
                    continue
                cached = cache[candidate]
                if cached and _copy_geo_fields(record, cached):
                    stats["cache_hits"] += 1
                    mark_changed()
                    used_cache = True
                    break
            if used_cache:
                save_partial()
                continue
            if cache_exhausted:
                if _apply_region_fallback(record):
                    mark_changed()
                    save_partial()
                    stats["geocoded"] += 1
                    continue
                stats["geocode_misses"] += 1
                stats["remaining_missing"] += 1
                continue

            found = False
            for candidate in candidates:
                if candidate in cache:
                    continue
                if stats["requests"] >= request_budget:
                    budget_exhausted = True
                    stats["paused_budget_exhausted"] = 1
                    break
                try:
                    result = _geocode_nominatim(candidate, user_agent=user_agent)
                    stats["requests"] += 1
                    cache[candidate] = result
                    if result and _copy_geo_fields(record, result):
                        if candidate != query:
                            record["geo_confidence"] = "fallback_geocode"
                            record["geo_query"] = candidate
                        stats["geocoded"] += 1
                        mark_changed()
                        found = True
                        break
                    stats["geocode_misses"] += 1
                    time.sleep(rate_limit_seconds)
                except Exception as exc:
                    cache[candidate] = None
                    stats["requests"] += 1
                    stats["errors"] += 1
                    log.warning("Geocode failed for %r: %s", candidate, exc)
                    time.sleep(rate_limit_seconds)
                if stats["requests"] % 20 == 0:
                    _save_cache(cache_path, cache)
                    log.info("Geocoded %s requests so far", stats["requests"])
            if not found:
                if budget_exhausted:
                    break
                if _apply_region_fallback(record):
                    mark_changed()
                    save_partial()
                    stats["geocoded"] += 1
                    continue
                stats["remaining_missing"] += 1
            else:
                save_partial()

        save_partial(force=True)

    _save_cache(cache_path, cache)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geocode crawled real-estate records with OSM Nominatim")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing crawled JSON files")
    parser.add_argument("--seed-dir", type=Path, default=None, help="Optional data directory to copy existing lat/lng from")
    parser.add_argument("--max-requests", type=int, default=None, help="Maximum Nominatim requests for this run")
    parser.add_argument("--rate-limit-seconds", type=float, default=1.1, help="Delay between Nominatim requests")
    parser.add_argument("--user-agent", default="RealEstateAssistant/1.0 (geocoding enrichment)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    stats = enrich_data_dir(
        data_dir=args.data_dir,
        seed_dir=args.seed_dir,
        max_requests=args.max_requests,
        rate_limit_seconds=args.rate_limit_seconds,
        user_agent=args.user_agent,
    )
    log.info("Geocode enrichment complete: %s", stats)


if __name__ == "__main__":
    main()
