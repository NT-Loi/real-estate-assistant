import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from crawlers.config import DATA_DIR

log = logging.getLogger("bds_crawler.osm_poi")

DEFAULT_POI_TYPES = [
    "school",
    "hospital",
    "transit_station",
    "park",
    "shopping_mall",
    "supermarket",
]

OSM_CATEGORY_TAGS = {
    "school": [("amenity", "school"), ("amenity", "kindergarten"), ("amenity", "university")],
    "hospital": [("amenity", "hospital"), ("amenity", "clinic"), ("amenity", "doctors")],
    "transit_station": [
        ("public_transport", "station"),
        ("public_transport", "stop_position"),
        ("railway", "station"),
        ("amenity", "bus_station"),
    ],
    "park": [("leisure", "park"), ("leisure", "garden")],
    "shopping_mall": [("shop", "mall"), ("building", "retail")],
    "supermarket": [("shop", "supermarket"), ("shop", "convenience")],
}


class OSMPOI:
    """Key-free OpenStreetMap geocoding and nearby POI retrieval with local caching."""

    def __init__(
        self,
        cache_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
        user_agent: str = "real-estate-assistant/1.0 (local crawler; contact: local)",
        nominatim_url: str = "https://nominatim.openstreetmap.org/search",
        overpass_url: str = "https://overpass-api.de/api/interpreter",
        nominatim_delay: float = 1.0,
    ):
        self.cache_file = cache_file or (DATA_DIR / ".osm_cache.json")
        self.output_file = output_file or (DATA_DIR / "pois.json")
        self.nearby_output_file = DATA_DIR / "nearby_amenities.json"
        self.nominatim_url = nominatim_url
        self.overpass_url = overpass_url
        self.nominatim_delay = nominatim_delay
        self._last_nominatim_call = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("geocode", {})
                    data.setdefault("poi", {})
                    return data
            except Exception as e:
                log.warning(f"Failed to read OSM cache from {self.cache_file}: {e}")
        return {"geocode": {}, "poi": {}}

    def _save_cache(self) -> None:
        temp_file = self.cache_file.with_suffix(".tmp")
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            temp_file.replace(self.cache_file)
        except Exception as e:
            log.error(f"Failed to save OSM cache: {e}")
            if temp_file.exists():
                temp_file.unlink()

    @staticmethod
    def _cache_key(raw: str) -> str:
        return hashlib.md5(raw.strip().lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _poi_cache_key(lat: float, lng: float, radius: int, category: str) -> str:
        raw = f"{round(lat, 4)},{round(lng, 4)}:{radius}:{category}"
        return OSMPOI._cache_key(raw)

    @staticmethod
    def distance_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Approximate great-circle distance between two WGS84 coordinates."""
        earth_radius_m = 6371000.0
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        d_phi = math.radians(float(lat2) - float(lat1))
        d_lambda = math.radians(float(lng2) - float(lng1))
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return 2 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def normalize_geocode_result(query: str, result: dict) -> Optional[dict]:
        try:
            lat = float(result["lat"])
            lng = float(result["lon"])
        except Exception:
            return None

        importance = result.get("importance")
        try:
            confidence = float(importance) if importance is not None else 0.5
        except Exception:
            confidence = 0.5

        return {
            "query": query,
            "latitude": lat,
            "longitude": lng,
            "display_name": result.get("display_name") or "",
            "geo_source": "osm_nominatim",
            "geo_confidence": max(0.0, min(confidence, 1.0)),
            "raw": result,
        }

    def geocode_address(self, address: str, countrycodes: str = "vn", use_cache: bool = True) -> Optional[dict]:
        query = " ".join(str(address or "").split())
        if not query:
            return None

        cache_key = self._cache_key(query)
        if use_cache and cache_key in self.cache["geocode"]:
            cached = self.cache["geocode"][cache_key]
            return cached if cached else None

        elapsed = time.time() - self._last_nominatim_call
        if elapsed < self.nominatim_delay:
            time.sleep(self.nominatim_delay - elapsed)

        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": countrycodes,
        }

        try:
            r = self.session.get(self.nominatim_url, params=params, timeout=20)
            self._last_nominatim_call = time.time()
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning(f"OSM geocode failed for '{query}': {e}")
            return None

        normalized = self.normalize_geocode_result(query, data[0]) if data else None
        self.cache["geocode"][cache_key] = normalized
        self._save_cache()
        return normalized

    @staticmethod
    def address_candidates(*values: object) -> List[str]:
        """Build increasingly broad Nominatim queries from noisy crawled Vietnamese addresses."""
        candidates = []
        seen = set()

        def add(raw: object) -> None:
            text = " ".join(str(raw or "").split())
            text = re.sub(r"\.?\s*Xem bản đồ\s*$", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"\([^)]*mới[^)]*\)", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"\s*,\s*", ", ", text)
            text = re.sub(r"\s+\.", ".", text)
            text = text.strip(" ,.")
            if len(text) <= 3:
                return
            key = text.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(text)

        for value in values:
            add(value)
            cleaned = candidates[-1] if candidates else ""
            if not cleaned:
                continue
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            for start in range(1, max(1, len(parts) - 1)):
                add(", ".join(parts[start:]))
            if len(parts) >= 2:
                add(", ".join(parts[-2:]))
            if parts:
                add(parts[-1])

        return candidates

    def geocode_record(self, record: dict) -> Optional[dict]:
        candidates = self.address_candidates(
            record.get("dia_chi"),
            record.get("khu_vuc"),
            record.get("ten_du_an"),
        )
        for idx, candidate in enumerate(candidates):
            geocoded = self.geocode_address(candidate)
            if not geocoded:
                continue
            if idx > 0:
                geocoded = {
                    **geocoded,
                    "geo_source": "osm_nominatim_fallback",
                    "geo_confidence": min(float(geocoded.get("geo_confidence") or 0.5), 0.55),
                    "fallback_query": candidate,
                }
            return geocoded
        return None

    @staticmethod
    def build_overpass_query(lat: float, lng: float, radius: int, category: str) -> str:
        tags = OSM_CATEGORY_TAGS.get(category)
        if not tags:
            raise ValueError(f"Unsupported OSM POI category: {category}")

        selectors = []
        for key, value in tags:
            selectors.append(f'node["{key}"="{value}"](around:{radius},{lat},{lng});')
            selectors.append(f'way["{key}"="{value}"](around:{radius},{lat},{lng});')
            selectors.append(f'relation["{key}"="{value}"](around:{radius},{lat},{lng});')

        return "[out:json][timeout:25];(" + "".join(selectors) + ");out center tags;"

    def _call_overpass(self, query: str, retries: int = 3, base_sleep: float = 2.0) -> dict:
        for attempt in range(retries):
            try:
                r = self.session.post(self.overpass_url, data={"data": query}, timeout=45)
                if r.status_code in (429, 502, 503, 504):
                    raise RuntimeError(f"Overpass temporary status {r.status_code}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == retries - 1:
                    log.warning(f"Overpass request failed: {e}")
                    return {}
                time.sleep(base_sleep * (attempt + 1))
        return {}

    @staticmethod
    def normalize_poi_element(element: dict, category: str) -> Optional[dict]:
        tags = element.get("tags") or {}
        name = tags.get("name") or tags.get("name:vi") or tags.get("name:en")
        lat = element.get("lat") or (element.get("center") or {}).get("lat")
        lng = element.get("lon") or (element.get("center") or {}).get("lon")
        if not name or lat is None or lng is None:
            return None

        osm_type = element.get("type")
        osm_id = element.get("id")
        place_id = f"osm:{osm_type}:{osm_id}"
        address_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:ward"),
            tags.get("addr:district"),
            tags.get("addr:city"),
        ]
        address = ", ".join(str(p) for p in address_parts if p)

        return {
            "id": hashlib.md5(place_id.encode("utf-8")).hexdigest(),
            "place_id": place_id,
            "osm_type": osm_type,
            "osm_id": osm_id,
            "name": name,
            "category": category,
            "address": address,
            "latitude": float(lat),
            "longitude": float(lng),
            "rating": None,
            "review_count": None,
            "source": "osm_overpass",
            "raw_json": element,
        }

    def search_nearby(
        self,
        lat: float,
        lng: float,
        radius: int = 2000,
        types: Optional[List[str]] = None,
        use_cache: bool = True,
        save_output: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        radius = max(1, min(int(radius), 5000))
        search_types = types or DEFAULT_POI_TYPES
        results_by_type: Dict[str, List[Dict[str, Any]]] = {}
        cache_updated = False

        for category in search_types:
            cache_key = self._poi_cache_key(lat, lng, radius, category)
            if use_cache and cache_key in self.cache["poi"]:
                results_by_type[category] = self.cache["poi"][cache_key]
                continue

            log.info(f"OSM Search POIs for type '{category}' at ({lat:.4f}, {lng:.4f}) within {radius}m")
            query = self.build_overpass_query(lat, lng, radius, category)
            data = self._call_overpass(query)
            seen_place_ids = set()
            pois = []
            for element in data.get("elements", []):
                poi = self.normalize_poi_element(element, category)
                if not poi or poi["place_id"] in seen_place_ids:
                    continue
                seen_place_ids.add(poi["place_id"])
                pois.append(poi)

            self.cache["poi"][cache_key] = pois
            results_by_type[category] = pois
            cache_updated = True

        if cache_updated:
            self._save_cache()
        if save_output:
            self.save_pois(results_by_type)
        return results_by_type

    def amenities_for_target(
        self,
        target: Dict[str, Any],
        results_by_type: Dict[str, List[Dict[str, Any]]],
        radius: int,
    ) -> Dict[str, Any]:
        """Build a per-listing/project amenity payload from grouped POI results."""
        target_lat = float(target["lat"])
        target_lng = float(target["lng"])
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for category, poi_list in results_by_type.items():
            amenities = []
            for poi in poi_list:
                poi_lat = poi.get("latitude")
                poi_lng = poi.get("longitude")
                if poi_lat is None or poi_lng is None:
                    continue
                amenities.append(
                    {
                        "place_id": poi.get("place_id"),
                        "osm_type": poi.get("osm_type"),
                        "osm_id": poi.get("osm_id"),
                        "name": poi.get("name"),
                        "category": poi.get("category") or category,
                        "address": poi.get("address"),
                        "latitude": float(poi_lat),
                        "longitude": float(poi_lng),
                        "distance_m": round(self.distance_meters(target_lat, target_lng, float(poi_lat), float(poi_lng)), 1),
                        "source": poi.get("source"),
                    }
                )
            amenities.sort(key=lambda item: item["distance_m"])
            grouped[category] = amenities

        return {
            "source_type": target.get("source_type"),
            "source_file": target.get("source_file"),
            "source_index": target.get("source_index"),
            "source_id": target.get("source_id"),
            "label": target.get("label"),
            "dia_chi": target.get("dia_chi"),
            "target_latitude": target_lat,
            "target_longitude": target_lng,
            "radius_m": int(radius),
            "amenities": grouped,
        }

    def save_nearby_amenities(self, payloads: List[Dict[str, Any]]) -> None:
        self.nearby_output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.nearby_output_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payloads, f, ensure_ascii=False, indent=2)
        temp_file.replace(self.nearby_output_file)
        log.info(f"Saved nearby amenities for {len(payloads)} source records to {self.nearby_output_file}")

    def save_pois(self, results_by_type: Dict[str, List[Dict[str, Any]]]) -> None:
        existing = []
        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    existing = data
            except Exception as e:
                log.warning(f"Failed to read existing POI output {self.output_file}: {e}")

        by_place_id = {p.get("place_id"): p for p in existing if p.get("place_id")}
        for poi_list in results_by_type.values():
            for poi in poi_list:
                by_place_id[poi["place_id"]] = poi

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.output_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(list(by_place_id.values()), f, ensure_ascii=False, indent=2)
        temp_file.replace(self.output_file)
        log.info(f"Saved {len(by_place_id)} unique OSM POIs to {self.output_file}")

    def enrich_data_files(self, max_records_per_file: Optional[int] = None) -> dict:
        targets = [
            DATA_DIR / "listings_ban.json",
            DATA_DIR / "listings_cho_thue.json",
            DATA_DIR / "projects.json",
        ]
        summary = {}

        for path in targets:
            if not path.exists():
                summary[path.name] = {"updated": 0, "missing": True}
                continue

            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                summary[path.name] = {"updated": 0, "missing": False, "error": "not a list"}
                continue

            updated = 0
            scanned = 0
            for record in records:
                if max_records_per_file is not None and scanned >= max_records_per_file:
                    break
                scanned += 1

                if record.get("latitude") and record.get("longitude"):
                    continue
                geocoded = self.geocode_record(record)
                if not geocoded:
                    continue
                record["latitude"] = geocoded["latitude"]
                record["longitude"] = geocoded["longitude"]
                record["geo_source"] = geocoded["geo_source"]
                record["geo_confidence"] = geocoded["geo_confidence"]
                updated += 1

            if updated:
                temp_file = path.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                temp_file.replace(path)

            summary[path.name] = {"updated": updated, "missing": False}
            log.info(f"Geocoded {updated} records in {path.name}")

        return summary
