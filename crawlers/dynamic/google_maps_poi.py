import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from crawlers.config import DATA_DIR, GOOGLE_MAPS_API_KEY

log = logging.getLogger("bds_crawler.poi")

DEFAULT_POI_TYPES = [
    "school",
    "hospital",
    "transit_station",
    "park",
    "shopping_mall",
    "supermarket"
]

class GoogleMapsPOI:
    """Dynamic context service fetching nearby Points of Interest (POIs) with caching support."""
    
    def __init__(self, cache_file: Optional[Path] = None, cache_ttl_days: int = 30):
        self.api_key = GOOGLE_MAPS_API_KEY
        self.base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        self.cache_file = cache_file or (DATA_DIR / ".poi_cache.json")
        self.cache_ttl_seconds = cache_ttl_days * 24 * 60 * 60
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load POI query cache from JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to read POI cache from {self.cache_file}: {e}")
        return {}

    def _save_cache(self) -> None:
        """Write POI query cache to JSON file atomically."""
        temp_file = self.cache_file.with_suffix(".tmp")
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            temp_file.replace(self.cache_file)
        except Exception as e:
            log.error(f"Failed to save POI cache: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _make_cache_key(self, lat: float, lng: float, radius: int, poi_type: str) -> str:
        """Generate deterministic cache key based on query params (rounded coordinate precision)."""
        # Round coords to 4 decimal places (~11 meters precision) to reuse cache effectively
        lat_r = round(lat, 4)
        lng_r = round(lng, 4)
        raw_key = f"{lat_r},{lng_r}_{radius}_{poi_type}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cached entry has expired based on TTL settings."""
        timestamp = cache_entry.get("timestamp", 0)
        return (time.time() - timestamp) < self.cache_ttl_seconds

    def _call_nearby_search(self, lat: float, lng: float, radius: int, poi_type: str) -> List[Dict[str, Any]]:
        """Call standard Google Places Nearby Search API."""
        if not self.api_key:
            log.error("GOOGLE_MAPS_API_KEY environment variable is not set. Cannot run Nearby Search.")
            return []

        pois = []
        next_page_token = None
        
        # Loop to retrieve paginated results (up to 60 POIs per search)
        while True:
            params = {
                "key": self.api_key,
                "location": f"{lat},{lng}",
                "radius": radius,
                "type": poi_type,
                "language": "vi"
            }
            if next_page_token:
                params["pagetoken"] = next_page_token

            try:
                r = requests.get(self.base_url, params=params, timeout=15)
                if r.status_code != 200:
                    log.warning(f"Places Nearby Search failed with status {r.status_code}")
                    break

                data = r.json()
                status = data.get("status")
                
                if status not in ("OK", "ZERO_RESULTS"):
                    log.warning(f"Places Nearby Search returned non-OK status: '{status}'")
                    break

                results = data.get("results", [])
                for res in results:
                    loc = res.get("geometry", {}).get("location", {})
                    pois.append({
                        "name": res.get("name"),
                        "place_id": res.get("place_id"),
                        "formatted_address": res.get("vicinity"),
                        "rating": res.get("rating"),
                        "coordinates": {
                            "lat": loc.get("lat"),
                            "lng": loc.get("lng")
                        },
                        "types": res.get("types", [])
                    })

                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break
                
                # Nearby search requires a short delay before next_page_token becomes valid
                time.sleep(2.0)
            except Exception as e:
                log.error(f"Error executing Nearby Search for {lat},{lng} (type: {poi_type}): {e}")
                break

        return pois

    def search_nearby(
        self,
        lat: float,
        lng: float,
        radius: int = 1000,
        types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve nearby Points of Interest, grouped by category type, with cache lookups."""
        search_types = types or DEFAULT_POI_TYPES
        results_by_type = {}
        cache_updated = False

        for t in search_types:
            cache_key = self._make_cache_key(lat, lng, radius, t)
            
            if use_cache and cache_key in self.cache:
                entry = self.cache[cache_key]
                if self._is_cache_valid(entry):
                    log.info(f"Retrieving cached POIs for type '{t}' at ({lat:.4f}, {lng:.4f})")
                    results_by_type[t] = entry.get("data", [])
                    continue
            
            # Cache miss or invalid - call API
            log.info(f"API Search POIs for type '{t}' at ({lat:.4f}, {lng:.4f}) within {radius}m")
            api_pois = self._call_nearby_search(lat, lng, radius, t)
            results_by_type[t] = api_pois
            
            # Store in cache
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "data": api_pois
            }
            cache_updated = True

        if cache_updated:
            self._save_cache()

        return results_by_type
