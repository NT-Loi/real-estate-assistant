import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from crawlers.base import BaseCrawler
from crawlers.config import DATA_DIR, GOOGLE_MAPS_API_KEY, REQUEST_DELAY

log = logging.getLogger("bds_crawler.google_reviews")

class GoogleReviewsCrawler(BaseCrawler):
    """Google Maps Place Reviews Crawler using the official Google Places Web Service API."""
    
    def __init__(self, output_file: Optional[Path] = None):
        super().__init__("google_reviews", output_file or (DATA_DIR / "google_reviews.json"))
        self.api_key = GOOGLE_MAPS_API_KEY
        self.base_url = "https://maps.googleapis.com/maps/api/place"

    def _call_api(self, service: str, params: dict) -> dict:
        """Call Google Places Web Services endpoints with credentials."""
        url = f"{self.base_url}/{service}/json"
        params["key"] = self.api_key
        
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            else:
                self.log.warning(f"Google Places API returned status code {r.status_code}")
        except Exception as e:
            self.log.error(f"Error calling Google Places API {service}: {e}")
        return {}

    def resolve_place_id(self, query: str) -> Optional[str]:
        """Search for a place query using Text Search API and return its unique Place ID."""
        self.log.info(f"Resolving Place ID for: '{query}'")
        params = {
            "query": query,
            "language": "vi"
        }
        data = self._call_api("textsearch", params)
        results = data.get("results", [])
        if results:
            place_id = results[0].get("place_id")
            name = results[0].get("name")
            self.log.info(f"  Resolved to: '{name}' (Place ID: {place_id})")
            return place_id
        
        self.log.warning(f"  No place resolved for query: '{query}'")
        return None

    def fetch_place_details(self, place_id: str) -> Dict[str, Any]:
        """Fetch detailed information, coordinates, and reviews for a place ID."""
        self.log.info(f"Fetching Place Details for ID: {place_id}")
        params = {
            "place_id": place_id,
            "fields": "name,formatted_address,geometry,rating,user_ratings_total,reviews",
            "language": "vi"
        }
        data = self._call_api("details", params)
        return data.get("result", {})

    async def crawl(
        self,
        queries: List[str],
        resume: bool = False
    ) -> List[Dict[str, Any]]:
        """Run Place resolution and Reviews retrieval flow for neighborhood target queries."""
        if not self.api_key:
            self.log.error("GOOGLE_MAPS_API_KEY environment variable is not set. Cannot run GoogleReviewsCrawler.")
            return []

        self.log.info(f"Starting Google Reviews Crawl. Target queries: {queries}")
        all_results = []

        if resume:
            self.checkpoint_mgr.load()
            all_results = self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resumed reviews crawl. Loaded {len(all_results)} places from checkpoints.")

        processed_queries = {item.get("query") for item in all_results if item.get("query")}

        for idx, q in enumerate(queries):
            if q in processed_queries:
                continue

            self.log.info(f"Processing query {idx+1}/{len(queries)}: '{q}'")
            place_id = self.resolve_place_id(q)
            
            if not place_id:
                # Store placeholder record to prevent repeatedly searching missing queries
                record = {
                    "query": q,
                    "resolved": False,
                    "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                all_results.append(record)
                processed_queries.add(q)
                self.checkpoint_mgr.save(idx, [record])
                continue

            details = self.fetch_place_details(place_id)
            if not details:
                continue

            # Format and save details
            location = details.get("geometry", {}).get("location", {})
            reviews_raw = details.get("reviews", [])
            reviews_formatted = []
            
            for rev in reviews_raw:
                reviews_formatted.append({
                    "author": rev.get("author_name"),
                    "rating": rev.get("rating"),
                    "text": rev.get("text"),
                    "relative_time": rev.get("relative_time_description"),
                    "timestamp": rev.get("time")  # unix epoch
                })

            record = {
                "query": q,
                "resolved": True,
                "place_id": place_id,
                "name": details.get("name"),
                "formatted_address": details.get("formatted_address"),
                "coordinates": {
                    "lat": location.get("lat"),
                    "lng": location.get("lng")
                },
                "rating": details.get("rating"),
                "user_ratings_total": details.get("user_ratings_total"),
                "reviews": reviews_formatted,
                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            all_results.append(record)
            processed_queries.add(q)
            
            # Save checkpoint
            self.checkpoint_mgr.save(idx, [record])
            
            # Sleep politely between API queries
            await self.sleep_polite(REQUEST_DELAY)

        self.save_final_results(all_results, resume)
        return all_results
