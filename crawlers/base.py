import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from crawlers.checkpoint import CheckpointManager
from crawlers.config import DATA_DIR, REQUEST_DELAY

log = logging.getLogger("bds_crawler.base")

class BaseCrawler:
    """Base abstract class for all crawlers, providing standardized logging, checkpointing, and output saving."""
    
    def __init__(self, name: str, output_file: Optional[Path] = None):
        self.name = name
        self.output_file = output_file or (DATA_DIR / f"{name}.json")
        self.log = logging.getLogger(f"bds_crawler.{name}")
        self.checkpoint_mgr = CheckpointManager(name, DATA_DIR / ".checkpoints")

    async def sleep_polite(self, delay: float = REQUEST_DELAY) -> None:
        """Inject request delays to respect the remote host's load limitations."""
        await asyncio.sleep(delay)

    def load_existing_data(self) -> List[Dict[str, Any]]:
        """Load already finalized JSON records if output file exists, preventing duplicate records."""
        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                self.log.warning(f"Failed to read existing data file {self.output_file}: {e}")
        return []

    def save_final_results(self, items: List[Dict[str, Any]], resume: bool = False) -> None:
        """Write records to file atomically, deduplicating elements by URL when possible."""
        existing_items = []
        if resume:
            existing_items = self.load_existing_data()

        # Combine items, preferring newer ones on overlapping keys
        seen_urls = set()
        combined: List[Dict[str, Any]] = []

        # Process existing items first
        for item in existing_items:
            url = item.get("url")
            if url:
                seen_urls.add(url)
            combined.append(item)

        # Append new unique items
        new_count = 0
        for item in items:
            url = item.get("url")
            # If no URL (like some POIs/general news), use unique identifier or fallback to title/coords
            if url:
                if url not in seen_urls:
                    seen_urls.add(url)
                    combined.append(item)
                    new_count += 1
            else:
                combined.append(item)
                new_count += 1

        # Write to temporary file first, then replace
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.output_file.with_suffix(".tmp")
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            temp_file.replace(self.output_file)
            self.log.info(f"Successfully saved {len(combined)} items ({new_count} new) to {self.output_file}")
            
            # Clear checkpoints upon completion of normal flow
            self.checkpoint_mgr.clear()
        except Exception as e:
            self.log.error(f"Failed to save final results to {self.output_file}: {e}")
            if temp_file.exists():
                temp_file.unlink()

    async def crawl(self, **kwargs) -> List[Dict[str, Any]]:
        """Method to be overridden by subclass crawlers."""
        raise NotImplementedError("Each crawler must implement the 'crawl' method.")
