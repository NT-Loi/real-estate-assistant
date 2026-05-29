import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

log = logging.getLogger("bds_crawler.checkpoint")

class CheckpointManager:
    """Manages crawl checkpoints and deduplication to ensure incremental crawl resume capability."""
    
    def __init__(self, crawler_name: str, checkpoint_dir: Path):
        self.crawler_name = crawler_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{crawler_name}.json"
        self.seen_urls: Set[str] = set()
        self.state: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load checkpoint file if it exists."""
        if not self.checkpoint_file.exists():
            self.state = {"last_page": 0, "processed_items": [], "seen_urls": []}
            return

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            
            seen_list = self.state.get("seen_urls", [])
            self.seen_urls = set(seen_list)
            log.info(f"Loaded checkpoint for {self.crawler_name}. Last page processed: {self.state.get('last_page', 0)}. Seen URLs count: {len(self.seen_urls)}")
        except Exception as e:
            log.error(f"Error loading checkpoint for {self.crawler_name}: {e}. Starting fresh.")
            self.state = {"last_page": 0, "processed_items": [], "seen_urls": []}

    def save(self, last_page: int, current_items: list, extra_state: Optional[Dict[str, Any]] = None) -> None:
        """Save the current crawl state atomically to checkpoint file."""
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        
        # Merge items
        processed_items = self.state.get("processed_items", [])
        # We can append new ones while avoiding duplicate objects
        existing_urls = {item.get("url") for item in processed_items if item.get("url")}
        for item in current_items:
            url = item.get("url")
            if url and url not in existing_urls:
                processed_items.append(item)
                self.seen_urls.add(url)
                existing_urls.add(url)

        self.state["last_page"] = last_page
        self.state["processed_items"] = processed_items
        self.state["seen_urls"] = list(self.seen_urls)
        
        if extra_state:
            self.state.update(extra_state)

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            temp_file.replace(self.checkpoint_file)
        except Exception as e:
            log.error(f"Failed to save checkpoint for {self.crawler_name}: {e}")

    def clear(self) -> None:
        """Clear the checkpoint file (typically when a crawl finishes successfully)."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                log.info(f"Cleared checkpoint file for {self.crawler_name}")
            except Exception as e:
                log.error(f"Failed to delete checkpoint file for {self.crawler_name}: {e}")
        self.state = {"last_page": 0, "processed_items": [], "seen_urls": []}
        self.seen_urls = set()

    def get_last_page(self) -> int:
        return self.state.get("last_page", 0)

    def get_processed_items(self) -> list:
        return self.state.get("processed_items", [])

    def is_seen(self, url: str) -> bool:
        return url in self.seen_urls

    def add_seen(self, url: str) -> None:
        self.seen_urls.add(url)
