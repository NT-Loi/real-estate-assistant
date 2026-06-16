import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from crawlers.base import BaseCrawler
from crawlers.config import DATA_DIR, YOUTUBE_API_KEY

log = logging.getLogger("bds_crawler.youtube")
YOUTUBE_MIN_PUBLISHED_AFTER = os.getenv("YOUTUBE_MIN_PUBLISHED_AFTER", "2025-01-01T00:00:00Z")

# EMOJI normalizer mappings from the user's notebook
CUSTOM_EMOJI_MAP = {
    "😂": "smile", "🤣": "laugh", "😁": "smile", "😀": "smile", "😊": "smile", "🙂": "smile",
    "😭": "cry", "😢": "cry", "😡": "angry", "👍": "thumbs up", "👎": "thumbs down",
    "❤️": "heart", "❤": "heart", "😍": "heart eyes", "🔥": "fire", "💯": "hundred"
}

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

def normalize_emoji(text: str) -> str:
    if not text:
        return text
    for k, v in CUSTOM_EMOJI_MAP.items():
        text = text.replace(k, v)
    if HAS_EMOJI:
        try:
            t = emoji.demojize(text).replace(":", " ").replace("_", " ")
            return " ".join(t.split())
        except Exception:
            pass
    return text

class YouTubeCrawler(BaseCrawler):
    """Crawler for collecting qualitative neighborhood feedback and reviews from YouTube comments & transcripts."""
    
    def __init__(self, output_file: Optional[Path] = None):
        super().__init__("youtube_neighborhood", output_file or (DATA_DIR / "youtube_comments.json"))
        self.api_key = YOUTUBE_API_KEY
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def _call_api(self, endpoint: str, params: dict, retries: int = 5, base_sleep: float = 1.5) -> dict:
        """Query Google YouTube Data API v3 with linear backoff retries."""
        url = f"{self.base_url}/{endpoint}"
        params["key"] = self.api_key
        
        for i in range(retries):
            try:
                r = requests.get(url, params=params, timeout=15)
                data = r.json()
                if "error" not in data:
                    return data
                
                reason = data["error"].get("errors", [{}])[0].get("reason", "")
                if reason in ("quotaExceeded", "dailyLimitExceeded"):
                    raise RuntimeError("YouTube API Quota exceeded. Please resume later.")
                if reason in ("commentsDisabled", "videoNotFound", "forbidden"):
                    self.log.warning(f"YouTube API Error ({reason}): {data['error'].get('message')}. Skipping.")
                    return {}
                
                self.log.warning(f"YouTube API Error ({reason}): {data['error'].get('message')}. Retrying...")
            except requests.RequestException as e:
                self.log.warning(f"Connection error to YouTube API: {e}. Retrying...")
            
            time.sleep(base_sleep * (i + 1))
        return {}

    def fetch_videos_by_keyword(self, keyword: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search YouTube for videos by keyword. Uses the keyword as-is for maximum flexibility."""
        self.log.info(
            "Searching YouTube videos for: '%s' (max_results=%s, published_after=%s)",
            keyword,
            max_results,
            YOUTUBE_MIN_PUBLISHED_AFTER,
        )
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": max_results,
            "relevanceLanguage": "vi",
            "publishedAfter": YOUTUBE_MIN_PUBLISHED_AFTER,
            "order": "date",
        }
        data = self._call_api("search", params)
        videos = []
        for item in data.get("items", []):
            vid_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if vid_id:
                videos.append({
                    "video_id": vid_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "channel": snippet.get("channelTitle"),
                    "publish_date": snippet.get("publishedAt"),
                    "url": f"https://www.youtube.com/watch?v={vid_id}"
                })
        self.log.info("YouTube search returned %s video(s) for keyword: '%s'", len(videos), keyword)
        for idx, video in enumerate(videos[:5], start=1):
            self.log.info(
                "  YouTube result %s/%s: id=%s published=%s channel=%s title=%r",
                idx,
                len(videos),
                video.get("video_id"),
                video.get("publish_date"),
                video.get("channel"),
                video.get("title"),
            )
        return videos

    def fetch_video_details(self, video_id: str) -> Dict[str, Any]:
        """Fetch full video details including complete description and statistics."""
        params = {
            "part": "snippet,statistics",
            "id": video_id
        }
        data = self._call_api("videos", params)
        items = data.get("items", [])
        if not items:
            return {}
        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        return {
            "full_description": snippet.get("description", ""),
            "tags": snippet.get("tags", []),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        }

    def fetch_video_transcript(self, video_id: str) -> Optional[str]:
        """Fetch video transcript safely using the youtube_transcript_api package."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            result = api.fetch(video_id, languages=["vi", "en"])
            text = " ".join([s.text for s in result.snippets])
            return text
        except ImportError:
            self.log.warning("youtube_transcript_api not installed. Skipping transcript crawl.")
        except Exception as e:
            self.log.warning(f"Could not retrieve transcript for video {video_id}: {e}")
        return None

    def fetch_missing_replies(self, parent_id: str, already_have: int, total_should_have: int, seen_ids: Set[str]) -> List[dict]:
        """Fetch extra replies that are paginated/missing from inline response."""
        rows = []
        next_token = None
        
        while already_have < total_should_have:
            params = {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": 100,
                "pageToken": next_token
            }
            data = self._call_api("comments", params)
            items = data.get("items", [])
            if not items:
                break
                
            for rep in items:
                rep_id = rep.get("id")
                if rep_id in seen_ids:
                    continue
                s = rep.get("snippet", {})
                rows.append({
                    "comment_id": rep_id,
                    "parent_id": s.get("parentId"),
                    "author": s.get("authorDisplayName"),
                    "comment_raw": s.get("textOriginal", ""),
                    "comment_norm": normalize_emoji(s.get("textOriginal", "")),
                    "published_at": s.get("publishedAt"),
                    "updated_at": s.get("updatedAt"),
                    "like_count": s.get("likeCount", 0),
                    "is_reply": True
                })
                seen_ids.add(rep_id)
                already_have += 1
                
            next_token = data.get("nextPageToken")
            if not next_token:
                break
                
        return rows

    def fetch_comments_for_video(self, video_id: str, max_comments: int = 100) -> List[Dict[str, Any]]:
        """Download comments and their replies for a target video ID."""
        self.log.info(f"Fetching comments for video: {video_id}")
        comments_list = []
        seen_ids = set()
        next_token_top = None
        total_written = 0

        while total_written < max_comments:
            params = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": 100,
                "pageToken": next_token_top,
                "textFormat": "plainText"
            }
            data = self._call_api("commentThreads", params)
            items = data.get("items", [])
            if not items:
                break

            for it in items:
                if total_written >= max_comments:
                    break

                top = it.get("snippet", {}).get("topLevelComment", {})
                top_id = top.get("id")
                s = top.get("snippet", {})
                
                if not top_id or top_id in seen_ids:
                    continue

                batch = []
                # top level comment
                row = {
                    "comment_id": top_id,
                    "parent_id": None,
                    "author": s.get("authorDisplayName"),
                    "comment_raw": s.get("textOriginal", ""),
                    "comment_norm": normalize_emoji(s.get("textOriginal", "")),
                    "published_at": s.get("publishedAt"),
                    "updated_at": s.get("updatedAt"),
                    "like_count": s.get("likeCount", 0),
                    "is_reply": False
                }
                batch.append(row)
                seen_ids.add(top_id)

                # replies
                total_replies = it.get("snippet", {}).get("totalReplyCount", 0)
                have_replies = 0
                if "replies" in it and "comments" in it["replies"]:
                    for rep in it["replies"]["comments"]:
                        rep_id = rep.get("id")
                        if rep_id in seen_ids:
                            continue
                        rs = rep.get("snippet", {})
                        batch.append({
                            "comment_id": rep_id,
                            "parent_id": rs.get("parentId"),
                            "author": rs.get("authorDisplayName"),
                            "comment_raw": rs.get("textOriginal", ""),
                            "comment_norm": normalize_emoji(rs.get("textOriginal", "")),
                            "published_at": rs.get("publishedAt"),
                            "updated_at": rs.get("updatedAt"),
                            "like_count": rs.get("likeCount", 0),
                            "is_reply": True
                        })
                        seen_ids.add(rep_id)
                        have_replies += 1
                        if total_written + len(batch) >= max_comments:
                            break

                # Missing replies check
                if total_replies > have_replies and total_written + len(batch) < max_comments:
                    missing = self.fetch_missing_replies(top_id, have_replies, total_replies, seen_ids)
                    if total_written + len(batch) + len(missing) > max_comments:
                        missing = missing[:max_comments - (total_written + len(batch))]
                    batch.extend(missing)

                # Append to comments
                comments_list.extend(batch)
                total_written += len(batch)

            next_token_top = data.get("nextPageToken")
            if not next_token_top:
                break

        top_count = sum(1 for c in comments_list if not c.get("is_reply"))
        reply_count = sum(1 for c in comments_list if c.get("is_reply"))
        self.log.info(
            "Fetched YouTube comments for video %s: total=%s top_level=%s replies=%s",
            video_id,
            len(comments_list),
            top_count,
            reply_count,
        )
        return comments_list

    async def crawl(
        self,
        keywords: List[str],
        max_videos_per_kw: int = 3,
        max_comments_per_video: int = 50,
        resume: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute complete keyword search and parsing pipeline."""
        if not self.api_key:
            self.log.error("YOUTUBE_API_KEY environment variable is not set. Cannot run YouTubeCrawler.")
            return []

        self.log.info(
            "Starting YouTube Neighborhood Crawl. keywords=%s max_videos_per_kw=%s max_comments_per_video=%s published_after=%s resume=%s",
            len(keywords or []),
            max_videos_per_kw,
            max_comments_per_video,
            YOUTUBE_MIN_PUBLISHED_AFTER,
            resume,
        )
        all_results = []
        stats = {
            "keywords": len(keywords or []),
            "search_results": 0,
            "skipped_existing": 0,
            "processed": 0,
            "comments": 0,
            "transcripts": 0,
        }
        
        if resume:
            self.checkpoint_mgr.load()
            all_results = self.load_existing_data() + self.checkpoint_mgr.get_processed_items()
            self.log.info(f"Resumed crawl. Loaded {len(all_results)} existing videos from output/checkpoints.")

        processed_vids = {item["video_id"] for item in all_results if "video_id" in item}

        for kw in keywords:
            videos = self.fetch_videos_by_keyword(kw, max_results=max_videos_per_kw)
            stats["search_results"] += len(videos)
            keyword_processed = 0
            keyword_skipped_existing = 0
            
            for idx, vid in enumerate(videos):
                vid_id = vid["video_id"]
                if vid_id in processed_vids:
                    keyword_skipped_existing += 1
                    stats["skipped_existing"] += 1
                    continue

                self.log.info(
                    "Processing YouTube video %s/%s for keyword=%r: id=%s published=%s title=%r",
                    idx + 1,
                    len(videos),
                    kw,
                    vid_id,
                    vid.get("publish_date"),
                    vid.get("title"),
                )
                
                # Fetch full video details (complete description + stats)
                details = self.fetch_video_details(vid_id)
                
                # Fetch comments and transcript
                comments_data = self.fetch_comments_for_video(vid_id, max_comments=max_comments_per_video)
                transcript = self.fetch_video_transcript(vid_id)
                if transcript:
                    self.log.info("Fetched YouTube transcript for video %s: chars=%s", vid_id, len(transcript))

                # Assemble
                vid_record = {
                    "keyword": kw,
                    "video_id": vid_id,
                    "title": vid["title"],
                    "url": vid["url"],
                    "description": details.get("full_description") or vid["description"],
                    "tags": details.get("tags", []),
                    "channel": vid["channel"],
                    "publish_date": vid["publish_date"],
                    "stats": {
                        "views": details.get("view_count"),
                        "likes": details.get("like_count"),
                        "comments": details.get("comment_count"),
                    },
                    "transcript_text": transcript,
                    "comments": comments_data,
                    "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                
                all_results.append(vid_record)
                processed_vids.add(vid_id)
                keyword_processed += 1
                stats["processed"] += 1
                stats["comments"] += len(comments_data)
                if transcript:
                    stats["transcripts"] += 1
                
                # Save progress checkpoint
                self.checkpoint_mgr.save(idx, [vid_record])
                
                # Sleep a short delay to stay polite
                await self.sleep_polite(1.0)

            self.log.info(
                "YouTube keyword summary: keyword=%r search_results=%s processed_new=%s skipped_existing=%s",
                kw,
                len(videos),
                keyword_processed,
                keyword_skipped_existing,
            )

        self.save_final_results(all_results, resume)
        self.log.info(
            "YouTube crawl summary: keywords=%s search_results=%s processed_new=%s skipped_existing=%s comments=%s transcripts=%s total_records_after_merge=%s",
            stats["keywords"],
            stats["search_results"],
            stats["processed"],
            stats["skipped_existing"],
            stats["comments"],
            stats["transcripts"],
            len(all_results),
        )
        return all_results
