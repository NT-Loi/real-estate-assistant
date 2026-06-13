"""
Chunker — convert raw crawled records into embeddable text chunks.

Strategies:
    - Listings & Projects: template-format all fields into a single natural
      language string (short enough to embed as one chunk).
    - Articles (News/Wiki): split long body text into overlapping segments
      using recursive character splitting.
"""
from __future__ import annotations

import re
from typing import Optional

from db.config import ARTICLE_CHUNK_SIZE, ARTICLE_CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Listing → text
# ---------------------------------------------------------------------------
def listing_to_text(record: dict) -> str:
    """
    Format a merged listing record into a natural language string
    suitable for embedding.
    """
    parts: list[str] = []

    # Date prefix for recency filtering
    date = record.get("ngay_dang") or record.get("posted_at")
    if date:
        parts.append(f"[{str(date)[:10]}]")

    # Title
    if record.get("tieu_de"):
        parts.append(record["tieu_de"])

    # Property type + location
    loc_parts = []
    if record.get("loai_nha_dat"):
        loc_parts.append(record["loai_nha_dat"])
    addr = record.get("dia_chi") or record.get("khu_vuc")
    if addr:
        loc_parts.append(f"tại {addr}")
    if loc_parts:
        parts.append(". ".join(loc_parts))

    # Project name (useful for lifestyle/project queries)
    if record.get("du_an"):
        parts.append(f"Thuộc dự án: {record['du_an']}")

    # Price + Area
    price_area = []
    if record.get("gia"):
        price_area.append(f"Giá: {record['gia']}")
    if record.get("dien_tich"):
        price_area.append(f"Diện tích: {record['dien_tich']}")
    if record.get("gia_per_m2"):
        price_area.append(f"Giá/m²: {record['gia_per_m2']}")
    if price_area:
        parts.append(", ".join(price_area))

    # Rooms
    rooms = []
    if record.get("so_phong_ngu"):
        rooms.append(f"{record['so_phong_ngu']} phòng ngủ")
    if record.get("so_phong_tam"):
        rooms.append(f"{record['so_phong_tam']} phòng tắm/toilet")
    if rooms:
        parts.append(", ".join(rooms))

    # Orientation
    orient = []
    if record.get("huong_nha"):
        orient.append(f"Hướng nhà: {record['huong_nha']}")
    if record.get("huong_ban_cong"):
        orient.append(f"Hướng ban công: {record['huong_ban_cong']}")
    if orient:
        parts.append(", ".join(orient))

    # Extra specs
    extras = []
    if record.get("phap_ly"):
        extras.append(f"Pháp lý: {record['phap_ly']}")
    if record.get("noi_that"):
        extras.append(f"Nội thất: {record['noi_that']}")
    if record.get("so_tang"):
        extras.append(f"Số tầng: {record['so_tang']}")
    if record.get("mat_tien"):
        extras.append(f"Mặt tiền: {record['mat_tien']}")
    if record.get("duong_vao"):
        extras.append(f"Đường vào: {record['duong_vao']}")
    if extras:
        parts.append(". ".join(extras))

    # Description
    desc = record.get("mo_ta_chi_tiet") or record.get("mo_ta")
    if desc:
        # Truncate very long descriptions
        parts.append(desc[:1500])

    text = ". ".join(parts)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Project → text
# ---------------------------------------------------------------------------
def project_to_text(record: dict) -> str:
    """Format a project record into embeddable text."""
    parts: list[str] = []

    # Date prefix
    date = record.get("ngay_dang") or record.get("posted_at")
    if date:
        parts.append(f"[{str(date)[:10]}]")

    if record.get("ten_du_an"):
        parts.append(f"Dự án: {record['ten_du_an']}")

    info = []
    if record.get("loai_du_an"):
        info.append(record["loai_du_an"])
    addr = record.get("dia_chi") or record.get("khu_vuc")
    if addr:
        info.append(f"tại {addr}")
    if info:
        parts.append(". ".join(info))

    if record.get("chu_dau_tu"):
        parts.append(f"Chủ đầu tư: {record['chu_dau_tu']}")
    if record.get("quy_mo"):
        parts.append(f"Quy mô: {record['quy_mo']}")
    if record.get("so_can_ho"):
        parts.append(f"Số căn hộ: {record['so_can_ho']}")
    if record.get("gia"):
        parts.append(f"Giá: {record['gia']}")
    if record.get("dien_tich"):
        parts.append(f"Diện tích: {record['dien_tich']}")
    if record.get("trang_thai"):
        parts.append(f"Trạng thái: {record['trang_thai']}")
    if record.get("phap_ly"):
        parts.append(f"Pháp lý: {record['phap_ly']}")
    if record.get("nam_ban_giao"):
        parts.append(f"Bàn giao: {record['nam_ban_giao']}")

    if record.get("tien_ich"):
        tien_ich = record["tien_ich"]
        if isinstance(tien_ich, list):
            tien_ich = ", ".join(tien_ich)
        parts.append(f"Tiện ích: {tien_ich}")

    desc = record.get("mo_ta_chi_tiet")
    if desc:
        parts.append(desc[:2000])

    text = ". ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Article → chunks
# ---------------------------------------------------------------------------
def _split_text(
    text: str,
    chunk_size: int = ARTICLE_CHUNK_SIZE,
    chunk_overlap: int = ARTICLE_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks by paragraph boundaries first,
    then by sentence boundaries, then by character.
    """
    if len(text) <= chunk_size:
        return [text]

    # Split by double newline (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed chunk_size
        if len(current_chunk) + len(para) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: keep tail of current chunk
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:]
                else:
                    current_chunk = ""

            # If a single paragraph is too long, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                                current_chunk = current_chunk[-chunk_overlap:]
                            else:
                                current_chunk = ""
                        # If a single sentence is still too long, hard-split
                        if len(sent) > chunk_size:
                            for i in range(0, len(sent), chunk_size - chunk_overlap):
                                chunks.append(sent[i : i + chunk_size])
                        else:
                            current_chunk = sent
                    else:
                        current_chunk = (current_chunk + " " + sent).strip()
            else:
                current_chunk = (current_chunk + " " + para).strip()
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def article_to_chunks(record: dict) -> list[tuple[str, int, int]]:
    """
    Convert an article record into a list of (chunk_text, chunk_index, total_chunks).

    The chunk text includes the article category + title as a prefix for retrieval context.

    Field name note: DB stores full body as 'mo_ta_chi_tiet', but some crawlers
    use 'noi_dung'. We check both to handle both sources correctly.
    """
    title = record.get("tieu_de") or ""
    # Read body from DB field (mo_ta_chi_tiet) or crawler field (noi_dung) or short summary (mo_ta)
    body = record.get("mo_ta_chi_tiet") or record.get("noi_dung") or record.get("mo_ta") or ""
    danh_muc = record.get("danh_muc") or ""
    date = record.get("ngay_dang") or record.get("published_at")
    date_prefix = f"[{str(date)[:10]}] " if date else ""
    cat_prefix = f"[{danh_muc}] " if danh_muc else ""

    if not body:
        # No content — use title + summary as single chunk
        text = f"{date_prefix}{cat_prefix}{title}"
        if record.get("mo_ta") and record["mo_ta"] not in text:
            text += ". " + record["mo_ta"]
        return [(text, 0, 1)] if text.strip() else []

    raw_chunks = _split_text(body)
    total = len(raw_chunks)

    result: list[tuple[str, int, int]] = []
    for i, chunk in enumerate(raw_chunks):
        # Prefix each chunk with date + category + title for retrieval context
        prefixed = f"{date_prefix}{cat_prefix}{title}. {chunk}" if title else f"{date_prefix}{cat_prefix}{chunk}"
        prefixed = re.sub(r"\s+", " ", prefixed).strip()
        result.append((prefixed, i, total))

    return result


# ---------------------------------------------------------------------------
# Social neighborhood → text
# ---------------------------------------------------------------------------
def social_to_text(record: dict) -> str:
    """Format social platform discussions into cohesive semantic text suitable for embedding."""
    parts = []
    source = record.get("source_type", "mạng xã hội").upper()
    kw = record.get("keyword")

    # Date prefix for recency
    date = record.get("published_at") or record.get("ngay_dang")
    if date:
        parts.append(f"[{str(date)[:10]}]")

    if kw:
        parts.append(f"Ý kiến thảo luận và đánh giá thực tế về {kw} trên {source}.")
    else:
        parts.append(f"Thảo luận thực tế trên mạng xã hội {source}.")

    title = record.get("title") or record.get("thread_title")
    if title:
        parts.append(f"Nội dung thảo luận: '{title}'.")

    desc = record.get("text_content") or record.get("description")
    if desc:
        parts.append(f"Chi tiết thảo luận: {desc[:600]}")

    # Capture top comments to represent actual neighborhood sentiment
    comments = record.get("comments") or record.get("posts") or []
    if comments:
        feedback = []
        for c in comments[:6]:
            txt = c.get("comment_norm") or c.get("comment_raw") or c.get("content")
            if txt and len(txt.strip()) > 5:
                # Remove extra formatting
                txt_clean = re.sub(r"\s+", " ", txt).strip()
                feedback.append(f"\"{txt_clean[:200]}\"")
        if feedback:
            parts.append("Nhận xét thực tế từ người dân: " + ", ".join(feedback))

    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()

