"""
Chunker - convert crawled records into focused embeddable text chunks.

The active embedding model is AITeamVN/Vietnamese_Embedding_v2 (BGE-M3 based),
which supports long Vietnamese context. We still chunk by semantic sections so
retrieval can hit the exact part of a listing, project, article, or discussion.
"""
from __future__ import annotations

import re
from typing import Any

from db.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    SOCIAL_COMMENT_BATCH_TOKENS,
)


Chunk = dict[str, Any]


def _clean(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _append(parts: list[str], label: str, value: Any) -> None:
    text = _clean(value)
    if text:
        parts.append(f"{label}: {text}")


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer for chunk sizing without loading the HF tokenizer."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def _split_text_by_tokens(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split long Vietnamese text into token-budgeted chunks."""
    text = _clean(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(_detokenize(current))
            current = current[-overlap_tokens:] if overlap_tokens > 0 else []

    for para in paragraphs or [text]:
        para_tokens = _tokenize(para)
        if not para_tokens:
            continue

        if len(para_tokens) > target_tokens:
            flush()
            step = max(1, target_tokens - overlap_tokens)
            for start in range(0, len(para_tokens), step):
                window = para_tokens[start : start + target_tokens]
                if window:
                    chunks.append(_detokenize(window))
            current = []
            continue

        if len(current) + len(para_tokens) > target_tokens:
            flush()

        current.extend(para_tokens)

    if current:
        chunks.append(_detokenize(current))

    return [c for c in chunks if c]


def _make_chunks(prefix: str, chunk_type: str, bodies: list[str]) -> list[Chunk]:
    texts = [f"{prefix}. {body}".strip(". ") if prefix else body for body in bodies if _clean(body)]
    total = len(texts)
    return [
        {
            "text": _clean(text),
            "chunk_type": chunk_type,
            "chunk_index": idx,
            "total_chunks": total,
        }
        for idx, text in enumerate(texts)
        if _clean(text)
    ]


def _date_prefix(record: dict) -> str:
    date = record.get("ngay_dang") or record.get("posted_at") or record.get("published_at")
    return f"[{str(date)[:10]}] " if date else ""


def listing_fact_text(record: dict) -> str:
    parts: list[str] = []
    if record.get("tieu_de"):
        parts.append(_clean(record["tieu_de"]))
    _append(parts, "Loại hình", record.get("loai_hinh"))
    _append(parts, "Loại nhà đất", record.get("loai_nha_dat"))
    _append(parts, "Địa chỉ", record.get("dia_chi") or record.get("khu_vuc"))
    _append(parts, "Dự án", record.get("du_an"))
    _append(parts, "Giá", record.get("gia"))
    _append(parts, "Giá mỗi m2", record.get("gia_per_m2"))
    _append(parts, "Diện tích", record.get("dien_tich"))
    _append(parts, "Phòng ngủ", record.get("so_phong_ngu"))
    _append(parts, "Phòng tắm", record.get("so_phong_tam"))
    _append(parts, "Hướng nhà", record.get("huong_nha"))
    _append(parts, "Hướng ban công", record.get("huong_ban_cong"))
    _append(parts, "Pháp lý", record.get("phap_ly"))
    _append(parts, "Nội thất", record.get("noi_that"))
    _append(parts, "Số tầng", record.get("so_tang"))
    _append(parts, "Mặt tiền", record.get("mat_tien"))
    _append(parts, "Đường vào", record.get("duong_vao"))
    return _clean(". ".join(parts))


def listing_to_chunks(record: dict) -> list[Chunk]:
    """Return fact and description chunks for one listing."""
    prefix = _date_prefix(record) + "Tin bất động sản"
    chunks = _make_chunks(prefix, "facts", [listing_fact_text(record)])

    desc = record.get("mo_ta_chi_tiet") or record.get("mo_ta")
    title = _clean(record.get("tieu_de"))
    desc_prefix = f"{_date_prefix(record)}Mô tả tin: {title}" if title else f"{_date_prefix(record)}Mô tả tin"
    desc_chunks = _make_chunks(
        desc_prefix,
        "description",
        _split_text_by_tokens(desc),
    )

    all_chunks = chunks + desc_chunks
    total = len(all_chunks)
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = idx
        chunk["total_chunks"] = total
    return all_chunks


def listing_to_text(record: dict) -> str:
    """Compatibility wrapper: join all listing chunks into one text."""
    return " ".join(chunk["text"] for chunk in listing_to_chunks(record))


def project_fact_text(record: dict) -> str:
    parts: list[str] = []
    _append(parts, "Dự án", record.get("ten_du_an"))
    _append(parts, "Loại dự án", record.get("loai_du_an"))
    _append(parts, "Địa chỉ", record.get("dia_chi") or record.get("khu_vuc"))
    _append(parts, "Chủ đầu tư", record.get("chu_dau_tu"))
    _append(parts, "Quy mô", record.get("quy_mo"))
    _append(parts, "Số căn hộ", record.get("so_can_ho"))
    _append(parts, "Giá", record.get("gia"))
    _append(parts, "Diện tích", record.get("dien_tich"))
    _append(parts, "Trạng thái", record.get("trang_thai"))
    _append(parts, "Pháp lý", record.get("phap_ly"))
    _append(parts, "Bàn giao", record.get("nam_ban_giao"))
    return _clean(". ".join(parts))


def _project_amenities(record: dict) -> str:
    amenities = record.get("tien_ich")
    if isinstance(amenities, list):
        amenities = ", ".join(_clean(x) for x in amenities if _clean(x))
    return _clean(amenities)


def project_to_chunks(record: dict) -> list[Chunk]:
    """Return fact, amenity, and description chunks for one project."""
    name = _clean(record.get("ten_du_an"))
    prefix = f"{_date_prefix(record)}Dự án {name}" if name else f"{_date_prefix(record)}Dự án bất động sản"

    chunks = _make_chunks(prefix, "facts", [project_fact_text(record)])
    chunks += _make_chunks(prefix, "amenities", [f"Tiện ích: {_project_amenities(record)}"])
    chunks += _make_chunks(
        prefix,
        "description",
        _split_text_by_tokens(record.get("mo_ta_chi_tiet")),
    )

    all_chunks = [c for c in chunks if c["text"]]
    total = len(all_chunks)
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = idx
        chunk["total_chunks"] = total
    return all_chunks


def project_to_text(record: dict) -> str:
    """Compatibility wrapper: join all project chunks into one text."""
    return " ".join(chunk["text"] for chunk in project_to_chunks(record))


def article_to_chunks(record: dict) -> list[Chunk]:
    """Convert one news/wiki article into token-budgeted chunks."""
    title = _clean(record.get("tieu_de") or record.get("title"))
    body = record.get("mo_ta_chi_tiet") or record.get("noi_dung") or record.get("content") or record.get("mo_ta")
    category = _clean(record.get("danh_muc") or record.get("category"))
    date = _date_prefix(record)
    prefix_parts = [date.strip(), f"[{category}]" if category else "", title]
    prefix = _clean(" ".join(p for p in prefix_parts if p))

    bodies = _split_text_by_tokens(body)
    if not bodies and title:
        bodies = [_clean(record.get("mo_ta") or title)]
    return _make_chunks(prefix, "article_body", bodies)


def _comment_text(comment: dict) -> str:
    text = (
        comment.get("comment_norm")
        or comment.get("comment_raw")
        or comment.get("content")
        or comment.get("text")
        or ""
    )
    return _clean(text)


def _comment_batches(comments: list[dict], target_tokens: int) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for comment in comments:
        text = _comment_text(comment)
        if len(text) <= 5:
            continue
        line = f"- {text}"
        size = len(_tokenize(line))
        if size > target_tokens:
            if current:
                batches.append("\n".join(current))
                current = []
                current_tokens = 0
            batches.extend(f"- {part}" for part in _split_text_by_tokens(text, target_tokens, 0))
            continue
        if current and current_tokens + size > target_tokens:
            batches.append("\n".join(current))
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += size

    if current:
        batches.append("\n".join(current))
    return batches


def social_to_chunks(record: dict) -> list[Chunk]:
    """Split one social/video/forum record into summary and comment chunks."""
    source = _clean(record.get("source_type") or "mạng xã hội").upper()
    keyword = _clean(record.get("keyword"))
    title = _clean(record.get("title") or record.get("thread_title"))
    desc = _clean(record.get("text_content") or record.get("description") or record.get("transcript_text"))

    summary_parts = [
        _date_prefix(record).strip(),
        f"Nguồn: {source}",
        f"Chủ đề/khu vực: {keyword}" if keyword else "",
        f"Tiêu đề: {title}" if title else "",
    ]
    chunks = _make_chunks("", "summary", [" ".join(p for p in summary_parts if p)])
    chunks += _make_chunks(
        f"Nội dung từ {source}" + (f" về {keyword}" if keyword else ""),
        "content",
        _split_text_by_tokens(desc),
    )

    comments = record.get("comments") or record.get("posts") or []
    if isinstance(comments, list):
        prefix = f"Ý kiến người dùng từ {source}"
        if keyword:
            prefix += f" về {keyword}"
        chunks += _make_chunks(
            prefix,
            "comments",
            _comment_batches(comments, SOCIAL_COMMENT_BATCH_TOKENS),
        )

    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["total_chunks"] = total
    return chunks


def social_to_text(record: dict) -> str:
    """Compatibility wrapper: join all social chunks into one text."""
    return " ".join(chunk["text"] for chunk in social_to_chunks(record))
