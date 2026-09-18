from __future__ import annotations

import re
from collections.abc import Iterable

from .models import TranscriptSegment

SHINGLE_SIZE = 5
_CHUNK_SIZE = 400


def transcript_text(transcript: Iterable[TranscriptSegment]) -> str:
    """Join the transcript into a single normalized-whitespace string."""
    return " ".join(segment.text.strip() for segment in transcript if segment.text.strip())


def transcript_chunks(
    transcript: Iterable[TranscriptSegment], size: int = _CHUNK_SIZE
) -> list[str]:
    """Pack consecutive segments into ~``size``-character windows.

    These windows are only used as semantic queries to recall candidate
    videos; the exact overlap is confirmed afterwards with shingles.
    """
    chunks: list[str] = []
    buffer = ""
    for segment in transcript:
        text = segment.text.strip()
        if not text:
            continue
        buffer = f"{buffer} {text}".strip()
        if len(buffer) >= size:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def normalize(text: str) -> str:
    """Keep only word characters (CJK included), lowercased.

    Dropping whitespace and punctuation makes the overlap robust to subtitle
    formatting differences between two uploads of the same content.
    """
    return re.sub(r"[^\w]", "", text.lower(), flags=re.UNICODE)


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    normalized = normalize(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Symmetric overlap of two shingle sets, in ``[0, 1]``."""
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0
