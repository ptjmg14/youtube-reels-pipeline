from __future__ import annotations

from typing import Protocol

from ..models import SegmentWindow, TranscriptSegment


class Prefilter(Protocol):
    def suggest(
        self, transcript: list[TranscriptSegment], max_clips: int
    ) -> list[SegmentWindow]: ...