from __future__ import annotations

from ..models import SegmentWindow, TranscriptSegment


class HeuristicPrefilter:
    """Cost-free structural pass: packs transcript segments into coherent
    15–55s windows as reference material for the rewrite.

    Window packing is kept because the LLM needs clean, bounded excerpts and
    sending them all costs little. Choosing the best clips is deliberately
    delegated to the LLM rewrite step — semantic selection by Gemini is more
    reliable than keyword/length scoring.
    """

    def suggest(
        self, transcript: list[TranscriptSegment], max_clips: int | None = None
    ) -> list[SegmentWindow]:
        if not transcript:
            return []
        return [window for window in self._pack(transcript) if window.end - window.start >= 15]

    def _pack(self, transcript: list[TranscriptSegment]) -> list[SegmentWindow]:
        windows: list[SegmentWindow] = []
        index = 0
        while index < len(transcript):
            start = transcript[index].start
            end = transcript[index].end
            texts = [transcript[index].text]
            cursor = index + 1
            while cursor < len(transcript) and transcript[cursor].end - start <= 55:
                end = transcript[cursor].end
                texts.append(transcript[cursor].text)
                cursor += 1
            windows.append(
                SegmentWindow(
                    start=start, end=end, text=" ".join(texts)
                )
            )
            index = max(cursor, index + 1)
        return windows