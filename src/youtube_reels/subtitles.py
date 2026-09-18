from __future__ import annotations

import re
from pathlib import Path

from .models import TranscriptSegment


def parse_vtt(path: Path) -> list[TranscriptSegment]:
    """Parse the subset of WebVTT emitted by yt-dlp, dropping duplicate cue text."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    segments: list[TranscriptSegment] = []
    index = 0
    while index < len(lines):
        match = re.match(r"(\d\d:\d\d:\d\d\.\d+)\s+-->\s+(\d\d:\d\d:\d\d\.\d+)", lines[index])
        if not match:
            index += 1
            continue
        start, end = (_parse_timestamp(value) for value in match.groups())
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip():
            clean = re.sub(r"<[^>]+>", "", lines[index]).strip()
            if clean:
                content.append(clean)
            index += 1
        text = " ".join(content)
        if text and (not segments or segments[-1].text != text):
            segments.append(TranscriptSegment(start=start, end=end, text=text))
        index += 1
    return segments


def write_srt(segments: list[TranscriptSegment], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for number, segment in enumerate(segments, start=1):
        rows.extend(
            [
                str(number),
                f"{_srt_time(segment.start)} --> {_srt_time(segment.end)}",
                segment.text,
                "",
            ]
        )
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _srt_time(seconds: float) -> str:
    milliseconds = round((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
