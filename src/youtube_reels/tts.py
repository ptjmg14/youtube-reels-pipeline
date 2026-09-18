from __future__ import annotations

import asyncio
import time
from pathlib import Path


class TtsError(RuntimeError):
    pass


def synthesize(text: str, output_path: Path, voice: str, rate: str = "+0%") -> Path:
    """Synthesise narration with edge-tts (free, no API key required)."""
    if not text.strip():
        raise TtsError("Empty narration — cannot generate audio.")
    try:
        import edge_tts
    except ImportError as error:  # pragma: no cover - dependency guard
        raise TtsError("Install the local TTS: pip install edge-tts") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            asyncio.run(
                _synthesize_async(edge_tts, text.strip(), str(output_path), voice, rate)
            )
            break
        except Exception as error:  # noqa: BLE001 - surface the underlying cause
            last_error = error
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    else:
        raise TtsError(
            f"Could not synthesise voice ({voice}) after 3 attempts: {last_error}"
        ) from last_error
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TtsError("edge-tts produced no audio.")
    return output_path


async def _synthesize_async(
    edge_tts: object, text: str, output_path: str, voice: str, rate: str
) -> None:
    communicator = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicator.save(output_path)