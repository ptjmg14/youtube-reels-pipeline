from __future__ import annotations

from pathlib import Path

from .models import TranscriptSegment


class TranscriptionError(RuntimeError):
    pass


# Groq service constants (OpenAI-compatible /audio/transcriptions endpoint).
_GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def transcribe(
    video_path: str,
    model_name: str,
    transcriber: str = "auto",
    groq_api_key: str | None = None,
    groq_model: str = "whisper-large-v3-turbo",
) -> list[TranscriptSegment]:
    """Transcribe audio into segments.

    - ``transcriber="groq"`` (or ``auto`` with a key): whisper-large-v3-turbo via
      API (free, ~200x faster than the local CPU model).
    - Otherwise, or on failure: local whisper (faster-whisper, offline).
    """
    if transcriber in {"groq", "auto"} and groq_api_key:
        try:
            segments = _transcribe_groq(video_path, groq_api_key, groq_model)
        except Exception as error:  # noqa: BLE001 - fallback to local whisper
            print(f"Warning: Groq failed ({error}); falling back to local whisper.")
        else:
            if segments:
                return segments
    return _transcribe_local(video_path, model_name)


def _transcribe_groq(
    video_path: str, api_key: str, model: str
) -> list[TranscriptSegment]:
    import requests

    with open(video_path, "rb") as file:
        response = requests.post(
            _GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(video_path).name, file, "audio/mpeg")},
            data={"model": model, "response_format": "verbose_json"},
            timeout=600,
        )
    if not response.ok:
        raise TranscriptionError(f"Groq {response.status_code}: {response.text[:400]}")
    payload = response.json()
    segments = []
    for segment in payload.get("segments", []):
        text = (segment.get("text") or "").strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=text,
                )
            )
    return segments


def _transcribe_local(video_path: str, model_name: str) -> list[TranscriptSegment]:
    """Run Whisper locally. This is a paid-API-free fallback for absent captions."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - dependency guard
        raise TranscriptionError(
            "Install local transcription: pip install -e '.[local]'"
        ) from error
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    raw_segments, _ = model.transcribe(
        video_path, vad_filter=True, beam_size=1, condition_on_previous_text=False
    )
    segments = [
        TranscriptSegment(start=segment.start, end=segment.end, text=segment.text.strip())
        for segment in raw_segments
        if segment.text.strip()
    ]
    if not segments:
        raise TranscriptionError("Could not obtain speech from this video.")
    return segments