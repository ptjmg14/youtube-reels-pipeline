from __future__ import annotations

import logging
import time

from google.genai import Client, errors, types

logger = logging.getLogger(__name__)

# List of models by capacity/stability (current 2026 lineup)
MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash"
]

RETRY_ROUNDS = 3
RETRY_SLEEP_S = 20


def generate_with_fallback(
    api_key: str,
    prompt: str,
    model: str | None = None,
    mime_type: str = "application/json",
) -> str:
    models = list(dict.fromkeys([model, *MODEL_FALLBACKS] if model else MODEL_FALLBACKS))
    client = Client(api_key=api_key)

    for _round in range(RETRY_ROUNDS):
        for candidate in models:
            try:
                logger.info(f"Attempting generation with model: {candidate}")
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type=mime_type),
                )
                return response.text or ""
            except errors.ClientError as e:
                if _retryable(e):
                    logger.warning(
                        f"Model {candidate} unavailable ({e}); trying next..."
                    )
                    continue
                logger.error(f"Model {candidate} failed with client error: {e}")
                raise
            except errors.ServerError as e:
                logger.warning(
                    f"Model {candidate} failed with 503/Server error: {e}. Trying next..."
                )
                continue
            except Exception as e:
                logger.error(f"Model {candidate} failed with unexpected error: {e}")
                raise
        if _round < RETRY_ROUNDS - 1:
            logger.info(f"All models busy; retrying in {RETRY_SLEEP_S}s...")
            time.sleep(RETRY_SLEEP_S)

    raise RuntimeError(
        "All Gemini models failed (server errors or free-tier quota exhausted)."
    )


def _retryable(error: errors.ClientError) -> bool:
    """True when a different fallback model might succeed (quota/retired/model)."""
    message = f"{error}".lower()
    return (
        error.code == 429
        or error.code == 404
        or "resource_exhausted" in message
        or "quota" in message
        or "no longer available" in message
    )
