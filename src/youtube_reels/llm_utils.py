from __future__ import annotations

import logging
import time

import httpx
from google.genai import Client, errors, types

logger = logging.getLogger(__name__)

# List of models by capacity/stability (current 2026 lineup)
MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash-lite",
]

# Groq free-tier models tried in order when Gemini quota is exhausted
# (best quality first, largest daily quota last). Rate limits are per model.
GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]

RETRY_ROUNDS = 3
RETRY_SLEEP_S = 20

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_with_fallback(
    api_key: str,
    prompt: str,
    model: str | None = None,
    mime_type: str = "application/json",
    groq_api_key: str | None = None,
    groq_models: list[str] | None = None,
) -> str:
    models = list(dict.fromkeys([model, *MODEL_FALLBACKS] if model else MODEL_FALLBACKS))
    client = Client(api_key=api_key)

    for _round in range(RETRY_ROUNDS):
        transient_503 = False
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
                transient_503 = True
                logger.warning(
                    f"Model {candidate} failed with 503/Server error: {e}. Trying next..."
                )
                continue
            except Exception as e:
                logger.error(f"Model {candidate} failed with unexpected error: {e}")
                raise
        if not transient_503:
            # Every candidate this round failed only with quota/retired-model
            # errors (429/404). Retrying is pointless — jump straight to Groq.
            logger.info("Gemini quota exhausted; skipping remaining retry rounds.")
            break
        if _round < RETRY_ROUNDS - 1:
            logger.info(f"All models busy; retrying in {RETRY_SLEEP_S}s...")
            time.sleep(RETRY_SLEEP_S)

    if groq_api_key:
        return _generate_with_groq(prompt, groq_api_key, groq_models)

    raise RuntimeError(
        "All Gemini models failed (server errors or free-tier quota exhausted)."
    )


def _generate_with_groq(
    prompt: str,
    groq_api_key: str,
    groq_models: list[str] | None,
) -> str:
    """Fallback to Groq free tier (OpenAI-compatible chat completions).

    Quotas are per model, so several models are tried in sequence to maximise
    the remaining daily budget before giving up.
    """
    models = [
        model
        for model in (groq_models or GROQ_FALLBACK_MODELS)
        if model.strip()
    ]
    errors_seen: list[str] = []
    for candidate in models:
        try:
            logger.info(f"Gemini quota exhausted; trying Groq model: {candidate}")
            response = httpx.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": candidate,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a precise assistant. Answer with "
                                "STRICT JSON only, no Markdown, no prose."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120.0,
            )
            if response.status_code in (200, 201):
                payload = response.json()
                content = (
                    payload.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content:
                    return content
                errors_seen.append(f"{candidate}: empty response")
                continue
            errors_seen.append(f"{candidate}: HTTP {response.status_code} {response.text[:200]}")
            logger.warning(errors_seen[-1])
        except httpx.HTTPError as e:
            errors_seen.append(f"{candidate}: {e}")
            logger.warning(errors_seen[-1])

    raise RuntimeError(
        "All Gemini models failed and Groq fallback is exhausted "
        f"(tried {len(models)} model(s): {'; '.join(errors_seen) or 'none available'})."
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
