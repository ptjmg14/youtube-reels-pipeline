import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from youtube_reels.llm_utils import (
    GROQ_FALLBACK_MODELS,
    MODEL_FALLBACKS,
    RETRY_ROUNDS,
    RETRY_SLEEP_S,
    _generate_with_groq,
)
from youtube_reels.pipeline import _groq_fallback


def _http_response(status_code: int, json_payload=None, text: str = "") -> object:
    response = SimpleNamespace(status_code=status_code, text=text)
    response.json = lambda: json_payload or {}
    return response


class GroqFallbackTests(unittest.TestCase):
    def test_groq_retries_next_model_on_429(self) -> None:
        """A 429 on the first model must fall through to the second one."""
        first = _http_response(429, text="rate limited")
        second = _http_response(
            200,
            {"choices": [{"message": {"content": '{"status":"PASS","reason":"None"}'}}]},
        )
        with patch("youtube_reels.llm_utils.httpx.post", side_effect=[first, second]) as post:
            result = _generate_with_groq("prompt", "key", ["m1", "m2"])
        self.assertEqual(result, '{"status":"PASS","reason":"None"}')
        self.assertEqual(post.call_count, 2)
        models_sent = [call.kwargs["json"]["model"] for call in post.call_args_list]
        self.assertEqual(models_sent, ["m1", "m2"])

    def test_groq_all_models_fail_raises(self) -> None:
        with (
            patch(
                "youtube_reels.llm_utils.httpx.post",
                side_effect=[_http_response(429, text="429")] * 3,
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            _generate_with_groq("prompt", "key", ["a", "b", "c"])
        self.assertIn("Groq fallback is exhausted", str(ctx.exception))

    def test_groq_uses_defaults_when_no_models(self) -> None:
        with patch("youtube_reels.llm_utils.httpx.post") as post:
            post.return_value = _http_response(
                200, {"choices": [{"message": {"content": "ok"}}]}
            )
            result = _generate_with_groq("prompt", "key", None)
        self.assertEqual(result, "ok")
        self.assertEqual(post.call_args_list[0].kwargs["json"]["model"], GROQ_FALLBACK_MODELS[0])

    def test_fallback_wiring_respects_settings(self) -> None:
        settings = MagicMock()
        settings.llm_fallback = "groq"
        settings.groq_api_key = "key"
        settings.groq_rewrite_models = ("llama-3.3-70b-versatile", "openai/gpt-oss-120b")
        kwargs = _groq_fallback(settings)
        self.assertEqual(kwargs["groq_api_key"], "key")
        self.assertEqual(kwargs["groq_models"], ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"])

        settings.llm_fallback = "off"
        self.assertEqual(_groq_fallback(settings), {})

        settings.llm_fallback = "groq"
        settings.groq_api_key = None
        self.assertEqual(_groq_fallback(settings), {})

    def test_generate_with_fallback_uses_groq_after_gemini_exhausted(self) -> None:
        """An exhausted Gemini key must delegate to the Groq fallback."""
        from google.genai import errors

        def gemini_error(*args, **kwargs):
            raise errors.ClientError(429, {"error": {"message": "Resource has been exhausted"}})

        groq_response = _http_response(
            200, {"choices": [{"message": {"content": "fallback text"}}]}
        )
        with patch("youtube_reels.llm_utils.Client") as client_cls:
            client_cls.return_value.models.generate_content.side_effect = gemini_error
            with patch("youtube_reels.llm_utils.httpx.post", return_value=groq_response) as post:
                from youtube_reels.llm_utils import generate_with_fallback

                result = generate_with_fallback(
                    "gemini-key",
                    "prompt",
                    model="gemini-test",
                    groq_api_key="groq-key",
                    groq_models=["llama-3.3-70b-versatile"],
                )
        self.assertEqual(result, "fallback text")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(
            client_cls.return_value.models.generate_content.call_count,
            1 + len(MODEL_FALLBACKS),
            msg="Quota exhaustion must skip the extra retry rounds and not re-try Gemini",
        )

    def test_transient_503_keeps_retry_rounds_before_groq(self) -> None:
        """Server errors (503) are transient and justify the full retry rounds."""
        from google.genai import errors

        busy = errors.ServerError(503, {"error": {"message": "unavailable"}})
        total_models = 1 + len(MODEL_FALLBACKS)

        with (
            patch("youtube_reels.llm_utils.Client") as client_cls,
            patch("youtube_reels.llm_utils.time") as mock_time,
            patch("youtube_reels.llm_utils.httpx.post") as post,
        ):
            from youtube_reels.llm_utils import generate_with_fallback

            attempts = {"n": 0}

            def gemini_busy(*args, **kwargs):
                attempts["n"] += 1
                if attempts["n"] <= total_models * (RETRY_ROUNDS - 1):
                    raise busy
                return SimpleNamespace(text="ok")

            client_cls.return_value.models.generate_content.side_effect = gemini_busy
            result = generate_with_fallback(
                "gemini-key",
                "prompt",
                model="gemini-test",
                groq_api_key="groq-key",
                groq_models=["llama-3.3-70b-versatile"],
            )
        self.assertEqual(result, "ok")
        self.assertEqual(mock_time.sleep.call_count, RETRY_ROUNDS - 1)
        mock_time.sleep.assert_called_with(RETRY_SLEEP_S)
        post.assert_not_called()