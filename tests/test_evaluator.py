import unittest
from unittest.mock import MagicMock, patch

from youtube_reels.analyzers.evaluator import EvalError, Evaluator
from youtube_reels.models import ChartSpec, RenderedScript, SegmentWindow, VideoAsset
from youtube_reels.pipeline import _evaluate


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Evaluator(api_key="fake-key", model="gemini-test", language="pt")

    def test_eval_prompt_includes_language_and_chart_rules(self) -> None:
        script = RenderedScript(
            number=1,
            hook="Hook teste",
            title="Título teste",
            narration="Segundo apurado pelo Canal, o PIB cresceu 3% este ano.",
            citation="Segundo apurado pelo Canal",
            reason="Momento chave",
            chart=ChartSpec(
                title="Crescimento", kind="bar", x_labels=["2023"], values=[3.0], unit="%"
            ),
            start=10.0,
            end=30.0,
            source_text="O PIB cresceu 3% este ano de acordo com os dados.",
        )
        prompt = self.evaluator._eval_prompt(script, "Canal")
        self.assertIn("Target Language: pt", prompt)
        self.assertIn("Original Source Text:", prompt)
        self.assertIn("grounded in the Original Source Text", prompt)
        self.assertIn("NOT required to recite every single data point", prompt)
        self.assertIn("Official Source Name:", prompt)
        self.assertIn('"Canal"', prompt)

    def test_parse_evaluation_variations(self) -> None:
        # Standard PASS
        res = self.evaluator._parse_evaluation('{"status": "PASS", "reason": "None"}')
        self.assertEqual(res["status"], "PASS")

        # Markdown wrapped
        res_md = self.evaluator._parse_evaluation('```json\n{"status": "pass", "reason": "ok"}\n```')
        self.assertEqual(res_md["status"], "pass")

        # Invalid JSON
        with self.assertRaises(EvalError):
            self.evaluator._parse_evaluation("Invalid output from model")

    @patch("youtube_reels.analyzers.evaluator.Evaluator.evaluate")
    @patch("youtube_reels.analyzers.rewriter.Rewriter.rewrite")
    def test_evaluate_pipeline_passes_first_time(self, mock_rewrite, mock_eval) -> None:
        mock_eval.return_value = True
        script = RenderedScript(
            number=1, hook="h", title="t", narration="n", citation="c",
            reason="r", chart=None, start=0.0, end=10.0, source_text="s"
        )
        settings = MagicMock()
        settings.gemini_api_key = "fake"
        settings.gemini_model = "fake-model"
        settings.source_name = "Channel"
        settings.output_language = "pt"

        asset = VideoAsset("id", "Title", "url", MagicMock(), None, None, "Channel")
        results = _evaluate([script], [SegmentWindow(0, 10, "s")], asset, settings)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].narration, "n")
        self.assertEqual(mock_eval.call_count, 1)
        mock_rewrite.assert_not_called()

    @patch("youtube_reels.analyzers.evaluator.Evaluator.evaluate")
    @patch("youtube_reels.analyzers.rewriter.Rewriter.rewrite")
    def test_evaluate_pipeline_retries_with_feedback(self, mock_rewrite, mock_eval) -> None:
        # First attempt fails, second attempt passes
        mock_eval.side_effect = [EvalError("Citação ausente"), True]

        refined_script = RenderedScript(
            number=1, hook="h", title="t", narration="n_corrigida", citation="c_nova",
            reason="r", chart=None, start=0.0, end=10.0, source_text="s"
        )
        mock_rewrite.return_value = [refined_script]

        original_script = RenderedScript(
            number=1, hook="h", title="t", narration="n_antiga", citation="c_velha",
            reason="r", chart=None, start=0.0, end=10.0, source_text="s"
        )
        settings = MagicMock()
        settings.gemini_api_key = "fake"
        settings.gemini_model = "fake-model"
        settings.source_name = "Channel"
        settings.output_language = "pt"

        asset = VideoAsset("id", "Title", "url", MagicMock(), None, None, "Channel")
        results = _evaluate([original_script], [SegmentWindow(0, 10, "s")], asset, settings)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].narration, "n_corrigida")
        self.assertEqual(results[0].citation, "c_nova")
        self.assertEqual(mock_eval.call_count, 2)
        mock_rewrite.assert_called_once()
        # Ensure feedback and previous narration were passed
        _, kwargs = mock_rewrite.call_args
        self.assertIn("Citação ausente", kwargs.get("feedback", ""))
        self.assertEqual(kwargs.get("previous_script"), "n_antiga")
