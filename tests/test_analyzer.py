import json
import unittest

from youtube_reels.analyzers.heuristic import HeuristicPrefilter
from youtube_reels.analyzers.rewriter import RewriteError, _parse_scripts
from youtube_reels.models import SegmentWindow, TranscriptSegment


def _transcript() -> list[TranscriptSegment]:
    return [TranscriptSegment(i, i + 1, "texto de exemplo para a janela") for i in range(30)]


class AnalyzerTests(unittest.TestCase):
    def test_heuristic_windows_are_coherent(self) -> None:
        windows = HeuristicPrefilter().suggest(_transcript())
        self.assertGreaterEqual(len(windows), 1)
        for window in windows:
            self.assertGreaterEqual(window.end - window.start, 15)

    def test_heuristic_empty_transcript_gives_no_windows(self) -> None:
        self.assertEqual(HeuristicPrefilter().suggest([]), [])

    def test_parse_scripts_selects_clip_with_timing_and_source_text(self) -> None:
        windows = [
            SegmentWindow(start=0.0, end=55.0, text="primeira parte"),
            SegmentWindow(start=55.0, end=110.0, text="segunda parte"),
        ]
        raw = json.dumps(
            [
                {
                    "start": 50,
                    "end": 60,
                    "hook": "h",
                    "title": "t",
                    "narration": "narração com citação",
                    "citation": "c",
                    "chart": None,
                }
            ]
        )
        scripts = _parse_scripts(raw, max_clips=2, windows=windows)
        self.assertEqual(len(scripts), 1)
        self.assertEqual(scripts[0].start, 50.0)
        self.assertEqual(scripts[0].end, 60.0)
        self.assertEqual(scripts[0].source_text, "primeira parte segunda parte")
        self.assertEqual(scripts[0].citation, "c")

    def test_parse_scripts_rejects_more_than_max_clips(self) -> None:
        raw = json.dumps(
            [
                {"start": 0, "end": 5, "hook": "h", "title": "t", "narration": "a", "citation": "c"},
                {"start": 6, "end": 9, "hook": "h", "title": "t", "narration": "b", "citation": "c"},
            ]
        )
        with self.assertRaises(RewriteError):
            _parse_scripts(raw, max_clips=1, windows=[SegmentWindow(0, 55, "x")])

    def test_gemini_json_shape(self) -> None:
        from youtube_reels.analyzers.rewriter import _parse_chart

        chart = _parse_chart({"title": "Taxa", "kind": "bar", "x_labels": ["a"], "values": [1]})
        self.assertEqual(chart.title, "Taxa")
        self.assertEqual(chart.values, [1.0])