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

    def test_heuristic_sliding_window_does_not_skip_boundaries(self) -> None:
        """Sliding window with 50 % step must not discard boundary segments."""
        # 30 x 1-second segments → greedy packer would skip every other window
        transcript = [TranscriptSegment(i, i + 1, f"seg {i}") for i in range(60)]
        windows = HeuristicPrefilter().suggest(transcript, top_n=100)
        # All segment content should appear in at least one window
        covered = set()
        for w in windows:
            for seg in transcript:
                if seg.start >= w.start and seg.end <= w.end:
                    covered.add(seg.start)
        # At least 90 % of segments covered (allow minor edge effects)
        self.assertGreaterEqual(len(covered), 54)

    def test_heuristic_scores_numeric_windows_higher(self) -> None:
        """Windows with numbers/percentages should rank above plain text."""
        from youtube_reels.analyzers.heuristic import HeuristicPrefilter as HP

        dense = "月付暴增30%！從1.5萬元跳升至4.5萬元，差距高達三倍！"
        bland = "今天天氣很好，我們聊聊生活上的一些事情吧。"
        score_dense = HP._score(dense)
        score_bland = HP._score(bland)
        self.assertGreater(score_dense, score_bland)

    def test_heuristic_top_n_cap_respected(self) -> None:
        """suggest() should return no more than top_n windows."""
        long_transcript = [TranscriptSegment(i * 2, i * 2 + 2, f"seg {i}") for i in range(200)]
        windows = HeuristicPrefilter().suggest(long_transcript, top_n=10)
        self.assertLessEqual(len(windows), 10)

    def test_heuristic_windows_chronologically_ordered(self) -> None:
        """Output should be in temporal order even after score-based ranking."""
        transcript = [TranscriptSegment(i * 2, i * 2 + 2, "text") for i in range(50)]
        windows = HeuristicPrefilter().suggest(transcript)
        starts = [w.start for w in windows]
        self.assertEqual(starts, sorted(starts))



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