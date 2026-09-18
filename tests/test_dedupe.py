import unittest

from youtube_reels.dedupe import jaccard, normalize, shingles, transcript_chunks, transcript_text
from youtube_reels.models import TranscriptSegment


class DedupeTests(unittest.TestCase):
    def test_normalize_drops_punctuation_and_whitespace(self) -> None:
        self.assertEqual(normalize("Hello, World! 你好！"), "helloworld你好")

    def test_shingles_match_duplicated_text(self) -> None:
        original = shingles("根據報導，這個利率今天上升了大約0.5")
        reupload = shingles("根據報導，這個利率今天上升了大約 0.5。")
        self.assertGreaterEqual(jaccard(original, reupload), 0.9)

    def test_jaccard_unrelated_text_is_low(self) -> None:
        finance = shingles("金融市場的利率持續上升")
        weather = shingles("台灣的天氣預報顯示明天會下雨")
        self.assertLess(jaccard(finance, weather), 0.5)

    def test_transcript_text_and_chunks(self) -> None:
        segments = [
            TranscriptSegment(0, 1, "你好"),
            TranscriptSegment(1, 2, "世界"),
        ]
        self.assertEqual(transcript_text(segments), "你好 世界")
        chunks = transcript_chunks(segments, size=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("你好", chunks[0])