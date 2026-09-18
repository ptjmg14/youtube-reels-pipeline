import tempfile
import unittest
from pathlib import Path

from youtube_reels.models import TranscriptSegment
from youtube_reels.subtitles import parse_vtt, write_srt


class SubtitleTests(unittest.TestCase):
    def test_vtt_to_segments_and_srt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vtt = root / "caption.vtt"
            vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nOlá mundo\n", encoding="utf-8")
            segments = parse_vtt(vtt)
            self.assertEqual(segments, [TranscriptSegment(1.0, 3.0, "Olá mundo")])
            srt = write_srt(segments, root / "caption.srt")
            self.assertIn("00:00:01,000 --> 00:00:03,000", srt.read_text(encoding="utf-8"))
