import unittest

from youtube_reels.vectordb import GeminiEmbeddingFunction


class VectorDbTests(unittest.TestCase):
    def test_embedding_adapter_retains_configuration(self) -> None:
        function = GeminiEmbeddingFunction("key", "text-embedding-004")
        self.assertEqual(function.model, "text-embedding-004")
