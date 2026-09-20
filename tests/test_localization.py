import unittest

from youtube_reels.localization import (
    citation_text,
    fallback_source_name,
    get_card_strings,
    normalize_lang,
    script_length_guideline,
)


class LocalizationTests(unittest.TestCase):
    def test_normalize_lang(self) -> None:
        self.assertEqual(normalize_lang("pt"), "pt")
        self.assertEqual(normalize_lang("pt-PT"), "pt")
        self.assertEqual(normalize_lang("Português"), "pt")
        self.assertEqual(normalize_lang("en"), "en")
        self.assertEqual(normalize_lang("English"), "en")
        self.assertEqual(normalize_lang("zh"), "zh")
        self.assertEqual(normalize_lang("繁體中文"), "zh")
        self.assertEqual(normalize_lang("unknown"), "en")

    def test_citation_text(self) -> None:
        self.assertIn("Segundo apurado por Canal X", citation_text("pt", "Canal X"))
        self.assertIn("According to BBC's reporting", citation_text("en", "BBC"))
        self.assertIn("根據 中央社 報導", citation_text("zh", "中央社"))

    def test_fallback_source_name(self) -> None:
        self.assertEqual(fallback_source_name("pt"), "a fonte original")
        self.assertEqual(fallback_source_name("en"), "the original source")
        self.assertEqual(fallback_source_name("zh"), "該影片來源")

    def test_card_strings(self) -> None:
        pt_cards = get_card_strings("pt")
        self.assertEqual(pt_cards.source_label, "Fonte:")
        self.assertEqual(pt_cards.cta_question, "Qual é a sua opinião?")

        en_cards = get_card_strings("en")
        self.assertEqual(en_cards.source_label, "Source:")

        zh_cards = get_card_strings("zh")
        self.assertEqual(zh_cards.source_label, "資料來源：")

    def test_script_length_guideline(self) -> None:
        self.assertIn("words", script_length_guideline("pt"))
        self.assertIn("characters", script_length_guideline("zh"))
