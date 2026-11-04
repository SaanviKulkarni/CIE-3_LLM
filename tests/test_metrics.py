import unittest

from evaluation.evaluate import cer, describe_errors, diff_type, edit_distance, word_accuracy


class MetricsTests(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(edit_distance('abc', 'axc'), 1)
        self.assertEqual(edit_distance('', 'abc'), 3)

    def test_cer(self):
        self.assertEqual(cer('राम', 'राम'), 0)
        self.assertEqual(cer('राम', 'रा'), 1 / 3)

    def test_word_accuracy(self):
        self.assertEqual(word_accuracy('यह सही है', 'यह सही है'), 1)
        self.assertEqual(word_accuracy('यह सही है', 'यह गलत है'), 2 / 3)

    def test_diff_labels(self):
        self.assertEqual(diff_type('हूँ', 'हूं'), 'chandrabindu/anusvara (ँ vs ं)')
        self.assertEqual(diff_type('कहीं', 'कहिं'), 'vowel length (इ/ई, उ/ऊ)')
        self.assertEqual(describe_errors('मैं ठीक हूँ', 'मैं ठीक')[0], 'word segmentation/count')
