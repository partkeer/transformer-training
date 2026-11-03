import unittest

from scripts.explore_tinystories import analyse, describe


class AnalysisTests(unittest.TestCase):
    def test_missing_empty_duplicates_and_word_definition(self):
        """Check EDA distinguishes null, invalid, blank, and duplicate text and counts
        apostrophe-containing regex words separately from characters.
        """
        rows = [{"text": text} for text in
                (None, 12, "", "  ", "Lily can't fly.", "Lily can't fly.")]
        summary, hashes, chars, lengths = analyse(rows)
        self.assertEqual(summary["missing_or_null_text"], 1)
        self.assertEqual(summary["non_string_text"], 1)
        self.assertEqual(summary["empty_or_whitespace_text"], 2)
        self.assertEqual(summary["duplicate_rows_beyond_first"], 1)
        self.assertEqual(lengths, [0, 0, 3, 3])
        self.assertEqual(chars, [0, 2, 15, 15])
        self.assertEqual(summary["total_words"], 6)
        self.assertEqual(summary["unique_lowercase_words"], 3)
        self.assertEqual(len(hashes), 3)

    def test_empty_input_and_quantiles(self):
        """Verify empty analysis returns absent length statistics and a simple nonempty
        sequence produces the expected median and mean.
        """
        summary, hashes, chars, lengths = analyse([])
        self.assertIsNone(summary["words"])
        self.assertEqual(summary["analysed_rows"], 0)
        self.assertEqual(len(hashes), 0)
        self.assertEqual(describe([0, 2, 4, 6])["median"], 3)
        self.assertEqual(describe([0, 2, 4, 6])["mean"], 3)


if __name__ == "__main__":
    unittest.main()
