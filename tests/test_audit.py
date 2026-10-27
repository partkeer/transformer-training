"""Small in-memory text fixtures for full-split leakage exclusion."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from transformer_training.audit import audit_overlap
from transformer_training.data import create_dataloaders
from test_data import fixture_tokenizer


def splits():
    return DatasetDict({
        "train": Dataset.from_dict({"text": ["a b", "c d", "a b", "e", "e", "", " g h "]}),
        "validation": Dataset.from_dict({"text": ["a b", "a b", "g h", " "]}),
    })


class AuditTests(unittest.TestCase):
    def test_full_audit_counts_exactness_cache_and_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = splits()
            result = audit_overlap(source, directory)
            self.assertEqual(result["removed_train_rows"], [0, 2])
            self.assertEqual(result["shared_distinct_texts"], 1)
            self.assertEqual(result["splits"]["train"]["duplicate_rows_beyond_first"], 2)
            self.assertEqual(result["splits"]["validation"]["duplicate_rows_beyond_first"], 1)
            self.assertEqual(result["splits"]["train"]["blank_stories"], 1)
            with patch("transformer_training.audit.hashlib.sha256", wraps=__import__(
                    "hashlib").sha256) as digest:
                self.assertEqual(audit_overlap(source, directory), result)
                self.assertEqual(digest.call_count, 1)  # Cache key only, no story hashing.
            cleaned = create_dataloaders(source, fixture_tokenizer(), sequence_length=3,
                                         exclude_validation_overlap=True, audit_cache_dir=directory)
            self.assertEqual(cleaned.metadata["audit"], result)
            self.assertEqual(list(cleaned.train.dataset["source_row"]), [1, 3, 4, 6])
            self.assertEqual(list(cleaned.validation.dataset["source_row"]), [0, 1, 2])
            sampled = create_dataloaders(source, fixture_tokenizer(), sequence_length=3,
                                         exclude_validation_overlap=True, max_stories=3,
                                         audit_cache_dir=directory)
            selected = sampled.metadata["splits"]["train"]["selected_row_indices"]
            self.assertTrue(set(selected).issubset({1, 3, 4, 5, 6}))
            self.assertEqual(set(sampled.train.dataset["source_row"]), set(selected) - {5})
            self.assertEqual(sampled.metadata["audit"]["removed_train_stories"], 2)
            self.assertEqual(sampled.metadata["audit"]["splits"]["train"]["stories"], 7)
            changed = splits()
            changed["validation"] = Dataset.from_dict({"text": ["c d"]})
            self.assertEqual(audit_overlap(changed, directory)["removed_train_rows"], [1])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_default_retains_overlap_and_opt_in_fails_if_no_usable_train(self):
        source = splits()
        unchanged = create_dataloaders(source, fixture_tokenizer(), sequence_length=3)
        self.assertNotIn("audit", unchanged.metadata)
        self.assertIn(0, list(unchanged.train.dataset["source_row"]))
        source["train"] = Dataset.from_dict({"text": ["a b", "", "a b"]})
        with self.assertRaisesRegex(ValueError, "no usable stories"):
            create_dataloaders(source, fixture_tokenizer(), exclude_validation_overlap=True)
        source["train"] = Dataset.from_dict({"text": ["a b"]})
        with self.assertRaisesRegex(ValueError, "no usable stories"):
            create_dataloaders(source, fixture_tokenizer(), exclude_validation_overlap=True)

    def test_invalid_text_is_rejected_even_outside_sample(self):
        source = splits()
        source["validation"] = Dataset.from_dict({"text": ["a b", None]})
        with self.assertRaisesRegex(ValueError, "string text"):
            create_dataloaders(source, fixture_tokenizer(), max_stories=1,
                               exclude_validation_overlap=True)


if __name__ == "__main__":
    unittest.main()
