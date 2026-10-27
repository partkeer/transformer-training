"""Offline correctness tests with a small, deterministic tokenizer and corpus."""

import unittest
from types import SimpleNamespace

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
import torch
import torch.nn.functional as F

from transformer_training.data import (
    IGNORE_INDEX, NextTokenCollator, _check_dataset_revision,
    create_dataloaders, tokenise_stories,
)


def fixture_tokenizer():
    tokenizer = Tokenizer(models.WordLevel(
        {"<unk>": 0, "<|endoftext|>": 1,
         **{letter: i + 2 for i, letter in enumerate("abcdefghij")}},
        unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def fixture_splits():
    return DatasetDict({
        "train": Dataset.from_dict({"text": ["a b c d e", "f", "", "  "]}),
        "validation": Dataset.from_dict({"text": ["g h", "i j"]}),
    })


class NextTokenTests(unittest.TestCase):
    def test_windows_cover_every_pair_without_crossing_stories(self):
        # Includes an exact multiple, a short tail, a one-token story and blanks.
        texts = ["a b c d", "e f g h i", "j", "", "  "]
        tokenizer = fixture_tokenizer()
        result = tokenise_stories({"text": texts}, list(range(5)),
                                  tokenizer=tokenizer, sequence_length=2, eos_id=1)
        for index, text in enumerate(texts[:3]):
            tokens = tokenizer.encode(text).ids + [1]
            windows = [w for w, row in zip(result["tokens"], result["source_row"])
                       if row == index]
            pairs = [(a, b) for w in windows for a, b in zip(w, w[1:])]
            self.assertEqual(pairs, list(zip(tokens, tokens[1:])))
            self.assertTrue(all(2 <= len(w) <= 3 for w in windows))
        self.assertNotIn(3, result["source_row"])
        self.assertNotIn(4, result["source_row"])

    def test_padding_eos_loss_and_gradient(self):
        batch = NextTokenCollator(4, 1)([
            {"tokens": [2, 3, 4, 5, 1]}, {"tokens": [6, 1]}])
        self.assertEqual(batch["input_ids"].tolist(), [[2, 3, 4, 5], [6, 1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[3, 4, 5, 1], [1, -100, -100, -100]])
        self.assertEqual(batch["attention_mask"].sum().item(), 5)
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["labels"].dtype, torch.long)
        logits = torch.randn(2, 4, 12, requires_grad=True)
        labels = batch["labels"]
        loss = F.cross_entropy(logits.reshape(-1, 12), labels.reshape(-1))
        valid = labels != IGNORE_INDEX
        expected = F.cross_entropy(logits[valid], labels[valid])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[valid].abs().sum().item(), 0)
        self.assertEqual(logits.grad[~valid].abs().sum().item(), 0)

    def test_sequence_length_one(self):
        result = tokenise_stories({"text": ["a b"]}, [0],
                                  tokenizer=fixture_tokenizer(), sequence_length=1, eos_id=1)
        self.assertEqual(result["tokens"], [[2, 3], [3, 1]])


class LoaderTests(unittest.TestCase):
    def make(self, **options):
        return create_dataloaders(fixture_splits(), fixture_tokenizer(),
                                  sequence_length=2, batch_size=3, **options)

    def test_splits_shapes_validation_order_and_partial_batches(self):
        loaders = self.make()
        self.assertEqual(len(loaders.train.dataset), 4)
        self.assertEqual(len(loaders.validation.dataset), 2)
        self.assertEqual([len(batch["labels"]) for batch in loaders.train], [3, 1])
        first = list(loaders.validation)
        second = list(loaders.validation)
        self.assertEqual(first[0]["input_ids"].tolist(), [[8, 9], [10, 11]])
        torch.testing.assert_close(first[0]["input_ids"], second[0]["input_ids"])
        self.assertEqual(first[0]["input_ids"].shape, (2, 2))
        self.assertEqual(first[0]["input_ids"].device.type, "cpu")

    def test_shuffling_is_reproducible_across_epochs_and_independent_of_global_rng(self):
        a, b = self.make(seed=7), self.make(seed=7)
        for _ in range(2):
            torch.rand(100)  # This must not affect either loader's dedicated RNG.
            order_a = torch.cat([batch["input_ids"] for batch in a.train])
            order_b = torch.cat([batch["input_ids"] for batch in b.train])
            torch.testing.assert_close(order_a, order_b)
        # Different seeds can legitimately collide on a four-element permutation.
        self.assertEqual(self.make(seed=8).train.generator.initial_seed(), 8)

    def test_sample_provenance_and_tokenizer_settings(self):
        tokenizer = fixture_tokenizer()
        tokenizer.enable_truncation(max_length=1)
        tokenizer.enable_padding(length=20, pad_id=1)
        splits = DatasetDict({name: Dataset.from_dict({"text": ["a b c"] * 20})
                              for name in ("train", "validation")})
        loaders = create_dataloaders(splits, tokenizer, max_stories=4,
                                     sequence_length=2)
        meta = loaders.metadata["splits"]["train"]
        self.assertEqual(meta["selected_stories"], 4)
        self.assertEqual(meta["windows"], 8)  # Caller truncation is disabled on the copy.
        self.assertEqual(sorted(set(loaders.train.dataset["source_row"])),
                         meta["selected_row_indices"])
        self.assertIsNotNone(tokenizer.truncation)  # Original instance is unchanged.

    def test_worker_process_matches_single_process(self):
        single, multi = self.make(), self.make(num_workers=2)
        for a, b in zip(single.train, multi.train):
            for key in a:
                torch.testing.assert_close(a[key], b[key])

    def test_errors_for_empty_invalid_and_missing_data(self):
        empty = fixture_splits()
        empty["train"] = Dataset.from_dict({"text": ["", "  "]})
        with self.assertRaisesRegex(ValueError, "no usable stories"):
            create_dataloaders(empty, fixture_tokenizer())
        for text in (None, 123):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "string text"):
                tokenise_stories({"text": [text]}, [0], tokenizer=fixture_tokenizer(),
                                 sequence_length=2, eos_id=1)
        with self.assertRaisesRegex(ValueError, "train and validation"):
            create_dataloaders({}, fixture_tokenizer())
        missing_text = fixture_splits()
        missing_text["train"] = Dataset.from_dict({"other": ["a"]})
        with self.assertRaisesRegex(ValueError, "no text column"):
            create_dataloaders(missing_text, fixture_tokenizer())
        for options in ({"sequence_length": 0}, {"batch_size": 0},
                        {"num_workers": -1}, {"max_stories": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_dataloaders(fixture_splits(), fixture_tokenizer(), **options)

    def test_cached_revision_must_match_recorded_revision(self):
        def cached(revision):
            return SimpleNamespace(cache_files=[
                {"filename": f"cache/default/0.0.0/{revision}/stories.arrow"}])

        splits = {name: cached("expected") for name in ("train", "validation")}
        _check_dataset_revision(splits, "expected")
        splits["validation"] = cached("different")
        with self.assertRaisesRegex(RuntimeError, "Cannot verify validation cache"):
            _check_dataset_revision(splits, "expected")
        splits["validation"] = SimpleNamespace(cache_files=[])
        with self.assertRaisesRegex(RuntimeError, "Cannot verify validation cache"):
            _check_dataset_revision(splits, "expected")


if __name__ == "__main__":
    unittest.main()
