"""Offline decoder correctness; no corpus or training run."""

from dataclasses import asdict
import json
import unittest

import torch
from torch import nn

from transformer_training.config import ModelConfig
from transformer_training.evaluation import next_token_loss
from transformer_training.model import DecoderOnlyTransformer


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = ModelConfig(vocab_size=12, sequence_length=5, layers=2,
                                  d_model=12, heads=3, d_ff=24, dropout=0)
        self.model = DecoderOnlyTransformer(self.config)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_shapes_tying_parameter_count_and_serialisation(self):
        self.assertEqual(self.model(torch.tensor([[2, 3, 4]])).shape, (1, 3, 12))
        self.assertEqual(self.model(torch.tensor([[2]])).shape, (1, 1, 12))
        self.assertEqual(self.model(torch.ones(2, 5, dtype=torch.long)).shape, (2, 5, 12))
        self.assertIs(self.model.token_embedding.weight, self.model.lm_head.weight)
        self.assertEqual(ModelConfig(**json.loads(json.dumps(asdict(self.config)))), self.config)
        with torch.device("meta"):
            default = DecoderOnlyTransformer(ModelConfig())
        self.assertEqual(sum(p.numel() for p in default.parameters()), 33691776)
        for module in self.model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                self.assertTrue((module.bias == 0).all())
            if isinstance(module, nn.LayerNorm):
                self.assertTrue((module.weight == 1).all())
                self.assertTrue((module.bias == 0).all())

    def test_causality_and_padding_isolation(self):
        self.model.eval()
        a = torch.tensor([[2, 3, 4, 5, 6]])
        b = torch.tensor([[2, 3, 9, 10, 11]])
        torch.testing.assert_close(self.model(a)[:, :2], self.model(b)[:, :2], rtol=0, atol=0)
        mask = torch.tensor([[True, True, False, False, False]])
        padded = self.model(b, mask)
        self.assertTrue(torch.isfinite(padded).all())
        torch.testing.assert_close(padded[:, :2], self.model(a, mask)[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(padded[:, :2], self.model(b[:, :2]), atol=1e-7, rtol=1e-6)

    def test_loss_alignment_and_gradients(self):
        ids = torch.tensor([[2, 3, 4, 1, 1], [5, 1, 1, 1, 1]])
        labels = torch.tensor([[3, 4, 1, -100, -100], [1, -100, -100, -100, -100]])
        logits = self.model(ids, labels != -100)
        logits.retain_grad()
        loss = next_token_loss(logits, labels)
        expected = -logits.log_softmax(-1)[labels != -100].gather(
            1, labels[labels != -100, None]).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad[labels == -100].abs().sum().item(), 0)
        self.assertGreater(logits.grad[..., 1][labels == 1].abs().sum().item(), 0)
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.blocks[0].attention.qkv.weight.grad.abs().sum(), 0)

    def test_invalid_configurations_and_inputs(self):
        for options in ({"heads": 5}, {"layers": 0}, {"dropout": 1},
                        {"dropout": float("nan")}, {"d_ff": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ModelConfig(**options)
        for ids in (torch.ones(5, dtype=torch.long), torch.ones(1, 2),
                    torch.ones(0, 2, dtype=torch.long), torch.ones(1, 0, dtype=torch.long),
                    torch.ones(1, 6, dtype=torch.long)):
            with self.subTest(shape=ids.shape), self.assertRaises(ValueError):
                self.model(ids)
        ids = torch.ones(1, 3, dtype=torch.long)
        for mask in (torch.tensor([[False, True, True]]), torch.tensor([[True, False, True]]),
                     torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool),
                     torch.ones(1, 3)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.model(ids, mask)


if __name__ == "__main__":
    unittest.main()
