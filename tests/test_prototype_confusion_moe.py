"""Architecture invariants for Prototype-Confusion Guided Dynamic MoE-FSCIL."""
import unittest
import torch
import torch.nn.functional as F

from models.prototype_confusion_moe import (
    UnifiedTopKRouter, DynamicLayerMoE, source_exclusive_means,
    prototype_confusion, make_apc_reference, balanced_risk,
)


class PrototypeConfusionTests(unittest.TestCase):
    def test_apc_uses_base_reference_percentiles_and_orders_layers(self):
        torch.manual_seed(4)
        base = F.normalize(torch.randn(12, 8), dim=-1)
        reference = make_apc_reference({2: base, 5: base})
        old = {2: base[:8], 5: base[:8]}
        labels = torch.tensor([10, 10, 11, 11])
        features = {
            2: torch.cat([base[:2], base[2:4]], dim=0),
            5: torch.cat([base[:2], -base[2:4]], dim=0),
        }
        scores = prototype_confusion(features, labels, [10, 11], old, reference)
        self.assertEqual(set(scores), {2, 5})
        for item in scores.values():
            self.assertTrue(torch.isfinite(item['scores']).all())
            self.assertGreaterEqual(item['mean'], 0.0)
            self.assertLessEqual(item['mean'], 1.0)

    def test_source_exclusion_removes_all_views_of_same_source(self):
        features = F.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]
        ]), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        sources = torch.tensor([7, 7, 8, 9])
        bank = source_exclusive_means(
            features, labels, sources, [0, 1], query_indices=[0, 2]
        )
        # Both views of source 7 are excluded for the first query.
        torch.testing.assert_close(bank[0, 0], features.new_tensor([0., 1.]))
        self.assertEqual(tuple(bank.shape), (2, 2, 2))

    def test_source_exclusive_banks_are_bounded_to_requested_queries(self):
        features = F.normalize(torch.randn(30, 12), dim=-1)
        labels = torch.arange(30) // 3
        sources = torch.arange(30)
        bank = source_exclusive_means(features, labels, sources, range(10),
                                      query_indices=[1, 4, 9])
        self.assertEqual(tuple(bank.shape), (3, 10, 12))

    def test_router_is_one_sparse_topk_softmax_over_all_experts(self):
        torch.manual_seed(2)
        router = UnifiedTopKRouter(6, 5)
        x = torch.randn(4, 6)
        logits, weights, selected = router(x, top_k=2)
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(weights.shape), (4, 5))
        self.assertTrue(torch.all((weights > 0).sum(1) == 2))
        torch.testing.assert_close(weights.sum(1), torch.ones(4))
        _, excluded_weights, _ = router(x, top_k=2, excluded=[4])
        self.assertTrue(torch.equal(excluded_weights[:, 4], torch.zeros(4)))
        self.assertTrue(torch.all((excluded_weights > 0).sum(1) == 2))

    def test_incremental_expert_is_appended_to_same_router(self):
        torch.manual_seed(3)
        layer = DynamicLayerMoE(8, 8, base_experts=3, base_reduction=2,
                               incremental_reduction=4, top_k=2, scale_init=0.5)
        self.assertEqual(layer.num_experts, 3)
        self.assertEqual(len(layer.router.columns), 3)
        new = torch.randn(5, 8) + 2.0
        old = torch.randn(20, 8) - 2.0
        layer.add_expert(1, new, old)
        self.assertEqual(layer.num_experts, 4)
        self.assertEqual(len(layer.router.columns), 4)
        trainable = [name for name, p in layer.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(('experts.3.', 'router.columns.3.'))
                            for name in trainable))
        layer.freeze_all()
        self.assertFalse(any(p.requires_grad for p in layer.parameters()))

    def test_balanced_risk_requires_every_query_label_in_global_bank(self):
        logits = torch.tensor([[4., 0., 0.], [0., 3., 0.], [0., 0., 2.]])
        labels = torch.tensor([10, 20, 30])
        risk, per_class = balanced_risk(logits, labels, [10, 20, 30])
        self.assertLess(float(risk), 0.2)
        self.assertEqual(set(per_class), {10, 20, 30})
        with self.assertRaises(ValueError):
            balanced_risk(logits, labels, [10, 20])


if __name__ == '__main__':
    unittest.main()
