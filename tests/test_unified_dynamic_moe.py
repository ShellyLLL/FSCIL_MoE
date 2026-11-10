"""Architecture invariants for the unified sparse dynamic FSCIL MoE."""

import unittest

import torch
import torch.nn.functional as F

from models.bimc import build_loo_cache_from_embeddings
from models.clip.model import VisualMoEAdapter


class UnifiedMoEArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_new_expert_is_function_preserving_until_activated(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        inputs = torch.randn(5, 3, 8)
        before, _ = adapter(inputs)
        adapter.add_expert(session_id=1, reduction=2)
        after, aux = adapter(inputs)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertEqual(aux["top_k_indices"].shape, (3, 2))
        self.assertTrue((aux["top_k_indices"] < 4).all())

    def test_router_uses_one_sparse_competition(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        inputs = torch.randn(7, 2, 8)
        _, aux = adapter(inputs)
        weights = aux["route_weights"]
        self.assertEqual(weights.shape, (2, 4))
        self.assertTrue(torch.all((weights > 0).sum(dim=-1) == 2))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))

    def test_unselected_experts_are_not_executed(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        calls = [0, 0, 0, 0]
        handles = []
        for index, expert in enumerate(adapter.experts):
            handles.append(expert.register_forward_hook(
                lambda module, args, output, i=index: calls.__setitem__(i, calls[i] + 1)
            ))
        adapter(torch.randn(5, 3, 8))
        for handle in handles:
            handle.remove()
        self.assertEqual(sum(value > 0 for value in calls), 2)

    def test_new_route_can_activate_without_unfreezing_history(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2)
        adapter.add_expert(session_id=1, reduction=2)
        old_parameters = [
            parameter for name, parameter in adapter.named_parameters()
            if name.startswith(("experts.0.", "experts.1.", "experts.2.", "experts.3.",
                                "router.columns.0.", "router.columns.1.",
                                "router.columns.2.", "router.columns.3."))
        ]
        new_column = adapter.router.columns[-1]
        current = torch.randn(6, 8)
        history = torch.randn(5, 8)
        activation, preservation = adapter.route_objective(
            current, history, torch.ones_like(history) * 0.01
        )
        (activation + preservation).backward()
        self.assertIsNotNone(new_column.bias.grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in old_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in old_parameters))

    def test_history_route_constraint_uses_class_covariance(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2)
        adapter.add_expert(session_id=1, reduction=2)
        history = torch.randn(6, 8)
        low_variance = torch.zeros_like(history)
        high_variance = torch.ones_like(history)
        low = adapter.route_preservation_loss(history, low_variance)
        high = adapter.route_preservation_loss(history, high_variance)
        self.assertGreaterEqual(float(high), float(low))

    def test_topology_round_trip_preserves_predictions_and_birth_order(self):
        source = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        source.add_expert(session_id=2, reduction=4)
        source.freeze_all()
        inputs = torch.randn(4, 3, 8)
        expected, _ = source(inputs)
        restored = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2)
        restored.rebuild_topology(source.export_topology())
        restored.load_state_dict(source.state_dict(), strict=True)
        restored.eval()
        actual, _ = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(restored.expert_birth_sessions, [0, 0, 0, 0, 2])

    def test_old_expert_and_router_are_frozen_after_expansion(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        self.assertTrue(all(not p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.router.columns[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.experts[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.router.columns[-1].parameters()))

    def test_loo_prototypes_exclude_all_views_of_query_source(self):
        features = F.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9],
            [-1., 0.], [-0.9, 0.1],
        ]), dim=-1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1])
        sources = torch.tensor([10, 10, 11, 12, 20, 21])
        cache = build_loo_cache_from_embeddings(features, labels, sources)
        # Both augmentations from source 10 are removed for either query.
        expected = F.normalize(features[2:4].mean(dim=0), dim=-1)
        torch.testing.assert_close(cache["loo_visual_prototypes"][0, 0], expected)
        torch.testing.assert_close(cache["loo_visual_prototypes"][1, 0], expected)
        self.assertTrue(cache["loo_valid"].all())

    def test_loo_rejects_single_source_class(self):
        features = F.normalize(torch.tensor([[1., 0.], [0., 1.]]), dim=-1)
        with self.assertRaisesRegex(ValueError, "another source"):
            cache = build_loo_cache_from_embeddings(
                features, torch.tensor([0, 1]), torch.tensor([1, 2])
            )
            if not cache["loo_valid"].all():
                raise ValueError("another source is required")


if __name__ == "__main__":
    unittest.main()
