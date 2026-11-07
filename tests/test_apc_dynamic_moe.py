"""Tests for the unified raw-reference APC and independent expert gates."""
import copy
import unittest
import torch
import torch.nn.functional as F
from models.clip.model import Transformer, VisionTransformer, VisualMoEAdapter


class UnifiedMoETests(unittest.TestCase):
    def test_base_softmax_and_incremental_independent_sigmoid(self):
        torch.manual_seed(3)
        adapter = VisualMoEAdapter(8, num_experts=3, reduction=2)
        adapter.eval()
        x = torch.randn(5, 4, 8)
        reference = torch.randn(4, 8)
        _, aux = adapter(x, reference)
        torch.testing.assert_close(aux["base_route_weights"].sum(-1), torch.ones(4))
        self.assertEqual(tuple(aux["incremental_gates"].shape), (4, 0))
        adapter.add_expert(session_id=1, reduction=2)
        adapter.add_expert(session_id=2, reduction=2)
        with torch.no_grad():
            for router in adapter.incremental_routers:
                router.weight.zero_()
                router.bias.fill_(2.0)
        adapter.eval()
        _, aux = adapter(x, reference)
        gates = aux["incremental_gates"]
        self.assertTrue(torch.all((gates > 0.5) & (gates < 1.0)))
        self.assertTrue(torch.all(gates.sum(-1) > 1.0))

    def test_zero_up_new_expert_preserves_current_function(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.eval()
        x, reference = torch.randn(6, 3, 8), torch.randn(3, 8)
        before, _ = adapter(x, reference)
        adapter.add_expert(session_id=1, reduction=2)
        adapter.eval()
        after, _ = adapter(x, reference)
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_only_new_expert_and_router_are_trainable(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        self.assertTrue(all(p.requires_grad for p in adapter.experts[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.incremental_routers[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.base_router.parameters()))

    def test_reference_stream_is_independent_of_earlier_adapters(self):
        torch.manual_seed(19)
        reference_model = Transformer(width=8, layers=2, heads=2).eval()
        enhanced_model = copy.deepcopy(reference_model)
        enhanced_model.resblocks[0].moe_adapter = VisualMoEAdapter(8, 2, 2)
        enhanced_model.resblocks[1].moe_adapter = VisualMoEAdapter(8, 2, 2)
        enhanced_model.eval()
        x = torch.randn(5, 3, 8)
        reference = x
        expected = []
        with torch.no_grad():
            for block in reference_model.resblocks:
                reference = block.forward_base(reference)
                expected.append(reference[0].clone())
        _, aux, _ = enhanced_model(x)
        self.assertEqual(len(aux), 2)
        torch.testing.assert_close(aux[0]["raw_reference_cls"], expected[0])
        torch.testing.assert_close(aux[1]["raw_reference_cls"], expected[1])
    def test_incremental_expert_uses_its_own_configured_bottleneck(self):
        adapter = VisualMoEAdapter(48, num_experts=2, reduction=4, incremental_reduction=12)
        adapter.add_expert(session_id=1)
        self.assertEqual(adapter.experts[0].c_fc.out_features, 12)
        self.assertEqual(adapter.experts[-1].c_fc.out_features, 4)
        topology = adapter.export_topology()
        self.assertEqual(topology["incremental_reduction"], 12)
    def test_tiny_vit_runs_both_streams_and_returns_one_image_embedding(self):
        torch.manual_seed(23)
        model = VisionTransformer(
            input_resolution=32, patch_size=8, width=16, layers=2, heads=2, output_dim=8
        )
        model.transformer.resblocks[0].moe_adapter = VisualMoEAdapter(16, 2, 4)
        model.eval()
        feature, aux = model(torch.randn(2, 3, 32, 32), return_moe_aux=True)
        self.assertEqual(tuple(feature.shape), (2, 8))
        self.assertEqual(len(aux), 1)
        self.assertEqual(tuple(aux[0]["raw_reference_cls"].shape), (2, 16))
        self.assertEqual(tuple(aux[0]["block_out"].shape), (2, 16))
    def test_newest_residual_audit_measures_actual_gated_expert_output(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        with torch.no_grad():
            adapter.experts[-1].c_proj.weight.normal_()
            adapter.experts[-1].c_proj.bias.normal_()
        raw = torch.randn(5, 8)
        hidden = torch.randn(5, 8)
        gate, residual = adapter.newest_residual(raw, hidden)
        self.assertEqual(tuple(gate.shape), (5,))
        self.assertEqual(tuple(residual.shape), (5, 8))
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.all(residual.norm(dim=-1) > 0))

    def test_router_requires_frozen_reference_feature(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        with self.assertRaises(ValueError):
            adapter(torch.randn(3, 2, 8), None)


if __name__ == "__main__":
    unittest.main()
