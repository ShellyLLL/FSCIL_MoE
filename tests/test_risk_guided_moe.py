"""Unit tests for the AE-driven, single-feature-space FSCIL architecture."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip.model import RepresentationDescriptor, VisualMoEAdapter
from models.bimc import BiMC
from engine.engine import Runner


class UnifiedDynamicMoETests(unittest.TestCase):
    def test_sema_descriptor_uses_linear_ae_and_rolling_500_error_buffer(self):
        torch.manual_seed(3)
        descriptor = RepresentationDescriptor(8, descriptor_dim=3, buffer_size=500)
        values = torch.randn(600, 8)
        optimizer = torch.optim.AdamW(descriptor.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            loss = F.mse_loss(descriptor(values), values)
            loss.backward()
            optimizer.step()
        errors = descriptor.reconstruction_error(values)
        descriptor.update_reference(errors)
        self.assertEqual(int(descriptor.error_count.item()), 500)
        torch.testing.assert_close(descriptor.error_history, errors[-500:])
        self.assertTrue(torch.isfinite(descriptor.standardized_score(values[:7])).all())

    def test_incremental_expert_append_preserves_function_at_initialization(self):
        torch.manual_seed(5)
        adapter = VisualMoEAdapter(8, num_experts=3, reduction=2,
                                   descriptor_dim=4, descriptor_buffer_size=500).eval()
        x = torch.randn(6, 5, 8)
        before, _ = adapter(x)
        expert_id = adapter.add_expert(session_id=1, reduction=4)
        after, aux = adapter(x)
        self.assertEqual(expert_id, 3)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        base_weights = aux["route_weights"][:, :adapter.base_expert_count]
        inc_weights = aux["route_weights"][:, adapter.base_expert_count:]
        torch.testing.assert_close(base_weights.sum(-1), torch.ones(x.size(1)))
        self.assertTrue(((inc_weights > 0) & (inc_weights < 1)).all())

    def test_new_expert_changes_current_unified_path_but_old_modules_are_frozen(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2).eval()
        x = torch.randn(4, 3, 8)
        before, _ = adapter(x)
        adapter.add_expert(1, reduction=2)
        with torch.no_grad():
            adapter.experts[-1].c_proj.bias.fill_(0.2)
        adapter.freeze_all()
        after, _ = adapter(x)
        self.assertGreater((after - before).abs().max().item(), 0.0)
        self.assertFalse(any(p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in adapter.router.columns[:-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))

    def test_topology_roundtrip_preserves_experts_and_descriptors(self):
        torch.manual_seed(11)
        source = VisualMoEAdapter(8, num_experts=2, reduction=2,
                                  descriptor_dim=4, descriptor_buffer_size=500)
        source.add_expert(2, reduction=4)
        topology = source.export_topology()
        restored = VisualMoEAdapter(8, num_experts=2, reduction=2,
                                    descriptor_dim=4, descriptor_buffer_size=500)
        restored.rebuild_topology(topology)
        restored.load_state_dict(source.state_dict(), strict=True)
        x = torch.randn(3, 2, 8)
        torch.testing.assert_close(source(x)[0], restored(x)[0], rtol=0, atol=0)
        self.assertEqual(restored.num_experts, 3)

    def test_expansion_requires_every_existing_descriptor_to_flag_shift(self):
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        block = nn.Module()
        block.moe_adapter = adapter
        transformer = nn.Module()
        transformer.resblocks = nn.ModuleList([block])
        visual = nn.Module()
        visual.transformer = transformer
        model.clip_model = nn.Module()
        model.clip_model.visual = visual
        model.device = "cpu"
        model.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            VISUAL_MOE=SimpleNamespace(EXPANSION_Z_THRESHOLD=1.0))))
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        adapter.descriptor_scores = lambda x, require_ready=True: torch.full(
            (x.size(0), 2), 2.0)
        features = {0: {"cls_in": torch.randn(4, 8), "ae_in": torch.randn(4, 8),
                        "labels": torch.tensor([2, 2, 3, 3])}}
        decision = model.plan_layer_expansion(0, features, [2, 3])
        self.assertTrue(decision.expand)
        adapter.descriptor_scores = lambda x, require_ready=True: torch.tensor(
            [[2.0, 0.1]] * x.size(0))
        decision = model.plan_layer_expansion(0, features, [2, 3])
        self.assertFalse(decision.expand)

    def test_semantic_guidance_transfers_visual_direction_from_semantically_related_old_class(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        runner.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            TEXT_CALIBRATION=False, LAMBDA_T=0.0, TAU=16.0,
            SEMANTIC_VISUAL_CALIBRATION=0.5)))
        old = {"image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
               "text_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
               "description_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        visual = torch.tensor([[0.0, 1.0]])
        semantic = torch.tensor([[1.0, 0.0]])
        calibrated = runner._semantic_visual_calibration(visual, semantic, [old])
        self.assertGreater(calibrated[0, 0].item(), calibrated[0, 1].item())

    def test_global_incremental_logits_use_one_current_encoder_and_global_columns(self):
        runner = Runner.__new__(Runner)
        class Encoder:
            def extract_img_feature_train(self, images):
                return F.normalize(images, dim=-1), []
            def extract_img_feature(self, images):
                return F.normalize(images, dim=-1)
        runner.model_without_dp = Encoder()
        bank = {
            "global_full": F.normalize(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]), dim=-1),
            "global_ids": torch.tensor([0, 2, 3]),
            "old_class_count": 1,
            "classes": torch.tensor([2, 3]),
            "source_ids": torch.tensor([10, 11]),
            "novel_full": F.normalize(torch.tensor([[0., 1.], [-1., 0.]]), dim=-1),
            "novel_loo": F.normalize(torch.tensor([[[0., 1.], [-1., 0.]],
                                                   [[0., 1.], [-1., 0.]]]), dim=-1),
        }
        images = torch.tensor([[0., 1.]])
        labels = torch.tensor([2])
        logits, targets = runner._global_incremental_logits(
            images, labels, torch.tensor([10]), bank, 100.0)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(int(targets.item()), 1)
        self.assertEqual(int(logits.argmax(dim=1).item()), 1)

    def test_anchor_suppression_is_differentiable_and_finite(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(1, reduction=2)
        mean = torch.randn(4, 8)
        var = torch.ones_like(mean) * 0.1
        loss = adapter.old_anchor_suppression_loss(mean, var)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(adapter.router.columns[-1].weight.grad)
        self.assertIsNotNone(adapter.experts[-1].c_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
