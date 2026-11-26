"""Unit tests for Session-mean AE-Z expansion and migrated dynamic experts."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from engine.engine import Runner
from models.bimc import BiMC, SESSION_EXPANSION_Z_THRESHOLD
from models.clip.model import VisualMoEAdapter


def make_bimc(adapter):
    model = BiMC.__new__(BiMC)
    nn.Module.__init__(model)
    block = nn.Module()
    block.moe_adapter = adapter
    transformer = nn.Module()
    transformer.resblocks = nn.ModuleList([block])
    visual = nn.Module()
    visual.transformer = transformer
    model.clip_model = nn.Module()
    model.clip_model.visual = visual
    model.device = torch.device("cpu")
    return model


class UnifiedDynamicMoETests(unittest.TestCase):
    def test_session_mean_z_uses_minimum_historical_rd_and_fixed_threshold(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        scores = torch.tensor([
            [0.2, 1.4],
            [0.4, 1.6],
            [0.6, 1.8],
            [0.8, 2.0],
        ])
        adapter.descriptor_scores = lambda values, require_ready=True: scores
        model = make_bimc(adapter)
        features = {0: {
            "cls_in": torch.randn(4, 8),
            "labels": torch.tensor([7, 7, 8, 8]),
            "source_ids": torch.tensor([10, 11, 12, 13]),
        }}
        decision = model.plan_layer_expansion(0, features, [7, 8])
        self.assertAlmostEqual(decision.session_mean_z, 0.5)
        self.assertEqual(decision.anchor_expert_id, 0)
        self.assertEqual(decision.threshold, 1.0)
        self.assertEqual(decision.expand, False)
        self.assertEqual(SESSION_EXPANSION_Z_THRESHOLD, 1.0)

    def test_expansion_decision_does_not_use_classwise_scores(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        scores = torch.tensor([
            [1.4, 0.2],
            [1.6, 0.4],
            [1.8, 0.6],
            [2.0, 0.8],
        ])
        adapter.descriptor_scores = lambda values, require_ready=True: scores
        model = make_bimc(adapter)
        features = {0: {
            "cls_in": torch.randn(4, 8),
            "labels": torch.tensor([1, 1, 2, 2]),
            "source_ids": torch.tensor([1, 2, 3, 4]),
        }}
        decision = model.plan_layer_expansion(0, features, [1, 2])
        self.assertAlmostEqual(decision.session_mean_z, 0.5)
        self.assertFalse(decision.expand)

    def test_router_and_experts_consume_adapted_stream_not_frozen_reference(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        adapted = torch.randn(3, 2, 8)
        reference = adapted.roll(1, dims=-1)
        _, aux = adapter(adapted, route_input=reference)
        expected_adapted = adapter.layer_norm(adapted)[0]
        expected_reference = adapter.layer_norm(reference)[0]
        torch.testing.assert_close(aux["cls_in"], expected_adapted)
        torch.testing.assert_close(aux["expert_in"], expected_adapted)
        torch.testing.assert_close(aux["reference_cls_in"], expected_reference)
        self.assertFalse(torch.allclose(aux["cls_in"], aux["reference_cls_in"]))

    def test_new_expert_and_rd_share_nearest_anchor_and_exact_parameters(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(16, num_experts=2, reduction=4, descriptor_dim=4)
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        with torch.no_grad():
            adapter.experts[1].c_fc.weight.fill_(0.125)
            adapter.experts[1].c_proj.weight.fill_(0.25)
        new_id = adapter.add_expert(
            session_id=1, anchor_expert_id=1, descriptor_residual_rank=3,
            gate_bias_init=-4.0,
        )
        self.assertEqual(new_id, 2)
        self.assertEqual(adapter.expert_anchor_ids[-1], 1)
        self.assertEqual(adapter.descriptors[-1].anchor_expert_id, 1)
        self.assertEqual(adapter.experts[-1].c_fc.out_features, adapter.experts[1].c_fc.out_features)
        for name, value in adapter.experts[1].state_dict().items():
            torch.testing.assert_close(adapter.experts[-1].state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(adapter.descriptors[-1].residual_rank, 3)

    def test_topology_roundtrip_restores_migration_anchor_and_hidden_width(self):
        torch.manual_seed(11)
        source = VisualMoEAdapter(16, num_experts=2, reduction=4, descriptor_dim=4)
        for descriptor in source.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        source.add_expert(session_id=1, anchor_expert_id=0, descriptor_residual_rank=3)
        topology = source.export_topology()
        state = {key: value.detach().clone() for key, value in source.state_dict().items()}
        restored = VisualMoEAdapter(16, num_experts=2, reduction=4, descriptor_dim=4)
        restored.rebuild_topology(topology)
        restored.load_state_dict(state, strict=True)
        self.assertEqual(restored.export_topology(), topology)
        self.assertEqual(restored.expert_anchor_ids, [-1, -1, 0])
        self.assertEqual(restored.experts[-1].c_fc.out_features, source.experts[-1].c_fc.out_features)

    def test_l2sp_is_zero_at_migration_and_penalizes_parameter_drift(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        adapter.add_expert(session_id=1, anchor_expert_id=0, descriptor_residual_rank=2)
        expert = adapter.experts[-1]
        anchor = {name: parameter.detach().clone() for name, parameter in expert.named_parameters()}
        loss = Runner._l2sp_anchor_loss(expert, anchor)
        self.assertEqual(float(loss.item()), 0.0)
        with torch.no_grad():
            expert.c_proj.bias.add_(0.1)
        loss = Runner._l2sp_anchor_loss(expert, anchor)
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(expert.c_proj.bias.grad)

    def test_only_new_expert_and_router_are_trainable_during_expansion(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        adapter.add_expert(session_id=1, anchor_expert_id=0, descriptor_residual_rank=2)
        self.assertTrue(all(not p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.experts[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.router.columns[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.router.columns[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.descriptors[-1].parameters()))
        adapter.freeze_all()
        self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))

    def test_global_logits_rejects_stale_prototype_bank_version(self):
        class DummyModel:
            def extract_img_feature_train(self, images):
                return images.float(), []

        runner = Runner.__new__(Runner)
        runner.model_without_dp = DummyModel()
        bank = {"encoder_step": 3}
        with self.assertRaisesRegex(RuntimeError, "prototype bank is stale"):
            runner._global_incremental_logits(
                torch.tensor([[1.0, 0.0]]), torch.tensor([1]), torch.tensor([10]),
                bank, 10.0, expected_encoder_step=4,
            )

    def test_global_logits_rejects_source_label_mismatch(self):
        class DummyModel:
            def extract_img_feature_train(self, images):
                return images.float(), []

        runner = Runner.__new__(Runner)
        runner.model_without_dp = DummyModel()
        bank = {
            "global_full": torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            "global_ids": torch.tensor([0, 1, 2]),
            "old_class_count": 1,
            "novel_full": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
            "novel_loo": torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]]),
            "source_ids": torch.tensor([10]),
            "labels": torch.tensor([1]),
        }
        with self.assertRaisesRegex(ValueError, "source ids and labels disagree"):
            runner._global_incremental_logits(
                torch.tensor([[1.0, 0.0]]), torch.tensor([2]), torch.tensor([10]),
                bank, 10.0,
            )

    def test_session_decision_preserves_global_rule_and_reports_classwise_coverage(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        scores = torch.tensor([
            [0.1, 2.0], [0.1, 2.0], [2.0, 0.1], [2.0, 0.1],
        ])
        adapter.descriptor_scores = lambda values, require_ready=True: scores
        model = make_bimc(adapter)
        features = {0: {
            "cls_in": torch.randn(4, 8),
            "labels": torch.tensor([7, 7, 8, 8]),
            "source_ids": torch.tensor([10, 11, 12, 13]),
        }}
        decision = model.plan_layer_expansion(0, features, [7, 8])
        self.assertFalse(decision.expand)
        self.assertAlmostEqual(decision.session_mean_z, 1.05)
        self.assertEqual(decision.class_diagnostics[7]["best_historical_mean_z"], 0.1)
        self.assertEqual(decision.class_diagnostics[8]["best_historical_mean_z"], 0.1)
        self.assertEqual(decision.class_diagnostics[7]["support_count"], 2)
        self.assertEqual(decision.class_diagnostics[8]["support_count"], 2)

    def test_rd_fit_and_reference_calibration_are_source_exclusive_and_class_balanced(self):
        labels = torch.tensor([0] * 5 + [1] * 5)
        sources = torch.tensor([10, 11, 12, 13, 14, 20, 21, 22, 23, 24])
        fit_mask, reference_mask = Runner._source_exclusive_rd_split(
            labels, sources, labels.clone(), sources.clone(), per_class_reference=1,
        )
        fit_ids = set(sources[fit_mask].tolist())
        reference_ids = set(sources[reference_mask].tolist())
        self.assertEqual(len(reference_ids), 2)
        self.assertTrue(fit_ids.isdisjoint(reference_ids))
        self.assertEqual(torch.unique(labels[reference_mask], return_counts=True)[1].tolist(), [1, 1])
        self.assertEqual(torch.unique(labels[fit_mask], return_counts=True)[1].tolist(), [4, 4])

    def test_rd_source_split_rejects_label_mismatch_and_insufficient_fit_sources(self):
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            Runner._source_exclusive_rd_split(
                torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 3, 4]),
                torch.tensor([0, 1, 1, 1]), torch.tensor([1, 2, 3, 4]),
                per_class_reference=1,
            )
        with self.assertRaisesRegex(ValueError, "at least three"):
            Runner._source_exclusive_rd_split(
                torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 3, 4]),
                torch.tensor([0, 0, 1, 1]), torch.tensor([1, 2, 3, 4]),
                per_class_reference=1,
            )

    def test_invalid_migration_anchor_does_not_mutate_topology(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        before = adapter.export_topology()
        with self.assertRaises(ValueError):
            adapter.add_expert(session_id=1, anchor_expert_id=9)
        self.assertEqual(adapter.export_topology(), before)


if __name__ == "__main__":
    unittest.main()
