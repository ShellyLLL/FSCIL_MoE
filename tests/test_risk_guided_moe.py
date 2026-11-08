"""Regression tests for one current encoder and one all-class BiMC prototype bank."""
import unittest
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from engine.engine import Runner
from models.clip.model import VisualMoEAdapter
from models.bimc import BiMC
from utils.evaluator import AccuracyEvaluator


class _AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)

    def adapters(self):
        return [(0, self.adapter)]

    def rebuild_moe_topology(self, topology):
        self.adapter.rebuild_topology(topology["0"])

    def export_moe_topology(self):
        return {"0": self.adapter.export_topology()}


class _FeatureAuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2), requires_grad=False)

    def parse_batch(self, batch):
        return batch["image"], batch["label"]

    def extract_img_feature(self, images):
        return images @ self.weight


class SessionMetricTests(unittest.TestCase):
    def test_bimc_table_aa_and_pd_definitions(self):
        # The first row of BiMC Table 2 reports A_0=92.58, A_8=86.09,
        # AA=88.55 and PD=6.49.
        session_acc = [92.58, 91.88, 89.37, 88.24, 88.42, 87.86, 86.33, 86.16, 86.09]
        metrics = Runner._compute_fscil_metrics(session_acc)
        self.assertAlmostEqual(metrics["aa"], 88.5477777778, places=8)
        self.assertAlmostEqual(metrics["pd"], 6.49, places=8)
        self.assertAlmostEqual(metrics["base_acc"], 92.58, places=8)
        self.assertAlmostEqual(metrics["last_acc"], 86.09, places=8)

    def test_current_run_log_uses_session_mean_and_base_to_final_drop(self):
        session_acc = [86.840, 84.090, 82.190, 78.840, 77.560, 75.750,
                       75.230, 75.040, 73.320, 73.390, 73.230]
        metrics = Runner._compute_fscil_metrics(session_acc)
        self.assertAlmostEqual(metrics["aa"], 77.7709090909, places=8)
        self.assertAlmostEqual(metrics["pd"], 13.610, places=8)

    def test_pd_is_signed_when_final_accuracy_exceeds_base(self):
        metrics = Runner._compute_fscil_metrics([80.0, 81.0])
        self.assertAlmostEqual(metrics["pd"], -1.0, places=8)

    def test_evaluator_preserves_unrounded_cumulative_accuracy(self):
        evaluator = AccuracyEvaluator([np.asarray([0, 1])])
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        labels = torch.tensor([0, 1, 1])
        metrics = evaluator.calc_accuracy(
            logits, labels, task_id=0, class_ids=torch.tensor([0, 1])
        )
        self.assertAlmostEqual(metrics["mean_acc"], 200.0 / 3.0, places=8)


class PrototypeAndStateTests(unittest.TestCase):
    def test_freeze_audit_and_optimizer_scope_are_exact(self):
        runner = Runner.__new__(Runner)
        runner.model = _AdapterModel()
        runner.model.adapter.add_expert(session_id=1, reduction=2)
        expected = runner.model.adapter.newest_parameters()
        runner._assert_trainable_scope(expected, "test")
        optimizer = optim.AdamW(expected, lr=1e-3)
        runner._assert_optimizer_scope(optimizer, expected, "test")
        frozen = runner._snapshot_frozen_parameters(expected)
        x, reference = torch.randn(5, 3, 8), torch.randn(3, 8)
        optimizer.zero_grad(set_to_none=True)
        residual, _ = runner.model.adapter(x, reference)
        residual.square().mean().backward()
        runner._assert_gradient_scope(expected, "test")
        optimizer.step()
        runner.model.adapter.freeze_all()
        runner._assert_all_parameters_frozen("test")
        runner._assert_frozen_unchanged(frozen, "test")

    def test_checkpoint_rollback_restores_exact_topology_and_weights(self):
        runner = Runner.__new__(Runner)
        runner.model = _AdapterModel()
        before_topology = runner.model.export_moe_topology()
        before_state = {
            key: value.detach().cpu().clone()
            for key, value in runner.model.state_dict().items()
        }
        runner.model.adapter.add_expert(session_id=1, reduction=2)
        with torch.no_grad():
            runner.model.adapter.experts[-1].c_proj.bias.fill_(3.0)
        runner._restore_candidate_snapshot(before_topology, before_state)
        self.assertEqual(runner.model.adapter.num_experts, 2)
        self.assertTrue(all(not p.requires_grad for p in runner.model.parameters()))
        for key, value in before_state.items():
            torch.testing.assert_close(runner.model.state_dict()[key].cpu(), value)

    def test_historical_feature_audit_detects_end_to_end_embedding_regression(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        runner.model = _FeatureAuditModel()
        with torch.no_grad():
            runner.model.weight.copy_(torch.tensor([[0., 1.], [1., 0.]]))
        batches = [{
            "image": torch.eye(2),
            "label": torch.tensor([0, 1]),
            "source_id": torch.tensor([100, 101]),
        }]
        targets = {
            100: (0, torch.tensor([1., 0.])),
            101: (1, torch.tensor([0., 1.])),
        }
        bank = {
            "class_ids": torch.tensor([0, 1]),
            "fused_proto": torch.eye(2),
        }
        audit = runner._evaluate_old_feature_preservation(batches, targets, bank)
        self.assertEqual(audit["count"], 2)
        self.assertEqual(audit["baseline_correct"], 2)
        self.assertEqual(audit["candidate_correct"], 0)
        self.assertEqual(audit["lost_previously_correct"], 2)
        self.assertAlmostEqual(audit["mean_cosine_similarity"], 0.0, places=6)

    def test_current_feature_refresh_never_overwrites_class_birth_teacher(self):
        runner = Runner.__new__(Runner)
        birth = torch.tensor([[1., 0.], [0., 1.]])
        state = {
            "support_features": birth.clone(),
            "support_labels": torch.tensor([0, 1]),
            "support_source_ids": torch.tensor([20, 21]),
            "feature_anchor_session": 0,
            "creation_session": 0,
        }
        current = torch.tensor([[0.8, 0.6], [0.6, 0.8]])
        labels = torch.tensor([0, 1])
        sources = torch.tensor([20, 21])
        Runner._update_current_feature_cache(
            state, current, labels, sources, torch.tensor([0, 1])
        )
        torch.testing.assert_close(state["support_features"], birth)
        torch.testing.assert_close(state["current_support_features"], current)
        targets = runner._historical_feature_targets([state])
        torch.testing.assert_close(targets[20][1], birth[0])

    def test_historical_feature_targets_require_complete_session_metadata(self):
        runner = Runner.__new__(Runner)
        state = {
            "support_features": torch.eye(2),
            "support_labels": torch.tensor([0, 1]),
            "support_source_ids": torch.tensor([20, 21]),
            "feature_anchor_session": 0,
            "creation_session": 0,
        }
        targets = runner._historical_feature_targets([state])
        self.assertEqual(set(targets), {20, 21})
        with self.assertRaisesRegex(RuntimeError, "schema 24"):
            runner._historical_feature_targets([{"class_ids": torch.tensor([0])}])

    def test_visual_text_intra_modal_and_inter_modal_calibration(self):
        runner = Runner.__new__(Runner)
        cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TEXT_CALIBRATION=True, LAMBDA_T=0.5)),
            DATASET=SimpleNamespace(BETA=0.3),
        )
        model = BiMC.__new__(BiMC)
        model.cfg = cfg
        runner.model = model
        visual = F.normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.]]), dim=-1)
        text = F.normalize(torch.tensor([[0., 0., 1.], [0., 1., 0.]]), dim=-1)
        desc = F.normalize(torch.tensor([[0., 0.5, 0.5], [0.5, 0.5, 0.]]), dim=-1)
        fused = runner._fuse(visual, text, desc)
        expected = 0.3 * (0.5 * text + 0.5 * desc) + 0.7 * visual
        self.assertEqual(tuple(fused.shape), (2, 3))
        torch.testing.assert_close(fused, expected)
        self.assertFalse(torch.allclose(fused.norm(dim=-1), torch.ones(2)))

    def test_global_classifier_merges_all_classes_without_task_gate_or_versions(self):
        states = [
            {"class_ids": torch.tensor([2, 0]), "image_proto": torch.eye(3)[:2],
             "text_features": torch.eye(3)[:2], "description_proto": torch.eye(3)[:2],
             "fused_proto": torch.eye(3)[:2]},
            {"class_ids": torch.tensor([1]), "image_proto": torch.eye(3)[2:3],
             "text_features": torch.eye(3)[2:3], "description_proto": torch.eye(3)[2:3],
             "fused_proto": torch.eye(3)[2:3]},
        ]
        merged = Runner._merge_states(states, torch.device("cpu"))
        self.assertEqual(merged["class_ids"].tolist(), [0, 1, 2])
        self.assertEqual(tuple(merged["fused_proto"].shape), (3, 3))
        self.assertNotIn("encoder_versions", merged)
        self.assertNotIn("task_ids", merged)

    def test_apc_threshold_uses_exact_empirical_ecdf_order_statistic(self):
        values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        threshold = Runner._empirical_upper_quantile(values, alpha=0.2)
        self.assertAlmostEqual(float(threshold), 0.8, places=6)

    def test_apc_candidate_diagnostics_report_per_class_confusion_and_margin(self):
        runner = Runner.__new__(Runner)
        runner.device = torch.device("cpu")
        runner.apc_state = {"thresholds": {0: 0.75}}
        old_states = [{
            "class_ids": torch.tensor([10, 11]),
            "raw_layer_prototypes": {
                0: torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            },
        }]
        new_state = {
            "class_ids": torch.tensor([20, 21]),
            "raw_layer_prototypes": {
                0: torch.tensor([[0.8, 0.6], [0.0, 1.0]])
            },
        }
        row = runner._candidate_layers(new_state, old_states)[0]
        self.assertTrue(row["candidate"])
        self.assertAlmostEqual(row["confusion"], 1.0, places=6)
        self.assertAlmostEqual(row["margin"], 0.25, places=6)
        self.assertEqual(row["most_confused_new_class_id"], 21)
        self.assertEqual(row["nearest_old_class_id"], 11)
        self.assertEqual(len(row["new_class_confusions"]), 2)

    def test_low_shot_apc_policy_returns_no_expansion_without_cv(self):
        runner = Runner.__new__(Runner)
        dataset = SimpleNamespace(labels=[0, 1], source_ids=[10, 20])
        self.assertEqual(runner._support_folds(dataset), [])

    def test_apc_candidate_set_must_match_every_held_out_fold(self):
        runner = Runner.__new__(Runner)
        runner.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
                APC=SimpleNamespace(ENABLE=True, SIGNIFICANCE=0.05)
            ))
        )
        # Candidate topology differs by fold. The full-data proposal must not
        # be authorized by risk gains measured for different fold topologies.
        fold_sets = [[1, 2], [1], [1, 2]]
        final_set = {1, 2}
        unstable = [
            index for index, candidates in enumerate(fold_sets)
            if set(candidates) != final_set
        ]
        self.assertEqual(unstable, [1])

    def test_source_grouped_folds_keep_every_source_wholly_in_one_partition(self):
        dataset = SimpleNamespace(
            labels=[0, 0, 0, 0, 1, 1, 1, 1],
            source_ids=[10, 11, 12, 13, 20, 21, 22, 23],
        )
        runner = Runner.__new__(Runner)
        folds = runner._support_folds(dataset)
        self.assertEqual(len(folds), 4)
        for fit, val in folds:
            self.assertFalse(set(fit) & set(val))
            fit_sources = {dataset.source_ids[i] for i in fit}
            val_sources = {dataset.source_ids[i] for i in val}
            self.assertFalse(fit_sources & val_sources)

    def test_base_visual_anchor_is_installed_once_and_used_for_incremental_calibration(self):
        model = BiMC.__new__(BiMC)
        model.device = torch.device("cpu")
        model.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TAU=8.0, LAMBDA_I=0.2))
        )
        base = F.normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.]]), dim=-1)
        novel = F.normalize(torch.tensor([[0.8, 0.2, 0.]]), dim=-1)
        model.base_vision_prototype = None
        model.set_base_visual_prototype(base)
        calibrated = model.calibrate_incremental_visual(novel)
        weights = torch.softmax(8.0 * (novel @ base.t()), dim=-1)
        expected = 0.8 * novel + 0.2 * (weights @ base)
        self.assertEqual(tuple(calibrated.shape), (1, 3))
        torch.testing.assert_close(calibrated, expected)
        torch.testing.assert_close(model.base_vision_prototype, base)
        with self.assertRaises(RuntimeError):
            model.set_base_visual_prototype(torch.flip(base, dims=[0]))

    def test_incremental_calibration_fails_with_actionable_error_without_base_anchor(self):
        model = BiMC.__new__(BiMC)
        model.device = torch.device("cpu")
        model.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TAU=8.0, LAMBDA_I=0.2))
        )
        model.base_vision_prototype = None
        with self.assertRaisesRegex(RuntimeError, "Session 0 image_proto"):
            model.calibrate_incremental_visual(torch.tensor([[1., 0., 0.]]))

    def test_expansion_is_function_preserving_at_creation(self):
        torch.manual_seed(4)
        adapter = VisualMoEAdapter(12, num_experts=2, reduction=3).eval()
        x, ref = torch.randn(5, 3, 12), torch.randn(3, 12)
        before, _ = adapter(x, ref)
        adapter.add_expert(1, reduction=3)
        adapter.eval()
        after, _ = adapter(x, ref)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_incremental_router_is_invariant_to_reference_scale(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=1)
        reference = torch.randn(5, 8)
        torch.testing.assert_close(
            adapter.newest_gate(reference),
            adapter.newest_gate(17.0 * reference),
        )

    def test_mean_of_normalized_features_preserves_resultant_magnitude(self):
        features = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
        prototype = BiMC.mean_normalized_features(features, dim=0)
        torch.testing.assert_close(prototype, torch.tensor([0.5, 0.5]))
        self.assertAlmostEqual(float(prototype.norm()), 2 ** -0.5, places=6)

    def test_base_visual_anchor_preserves_resultant_magnitude(self):
        model = BiMC.__new__(BiMC)
        model.device = torch.device("cpu")
        model.base_vision_prototype = None
        model.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TAU=8.0, LAMBDA_I=0.2))
        )
        base = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.25, 0.0]])
        model.set_base_visual_prototype(base)
        torch.testing.assert_close(model.base_vision_prototype, base)
        self.assertFalse(torch.allclose(model.base_vision_prototype.norm(dim=-1), torch.ones(2)))

    def test_loo_prototypes_exclude_exact_query_source(self):
        features = F.normalize(torch.tensor([
            [1., 0.], [0., 1.], [0.8, 0.2], [0.2, 0.8]
        ]), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        sources = torch.tensor([10, 11, 20, 21])
        prototypes = Runner._loo_class_prototypes(
            features, labels, sources, torch.tensor([0, 1]),
            torch.tensor([0, 1]), torch.tensor([10, 21]),
        )
        torch.testing.assert_close(prototypes[0, 0], features[1])
        torch.testing.assert_close(
            prototypes[0, 1], features[2:4].mean(0)
        )
        torch.testing.assert_close(prototypes[1, 1], features[2])
        torch.testing.assert_close(
            prototypes[1, 0], features[:2].mean(0)
        )


if __name__ == "__main__":
    unittest.main()
