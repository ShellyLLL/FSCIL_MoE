"""Regression tests for one current encoder and one all-class BiMC prototype bank."""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from engine.engine import Runner
from models.clip.model import VisualMoEAdapter
from models.bimc import BiMC


class PrototypeAndStateTests(unittest.TestCase):
    def test_visual_text_intra_modal_and_inter_modal_calibration(self):
        runner = Runner.__new__(Runner)
        runner.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(TEXT_CALIBRATION=True, LAMBDA_T=0.5)),
            DATASET=SimpleNamespace(BETA=0.3),
        )
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
            prototypes[0, 1], F.normalize(features[2:4].mean(0), dim=-1)
        )
        torch.testing.assert_close(prototypes[1, 1], features[2])
        torch.testing.assert_close(
            prototypes[1, 0], F.normalize(features[:2].mean(0), dim=-1)
        )


if __name__ == "__main__":
    unittest.main()
