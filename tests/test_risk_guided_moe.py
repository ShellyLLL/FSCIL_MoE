"""Regression tests for one current encoder and one all-class BiMC prototype bank."""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from engine.engine import Runner
from models.clip.model import VisualMoEAdapter


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
        self.assertEqual(tuple(fused.shape), (2, 3))
        torch.testing.assert_close(fused.norm(dim=-1), torch.ones(2))

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

    def test_expansion_is_function_preserving_at_creation(self):
        torch.manual_seed(4)
        adapter = VisualMoEAdapter(12, num_experts=2, reduction=3).eval()
        x, ref = torch.randn(5, 3, 12), torch.randn(3, 12)
        before, _ = adapter(x, ref)
        adapter.add_expert(1, reduction=3)
        adapter.eval()
        after, _ = adapter(x, ref)
        torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
