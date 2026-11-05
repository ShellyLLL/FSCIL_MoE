import unittest

import torch
import torch.nn.functional as F

from engine.apc import AdaptationAwarePrototypeController
from models.clip.model import VisualMoEAdapter


def make_layer_features(class_vectors, shots=8, noise=0.005, seed=7):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(len(class_vectors)).repeat_interleave(shots)
    centers = torch.as_tensor(class_vectors, dtype=torch.float32)
    features = centers[labels] + noise * torch.randn(
        len(labels), centers.size(1), generator=generator
    )
    return {
        "cls_in": features.clone(),
        "cls_out": features.clone(),
        "labels": labels,
        "source_ids": torch.arange(len(labels)),
    }


class MultiLayerAPCTests(unittest.TestCase):
    def test_calibration_covers_every_inserted_moe_block(self):
        centers = torch.eye(8)
        features = {
            8: make_layer_features(centers, seed=8),
            9: make_layer_features(centers.roll(1, dims=1), seed=9),
            10: make_layer_features(centers.roll(2, dims=1), seed=10),
            11: make_layer_features(centers.roll(3, dims=1), seed=11),
        }
        controller = AdaptationAwarePrototypeController(.05, 64, 3)
        state = controller.calibrate(features, n_new_classes=2, k_shot=5)
        self.assertEqual(state["version"], 2)
        self.assertEqual(sorted(map(int, state["blocks"])), [8, 9, 10, 11])
        for block in (8, 9, 10, 11):
            self.assertIn("risk_threshold", state["blocks"][str(block)])
            self.assertIn("capacity_threshold", state["blocks"][str(block)])

    def test_decision_reports_risk_and_existing_expert_capacity_per_layer(self):
        centers = torch.eye(8)
        base = {
            8: make_layer_features(centers, seed=8),
            9: make_layer_features(centers.roll(1, dims=1), seed=9),
        }
        controller = AdaptationAwarePrototypeController(.05, 128, 3)
        state = controller.calibrate(base, n_new_classes=2, k_shot=5)
        current = {
            8: make_layer_features(centers, seed=20),
            9: make_layer_features(centers.roll(1, dims=1), seed=21),
        }
        decisions = controller.decide(
            current, old_ids=torch.arange(6), new_ids=torch.tensor([6, 7]), state=state
        )
        self.assertEqual([row["block_idx"] for row in decisions], [8, 9])
        for row in decisions:
            self.assertIn("risk", row)
            self.assertIn("adaptation_loss", row)
            self.assertIn("adaptation_accuracy", row)
            self.assertIn("reason", row)

    def test_confusion_and_low_adaptation_trigger_candidate(self):
        centers = torch.eye(8)
        base = {
            4: make_layer_features(centers, seed=1),
            5: make_layer_features(centers.roll(1, dims=1), seed=2),
        }
        controller = AdaptationAwarePrototypeController(.05, 128, 9)
        state = controller.calibrate(base, n_new_classes=2, k_shot=5)
        # New class 6 is deliberately mapped onto old class 0 at block 4.
        current_centers = centers.clone()
        current_centers[6] = centers[0]
        current_centers[7] = centers[7]
        current = {
            4: make_layer_features(current_centers, seed=31),
            5: make_layer_features(centers.roll(1, dims=1), seed=32),
        }
        decisions = controller.decide(
            current, torch.arange(6), torch.tensor([6, 7]), state
        )
        self.assertTrue(decisions[0]["expand"])
        self.assertIn(decisions[0]["reason"], {"prototype_confusion", "both"})

    def test_missing_adapted_output_is_rejected(self):
        controller = AdaptationAwarePrototypeController(.05, 16, 1)
        invalid = {2: {"cls_in": torch.randn(12, 4),
                       "labels": torch.arange(3).repeat_interleave(4),
                       "source_ids": torch.arange(12)}}
        with self.assertRaises(KeyError):
            controller.calibrate(invalid, 1, 2)

    def test_zero_up_expert_preserves_old_function_and_cls_out_is_exposed(self):
        adapter = VisualMoEAdapter(8, 2, 2).eval()
        x = torch.randn(5, 3, 8)
        before, aux_before = adapter(x)
        adapter.add_expert(session_id=1, reduction=2)
        after, aux_after = adapter(x)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertIn("cls_out", aux_after)
        torch.testing.assert_close(
            aux_after["cls_out"],
            F.normalize(aux_after["cls_in"] + after[0], dim=-1),
            rtol=1e-5, atol=1e-6,
        )
        self.assertLess(float(aux_after["route_weights"][:, 2:].sum(-1).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
