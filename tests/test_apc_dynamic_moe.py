import unittest

import torch
import torch.nn.functional as F

from engine.apc import AdaptationAwarePrototypeController
from models.clip.model import ResidualAttentionBlock, VisualMoEAdapter


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
        "block_out": features.clone(),
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
        self.assertEqual(state["version"], 4)
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

    def test_zero_up_expert_preserves_old_function_and_version_is_explicit(self):
        adapter = VisualMoEAdapter(8, 2, 2).eval()
        x = torch.randn(5, 3, 8)
        before, _ = adapter(x, max_version=0)
        adapter.add_expert(session_id=1, reduction=2, version_id=4)
        after_old, aux_old = adapter(x, max_version=3)
        after_new, aux_new = adapter(x, max_version=4)
        torch.testing.assert_close(before, after_old, rtol=0, atol=0)
        torch.testing.assert_close(before, after_new, rtol=0, atol=0)
        self.assertEqual(adapter.expert_birth_sessions, [0, 0, 1])
        self.assertEqual(adapter.expert_birth_versions, [0, 0, 4])
        self.assertIn("cls_out", aux_new)
        self.assertLess(float(aux_old["route_weights"].shape[-1]), float(aux_new["route_weights"].shape[-1]))

    def test_apc_block_feature_is_actual_post_residual_transformer_state(self):
        block = ResidualAttentionBlock(8, 2).eval()
        block.moe_adapter = VisualMoEAdapter(8, 2, 2).eval()
        x = torch.randn(5, 3, 8)
        output, aux = block(x, max_version=0)
        torch.testing.assert_close(aux["block_out"], output[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(aux["block_out"], aux["adapter_output_cls"]))

    def test_topology_round_trip_preserves_monotonic_versions(self):
        adapter = VisualMoEAdapter(8, 2, 2).eval()
        adapter.add_expert(session_id=2, reduction=2, version_id=5)
        adapter.add_expert(session_id=2, reduction=2, version_id=6)
        topology = adapter.export_topology()
        clone = VisualMoEAdapter(8, 2, 2).eval()
        clone.rebuild_topology(topology)
        self.assertEqual(clone.expert_birth_sessions, [0, 0, 2, 2])
        self.assertEqual(clone.expert_birth_versions, [0, 0, 5, 6])


    def test_topology_bound_calibration_rejects_stale_topology(self):
        centers = torch.eye(8)
        features = {8: make_layer_features(centers, seed=8)}
        controller = AdaptationAwarePrototypeController(.05, 32, 3)
        state = controller.calibrate(
            features, n_new_classes=2, k_shot=5,
            topology_version=3, topology_signature="topology-A",
            support_manifest_hash="manifest-A",
        )
        current = {8: make_layer_features(centers, seed=19)}
        with self.assertRaisesRegex(ValueError, "different MoE topology"):
            controller.decide(
                current, torch.arange(6), torch.tensor([6, 7]), state,
                topology_version=3, topology_signature="topology-B",
            )
        with self.assertRaisesRegex(ValueError, "different topology version"):
            controller.decide(
                current, torch.arange(6), torch.tensor([6, 7]), state,
                topology_version=4, topology_signature="topology-A",
            )


if __name__ == "__main__":
    unittest.main()
