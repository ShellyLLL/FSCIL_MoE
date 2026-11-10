"""Few-shot APC calibration invariants."""
import unittest
import torch
import torch.nn.functional as F
from utils.apc import build_few_shot_apc_reference, evaluate_few_shot_apc, update_few_shot_apc_memory


class FewShotAPCTests(unittest.TestCase):
    def _base_features(self):
        generator = torch.Generator().manual_seed(91)
        result = {}
        labels = torch.arange(6).repeat_interleave(12)
        for block in (8, 9):
            centers = F.normalize(torch.randn(6, 16, generator=generator), dim=-1)
            values = []
            for class_id in range(6):
                noise = 0.15 * torch.randn(12, 16, generator=generator)
                values.append(F.normalize(centers[class_id] + noise, dim=-1))
            result[block] = {"cls_in": torch.cat(values), "labels": labels.clone(),
                             "source_ids": torch.arange(len(labels))}
        return result

    def test_calibration_matches_five_shot_and_competitor_count(self):
        state = build_few_shot_apc_reference(
            self._base_features(), list(range(6)),
            [{"task_id": 1, "competitor_count": 7, "new_class_count": 2}],
            shot=5, episodes=64, alpha=0.05, seed=12, device="cpu")
        self.assertEqual(state["shot"], 5)
        self.assertEqual(state["query_shot"], 4)
        self.assertEqual(set(state["layers"]), {"8", "9"})
        for layer in state["layers"].values():
            calibration = layer["calibration"]["7:2"]
            self.assertEqual(calibration["null_count"], 64)
            self.assertTrue(torch.isfinite(torch.tensor(calibration["threshold"])))

    def test_cross_fitted_queries_and_memory_commit(self):
        state = build_few_shot_apc_reference(
            self._base_features(), list(range(6)),
            [{"task_id": 1, "competitor_count": 7, "new_class_count": 2}],
            shot=5, episodes=64, alpha=0.05, seed=13, device="cpu")
        folds = []
        for _ in range(5):
            fold = {}
            for block in (8, 9):
                rows, labels = [], []
                for class_id in (6, 7):
                    center = torch.zeros(16); center[class_id] = 1.0
                    rows.append(F.normalize(center + 0.1 * torch.randn(4, 16), dim=-1))
                    labels.extend([class_id] * 4)
                fold[block] = {"cls_in": torch.cat(rows), "labels": torch.tensor(labels)}
            folds.append(fold)
        result = evaluate_few_shot_apc(folds, state, [6, 7], device="cpu")
        self.assertEqual(set(result["layers"]), {"8", "9"})
        for item in result["layers"].values():
            self.assertEqual(item["competitor_count"], 7)
            self.assertEqual(item["new_class_count"], 2)
        full = {}
        for block in (8, 9):
            rows, labels = [], []
            for class_id in (6, 7):
                center = torch.zeros(16); center[class_id] = 1.0
                rows.append(F.normalize(center + 0.1 * torch.randn(5, 16), dim=-1))
                labels.extend([class_id] * 5)
            full[block] = {"cls_in": torch.cat(rows), "labels": torch.tensor(labels)}
        update_few_shot_apc_memory(state, full, [6, 7])
        for layer in state["layers"].values():
            self.assertEqual(set(layer["prototypes"]), {str(i) for i in range(8)})

    def test_calibration_is_deterministic(self):
        features = self._base_features()
        specs = [{"task_id": 1, "competitor_count": 7, "new_class_count": 2}]
        first = build_few_shot_apc_reference(features, list(range(6)), specs,
            shot=5, episodes=64, alpha=0.05, seed=77, device="cpu")
        second = build_few_shot_apc_reference(features, list(range(6)), specs,
            shot=5, episodes=64, alpha=0.05, seed=77, device="cpu")
        for block in ("8", "9"):
            self.assertEqual(first["layers"][block]["thresholds"],
                             second["layers"][block]["thresholds"])


if __name__ == "__main__":
    unittest.main()
