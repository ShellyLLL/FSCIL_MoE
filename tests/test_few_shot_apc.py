"""APC tests using only real class prototypes and real support samples."""
import unittest
import torch
import torch.nn.functional as F
from utils.apc import build_few_shot_apc_reference, evaluate_few_shot_apc, update_few_shot_apc_memory


class FewShotAPCTests(unittest.TestCase):
    @staticmethod
    def _features(vectors, labels, shots):
        rows, targets = [], []
        for vector, label in zip(vectors, labels):
            for _ in range(shots):
                rows.append(torch.tensor(vector, dtype=torch.float32))
                targets.append(label)
        return {"cls_in": torch.stack(rows), "labels": torch.tensor(targets)}

    def test_threshold_comes_from_real_base_nearest_neighbours(self):
        vectors = [[1., 0., 0.], [.8, .6, 0.], [0., 1., 0.]]
        base = self._features(vectors, [0, 1, 2], 8)
        state = build_few_shot_apc_reference({5: base}, [0, 1, 2], quantile=.5)
        prototypes = F.normalize(torch.tensor(vectors), dim=-1)
        similarities = prototypes @ prototypes.T
        similarities.fill_diagonal_(float("-inf"))
        expected = torch.quantile(similarities.max(dim=1).values, .5).item()
        self.assertAlmostEqual(state["layers"]["5"]["threshold"], expected, places=6)
        self.assertEqual(state["seen_class_ids"], [0, 1, 2])

    def test_real_five_shot_support_controls_expansion(self):
        vectors = [[1., 0., 0.], [.8, .6, 0.], [0., 1., 0.]]
        state = build_few_shot_apc_reference(
            {5: self._features(vectors, [0, 1, 2], 8)}, [0, 1, 2], quantile=.5)
        confused = self._features([[.82, .57, 0.]], [3], 5)
        decision = evaluate_few_shot_apc({5: confused}, state, [3], expected_shot=5)
        self.assertEqual(decision["candidate_blocks"], [5])
        self.assertEqual(decision["layers"]["5"]["nearest_seen_class_by_new_class"][3], 1)
        separated = self._features([[0., 0., 1.]], [4], 5)
        decision = evaluate_few_shot_apc({5: separated}, state, [4], expected_shot=5)
        self.assertEqual(decision["candidate_blocks"], [])

    def test_memory_commit_uses_actual_support_count(self):
        base = self._features([[1., 0.], [0., 1.]], [0, 1], 6)
        state = build_few_shot_apc_reference({2: base}, [0, 1], quantile=.9)
        support = self._features([[0., 0., 1.]], [2], 5)
        update_few_shot_apc_memory(state, {2: support}, [2], expected_shot=5)
        self.assertEqual(state["seen_class_ids"], [0, 1, 2])
        self.assertIn("2", state["layers"]["2"]["prototypes"])
        with self.assertRaisesRegex(ValueError, "received 5 samples"):
            update_few_shot_apc_memory(state, {2: support}, [3], expected_shot=4)


if __name__ == "__main__":
    unittest.main()
