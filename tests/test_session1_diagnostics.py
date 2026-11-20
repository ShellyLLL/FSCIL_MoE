import unittest

import torch

from tools.session1_diagnostics import (
    RESPONSE_METRICS,
    binary_auc,
    classification_diagnostics,
    metric_summary,
    response_posthoc,
)


class Session1DiagnosticsTests(unittest.TestCase):
    def test_metric_summary_reports_distribution_and_empty_values(self):
        summary = metric_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(summary["std"], 1.1180339, places=5)
        self.assertEqual(metric_summary([])["count"], 0)
        self.assertIsNone(metric_summary([])["mean"])

    def test_binary_auc_handles_order_and_ties(self):
        self.assertAlmostEqual(binary_auc([False, False, True, True], [0, 1, 2, 3]), 1.0)
        self.assertAlmostEqual(binary_auc([False, True], [1, 1]), 0.5)
        self.assertAlmostEqual(binary_auc([True, True], [1, 2]), 0.5) if False else None
        self.assertIsNone(binary_auc([True, True], [1, 2]))

    def test_classification_diagnostics_maps_global_ids_and_margin(self):
        # Deliberately non-contiguous global IDs: columns are [2, 7, 11].
        class_ids = torch.tensor([2, 7, 11])
        labels = torch.tensor([2, 7, 11, 2])
        logits = torch.tensor([
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 1.0],  # 11 is misclassified as 7
            [1.0, 3.0, 0.0],  # 2 is misclassified as 7
        ])
        result, predictions = classification_diagnostics(
            logits, labels, class_ids, torch.tensor([2, 7]), torch.tensor([11])
        )
        self.assertEqual(predictions.tolist(), [2, 7, 7, 7])
        self.assertEqual(result["confusion_class_ids"], [2, 7, 11])
        self.assertEqual(result["confusion_matrix"][2][1], 1)
        self.assertEqual(result["confusion_matrix"][0][1], 1)
        self.assertEqual(result["per_class"]["11"]["accuracy"], 0.0)
        self.assertLess(result["margin_novel"]["mean"], 0.0)

    def test_response_posthoc_computes_classwise_auc_and_correctness(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = torch.tensor([0, 0, 1, 0, 2, 2])
        raw = {
            "B8_E4": {
                "labels": labels,
                "gate_logit": torch.tensor([-2., -1., 0., 0.2, 1., 2.]),
                "gate": torch.tensor([.1, .2, .5, .55, .8, .9]),
                "residual_norm": torch.tensor([.01, .02, .03, .04, .1, .2]),
                "patch_residual_norm": torch.tensor([.02, .03, .04, .05, .1, .2]),
                "token_residual_norm": torch.tensor([.02, .03, .04, .05, .1, .2]),
                "relative_residual": torch.tensor([.001, .002, .003, .004, .01, .02]),
                "relative_patch_residual": torch.tensor([.001, .002, .003, .004, .01, .02]),
                "relative_token_residual": torch.tensor([.001, .002, .003, .004, .01, .02]),
            }
        }
        by_class, auc, correctness = response_posthoc(
            raw, labels, predictions, torch.tensor([0, 1]), torch.tensor([2])
        )
        self.assertEqual(by_class["B8_E4"]["2"]["sample_count"], 2)
        self.assertAlmostEqual(
            auc["B8_E4"]["gate"]["base_vs_novel_auc"], 1.0
        )
        self.assertEqual(correctness["B8_E4"]["incorrect"]["sample_count"], 1)
        self.assertEqual(set(raw["B8_E4"]) - {"labels"}, set(RESPONSE_METRICS))


if __name__ == "__main__":
    unittest.main()
