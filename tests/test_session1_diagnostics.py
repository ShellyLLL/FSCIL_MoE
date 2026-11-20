import unittest

import torch

from tools.session1_diagnostics import (
    RESPONSE_METRICS,
    binary_auc,
    classification_diagnostics,
    metric_summary,
    response_posthoc,
    gradient_objective_diagnostics,
    prototype_generalization_diagnostics,
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

    def test_weighted_gradient_diagnostics_match_training_objective_and_clip(self):
        ce = torch.tensor([3.0, 4.0])
        pd = torch.tensor([-1.0, 2.0])
        result = gradient_objective_diagnostics(
            ce, pd, pd_weight=0.5, clip_max_norm=1.0
        )
        expected = ce + 0.5 * pd
        expected_norm = torch.linalg.vector_norm(expected).item()
        self.assertAlmostEqual(
            result["global_combined_gradient_norm_pre_clip"], expected_norm, places=6
        )
        self.assertAlmostEqual(
            result["global_clip_coefficient"], 1.0 / (expected_norm + 1e-6), places=6
        )
        self.assertAlmostEqual(
            result["ce_weighted_pd_cosine"],
            float(torch.dot(ce, 0.5 * pd) / (
                torch.linalg.vector_norm(ce) * torch.linalg.vector_norm(0.5 * pd)
            )),
            places=6,
        )
        no_clip = gradient_objective_diagnostics(ce, pd, pd_weight=0.0, clip_max_norm=100.0)
        self.assertEqual(no_clip["global_clip_coefficient"], 1.0)

    def test_prototype_generalization_uses_source_exclusive_loo(self):
        class_ids = torch.tensor([0, 1, 2])
        base_ids = torch.tensor([0])
        novel_ids = torch.tensor([1, 2])
        global_prototypes = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        support_features = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ])
        support_labels = torch.tensor([1, 1, 2, 2])
        loo = torch.tensor([
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ])
        test_features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        test_labels = torch.tensor([0, 1, 2])
        result = prototype_generalization_diagnostics(
            support_features, support_labels, loo,
            test_features, test_labels, global_prototypes, class_ids,
            old_class_count=1, base_ids=base_ids, novel_ids=novel_ids,
            logit_scale=10.0,
        )
        self.assertAlmostEqual(result["support_loo"]["novel_accuracy"], 1.0)
        self.assertAlmostEqual(result["test"]["base_accuracy"], 1.0)
        self.assertAlmostEqual(result["test"]["novel_accuracy"], 1.0)
        self.assertAlmostEqual(result["novel_support_loo_minus_test_accuracy"], 0.0)
        self.assertIn("participated in incremental optimization", result["interpretation_limit"])

    def test_prototype_generalization_rejects_misaligned_loo_cache(self):
        with self.assertRaises(ValueError):
            prototype_generalization_diagnostics(
                torch.randn(2, 3), torch.tensor([1, 1]), torch.randn(2, 1, 2),
                torch.randn(2, 3), torch.tensor([0, 1]),
                torch.randn(2, 3), torch.tensor([0, 1]),
                old_class_count=1, base_ids=torch.tensor([0]),
                novel_ids=torch.tensor([1]), logit_scale=10.0,
            )


if __name__ == "__main__":
    unittest.main()
