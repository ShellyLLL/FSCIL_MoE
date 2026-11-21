import unittest

import torch

from engine.engine import Runner
from tools.session1_diagnostics import (
    RESPONSE_METRICS,
    binary_auc,
    classification_diagnostics,
    metric_summary,
    response_posthoc,
    gradient_objective_diagnostics,
    prototype_generalization_diagnostics,
    collect_support_generalization_diagnostics,
    preserve_diagnostic_state,
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-device regression")
    def test_prototype_generalization_handles_cpu_predictions_and_cuda_masks(self):
        device = torch.device("cuda")
        class_ids = torch.tensor([0, 1, 2])
        base_ids = torch.tensor([0])
        novel_ids = torch.tensor([1, 2])
        global_prototypes = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        ], device=device)
        support_features = torch.tensor([
            [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0],
        ])
        support_labels = torch.tensor([1, 1, 2, 2])
        loo = torch.tensor([
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ])
        test_features = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
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

    def test_prototype_generalization_rejects_misaligned_loo_cache(self):
        with self.assertRaises(ValueError):
            prototype_generalization_diagnostics(
                torch.randn(2, 3), torch.tensor([1, 1]), torch.randn(2, 1, 2),
                torch.randn(2, 3), torch.tensor([0, 1]),
                torch.randn(2, 3), torch.tensor([0, 1]),
                old_class_count=1, base_ids=torch.tensor([0]),
                novel_ids=torch.tensor([1]), logit_scale=10.0,
            )

    def test_incremental_bank_exposes_support_features_with_public_diagnostic_schema(self):
        features = torch.tensor([
            [1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8],
        ])
        labels = torch.tensor([1, 1, 2, 2])
        source_ids = torch.tensor([10, 11, 20, 21])
        cache = {
            "features": features,
            "labels": labels,
            "source_ids": source_ids,
            "classes": torch.tensor([1, 2]),
            "loo_visual_prototypes": torch.tensor([
                [[0.8, 0.6], [0.6, 0.8]],
                [[1.0, 0.0], [0.6, 0.8]],
                [[1.0, 0.0], [0.6, 0.8]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]),
            "loo_valid": torch.ones(4, dtype=torch.bool),
        }

        class DummyModel:
            def build_leave_one_out_prototypes(self, loader, class_index):
                return cache

        runner = Runner.__new__(Runner)
        runner.model_without_dp = DummyModel()
        runner.device = torch.device("cpu")
        bank = runner._build_incremental_bank(
            [], torch.tensor([1, 2]),
            [{"image_proto": torch.tensor([[1.0, 0.0]]), "class_ids": [0]}],
        )

        self.assertTrue(torch.equal(bank["features"], features))
        self.assertTrue(torch.equal(bank["labels"], labels))
        self.assertTrue(torch.equal(bank["source_ids"], source_ids))
        self.assertEqual(bank["global_ids"].tolist(), [0, 1, 2])
        self.assertEqual(bank["global_full"].shape, (3, 2))
        self.assertNotIn("loo_valid", bank)

    def test_support_generalization_builds_loo_with_novel_ids_not_merged_ids(self):
        support_features = torch.tensor([
            [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0],
        ])
        support_labels = torch.tensor([1, 1, 2, 2])
        loo = torch.tensor([
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ])
        test_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        test_labels = torch.tensor([0, 1, 2])
        bank = {
            "features": support_features,
            "labels": support_labels,
            "novel_loo": loo,
            "global_full": torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            "global_ids": torch.tensor([0, 1, 2]),
            "old_class_count": 1,
        }

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def inference_all_img_feature(self, loader):
                return test_features, test_labels, None, None

        class DummyRunner:
            def __init__(self):
                self.model = DummyModel()
                self.model_without_dp = self.model
                self.received_class_ids = None

            def _moe_adapters(self):
                return []

            def _build_incremental_bank(self, loader, class_ids, old_states):
                self.received_class_ids = torch.as_tensor(class_ids).clone()
                return bank

        runner = DummyRunner()
        result = collect_support_generalization_diagnostics(
            runner, [], [], torch.tensor([2, 1]), [{"class_ids": [0]}],
            torch.tensor([0]), torch.tensor([1, 2]), logit_scale=10.0,
        )
        self.assertEqual(runner.received_class_ids.tolist(), [1, 2])
        self.assertAlmostEqual(result["support_loo"]["novel_accuracy"], 1.0)
        self.assertAlmostEqual(result["test"]["base_accuracy"], 1.0)
        self.assertAlmostEqual(result["test"]["novel_accuracy"], 1.0)

    def test_support_generalization_rejects_merged_base_novel_support_class_ids(self):
        class DummyRunner:
            def __init__(self):
                self.builder_called = False

            def _build_incremental_bank(self, *args):
                self.builder_called = True
                raise AssertionError("The LOO bank must not be built with merged class IDs.")

        runner = DummyRunner()
        with self.assertRaisesRegex(ValueError, "must match novel_ids exactly"):
            collect_support_generalization_diagnostics(
                runner, [], [], torch.tensor([0, 1, 2]), [{"class_ids": [0]}],
                torch.tensor([0]), torch.tensor([1, 2]), logit_scale=10.0,
            )
        self.assertFalse(runner.builder_called)

    def test_diagnostic_context_restores_model_parameters_modes_gradients_and_rng(self):
        class DummyRunner:
            def __init__(self):
                self.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
                self.model_without_dp = self.model
                self.adapter = self.model[0]
                self.adapter._inference_disabled_expert_ids = frozenset({3})
            def _moe_adapters(self):
                return [(0, self.adapter)]

        runner = DummyRunner()
        runner.model.train()
        runner.model[1].eval()
        parameter = runner.adapter.weight
        parameter.grad = torch.ones_like(parameter)
        original_weight = parameter.detach().clone()
        original_grad = parameter.grad.detach().clone()
        original_modes = [module.training for module in runner.model.modules()]
        original_rng = torch.get_rng_state().clone()
        original_mask = runner.adapter._inference_disabled_expert_ids

        with preserve_diagnostic_state(runner):
            with torch.no_grad():
                parameter.add_(10.0)
            parameter.requires_grad_(False)
            parameter.grad = None
            runner.model.eval()
            runner.adapter._inference_disabled_expert_ids = frozenset()
            torch.rand(5)

        self.assertTrue(torch.equal(parameter, original_weight))
        self.assertTrue(torch.equal(parameter.grad, original_grad))
        self.assertTrue(parameter.requires_grad)
        self.assertEqual([module.training for module in runner.model.modules()], original_modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        self.assertEqual(runner.adapter._inference_disabled_expert_ids, original_mask)


if __name__ == "__main__":
    unittest.main()
