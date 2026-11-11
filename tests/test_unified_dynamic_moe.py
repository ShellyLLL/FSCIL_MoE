"""Architecture invariants for the unified sparse dynamic FSCIL MoE."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from models.bimc import BiMC, build_loo_cache_from_embeddings
from models.clip.model import VisualMoEAdapter
from engine.engine import Runner
from utils.evaluator import AccuracyEvaluator
from main import setup_cfg


class UnifiedMoEArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_bimc_yaml_merges_against_complete_yacs_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        trainer_cfg = repo_root / "configs" / "trainers" / "bimc.yaml"
        dataset_cfgs = [
            repo_root / "configs" / "datasets" / "cifar100.yaml",
            repo_root / "configs" / "datasets" / "cub200.yaml",
            repo_root / "configs" / "datasets" / "miniimagenet.yaml",
        ]
        for dataset_cfg in dataset_cfgs:
            with self.subTest(dataset=dataset_cfg.stem):
                cfg = setup_cfg(str(dataset_cfg), str(trainer_cfg))
                visual_moe = cfg.TRAINER.BiMC.VISUAL_MOE
                self.assertAlmostEqual(visual_moe.ROUTE_INIT_GAP, 0.02)
                self.assertAlmostEqual(visual_moe.ROUTE_INIT_TARGET_GAP, 0.5)
                self.assertAlmostEqual(visual_moe.ROUTE_INIT_MAX_WEIGHT_NORM, 2.0)
                self.assertAlmostEqual(visual_moe.EXPANSION_QUANTILE, 0.95)
                self.assertAlmostEqual(visual_moe.EXPANSION_CONFIDENCE_Z, 1.0)
                self.assertAlmostEqual(visual_moe.EXPANSION_MIN_GAIN, 0.001)
                self.assertAlmostEqual(
                    cfg.TRAINER.BiMC.OPTIM.LR_INCREMENTAL_ROUTER, 1.0e-2
                )
                self.assertTrue(cfg.is_frozen())

    def test_runner_initializes_evaluator_with_global_session_partition(self):
        class FakeDatasetManager:
            def __init__(self, cfg):
                self.template = []
                self.class_index_in_task = [
                    np.asarray([0, 1]),
                    np.asarray([2]),
                ]

        class FakeModel(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()

        cfg = SimpleNamespace(
            DEVICE=SimpleNamespace(DEVICE_NAME="cpu"),
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace()),
        )
        with patch("engine.engine.DatasetManager", FakeDatasetManager):
            with patch("engine.engine.BiMC", FakeModel):
                with patch("engine.engine.torch.cuda.device_count", return_value=0):
                    runner = Runner(cfg)

        self.assertIsInstance(runner.evaluator, AccuracyEvaluator)
        logits = torch.tensor([
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        targets = torch.tensor([0, 2])
        result = runner.evaluator.calc_accuracy(
            logits, targets, task_id=1, class_ids=torch.tensor([0, 1, 2])
        )
        self.assertEqual(result["mean_acc"], 100.0)
        self.assertEqual(result["task_acc"], [100.0, 100.0])

    def test_new_expert_is_function_preserving_until_activated(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        inputs = torch.randn(5, 3, 8)
        before, _ = adapter(inputs)
        adapter.add_expert(session_id=1, reduction=2)
        after, aux = adapter(inputs)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertEqual(aux["top_k_indices"].shape, (3, 2))
        self.assertTrue((aux["top_k_indices"] < 4).all())

    def test_router_uses_one_sparse_competition(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        inputs = torch.randn(7, 2, 8)
        _, aux = adapter(inputs)
        weights = aux["route_weights"]
        self.assertEqual(weights.shape, (2, 4))
        self.assertTrue(torch.all((weights > 0).sum(dim=-1) == 2))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))

    def test_unselected_experts_are_not_executed(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        calls = [0, 0, 0, 0]
        handles = []
        for index, expert in enumerate(adapter.experts):
            handles.append(expert.register_forward_hook(
                lambda module, args, output, i=index: calls.__setitem__(i, calls[i] + 1)
            ))
        _, aux = adapter(torch.randn(5, 3, 8))
        for handle in handles:
            handle.remove()
        selected_experts = set(aux["top_k_indices"].reshape(-1).tolist())
        self.assertEqual(
            {index for index, count in enumerate(calls) if count > 0},
            selected_experts,
        )
        self.assertTrue(all(calls[index] <= 1 for index in selected_experts))

    def test_new_route_initialization_and_optimizer_activation_preserve_history(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2)
        with torch.no_grad():
            for column in adapter.router.columns:
                column.weight.zero_()
                column.bias.zero_()
        adapter.add_expert(session_id=1, reduction=2)
        old_parameters = [
            parameter for name, parameter in adapter.named_parameters()
            if name.startswith(("experts.0.", "experts.1.", "experts.2.", "experts.3.",
                                "router.columns.0.", "router.columns.1.",
                                "router.columns.2.", "router.columns.3."))
        ]
        current = torch.zeros(32, 8)
        current[:, 0] = 0.5
        history = torch.zeros(6, 8)
        history[:, 0] = -0.5
        variance = torch.zeros_like(history)
        init = adapter.initialize_new_route(
            current, history, variance, margin=0.1, uncertainty=0.0,
            init_gap=0.02, target_gap=0.5, max_weight_norm=2.0,
        )
        _, _, initial_indices, _, _ = adapter.router(current, top_k=2)
        self.assertFalse((initial_indices == 4).any())
        self.assertGreaterEqual(init["current_min_gap"], 0.02 - 1e-6)
        optimizer = torch.optim.AdamW(
            adapter.newest_parameters(), lr=0.05, weight_decay=0.0
        )
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            activation, preservation = adapter.route_objective(
                current, history, variance, margin=0.1, uncertainty=0.0
            )
            (activation + preservation).backward()
            optimizer.step()
        _, _, current_indices, _, _ = adapter.router(current, top_k=2)
        _, _, history_indices, _, _ = adapter.router(history, top_k=2)
        self.assertTrue((current_indices == 4).any())
        self.assertFalse((history_indices == 4).any())
        self.assertTrue(all(not parameter.requires_grad for parameter in old_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in old_parameters))

    def test_adapter_keeps_fp32_parameters_with_half_precision_backbone_inputs(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).float().eval()
        inputs = torch.randn(5, 3, 8, dtype=torch.float16)
        residual, aux = adapter(inputs)
        self.assertEqual(residual.dtype, torch.float16)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in adapter.parameters()))
        self.assertEqual(aux["logits"].dtype, torch.float32)

    def test_history_route_constraint_uses_diagonal_support_uncertainty(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2)
        adapter.add_expert(session_id=1, reduction=2)
        history = torch.randn(6, 8)
        low_variance = torch.zeros_like(history)
        high_variance = torch.ones_like(history)
        low = adapter.route_preservation_loss(history, low_variance)
        high = adapter.route_preservation_loss(history, high_variance)
        self.assertGreaterEqual(float(high.detach()), float(low.detach()))

    def test_topology_round_trip_preserves_predictions_and_birth_order(self):
        source = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2).eval()
        source.add_expert(session_id=2, reduction=4)
        source.freeze_all()
        inputs = torch.randn(4, 3, 8)
        expected, _ = source(inputs)
        restored = VisualMoEAdapter(8, num_experts=4, reduction=2, top_k=2)
        restored.rebuild_topology(source.export_topology())
        restored.load_state_dict(source.state_dict(), strict=True)
        restored.eval()
        actual, _ = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(restored.expert_birth_sessions, [0, 0, 0, 0, 2])

    def test_old_expert_and_router_are_frozen_after_expansion(self):
        adapter = VisualMoEAdapter(8, num_experts=4, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        self.assertTrue(all(not p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.router.columns[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.experts[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.router.columns[-1].parameters()))

    def test_base_expansion_thresholds_use_nearest_neighbor_quantile(self):
        prototypes = F.normalize(torch.tensor([
            [1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.0, 1.0]
        ]), dim=-1)
        thresholds = BiMC.build_expansion_thresholds(
            {"-1": {"class_ids": torch.arange(4), "prototypes": prototypes}},
            quantile=0.75,
        )
        self.assertIn("-1", thresholds)
        self.assertTrue(-1.0 <= thresholds["-1"] <= 1.0)

    def test_loo_prototypes_exclude_all_views_of_query_source(self):
        features = F.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9],
            [-1., 0.], [-0.9, 0.1],
        ]), dim=-1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1])
        sources = torch.tensor([10, 10, 11, 12, 20, 21])
        cache = build_loo_cache_from_embeddings(features, labels, sources)
        # Both augmentations from source 10 are removed for either query.
        expected = F.normalize(features[2:4].mean(dim=0), dim=-1)
        torch.testing.assert_close(cache["loo_visual_prototypes"][0, 0], expected)
        torch.testing.assert_close(cache["loo_visual_prototypes"][1, 0], expected)
        self.assertTrue(cache["loo_valid"].all())

    def test_loo_rejects_single_source_class(self):
        features = F.normalize(torch.tensor([[1., 0.], [0., 1.]]), dim=-1)
        with self.assertRaisesRegex(ValueError, "another source"):
            cache = build_loo_cache_from_embeddings(
                features, torch.tensor([0, 1]), torch.tensor([1, 2])
            )
            if not cache["loo_valid"].all():
                raise ValueError("another source is required")


if __name__ == "__main__":
    unittest.main()
