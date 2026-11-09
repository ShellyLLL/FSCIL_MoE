"""Architecture invariants for Prototype-Confusion Guided Dynamic MoE-FSCIL."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from models.prototype_confusion_moe import (
    UnifiedTopKRouter, DynamicLayerMoE, source_exclusive_means,
    prototype_confusion, make_apc_reference, balanced_risk,
)
from engine.prototype_confusion_engine import PrototypeConfusionRunner


class PrototypeConfusionTests(unittest.TestCase):
    def test_apc_uses_base_reference_percentiles_and_orders_layers(self):
        torch.manual_seed(4)
        base = F.normalize(torch.randn(12, 8), dim=-1)
        reference = make_apc_reference({2: base, 5: base})
        old = {2: base[:8], 5: base[:8]}
        labels = torch.tensor([10, 10, 11, 11])
        features = {
            2: torch.cat([base[:2], base[2:4]], dim=0),
            5: torch.cat([base[:2], -base[2:4]], dim=0),
        }
        scores = prototype_confusion(features, labels, [10, 11], old, reference)
        self.assertEqual(set(scores), {2, 5})
        for item in scores.values():
            self.assertTrue(torch.isfinite(item['scores']).all())
            self.assertGreaterEqual(item['mean'], 0.0)
            self.assertLessEqual(item['mean'], 1.0)

    def test_source_exclusion_removes_all_views_of_same_source(self):
        features = F.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0.8, 0.2], [0., 1.], [-1., 0.]
        ]), dim=-1)
        labels = torch.tensor([0, 0, 0, 1, 1])
        sources = torch.tensor([7, 7, 10, 8, 9])
        bank = source_exclusive_means(
            features, labels, sources, [0, 1], query_indices=[0, 3]
        )
        # Both views of source 7 are excluded for the first query.
        torch.testing.assert_close(bank[0, 0], features[2])
        # The second query excludes source 8, not its other class-1 peer.
        torch.testing.assert_close(bank[1, 1], features[4])
        self.assertEqual(tuple(bank.shape), (2, 2, 2))

    def test_source_exclusive_banks_are_bounded_to_requested_queries(self):
        features = F.normalize(torch.randn(30, 12), dim=-1)
        labels = torch.arange(30) // 3
        sources = torch.arange(30)
        bank = source_exclusive_means(features, labels, sources, range(10),
                                      query_indices=[1, 4, 9])
        self.assertEqual(tuple(bank.shape), (3, 10, 12))

    def test_router_is_one_sparse_topk_softmax_over_all_experts(self):
        torch.manual_seed(2)
        router = UnifiedTopKRouter(6, 5)
        x = torch.randn(4, 6)
        logits, weights, selected = router(x, top_k=2)
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(weights.shape), (4, 5))
        self.assertTrue(torch.all((weights > 0).sum(1) == 2))
        torch.testing.assert_close(weights.sum(1), torch.ones(4))
        _, excluded_weights, _ = router(x, top_k=2, excluded=[4])
        self.assertTrue(torch.equal(excluded_weights[:, 4], torch.zeros(4)))
        self.assertTrue(torch.all((excluded_weights > 0).sum(1) == 2))

    def test_incremental_expert_is_appended_to_same_router(self):
        torch.manual_seed(3)
        layer = DynamicLayerMoE(8, 8, base_experts=3, base_reduction=2,
                               incremental_reduction=4, top_k=2, scale_init=0.5)
        self.assertEqual(layer.num_experts, 3)
        self.assertEqual(len(layer.router.columns), 3)
        new = torch.randn(5, 8) + 2.0
        old = torch.randn(20, 8) - 2.0
        layer.add_expert(1, new, old)
        self.assertEqual(layer.num_experts, 4)
        self.assertEqual(len(layer.router.columns), 4)
        trainable = [name for name, p in layer.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(('experts.3.', 'router.columns.3.'))
                            for name in trainable))
        layer.freeze_all()
        self.assertFalse(any(p.requires_grad for p in layer.parameters()))

    def test_new_gate_initialization_preserves_historical_topk_exactly(self):
        torch.manual_seed(31)
        layer = DynamicLayerMoE(8, 8, base_experts=4, base_reduction=2,
                               incremental_reduction=4, top_k=2, scale_init=0.5)
        old = torch.randn(32, 8)
        with torch.no_grad():
            old_x = layer.norm(old)
            old_logits, old_weights, old_ids = layer.router(old_x, layer.top_k)
            old_output, _ = layer(old)
            layer.add_expert(1, torch.randn(6, 8) + 1.5, old)
            new_logits, new_weights, new_ids = layer.router(layer.norm(old), layer.top_k)
            new_output, _ = layer(old)
        torch.testing.assert_close(new_ids, old_ids)
        torch.testing.assert_close(new_weights[:, :-1], old_weights)
        torch.testing.assert_close(new_weights[:, -1], torch.zeros(old.size(0)))
        torch.testing.assert_close(new_output, old_output, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(new_logits[:, -1] < old_logits.topk(2, dim=1).values[:, -1]))

    def test_new_gate_receives_gradient_before_sparse_activation(self):
        torch.manual_seed(32)
        layer = DynamicLayerMoE(8, 8, base_experts=3, base_reduction=2,
                               incremental_reduction=4, top_k=2, scale_init=0.5)
        new = torch.randn(7, 8) + 1.0
        old = torch.randn(24, 8)
        layer.add_expert(1, new, old)
        terms = layer.route_margin_loss(new, old, margin=0.05)
        loss = terms["new"] + terms["historical"]
        loss.backward()
        gate = layer.router.columns[-1]
        self.assertIsNotNone(gate.weight.grad)
        self.assertGreater(float(gate.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(terms["new"].item()), 0.0)
        self.assertEqual(float(terms["historical_intrusion_rate"].item()), 0.0)

    def test_router_rejects_invalid_topk_instead_of_silently_changing_it(self):
        router = UnifiedTopKRouter(4, 3)
        with self.assertRaises(ValueError):
            router(torch.randn(2, 4), top_k=4)
        with self.assertRaises(ValueError):
            router(torch.randn(2, 4), top_k=0)

    def test_visual_intra_modal_calibration_is_applied_to_per_query_banks(self):
        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.cfg = type("Cfg", (), {
            "DATASET": type("DatasetCfg", (), {"NUM_INIT_CLS": 2})(),
            "TRAINER": type("TrainerCfg", (), {
                "PCMOE": type("PCMOECfg", (), {
                    "CALIBRATION": type("CalibrationCfg", (), {
                        "LAMBDA_VISUAL": 0.5, "VISUAL_TAU": 8.0
                    })()
                })()
            })()
        })()
        prototypes = F.normalize(torch.tensor([
            [[1., 0.], [0., 1.], [0.8, 0.6]],
            [[1., 0.], [0., 1.], [0.6, 0.8]],
        ]), dim=-1)
        calibrated = runner._calibrate_visual_bank(prototypes)
        self.assertEqual(tuple(calibrated.shape), tuple(prototypes.shape))
        torch.testing.assert_close(calibrated[:, :2], prototypes[:, :2])
        self.assertFalse(torch.allclose(calibrated[:, 2], prototypes[:, 2]))

    def test_source_exclusive_bank_supports_old_centroids_and_current_new_samples(self):
        features = F.normalize(torch.tensor([
            [1., 0.], [0.9, 0.1], [0.8, 0.2], [0., 1.]
        ]), dim=-1)
        labels = torch.tensor([0, 1, 1, 1])
        # Class 0 is a single historical centroid with a negative synthetic
        # source id; class 1 has current-session samples with real source ids.
        sources = torch.tensor([-1, 20, 21, 22])
        bank = source_exclusive_means(
            features, labels, sources, [0, 1], query_indices=[1, 2, 3]
        )
        self.assertEqual(tuple(bank.shape), (3, 2, 2))
        torch.testing.assert_close(bank[:, 0], features[0].expand(3, -1))
        for row, query in enumerate([1, 2, 3]):
            remaining = features[(labels == 1) & (sources != sources[query])]
            torch.testing.assert_close(bank[row, 1], F.normalize(remaining.mean(0), dim=0))

    def test_completed_class_memory_is_compressed_to_single_prototypes(self):
        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.model = type("Model", (), {"layer_ids": [2, 5]})()
        runner.class_memory = {
            10: {
                "fit": {
                    "kind": "samples",
                    "raw": torch.tensor([[1., 0.], [0.8, 0.2], [0.9, 0.1]]),
                    "sources": torch.tensor([1, 2, 3]),
                    "layers": {
                        2: torch.tensor([[1., 0.], [0.8, 0.2], [0.9, 0.1]]),
                        5: torch.tensor([[0., 1.], [0.2, 0.8], [0.1, 0.9]]),
                    },
                }
            }
        }
        runner._compress_class_memory()
        runner._assert_prototype_only_memory()
        item = runner.class_memory[10]["fit"]
        self.assertEqual(item["kind"], "prototype")
        self.assertNotIn("sources", item)
        self.assertEqual(tuple(item["raw"].shape), (1, 2))
        self.assertEqual(tuple(item["layers"][2].shape), (1, 2))
        self.assertEqual(item["count"], 3)

    def test_checkpoint_guard_rejects_sample_level_historical_memory(self):
        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.class_memory = {
            10: {"fit": {
                "kind": "samples",
                "raw": torch.randn(3, 4),
                "sources": torch.tensor([1, 2, 3]),
                "layers": {2: torch.randn(3, 4)},
            }}
        }
        with self.assertRaises(RuntimeError):
            runner._assert_prototype_only_memory()

    def test_memory_cache_uses_one_old_centroid_and_current_novel_supports(self):
        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.model = type("Model", (), {"layer_ids": [2]})()
        runner.device = "cpu"
        runner.class_ids = [10, 20]
        runner.class_memory = {
            10: {"fit": {
                "kind": "prototype", "raw": torch.tensor([[1., 0.]]),
                "layers": {2: torch.tensor([[0.5, 0.5]])}, "count": 5,
            }},
            20: {"fit": {
                "kind": "samples", "raw": torch.tensor([[0., 1.], [0.1, 0.9]]),
                "sources": torch.tensor([100, 101]),
                "layers": {2: torch.tensor([[0.2, 0.8], [0.3, 0.7]])},
            }},
        }
        cache = runner._memory_cache("fit")
        self.assertEqual(tuple(cache["raw"].shape), (3, 2))
        self.assertEqual(cache["class_counts"], {10: 5, 20: 2})
        self.assertEqual(cache["sources"][0].item(), -1)
        self.assertTrue(torch.equal(cache["prototype_mask"], torch.tensor([True, False, False])))

    def test_support_split_reserves_two_fit_sources_for_leave_one_out(self):
        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.cfg = type("Cfg", (), {"SEED": 7})()
        cache = {
            "raw": torch.randn(3, 4),
            "labels": torch.tensor([10, 10, 10]),
            "sources": torch.tensor([100, 101, 102]),
            "layers": {2: torch.randn(3, 4)},
        }
        with self.assertRaisesRegex(ValueError, "requires >=4 distinct source images"):
            runner._split(cache, 1)

        cache = {
            "raw": torch.randn(4, 4),
            "labels": torch.tensor([10, 10, 10, 10]),
            "sources": torch.tensor([100, 101, 102, 103]),
            "layers": {2: torch.randn(4, 4)},
        }
        fit, calibration, risk = runner._split(cache, 1)
        self.assertEqual(fit["raw"].size(0), 2)
        self.assertEqual(calibration["raw"].size(0), 1)
        self.assertEqual(risk["raw"].size(0), 1)

    def test_temperature_fitting_runs_and_respects_configured_bounds(self):
        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.device = "cpu"
        runner.visual_temperature = 0.1
        runner.text_temperature = 0.1
        runner.cfg = type("Cfg", (), {
            "TRAINER": type("TrainerCfg", (), {
                "PCMOE": type("PCMOECfg", (), {
                    "CALIBRATION": type("CalibrationCfg", (), {
                        "MIN_TEMPERATURE": 0.02, "MAX_TEMPERATURE": 2.0
                    })()
                })()
            })()
        })()
        class Model:
            def compose(self, raw, layers, disabled_by_layer=None):
                return F.normalize(raw, dim=-1)
        runner.model = Model()
        runner._calibration_features = lambda: {
            "raw": F.normalize(torch.tensor([
                [1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]
            ]), dim=-1),
            "layers": {},
            "labels": torch.tensor([10, 10, 20, 20]),
        }
        bank = {
            "visual": F.normalize(torch.tensor([[1., 0.], [0., 1.]]), dim=-1),
            "text": F.normalize(torch.tensor([[0.9, 0.1], [0.1, 0.9]]), dim=-1),
            "ids": torch.tensor([10, 20]),
        }
        visual, textual = runner._fit_temperatures(bank)
        self.assertTrue(0.02 <= visual <= 2.0)
        self.assertTrue(0.02 <= textual <= 2.0)
        self.assertTrue(np.isfinite([visual, textual]).all())

    def test_incremental_transaction_exception_restores_exact_model_topology_and_weights(self):
        from torch import nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_ids = [2]
                self.adapters = nn.ModuleDict({
                    "2": DynamicLayerMoE(
                        4, 4, base_experts=2, base_reduction=2,
                        incremental_reduction=2, top_k=1, scale_init=0.5
                    )
                })
                self.register_buffer("temperature", torch.tensor([0.1, 0.1]))
            def topology(self):
                return {key: value.topology() for key, value in self.adapters.items()}
            def add_expert(self, layer, session, new, old, route_margin=0.05):
                return self.adapters[str(layer)].add_expert(
                    session, new, old, route_margin=route_margin
                )
            def set_incremental_trainable(self, layers):
                for adapter in self.adapters.values():
                    adapter.freeze_all()
                for layer in layers:
                    self.adapters[str(layer)].set_newest_trainable()
            def freeze_all(self):
                for adapter in self.adapters.values():
                    adapter.freeze_all()
            def set_temperatures(self, visual, text):
                self.temperature.copy_(torch.tensor([visual, text]))
            def rebuild_topology(self, topology):
                for key, spec in topology.items():
                    self.adapters[key].rebuild(spec)

        runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
        runner.device = "cpu"
        runner.model = FakeModel()
        runner.class_ids = [10, 20]
        runner.class_memory = {
            10: {"fit": {
                "kind": "prototype", "raw": torch.tensor([[1., 0., 0., 0.]]),
                "sources": None,
                "layers": {2: torch.tensor([[0.2, 0.1, 0.3, 0.4]])}, "count": 4,
            }},
            20: {"fit": {
                "kind": "samples",
                "raw": torch.tensor([[0., 1., 0., 0.], [0., 0.9, 0.1, 0.]]),
                "sources": torch.tensor([200, 201]),
                "layers": {2: torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])},
            }},
        }
        runner.visual_temperature, runner.text_temperature = 0.1, 0.1
        runner.text_prototypes = {10: torch.tensor([1., 0., 0., 0.]),
                                  20: torch.tensor([0., 1., 0., 0.])}
        runner.description_prototypes = dict(runner.text_prototypes)
        runner.text_consistency = {10: torch.tensor(1.), 20: torch.tensor(1.)}
        runner.cfg = type("Cfg", (), {
            "TRAINER": type("TrainerCfg", (), {
                "PCMOE": type("PCMOECfg", (), {
                    "LOSS": type("LossCfg", (), {
                        "ROUTE_MARGIN": 0.05, "NEW_ROUTE_WEIGHT": 1.0,
                        "HISTORICAL_ROUTE_WEIGHT": 1.0
                    })(),
                    "OPTIM": type("OptimCfg", (), {
                        "INCREMENTAL_LR": 1e-3, "WEIGHT_DECAY": 0.0,
                        "INCREMENTAL_EPOCHS": 0
                    })(),
                    "CALIBRATION": type("CalibrationCfg", (), {
                        "LAMBDA_TEXT": 0.5
                    })(),
                    "EXPANSION": type("ExpansionCfg", (), {
                        "APC_MIN_PERCENTILE": 0.8, "MAX_LAYERS_PER_SESSION": 1
                    })()
                })()
            })()
        })()
        before_topology = runner.model.topology()
        before_state = {k: v.clone() for k, v in runner.model.state_dict().items()}
        fit_new = {
            "raw": torch.tensor([[0., 1., 0., 0.], [0., 0.8, 0.2, 0.]]),
            "labels": torch.tensor([20, 20]),
            "sources": torch.tensor([200, 201]),
            "layers": {2: torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])},
        }
        def fail_after_candidate_training(*args, **kwargs):
            raise RuntimeError("injected failure after candidate expert training")
        runner._fit_bank_and_temperatures = fail_after_candidate_training
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            runner._train_incremental(
                1, fit_new, [20], {2: {"mean": 0.95}}
            )
        self.assertEqual(runner.model.topology(), before_topology)
        after_state = runner.model.state_dict()
        self.assertEqual(set(after_state), set(before_state))
        for key in before_state:
            torch.testing.assert_close(after_state[key], before_state[key])
        self.assertEqual(runner.class_ids, [10, 20])
        self.assertEqual(runner.class_memory[20]["fit"]["kind"], "samples")

    def test_checkpoint_round_trip_restores_prototype_memory_and_session_state(self):
        import tempfile
        from pathlib import Path

        class FakeModel:
            def __init__(self):
                self.value = torch.tensor([3.0])
                self.temperatures = None
                self.restored_topology = None
            def state_dict(self):
                return {"value": self.value.clone()}
            def topology(self):
                return {"experts": [0, 1]}
            def rebuild_topology(self, topology):
                self.restored_topology = topology
            def load_state_dict(self, state, strict=True):
                self.value = state["value"].clone()
            def freeze_all(self):
                pass
            def set_temperatures(self, visual, text):
                self.temperatures = (float(visual), float(text))

        class FakeData:
            def get_support_manifest(self):
                return {"version": 1, "sessions": {}}
            def load_support_manifest(self, manifest):
                self.loaded_manifest = manifest

        with tempfile.TemporaryDirectory() as directory:
            runner = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
            runner.cfg = type("Cfg", (), {
                "TRAINER": type("TrainerCfg", (), {
                    "PCMOE": type("PCMOECfg", (), {
                        "CHECKPOINT": type("CheckpointCfg", (), {"DIR": directory, "RESUME": ""})()
                    })()
                })()
            })()
            runner.model = FakeModel()
            runner.data = FakeData()
            runner.class_ids = [10]
            runner.class_memory = {10: {"fit": {
                "kind": "prototype", "raw": torch.tensor([[1., 0.]]),
                "layers": {2: torch.tensor([[0.5, 0.5]])}, "count": 3,
            }}}
            runner.text_prototypes = {10: torch.tensor([1., 0.])}
            runner.description_prototypes = {10: torch.tensor([1., 0.])}
            runner.text_consistency = {10: torch.tensor(0.9)}
            runner.apc_reference = {2: torch.tensor([0.2])}
            runner.base_layer_prototypes = {2: {10: torch.tensor([0.5, 0.5])}}
            runner.visual_temperature, runner.text_temperature = 0.2, 0.3
            runner.inter_modal_beta = 0.4
            runner.diagnostics = [{"session": 2}]
            runner.base_accuracy = 71.5
            runner.start_session = 0
            runner._save(2)

            restored = PrototypeConfusionRunner.__new__(PrototypeConfusionRunner)
            restored.cfg = type("Cfg", (), {
                "TRAINER": type("TrainerCfg", (), {
                    "PCMOE": type("PCMOECfg", (), {
                        "CHECKPOINT": type("CheckpointCfg", (), {
                            "RESUME": str(Path(directory) / "session_02.pth")
                        })()
                    })()
                })()
            })()
            restored.model = FakeModel()
            restored.data = FakeData()
            restored._load_resume()
            self.assertEqual(restored.start_session, 3)
            self.assertEqual(restored.class_ids, [10])
            self.assertEqual(restored.class_memory[10]["fit"]["kind"], "prototype")
            self.assertEqual(restored.inter_modal_beta, 0.4)
            self.assertEqual(restored.model.temperatures, (0.2, 0.3))
            self.assertEqual(restored.model.restored_topology, {"experts": [0, 1]})
            torch.testing.assert_close(restored.model.value, torch.tensor([3.0]))

    def test_balanced_risk_requires_every_query_label_in_global_bank(self):
        logits = torch.tensor([[4., 0., 0.], [0., 3., 0.], [0., 0., 2.]])
        labels = torch.tensor([10, 20, 30])
        risk, per_class = balanced_risk(logits, labels, [10, 20, 30])
        self.assertLess(float(risk), 0.2)
        self.assertEqual(set(per_class), {10, 20, 30})
        with self.assertRaises(ValueError):
            balanced_risk(logits, labels, [10, 20])
        with self.assertRaises(ValueError):
            balanced_risk(logits[:2], labels[:2], [10, 20, 30])


if __name__ == '__main__':
    unittest.main()
