"""Unit tests for the AE-driven, single-feature-space FSCIL architecture."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip.model import RepresentationDescriptor, VisualMoEAdapter, VisionTransformer
from models.bimc import BiMC
from engine.engine import Runner


class TinyVisualCLIP(nn.Module):
    """Small real ViT used to test the production CLS/Router feature path."""

    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer(
            input_resolution=8, patch_size=4, width=8, layers=2, heads=2, output_dim=6
        )
        for block in self.visual.transformer.resblocks:
            block.moe_adapter = VisualMoEAdapter(
                8, num_experts=2, reduction=2, descriptor_dim=4, descriptor_buffer_size=16
            )

    def encode_image(self, images, return_moe_aux=False):
        return self.visual(images, return_moe_aux=return_moe_aux)


class UnifiedDynamicMoETests(unittest.TestCase):
    def test_sema_descriptor_uses_linear_ae_and_rolling_500_error_buffer(self):
        torch.manual_seed(3)
        descriptor = RepresentationDescriptor(8, descriptor_dim=3, buffer_size=500)
        values = torch.randn(600, 8)
        optimizer = torch.optim.AdamW(descriptor.parameters(), lr=1e-2)
        for _ in range(3):
            optimizer.zero_grad()
            loss = F.mse_loss(descriptor(values), values)
            loss.backward()
            optimizer.step()
        errors = descriptor.reconstruction_error(values)
        descriptor.update_reference(errors)
        self.assertEqual(int(descriptor.error_count.item()), 500)
        torch.testing.assert_close(descriptor.error_history, errors[-500:])
        self.assertTrue(torch.isfinite(descriptor.standardized_score(values[:7])).all())

    def test_descriptor_reference_ring_buffer_wraps_without_stale_statistics(self):
        descriptor = RepresentationDescriptor(4, descriptor_dim=2, buffer_size=5)
        descriptor.update_reference(torch.tensor([1., 2., 3.]))
        descriptor.update_reference(torch.tensor([4., 5., 6., 7.]))
        self.assertEqual(int(descriptor.error_count.item()), 5)
        torch.testing.assert_close(
            descriptor.error_history.sort().values, torch.tensor([3., 4., 5., 6., 7.])
        )
        torch.testing.assert_close(descriptor.error_mean, torch.tensor(5.0))
        torch.testing.assert_close(descriptor.error_std, torch.tensor(2.5).sqrt())
        descriptor.update_reference(torch.tensor([8., 9.]))
        torch.testing.assert_close(
            descriptor.error_history.sort().values, torch.tensor([5., 6., 7., 8., 9.])
        )
        torch.testing.assert_close(descriptor.error_mean, torch.tensor(7.0))

    def test_descriptor_expansion_score_is_signed_and_high_error_only(self):
        descriptor = RepresentationDescriptor(8, descriptor_dim=3, buffer_size=8)
        descriptor.update_reference(torch.tensor([1.0, 2.0, 3.0]))
        descriptor.reconstruction_error = lambda x: torch.tensor([0.0, 2.0, 4.0])
        scores = descriptor.standardized_score(torch.zeros(3, 8))
        torch.testing.assert_close(scores, torch.tensor([-2.0, 0.0, 2.0]))
        # An unusually low reconstruction error is evidence of coverage,
        # not a novelty signal that should trigger expert expansion.
        self.assertLess(float(scores[0]), 0.0)
        self.assertGreater(float(scores[2]), 0.0)

    def test_router_and_descriptor_share_the_exact_layer_normalized_cls_input(self):
        torch.manual_seed(13)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2,
                                   descriptor_dim=4, descriptor_buffer_size=8).eval()
        tokens = torch.randn(5, 3, 8)
        captured = []
        handle = adapter.router.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        try:
            _, aux = adapter(tokens)
        finally:
            handle.remove()
        expected_cls = adapter.layer_norm(tokens)[0]
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0], expected_cls)
        torch.testing.assert_close(aux["cls_in"], expected_cls)
        self.assertEqual(tuple(aux["cls_in"].shape), (tokens.size(1), tokens.size(2)))
        self.assertFalse(torch.allclose(aux["cls_in"], adapter.layer_norm(tokens).mean(dim=0)))

    def test_incremental_expert_append_preserves_function_at_initialization(self):
        torch.manual_seed(5)
        adapter = VisualMoEAdapter(8, num_experts=3, reduction=2,
                                   descriptor_dim=4, descriptor_buffer_size=500).eval()
        x = torch.randn(6, 5, 8)
        before, before_aux = adapter(x)
        expert_id = adapter.add_expert(session_id=1, reduction=4)
        after, aux = adapter(x)
        self.assertEqual(expert_id, 3)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(
            before_aux["route_weights"], aux["route_weights"][:, :adapter.base_expert_count],
            rtol=0, atol=0
        )
        base_weights = aux["route_weights"][:, :adapter.base_expert_count]
        inc_weights = aux["route_weights"][:, adapter.base_expert_count:]
        torch.testing.assert_close(base_weights.sum(-1), torch.ones(x.size(1)))
        self.assertTrue(((inc_weights > 0) & (inc_weights < 1)).all())

    def test_incremental_append_only_unfreezes_new_expert_and_new_router_column(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=3, reduction=4)
        self.assertTrue(all(not p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.experts[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.router.columns[:-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.router.columns[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for d in adapter.descriptors for p in d.parameters()))
        self.assertEqual(adapter.expert_birth_sessions, [0, 0, 3])

    def test_descriptor_scores_reject_uncalibrated_deployed_experts(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        with self.assertRaisesRegex(RuntimeError, "uncalibrated descriptors"):
            adapter.descriptor_scores(torch.randn(3, 8))

    def test_blockwise_collection_uses_exact_router_cls_and_preserves_sample_identity(self):
        torch.manual_seed(17)
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        model.clip_model = TinyVisualCLIP()
        model.device = torch.device("cpu")
        model.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            VISUAL_MOE=SimpleNamespace(ENABLE=True)
        )))
        model.parse_batch = lambda batch: (batch["images"], batch["labels"])
        captured = {0: [], 1: []}
        hooks = []
        for index, block in enumerate(model.clip_model.visual.transformer.resblocks):
            hooks.append(block.moe_adapter.router.register_forward_pre_hook(
                lambda _module, args, i=index: captured[i].append(args[0].detach().cpu().clone())
            ))
        batches = [
            {"images": torch.randn(2, 3, 8, 8), "labels": torch.tensor([4, 5]),
             "source_ids": torch.tensor([101, 102])},
            {"images": torch.randn(1, 3, 8, 8), "labels": torch.tensor([5]),
             "source_ids": torch.tensor([103])},
        ]
        try:
            features = model.collect_blockwise_cls_features(batches)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(sorted(features), [0, 1])
        for block_idx in (0, 1):
            torch.testing.assert_close(
                features[block_idx]["cls_in"], torch.cat(captured[block_idx], dim=0)
            )
            self.assertEqual(features[block_idx]["labels"].tolist(), [4, 5, 5])
            self.assertEqual(features[block_idx]["source_ids"].tolist(), [101, 102, 103])

    def test_base_freeze_policy_only_opens_adapters_and_keeps_backbone_eval(self):
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        model.clip_model = TinyVisualCLIP()
        model.device = torch.device("cpu")
        model.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            VISUAL_MOE=SimpleNamespace(ENABLE=True)
        )))
        model._apply_freeze_policy()
        trainable = [name for name, parameter in model.clip_model.named_parameters()
                     if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("moe_adapter" in name and ".descriptors." not in name
                            for name in trainable))
        self.assertFalse(model.clip_model.visual.proj.requires_grad)
        model.train()
        self.assertFalse(model.clip_model.training)
        for block in model.clip_model.visual.transformer.resblocks:
            self.assertTrue(block.moe_adapter.training)
            self.assertFalse(any(p.requires_grad for p in block.moe_adapter.descriptors.parameters()))

    def test_adapter_forward_matches_sample_gates_broadcast_over_all_tokens(self):
        torch.manual_seed(19)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2).eval()
        adapter.add_expert(session_id=1, reduction=4)
        tokens = torch.randn(7, 3, 8)
        output, aux = adapter(tokens)
        normalized = adapter.layer_norm(tokens)
        base_logits, base_weights = adapter.router(normalized[0], adapter.base_expert_count)
        new_logits = adapter.router.columns[-1](normalized[0])
        expected_weights = torch.cat([base_weights, torch.sigmoid(new_logits)], dim=-1)
        expected = torch.zeros_like(normalized)
        for expert_id, expert in enumerate(adapter.experts):
            expected = expected + expert(normalized) * expected_weights[:, expert_id][None, :, None]
        expected = adapter.residual_scale * expected
        torch.testing.assert_close(aux["route_weights"], expected_weights)
        torch.testing.assert_close(aux["logits"], torch.cat([base_logits, new_logits], dim=-1))
        torch.testing.assert_close(output, expected)
        self.assertEqual(tuple(output.shape), tuple(tokens.shape))

    def test_adapter_rejects_invalid_token_batch_feature_shape(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        with self.assertRaisesRegex(ValueError, r"\[tokens, batch, dim\]"):
            adapter(torch.randn(3, 8))

    def test_new_expert_changes_current_unified_path_but_old_modules_are_frozen(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2).eval()
        x = torch.randn(4, 3, 8)
        before, _ = adapter(x)
        adapter.add_expert(1, reduction=2)
        with torch.no_grad():
            adapter.experts[-1].c_proj.bias.fill_(0.2)
        adapter.freeze_all()
        after, _ = adapter(x)
        self.assertGreater((after - before).abs().max().item(), 0.0)
        self.assertFalse(any(p.requires_grad for p in adapter.experts[:-1].parameters()))
        self.assertFalse(any(p.requires_grad for p in adapter.router.columns[:-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))

    def test_topology_roundtrip_preserves_experts_and_descriptors(self):
        torch.manual_seed(11)
        source = VisualMoEAdapter(8, num_experts=2, reduction=2,
                                  descriptor_dim=4, descriptor_buffer_size=500)
        source.add_expert(2, reduction=4)
        topology = source.export_topology()
        restored = VisualMoEAdapter(8, num_experts=2, reduction=2,
                                    descriptor_dim=4, descriptor_buffer_size=500)
        restored.rebuild_topology(topology)
        restored.load_state_dict(source.state_dict(), strict=True)
        x = torch.randn(3, 2, 8)
        torch.testing.assert_close(source(x)[0], restored(x)[0], rtol=0, atol=0)
        self.assertEqual(restored.num_experts, 3)

    def test_class_level_expansion_uses_each_classes_best_existing_descriptor(self):
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2, descriptor_dim=4)
        block = nn.Module()
        block.moe_adapter = adapter
        transformer = nn.Module()
        transformer.resblocks = nn.ModuleList([block])
        visual = nn.Module()
        visual.transformer = transformer
        model.clip_model = nn.Module()
        model.clip_model.visual = visual
        model.device = "cpu"
        model.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            VISUAL_MOE=SimpleNamespace(EXPANSION_Z_THRESHOLD=1.0))))
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        features = {0: {"cls_in": torch.randn(4, 8),
                        "labels": torch.tensor([2, 2, 3, 3])}}

        adapter.descriptor_scores = lambda x, require_ready=True: torch.tensor(
            [[0.2, 0.3], [0.2, 0.3], [1.5, 1.6], [1.5, 1.6]])
        decision = model.plan_layer_expansion(0, features, [2, 3])
        self.assertTrue(decision.expand)
        self.assertEqual(decision.uncovered_classes, [3])
        self.assertAlmostEqual(decision.class_scores[2], 0.2, places=5)
        self.assertAlmostEqual(decision.class_scores[3], 1.5, places=5)

        adapter.descriptor_scores = lambda x, require_ready=True: torch.tensor(
            [[1.5, 0.2], [1.5, 0.2], [0.2, 1.5], [0.2, 1.5]])
        decision = model.plan_layer_expansion(0, features, [2, 3])
        self.assertFalse(decision.expand)
        self.assertEqual(decision.uncovered_classes, [])

    def test_semantic_guidance_uses_only_fixed_base_visual_prototypes(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        runner.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            TEXT_CALIBRATION=False, LAMBDA_T=0.0, VISION_CALIBRATION=True,
            LAMBDA_I=0.8, TAU=16.0)))
        base = {"image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "text_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "description_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        old_incremental = {"image_proto": torch.tensor([[0.0, -1.0]]),
                           "text_features": torch.tensor([[0.0, -1.0]]),
                           "description_proto": torch.tensor([[0.0, -1.0]])}
        visual = torch.tensor([[0.0, 1.0]])
        semantic = torch.tensor([[1.0, 0.0]])
        first = runner._semantic_visual_calibration(visual, semantic, [base, old_incremental])
        changed_old = dict(old_incremental)
        changed_old["image_proto"] = torch.tensor([[-1.0, 0.0]])
        changed_old["text_features"] = torch.tensor([[-1.0, 0.0]])
        changed_old["description_proto"] = torch.tensor([[-1.0, 0.0]])
        second = runner._semantic_visual_calibration(visual, semantic, [base, changed_old])
        torch.testing.assert_close(first, second)
        self.assertGreater(first[0, 0].item(), first[0, 1].item())


    def test_base_training_logits_use_visual_prototypes_not_text_fusion(self):
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        model.device = "cpu"
        model.clip_model = nn.Module()
        model.clip_model.logit_scale = nn.Parameter(torch.tensor(0.0))
        model.extract_img_feature_train = lambda images: (torch.tensor([[1.0, 0.0]]), [])
        task_state = {
            "image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "fused_proto": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        }
        logits, _, teacher, _ = model.forward_train(torch.zeros(1, 3, 2, 2), task_state)
        self.assertIsNone(teacher)
        self.assertEqual(int(logits.argmax(dim=1).item()), 0)

    def test_merge_dicts_builds_visual_bank_without_fused_prototypes(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        states = [
            {"class_ids": torch.tensor([0, 1]),
             "image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "text_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "description_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "fused_proto": torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
            {"class_ids": torch.tensor([2]),
             "image_proto": torch.tensor([[-1.0, 0.0]]),
             "text_features": torch.tensor([[-1.0, 0.0]]),
             "description_proto": torch.tensor([[-1.0, 0.0]]),
             "fused_proto": torch.tensor([[0.0, -1.0]])},
        ]
        merged = runner.merge_dicts(states)
        self.assertEqual(merged["class_ids"].tolist(), [0, 1, 2])
        self.assertNotIn("fused_proto", merged)
        torch.testing.assert_close(
            merged["image_proto"],
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        )

    def test_uncovered_batch_selection_keeps_images_labels_and_source_ids_aligned(self):
        images = torch.arange(4).float().unsqueeze(1)
        labels = torch.tensor([2, 3, 4, 3])
        source_ids = torch.tensor([10, 11, 12, 13])
        selected_images, selected_labels, selected_sources = Runner._select_uncovered_batch(
            images, labels, source_ids, [3]
        )
        self.assertEqual(selected_images.squeeze(1).tolist(), [1.0, 3.0])
        self.assertEqual(selected_labels.tolist(), [3, 3])
        self.assertEqual(selected_sources.tolist(), [11, 13])

    def test_global_incremental_logits_use_one_current_encoder_and_global_columns(self):
        runner = Runner.__new__(Runner)
        class Encoder:
            def extract_img_feature_train(self, images):
                return F.normalize(images, dim=-1), []
            def extract_img_feature(self, images):
                return F.normalize(images, dim=-1)
        runner.model_without_dp = Encoder()
        bank = {
            "global_full": F.normalize(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]), dim=-1),
            "global_ids": torch.tensor([0, 2, 3]),
            "old_class_count": 1,
            "classes": torch.tensor([2, 3]),
            "source_ids": torch.tensor([10, 11]),
            "novel_full": F.normalize(torch.tensor([[0., 1.], [-1., 0.]]), dim=-1),
            "novel_loo": F.normalize(torch.tensor([[[0., 1.], [-1., 0.]],
                                                   [[0., 1.], [-1., 0.]]]), dim=-1),
        }
        images = torch.tensor([[0., 1.]])
        labels = torch.tensor([2])
        logits, targets = runner._global_incremental_logits(
            images, labels, torch.tensor([10]), bank, 100.0)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(int(targets.item()), 1)
        self.assertEqual(int(logits.argmax(dim=1).item()), 1)

    def test_anchor_suppression_is_differentiable_and_finite(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(1, reduction=2)
        mean = torch.randn(4, 8)
        var = torch.ones_like(mean) * 0.1
        loss = adapter.old_anchor_suppression_loss(mean, var)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(adapter.router.columns[-1].weight.grad)
        self.assertIsNotNone(adapter.experts[-1].c_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
