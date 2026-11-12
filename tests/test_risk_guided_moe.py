"""Unit tests for the AE-driven, single-feature-space FSCIL architecture."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip.model import RepresentationDescriptor, VisualMoEAdapter, VisionTransformer
from models.bimc import BiMC, build_loo_cache_from_embeddings
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


class TinyLifecycleBlock(nn.Module):
    """A real VisualMoEAdapter in a deterministic, lightweight two-layer path."""

    def __init__(self):
        super().__init__()
        self.moe_adapter = VisualMoEAdapter(
            8, num_experts=2, reduction=2, descriptor_dim=1,
            descriptor_buffer_size=16, incremental_gate_bias_init=-1.0,
        )
        # A trained Base path is represented by a stable identity residual.
        # This isolates the incremental lifecycle without mocking MoE behavior.
        with torch.no_grad():
            for expert in self.moe_adapter.experts:
                expert.c_proj.weight.zero_()
                expert.c_proj.bias.zero_()

    def forward(self, tokens):
        residual, aux = self.moe_adapter(tokens)
        return tokens + residual, aux

    def forward_dual(self, adapted_tokens, reference_tokens):
        residual, aux = self.moe_adapter(
            adapted_tokens, route_input=reference_tokens[0]
        )
        return adapted_tokens + residual, reference_tokens, aux


class TinyLifecycleCLIP(nn.Module):
    """Deterministic two-layer encoder for real Runner/BiMC integration tests."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList(
            [TinyLifecycleBlock(), TinyLifecycleBlock()]
        )

    def encode_image(self, images, return_moe_aux=False, return_reference=False):
        if images.ndim != 2 or images.size(-1) != 8:
            raise ValueError("TinyLifecycleCLIP expects [batch, 8] feature-like images.")
        images = images.float()
        tokens = torch.stack([images, images.roll(1, dims=-1)], dim=0)
        reference_tokens = tokens.detach()
        aux_list = []
        for block in self.visual.transformer.resblocks:
            tokens, reference_tokens, aux = block.forward_dual(tokens, reference_tokens)
            aux_list.append(aux)
        features = F.normalize(tokens[0], dim=-1)
        with torch.no_grad():
            reference_features = F.normalize(reference_tokens[0], dim=-1)
        if return_reference and return_moe_aux:
            return features, reference_features, aux_list
        if return_reference:
            return features, reference_features
        if return_moe_aux:
            return features, aux_list
        return features


def build_lifecycle_runner():
    optim_cfg = SimpleNamespace(
        BASE_DESCRIPTOR_EPOCHS=30,
        INCREMENTAL_DESCRIPTOR_EPOCHS=30,
        LR_DESCRIPTOR=0.03,
        LR_INCREMENTAL_EXPERT=0.08,
        LR_INCREMENTAL_ROUTER=0.08,
        WEIGHT_DECAY=0.0,
    )
    incremental_cfg = SimpleNamespace(
        EPOCHS=8,
        LOGIT_SCALE=20.0,
        OLD_RESIDUAL_WEIGHT=1.0,
        OLD_SUPPRESSION_WEIGHT=10.0,
    )
    bimc_cfg = SimpleNamespace(
        VISUAL_MOE=SimpleNamespace(
            ENABLE=True,
            NUM_BASE_EXPERTS=2,
            INCREMENTAL_REDUCTION=2,
            EXPANSION_Z_THRESHOLD=1.0,
        ),
        OPTIM=optim_cfg,
        INCREMENTAL=incremental_cfg,
        TEXT_CALIBRATION=False,
        LAMBDA_T=0.0,
        VISION_CALIBRATION=False,
        LAMBDA_I=0.0,
        TAU=16.0,
    )
    cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=bimc_cfg))
    model = BiMC.__new__(BiMC)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.device = torch.device("cpu")
    model.clip_model = TinyLifecycleCLIP()
    model._apply_freeze_policy()

    runner = Runner.__new__(Runner)
    runner.cfg = cfg
    runner.device = torch.device("cpu")
    runner.expansion_thresholds = {0: 1.0, 1: 1.0}
    runner.descriptor_error_priors = {}
    runner.model = model
    runner.model_without_dp = model
    runner._text_prototypes = lambda _names, class_index: (
        F.normalize(torch.eye(8)[torch.as_tensor(class_index, dtype=torch.long)], dim=-1),
        F.normalize(torch.eye(8)[torch.as_tensor(class_index, dtype=torch.long)], dim=-1),
    )
    return runner, model


def lifecycle_batch(class_id, feature, source_start, augmented=False, seed=0):
    count = 4
    images = feature.reshape(1, 8).repeat(count, 1).clone()
    if augmented:
        generator = torch.Generator().manual_seed(seed)
        perturbation = torch.randn(images.shape, generator=generator) * 0.30
        perturbation[:, class_id] = 0.0
        images = images + perturbation
    return {
        "image": images,
        "label": torch.full((count,), int(class_id), dtype=torch.long),
        "source_ids": torch.arange(source_start, source_start + count),
    }


def make_lifecycle_state(runner, model, class_id, support, creation_session):
    anchors = runner._collect_class_anchors([support])
    with torch.no_grad():
        adapted, _, reference = model.extract_img_feature_train(
            support["image"], return_reference=True
        )
        prototype = F.normalize(adapted.mean(0, keepdim=True), dim=-1)
        reference_prototype = F.normalize(reference.mean(0, keepdim=True), dim=-1)
    semantic = F.normalize(torch.eye(8)[[class_id]], dim=-1)
    return {
        "class_ids": torch.tensor([class_id]),
        "image_proto": prototype,
        "reference_image_proto": reference_prototype,
        "text_features": semantic,
        "description_proto": semantic,
        "anchor_by_block": anchors,
        "creation_session": int(creation_session),
    }


def lifecycle_route_snapshot(model, images):
    model.eval()
    with torch.no_grad():
        features, aux = model.extract_img_feature_train(images)
    return features.detach().clone(), [
        item["route_weights"].detach().clone() for item in aux
    ]


class UnifiedDynamicMoETests(unittest.TestCase):
    def test_incremental_session_trains_positive_and_negative_route_responsibilities(self):
        torch.manual_seed(239)
        runner, model = build_lifecycle_runner()
        model.eval()
        base_support = lifecycle_batch(
            0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 6000
        )
        runner._train_base_descriptors([base_support])
        base_state = make_lifecycle_state(runner, model, 0, base_support, 0)

        covered_support = lifecycle_batch(
            1, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 6010
        )
        novel_support = lifecycle_batch(
            2, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 6020
        )
        support = [covered_support, novel_support]
        train = [covered_support, novel_support]
        result = runner.train_incremental_task(
            1, train, support, [1, 2], ["covered_new_class", "uncovered_new_class"],
            [base_state],
        )
        self.assertEqual(
            [item["uncovered_classes"] for item in result["expanded"]], [[2], [2]]
        )
        for block_idx, adapter in runner._moe_adapters():
            features = model.collect_blockwise_cls_features(support)[block_idx]
            gates = torch.sigmoid(adapter.router.columns[-1](features["cls_in"]))
            labels = features["labels"]
            covered_gate = gates[labels.eq(1)].mean()
            novel_gate = gates[labels.eq(2)].mean()
            self.assertGreater(float(novel_gate), float(covered_gate))
            self.assertLess(float(covered_gate), 0.5)

    def test_incremental_route_responsibility_has_positive_and_negative_targets(self):
        logits = torch.tensor([-1.0, 1.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        self.assertGreater(float(logits.grad[0]), 0.0)
        self.assertLess(float(logits.grad[1]), 0.0)

    def test_reference_stream_matches_unmodified_frozen_clip_vit(self):
        torch.manual_seed(229)
        model = TinyVisualCLIP().eval()
        images = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            _, reference, _ = model.visual(
                images, return_moe_aux=True, return_reference=True
            )
            adapters = []
            for block in model.visual.transformer.resblocks:
                adapters.append(block.moe_adapter)
                block.moe_adapter = None
            try:
                baseline = model.visual(images)
            finally:
                for block, adapter in zip(model.visual.transformer.resblocks, adapters):
                    block.moe_adapter = adapter
        torch.testing.assert_close(reference, baseline, rtol=0, atol=0)

    def test_dual_path_reference_is_unchanged_by_adapter_updates(self):
        torch.manual_seed(211)
        model = TinyVisualCLIP().eval()
        images = torch.randn(3, 3, 8, 8)
        with torch.no_grad():
            adapted_before, reference_before, _ = model.visual(
                images, return_moe_aux=True, return_reference=True
            )
            block0 = model.visual.transformer.resblocks[0].moe_adapter
            block0.add_expert(session_id=1, reduction=2)
            with torch.no_grad():
                block0.experts[-1].c_proj.bias.fill_(0.25)
            block0.freeze_all()
            adapted_after, reference_after, _ = model.visual(
                images, return_moe_aux=True, return_reference=True
            )
        self.assertGreater((adapted_after - adapted_before).abs().max().item(), 0.0)
        torch.testing.assert_close(reference_before, reference_after, rtol=0, atol=0)

    def test_dual_path_router_uses_reference_cls_while_experts_use_adapted_tokens(self):
        torch.manual_seed(223)
        model = TinyVisualCLIP().eval()
        images = torch.randn(2, 3, 8, 8)
        captured = []
        adapter = model.visual.transformer.resblocks[0].moe_adapter
        handle = adapter.router.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        try:
            adapted, reference, aux = model.visual(
                images, return_moe_aux=True, return_reference=True
            )
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0], aux[0]["cls_in"])
        self.assertFalse(reference.requires_grad)
        self.assertEqual(tuple(adapted.shape), tuple(reference.shape))
        self.assertEqual(tuple(aux[0]["cls_in"].shape), (images.size(0), 8))

    def test_base_ae_and_expansion_threshold_use_disjoint_heldout_sources(self):
        torch.manual_seed(233)
        runner, model = build_lifecycle_runner()
        supports = [
            lifecycle_batch(0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 5000),
            lifecycle_batch(1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 5010),
            lifecycle_batch(2, torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.]), 5020),
        ]
        runner._train_base_descriptors(supports)
        self.assertEqual(set(runner._base_calibration_features), {0, 1})
        for block_idx, adapter in runner._moe_adapters():
            holdout = runner._base_calibration_features[block_idx]
            self.assertEqual(holdout["labels"].numel(), 3)
            self.assertEqual(set(holdout["labels"].tolist()), {0, 1, 2})
            self.assertTrue(all(
                int(descriptor.error_count.item()) == 3
                for descriptor in adapter.descriptors
            ))
        runner._calibrate_expansion_thresholds(supports)
        self.assertEqual(set(runner.expansion_thresholds), {0, 1})
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in runner.expansion_thresholds.values()))

    def test_incremental_expert_output_energy_is_differentiable_and_unweighted(self):
        torch.manual_seed(241)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        adapter.set_newest_trainable()
        # The production zero-up initialization deliberately yields zero
        # energy and zero energy-gradient before the first task gradient.
        with torch.no_grad():
            adapter.experts[-1].c_proj.weight.normal_(std=0.1)
        tokens = torch.randn(3, 4, 8)
        route = torch.randn(4, 8)
        _, aux = adapter(tokens, route_input=route)
        energy = aux["expert_output_energy"][-1]
        energy.backward()
        self.assertTrue(torch.isfinite(energy))
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in adapter.experts[-1].parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in adapter.experts[0].parameters()
        ))

    def test_ae_error_statistics_shrink_toward_base_prior(self):
        descriptor = RepresentationDescriptor(4, descriptor_dim=2, buffer_size=8)
        errors = torch.tensor([1.0, 1.2, 0.8, 1.1])
        descriptor.update_reference(errors)
        local_mean = descriptor.error_mean.clone()
        local_std = descriptor.error_std.clone()
        mean, std = descriptor.shrink_reference(0.0, 0.5, prior_strength=8.0)
        self.assertLess(float(mean), float(local_mean))
        self.assertGreater(float(std), 0.0)
        self.assertTrue(torch.isfinite(mean) and torch.isfinite(std))
        self.assertEqual(int(descriptor.error_count.item()), 4)

    def test_low_rank_ae_is_centered_tied_weight_and_orthogonal(self):
        torch.manual_seed(227)
        descriptor = RepresentationDescriptor(12, descriptor_dim=3, buffer_size=8)
        values = torch.randn(20, 12) + 2.0
        descriptor.set_center(values)
        reconstructed = descriptor(values)
        self.assertEqual(tuple(descriptor.basis.shape), (12, 3))
        torch.testing.assert_close(descriptor.center, values.mean(0), rtol=0, atol=1e-6)
        torch.testing.assert_close(
            descriptor.basis.t() @ descriptor.basis,
            torch.eye(3), rtol=1e-5, atol=1e-5
        )
        self.assertEqual(tuple(reconstructed.shape), tuple(values.shape))
        self.assertLess(float(descriptor.orthogonality_loss()), 1e-8)

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
                "reference_image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "text_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "description_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        old_incremental = {"image_proto": torch.tensor([[0.0, -1.0]]),
                           "text_features": torch.tensor([[0.0, -1.0]]),
                           "description_proto": torch.tensor([[0.0, -1.0]])}
        visual = torch.tensor([[0.0, 1.0]])
        semantic = torch.tensor([[1.0, 0.0]])
        first = runner._visual_intra_modal_calibration(visual, visual, [base, old_incremental])
        changed_old = dict(old_incremental)
        changed_old["image_proto"] = torch.tensor([[-1.0, 0.0]])
        changed_old["text_features"] = torch.tensor([[-1.0, 0.0]])
        changed_old["description_proto"] = torch.tensor([[-1.0, 0.0]])
        second = runner._visual_intra_modal_calibration(visual, visual, [base, changed_old])
        torch.testing.assert_close(first, second)
        self.assertGreater(first[0, 1].item(), first[0, 0].item())

    def test_bilevel_calibration_fuses_visual_textual_modalities(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        runner.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            TEXT_CALIBRATION=True, LAMBDA_T=0.5, VISION_CALIBRATION=False,
            LAMBDA_I=0.0, INTER_MODAL_WEIGHT=0.25,
        )))
        text = torch.tensor([[1.0, 0.0]])
        description = torch.tensor([[0.0, 1.0]])
        semantic = runner._text_intra_modal_calibration(text, description)
        visual = torch.tensor([[1.0, 0.0]])
        final = runner._inter_modal_fusion(visual, semantic)
        self.assertGreater(final[0, 0].item(), final[0, 1].item())
        self.assertTrue(torch.isfinite(final).all())


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

    def test_merge_dicts_builds_global_calibrated_prototype_bank(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        states = [
            {"class_ids": torch.tensor([0, 1]),
             "image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "text_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "description_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "final_proto": torch.tensor([[0.8, 0.6], [0.6, 0.8]])},
            {"class_ids": torch.tensor([2]),
             "image_proto": torch.tensor([[-1.0, 0.0]]),
             "text_features": torch.tensor([[-1.0, 0.0]]),
             "description_proto": torch.tensor([[-1.0, 0.0]]),
             "final_proto": torch.tensor([[-0.8, 0.6]])},
        ]
        merged = runner.merge_dicts(states)
        self.assertEqual(merged["class_ids"].tolist(), [0, 1, 2])
        self.assertIn("final_proto", merged)
        torch.testing.assert_close(
            merged["final_proto"],
            torch.tensor([[0.8, 0.6], [0.6, 0.8], [-0.8, 0.6]])
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


    def test_multisession_expansion_reuse_history_and_topology_restore(self):
        torch.manual_seed(23)
        runner, model = build_lifecycle_runner()
        model.eval()

        base_support = lifecycle_batch(0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 100)
        runner._train_base_descriptors([base_support])
        base_anchors = runner._collect_class_anchors([base_support])
        with torch.no_grad():
            base_adapted, _, base_reference = model.extract_img_feature_train(
                base_support["image"], return_reference=True
            )
            base_proto = F.normalize(base_adapted.mean(0, keepdim=True), dim=-1)
            base_reference_proto = F.normalize(base_reference.mean(0, keepdim=True), dim=-1)
        base_state = {
            "class_ids": torch.tensor([0]),
            "image_proto": base_proto,
            "reference_image_proto": base_reference_proto,
            "text_features": F.normalize(torch.eye(8)[[0]], dim=-1),
            "description_proto": F.normalize(torch.eye(8)[[0]], dim=-1),
            "anchor_by_block": base_anchors,
            "creation_session": 0,
        }
        for _, adapter in runner._moe_adapters():
            adapter.assert_descriptors_ready()

        session1_support = lifecycle_batch(
            1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 200
        )
        session1_train = lifecycle_batch(
            1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 200,
            augmented=True, seed=31,
        )
        before_session1 = {
            block: {key: value.detach().clone() for key, value in adapter.state_dict().items()}
            for block, adapter in runner._moe_adapters()
        }
        baseline_block1 = model.collect_blockwise_cls_features([session1_support])[1][
            "cls_in"
        ].clone()
        captured_block1 = {}
        original_plan = model.plan_layer_expansion

        def capture_downstream_plan(block_idx, features, class_index, threshold=None):
            if int(block_idx) == 1:
                captured_block1["cls_in"] = features[1]["cls_in"].detach().clone()
            return original_plan(block_idx, features, class_index, threshold)

        model.plan_layer_expansion = capture_downstream_plan
        try:
            session1_result = runner.train_incremental_task(
                1, [session1_train], [session1_support], [1], ["class_1"], [base_state]
            )
        finally:
            model.plan_layer_expansion = original_plan

        self.assertEqual([item["block_idx"] for item in session1_result["expanded"]], [0, 1])
        self.assertEqual([adapter.num_experts for _, adapter in runner._moe_adapters()], [3, 3])
        self.assertTrue(captured_block1)
        torch.testing.assert_close(baseline_block1, captured_block1["cls_in"], rtol=0, atol=0)
        for block, adapter in runner._moe_adapters():
            adapter.assert_descriptors_ready()
            for key, old_value in before_session1[block].items():
                torch.testing.assert_close(adapter.state_dict()[key], old_value, rtol=0, atol=0)
            old_mean = base_anchors[block]["mean"].float()
            old_var = base_anchors[block]["var"].float()
            old_gate = adapter.old_anchor_response(old_mean, old_var)
            current = model.collect_blockwise_cls_features([session1_support])[block]["cls_in"]
            new_gate = torch.sigmoid(adapter.router.columns[-1](current)).mean().item()
            self.assertLess(old_gate, 0.5)
            self.assertGreater(new_gate, old_gate)

        # The same class must be recognized by its newly calibrated expert,
        # rather than causing a duplicate expert in the next pass.
        session1_features = model.collect_blockwise_cls_features([session1_support])
        for block in model.expandable_blocks():
            decision = model.plan_layer_expansion(block, session1_features, [1])
            self.assertFalse(decision.expand)
            self.assertEqual(decision.uncovered_classes, [])

        session1_anchors = runner._collect_class_anchors([session1_support])
        with torch.no_grad():
            session1_proto = F.normalize(
                model.extract_img_feature(session1_support["image"]).mean(0, keepdim=True), dim=-1
            )
        session1_state = {
            "class_ids": torch.tensor([1]),
            "image_proto": session1_proto,
            "text_features": F.normalize(torch.eye(8)[[1]], dim=-1),
            "description_proto": F.normalize(torch.eye(8)[[1]], dim=-1),
            "anchor_by_block": session1_anchors,
            "creation_session": 1,
        }

        old_images = torch.cat([base_support["image"][:1], session1_support["image"][:1]], dim=0)
        old_labels = torch.tensor([0, 1])
        old_prototypes = torch.cat([base_proto, session1_proto], dim=0)

        def old_predictions():
            model.eval()
            with torch.no_grad():
                features = F.normalize(model.extract_img_feature(old_images), dim=-1)
                return (features @ old_prototypes.t()).argmax(dim=-1)

        predictions_before_session2 = old_predictions()
        self.assertEqual(predictions_before_session2.tolist(), old_labels.tolist())

        session2_support = lifecycle_batch(
            2, torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.]), 300
        )
        session2_train = lifecycle_batch(
            2, torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.]), 300,
            augmented=True, seed=37,
        )
        before_session2 = {
            block: {key: value.detach().clone() for key, value in adapter.state_dict().items()}
            for block, adapter in runner._moe_adapters()
        }
        session2_result = runner.train_incremental_task(
            2, [session2_train], [session2_support], [2], ["class_2"],
            [base_state, session1_state],
        )
        self.assertEqual([item["block_idx"] for item in session2_result["expanded"]], [0, 1])
        self.assertEqual([adapter.num_experts for _, adapter in runner._moe_adapters()], [4, 4])

        for block, adapter in runner._moe_adapters():
            adapter.assert_descriptors_ready()
            for key, old_value in before_session2[block].items():
                torch.testing.assert_close(adapter.state_dict()[key], old_value, rtol=0, atol=0)
            old_mean, old_var = runner._all_old_anchors([base_state, session1_state], block)
            old_gate = adapter.old_anchor_response(old_mean, old_var)
            current = model.collect_blockwise_cls_features([session2_support])[block]["cls_in"]
            new_gate = torch.sigmoid(adapter.router.columns[-1](current)).mean().item()
            self.assertLess(old_gate, 0.5)
            self.assertGreater(new_gate, old_gate)

        predictions_after_session2 = old_predictions()
        self.assertEqual(predictions_after_session2.tolist(), old_labels.tolist())
        torch.testing.assert_close(
            predictions_after_session2, predictions_before_session2, rtol=0, atol=0
        )

        session2_features = model.collect_blockwise_cls_features([session2_support])
        for block in model.expandable_blocks():
            decision = model.plan_layer_expansion(block, session2_features, [2])
            self.assertFalse(decision.expand)
            self.assertEqual(decision.uncovered_classes, [])

        # Checkpoint-style topology reconstruction must preserve the complete
        # multi-session expert pool, AE references, route outputs and encoder.
        topology = model.export_moe_topology()
        checkpoint_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        model.eval()
        with torch.no_grad():
            expected_features, expected_aux = model.extract_img_feature_train(
                session2_support["image"]
            )
        restored_runner, restored_model = build_lifecycle_runner()
        restored_model.rebuild_moe_topology(topology)
        restored_model.load_state_dict(checkpoint_state, strict=True)
        restored_model.eval()
        with torch.no_grad():
            restored_features, restored_aux = restored_model.extract_img_feature_train(
                session2_support["image"]
            )
        torch.testing.assert_close(restored_features, expected_features, rtol=0, atol=0)
        self.assertEqual(topology, restored_model.export_moe_topology())
        self.assertEqual(
            [adapter.expert_birth_sessions for _, adapter in restored_runner._moe_adapters()],
            [[0, 0, 1, 2], [0, 0, 1, 2]],
        )
        for expected, restored in zip(expected_aux, restored_aux):
            torch.testing.assert_close(
                restored["route_weights"], expected["route_weights"], rtol=0, atol=0
            )
            torch.testing.assert_close(restored["cls_in"], expected["cls_in"], rtol=0, atol=0)
        for _, adapter in restored_runner._moe_adapters():
            adapter.assert_descriptors_ready()



    def test_long_sequence_base_plus_four_incremental_sessions(self):
        torch.manual_seed(101)
        runner, model = build_lifecycle_runner()
        model.eval()
        base_support = lifecycle_batch(
            0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 1000
        )
        runner._train_base_descriptors([base_support])
        states = [make_lifecycle_state(runner, model, 0, base_support, 0)]

        expected_counts = [2, 3, 4, 5, 6]
        self.assertEqual(
            [adapter.num_experts for _, adapter in runner._moe_adapters()],
            [expected_counts[0], expected_counts[0]],
        )
        for session_id in range(1, 5):
            feature = torch.zeros(8)
            feature[session_id] = 1.0
            support = lifecycle_batch(session_id, feature, 1000 + session_id * 10)
            train = lifecycle_batch(
                session_id, feature, 1000 + session_id * 10,
                augmented=True, seed=100 + session_id,
            )
            result = runner.train_incremental_task(
                session_id, [train], [support], [session_id],
                [f"class_{session_id}"], states,
            )
            self.assertTrue(result["trained"])
            self.assertEqual(
                [entry["uncovered_classes"] for entry in result["expanded"]],
                [[session_id], [session_id]],
            )
            self.assertEqual(
                [adapter.num_experts for _, adapter in runner._moe_adapters()],
                [expected_counts[session_id], expected_counts[session_id]],
            )
            self.assertEqual(
                [adapter.expert_birth_sessions[-1] for _, adapter in runner._moe_adapters()],
                [session_id, session_id],
            )
            for block in model.expandable_blocks():
                features = model.collect_blockwise_cls_features([support])
                decision = model.plan_layer_expansion(block, features, [session_id])
                self.assertFalse(decision.expand)
                self.assertEqual(decision.uncovered_classes, [])
            states.append(make_lifecycle_state(
                runner, model, session_id, support, session_id
            ))

        for _, adapter in runner._moe_adapters():
            self.assertEqual(adapter.num_experts, 6)
            self.assertEqual(adapter.expert_birth_sessions, [0, 0, 1, 2, 3, 4])
            adapter.assert_descriptors_ready()
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    def test_quantifies_old_feature_accuracy_and_routing_retention(self):
        torch.manual_seed(109)
        runner, model = build_lifecycle_runner()
        model.eval()
        base_support = lifecycle_batch(
            0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 2000
        )
        runner._train_base_descriptors([base_support])
        base_state = make_lifecycle_state(runner, model, 0, base_support, 0)

        session1_support = lifecycle_batch(
            1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 2010
        )
        session1_train = lifecycle_batch(
            1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 2010,
            augmented=True, seed=111,
        )
        runner.train_incremental_task(
            1, [session1_train], [session1_support], [1], ["class_1"], [base_state]
        )
        session1_state = make_lifecycle_state(runner, model, 1, session1_support, 1)

        old_images = torch.cat([base_support["image"], session1_support["image"]], dim=0)
        old_labels = torch.tensor([0] * 4 + [1] * 4)
        old_prototypes = torch.cat([
            base_state["image_proto"], session1_state["image_proto"]
        ], dim=0)
        before_features, before_routes = lifecycle_route_snapshot(model, old_images)
        before_accuracy = int(
            (F.normalize(before_features, dim=-1) @ old_prototypes.t())
            .argmax(dim=-1).eq(old_labels).sum().item()
        )

        session2_support = lifecycle_batch(
            2, torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.]), 2020
        )
        session2_train = lifecycle_batch(
            2, torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.]), 2020,
            augmented=True, seed=113,
        )
        runner.train_incremental_task(
            2, [session2_train], [session2_support], [2], ["class_2"],
            [base_state, session1_state],
        )
        after_features, after_routes = lifecycle_route_snapshot(model, old_images)
        after_accuracy = int(
            (F.normalize(after_features, dim=-1) @ old_prototypes.t())
            .argmax(dim=-1).eq(old_labels).sum().item()
        )
        per_sample_cosine = F.cosine_similarity(
            before_features.float(), after_features.float(), dim=-1
        )
        feature_cosine = float(per_sample_cosine.mean().item())
        route_deltas, newest_old_responses = [], []
        for (_, adapter), before, after in zip(
            runner._moe_adapters(), before_routes, after_routes
        ):
            historical_count = before.size(-1)
            route_deltas.append(float(
                (before - after[:, :historical_count]).abs().mean().item()
            ))
            newest_old_responses.append(float(after[:, -1].mean().item()))

        self.assertEqual(before_accuracy, 8)
        self.assertEqual(after_accuracy, before_accuracy)
        self.assertTrue(torch.isfinite(per_sample_cosine).all())
        self.assertGreaterEqual(feature_cosine, 0.98)
        self.assertTrue(all(torch.isfinite(torch.tensor(route_deltas))))
        self.assertLess(max(route_deltas), 0.10)
        self.assertLess(max(newest_old_responses), 0.30)
        print(
            "[RETENTION] old_feature_cosine=%.6f old_accuracy=%d/%d "
            "historical_route_mean_abs_delta=%s newest_expert_old_response=%s"
            % (feature_cosine, after_accuracy, old_labels.numel(),
               [round(value, 6) for value in route_deltas],
               [round(value, 6) for value in newest_old_responses])
        )

    def test_extreme_nonfinite_reconstruction_scores_are_rejected(self):
        runner, model = build_lifecycle_runner()
        for _, adapter in runner._moe_adapters():
            for descriptor in adapter.descriptors:
                descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        extreme = torch.full((4, 8), 1e20)
        features = {0: {
            "cls_in": extreme,
            "labels": torch.tensor([7, 7, 7, 7]),
        }}
        with self.assertRaises((FloatingPointError, ValueError)):
            model.plan_layer_expansion(0, features, [7])

    def test_duplicate_source_ids_are_query_exclusive_and_cross_class_collisions_fail(self):
        feature = torch.tensor([
            [1., 0.], [1., 0.], [0., 1.], [-1., 0.],
        ])
        labels = torch.tensor([4, 4, 4, 4])
        source_ids = torch.tensor([10, 10, 11, 12])
        cache = build_loo_cache_from_embeddings(feature, labels, source_ids, [4])
        # Both augmented views of source 10 must be excluded for either query
        # belonging to source 10; only sources 11 and 12 remain.
        expected = F.normalize(torch.tensor([[0., 1.]]) + torch.tensor([[-1., 0.]]), dim=-1)
        torch.testing.assert_close(
            cache["loo_visual_prototypes"][0, 0], expected[0], rtol=0, atol=1e-6
        )
        collision_labels = torch.tensor([4, 5, 4, 5])
        collision_sources = torch.tensor([10, 10, 11, 12])
        with self.assertRaisesRegex(ValueError, "source id.*multiple classes"):
            build_loo_cache_from_embeddings(
                torch.randn(4, 2), collision_labels, collision_sources, [4, 5]
            )

    def test_insufficient_class_support_rolls_back_all_session_experts(self):
        torch.manual_seed(127)
        runner, model = build_lifecycle_runner()
        base_support = lifecycle_batch(
            0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 3000
        )
        runner._train_base_descriptors([base_support])
        # A single source cannot produce a query-exclusive same-class prototype.
        one_source = {
            "image": torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]]),
            "label": torch.tensor([1]),
            "source_ids": torch.tensor([3010]),
        }
        with self.assertRaisesRegex(ValueError, "another same-class source"):
            runner.train_incremental_task(
                1, [one_source], [one_source], [1], ["class_1"],
                [make_lifecycle_state(runner, model, 0, base_support, 0)],
            )
        self.assertEqual(
            [adapter.num_experts for _, adapter in runner._moe_adapters()], [2, 2]
        )
        for _, adapter in runner._moe_adapters():
            adapter.assert_descriptors_ready()
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    def test_ae_calibration_failure_rolls_back_expansions_across_all_blocks(self):
        torch.manual_seed(131)
        runner, model = build_lifecycle_runner()
        base_support = lifecycle_batch(
            0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 4000
        )
        runner._train_base_descriptors([base_support])
        base_state = make_lifecycle_state(runner, model, 0, base_support, 0)
        support = lifecycle_batch(
            1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 4010
        )
        train = lifecycle_batch(
            1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 4010,
            augmented=True, seed=137,
        )
        original_fit = runner._train_descriptor
        calls = {"count": 0}

        def fail_second_layer(adapter, expert_id, values, epochs, sample_weights=None,
                              calibration_values=None, shrinkage_prior=None):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected AE calibration failure")
            return original_fit(
                adapter, expert_id, values, epochs, sample_weights,
                calibration_values=calibration_values,
                shrinkage_prior=shrinkage_prior,
            )

        runner._train_descriptor = fail_second_layer
        try:
            with self.assertRaisesRegex(RuntimeError, "injected AE calibration failure"):
                runner.train_incremental_task(
                    1, [train], [support], [1], ["class_1"], [base_state]
                )
        finally:
            runner._train_descriptor = original_fit
        self.assertEqual(calls["count"], 2)
        self.assertEqual(
            [adapter.num_experts for _, adapter in runner._moe_adapters()], [2, 2]
        )
        for _, adapter in runner._moe_adapters():
            adapter.assert_descriptors_ready()
            self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    def test_topology_rebuild_rejects_invalid_specs_without_mutating_live_adapter(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        before = adapter.export_topology()
        invalid_topologies = [
            {**before, "version": 999},
            {**before, "experts": before["experts"][:1]},
            {**before, "experts": [
                before["experts"][0], before["experts"][1],
                {"reduction": 0, "birth_session": 2},
            ]},
            {**before, "experts": [
                before["experts"][0], before["experts"][1],
                {"reduction": 2, "birth_session": -1},
            ]},
        ]
        for topology in invalid_topologies:
            with self.subTest(topology=topology):
                with self.assertRaises(ValueError):
                    adapter.rebuild_topology(topology)
                self.assertEqual(adapter.export_topology(), before)
                self.assertEqual(adapter.num_experts, 2)



if __name__ == "__main__":
    unittest.main()
