"""Unit tests for the AE-driven, single-feature-space FSCIL architecture."""

import tempfile
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
        OLD_SUPPRESSION_WEIGHT=10.0,
        RD_WEIGHT=1.0,
    )
    bimc_cfg = SimpleNamespace(
        VISUAL_MOE=SimpleNamespace(
            ENABLE=True,
            NUM_BASE_EXPERTS=2,
            INCREMENTAL_REDUCTION=2,
            INCREMENTAL_DROPOUT=0.1,
            EXPANSION_THRESHOLD_QUANTILE=0.95,
            EXPANSION_CALIBRATION_GROUPS=20,
            EXPANSION_CALIBRATION_SEED=271,
            SESSION_SUPPORT_SIZES=[8, 16],
        ),
        OPTIM=optim_cfg,
        INCREMENTAL=incremental_cfg,
        TEXT_CALIBRATION=False,
        LAMBDA_T=0.0,
        VISION_CALIBRATION=False,
        LAMBDA_I=0.0,
        TAU=16.0,
    )
    cfg = SimpleNamespace(
        TRAINER=SimpleNamespace(BiMC=bimc_cfg),
        DATASET=SimpleNamespace(NUM_INC_CLS=1, NUM_INC_SHOT=8, SESSION_SUPPORT_SIZES=[8, 16]),
    )
    model = BiMC.__new__(BiMC)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.device = torch.device("cpu")
    model.clip_model = TinyLifecycleCLIP()
    model._apply_freeze_policy()

    runner = Runner.__new__(Runner)
    runner.cfg = cfg
    runner.device = torch.device("cpu")
    runner.model = model
    runner.model_without_dp = model
    runner._text_prototypes = lambda _names, class_index: (
        F.normalize(torch.eye(8)[torch.as_tensor(class_index, dtype=torch.long)], dim=-1),
        F.normalize(torch.eye(8)[torch.as_tensor(class_index, dtype=torch.long)], dim=-1),
    )
    return runner, model


def lifecycle_batch(class_id, feature, source_start, augmented=False, seed=0):
    count = 8
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
        "support_count": torch.tensor([support["label"].numel()], dtype=torch.long),
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
    def test_incremental_session_trains_all_current_classes_as_positive_route_responsibilities(self):
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
            1, train, support, [1, 2], ["covered_new_class", "novel_new_class"],
            [base_state],
        )
        self.assertEqual([item["block_idx"] for item in result["expanded"]], [0, 1])
        for block_idx, adapter in runner._moe_adapters():
            history = result["training"][block_idx]
            self.assertEqual(history["classification_samples_per_epoch"], [16] * 8)
            self.assertEqual(history["classification_samples_total"], 128)
            self.assertEqual(len(history["ce_loss"]), 8)
            self.assertEqual(len(history["rd_loss"]), 8)
            self.assertEqual(len(history["pd_loss"]), 8)
            self.assertEqual(history["rd_training_samples"], 16)
            self.assertEqual(history["ae_calibration_sources"], 16)
            self.assertEqual(history["ae_fit_classes"], [1, 2])
            self.assertEqual(history["ae_calibration_classes"], [1, 2])
            self.assertGreater(history["ae_effective_calibration_count"], 1.0)
            self.assertTrue(all(torch.isfinite(torch.tensor(history[key])).all()
                                for key in ("ce_loss", "rd_loss", "pd_loss")))
            features = model.collect_blockwise_cls_features(support)[block_idx]
            gates = torch.sigmoid(adapter.router.columns[-1](features["cls_in"]))
            self.assertTrue(torch.isfinite(gates).all())

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

    def test_base_ae_uses_disjoint_reference_sources_without_task_threshold_calibration(self):
        torch.manual_seed(233)
        runner, model = build_lifecycle_runner()
        supports = [
            lifecycle_batch(0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 5000),
            lifecycle_batch(1, torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.]), 5010),
            lifecycle_batch(2, torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.]), 5020),
        ]
        runner._train_base_descriptors(supports)
        for _, adapter in runner._moe_adapters():
            adapter.assert_descriptors_ready()
            for descriptor in adapter.descriptors:
                self.assertEqual(int(descriptor.error_count.item()), 9)
                self.assertGreater(float(descriptor.effective_error_count.item()), 0.0)
                self.assertTrue(torch.isfinite(descriptor.error_mean))
                self.assertTrue(torch.isfinite(descriptor.error_std))

    def test_old_route_suppression_only_updates_new_sigmoid_gate(self):
        torch.manual_seed(17)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        adapter.set_newest_trainable()
        route_mean = torch.randn(3, 8)
        route_var = torch.ones_like(route_mean) * 0.1
        loss = adapter.old_route_suppression_loss(route_mean, route_var)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(adapter.router.columns[-1].weight.grad)
        self.assertIsNone(adapter.experts[-1].c_fc.weight.grad)
        self.assertIsNone(adapter.experts[-1].c_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in adapter.experts[0].parameters()))

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

    def test_weighted_ae_rejects_degenerate_effective_sample_size_atomically(self):
        descriptor = RepresentationDescriptor(4, descriptor_dim=2, buffer_size=8)
        before = {key: value.detach().clone() for key, value in descriptor.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "effective sample size greater than one"):
            descriptor.update_reference(
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([1.0, 1e-8, 1e-8]),
            )
        self.assertFalse(descriptor.is_ready())
        for key, value in before.items():
            torch.testing.assert_close(descriptor.state_dict()[key], value, rtol=0, atol=0)

    def test_weighted_ae_requires_two_independent_reference_errors(self):
        descriptor = RepresentationDescriptor(4, descriptor_dim=2, buffer_size=8)
        with self.assertRaisesRegex(ValueError, "at least two finite independent errors"):
            descriptor.update_reference(torch.tensor([0.1]), torch.tensor([1.0]))
        self.assertFalse(descriptor.is_ready())

    def test_weighted_ae_reference_uses_exact_responsibility_mass(self):
        descriptor = RepresentationDescriptor(4, descriptor_dim=2, buffer_size=8)
        errors = torch.tensor([1.0, 3.0, 5.0])
        weights = torch.tensor([1e-8, 2e-8, 3e-8])
        descriptor.update_reference(errors, weights)
        expected_mean = (errors * weights).sum() / weights.sum()
        expected_effective_n = weights.sum().square() / weights.square().sum()
        torch.testing.assert_close(descriptor.error_mean, expected_mean)
        torch.testing.assert_close(descriptor.effective_error_count, expected_effective_n)
        self.assertTrue(torch.isfinite(descriptor.error_std))
        self.assertGreater(float(descriptor.error_std), 0.0)

    def test_incremental_calibration_mask_keeps_one_source_out_of_fit(self):
        labels = torch.tensor([3, 3, 3, 3, 3])
        source_ids = torch.tensor([101, 102, 103, 104, 105])
        mask = Runner._make_calibration_mask(
            labels, source_ids, min_calibration_count=1, min_fit_per_class=3
        )
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(int((~mask).sum()), 4)
        self.assertFalse(torch.isin(source_ids[mask], source_ids[~mask]).any())

    def test_calibration_split_is_source_order_invariant_and_reproducible(self):
        labels = torch.tensor([0] * 6 + [1] * 6)
        source_ids = torch.tensor([11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26])
        first = Runner._make_calibration_mask(
            labels, source_ids, min_calibration_count=2, min_fit_per_class=3
        )
        permutation = torch.tensor([7, 2, 10, 0, 8, 4, 11, 1, 5, 9, 3, 6])
        second = Runner._make_calibration_mask(
            labels[permutation], source_ids[permutation],
            min_calibration_count=2, min_fit_per_class=3
        )
        selected_first = set(source_ids[first].tolist())
        selected_second = set(source_ids[permutation][second].tolist())
        self.assertEqual(selected_first, selected_second)
        self.assertEqual(len(selected_first), 2)
        for class_id in (0, 1):
            class_selected = [
                sid for sid in selected_first
                if sid // 10 == class_id + 1
            ]
            self.assertEqual(len(class_selected), 1)

    def test_base_calibration_mask_never_reuses_fit_sources_as_holdout(self):
        labels = torch.tensor([3, 3, 3, 3])
        source_ids = torch.tensor([201, 202, 203, 204])
        with self.assertRaisesRegex(ValueError, "Insufficient independent support sources"):
            Runner._make_calibration_mask(
                labels, source_ids, min_calibration_count=3, min_fit_per_class=3
            )

    def test_ae_calibration_rejects_source_identity_cross_class_collisions(self):
        labels = torch.tensor([0, 0, 1, 1, 1, 1])
        source_ids = torch.tensor([10, 11, 10, 12, 13, 14])
        with self.assertRaisesRegex(ValueError, "source id 10.*multiple classes"):
            Runner._make_calibration_mask(
                labels, source_ids, min_calibration_count=1,
                min_fit_per_class=3, min_calibration_per_class=1,
            )

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
        torch.testing.assert_close(aux["expert_in"], expected_cls)
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

    def test_incremental_rd_reuses_nearest_historical_anchor_with_zero_up_residual(self):
        torch.manual_seed(271)
        adapter = VisualMoEAdapter(
            8, num_experts=2, reduction=2, descriptor_dim=4,
            descriptor_buffer_size=16, descriptor_residual_rank=3,
        )
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        adapter.descriptors[0].standardized_score = lambda x: torch.full(
            (x.size(0),), 2.0, device=x.device
        )
        adapter.descriptors[1].standardized_score = lambda x: torch.full(
            (x.size(0),), -1.0, device=x.device
        )
        session_features = torch.randn(7, 8)
        anchor_id, anchor_q90 = adapter.select_descriptor_anchor(session_features)
        self.assertEqual(anchor_id, 1)
        torch.testing.assert_close(anchor_q90, torch.tensor([2.0, -1.0]))

        probe = torch.randn(9, 8)
        historical_anchor = adapter.descriptors[anchor_id]
        expected = historical_anchor(probe).detach().clone()
        expert_id = adapter.add_expert(
            session_id=1, reduction=2, anchor_expert_id=anchor_id
        )
        descriptor = adapter.descriptors[expert_id]
        self.assertTrue(descriptor.is_anchored)
        self.assertEqual(descriptor.anchor_expert_id, anchor_id)
        self.assertEqual(descriptor.residual_rank, 3)
        torch.testing.assert_close(descriptor(probe), expected, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(descriptor.residual_up.weight).item()), 0)
        self.assertFalse(descriptor.basis.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in descriptor.trainable_parameters()))
        descriptor.set_trainable(True)
        self.assertTrue(all(parameter.requires_grad for parameter in descriptor.trainable_parameters()))
        self.assertFalse(descriptor.basis.requires_grad)
        descriptor.set_trainable(False)

        # Anchoring a later RD to an already adapted historical RD preserves
        # the complete chained reconstruction function, not just its base basis.
        chained_expected = descriptor(probe).detach().clone()
        chained_id = adapter.add_expert(
            session_id=2, reduction=2, anchor_expert_id=expert_id
        )
        torch.testing.assert_close(
            adapter.descriptors[chained_id](probe), chained_expected, rtol=0, atol=0
        )
        self.assertEqual(adapter.descriptors[chained_id].anchor_expert_id, expert_id)

    def test_anchored_rd_training_updates_only_low_rank_residual_and_calibrates(self):
        torch.manual_seed(277)
        runner, model = build_lifecycle_runner()
        adapter = model.clip_model.visual.transformer.resblocks[0].moe_adapter
        fit_values = torch.randn(30, 8) + torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        calibration_values = torch.randn(8, 8) + torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        adapter.set_base_feature_stats(fit_values)
        anchor = adapter.descriptors[0]
        anchor_basis_before = anchor.basis.detach().clone()
        anchor_center_before = anchor.center.detach().clone()
        expert_id = adapter.add_expert(session_id=1, anchor_expert_id=0)
        descriptor = adapter.descriptors[expert_id]
        basis_before = descriptor.basis.detach().clone()

        loss = runner._train_descriptor(
            adapter, expert_id, fit_values, epochs=8,
            sample_weights=torch.ones(fit_values.size(0)),
            calibration_values=calibration_values,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(descriptor.is_ready())
        torch.testing.assert_close(descriptor.basis, basis_before, rtol=0, atol=0)
        torch.testing.assert_close(anchor.basis, anchor_basis_before, rtol=0, atol=0)
        torch.testing.assert_close(anchor.center, anchor_center_before, rtol=0, atol=0)
        self.assertGreater(float(descriptor.residual_up.weight.abs().sum().item()), 0.0)
        self.assertTrue(all(not parameter.requires_grad for parameter in descriptor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in descriptor.parameters()))

    def test_descriptor_scores_reject_uncalibrated_deployed_experts(self):
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        with self.assertRaisesRegex(RuntimeError, "uncalibrated descriptors"):
            adapter.descriptor_scores(torch.randn(3, 8))

    def test_expert_responsibility_matches_exact_deployed_forward_gates(self):
        torch.manual_seed(257)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.add_expert(session_id=1, reduction=2)
        route = torch.randn(5, 8)
        tokens = torch.randn(3, 5, 8)
        _, aux = adapter(tokens, route_input=route)
        for expert_id in range(adapter.num_experts):
            expected = aux["route_weights"][:, expert_id]
            actual = adapter.expert_responsibility(aux["cls_in"], expert_id)
            torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(
            adapter.expert_responsibility(route, 2),
            torch.softmax(adapter.router(torch.nn.functional.layer_norm(
                route, (8,), eps=1e-5
            ))[0], dim=-1)[:, 0],
        ))

    def test_descriptor_rejects_zero_effective_weight_calibration_state(self):
        descriptor = RepresentationDescriptor(4, descriptor_dim=2, buffer_size=8)
        descriptor.update_reference(torch.tensor([0.1, 0.2, 0.3]))
        descriptor.effective_error_count.zero_()
        self.assertFalse(descriptor.is_ready())
        with self.assertRaisesRegex(RuntimeError, "weighted reference errors"):
            descriptor.standardized_score(torch.randn(2, 4))

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
        self.assertTrue(all(
            "moe_adapter" in name and ".descriptors." not in name and ".layer_norm." not in name
            for name in trainable
        ))
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

    def test_checkpoint_roundtrip_restores_thresholds_and_reference_prototypes(self):
        runner, model = build_lifecycle_runner()
        with tempfile.TemporaryDirectory() as directory:
            runner.cfg.TRAINER.BiMC.CHECKPOINT = SimpleNamespace(
                ENABLE=True, DIR=directory, RESUME="", SAVE_EVERY_SESSION=True
            )
            runner.data_manager = SimpleNamespace(
                get_support_manifest=lambda: {"seed": 7, "support_ids": [1, 2, 3]}
            )
            runner.acc_list = [50.0]
            runner.task_acc_list = [[50.0]]
            runner.session_diagnostics = [{"session": 0}]
            model.base_enhanced_visual_prototype = torch.tensor([[1.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.0]])
            runner._train_base_descriptors([lifecycle_batch(
                0, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), 7000
            )])
            prototype = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            anchor_var = torch.full_like(prototype, 0.1)
            anchor_bank = {
                "class_ids": torch.tensor([0]),
                "route_mean": prototype.clone(), "route_var": anchor_var.clone(),
                "mean": prototype.clone(), "var": anchor_var.clone(),
            }
            state = {
                "class_ids": torch.tensor([0]),
                "image_proto": prototype.clone(),
                "reference_image_proto": prototype.clone(),
                "text_features": prototype.clone(),
                "support_count": torch.tensor([5]),
                "description_proto": prototype.clone(),
                "anchor_by_block": {0: anchor_bank, 1: anchor_bank},
                "creation_session": 0,
            }
            checkpoint_path = runner.save_checkpoint(0, [state])
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], 26)
            self.assertIn("expansion_thresholds", payload)
            self.assertEqual(payload["schema_version"], 27)
            self.assertEqual(payload["expansion_thresholds"], runner.expansion_thresholds)
            self.assertIn("reference_image_proto", payload["task_states"][0])

            restored, restored_model = build_lifecycle_runner()
            restored.cfg.TRAINER.BiMC.CHECKPOINT = SimpleNamespace(
                ENABLE=True, DIR=directory, RESUME=str(checkpoint_path),
                SAVE_EVERY_SESSION=True
            )
            restored.data_manager = SimpleNamespace(
                num_tasks=5,
                class_index_in_task=[torch.tensor([0])] + [
                    torch.tensor([index]) for index in range(1, 5)
                ],
                load_support_manifest=lambda manifest: self.assertEqual(
                    manifest, {"seed": 7, "support_ids": [1, 2, 3]}
                )
            )
            restored._load_resume_checkpoint_if_requested()
            self.assertEqual(restored._resume_completed_session, 0)
            self.assertEqual(len(restored._resume_state_dict_list), 1)
            torch.testing.assert_close(
                restored_model.base_enhanced_visual_prototype, model.base_enhanced_visual_prototype
            )
            self.assertTrue(all(
                not parameter.requires_grad
                for _, adapter in restored._moe_adapters()
                for parameter in adapter.parameters()
            ))
            probe = torch.randn(3, 8)
            model.eval()
            restored_model.eval()
            with torch.no_grad():
                expected_features = model.extract_img_feature(probe)
                restored_features = restored_model.extract_img_feature(probe)
            torch.testing.assert_close(
                restored_features, expected_features, rtol=0, atol=0
            )

            mismatched, _ = build_lifecycle_runner()
            mismatched.cfg.TRAINER.BiMC.VISUAL_MOE.EXPANSION_THRESHOLD_QUANTILE = 0.90
            mismatched.cfg.TRAINER.BiMC.CHECKPOINT = SimpleNamespace(
                ENABLE=True, DIR=directory, RESUME=str(checkpoint_path),
                SAVE_EVERY_SESSION=True
            )
            with self.assertRaisesRegex(ValueError, "expansion policy does not match"):
                mismatched._load_resume_checkpoint_if_requested()

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
        self.assertTrue(restored.descriptors[2].is_anchored)
        self.assertEqual(restored.descriptors[2].anchor_expert_id, 0)
        self.assertEqual(restored.descriptors[2].residual_rank, source.descriptors[2].residual_rank)

    def test_session_expansion_uses_q90_of_best_historical_rd_scores(self):
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
            VISUAL_MOE=SimpleNamespace(EXPANSION_THRESHOLD_QUANTILE=0.95))))
        for descriptor in adapter.descriptors:
            descriptor.update_reference(torch.tensor([0.1, 0.11, 0.09, 0.1]))
        features = {0: {
            "cls_in": torch.randn(8, 8),
            "labels": torch.tensor([2, 2, 2, 2, 3, 3, 3, 3]),
            "source_ids": torch.arange(10, 18),
        }}
        adapter.descriptor_scores = lambda x, require_ready=True: torch.tensor([
            [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
            [2.0, 3.0], [2.1, 3.1], [2.2, 3.2], [4.0, 5.0],
        ])
        decision = model.plan_layer_expansion(0, features, [2, 3], 1.0)
        expected_q90 = torch.quantile(torch.tensor([0.1, 0.2, 0.3, 0.4, 2.0, 2.1, 2.2, 4.0]), 0.90)
        self.assertTrue(decision.expand)
        self.assertAlmostEqual(decision.task_shift_score, float(expected_q90), places=6)
        self.assertEqual(decision.support_count, 8)
        self.assertEqual(decision.threshold, 1.0)

        # Session decisions aggregate all samples: class labels do not create
        # separate tests or separate expert expansion decisions.
        adapter.descriptor_scores = lambda x, require_ready=True: torch.full((8, 2), 0.25)
        decision = model.plan_layer_expansion(0, features, [2, 3], 0.5)
        self.assertFalse(decision.expand)
        self.assertEqual(decision.support_count, 8)

    def test_semantic_guided_calibration_uses_only_base_description_to_retrieve_enhanced_visual(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        runner.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            VISUAL_MOE=SimpleNamespace(
                SEMANTIC_TEMPERATURE=0.05,
                SEMANTIC_CALIBRATION_MAX_ALPHA=0.5,
                PROTOTYPE_RELIABILITY_K=5.0,
            )
        )))
        base = {
            "image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "description_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        old_incremental = {
            "image_proto": torch.tensor([[0.0, -1.0]]),
            "description_proto": torch.tensor([[0.0, -1.0]]),
        }
        visual = torch.tensor([[0.0, 1.0]])
        semantic = torch.tensor([[1.0, 0.0]])
        calibrated, diagnostics = runner._semantic_guided_visual_prototype_calibration(
            visual, semantic, torch.tensor([1]), [base, old_incremental]
        )
        self.assertGreater(calibrated[0, 0].item(), visual[0, 0].item())
        self.assertLess(calibrated[0, 1].item(), visual[0, 1].item())
        self.assertGreater(diagnostics["alpha"].item(), 0.0)
        changed_incremental = dict(old_incremental)
        changed_incremental["image_proto"] = torch.tensor([[-1.0, 0.0]])
        changed, _ = runner._semantic_guided_visual_prototype_calibration(
            visual, semantic, torch.tensor([1]), [base, changed_incremental]
        )
        torch.testing.assert_close(calibrated, changed)
        reliable, reliable_diag = runner._semantic_guided_visual_prototype_calibration(
            visual, semantic, torch.tensor([100]), [base]
        )
        self.assertLess(reliable_diag["alpha"].item(), diagnostics["alpha"].item())
        self.assertGreater(reliable[0, 1].item(), calibrated[0, 1].item())

    def test_classifier_bank_never_fuses_text_semantics_into_visual_prototypes(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        states = [
            {"class_ids": torch.tensor([0]),
             "image_proto": torch.tensor([[1.0, 0.0]]),
             "text_features": torch.tensor([[0.0, 1.0]]),
             "description_proto": torch.tensor([[0.0, 1.0]])},
            {"class_ids": torch.tensor([1]),
             "image_proto": torch.tensor([[0.0, 1.0]]),
             "text_features": torch.tensor([[1.0, 0.0]]),
             "description_proto": torch.tensor([[1.0, 0.0]])},
        ]
        merged = runner.merge_dicts(states)
        torch.testing.assert_close(
            merged["image_proto"], torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )
        self.assertNotIn("final_proto", merged)

    def test_base_training_logits_use_frozen_reference_visual_prototypes(self):
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        model.device = "cpu"
        model.clip_model = nn.Module()
        model.clip_model.logit_scale = nn.Parameter(torch.tensor(0.0))
        model.extract_img_feature_train = lambda images: (torch.tensor([[1.0, 0.0]]), [])
        task_state = {
            "image_proto": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "reference_image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        logits, _, _ = model.forward_train(torch.zeros(1, 3, 2, 2), task_state)
        self.assertEqual(int(logits.argmax(dim=1).item()), 0)

    def test_merge_dicts_builds_one_global_visual_prototype_bank(self):
        runner = Runner.__new__(Runner)
        runner.device = "cpu"
        states = [
            {"class_ids": torch.tensor([0, 1]),
             "image_proto": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
             "text_features": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
             "description_proto": torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
            {"class_ids": torch.tensor([2]),
             "image_proto": torch.tensor([[-1.0, 0.0]]),
             "text_features": torch.tensor([[1.0, 0.0]]),
             "description_proto": torch.tensor([[1.0, 0.0]])},
        ]
        merged = runner.merge_dicts(states)
        self.assertEqual(merged["class_ids"].tolist(), [0, 1, 2])
        torch.testing.assert_close(
            merged["image_proto"],
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        )
        self.assertNotIn("final_proto", merged)

    def test_leave_one_out_cache_rejects_missing_requested_classes(self):
        features = torch.randn(4, 8)
        labels = torch.tensor([0, 0, 1, 1])
        source_ids = torch.tensor([10, 11, 20, 21])
        with self.assertRaisesRegex(ValueError, "missing requested classes"):
            build_loo_cache_from_embeddings(
                features, labels, source_ids, class_index=[0, 1, 2]
            )

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
            return original_plan(
                block_idx, features, class_index, threshold
            )

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
            decision = model.plan_layer_expansion(block, session1_features, [1], 0.05)
            self.assertFalse(decision.expand)

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
            decision = model.plan_layer_expansion(block, session2_features, [2], 0.05)
            self.assertFalse(decision.expand)

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
                [adapter.num_experts for _, adapter in runner._moe_adapters()],
                [expected_counts[session_id], expected_counts[session_id]],
            )
            self.assertEqual(
                [adapter.expert_birth_sessions[-1] for _, adapter in runner._moe_adapters()],
                [session_id, session_id],
            )
            for block in model.expandable_blocks():
                features = model.collect_blockwise_cls_features([support])
                decision = model.plan_layer_expansion(block, features, [session_id], 0.05)
                self.assertFalse(decision.expand)
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
        old_labels = torch.tensor([0] * base_support["image"].size(0) + [1] * session1_support["image"].size(0))
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

        self.assertEqual(before_accuracy, old_labels.numel())
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
            "source_ids": torch.tensor([70, 71, 72, 73]),
        }}
        with self.assertRaises((FloatingPointError, ValueError)):
            model.plan_layer_expansion(0, features, [7], 0.05)

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
                              calibration_values=None, calibration_weights=None):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected AE calibration failure")
            return original_fit(
                adapter, expert_id, values, epochs, sample_weights,
                calibration_values=calibration_values,
                calibration_weights=calibration_weights,
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
