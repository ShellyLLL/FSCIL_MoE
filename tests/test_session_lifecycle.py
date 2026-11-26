"""Synthetic integration test for the complete Base-to-Session-1 MoE lifecycle."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.engine import Runner
from models.bimc import BiMC, LayerExpansionDecision
from models.clip.model import VisualMoEAdapter


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_adapter = VisualMoEAdapter(
            8, num_experts=2, reduction=2, descriptor_dim=1,
            descriptor_buffer_size=32, incremental_gate_bias_init=-1.0,
        )
        with torch.no_grad():
            for expert in self.moe_adapter.experts:
                expert.c_proj.weight.zero_()
                expert.c_proj.bias.zero_()

    def forward_dual(self, adapted, reference):
        residual, aux = self.moe_adapter(adapted, route_input=reference)
        return adapted + residual, reference, aux


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    def extract_reference_prefix_tokens(self, images):
        return images.detach().unsqueeze(1)

    def extract_img_feature_from_reference_prefix(self, prefix):
        return self.encode_image(prefix[:, 0, :])

    def encode_image(self, images, return_moe_aux=False, return_reference=False):
        tokens = torch.stack([images.float(), images.float().roll(1, dims=-1)], dim=0)
        reference = tokens.detach()
        aux = []
        for block in self.visual.transformer.resblocks:
            tokens, reference, item = block.forward_dual(tokens, reference)
            aux.append(item)
        adapted = F.normalize(tokens[0], dim=-1)
        reference_feature = F.normalize(reference[0], dim=-1)
        if return_reference and return_moe_aux:
            return adapted, reference_feature, aux
        if return_reference:
            return adapted, reference_feature
        if return_moe_aux:
            return adapted, aux
        return adapted


def make_batch(class_id, vector, source, count=8):
    generator = torch.Generator().manual_seed(source)
    images = vector.reshape(1, 8).repeat(count, 1)
    images = images + torch.randn(images.shape, generator=generator) * 0.03
    return {
        "image": images,
        "label": torch.full((count,), class_id, dtype=torch.long),
        "source_ids": torch.arange(source * 100, source * 100 + count),
    }


class CheckpointSourceIdentityTests(unittest.TestCase):
    def test_base_reference_subset_and_feature_anchors_validate_against_full_manifest(self):
        # Base RD calibration intentionally retains only a balanced holdout
        # subset, while final-feature anchors are selected from all Base
        # support sources. Both sets must belong to the manifest, but need
        # not be subsets of one another.
        source_to_class = {10: 0, 11: 0, 20: 1, 21: 1}
        reference_sources = Runner._validate_checkpoint_source_rows(
            0, source_to_class, torch.tensor([10, 20]), torch.tensor([0, 1]),
            "reference CLS cache",
        )
        anchor_sources = Runner._validate_checkpoint_source_rows(
            0, source_to_class, torch.tensor([11, 21]), torch.tensor([0, 1]),
            "final feature anchors",
        )
        self.assertEqual(reference_sources, {10, 20})
        self.assertEqual(anchor_sources, {11, 21})
        self.assertTrue(reference_sources.isdisjoint(anchor_sources))

    def test_rejects_anchor_sources_outside_manifest(self):
        source_to_class = {10: 0, 20: 1}
        with self.assertRaisesRegex(ValueError, "outside its support manifest"):
            Runner._validate_checkpoint_source_rows(
                0, source_to_class, torch.tensor([10, 99]), torch.tensor([0, 1]),
                "final feature anchors",
            )

    def test_rejects_source_label_mismatch_against_manifest(self):
        source_to_class = {10: 0, 20: 1}
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            Runner._validate_checkpoint_source_rows(
                0, source_to_class, torch.tensor([10, 20]), torch.tensor([1, 0]),
                "reference CLS cache",
            )


class SessionLifecycleTests(unittest.TestCase):
    def test_base_to_incremental_session_trains_migrated_experts_and_freezes(self):
        torch.manual_seed(23)
        optim = SimpleNamespace(
            BASE_DESCRIPTOR_EPOCHS=8, LR_DESCRIPTOR=0.01,
            LR_INCREMENTAL_EXPERT=0.01, LR_INCREMENTAL_ROUTER=0.01,
            WEIGHT_DECAY=0.0,
        )
        inc = SimpleNamespace(
            EPOCHS=2, LOGIT_SCALE=20.0, L2SP_WEIGHT=1e-3,
            FEATURE_SPACE_PRESERVATION_WEIGHT=1.0, RD_WEIGHT=1.0,
        )
        moe = SimpleNamespace(
            ENABLE=True, NUM_BASE_EXPERTS=2, DESCRIPTOR_RESIDUAL_RANK=2,
            DESCRIPTOR_DENOISE_SCALE=0.02, DESCRIPTOR_RESIDUAL_WEIGHT=1e-3,
            INCREMENTAL_GATE_BIAS_INIT=-4.0, INCREMENTAL_DROPOUT=0.0,
            SEMANTIC_TEMPERATURE=0.07, SEMANTIC_CALIBRATION_MAX_ALPHA=0.5,
            PROTOTYPE_RELIABILITY_K=5.0, FEATURE_SPACE_ANCHORS_PER_CLASS=2,
            FEATURE_SPACE_ANCHOR_BATCH_SIZE=8, FEATURE_SPACE_MIN_COSINE=0.0,
            FEATURE_SPACE_MAX_ACCURACY_DROP=100.0,
        )
        cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
                VISUAL_MOE=moe, OPTIM=optim, INCREMENTAL=inc,
            )),
            DATASET=SimpleNamespace(NUM_INC_CLS=2),
            SEED=1,
        )
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        model.cfg, model.device, model.clip_model = cfg, torch.device("cpu"), TinyCLIP()
        model._apply_freeze_policy()

        runner = Runner.__new__(Runner)
        runner.cfg, runner.device = cfg, torch.device("cpu")
        runner.model = runner.model_without_dp = model
        runner._text_prototypes = lambda _names, ids: (
            F.normalize(torch.eye(8)[torch.as_tensor(ids, dtype=torch.long)], dim=-1),
            F.normalize(torch.eye(8)[torch.as_tensor(ids, dtype=torch.long)], dim=-1),
        )

        @torch.no_grad()
        def build_stats(names, loader, class_ids):
            _, labels, visual, reference = model.inference_all_img_feature(loader)
            ids = torch.as_tensor(class_ids, dtype=torch.long)
            semantic, _ = runner._text_prototypes(names, ids)
            counts = torch.stack([labels.eq(int(cid)).sum() for cid in ids])
            return {
                "class_ids": ids, "image_proto": visual,
                "reference_image_proto": reference, "text_features": semantic,
                "description_proto": semantic.clone(), "support_count": counts,
            }
        model.build_task_statistics = build_stats

        base = make_batch(0, torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]), 10)
        runner._train_base_descriptors([base], [base])
        with torch.no_grad():
            adapted, _, reference = model.extract_img_feature_train(
                base["image"], return_reference=True
            )
        semantic = F.normalize(torch.eye(8)[[0]], dim=-1)
        base_state = {
            "class_ids": torch.tensor([0]),
            "image_proto": F.normalize(adapted.mean(0, keepdim=True), dim=-1),
            "reference_image_proto": F.normalize(reference.mean(0, keepdim=True), dim=-1),
            "text_features": semantic, "description_proto": semantic,
            "support_count": torch.tensor([8]),
            "anchor_by_block": runner._collect_class_anchors([base]),
            "final_feature_anchors": runner._collect_final_feature_anchors([base], 0),
            "creation_session": 0,
        }

        # A REUSE-only Session must preserve the original five-shot prototype
        # exactly; it must not replace it with the compact feature-anchor mean.
        original_base_prototype = base_state["image_proto"].clone()
        def force_reuse(_self, block_idx, feature_map, class_ids):
            count = int(feature_map[int(block_idx)]["cls_in"].size(0))
            return LayerExpansionDecision(int(block_idx), False, 0.2, 1.0, 0, count)
        model.plan_layer_expansion = force_reuse.__get__(model, BiMC)
        reuse_result = runner.train_incremental_task(
            1, [make_batch(1, torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.]), 20),
                make_batch(2, torch.tensor([0.,0.,1.,0.,0.,0.,0.,0.]), 30)],
            [make_batch(1, torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.]), 20),
             make_batch(2, torch.tensor([0.,0.,1.,0.,0.,0.,0.,0.]), 30)],
            [1,2], ["novel_a", "novel_b"], [base_state],
        )
        self.assertEqual(len(reuse_result["expanded"]), 0)
        self.assertEqual(len(reuse_result["reused"]), 2)
        torch.testing.assert_close(base_state["image_proto"], original_base_prototype, rtol=0, atol=0)
        self.assertNotIn("refreshed_old_states", reuse_result)

        # Force the two layer decisions only in this integration test; the
        # numeric Session-mean Z boundary is independently tested in unit tests.
        sequential_checks = []
        def force_expand(_self, block_idx, feature_map, class_ids):
            if int(block_idx) == 1:
                previous = model.clip_model.visual.transformer.resblocks[0].moe_adapter
                self.assertEqual(previous.num_experts, 3)
                self.assertTrue(previous.descriptors[-1].is_ready())
                self.assertTrue(all(not parameter.requires_grad for parameter in previous.parameters()))
                sequential_checks.append("block0_trained_and_frozen_before_block1_decision")
            count = int(feature_map[int(block_idx)]["cls_in"].size(0))
            return LayerExpansionDecision(int(block_idx), True, 2.0, 1.0, 0, count)
        model.plan_layer_expansion = force_expand.__get__(model, BiMC)

        novel_a = make_batch(1, torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.]), 20)
        novel_b = make_batch(2, torch.tensor([0.,0.,1.,0.,0.,0.,0.,0.]), 30)
        result = runner.train_incremental_task(
            1, [novel_a, novel_b], [novel_a, novel_b], [1,2],
            ["novel_a", "novel_b"], [base_state],
        )
        self.assertEqual(len(result["expanded"]), 2)
        self.assertEqual(sequential_checks, ["block0_trained_and_frozen_before_block1_decision"])
        self.assertEqual(result["task_state"]["class_ids"].tolist(), [1,2])
        torch.testing.assert_close(base_state["image_proto"], original_base_prototype, rtol=0, atol=0)
        self.assertNotIn("refreshed_old_states", result)
        self.assertTrue(all(
            item["expert_anchor_id"] == item["descriptor_anchor_id"]
            for item in result["expanded"]
        ))
        for _, adapter in runner._moe_adapters():
            self.assertEqual(adapter.num_experts, 3)
            self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))
        for history in result["training"].values():
            self.assertEqual(history["classification_samples_per_epoch"], [16,16])
            self.assertEqual(len(history["l2sp_loss"]), 2)
            self.assertEqual(len(history["ce_loss"]), 2)
            self.assertEqual(len(history["feature_space_loss"]), 2)
            self.assertNotIn("rd_loss", history)
            self.assertGreaterEqual(history["rd_effective_reference_count"], 1.0)

        # A second Session must detect and migrate from the already deployed
        # incremental generation, not assume that all anchors are Base experts.
        def expand_from_generation_two(_self, block_idx, feature_map, class_ids):
            count = int(feature_map[int(block_idx)]["cls_in"].size(0))
            return LayerExpansionDecision(int(block_idx), True, 2.5, 1.0, 2, count)
        model.plan_layer_expansion = expand_from_generation_two.__get__(model, BiMC)
        novel_c = make_batch(3, torch.tensor([0.,0.,0.,1.,0.,0.,0.,0.]), 40)
        novel_d = make_batch(4, torch.tensor([0.,0.,0.,0.,1.,0.,0.,0.]), 50)
        result2 = runner.train_incremental_task(
            2, [novel_c, novel_d], [novel_c, novel_d], [3,4],
            ["novel_c", "novel_d"], [base_state, result["task_state"]],
        )
        self.assertEqual(len(result2["expanded"]), 2)
        self.assertTrue(all(item["expert_anchor_id"] == 2 for item in result2["expanded"]))
        self.assertTrue(all(item["descriptor_anchor_id"] == 2 for item in result2["expanded"]))
        for _, adapter in runner._moe_adapters():
            self.assertEqual(adapter.num_experts, 4)
            self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))


if __name__ == "__main__":
    unittest.main()
