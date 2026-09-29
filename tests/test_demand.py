"""CPU-only invariants for versioned FSCIL expert growth and source-exclusive CV.

Run: python -m unittest discover -s tests -p 'test_demand.py'
No dataset or pretrained CLIP checkpoint is required.
"""
import types
import unittest

import torch
import torch.nn.functional as F

# Load the standalone layer file without models.clip.__init__, which imports
# runtime image downloading/transforms irrelevant to these CPU-only invariants.
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location(
    "fscil_clip_layers", Path(__file__).resolve().parents[1] / "models" / "clip" / "model.py"
)
_layers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layers)
VisionTransformer = _layers.VisionTransformer
VisualMoEAdapter = _layers.VisualMoEAdapter
from engine.demand import (
    VersionedScorer, class_prototypes, source_stratified_folds, topology_key,
)


class TestVersionedRouter(unittest.TestCase):
    def test_historical_prefix_exact_after_growth_and_rollback(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(
            d_model=16, num_experts=2, reduction=4, descriptor_dim=4,
        )
        adapter.eval()
        x = torch.randn(5, 3, 16)
        before, aux_before = adapter(x, active_count=2)
        adapter.add_expert(session_id=1)
        with torch.no_grad():
            adapter.experts[-1].c_proj.weight.normal_(0.0, 0.2)
            adapter.experts[-1].c_proj.bias.fill_(0.1)
            adapter.router.columns[-1].bias.fill_(2.0)
        after, aux_after = adapter(x, active_count=2)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        torch.testing.assert_close(
            aux_after["route_weights"], aux_before["route_weights"], atol=0, rtol=0
        )
        new, _ = adapter(x)
        self.assertFalse(torch.equal(new, before))
        adapter.set_descriptor_topology(2, {8: 2, 11: 3})
        topology = adapter.export_topology()
        self.assertEqual(topology["experts"][-1]["descriptor_topology"], {8: 2, 11: 3})
        adapter.discard_newest()
        self.assertEqual(adapter.num_experts, 2)
        rolled, _ = adapter(x)
        torch.testing.assert_close(rolled, before, atol=0, rtol=0)

    def test_full_vit_old_path_unchanged(self):
        torch.manual_seed(31)
        model = VisionTransformer(
            input_resolution=16, patch_size=8, width=32,
            layers=2, heads=4, output_dim=16,
        ).eval()
        model.transformer.resblocks[0].moe_adapter = VisualMoEAdapter(
            32, num_experts=2, reduction=4, descriptor_dim=8
        )
        model.transformer.resblocks[1].moe_adapter = VisualMoEAdapter(
            32, num_experts=2, reduction=4, descriptor_dim=8
        )
        images = torch.randn(3, 3, 16, 16)
        ref = {0: 2, 1: 2}
        with torch.no_grad():
            old = model(images, active_counts=ref)
            first = model.transformer.resblocks[0].moe_adapter
            first.add_expert(1)
            first.experts[-1].c_proj.bias.fill_(0.7)
            first.router.columns[-1].bias.fill_(3.0)
            pinned = model(images, active_counts=ref)
            current = model(images)
        torch.testing.assert_close(old, pinned, atol=0, rtol=0)
        self.assertFalse(torch.equal(old, current))


class TestFoldIsolation(unittest.TestCase):
    def test_class_stratified_loo_source_exclusivity(self):
        dataset = types.SimpleNamespace(
            labels=[2] * 5 + [9] * 5,
            source_ids=[100, 103, 110, 101, 120, 202, 201, 215, 211, 230],
        )
        folds = list(source_stratified_folds(dataset, 5))
        self.assertEqual(len(folds), 5)
        held_out = []
        for train, val in folds:
            self.assertEqual(len(train), 8)
            self.assertEqual(len(val), 2)
            self.assertFalse(set(train) & set(val))
            self.assertEqual({dataset.labels[i] for i in val}, {2, 9})
            held_out.extend(val)
        self.assertEqual(sorted(held_out), list(range(10)))

    def test_leave_out_self_from_class_prototype(self):
        f = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        labels = torch.tensor([1, 1, 2, 2])
        mask = torch.tensor([True, False, False, False])
        p = class_prototypes(f, labels, torch.tensor([1, 2]), mask)
        torch.testing.assert_close(p[0], torch.tensor([0., 1.]))
        with self.assertRaises(ValueError):
            class_prototypes(f[:2], labels[:2], torch.tensor([1]), torch.tensor([True, True]))


class TinyClip:
    """Minimal owner-topology encoder for deterministic S7 score invariance."""
    def encode_image(self, images, active_counts=None):
        count = int(active_counts[0])
        features = images.clone()
        if count > 2:
            features = features + torch.tensor([0.7, -0.2])
        return features


class TestVersionedScoring(unittest.TestCase):
    def test_old_logits_unchanged_after_new_version(self):
        cfg = types.SimpleNamespace(
            DATASET=types.SimpleNamespace(BETA=0.3),
            TRAINER=types.SimpleNamespace(
                BiMC=types.SimpleNamespace(
                    VISION_CALIBRATION=False,
                    INCREMENTAL=types.SimpleNamespace(LOGIT_SCALE=25.0)
                )
            )
        )
        model = types.SimpleNamespace(
            cfg=cfg, device="cpu",
            clip_model=TinyClip(),
            base_vision_prototype=F.normalize(torch.eye(2), dim=-1),
        )
        old = [{
            "class_ids": torch.tensor([0]),
            "topology": {0: 2},
            "ref_fused_proto": F.normalize(torch.tensor([[1., 0.]]), dim=-1),
            "fused_proto": F.normalize(torch.tensor([[1., 0.]]), dim=-1),
        }]
        new_text = {
            "class_ids": torch.tensor([1]),
            "calibrated_text": torch.tensor([[0., 1.]]),
        }
        score = VersionedScorer(model, old, new_text, {0: 2}, eta=0.5)
        images = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        ref = torch.tensor([[0., 1.]])
        dyn = torch.tensor([[0., 1.]])
        a = score.logits(images, {0: 2}, ref, dyn)
        b = score.logits(images, {0: 3}, ref, dyn)
        torch.testing.assert_close(a[:, 0], b[:, 0], atol=0, rtol=0)
        self.assertFalse(torch.equal(a[:, 1], b[:, 1]))
        self.assertEqual(topology_key({11: 2, 8: 1}), topology_key({"8": 1, "11": 2}))


if __name__ == "__main__":
    unittest.main()
