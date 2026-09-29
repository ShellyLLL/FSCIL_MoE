"""Invariants for function-preserving expansion and versioned classifiers."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.clip.model import VisionTransformer, VisualMoEAdapter
from models.bimc import BiMC, build_loo_cache_from_embeddings
from engine.engine import Runner


class TinySupport(Dataset):
    def __init__(self, shots=5):
        self.labels = [2] * shots + [3] * shots
        self.source_ids = list(range(2 * shots))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = F.one_hot(torch.tensor(self.labels[index]), num_classes=4).float()
        return {"image": value, "label": self.labels[index],
                "source_id": self.source_ids[index]}


class ExpansionInvariantTests(unittest.TestCase):
    def test_appending_an_expert_preserves_the_old_function(self):
        torch.manual_seed(7)
        adapter = VisualMoEAdapter(8, num_experts=2, reduction=2)
        adapter.eval()
        inputs = torch.randn(5, 3, 8)
        before, _ = adapter(inputs)
        adapter.add_expert(session_id=1, reduction=2)
        at_creation, _ = adapter(inputs)
        torch.testing.assert_close(at_creation, before, rtol=0, atol=0)
        with torch.no_grad():
            adapter.experts[-1].c_proj.bias.fill_(0.3)
        old, _ = adapter(inputs, max_session=0)
        current, _ = adapter(inputs, max_session=1)
        torch.testing.assert_close(old, before, rtol=0, atol=0)
        self.assertGreater((current - old).abs().max().item(), 0)
        adapter.remove_newest_expert()
        restored, _ = adapter(inputs)
        torch.testing.assert_close(restored, before, rtol=0, atol=0)

    def test_later_experts_cannot_change_an_old_visual_encoder(self):
        torch.manual_seed(11)
        visual = VisionTransformer(8, 4, 8, 2, 2, 8).eval()
        visual.transformer.resblocks[-1].moe_adapter = VisualMoEAdapter(
            8, num_experts=2, reduction=2)
        images = torch.randn(2, 3, 8, 8)
        before = visual(images, max_session=0)
        adapter = visual.transformer.resblocks[-1].moe_adapter
        adapter.add_expert(1, reduction=2)
        with torch.no_grad():
            adapter.experts[-1].c_proj.bias.fill_(0.4)
        adapter.freeze_all()
        torch.testing.assert_close(visual(images, max_session=0), before, rtol=0, atol=0)
        self.assertGreater((visual(images, max_session=1) - before).abs().max().item(), 0)

    def test_topology_and_weights_restore_versioned_predictions(self):
        torch.manual_seed(13)
        source = VisualMoEAdapter(8, num_experts=2, reduction=2).eval()
        source.add_expert(1, reduction=4)
        with torch.no_grad():
            source.experts[-1].c_proj.bias.fill_(0.2)
        source.freeze_all()
        inputs = torch.randn(3, 2, 8)
        expected = [source(inputs, max_session=version)[0] for version in (0, 1)]
        restored = VisualMoEAdapter(8, num_experts=2, reduction=2)
        restored.rebuild_topology(source.export_topology())
        restored.load_state_dict(source.state_dict(), strict=True)
        for version, value in enumerate(expected):
            torch.testing.assert_close(restored(inputs, max_session=version)[0],
                                       value, rtol=0, atol=0)

    def test_visual_calibration_transports_only_the_paired_shift(self):
        model = BiMC.__new__(BiMC)
        torch.nn.Module.__init__(model)
        model.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            LAMBDA_I=0.25, TAU=4)))
        model.base_vision_prototype = F.normalize(torch.eye(3)[:2], dim=-1)
        reference = F.normalize(torch.tensor([[0.7, 0.7, 0.0]]), dim=-1)
        current = F.normalize(torch.tensor([[0.6, 0.7, 0.4]]), dim=-1)
        expected = F.normalize(model.soft_calibration(model.base_vision_prototype,
                                                       reference) + current - reference, dim=-1)
        torch.testing.assert_close(model.calibrate_incremental_visual(current, reference),
                                   expected)

    def test_leave_one_out_excludes_the_query_source(self):
        features = F.normalize(torch.tensor([[1., 0.], [0., 1.],
                                             [-1., 0.], [0., -1.]]), dim=-1)
        cache = build_loo_cache_from_embeddings(
            features, torch.tensor([2, 2, 3, 3]), torch.arange(4))
        torch.testing.assert_close(cache["loo_visual_prototypes"][0, 0], features[1])
        self.assertTrue(cache["loo_valid"].all())

    def test_visual_proto_is_not_recovered_from_normalized_fusion(self):
        model = BiMC.__new__(BiMC)
        torch.nn.Module.__init__(model)
        model.device = "cpu"
        model.template = ["a photo of a {}"]
        model.cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
                TEXT_CALIBRATION=True, LAMBDA_T=0.5,
                FUSED_CONFLICT_STRENGTH=0.0, FUSED_CONFLICT_MARGIN=0.05)),
            DATASET=SimpleNamespace(GPT_PATH="unused", BETA=0.3))
        model.inference_text_feature = lambda *args: (
            F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1), None)
        model.inference_all_description_feature = lambda *args: (
            None, None, F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1))
        model.inference_all_img_feature = lambda *args, **kwargs: (
            [torch.tensor([1., 0.])], None, torch.tensor([[1., 0.]]))
        state = model.build_task_statistics(["class"], None, [0], encoder_version=0)
        torch.testing.assert_close(state["image_proto"], torch.tensor([[1., 0.]]))
        self.assertEqual(state["encoder_version"], 0)

    def test_support_folds_exclude_every_held_out_source(self):
        runner = Runner.__new__(Runner)
        dataset = TinySupport()
        train = DataLoader(dataset, batch_size=4, shuffle=True)
        extract = DataLoader(dataset, batch_size=4, shuffle=False)
        folds = runner._support_folds(train, extract)
        self.assertEqual(len(folds), 5)
        for fit_train, fit_extract, validation in folds:
            fitting = set(fit_extract.dataset.indices)
            held_out = set(validation.dataset.indices)
            self.assertFalse(fitting & held_out)
            self.assertEqual(len(held_out), 2)
            self.assertEqual(len(fit_train.dataset), 8)

    def test_versioned_scores_keep_historical_columns_fixed(self):
        runner = Runner.__new__(Runner)

        class Encoder:
            shift = 0.0

            def extract_img_feature(self, images, max_session=None):
                return images + (self.shift if max_session == 1 else 0.0) * torch.tensor([0., 1.])

        runner.model_without_dp = Encoder()
        images = torch.tensor([[1., 0.], [0., 1.]])
        prototypes = torch.tensor([[1., 0.], [0., 1.]])
        versions = torch.tensor([0, 1])
        first = runner._versioned_scores(images, prototypes, versions, 100.0)
        runner.model_without_dp.shift = 0.8
        second = runner._versioned_scores(images, prototypes, versions, 100.0)
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
        self.assertGreater((first[:, 1] - second[:, 1]).abs().max().item(), 0)

    def test_candidate_training_uses_fused_leave_one_out_scores(self):
        from torch import nn

        class TinyModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                block = nn.Module()
                block.moe_adapter = VisualMoEAdapter(4, num_experts=2, reduction=2)
                transformer = nn.Module()
                transformer.resblocks = nn.ModuleList([block])
                visual = nn.Module()
                visual.transformer = transformer
                self.clip_model = nn.Module()
                self.clip_model.visual = visual
                self.base_vision_prototype = torch.eye(4)[:2]
                self.cfg, self.template = cfg, ["a {}"]

            def extract_img_feature(self, images, max_session=None):
                adapter = self.clip_model.visual.transformer.resblocks[0].moe_adapter
                residual, _ = adapter(torch.stack([images, images]), max_session=max_session)
                return images + residual[0]

            def build_leave_one_out_prototypes(self, loader, class_index, max_session=None):
                features, labels, sources = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        features.append(F.normalize(self.extract_img_feature(
                            batch["image"], max_session), dim=-1))
                        labels.append(batch["label"])
                        sources.append(batch["source_id"])
                return build_loo_cache_from_embeddings(torch.cat(features),
                                                       torch.cat(labels), torch.cat(sources), class_index)

            def soft_calibration(self, base, current):
                return BiMC.soft_calibration(self, base, current)

            def calibrate_incremental_visual(self, current, reference):
                return BiMC.calibrate_incremental_visual(self, current, reference)

            def inference_text_feature(self, names, template, first):
                return torch.eye(4)[first:first + len(names)], None

            def inference_all_description_feature(self, names, path, first):
                return None, None, torch.eye(4)[first:first + len(names)]

            def expandable_blocks(self):
                return [0]

            def add_expert(self, block_idx, session_id):
                return self.clip_model.visual.transformer.resblocks[block_idx].moe_adapter.add_expert(
                    session_id, reduction=2)

        cfg = SimpleNamespace(
            DATASET=SimpleNamespace(BETA=0.3, GPT_PATH="unused"),
            TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
                TEXT_CALIBRATION=True, LAMBDA_T=0.5, LAMBDA_I=0.1, TAU=16,
                INCREMENTAL=SimpleNamespace(EPOCHS=1, LOGIT_SCALE=100.0),
                OPTIM=SimpleNamespace(LR_INCREMENTAL_EXPERT=1e-3,
                                      LR_INCREMENTAL_ROUTER=1e-3, WEIGHT_DECAY=0.0))))
        runner = Runner.__new__(Runner)
        runner.cfg, runner.device = cfg, "cpu"
        runner.model = runner.model_without_dp = TinyModel(cfg)
        dataset = TinySupport(shots=3)
        training = DataLoader(dataset, batch_size=4, shuffle=True)
        extraction = DataLoader(dataset, batch_size=4, shuffle=False)
        old = {"class_ids": torch.tensor([0, 1]), "encoder_version": 0,
               "image_proto": torch.eye(4)[:2], "text_features": torch.eye(4)[:2],
               "description_proto": torch.eye(4)[:2], "fused_proto": torch.eye(4)[:2]}
        result = runner.train_incremental_task(
            1, training, extraction, [2, 3], ["two", "three"], [old], True)
        self.assertEqual(len(result["decisions"]), 1)
        self.assertFalse(result["trained"])
        self.assertEqual(runner._adapter_by_block(0).num_experts, 2)

        # Exercise the acceptance and full-support retraining path independently
        # of this synthetic support set's actual predictive quality.
        runner._held_out_loss = lambda *args: 1.0 if args[6] == 0 else 0.0
        accepted = runner.train_incremental_task(
            1, training, extraction, [2, 3], ["two", "three"], [old], True)
        self.assertTrue(accepted["trained"])
        self.assertEqual(accepted["encoder_version"], 1)
        self.assertEqual(runner._adapter_by_block(0).num_experts, 3)


if __name__ == "__main__":
    unittest.main()
