"""End-to-end synthetic APCRunner, versioned-classifier and checkpoint tests.

These tests use the real APCRunner/Runner/VisualMoEAdapter implementations. Only
the CLIP image/text source and FSCIL data manager are replaced by deterministic
in-memory fixtures; candidate training, held-out global classification, task
gating, topology mutation and checkpoint restore are not mocked.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from engine.apc import AdaptationAwarePrototypeController
from engine.apc_runner import APCRunner
from models.bimc import build_loo_cache_from_embeddings
from models.clip.model import VisualMoEAdapter
from utils.evaluator import AccuracyEvaluator


DIM = 8
TASK_CLASSES = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))


def class_center(class_id):
    value = torch.zeros(DIM)
    if class_id == 0:
        value[0] = 1.0
    elif class_id == 1:
        value[1] = 1.0
    elif class_id in (2, 3):
        value[2] = 1.0
        value[3] = 0.018 if class_id == 2 else -0.018
    elif class_id in (4, 5):
        value[4] = 1.0
        value[5] = 0.018 if class_id == 4 else -0.018
    return F.normalize(value, dim=0)


class SyntheticSupport(Dataset):
    def __init__(self, class_ids, shots=5, seed=17, noise=0.002):
        self.labels = np.asarray(
            [int(cls) for cls in class_ids for _ in range(shots)], dtype=np.int64
        )
        self.source_ids = np.asarray(
            [int(cls) * 1000 + shot for cls in class_ids for shot in range(shots)],
            dtype=np.int64,
        )
        values = []
        for class_id in class_ids:
            for shot in range(shots):
                generator = torch.Generator().manual_seed(seed + int(class_id) * 101 + shot)
                values.append(class_center(int(class_id)) + noise * torch.randn(
                    DIM, generator=generator
                ))
        self.images = torch.stack(values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "image": self.images[index].clone(),
            "label": int(self.labels[index]),
            "source_id": int(self.source_ids[index]),
        }


class SyntheticDataManager:
    def __init__(self):
        self.class_index_in_task = list(TASK_CLASSES)
        self.class_names = [f"class_{index}" for index in range(6)]
        self.num_tasks = len(self.class_index_in_task)
        self.support_manifest = {
            "seed": 17,
            "sessions": {
                str(task): {
                    str(int(cls)): [int(cls) * 1000 + shot for shot in range(5)]
                    for cls in classes
                }
                for task, classes in enumerate(self.class_index_in_task)
            },
        }

    def _classes(self, task_id, accumulate_past):
        if accumulate_past:
            return np.concatenate(self.class_index_in_task[:task_id + 1])
        return self.class_index_in_task[task_id]

    def get_dataloader(self, task_id, source, mode="test", accumulate_past=False):
        if source == "train":
            classes = self._classes(task_id, accumulate_past)
            dataset = SyntheticSupport(classes, shots=5, seed=17, noise=0.002)
            return DataLoader(dataset, batch_size=8, shuffle=(mode == "train"))
        if source == "test":
            classes = np.concatenate(self.class_index_in_task[:task_id + 1])
            dataset = SyntheticSupport(classes, shots=5, seed=901, noise=0.002)
            return DataLoader(dataset, batch_size=10, shuffle=False)
        raise ValueError(source)

    def get_support_manifest(self):
        return self.support_manifest

    def load_support_manifest(self, manifest):
        self.support_manifest = manifest


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_adapter = VisualMoEAdapter(
            DIM, num_experts=2, reduction=1, residual_scale_init=1.0
        )
        # The base visual path is an identity map in this controlled experiment.
        # Incremental experts still receive the real zero-up initialization.
        with torch.no_grad():
            for expert in self.moe_adapter.experts:
                expert.c_proj.weight.zero_()
                expert.c_proj.bias.zero_()


class _TinyVisual(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        transformer = nn.Module()
        transformer.resblocks = nn.ModuleList([_TinyBlock() for _ in range(blocks)])
        self.transformer = transformer


class _TinyClip(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.visual = _TinyVisual(blocks)


class TinyVersionedModel(nn.Module):
    """Small differentiable encoder that executes the production MoE adapters."""

    def __init__(self, blocks=2):
        super().__init__()
        self.clip_model = _TinyClip(blocks)
        self.template = ["a photo of a {}"]
        self.base_vision_prototype = None
        self._fused_history_prototypes = None
        self._fused_history_class_ids = None

    def _blocks(self):
        return self.clip_model.visual.transformer.resblocks

    def extract_img_feature_train(self, images, max_session=None):
        x = images
        aux_list = []
        for block in self._blocks():
            residual, aux = block.moe_adapter(x.unsqueeze(0), max_version=max_session)
            x = x + residual[0]
            aux["block_out"] = x
            aux_list.append(aux)
        return x, aux_list

    @torch.no_grad()
    def extract_img_feature(self, images, max_version=None, max_session=None):
        if max_session is not None:
            if max_version is not None:
                raise ValueError("Specify max_version or max_session, not both.")
            max_version = max_session
        return self.extract_img_feature_train(images, max_version)[0]

    @torch.no_grad()
    def collect_blockwise_cls_features(self, loader, max_version=None):
        output = {}
        for batch in loader:
            x = batch["image"]
            labels = batch["label"].long()
            sources = batch["source_id"].long()
            for block_idx, block in enumerate(self._blocks()):
                residual, aux = block.moe_adapter(x.unsqueeze(0), max_version=max_version)
                x = x + residual[0]
                aux["block_out"] = x
                item = output.setdefault(block_idx, {
                    "cls_in": [], "cls_out": [], "block_out": [],
                    "patch_mean_in": [], "labels": [], "source_ids": [],
                })
                item["cls_in"].append(aux["cls_in"].cpu())
                item["cls_out"].append(aux["cls_out"].cpu())
                item["block_out"].append(aux["block_out"].cpu())
                item["labels"].append(labels.cpu())
                item["source_ids"].append(sources.cpu())
        for item in output.values():
            for key in ("cls_in", "cls_out", "block_out", "labels", "source_ids"):
                item[key] = torch.cat(item[key], dim=0)
            item["patch_mean_in"] = item["cls_in"].new_empty((0, DIM))
        return output

    @torch.no_grad()
    def build_leave_one_out_prototypes(self, loader, class_index=None,
                                       max_version=None, max_session=None):
        if max_session is not None:
            if max_version is not None:
                raise ValueError("Specify max_version or max_session, not both.")
            max_version = max_session
        features, labels, sources = [], [], []
        for batch in loader:
            features.append(F.normalize(self.extract_img_feature(
                batch["image"], max_version), dim=-1))
            labels.append(batch["label"].long())
            sources.append(batch["source_id"].long())
        return build_loo_cache_from_embeddings(
            torch.cat(features), torch.cat(labels), torch.cat(sources), class_index
        )

    @torch.no_grad()
    def inference_all_img_feature(self, loader, max_version=None, max_session=None):
        if max_session is not None:
            if max_version is not None:
                raise ValueError("Specify max_version or max_session, not both.")
            max_version = max_session
        features, labels = [], []
        for batch in loader:
            features.append(F.normalize(self.extract_img_feature(
                batch["image"], max_version), dim=-1))
            labels.append(batch["label"].long())
        features, labels = torch.cat(features), torch.cat(labels)
        prototypes = torch.stack([
            features[labels.eq(cls)].mean(dim=0) for cls in torch.unique(labels, sorted=True)
        ])
        return features, labels, F.normalize(prototypes, dim=-1)

    def _text_for_names(self, names):
        ids = [int(str(name).split("_")[-1]) for name in names]
        return F.normalize(torch.stack([class_center(value) for value in ids]), dim=-1)

    @torch.no_grad()
    def inference_text_feature(self, names, template, first):
        return self._text_for_names(names), None

    @torch.no_grad()
    def inference_all_description_feature(self, names, path, first):
        return None, None, self._text_for_names(names)

    def soft_calibration(self, base, current):
        return F.normalize(current, dim=-1)

    def calibrate_incremental_visual(self, current, reference):
        return F.normalize(current, dim=-1)

    @staticmethod
    def _source_ids(loader, count):
        dataset = loader.dataset
        if isinstance(dataset, Subset):
            base = dataset.dataset
            ids = torch.as_tensor(base.source_ids, dtype=torch.long)[
                torch.as_tensor(dataset.indices, dtype=torch.long)
            ]
        else:
            ids = torch.as_tensor(dataset.source_ids, dtype=torch.long)
        if ids.numel() != count:
            raise ValueError("Synthetic source IDs are not aligned with support features.")
        return ids

    @torch.no_grad()
    def build_task_statistics(self, class_names, loader, class_index,
                              calibrate_novel_vision_proto=False, encoder_version=None):
        version = 0 if encoder_version is None else int(encoder_version)
        image_features, labels, _ = self.inference_all_img_feature(
            loader, max_version=version
        )
        ids = torch.as_tensor(class_index, dtype=torch.long)
        image_proto = F.normalize(torch.stack([
            image_features[labels.eq(cls)].mean(dim=0) for cls in ids
        ]), dim=-1)
        anchor_features, anchor_labels = image_features, labels
        anchor_proto = image_proto
        if int(ids[0]) != 0 and version != 0:
            anchor_features, anchor_labels, _ = self.inference_all_img_feature(
                loader, max_version=0
            )
            anchor_proto = F.normalize(torch.stack([
                anchor_features[anchor_labels.eq(cls)].mean(dim=0) for cls in ids
            ]), dim=-1)
        text = self._text_for_names(class_names)
        description = text.clone()
        beta = float(self._test_cfg.DATASET.BETA)
        fused = F.normalize(beta * text + (1.0 - beta) * image_proto, dim=-1)
        anchor_fused = F.normalize(beta * text + (1.0 - beta) * anchor_proto, dim=-1)
        if int(ids[0]) == 0:
            self.base_vision_prototype = image_proto.detach().clone()
        if self._fused_history_prototypes is None:
            self._fused_history_prototypes = fused.detach().clone()
            self._fused_history_class_ids = ids.clone()
        else:
            keep = ~torch.isin(self._fused_history_class_ids, ids)
            self._fused_history_prototypes = torch.cat([
                self._fused_history_prototypes[keep], fused.detach()
            ], dim=0)
            self._fused_history_class_ids = torch.cat([
                self._fused_history_class_ids[keep], ids
            ], dim=0)
        return {
            "description_proto": description,
            "text_features": text,
            "image_proto": image_proto,
            "class_index": ids,
            "class_ids": ids,
            "sample_cnt": int(len(image_features)),
            "initial_fused_proto": fused,
            "fused_proto": fused,
            "anchor_fused_proto": anchor_fused,
            "anchor_features": anchor_features.detach().cpu().half(),
            "anchor_labels": anchor_labels.detach().cpu(),
            "anchor_source_ids": self._source_ids(loader, len(anchor_labels)),
            "encoder_version": version,
        }

    @staticmethod
    def fuse_prototypes(state):
        return F.normalize(state["fused_proto"], dim=-1)

    def add_expert(self, block_idx, session_id, version_id=None):
        return self._blocks()[int(block_idx)].moe_adapter.add_expert(
            session_id=session_id, reduction=1, version_id=version_id
        )

    def export_moe_topology(self):
        return {
            str(index): block.moe_adapter.export_topology()
            for index, block in enumerate(self._blocks())
        }

    def rebuild_moe_topology(self, topology):
        for index, block in enumerate(self._blocks()):
            block.moe_adapter.rebuild_topology(topology[str(index)])

    @torch.no_grad()
    def restore_fused_history(self, states):
        if not states:
            self._fused_history_prototypes = None
            self._fused_history_class_ids = None
            return
        self._fused_history_prototypes = torch.cat([
            self.fuse_prototypes(state) for state in states
        ], dim=0)
        self._fused_history_class_ids = torch.cat([
            torch.as_tensor(state["class_ids"]).long() for state in states
        ])


class AlwaysCandidateAPC(AdaptationAwarePrototypeController):
    """Use real APC features/calibration, but force candidate evaluation for coverage."""

    def decide(self, features_by_block, old_ids, new_ids, state):
        decisions = super().decide(features_by_block, old_ids, new_ids, state)
        for row in decisions:
            row["expand"] = True
            row["reason"] = "integration_candidate"
        return decisions


class SyntheticAPCRunner(APCRunner):
    def train_base_task(self, task_stat, train_loader, extract_loader):
        # Base path is intentionally identity-initialized; freeze it exactly as
        # the production base stage does after fitting.
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

    def _routing_diagnostics(self, loader, tag, current_task=None):
        return {}

    def _log_expert_topology(self, tag):
        return


def make_cfg(checkpoint_dir, resume=""):
    return SimpleNamespace(
        SEED=17,
        DATASET=SimpleNamespace(
            NUM_INC_CLS=2, NUM_INC_SHOT=2, BETA=0.0, GPT_PATH="unused"
        ),
        TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            END_SESSION=2,
            VISION_CALIBRATION=False,
            TEXT_CALIBRATION=False,
            LAMBDA_T=0.0,
            APC=SimpleNamespace(ALPHA=0.05, CALIBRATION_TRIALS=8, SEED=3),
            INCREMENTAL=SimpleNamespace(
                EPOCHS=8, LOGIT_SCALE=30.0, STABILITY_WEIGHT=0.0
            ),
            OPTIM=SimpleNamespace(
                LR_INCREMENTAL_EXPERT=0.04,
                LR_INCREMENTAL_ROUTER=0.005,
                WEIGHT_DECAY=0.0,
            ),
            PROTECTION=SimpleNamespace(
                GATE_WEIGHT=0.1, OLD_TASK_RECALL_TOLERANCE=0.02
            ),
            CHECKPOINT=SimpleNamespace(
                ENABLE=True, DIR=str(checkpoint_dir), RESUME=resume,
                SAVE_EVERY_SESSION=True,
            ),
        )),
    )


def make_runner(checkpoint_dir, resume=""):
    cfg = make_cfg(checkpoint_dir, resume)
    data = SyntheticDataManager()
    model = TinyVersionedModel(blocks=2)
    model._test_cfg = cfg
    runner = SyntheticAPCRunner.__new__(SyntheticAPCRunner)
    runner.cfg = cfg
    runner.data_manager = data
    runner.device = "cpu"
    runner.model = runner.model_without_dp = model
    runner.acc_list = []
    runner.task_acc_list = []
    runner.session_diagnostics = []
    runner.task_gate_biases = []
    runner.apc_state = None
    runner._resume_completed_session = -1
    runner._resume_state_dict_list = []
    runner.evaluator = AccuracyEvaluator(data.class_index_in_task)
    runner.apc_controller = AlwaysCandidateAPC(
        alpha=cfg.TRAINER.BiMC.APC.ALPHA,
        calibration_trials=cfg.TRAINER.BiMC.APC.CALIBRATION_TRIALS,
        seed=cfg.TRAINER.BiMC.APC.SEED,
        logit_scale=cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE,
    )
    return runner


class APCRunnerMultiSessionIntegrationTests(unittest.TestCase):
    def test_real_apc_runner_two_incremental_sessions_and_checkpoint_resume(self):
        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(directory)
            runner.run()

            self.assertEqual(runner._resume_completed_session, 2)
            self.assertEqual(len(runner._resume_state_dict_list), 3)
            self.assertEqual(len(runner.acc_list), 3)
            self.assertEqual(len(runner.task_acc_list), 3)
            self.assertEqual(len(runner.session_diagnostics), 3)
            self.assertEqual(runner.task_gate_biases[0], 0.0)
            self.assertTrue(all(np.isfinite(value) for value in runner.acc_list))
            self.assertTrue(all(np.isfinite(value) for value in runner.task_gate_biases))

            versions = [
                version
                for _, adapter in runner._moe_adapters()
                for version in adapter.expert_birth_versions[adapter.base_expert_count:]
            ]
            self.assertTrue(versions, "The real APCRunner path did not commit an expert.")
            self.assertEqual(sorted(versions), sorted(set(versions)))
            self.assertTrue(any(
                item["adaptation"] and item["adaptation"]["expanded"]
                for item in runner.session_diagnostics[1:]
            ))
            for _, adapter in runner._moe_adapters():
                self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

            checkpoint = Path(directory) / "session_02.pth"
            self.assertTrue(checkpoint.is_file())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], 11)
            self.assertEqual(payload["completed_session"], 2)
            self.assertEqual(len(payload["task_states"]), 3)
            self.assertEqual(len(payload["task_gate_biases"]), 3)
            self.assertEqual(payload["apc_state"]["version"], 3)

            resumed = make_runner(directory, resume=str(checkpoint))
            resumed._load_resume_checkpoint_if_requested()
            self.assertEqual(resumed._resume_completed_session, 2)
            self.assertEqual(resumed.task_gate_biases, runner.task_gate_biases)
            self.assertEqual(resumed.acc_list, runner.acc_list)
            self.assertEqual(resumed.task_acc_list, runner.task_acc_list)
            self.assertEqual(len(resumed._resume_state_dict_list), 3)
            self.assertEqual(resumed.apc_state["version"], runner.apc_state["version"])

            for (block_a, adapter_a), (block_b, adapter_b) in zip(
                runner._moe_adapters(), resumed._moe_adapters()
            ):
                self.assertEqual(block_a, block_b)
                self.assertEqual(adapter_a.export_topology(), adapter_b.export_topology())

            merged_a = runner.merge_dicts(runner._resume_state_dict_list)
            merged_b = resumed.merge_dicts(resumed._resume_state_dict_list)
            for key in ("class_ids", "encoder_versions", "task_ids"):
                torch.testing.assert_close(merged_a[key], merged_b[key], rtol=0, atol=0)
            for key in ("image_proto", "text_features", "description_proto",
                        "fused_proto", "anchor_fused_proto"):
                torch.testing.assert_close(merged_a[key], merged_b[key], rtol=0, atol=0)

            probe = torch.stack([class_center(index) for index in range(6)])
            logits_a = runner._versioned_scores(
                probe, merged_a["fused_proto"], merged_a["encoder_versions"], 30.0
            )
            logits_b = resumed._versioned_scores(
                probe, merged_b["fused_proto"], merged_b["encoder_versions"], 30.0
            )
            torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
            anchor_a = F.normalize(runner.model_without_dp.extract_img_feature(
                probe, max_session=0), dim=-1)
            anchor_b = F.normalize(resumed.model_without_dp.extract_img_feature(
                probe, max_session=0), dim=-1)
            retained_a = runner._retained_scores(anchor_a, merged_a, logits_a)
            retained_b = resumed._retained_scores(anchor_b, merged_b, logits_b)
            torch.testing.assert_close(retained_a, retained_b, rtol=0, atol=0)

            for left, right in zip(
                runner.model_without_dp.state_dict().values(),
                resumed.model_without_dp.state_dict().values()
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

            for left, right in zip(runner.session_diagnostics, resumed.session_diagnostics):
                self.assertEqual(left["session"], right["session"])
                self.assertEqual(left["expert_counts"], right["expert_counts"])
                self.assertEqual(left["classifier"]["AA"], right["classifier"]["AA"])
                self.assertEqual(left["classifier"]["PD"], right["classifier"]["PD"])

    def test_versioned_global_classifier_preserves_old_columns_after_two_expansions(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(directory)
            model = runner.model_without_dp
            # Simulate two committed topology versions in distinct Transformer
            # blocks, then a third version in the first block.
            model.add_expert(0, session_id=1, version_id=1)
            model.add_expert(1, session_id=1, version_id=2)
            model.add_expert(0, session_id=2, version_id=3)
            for _, adapter in runner._moe_adapters():
                adapter.freeze_all()

            probe = torch.stack([class_center(index) for index in range(6)])
            old_prototype = F.normalize(class_center(0), dim=0).unsqueeze(0)
            before = runner._versioned_scores(
                probe, old_prototype, torch.tensor([0]), 30.0
            )
            with torch.no_grad():
                for _, adapter in runner._moe_adapters():
                    for expert in adapter.experts[adapter.base_expert_count:]:
                        expert.c_proj.bias.add_(torch.randn_like(expert.c_proj.bias) * 0.1)
            after = runner._versioned_scores(
                probe, old_prototype, torch.tensor([0]), 30.0
            )
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(runner._current_encoder_version(), 3)


if __name__ == "__main__":
    unittest.main()
