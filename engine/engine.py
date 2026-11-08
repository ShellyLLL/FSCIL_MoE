"""Single-classifier FSCIL training and APC-guided dynamic MoE expansion."""
import copy
import json
import math
import os
import random
import time
from statistics import NormalDist
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets.data_manager import DatasetManager
from models.bimc import BiMC
from utils.evaluator import AccuracyEvaluator


def _get(node, name, default):
    return getattr(node, name, default) if node is not None else default


class HistoricalFeatureConstraintError(RuntimeError):
    """Raised when a candidate encoder regresses on previously learned support data."""


class Runner:
    """All sessions share one current encoder and one all-seen BiMC cosine classifier."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE.DEVICE_NAME
        self.data_manager = DatasetManager(cfg)
        self.model = BiMC(cfg, self.data_manager.template, self.device)
        self.acc_list, self.task_acc_list, self.session_diagnostics = [], [], []
        self.states = []
        self.apc_state = None
        self.start_session = 0
        self._load_checkpoint()

    def parse_batch(self, batch):
        return self.model.parse_batch(batch)

    def _adapters(self):
        return self.model.adapters()

    @staticmethod
    def _source(batch):
        value = batch.get("source_ids", batch.get("source_id", batch.get("index")))
        if value is None:
            raise KeyError("FSCIL training and validation require stable source IDs.")
        return torch.as_tensor(value).long().reshape(-1)

    @staticmethod
    def _compute_fscil_metrics(session_accuracies):
        """Compute the BiMC FSCIL metrics from cumulative session accuracies.

        A_t is the all-seen-class test accuracy after session t.
        AA_t = mean(A_0, ..., A_t), and PD_t = A_0 - A_t.
        PD is signed: a negative value means the final accuracy exceeds base.
        """
        values = np.asarray(session_accuracies, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("FSCIL metrics require a non-empty finite session-accuracy history.")
        return {
            "aa": float(values.mean()),
            "pd": float(values[0] - values[-1]),
            "base_acc": float(values[0]),
            "last_acc": float(values[-1]),
        }

    @staticmethod
    def _merge_states(states, device):
        rows = []
        for state in states:
            for i, class_id in enumerate(torch.as_tensor(state["class_ids"]).tolist()):
                rows.append((int(class_id), state, i))
        rows.sort(key=lambda row: row[0])
        ordered_ids = [r[0] for r in rows]
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("Global prototype bank contains duplicate class IDs.")
        ids = torch.tensor(ordered_ids, dtype=torch.long, device=device)
        for state_key in ("image_proto", "text_features", "description_proto", "fused_proto"):
            values = [torch.as_tensor(state[state_key])[i] for _, state, i in rows]
            if not values:
                raise ValueError("Cannot build an empty global classifier.")
            # Every prototype must share the current global embedding dimension.
            if len({tuple(v.shape) for v in values}) != 1:
                raise ValueError(f"Global prototype field {state_key} has inconsistent dimensions.")
            rows_value = torch.stack(values).to(device)
            if not torch.isfinite(rows_value).all():
                raise FloatingPointError(f"Non-finite values in global {state_key}.")
            if state_key == "fused_proto":
                fused = rows_value
            elif state_key == "image_proto":
                image = rows_value
            elif state_key == "text_features":
                text = rows_value
            else:
                desc = rows_value
        return {
            "class_ids": ids,
            "image_proto": image,
            "text_features": text,
            "description_proto": desc,
            "fused_proto": F.normalize(fused.float(), dim=-1),
        }

    def _columns(self, labels, class_ids):
        ids = torch.as_tensor(class_ids, device=labels.device, dtype=torch.long)
        match = labels.reshape(-1, 1).eq(ids.reshape(1, -1))
        if not match.sum(1).eq(1).all():
            raise ValueError("Each target must occur exactly once in the global class bank.")
        return match.long().argmax(1)

    def _support_folds(self, dataset, n_splits=None):
        labels = np.asarray(dataset.labels)
        sources = np.asarray(dataset.source_ids)
        if labels.shape != sources.shape:
            raise ValueError("Support labels and source IDs do not align.")
        by_class = {int(c): np.unique(sources[labels == c]) for c in np.unique(labels)}
        n = min(map(len, by_class.values()))
        if n < 3:
            return []
        folds = []
        for k in range(n):
            held = {c: s[k] for c, s in by_class.items()}
            val_idx = [i for i, (y, sid) in enumerate(zip(labels, sources)) if held[int(y)] == sid]
            fit_idx = sorted(set(range(len(labels))) - set(val_idx))
            if not val_idx or not fit_idx:
                raise RuntimeError("Empty source-exclusive CV fold.")
            folds.append((fit_idx, val_idx))
        return folds

    def _assert_aligned_support_datasets(self, train_dataset, extract_dataset):
        """Require optimizer and deterministic extraction views to index identical sources."""
        train_labels = np.asarray(train_dataset.labels)
        extract_labels = np.asarray(extract_dataset.labels)
        train_sources = np.asarray(train_dataset.source_ids)
        extract_sources = np.asarray(extract_dataset.source_ids)
        if (train_labels.shape != extract_labels.shape
                or train_sources.shape != extract_sources.shape
                or not np.array_equal(train_labels, extract_labels)
                or not np.array_equal(train_sources, extract_sources)):
            raise RuntimeError(
                "APC fitting and extraction datasets do not have identical label/source ordering."
            )
        if len(np.unique(train_sources)) != len(train_sources):
            raise RuntimeError("A session support set contains duplicate source IDs.")

    def _trainable_parameter_ids(self):
        return {id(parameter) for parameter in self.model.parameters() if parameter.requires_grad}

    def _assert_trainable_scope(self, expected_parameters, context):
        expected = {id(parameter) for parameter in expected_parameters}
        actual = self._trainable_parameter_ids()
        if actual != expected:
            names = {
                name: ("expected" if id(parameter) in expected else "unexpected")
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            raise RuntimeError(
                f"{context}: requires_grad scope differs from the architecture contract; "
                f"unexpected_trainables={names}"
            )

    def _assert_optimizer_scope(self, optimizer, expected_parameters, context):
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        ids = [id(parameter) for parameter in parameters]
        expected = {id(parameter) for parameter in expected_parameters}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError(
                f"{context}: optimizer parameters must contain each and only each "
                "architecture-authorized trainable parameter exactly once."
            )

    def _snapshot_frozen_parameters(self, expected_parameters):
        trainable = {id(parameter) for parameter in expected_parameters}
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if id(parameter) not in trainable
        }

    def _assert_gradient_scope(self, expected_parameters, context):
        trainable = {id(parameter) for parameter in expected_parameters}
        unexpected, non_finite = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and id(parameter) not in trainable:
                unexpected.append(name)
            elif parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                non_finite.append(name)
        if unexpected or non_finite:
            raise RuntimeError(
                f"{context}: gradient scope violation; unexpected={unexpected}, "
                f"non_finite={non_finite}"
            )

    def _assert_frozen_unchanged(self, snapshot, context):
        changed = []
        current = dict(self.model.named_parameters())
        for name, before in snapshot.items():
            parameter = current.get(name)
            if parameter is None or not torch.equal(parameter.detach().cpu(), before):
                changed.append(name)
        if changed:
            raise RuntimeError(f"{context}: frozen parameters changed: {changed}")

    def _assert_all_parameters_frozen(self, context):
        trainable = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"{context}: parameters remain trainable after the phase: {trainable}")

    def _restore_candidate_snapshot(self, topology, state):
        self.model.rebuild_moe_topology(topology)
        self.model.load_state_dict(state, strict=True)
        for _, adapter in self._adapters():
            adapter.freeze_all()
        self._assert_all_parameters_frozen("APC candidate rollback")
        current = self.model.state_dict()
        changed = [
            key for key, value in state.items()
            if key not in current or not torch.equal(current[key].detach().cpu(), value.detach().cpu())
        ]
        if changed:
            raise RuntimeError(f"APC rollback did not restore the exact model state: {changed}")

    def _historical_feature_targets(self, old_states):
        targets = {}
        for state in old_states:
            required = (
                "support_features", "support_labels", "support_source_ids",
                "feature_anchor_session",
            )
            if any(key not in state for key in required):
                raise RuntimeError(
                    "Historical feature targets are missing. Restart from Session 0 "
                    "with checkpoint schema 24."
                )
            if int(state["feature_anchor_session"]) != int(state.get("creation_session", -1)):
                raise RuntimeError(
                    "Historical feature anchor session metadata is inconsistent; "
                    "restart from Session 0 with checkpoint schema 24."
                )
            features = torch.as_tensor(state["support_features"]).float().cpu()
            labels = torch.as_tensor(state["support_labels"]).long().cpu().reshape(-1)
            sources = torch.as_tensor(state["support_source_ids"]).long().cpu().reshape(-1)
            if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(sources):
                raise ValueError("Historical support features, labels, and source IDs do not align.")
            if not torch.isfinite(features).all():
                raise FloatingPointError("Historical support feature targets contain non-finite values.")
            for feature, label, source in zip(features, labels, sources):
                source_id = int(source)
                if source_id in targets:
                    raise ValueError(f"Historical support source {source_id} appears more than once.")
                targets[source_id] = (int(label), F.normalize(feature.float(), dim=-1))
        if not targets:
            raise ValueError("Historical support feature target bank is empty.")
        return targets

    def _historical_support_loader(self, task_id, old_states):
        if int(task_id) <= 0 or not old_states:
            raise ValueError("Historical support preservation is only defined after Session 0.")
        targets = self._historical_feature_targets(old_states)
        loader = self.data_manager.get_dataloader(
            int(task_id) - 1, source="train", mode="test", accumulate_past=True
        )
        dataset_sources = np.asarray(loader.dataset.source_ids, dtype=np.int64)
        dataset_labels = np.asarray(loader.dataset.labels, dtype=np.int64)
        if len(dataset_sources) != len(targets) or set(dataset_sources.tolist()) != set(targets):
            raise RuntimeError(
                "Checkpointed historical support sources do not match the current support manifest."
            )
        for source_id, label in zip(dataset_sources, dataset_labels):
            if targets[int(source_id)][0] != int(label):
                raise RuntimeError(
                    f"Historical support label mismatch for source {int(source_id)}."
                )
        return loader, targets

    def _historical_target_batch(self, batch, targets):
        source_ids = self._source(batch).tolist()
        labels = torch.as_tensor(batch["label"]).long().reshape(-1).tolist()
        rows = []
        for source_id, label in zip(source_ids, labels):
            item = targets.get(int(source_id))
            if item is None:
                raise KeyError(f"No historical feature target for source {int(source_id)}.")
            target_label, feature = item
            if int(target_label) != int(label):
                raise ValueError(f"Historical target label mismatch for source {int(source_id)}.")
            rows.append(feature)
        return torch.stack(rows).to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def _evaluate_old_feature_preservation(self, loader, targets, old_bank):
        self.model.eval()
        prototypes = F.normalize(old_bank["fused_proto"].float(), dim=-1)
        class_ids = old_bank["class_ids"].to(self.device)
        cosine_values, drift_values = [], []
        baseline_correct = candidate_correct = lost_correct = count = 0
        for batch in loader:
            images, labels = self.parse_batch(batch)
            target_features = F.normalize(self._historical_target_batch(batch, targets), dim=-1)
            current_features = F.normalize(
                self.model.extract_img_feature(images).float(), dim=-1
            )
            cosine = F.cosine_similarity(current_features, target_features, dim=-1)
            drift = (current_features - target_features).norm(dim=-1)
            target_columns = self._columns(labels, class_ids)
            baseline_prediction = (target_features @ prototypes.t()).argmax(dim=-1)
            current_prediction = (current_features @ prototypes.t()).argmax(dim=-1)
            baseline_ok = baseline_prediction.eq(target_columns)
            current_ok = current_prediction.eq(target_columns)
            cosine_values.extend(cosine.cpu().tolist())
            drift_values.extend(drift.cpu().tolist())
            baseline_correct += int(baseline_ok.sum().item())
            candidate_correct += int(current_ok.sum().item())
            lost_correct += int((baseline_ok & ~current_ok).sum().item())
            count += int(labels.numel())
        if count == 0:
            raise RuntimeError("Historical support preservation audit received no samples.")
        return {
            "count": count,
            "mean_cosine_similarity": float(np.mean(cosine_values)),
            "min_cosine_similarity": float(np.min(cosine_values)),
            "mean_l2_drift": float(np.mean(drift_values)),
            "max_l2_drift": float(np.max(drift_values)),
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "baseline_accuracy": float(baseline_correct / count),
            "candidate_accuracy": float(candidate_correct / count),
            "lost_previously_correct": lost_correct,
        }

    def _loader(self, dataset, indices, batch_size, shuffle):
        return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.cfg.DATALOADER.NUM_WORKERS, drop_last=False)

    @torch.no_grad()
    def _extract_support(self, loader):
        feats, labels, sources = [], [], []
        self.model.eval()
        for batch in loader:
            images, y = self.parse_batch(batch)
            feats.append(F.normalize(self.model.extract_img_feature(images).float(), dim=-1))
            labels.append(y)
            sources.append(self._source(batch).to(self.device))
        return torch.cat(feats), torch.cat(labels), torch.cat(sources)

    @torch.no_grad()
    def _class_prototypes(self, features, labels, class_ids, exclude_sources=None, query_sources=None):
        class_ids = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
        sources = None if exclude_sources is None else torch.as_tensor(
            exclude_sources, device=features.device
        ).reshape(-1)
        if query_sources is None:
            result = []
            for cls in class_ids:
                selected = features[labels.eq(cls)]
                if selected.numel() == 0:
                    raise ValueError(f"No features for class {int(cls)}.")
                result.append(BiMC.mean_normalized_features(selected, dim=0))
            return torch.stack(result)
        banks = []
        for sid, query_label in zip(query_sources, exclude_sources):
            prototypes = []
            for cls in class_ids:
                mask = labels.eq(cls)
                if int(cls) == int(query_label):
                    mask = mask & sources.ne(sid)
                selected = features[mask]
                if selected.numel() == 0:
                    raise ValueError("A query has no source-exclusive support prototype.")
                prototypes.append(BiMC.mean_normalized_features(selected, dim=0))
            banks.append(torch.stack(prototypes))
        return torch.stack(banks)

    def _text(self, class_ids):
        names = np.asarray(self.data_manager.class_names)[np.asarray(class_ids, dtype=np.int64)].tolist()
        text = self.model.inference_text_feature(names, self.data_manager.template)
        desc_list, desc = self.model.inference_all_description_feature(names, self.cfg.DATASET.GPT_PATH)
        return text, desc, desc_list

    @staticmethod
    def _loo_class_prototypes(features, labels, source_ids, class_ids,
                              query_labels, query_source_ids):
        """Build per-query prototypes with the query's source removed.

        Inputs are L2-normalized image embeddings. This is used in both base
        and incremental optimization so no query can appear in its own class
        prototype, even when the training augmentation differs from extraction.
        """
        class_ids = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
        query_labels = torch.as_tensor(query_labels, device=features.device, dtype=torch.long)
        query_source_ids = torch.as_tensor(query_source_ids, device=features.device).reshape(-1)
        source_ids = torch.as_tensor(source_ids, device=features.device).reshape(-1)
        sums, counts = [], []
        for cls in class_ids:
            mask = labels.eq(cls)
            selected = features[mask]
            if selected.numel() == 0:
                raise ValueError(f"No support features for class {int(cls)}.")
            sums.append(selected.sum(0))
            counts.append(int(mask.sum().item()))
        sums = torch.stack(sums)
        counts = torch.tensor(counts, device=features.device, dtype=features.dtype)
        rows = []
        for label, source_id in zip(query_labels, query_source_ids):
            row = []
            for index, cls in enumerate(class_ids):
                total = sums[index]
                count = counts[index]
                if int(label) == int(cls):
                    own = labels.eq(cls) & source_ids.eq(source_id)
                    if int(own.sum().item()) != 1 or count.item() <= 1:
                        raise ValueError("LOO prototype requires exactly one query source and another support.")
                    total = total - features[own][0]
                    count = count - 1
                # Features are normalized sample embeddings. The LOO
                # prototype is their arithmetic mean, not a normalized mean.
                row.append(total / count)
            rows.append(torch.stack(row))
        return torch.stack(rows)

    def _fuse(self, visual, text, desc):
        return self.model.fuse_modalities(visual, text, desc)

    def _build_state(self, class_ids, loader, old_states, calibrate_visual):
        feat, labels, sources = self._extract_support(loader)
        visual = self._class_prototypes(feat, labels, class_ids)
        if old_states and calibrate_visual:
            visual = self.model.calibrate_incremental_visual(visual)
        names = np.asarray(self.data_manager.class_names)[np.asarray(class_ids, dtype=np.int64)].tolist()
        text = self.model.inference_text_feature(names, self.data_manager.template)
        desc_list, desc = self.model.inference_all_description_feature(names, self.cfg.DATASET.GPT_PATH)
        fused = self._fuse(visual, text, desc)
        raw_layers = self.model.collect_blockwise_cls_features(loader)
        raw_prototypes, adapter_input_anchors, raw_reference_anchors = {}, {}, {}
        for block, item in raw_layers.items():
            vals, ys = item["raw_reference_cls"], item["labels"]
            adapter_input_anchors[int(block)] = item["adapter_input_cls"].float().cpu()
            raw_reference_anchors[int(block)] = vals.float().cpu()
            rows = []
            for cls in class_ids:
                v = vals[ys.eq(int(cls))]
                if not len(v):
                    raise ValueError(f"Missing raw reference features for class {cls}, block {block}.")
                rows.append(BiMC.mean_normalized_features(v.float(), dim=0))
            raw_prototypes[int(block)] = torch.stack(rows).cpu()
        return {
            "class_ids": torch.as_tensor(class_ids, dtype=torch.long),
            "image_proto": visual.detach().cpu(),
            "text_features": text.detach().cpu(),
            "description_proto": desc.detach().cpu(),
            "description_features": [x.detach().cpu() for x in desc_list],
            "fused_proto": fused.detach().cpu(),
            "support_features": feat.detach().cpu(),
            "support_labels": labels.detach().cpu(),
            "support_source_ids": sources.detach().cpu(),
            "feature_anchor_session": len(old_states),
            "current_support_features": feat.detach().cpu(),
            "raw_layer_prototypes": raw_prototypes,
            # Historical adapter-input CLS anchors permit auditing the actual
            # residual emitted by a newly inserted expert without retaining images.
            "adapter_input_anchors": adapter_input_anchors,
            "raw_reference_anchors": raw_reference_anchors,
            "creation_session": len(old_states),
        }

    def _raw_history(self, states, block):
        ids, protos = [], []
        for state in states:
            raw = state.get("raw_layer_prototypes", {}).get(int(block))
            if raw is None:
                raise RuntimeError(f"Historical raw prototype missing for MoE block {block}.")
            ids.extend(torch.as_tensor(state["class_ids"]).tolist())
            protos.extend(torch.as_tensor(raw).unbind(0))
        return torch.tensor(ids, dtype=torch.long), torch.stack(protos).to(self.device)

    def _raw_reference_anchor_history(self, states, block):
        values = []
        for state in states:
            anchors = state.get("raw_reference_anchors", {}).get(int(block))
            if anchors is None or anchors.numel() == 0:
                raise RuntimeError(
                    f"Historical raw-reference anchors missing for MoE block {block}; "
                    "the checkpoint cannot certify actual expert residual protection."
                )
            values.append(torch.as_tensor(anchors).float())
        return torch.cat(values, dim=0).to(self.device)

    def _adapter_input_history(self, states, block):
        values = []
        for state in states:
            anchors = state.get("adapter_input_anchors", {}).get(int(block))
            if anchors is None or anchors.numel() == 0:
                raise RuntimeError(
                    f"Historical adapter-input anchors missing for MoE block {block}; "
                    "the checkpoint cannot certify actual expert residual protection."
                )
            values.append(torch.as_tensor(anchors).float())
        return torch.cat(values, dim=0).to(self.device)

    @staticmethod
    def _update_current_feature_cache(state, features, labels, sources, indices):
        """Refresh diagnostics/cache without mutating immutable class-birth targets."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=features.device)
        current_features = features.index_select(0, indices).detach().cpu()
        current_labels = labels.index_select(0, indices).detach().cpu()
        current_sources = sources.index_select(0, indices).detach().cpu()
        birth_labels = torch.as_tensor(state["support_labels"]).long().cpu()
        birth_sources = torch.as_tensor(state["support_source_ids"]).long().cpu()
        if not torch.equal(current_labels, birth_labels):
            raise RuntimeError("Historical support labels changed after their feature-anchor session.")
        if not torch.equal(current_sources, birth_sources):
            raise RuntimeError("Historical support source ordering changed after feature anchoring.")
        # support_features remains immutable and is the teacher target for all future sessions.
        state["current_support_features"] = current_features

    @torch.no_grad()
    def _refresh_historical_support_features(self, task_id):
        """Refresh every stored support target/anchor in the current encoder space."""
        loader = self.data_manager.get_dataloader(
            int(task_id), source="train", mode="test", accumulate_past=True
        )
        features, labels, sources = self._extract_support(loader)
        source_ids = sources.detach().cpu().long().tolist()
        if len(set(source_ids)) != len(source_ids):
            raise RuntimeError("Accumulated historical support loader contains duplicate source IDs.")
        index_by_source = {int(source): index for index, source in enumerate(source_ids)}
        all_state_sources = [
            int(source)
            for state in self.states
            for source in torch.as_tensor(state["support_source_ids"]).tolist()
        ]
        if len(all_state_sources) != len(set(all_state_sources)) or set(all_state_sources) != set(source_ids):
            raise RuntimeError(
                "Stored per-session support sources do not exactly match the accumulated support manifest."
            )
        for state in self.states:
            ordered_sources = torch.as_tensor(state["support_source_ids"]).long().tolist()
            indices = torch.tensor(
                [index_by_source[int(source)] for source in ordered_sources],
                device=features.device, dtype=torch.long,
            )
            self._update_current_feature_cache(
                state, features, labels, sources, indices
            )

        raw_layers = self.model.collect_blockwise_cls_features(loader)
        for block, item in raw_layers.items():
            layer_sources = torch.as_tensor(item["source_ids"]).long().tolist()
            if len(set(layer_sources)) != len(layer_sources) or set(layer_sources) != set(source_ids):
                raise RuntimeError(f"Block {block} historical anchor sources are incomplete or duplicated.")
            layer_index = {int(source): index for index, source in enumerate(layer_sources)}
            raw_reference = item["raw_reference_cls"].float()
            adapter_input = item["adapter_input_cls"].float()
            layer_labels = item["labels"].long()
            for state in self.states:
                ordered_sources = torch.as_tensor(state["support_source_ids"]).long().tolist()
                indices = torch.tensor([layer_index[int(source)] for source in ordered_sources])
                state["raw_reference_anchors"][int(block)] = raw_reference.index_select(0, indices).cpu()
                state["adapter_input_anchors"][int(block)] = adapter_input.index_select(0, indices).cpu()
                state_labels = torch.as_tensor(state["support_labels"]).long()
                if not torch.equal(layer_labels.index_select(0, indices), state_labels):
                    raise RuntimeError(f"Block {block} historical anchor labels do not align.")
                prototypes = []
                class_ids = torch.as_tensor(state["class_ids"]).long()
                for class_id in class_ids:
                    selected = raw_reference.index_select(0, indices)[state_labels.eq(int(class_id))]
                    if not selected.numel():
                        raise RuntimeError(
                            f"Missing refreshed raw-reference anchors for class {int(class_id)}, block {block}."
                        )
                    prototypes.append(BiMC.mean_normalized_features(selected, dim=0))
                state["raw_layer_prototypes"][int(block)] = torch.stack(prototypes).cpu()

    @staticmethod
    def _empirical_upper_quantile(values, alpha):
        """Return the exact ECDF order statistic used by the APC definition."""
        values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("APC calibration requires finite, non-empty base similarities.")
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("APC alpha must be strictly between zero and one.")
        rank = max(1, min(values.numel(), math.ceil((1.0 - alpha) * values.numel())))
        return torch.kthvalue(values, rank).values

    def _calibrate_apc(self, base_state):
        thresholds, base_statistics = {}, {}
        alpha = float(self.cfg.TRAINER.BiMC.APC.ALPHA)
        for block, item in base_state["raw_layer_prototypes"].items():
            proto = F.normalize(item.float(), dim=-1)
            if proto.ndim != 2 or proto.shape[0] < 2:
                raise ValueError(f"APC block {block} requires at least two base-class prototypes.")
            similarity = proto @ proto.t()
            similarity.fill_diagonal_(-1.0)
            nearest = similarity.max(dim=1).values
            threshold = self._empirical_upper_quantile(nearest, alpha)
            thresholds[int(block)] = float(threshold.item())
            base_statistics[int(block)] = {
                "num_base_classes": int(proto.shape[0]),
                "nearest_similarity_min": float(nearest.min().item()),
                "nearest_similarity_median": float(nearest.median().item()),
                "nearest_similarity_max": float(nearest.max().item()),
                "threshold": float(threshold.item()),
                "threshold_rank": int(math.ceil((1.0 - alpha) * proto.shape[0])),
            }
        if not thresholds:
            raise RuntimeError("APC calibration produced no expandable-layer thresholds.")
        self.apc_state = {
            "thresholds": thresholds,
            "base_statistics": base_statistics,
            "alpha": alpha,
            "schema": 2,
        }

    def _candidate_layers(self, new_state, old_states):
        result = []
        new_ids = torch.as_tensor(new_state["class_ids"]).long().cpu()
        for block, threshold in self.apc_state["thresholds"].items():
            if block not in new_state["raw_layer_prototypes"]:
                raise RuntimeError(f"New-session raw prototypes are missing APC block {block}.")
            new_proto = F.normalize(
                new_state["raw_layer_prototypes"][block].to(self.device).float(), dim=-1
            )
            old_ids, old_proto = self._raw_history(old_states, block)
            if new_proto.ndim != 2 or old_proto.ndim != 2 or new_proto.shape[1] != old_proto.shape[1]:
                raise ValueError(f"APC prototype dimensions do not align at block {block}.")
            similarities = new_proto @ old_proto.t()
            per_new_max, nearest_old_index = similarities.max(dim=1)
            max_new_index = int(per_new_max.argmax().item())
            confusion = float(per_new_max[max_new_index].item())
            nearest_old_id = int(old_ids[nearest_old_index[max_new_index]].item())
            new_class_confusions = [
                {
                    "class_id": int(class_id),
                    "max_similarity": float(score.item()),
                    "nearest_old_class_id": int(old_ids[old_index].item()),
                }
                for class_id, score, old_index in zip(new_ids, per_new_max, nearest_old_index)
            ]
            result.append({
                "block": int(block),
                "confusion": confusion,
                "threshold": float(threshold),
                "margin": float(confusion - float(threshold)),
                "candidate": confusion > float(threshold),
                "most_confused_new_class_id": int(new_ids[max_new_index].item()),
                "nearest_old_class_id": nearest_old_id,
                "new_class_confusions": new_class_confusions,
            })
        return result

    def _candidate_train(self, task_id, selected_blocks, fit_loader, fit_extract_loader,
                         old_states, class_ids, text, desc, old_support_loader,
                         old_feature_targets):
        adapters = dict(self._adapters())
        selected_blocks = sorted(set(int(block) for block in selected_blocks))
        if not selected_blocks or any(block not in adapters for block in selected_blocks):
            raise ValueError("Candidate training requires valid, non-empty expandable block indices.")
        for _, adapter in self._adapters():
            adapter.freeze_all()
        for block in selected_blocks:
            self.model.add_expert(block, task_id)

        expected_params = []
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        params = []
        for block in selected_blocks:
            adapter = adapters[block]
            expert_params = list(adapter.experts[-1].parameters())
            router_params = list(adapter.incremental_routers[-1].parameters())
            expected_params.extend(expert_params)
            expected_params.extend(router_params)
            params += [
                {"params": expert_params, "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                {"params": router_params, "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)},
            ]
        self._assert_trainable_scope(expected_params, "incremental expert training")
        optimizer = optim.AdamW(params, weight_decay=float(optim_cfg.WEIGHT_DECAY))
        self._assert_optimizer_scope(optimizer, expected_params, "incremental expert training")
        frozen_snapshot = self._snapshot_frozen_parameters(expected_params)

        epochs = int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)
        old_bank = self._merge_states(old_states, self.device)
        old_proto = old_bank["fused_proto"]
        lambda_t = float(self.cfg.TRAINER.BiMC.LAMBDA_T) if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        new_text = (1.0-lambda_t)*text.float() + lambda_t*desc.float()
        beta = float(self.cfg.DATASET.BETA)
        logit_scale = self.model.classification_scale()
        history_raw = {
            block: F.normalize(self._raw_reference_anchor_history(old_states, block), dim=-1)
            for block in selected_blocks
        }
        history_inputs = {
            block: self._adapter_input_history(old_states, block)
            for block in selected_blocks
        }
        old_count = len(old_support_loader.dataset)
        if old_count <= 0:
            raise RuntimeError("Historical feature preservation requires a non-empty old support set.")

        logs = []
        for epoch in range(epochs):
            self.model.train()
            total, distillation_total = 0.0, 0.0
            for batch in fit_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._source(batch).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                query_features, _ = self.model.extract_img_feature_train(images)
                query_features = F.normalize(query_features.float(), dim=-1)

                # Query-exclusive prototypes are rebuilt under the current candidate weights.
                current_support, current_labels, current_sources = self._extract_support(fit_extract_loader)
                self.model.train()
                prototype_rows = []
                for query_label, query_source in zip(labels, source_ids):
                    class_rows = []
                    for class_id in class_ids:
                        mask = current_labels.eq(int(class_id))
                        if int(class_id) == int(query_label):
                            mask = mask & current_sources.ne(query_source)
                        chosen = current_support[mask]
                        if not chosen.numel():
                            raise ValueError(
                                "Incremental classification needs another source after excluding the query source."
                            )
                        class_rows.append(BiMC.mean_normalized_features(chosen, dim=0))
                    prototype_rows.append(torch.stack(class_rows))
                visual_loo = torch.stack(prototype_rows)
                if self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
                    visual_loo = self.model.calibrate_incremental_visual(
                        visual_loo.reshape(-1, visual_loo.size(-1))
                    ).reshape_as(visual_loo)
                fused_loo = beta * new_text[None] + (1.0 - beta) * visual_loo
                full_bank = torch.cat(
                    [old_proto[None].expand(len(labels), -1, -1), fused_loo], dim=1
                )
                logits = logit_scale * torch.einsum(
                    "nd,ncd->nc", query_features, F.normalize(full_bank, dim=-1)
                )
                all_ids = torch.cat([
                    old_bank["class_ids"],
                    torch.as_tensor(class_ids, device=self.device),
                ])
                loss = F.cross_entropy(logits, self._columns(labels, all_ids))

                # Local residual suppression is an auxiliary constraint. The
                # authoritative preservation test is the final image embedding.
                protection_terms = []
                for block in selected_blocks:
                    adapter = adapters[block]
                    gates, residual = adapter.newest_residual(
                        history_raw[block], history_inputs[block]
                    )
                    relative_residual = residual.float().norm(dim=-1) / (
                        history_inputs[block].float().norm(dim=-1).clamp_min(1e-12)
                    )
                    protection_terms.extend([
                        gates.square().mean(),
                        relative_residual.square().mean(),
                    ])
                if protection_terms:
                    loss = loss + float(
                        self.cfg.TRAINER.BiMC.PROTECTION.OLD_GATE_WEIGHT
                    ) * torch.stack(protection_terms).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite all-class incremental loss.")
                loss.backward()
                self._assert_gradient_scope(expected_params, "incremental classification backward")
                torch.nn.utils.clip_grad_norm_(expected_params, 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(labels)

            # A complete old-support distillation pass refreshes the actual
            # final image embedding, including all downstream attention blocks.
            # The source IDs resolve targets captured by the previous session.
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            for old_batch in old_support_loader:
                old_images, _ = self.parse_batch(old_batch)
                old_targets = F.normalize(
                    self._historical_target_batch(old_batch, old_feature_targets), dim=-1
                )
                old_current, _ = self.model.extract_img_feature_train(old_images)
                old_current = F.normalize(old_current.float(), dim=-1)
                distillation_loss = (
                    1.0 - F.cosine_similarity(old_current, old_targets, dim=-1)
                ).sum() / old_count
                if not torch.isfinite(distillation_loss):
                    raise FloatingPointError("Non-finite historical feature distillation loss.")
                distillation_loss.backward()
                self._assert_gradient_scope(expected_params, "historical feature distillation backward")
                distillation_total += float(distillation_loss.detach()) * len(old_images)
            torch.nn.utils.clip_grad_norm_(expected_params, 1.0)
            optimizer.step()
            logs.append({
                "epoch": epoch + 1,
                "classification_loss": total / max(1, len(fit_extract_loader.dataset)),
                "old_feature_distillation_loss": distillation_total / old_count,
            })

        for _, adapter in self._adapters():
            adapter.freeze_all()
        self._assert_all_parameters_frozen("incremental expert training completion")
        self._assert_frozen_unchanged(frozen_snapshot, "incremental expert training")

        gate_audit, residual_audit = {}, {}
        with torch.no_grad():
            for block in selected_blocks:
                adapter = adapters[block]
                gates, residual = adapter.newest_residual(
                    history_raw[block], history_inputs[block]
                )
                gates, residual = gates.detach(), residual.detach().float()
                relative_residual = residual.norm(dim=-1) / (
                    history_inputs[block].float().norm(dim=-1).clamp_min(1e-12)
                )
                gate_audit[int(block)] = {
                    "mean": float(gates.mean().item()),
                    "max": float(gates.max().item()),
                }
                residual_audit[int(block)] = {
                    "mean_relative_residual": float(relative_residual.mean().item()),
                    "max_relative_residual": float(relative_residual.max().item()),
                }
        self._last_old_gate_audit = gate_audit
        self._last_old_residual_audit = residual_audit
        old_feature_audit = self._evaluate_old_feature_preservation(
            old_support_loader, old_feature_targets, old_bank
        )
        self._last_old_feature_audit = old_feature_audit
        if old_feature_audit["lost_previously_correct"] > 0:
            raise HistoricalFeatureConstraintError(
                "Candidate expansion changed previously correct historical support predictions: "
                f"{old_feature_audit}"
            )
        return logs

    @torch.no_grad()

    @torch.no_grad()
    def _evaluate_candidate(self, validation_loader, fit_extract_loader, old_states,
                            class_ids, text, desc):
        self.model.eval()
        feat, labels, sources = self._extract_support(fit_extract_loader)
        old_bank = self._merge_states(old_states, self.device)
        visual = self._class_prototypes(feat, labels, class_ids)
        if self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
            visual = self.model.calibrate_incremental_visual(visual)
        new_fused = self._fuse(visual, text, desc)
        all_proto = torch.cat([old_bank["fused_proto"], new_fused], dim=0)
        all_ids = torch.cat([old_bank["class_ids"], torch.as_tensor(class_ids, device=self.device)])
        loss_sum, count = 0.0, 0
        for batch in validation_loader:
            images, y = self.parse_batch(batch)
            q = F.normalize(self.model.extract_img_feature(images).float(), dim=-1)
            logits = self.model.classification_scale() * (
                q @ F.normalize(all_proto, dim=-1).t()
            )
            target = self._columns(y, all_ids)
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
            count += len(y)
        return loss_sum/max(1,count)

    def _select_and_train_expansion(self, task_id, train_loader, extract_loader,
                                    old_states, class_ids, class_names):
        """Validate the fixed APC proposal policy and commit or roll back atomically."""
        self._last_old_gate_audit = {}
        self._last_old_residual_audit = {}
        self._last_old_feature_audit = {}
        apc_cfg = self.cfg.TRAINER.BiMC.APC
        if not bool(_get(apc_cfg, "ENABLE", True)):
            return {
                "expanded": [], "candidate_layers": [],
                "reason": "APC disabled by configuration",
                "validation_status": "disabled", "logs": [],
            }
        if not self.apc_state:
            raise RuntimeError("Base APC thresholds are not initialized.")
        self._assert_aligned_support_datasets(train_loader.dataset, extract_loader.dataset)
        old_support_loader, old_feature_targets = self._historical_support_loader(
            task_id, old_states
        )
        folds = self._support_folds(train_loader.dataset)
        if not folds:
            return {
                "expanded": [], "candidate_layers": [],
                "reason": "Insufficient independent support sources for source-held-out APC validation",
                "validation_status": "insufficient_support_sources",
                "logs": [],
            }

        text, desc, _ = self._text(class_ids)
        fold_candidate_sets, fold_baseline_losses = [], []
        fold_expanded_losses, fold_gains, fold_old_feature_audits = [], [], []
        fold_apc_layer_diagnostics = []
        cv_rng = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        for fit_idx, val_idx in folds:
            random.setstate(cv_rng["python"])
            np.random.set_state(cv_rng["numpy"])
            torch.set_rng_state(cv_rng["torch"])
            if torch.cuda.is_available() and cv_rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(cv_rng["cuda"])
            fit_train = self._loader(train_loader.dataset, fit_idx, train_loader.batch_size, True)
            fit_extract = self._loader(extract_loader.dataset, fit_idx, extract_loader.batch_size, False)
            val_loader = self._loader(extract_loader.dataset, val_idx, extract_loader.batch_size, False)

            fold_state = self._build_state(class_ids, fit_extract, old_states, False)
            fold_layer_diagnostics = self._candidate_layers(fold_state, old_states)
            fold_apc_layer_diagnostics.append(fold_layer_diagnostics)
            fold_candidates = sorted(
                row["block"] for row in fold_layer_diagnostics if row["candidate"]
            )
            fold_candidate_sets.append(fold_candidates)

            topology_before = self.model.export_moe_topology()
            state_before = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            try:
                baseline_loss = self._evaluate_candidate(
                    val_loader, fit_extract, old_states, class_ids, text, desc
                )
                fold_baseline_losses.append(float(baseline_loss))
                if fold_candidates:
                    try:
                        self._candidate_train(
                            task_id, fold_candidates, fit_train, fit_extract,
                            old_states, class_ids, text, desc,
                            old_support_loader, old_feature_targets,
                        )
                        fold_old_feature_audits.append(dict(self._last_old_feature_audit))
                        expanded_loss = self._evaluate_candidate(
                            val_loader, fit_extract, old_states, class_ids, text, desc
                        )
                    except HistoricalFeatureConstraintError:
                        fold_old_feature_audits.append(dict(self._last_old_feature_audit))
                        expanded_loss = float("inf")
                    fold_expanded_losses.append(float(expanded_loss))
                    fold_gains.append(float(baseline_loss - expanded_loss))
                else:
                    # No counterfactual expert was trained; do not fabricate zero risk gains.
                    fold_old_feature_audits.append({})
                    fold_expanded_losses.append(None)
                    fold_gains.append(None)
            finally:
                self._restore_candidate_snapshot(topology_before, state_before)

        # The final proposal is re-derived from all current support sources;
        # validation folds estimate the policy and never select a layer subset.
        full_state = self._build_state(class_ids, extract_loader, old_states, False)
        full_apc_layer_diagnostics = self._candidate_layers(full_state, old_states)
        final_candidates = sorted(
            row["block"] for row in full_apc_layer_diagnostics if row["candidate"]
        )
        # The exact layer set that will be trained must be the layer set whose
        # held-out risk was measured in every fold. Otherwise the CV gains
        # estimate a different architecture and cannot authorize deployment.
        final_candidate_set = set(final_candidates)
        unstable_folds = [
            index for index, candidates in enumerate(fold_candidate_sets)
            if set(candidates) != final_candidate_set
        ]
        if unstable_folds:
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": "APC candidate topology is not stable across source-held-out folds",
                "validation_status": "candidate_set_unstable",
                "unstable_fold_indices": unstable_folds,
                "fold_candidate_layers": fold_candidate_sets,
                "fold_apc_layer_diagnostics": fold_apc_layer_diagnostics,
                "apc_layer_diagnostics": full_apc_layer_diagnostics,
                "final_candidate_layers": final_candidates,
                "risk_validation_performed": False,
                "mean_risk_gain": None,
                "required_risk_gain": None,
                "logs": [],
            }
        if not any(fold_candidate_sets):
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": "APC proposed no layer in any fitting fold",
                "validation_status": "no_fold_candidates",
                "fold_candidate_layers": fold_candidate_sets,
                "fold_baseline_nll": fold_baseline_losses,
                "fold_expanded_nll": fold_expanded_losses,
                "fold_risk_gains": fold_gains,
                "fold_apc_layer_diagnostics": fold_apc_layer_diagnostics,
                "apc_layer_diagnostics": full_apc_layer_diagnostics,
                "final_candidate_layers": final_candidates,
                "risk_validation_performed": False,
                "mean_risk_gain": None,
                "required_risk_gain": None,
                "logs": [],
            }
        gains = np.asarray(fold_gains, dtype=np.float64)
        mean_gain = float(gains.mean()) if len(gains) else 0.0
        if len(gains) > 1:
            sample_variance = float(gains.var(ddof=1))
            test_train_ratio = 1.0 / max(1, len(gains) - 1)
            corrected_se = float(np.sqrt((1.0 / len(gains) + test_train_ratio) * sample_variance))
            from scipy.stats import t as student_t
            significance = float(_get(apc_cfg, "SIGNIFICANCE", 0.05))
            critical = float(student_t.ppf(1.0 - significance, df=len(gains) - 1))
            required_gain = critical * corrected_se
        else:
            corrected_se, critical, required_gain = float("inf"), float("inf"), float("inf")

        diagnostics = {
            "fold_candidate_layers": fold_candidate_sets,
            "fold_apc_layer_diagnostics": fold_apc_layer_diagnostics,
            "apc_layer_diagnostics": full_apc_layer_diagnostics,
            "fold_baseline_nll": fold_baseline_losses,
            "fold_expanded_nll": fold_expanded_losses,
            "fold_risk_gains": fold_gains,
            "fold_old_feature_audits": fold_old_feature_audits,
            "mean_risk_gain": mean_gain,
            "corrected_standard_error": corrected_se,
            "critical_value": critical,
            "required_risk_gain": required_gain,
            "final_candidate_layers": final_candidates,
        }
        if not any(fold_candidate_sets):
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": "APC proposed no layer in any fitting fold",
                "validation_status": "no_fold_candidates",
                **diagnostics, "logs": [],
            }
        if not final_candidates:
            return {
                "expanded": [], "candidate_layers": [],
                "reason": "APC proposed no layer on the full support set",
                "validation_status": "no_full_support_candidates",
                **diagnostics, "logs": [],
            }
        if not np.isfinite(mean_gain) or mean_gain <= required_gain:
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": "Held-out full-class risk improvement did not clear the uncertainty bound",
                "validation_status": "risk_not_verified",
                **diagnostics, "logs": [],
            }

        topology_before = self.model.export_moe_topology()
        state_before = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        try:
            logs = self._candidate_train(
                task_id, final_candidates, train_loader, extract_loader,
                old_states, class_ids, text, desc,
                old_support_loader, old_feature_targets,
            )
        except HistoricalFeatureConstraintError as exc:
            self._restore_candidate_snapshot(topology_before, state_before)
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": f"Historical feature preservation rejected expansion: {exc}",
                "validation_status": "historical_feature_regression",
                "old_feature_audit": dict(self._last_old_feature_audit),
                **diagnostics, "logs": [],
            }
        except Exception:
            self._restore_candidate_snapshot(topology_before, state_before)
            raise
        return {
            "expanded": list(final_candidates),
            "candidate_layers": final_candidates,
            "reason": "Held-out risk and historical feature preservation accepted",
            "validation_status": "accepted",
            "old_feature_audit": dict(self._last_old_feature_audit),
            **diagnostics, "logs": logs,
        }

    def _base_train(self, state, train_loader, extract_loader):
        adapters = self._adapters()
        if not adapters:
            raise RuntimeError("Base MoE training was requested without expandable adapters.")
        expected_params = []
        for _, adapter in adapters:
            adapter.set_base_trainable()
            expected_params.extend(adapter.base_router.parameters())
            for expert in adapter.experts[:adapter.base_expert_count]:
                expected_params.extend(expert.parameters())
        self._assert_trainable_scope(expected_params, "base expert training")
        optimizer = optim.AdamW(
            expected_params,
            lr=float(self.cfg.TRAINER.BiMC.OPTIM.LR_BASE_MOE),
            weight_decay=float(self.cfg.TRAINER.BiMC.OPTIM.WEIGHT_DECAY),
        )
        self._assert_optimizer_scope(optimizer, expected_params, "base expert training")
        frozen_snapshot = self._snapshot_frozen_parameters(expected_params)
        epochs = int(self.cfg.TRAINER.BiMC.OPTIM.BASE_EPOCHS)
        class_ids = state["class_ids"].to(self.device)
        text = state["text_features"].to(self.device)
        desc = state["description_proto"].to(self.device)
        for epoch in range(epochs):
            self.model.eval()
            support_feat, support_labels, support_sources = self._extract_support(extract_loader)
            self.model.train()
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                query_sources = self._source(batch).to(self.device)
                visual_loo = self._loo_class_prototypes(
                    support_feat, support_labels, support_sources,
                    class_ids, labels, query_sources,
                )
                prototypes = self._fuse(visual_loo, text[None], desc[None])
                optimizer.zero_grad(set_to_none=True)
                feature, _ = self.model.extract_img_feature_train(images)
                logits = self.model.classification_scale() * torch.einsum(
                    "bd,bcd->bc",
                    F.normalize(feature.float(), dim=-1),
                    F.normalize(prototypes.float(), dim=-1),
                )
                loss = F.cross_entropy(logits, self._columns(labels, class_ids))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite base-session LOO classification loss.")
                loss.backward()
                self._assert_gradient_scope(expected_params, "base classification backward")
                torch.nn.utils.clip_grad_norm_(expected_params, 1.0)
                optimizer.step()
            print(f"=> Base epoch {epoch+1}/{epochs} complete")
        for _, adapter in self._adapters():
            adapter.freeze_all()
        self._assert_all_parameters_frozen("base expert training completion")
        self._assert_frozen_unchanged(frozen_snapshot, "base expert training")

    @torch.no_grad()

    @torch.no_grad()
    def _evaluate_session(self, task_id, global_state):
        loader = self.data_manager.get_dataloader(task_id, source="test", mode="test")
        prototypes = global_state["fused_proto"].to(self.device)
        ids = global_state["class_ids"].to(self.device)
        logits_all, labels_all = [], []
        self.model.eval()
        for batch in loader:
            images, labels = self.parse_batch(batch)
            feat = F.normalize(self.model.extract_img_feature(images).float(), dim=-1)
            logits_all.append(feat @ prototypes.t())
            labels_all.append(labels)
        logits, labels = torch.cat(logits_all), torch.cat(labels_all)
        return self.evaluator.calc_accuracy(logits, labels, task_id, class_ids=ids)

    def _save_checkpoint(self, completed_session):
        cfg = self.cfg.TRAINER.BiMC.CHECKPOINT
        if not bool(_get(cfg, "ENABLE", True)):
            return
        path = Path(_get(cfg, "DIR", "checkpoints/fscil_moe"))
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 24,
            "completed_session": int(completed_session),
            "model_state": {k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            "moe_topology": self.model.export_moe_topology(),
            "states": self.states,
            "apc_state": self.apc_state,
            "acc_list": self.acc_list,
            "task_acc_list": self.task_acc_list,
            "session_diagnostics": self.session_diagnostics,
            "support_manifest": self.data_manager.get_support_manifest(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        if bool(_get(cfg, "SAVE_EVERY_SESSION", True)):
            target = path / f"session_{completed_session:02d}.pth"
            temp = target.with_suffix(".tmp")
            torch.save(payload, temp)
            os.replace(temp, target)
        latest = path / "latest.pth"
        latest_tmp = latest.with_suffix(".tmp")
        torch.save(payload, latest_tmp)
        os.replace(latest_tmp, latest)

    def _load_checkpoint(self):
        resume = _get(self.cfg.TRAINER.BiMC.CHECKPOINT, "RESUME", "")
        if not resume:
            return
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if int(payload.get("schema_version", -1)) != 24:
            raise ValueError("This unified architecture requires checkpoint schema 24 with immutable class-birth feature anchors; restart from Session 0.")
        self.model.rebuild_moe_topology(payload["moe_topology"])
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.states = payload["states"]
        if not self.states:
            raise ValueError("Unified checkpoint has no base-session prototype state.")
        self._historical_feature_targets(self.states)
        # Restore the exact post-base-training anchor used by every later
        # incremental visual calibration step.
        self.model.set_base_visual_prototype(self.states[0]["image_proto"])
        self.apc_state = payload["apc_state"]
        self.acc_list = payload["acc_list"]
        self.task_acc_list = payload["task_acc_list"]
        self.session_diagnostics = payload.get("session_diagnostics", [])
        # Checkpoints written by earlier revisions stored the same cumulative
        # session accuracies but used a different PD/AA definition. Recompute
        # their derived metrics from the authoritative accuracy history.
        for index, diagnostic in enumerate(self.session_diagnostics):
            if index >= len(self.acc_list):
                break
            historical_metrics = self._compute_fscil_metrics(self.acc_list[:index + 1])
            diagnostic.update({
                "session_acc": float(self.acc_list[index]),
                "aa": historical_metrics["aa"],
                "pd": historical_metrics["pd"],
                "base_session_acc": historical_metrics["base_acc"],
                "final_session_acc": historical_metrics["last_acc"],
                "metric_definition": "AA=mean(A_0..A_t); PD=A_0-A_t",
            })
        self.data_manager.load_support_manifest(payload["support_manifest"])
        self.start_session = int(payload["completed_session"]) + 1
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])

    def run(self):
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)
        end = self.data_manager.num_tasks - 1
        configured = int(self.cfg.TRAINER.BiMC.END_SESSION)
        if configured >= 0:
            end = min(end, configured)
        for task_id in range(self.start_session, end+1):
            # Session 0 does not run incremental expansion, but the common
            # reporting path below still records the (empty) expansion result.
            adaptation = {"expanded": []}
            classes = self.data_manager.class_index_in_task[task_id]
            names = np.asarray(self.data_manager.class_names)[classes].tolist()
            train = self.data_manager.get_dataloader(task_id, "train", mode="train", accumulate_past=False)
            extract = self.data_manager.get_dataloader(task_id, "train", mode="test", accumulate_past=False)
            if task_id == 0:
                state = self._build_state(classes, extract, [], False)
                if self._adapters():
                    self._base_train(state, train, extract)
                    # Rebuild prototypes after base expert training. This is
                    # the final base feature space and therefore the only
                    # valid anchor for all incremental visual calibration.
                    state = self._build_state(classes, extract, [], False)
                self.model.set_base_visual_prototype(state["image_proto"])
                self.states.append(state)
                if self._adapters():
                    self._calibrate_apc(state)
                self._refresh_historical_support_features(task_id)
            else:
                adaptation = self._select_and_train_expansion(
                    task_id, train, extract, self.states, classes, names
                ) if self._adapters() else {"expanded": []}
                state = self._build_state(
                    classes, extract, self.states,
                    bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                )
                self.states.append(state)
                self._refresh_historical_support_features(task_id)
            global_state = self._merge_states(self.states, self.device)
            metrics = self._evaluate_session(task_id, global_state)
            task_acc = metrics["task_acc"]
            # Acc_t is the cumulative all-seen-class test accuracy at this
            # session. AA and PD follow the BiMC FSCIL table definitions,
            # not the task-wise average-forgetting statistic.
            self.acc_list.append(float(metrics["mean_acc"]))
            self.task_acc_list.append(task_acc)
            continual_metrics = self._compute_fscil_metrics(self.acc_list)
            aa = continual_metrics["aa"]
            pd = continual_metrics["pd"]
            diagnostic = {
                "session": int(task_id),
                "candidate_layers": list(adaptation.get("candidate_layers", [])),
                "expanded_layers": list(adaptation.get("expanded", [])),
                "reason": adaptation.get("reason", "base session"),
                "validation_status": adaptation.get("validation_status", "not_applicable"),
                "fold_candidate_layers": adaptation.get("fold_candidate_layers", []),
                "fold_apc_layer_diagnostics": adaptation.get("fold_apc_layer_diagnostics", []),
                "apc_layer_diagnostics": adaptation.get("apc_layer_diagnostics", []),
                "risk_validation_performed": bool(adaptation.get("risk_validation_performed", adaptation.get("mean_risk_gain") is not None)),
                "fold_baseline_nll": adaptation.get("fold_baseline_nll", []),
                "fold_expanded_nll": adaptation.get("fold_expanded_nll", []),
                "fold_risk_gains": adaptation.get("fold_risk_gains", []),
                "mean_risk_gain": adaptation.get("mean_risk_gain"),
                "required_risk_gain": adaptation.get("required_risk_gain"),
                "old_gate_audit": getattr(self, "_last_old_gate_audit", {}),
                "old_residual_audit": getattr(self, "_last_old_residual_audit", {}),
                "old_feature_audit": adaptation.get(
                    "old_feature_audit", getattr(self, "_last_old_feature_audit", {})
                ),
                "expert_topology": self.model.export_moe_topology(),
                "mean_acc": float(metrics["mean_acc"]),
                "session_acc": float(metrics["mean_acc"]),
                "aa": float(aa),
                "base_session_acc": continual_metrics["base_acc"],
                "final_session_acc": continual_metrics["last_acc"],
                "metric_definition": "AA=mean(A_0..A_t); PD=A_0-A_t",
                "macro_task_acc": float(metrics["macro_task_acc"]),
                "task_acc": [float(x) for x in task_acc],
                "pd": float(pd),
            }
            self.session_diagnostics.append(diagnostic)
            self._save_checkpoint(task_id)
            risk_gain = adaptation.get("mean_risk_gain")
            required_gain = adaptation.get("required_risk_gain")
            risk_gain_text = "n/a" if risk_gain is None else f"{float(risk_gain):.5f}"
            required_gain_text = "n/a" if required_gain is None else f"{float(required_gain):.5f}"
            apc_summary = [
                f"{row['block']}:{row['confusion']:.4f}/{row['threshold']:.4f}"
                for row in adaptation.get("apc_layer_diagnostics", [])
            ]
            print(
                f"=> Session {task_id}: Acc={metrics['mean_acc']:.3f}, "
                f"AA={aa:.3f}, PD={pd:.3f}, "
                f"expanded={adaptation.get('expanded', [])}, "
                f"candidates={adaptation.get('candidate_layers', [])}, "
                f"APC(confusion/threshold)={apc_summary}, "
                f"risk_gain={risk_gain_text}, required_gain={required_gain_text}, "
                f"risk_tested={adaptation.get('risk_validation_performed', adaptation.get('mean_risk_gain') is not None)}, "
                f"reason={adaptation.get('reason', 'base session')}"
            )
