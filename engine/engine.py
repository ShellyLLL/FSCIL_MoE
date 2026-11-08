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


class HistoricalPrototypeConstraintError(RuntimeError):
    """Raised when a new expert violates stored historical-prototype route constraints."""


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


    @staticmethod
    def _capture_rng_state():
        return {
            "python": copy.deepcopy(random.getstate()),
            "numpy": copy.deepcopy(np.random.get_state()),
            "torch": torch.get_rng_state().clone(),
            "cuda": (
                [value.clone() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None
            ),
        }

    @staticmethod
    def _restore_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _classification_logits(self, features, prototypes):
        """The single scaled-cosine scorer used by training, risk validation and inference."""
        query = F.normalize(torch.as_tensor(features).float(), dim=-1)
        bank = torch.as_tensor(prototypes, device=query.device).float()
        bank = F.normalize(bank, dim=-1)
        scale = torch.as_tensor(
            self.model.classification_scale(), device=query.device, dtype=query.dtype
        )
        if bank.ndim == 2:
            return scale * (query @ bank.t())
        if bank.ndim == 3:
            if bank.shape[0] != query.shape[0]:
                raise ValueError("Per-query prototype banks must align with query features.")
            return scale * torch.einsum("bd,bcd->bc", query, bank)
        raise ValueError("The classifier prototype bank must have shape [C,D] or [B,C,D].")

    @staticmethod
    def _risk_gain_statistics(gains, significance, test_train_ratio):
        """One-sided corrected-resampled-t lower bound for paired fold risk gains."""
        values = np.asarray(gains, dtype=np.float64).reshape(-1)
        if values.size < 2 or not np.isfinite(values).all():
            raise ValueError("Risk estimation requires at least two finite paired fold gains.")
        significance = float(significance)
        test_train_ratio = float(test_train_ratio)
        if not 0.0 < significance < 1.0 or test_train_ratio < 0.0:
            raise ValueError("Invalid significance level or test/train ratio.")
        from scipy.stats import t as student_t
        mean_gain = float(values.mean())
        variance = float(values.var(ddof=1))
        corrected_se = float(np.sqrt((1.0 / values.size + test_train_ratio) * variance))
        critical = float(student_t.ppf(1.0 - significance, df=values.size - 1))
        required_gain = float(critical * corrected_se)
        return {
            "fold_count": int(values.size),
            "mean_risk_gain": mean_gain,
            "corrected_standard_error": corrected_se,
            "critical_value": critical,
            "required_risk_gain": required_gain,
            "lower_confidence_bound": float(mean_gain - required_gain),
        }

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
        """Snapshot frozen adapter parameters only; the CLIP backbone is immutable by contract."""
        trainable = {id(parameter) for parameter in expected_parameters}
        return {
            f"{block}.{name}": parameter.detach().cpu().clone()
            for block, adapter in self._adapters()
            for name, parameter in adapter.named_parameters()
            if id(parameter) not in trainable
        }

    def _assert_frozen_unchanged(self, snapshot, context):
        current = {
            f"{block}.{name}": parameter
            for block, adapter in self._adapters()
            for name, parameter in adapter.named_parameters()
        }
        changed = [
            name for name, before in snapshot.items()
            if name not in current or not torch.equal(current[name].detach().cpu(), before)
        ]
        if changed:
            raise RuntimeError(f"{context}: frozen adapter parameters changed: {changed}")

    def _assert_all_parameters_frozen(self, context):
        trainable = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"{context}: parameters remain trainable after the phase: {trainable}")

    def _snapshot_adapter_state(self):
        return {
            int(block): {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
            for block, adapter in self._adapters()
        }

    @staticmethod
    def _snapshot_prototype_states(states):
        keys = (
            "class_ids", "image_proto", "text_features",
            "description_proto", "description_features",
            "fused_proto", "raw_layer_prototypes",
        )
        snapshots = []
        for state in states:
            snapshot = {}
            for key in keys:
                if key == "raw_layer_prototypes":
                    snapshot[key] = {
                        int(block): value.detach().cpu().clone()
                        for block, value in state[key].items()
                    }
                elif key == "description_features":
                    snapshot[key] = [
                        value.detach().cpu().clone() for value in state[key]
                    ]
                else:
                    snapshot[key] = torch.as_tensor(state[key]).detach().cpu().clone()
            snapshots.append(snapshot)
        return snapshots

    @staticmethod
    def _assert_historical_prototypes_unchanged(states, snapshots):
        if len(states) != len(snapshots):
            raise RuntimeError("Historical prototype state count changed during an incremental transaction.")
        for index, (state, snapshot) in enumerate(zip(states, snapshots)):
            for key, before in snapshot.items():
                current = state.get(key)
                if key == "raw_layer_prototypes":
                    if current is None or set(map(int, current)) != set(before):
                        raise RuntimeError(f"Historical raw prototype topology changed in state {index}.")
                    changed = [
                        block for block, value in before.items()
                        if not torch.equal(torch.as_tensor(current[block]).detach().cpu(), value)
                    ]
                    if changed:
                        raise RuntimeError(
                            f"Historical raw prototypes changed in state {index}, blocks {changed}."
                        )
                elif key == "description_features":
                    if current is None or len(current) != len(before) or any(
                        not torch.equal(torch.as_tensor(now).detach().cpu(), old)
                        for now, old in zip(current, before)
                    ):
                        raise RuntimeError(f"Historical description features changed in state {index}.")
                elif current is None or not torch.equal(
                    torch.as_tensor(current).detach().cpu(), before
                ):
                    raise RuntimeError(f"Historical prototype field {key} changed in state {index}.")

    @staticmethod
    def _restore_historical_prototypes(states, snapshots):
        for state, snapshot in zip(states, snapshots):
            for key, value in snapshot.items():
                if key == "raw_layer_prototypes":
                    state[key] = {
                        int(block): tensor.detach().cpu().clone()
                        for block, tensor in value.items()
                    }
                elif key == "description_features":
                    state[key] = [tensor.detach().cpu().clone() for tensor in value]
                else:
                    state[key] = value.detach().cpu().clone()

    def _old_route_suppression_loss(self, adapter, old_raw_prototypes):
        gates = adapter.newest_gate(old_raw_prototypes)
        if not torch.isfinite(gates).all():
            raise FloatingPointError("Non-finite new-expert gates on historical raw prototypes.")
        return gates.square().mean(), gates

    def _audit_historical_prototype_gates(self, selected_blocks, old_states):
        if not old_states:
            raise RuntimeError("Historical prototype protection requires at least one old session.")
        protection_cfg = self.cfg.TRAINER.BiMC.PROTECTION
        max_allowed = float(_get(protection_cfg, "MAX_OLD_GATE", 0.05))
        if not 0.0 < max_allowed < 1.0:
            raise ValueError("MAX_OLD_GATE must lie strictly between zero and one.")
        adapters = dict(self._adapters())
        audits = {}
        violations = []
        for block in selected_blocks:
            class_ids, raw_prototypes = self._raw_history(old_states, block)
            raw_prototypes = F.normalize(raw_prototypes.float(), dim=-1)
            _, gates = self._old_route_suppression_loss(adapters[int(block)], raw_prototypes)
            max_index = int(gates.argmax().item())
            maximum = float(gates[max_index].item())
            audits[int(block)] = {
                "count": int(gates.numel()),
                "mean": float(gates.mean().item()),
                "max": maximum,
                "max_class_id": int(class_ids[max_index].item()),
                "max_allowed": max_allowed,
                "per_class": [
                    {"class_id": int(class_id), "gate": float(gate)}
                    for class_id, gate in zip(class_ids.tolist(), gates.detach().cpu().tolist())
                ],
            }
            if maximum > max_allowed:
                violations.append(
                    f"block {int(block)} class {int(class_ids[max_index])}: "
                    f"gate {maximum:.6f} > {max_allowed:.6f}"
                )
        self._last_old_gate_audit = audits
        if violations:
            raise HistoricalPrototypeConstraintError(
                "New expert activation exceeds the historical-prototype protection bound: "
                + "; ".join(violations)
            )
        return audits

    def _restore_candidate_snapshot(self, topology, state):
        self.model.rebuild_moe_topology(topology)
        adapters = dict(self._adapters())
        if set(adapters) != set(state):
            raise RuntimeError("APC rollback did not restore the exact adapter block set.")
        for block, adapter in adapters.items():
            adapter.load_state_dict(state[int(block)], strict=True)
            adapter.freeze_all()
        self._assert_all_parameters_frozen("APC candidate rollback")
        restored = self._snapshot_adapter_state()
        changed = []
        for block, values in state.items():
            for name, value in values.items():
                if name not in restored.get(int(block), {}) or not torch.equal(
                    restored[int(block)][name], value
                ):
                    changed.append(f"{block}.{name}")
        if changed:
            raise RuntimeError(f"APC rollback did not restore exact adapter state: {changed}")

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
        """Build immutable class prototypes and frozen-reference APC prototypes for this session."""
        feat, labels, _ = self._extract_support(loader)
        visual = self._class_prototypes(feat, labels, class_ids)
        if old_states and calibrate_visual:
            visual = self.model.calibrate_incremental_visual(visual)
        names = np.asarray(self.data_manager.class_names)[
            np.asarray(class_ids, dtype=np.int64)
        ].tolist()
        text = self.model.inference_text_feature(names, self.data_manager.template)
        desc_list, desc = self.model.inference_all_description_feature(
            names, self.cfg.DATASET.GPT_PATH
        )
        fused = self._fuse(visual, text, desc)

        raw_layers = self.model.collect_blockwise_cls_features(loader)
        raw_prototypes = {}
        for block, item in raw_layers.items():
            values, layer_labels = item["raw_reference_cls"], item["labels"]
            rows = []
            for class_id in class_ids:
                selected = values[layer_labels.eq(int(class_id))]
                if not len(selected):
                    raise ValueError(
                        f"Missing frozen-reference samples for class {class_id}, block {block}."
                    )
                rows.append(BiMC.mean_normalized_features(selected.float(), dim=0))
            raw_prototypes[int(block)] = torch.stack(rows).detach().cpu()

        return {
            "class_ids": torch.as_tensor(class_ids, dtype=torch.long).detach().cpu(),
            "image_proto": visual.detach().cpu(),
            "text_features": text.detach().cpu(),
            "description_proto": desc.detach().cpu(),
            "description_features": [value.detach().cpu() for value in desc_list],
            "fused_proto": fused.detach().cpu(),
            "raw_layer_prototypes": raw_prototypes,
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


    def _candidate_train(self, task_id, selected_blocks, train_loader, extract_loader,
                         old_states, class_ids, text, desc):
        """Train only new experts against the immutable all-seen prototype bank."""
        if not old_states:
            raise RuntimeError("Incremental expert training requires immutable historical prototypes.")
        adapters = dict(self._adapters())
        for block in selected_blocks:
            if int(block) not in adapters:
                raise ValueError(f"Selected block {block} has no expandable adapter.")
            self.model.add_expert(block, task_id)
        adapters = dict(self._adapters())
        for _, adapter in self._adapters():
            adapter.freeze_all()
        expected_params = []
        for block in selected_blocks:
            adapter = adapters[int(block)]
            adapter.set_newest_trainable()
            expected_params.extend(adapter.newest_parameters())
        self._assert_trainable_scope(expected_params, "incremental expert training")
        optimizer = optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for block in selected_blocks
                        for parameter in adapters[int(block)].experts[-1].parameters()
                    ],
                    "lr": float(self.cfg.TRAINER.BiMC.OPTIM.LR_INCREMENTAL_EXPERT),
                },
                {
                    "params": [
                        parameter
                        for block in selected_blocks
                        for parameter in adapters[int(block)].incremental_routers[-1].parameters()
                    ],
                    "lr": float(self.cfg.TRAINER.BiMC.OPTIM.LR_INCREMENTAL_ROUTER),
                },
            ],
            weight_decay=float(self.cfg.TRAINER.BiMC.OPTIM.WEIGHT_DECAY),
        )
        self._assert_optimizer_scope(optimizer, expected_params, "incremental expert training")
        frozen_snapshot = self._snapshot_frozen_parameters(expected_params)
        historical_snapshot = self._snapshot_prototype_states(old_states)
        old_bank = self._merge_states(old_states, self.device)
        old_proto = old_bank["fused_proto"].to(self.device)
        old_ids = old_bank["class_ids"].to(self.device)
        all_ids = torch.cat([
            old_ids,
            torch.as_tensor(class_ids, device=self.device, dtype=torch.long),
        ])
        history_raw = {}
        for block in selected_blocks:
            _, raw = self._raw_history(old_states, block)
            history_raw[int(block)] = F.normalize(raw.float(), dim=-1).to(self.device)

        epochs = int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)
        gate_weight = float(
            _get(self.cfg.TRAINER.BiMC.PROTECTION, "OLD_GATE_WEIGHT", 0.1)
        )
        logs = []
        for epoch in range(epochs):
            classification_total = 0.0
            suppression_total = 0.0
            sample_count = 0
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._source(batch).to(self.device)
                optimizer.zero_grad(set_to_none=True)

                query_features, _ = self.model.extract_img_feature_train(images)
                query_features = F.normalize(query_features.float(), dim=-1)

                # Recompute support features after each optimizer step so the LOO
                # classifier is expressed in the current encoder feature space.
                current_support, current_labels, current_sources = self._extract_support(
                    extract_loader
                )
                self.model.train()
                visual_loo = self._loo_class_prototypes(
                    current_support, current_labels, current_sources,
                    class_ids, labels, source_ids,
                )
                if self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
                    visual_loo = self.model.calibrate_incremental_visual(
                        visual_loo.reshape(-1, visual_loo.size(-1))
                    ).reshape_as(visual_loo)
                fused_loo = self._fuse(visual_loo, text[None], desc[None])
                full_bank = torch.cat(
                    [old_proto[None].expand(len(labels), -1, -1), fused_loo], dim=1
                )
                logits = self._classification_logits(query_features, full_bank)
                classification_loss = F.cross_entropy(
                    logits, self._columns(labels, all_ids)
                )

                suppression_terms = []
                for block in selected_blocks:
                    suppression_loss, _ = self._old_route_suppression_loss(
                        adapters[int(block)], history_raw[int(block)]
                    )
                    suppression_terms.append(suppression_loss)
                suppression_loss = torch.stack(suppression_terms).mean()
                loss = classification_loss + gate_weight * suppression_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite incremental all-class objective.")
                loss.backward()
                self._assert_gradient_scope(expected_params, "incremental classification backward")
                torch.nn.utils.clip_grad_norm_(expected_params, 1.0)
                optimizer.step()

                batch_count = len(labels)
                classification_total += float(classification_loss.detach()) * batch_count
                suppression_total += float(suppression_loss.detach()) * batch_count
                sample_count += batch_count

            logs.append({
                "epoch": epoch + 1,
                "classification_loss": classification_total / max(1, sample_count),
                "old_route_suppression_loss": suppression_total / max(1, sample_count),
            })

        for _, adapter in self._adapters():
            adapter.freeze_all()
        self._assert_all_parameters_frozen("incremental expert training completion")
        self._assert_frozen_unchanged(frozen_snapshot, "incremental expert training")
        self._assert_historical_prototypes_unchanged(old_states, historical_snapshot)
        self._audit_historical_prototype_gates(selected_blocks, old_states)
        return logs

    @torch.no_grad()
    def _evaluate_candidate(self, validation_loader, fit_extract_loader, old_states,
                            class_ids, text, desc):
        self.model.eval()
        features, labels, _ = self._extract_support(fit_extract_loader)
        old_bank = self._merge_states(old_states, self.device)
        visual = self._class_prototypes(features, labels, class_ids)
        if self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
            visual = self.model.calibrate_incremental_visual(visual)
        new_fused = self._fuse(visual, text, desc)
        all_proto = torch.cat([old_bank["fused_proto"], new_fused], dim=0)
        all_ids = torch.cat([
            old_bank["class_ids"],
            torch.as_tensor(class_ids, device=self.device, dtype=torch.long),
        ])
        loss_sum, count = 0.0, 0
        for batch in validation_loader:
            images, targets = self.parse_batch(batch)
            query = self.model.extract_img_feature(images).float()
            logits = self._classification_logits(query, all_proto)
            target_columns = self._columns(targets, all_ids)
            loss_sum += float(F.cross_entropy(logits, target_columns, reduction="sum"))
            count += len(targets)
        if count == 0:
            raise RuntimeError("Candidate risk validation received no held-out queries.")
        return loss_sum / count


    def _select_and_train_expansion(self, task_id, train_loader, extract_loader,
                                    old_states, class_ids, class_names):
        """Cross-validate the APC expansion policy, then refit its full-support proposal."""
        self._last_old_gate_audit = {}
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
        folds = self._support_folds(train_loader.dataset)
        if not folds:
            return {
                "expanded": [], "candidate_layers": [],
                "reason": "Insufficient independent support sources for source-held-out APC validation",
                "validation_status": "insufficient_support_sources",
                "logs": [],
            }

        text, desc, _ = self._text(class_ids)
        rng_before = self._capture_rng_state()
        committed = False
        fold_candidate_sets = []
        fold_baseline_losses, fold_expanded_losses, fold_gains = [], [], []
        fold_statuses, fold_old_gate_audits = [], []
        fold_apc_layer_diagnostics = []
        test_train_ratios = []

        try:
            for fit_idx, val_idx in folds:
                self._restore_rng_state(rng_before)
                fit_train = self._loader(
                    train_loader.dataset, fit_idx, train_loader.batch_size, True
                )
                fit_extract = self._loader(
                    extract_loader.dataset, fit_idx, extract_loader.batch_size, False
                )
                val_loader = self._loader(
                    extract_loader.dataset, val_idx, extract_loader.batch_size, False
                )
                test_train_ratios.append(len(val_idx) / max(1, len(fit_idx)))

                fold_state = self._build_state(class_ids, fit_extract, old_states, False)
                layer_diagnostics = self._candidate_layers(fold_state, old_states)
                fold_apc_layer_diagnostics.append(layer_diagnostics)
                fold_candidates = sorted(
                    row["block"] for row in layer_diagnostics if row["candidate"]
                )
                fold_candidate_sets.append(fold_candidates)

                topology_before = self.model.export_moe_topology()
                state_before = self._snapshot_adapter_state()
                baseline_loss = self._evaluate_candidate(
                    val_loader, fit_extract, old_states, class_ids, text, desc
                )
                if not np.isfinite(baseline_loss):
                    raise FloatingPointError("Non-finite no-expansion validation risk.")
                fold_baseline_losses.append(float(baseline_loss))

                try:
                    if fold_candidates:
                        try:
                            self._candidate_train(
                                task_id, fold_candidates, fit_train, fit_extract,
                                old_states, class_ids, text, desc,
                            )
                            expanded_loss = self._evaluate_candidate(
                                val_loader, fit_extract, old_states, class_ids, text, desc
                            )
                            if not np.isfinite(expanded_loss):
                                raise FloatingPointError("Non-finite expanded validation risk.")
                            fold_statuses.append("expanded")
                            fold_old_gate_audits.append(copy.deepcopy(self._last_old_gate_audit))
                        except HistoricalPrototypeConstraintError:
                            expanded_loss = baseline_loss
                            fold_statuses.append("historical_prototype_rejected")
                            fold_old_gate_audits.append(copy.deepcopy(self._last_old_gate_audit))
                    else:
                        expanded_loss = baseline_loss
                        fold_statuses.append("no_candidate")
                        fold_old_gate_audits.append({})
                finally:
                    self._restore_candidate_snapshot(topology_before, state_before)

                fold_expanded_losses.append(float(expanded_loss))
                fold_gains.append(float(baseline_loss - expanded_loss))

            # Fold proposals may differ: CV estimates the adaptive APC policy,
            # not one fixed subset selected using held-out support sources.
            self._restore_rng_state(rng_before)
            full_state = self._build_state(class_ids, extract_loader, old_states, False)
            full_diagnostics = self._candidate_layers(full_state, old_states)
            final_candidates = sorted(
                row["block"] for row in full_diagnostics if row["candidate"]
            )
            diagnostics = {
                "fold_candidate_layers": fold_candidate_sets,
                "fold_apc_layer_diagnostics": fold_apc_layer_diagnostics,
                "apc_layer_diagnostics": full_diagnostics,
                "fold_baseline_nll": fold_baseline_losses,
                "fold_expanded_nll": fold_expanded_losses,
                "fold_risk_gains": fold_gains,
                "fold_statuses": fold_statuses,
                "fold_old_gate_audits": fold_old_gate_audits,
                "final_candidate_layers": final_candidates,
                "risk_validation_performed": bool(any(fold_candidate_sets)),
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

            risk = self._risk_gain_statistics(
                fold_gains,
                significance=float(_get(apc_cfg, "SIGNIFICANCE", 0.05)),
                test_train_ratio=float(np.mean(test_train_ratios)),
            )
            diagnostics.update(risk)
            if (
                not np.isfinite(risk["mean_risk_gain"])
                or risk["lower_confidence_bound"] <= 0.0
            ):
                return {
                    "expanded": [], "candidate_layers": final_candidates,
                    "reason": "Held-out all-class classification-risk improvement did not clear the corrected uncertainty bound",
                    "validation_status": "risk_not_verified",
                    **diagnostics, "logs": [],
                }

            self._restore_rng_state(rng_before)
            topology_before = self.model.export_moe_topology()
            state_before = self._snapshot_adapter_state()
            try:
                logs = self._candidate_train(
                    task_id, final_candidates, train_loader, extract_loader,
                    old_states, class_ids, text, desc,
                )
            except HistoricalPrototypeConstraintError as exc:
                old_gate_audit = copy.deepcopy(self._last_old_gate_audit)
                self._restore_candidate_snapshot(topology_before, state_before)
                return {
                    "expanded": [], "candidate_layers": final_candidates,
                    "reason": f"Historical prototype gate constraint rejected expansion: {exc}",
                    "validation_status": "historical_prototype_regression",
                    "old_gate_audit": old_gate_audit,
                    **diagnostics, "logs": [],
                }
            except Exception:
                self._restore_candidate_snapshot(topology_before, state_before)
                raise

            committed = True
            return {
                "expanded": list(final_candidates),
                "candidate_layers": final_candidates,
                "reason": "Corrected held-out risk and historical prototype gate constraints accepted",
                "validation_status": "accepted",
                "old_gate_audit": copy.deepcopy(self._last_old_gate_audit),
                **diagnostics, "logs": logs,
            }
        finally:
            if not committed:
                self._restore_rng_state(rng_before)

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
                logits = self._classification_logits(feature, prototypes)
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
            feat = self.model.extract_img_feature(images).float()
            logits_all.append(self._classification_logits(feat, prototypes))
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
            "schema_version": 25,
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
        if int(payload.get("schema_version", -1)) != 25:
            raise ValueError(
                "This architecture requires checkpoint schema 25 with prototype-only historical "
                "memory and no historical-image replay; restart from Session 0."
            )
        self.model.rebuild_moe_topology(payload["moe_topology"])
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.states = payload["states"]
        if not self.states:
            raise ValueError("Unified checkpoint has no base-session prototype state.")
        # Restore the immutable post-base-training visual calibration anchor.
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

        for task_id in range(self.start_session, end + 1):
            adaptation = {"expanded": []}
            classes = self.data_manager.class_index_in_task[task_id]
            names = np.asarray(self.data_manager.class_names)[classes].tolist()

            transaction = None
            if task_id > 0:
                transaction = {
                    "topology": self.model.export_moe_topology(),
                    "adapter_state": self._snapshot_adapter_state(),
                    "state_count": len(self.states),
                    "historical_prototypes": self._snapshot_prototype_states(self.states),
                    "apc_state": copy.deepcopy(self.apc_state),
                    "acc_list": list(self.acc_list),
                    "task_acc_list": copy.deepcopy(self.task_acc_list),
                    "session_diagnostics": copy.deepcopy(self.session_diagnostics),
                    "rng": self._capture_rng_state(),
                }

            try:
                train = self.data_manager.get_dataloader(
                    task_id, "train", mode="train", accumulate_past=False
                )
                extract = self.data_manager.get_dataloader(
                    task_id, "train", mode="test", accumulate_past=False
                )

                if task_id == 0:
                    state = self._build_state(classes, extract, [], False)
                    if self._adapters():
                        self._base_train(state, train, extract)
                        state = self._build_state(classes, extract, [], False)
                    self.model.set_base_visual_prototype(state["image_proto"])
                    self.states.append(state)
                    if self._adapters():
                        self._calibrate_apc(state)
                else:
                    adaptation = self._select_and_train_expansion(
                        task_id, train, extract, self.states, classes, names
                    ) if self._adapters() else {"expanded": []}
                    state = self._build_state(
                        classes, extract, self.states,
                        bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                    )
                    self.states.append(state)
                    self._assert_historical_prototypes_unchanged(
                        self.states[:-1], transaction["historical_prototypes"]
                    )

                global_state = self._merge_states(self.states, self.device)
                expected_ids = np.sort(np.concatenate(
                    self.data_manager.class_index_in_task[:task_id + 1]
                )).astype(np.int64)
                if not np.array_equal(
                    global_state["class_ids"].detach().cpu().numpy(), expected_ids
                ):
                    raise RuntimeError("Final classifier does not contain exactly all seen classes.")

                metrics = self._evaluate_session(task_id, global_state)
                task_acc = metrics["task_acc"]
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
                    "risk_validation_performed": bool(
                        adaptation.get("risk_validation_performed", False)
                    ),
                    "fold_baseline_nll": adaptation.get("fold_baseline_nll", []),
                    "fold_expanded_nll": adaptation.get("fold_expanded_nll", []),
                    "fold_risk_gains": adaptation.get("fold_risk_gains", []),
                    "fold_statuses": adaptation.get("fold_statuses", []),
                    "mean_risk_gain": adaptation.get("mean_risk_gain"),
                    "required_risk_gain": adaptation.get("required_risk_gain"),
                    "risk_lower_confidence_bound": adaptation.get("lower_confidence_bound"),
                    "fold_old_gate_audits": adaptation.get("fold_old_gate_audits", []),
                    "old_gate_audit": adaptation.get(
                        "old_gate_audit", getattr(self, "_last_old_gate_audit", {})
                    ),
                    "expert_topology": self.model.export_moe_topology(),
                    "mean_acc": float(metrics["mean_acc"]),
                    "session_acc": float(metrics["mean_acc"]),
                    "aa": float(aa),
                    "base_session_acc": continual_metrics["base_acc"],
                    "final_session_acc": continual_metrics["last_acc"],
                    "metric_definition": "AA=mean(A_0..A_t); PD=A_0-A_t",
                    "macro_task_acc": float(metrics["macro_task_acc"]),
                    "task_acc": [float(value) for value in task_acc],
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
                    f"risk_tested={adaptation.get('risk_validation_performed', False)}, "
                    f"reason={adaptation.get('reason', 'base session')}"
                )
            except Exception:
                if transaction is not None:
                    self._restore_candidate_snapshot(
                        transaction["topology"], transaction["adapter_state"]
                    )
                    self.states = self.states[:transaction["state_count"]]
                    self._restore_historical_prototypes(
                        self.states, transaction["historical_prototypes"]
                    )
                    self.apc_state = transaction["apc_state"]
                    self.acc_list = transaction["acc_list"]
                    self.task_acc_list = transaction["task_acc_list"]
                    self.session_diagnostics = transaction["session_diagnostics"]
                    self._restore_rng_state(transaction["rng"])
                raise
