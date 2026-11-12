import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.data_manager import DatasetManager
from models.bimc import BiMC, build_loo_cache_from_embeddings
from utils.evaluator import AccuracyEvaluator


def compute_load_balance_loss(moe_aux_list, num_experts):
    """Base-session load balance over the unified soft router."""
    if not moe_aux_list:
        return torch.tensor(0.0)
    losses = []
    for aux in moe_aux_list:
        weights = aux["route_weights"].float()
        top1 = F.one_hot(weights.argmax(dim=-1), num_classes=num_experts).float()
        losses.append(num_experts * torch.sum(weights.mean(dim=0) * top1.mean(dim=0)))
    return torch.stack(losses).mean()


def _cfg_value(node, name, default):
    return getattr(node, name, default) if node is not None else default


class Runner:
    """FSCIL runner with SEMA-style AE expansion and one global prototype space."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg)
        self.device = cfg.DEVICE.DEVICE_NAME
        self.model = BiMC(cfg, self.data_manager.template, self.device)
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False
        self.model_without_dp = self.model.module if self.is_distributed else self.model
        self.acc_list, self.task_acc_list = [], []
        self.session_diagnostics = []
        self.expansion_thresholds = {}
        self._base_calibration_features = {}
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)
        self._resume_completed_session = -1
        self._resume_state_dict_list = []
        self._load_resume_checkpoint_if_requested()

    # ------------------------------------------------------------------
    # Generic state / model helpers
    # ------------------------------------------------------------------
    def _moe_adapters(self):
        result = []
        for block_idx, block in enumerate(self.model_without_dp.clip_model.visual.transformer.resblocks):
            adapter = getattr(block, "moe_adapter", None)
            if adapter is not None:
                result.append((block_idx, adapter))
        return result

    def _adapter_by_block(self, block_idx):
        block = self.model_without_dp.clip_model.visual.transformer.resblocks[int(block_idx)]
        adapter = getattr(block, "moe_adapter", None)
        if adapter is None:
            raise KeyError(f"Block {block_idx} has no VisualMoEAdapter.")
        return adapter

    def _visual_moe_cfg(self):
        return self.cfg.TRAINER.BiMC.VISUAL_MOE

    def _checkpoint_cfg(self):
        return getattr(self.cfg.TRAINER.BiMC, "CHECKPOINT", None)

    @staticmethod
    def _cpu_copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: Runner._cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Runner._cpu_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(Runner._cpu_copy(item) for item in value)
        return copy.deepcopy(value)

    @staticmethod
    def _move_state_to_device(state, device):
        result = {}
        for key, value in state.items():
            result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return result

    def merge_dicts(self, dict_list):
        """Merge task classifier states by global class id, not append order."""
        if not dict_list:
            raise ValueError("Cannot merge an empty classifier state list.")
        class_ids = []
        per_state_ids = []
        for state in dict_list:
            ids = state.get("class_ids", state.get("class_index"))
            if ids is None:
                raise KeyError("Each task classifier state needs class_ids or class_index.")
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
            per_state_ids.append(ids)
            class_ids.append(ids)
        all_ids = torch.cat(class_ids, dim=0)
        if torch.unique(all_ids).numel() != all_ids.numel():
            raise ValueError("Classifier states contain duplicate global class ids.")
        order = all_ids.argsort()
        result = {"class_ids": all_ids[order]}
        result["task_ids"] = torch.cat([
            torch.full_like(ids, int(state.get("creation_session", index)))
            for index, (ids, state) in enumerate(zip(per_state_ids, dict_list))
        ])[order]
        candidate_keys = ("description_proto", "text_features", "image_proto", "final_proto")
        for key in candidate_keys:
            if not all(key in state for state in dict_list):
                continue
            chunks = []
            valid = True
            for state, ids in zip(dict_list, per_state_ids):
                value = torch.as_tensor(state[key], device=self.device)
                if value.ndim < 1 or value.size(0) != ids.numel():
                    valid = False
                    break
                chunks.append(value)
            if valid:
                result[key] = torch.cat(chunks, dim=0)[order]
        if "image_proto" not in result or "text_features" not in result or "final_proto" not in result:
            raise KeyError("Classifier state is missing calibrated visual/text/final prototypes.")
        result["class_index"] = result["class_ids"]
        return result

    @staticmethod
    def _targets_to_columns(labels, classifier_state):
        class_ids = torch.as_tensor(classifier_state["class_ids"], device=labels.device, dtype=torch.long)
        if class_ids.ndim != 1 or class_ids.numel() == 0:
            raise ValueError("Classifier class_ids must be a non-empty one-dimensional tensor.")
        matches = labels.reshape(-1, 1).eq(class_ids.reshape(1, -1))
        valid = matches.sum(dim=1).eq(1)
        if not valid.all():
            missing = labels[~valid].unique().tolist()
            raise ValueError(
                "Classifier bank must contain each target exactly once; invalid labels: "
                f"{missing}"
            )
        return matches.to(dtype=torch.long).argmax(dim=1)

    @staticmethod
    def _batch_source_ids(batch, expected_size, device):
        raw = None
        if isinstance(batch, dict):
            raw = batch.get("source_ids", batch.get("source_id", batch.get("idx")))
        if raw is None:
            raise KeyError("Incremental batch requires stable source_id for query-exclusive LOO training.")
        source_ids = torch.as_tensor(raw, device=device).reshape(-1)
        if source_ids.numel() != expected_size:
            raise ValueError("Batch source ids must align with image batch size.")
        return source_ids

    @staticmethod
    def _select_uncovered_batch(images, labels, source_ids, uncovered_classes):
        class_ids = torch.as_tensor(uncovered_classes, device=labels.device, dtype=torch.long).reshape(-1)
        mask = torch.isin(labels, class_ids)
        return images[mask], labels[mask], source_ids[mask]

    def _support_manifest_hash(self):
        """Short stable identity for the exact support protocol in this run."""
        manifest = self.data_manager.get_support_manifest()
        payload = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _log_expert_topology(self, tag):
        rows = []
        for block_idx, adapter in self._moe_adapters():
            rows.append(
                f"B{block_idx}:experts={adapter.num_experts} "
                f"birth_sessions={adapter.expert_birth_sessions}"
            )
        if rows:
            print(f"=> [Experts][{tag}] " + " | ".join(rows))

    @torch.no_grad()
    def _routing_diagnostics(self, loader, tag, current_task=None):
        adapters = dict(self._moe_adapters())
        if not adapters:
            return {}
        totals = {
            block: {"weight_sum": torch.zeros(adapter.num_experts),
                    "base_top1": torch.zeros(adapter.base_expert_count), "count": 0}
            for block, adapter in adapters.items()
        }
        self.model.eval()
        for batch in loader:
            images, _ = self.parse_batch(batch)
            _, aux_list = self.model_without_dp.extract_img_feature_train(images)
            for (block_idx, adapter), aux in zip(self._moe_adapters(), aux_list):
                weights = aux["route_weights"].detach().float().cpu()
                item = totals[block_idx]
                item["weight_sum"] += weights.sum(dim=0)
                base_top1 = weights[:, :adapter.base_expert_count].argmax(dim=-1)
                item["base_top1"] += F.one_hot(
                    base_top1, adapter.base_expert_count).sum(dim=0)
                item["count"] += weights.size(0)
        output = {}
        for block_idx, item in totals.items():
            count = max(1, item["count"])
            means, usage = item["weight_sum"] / count, item["base_top1"] / count
            rows = []
            adapter = adapters[block_idx]
            for expert_id in range(adapter.num_experts):
                born = adapter.expert_birth_sessions[expert_id]
                role = "new" if current_task is not None and born == int(current_task) else "history"
                if expert_id < adapter.base_expert_count:
                    rows.append(f"e{expert_id}({role},birth={born}) base_weight={means[expert_id]:.3f} "
                                f"base_top1={usage[expert_id]:.3f}")
                else:
                    rows.append(f"e{expert_id}({role},birth={born}) gate_mean={means[expert_id]:.3f}")
            print(f"=> [Router][{tag}][B{block_idx}] " + " | ".join(rows))
            output[block_idx] = {"mean_weight": means, "base_top1_usage": usage}
        return output

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------
    def _checkpoint_enabled(self):
        return bool(_cfg_value(self._checkpoint_cfg(), "ENABLE", False))

    def _checkpoint_dir(self):
        raw = _cfg_value(self._checkpoint_cfg(), "DIR", "checkpoints/fscil_moe")
        return Path(raw)

    @staticmethod
    def _rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _load_resume_checkpoint_if_requested(self):
        resume = _cfg_value(self._checkpoint_cfg(), "RESUME", "")
        if not resume:
            return
        path = Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        try:
            # RNG state intentionally contains Python/NumPy objects, so this
            # is a trusted local experiment checkpoint rather than a
            # weights-only interchange file.  Explicitly state that for
            # PyTorch versions whose default changed to ``weights_only=True``.
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Resume checkpoint must contain a dictionary payload.")
        required = (
            "schema_version", "completed_session", "model_state", "task_states",
            "support_manifest", "moe_topology", "expansion_thresholds",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
        schema = int(payload["schema_version"])
        if schema != 14:
            raise ValueError(
                "This branch uses reference-transported BiMC checkpoint schema (14); "
                "older single-path or pre-reference-prototype checkpoints are incompatible."
            )
        if not isinstance(payload["model_state"], dict):
            raise TypeError("Resume checkpoint model_state must be a state_dict mapping.")
        if not isinstance(payload["task_states"], list):
            raise TypeError("Resume checkpoint task_states must be a list.")
        diagnostics = payload.get("session_diagnostics", [])
        if not isinstance(diagnostics, list):
            raise TypeError("Resume checkpoint session_diagnostics must be a list.")
        completed_session = int(payload["completed_session"])
        if completed_session != len(payload["task_states"]) - 1:
            raise ValueError(
                "Resume checkpoint has inconsistent completed_session/task_states length: "
                f"{completed_session} versus {len(payload['task_states'])} states."
            )
        if diagnostics and len(diagnostics) != len(payload["task_states"]):
            raise ValueError(
                "Resume checkpoint has inconsistent session_diagnostics/task_states length: "
                f"{len(diagnostics)} versus {len(payload['task_states'])}."
            )
        topology = payload.get("moe_topology", {})
        if not isinstance(topology, dict):
            raise TypeError("Resume checkpoint moe_topology must be a mapping.")
        self.model_without_dp.rebuild_moe_topology(topology)
        self.model_without_dp.load_state_dict(payload["model_state"], strict=True)
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        thresholds = payload.get("expansion_thresholds")
        if not isinstance(thresholds, dict):
            raise TypeError("Checkpoint expansion_thresholds must be a mapping.")
        self.expansion_thresholds = {
            int(key): float(value) for key, value in thresholds.items()
        }
        if not all(np.isfinite(value) for value in self.expansion_thresholds.values()):
            raise ValueError("Checkpoint expansion thresholds must be finite.")
        if bool(self._visual_moe_cfg().ENABLE):
            expected_blocks = {int(block) for block in self.model_without_dp.expandable_blocks()}
            if set(self.expansion_thresholds) != expected_blocks:
                raise ValueError(
                    "Checkpoint expansion thresholds do not cover the deployed MoE blocks: "
                    f"{set(self.expansion_thresholds)} versus {expected_blocks}."
                )
        checkpoint_class_ids = []
        for session_index, state in enumerate(payload["task_states"]):
            required_state = ("class_ids", "image_proto", "reference_image_proto",
                              "text_features", "description_proto", "final_proto")
            missing_state = [key for key in required_state if key not in state]
            if missing_state:
                raise ValueError(
                    f"Checkpoint task state {session_index} is missing calibrated fields: {missing_state}"
                )
            class_ids = torch.as_tensor(state["class_ids"], dtype=torch.long).reshape(-1)
            checkpoint_class_ids.append(class_ids)
            for key in ("image_proto", "reference_image_proto", "text_features",
                        "description_proto", "final_proto"):
                prototype = torch.as_tensor(state[key]).float()
                if prototype.ndim != 2 or prototype.size(0) != class_ids.numel():
                    raise ValueError(
                        f"Checkpoint task state {session_index} has invalid {key}/class-id alignment."
                    )
                if not torch.isfinite(prototype).all():
                    raise ValueError(
                        f"Checkpoint task state {session_index} has non-finite {key}."
                    )
        all_checkpoint_ids = torch.cat(checkpoint_class_ids)
        if torch.unique(all_checkpoint_ids).numel() != all_checkpoint_ids.numel():
            raise ValueError("Checkpoint task states contain duplicate global class ids.")
        manifest = payload.get("support_manifest")
        if manifest is None:
            raise ValueError("Resume checkpoint support_manifest must not be null.")
        self.data_manager.load_support_manifest(manifest)
        base_proto = payload.get("base_vision_prototype")
        self.model_without_dp.base_vision_prototype = (
            None if base_proto is None else base_proto.to(self.device)
        )
        self._resume_state_dict_list = payload.get("task_states", [])
        self.acc_list = payload.get("acc_list", [])
        self.task_acc_list = payload.get("task_acc_list", [])
        self.session_diagnostics = diagnostics
        # Thresholds were validated and installed before restoring support data.
        self._resume_completed_session = completed_session
        self._restore_rng_state(payload.get("rng_state"))
        print(f"=> Resumed session {self._resume_completed_session} from {path}")

    def save_checkpoint(self, completed_session, state_dict_list):
        if not self._checkpoint_enabled():
            return None
        directory = self._checkpoint_dir()
        directory.mkdir(parents=True, exist_ok=True)
        topology = self.model_without_dp.export_moe_topology()
        payload = {
            "schema_version": 14,
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
            "task_states": self._cpu_copy(state_dict_list),
            "base_vision_prototype": self._cpu_copy(self.model_without_dp.base_vision_prototype),
            "support_manifest": self.data_manager.get_support_manifest(),
            "acc_list": list(self.acc_list),
            "task_acc_list": copy.deepcopy(self.task_acc_list),
            "session_diagnostics": self._cpu_copy(self.session_diagnostics),
            "expansion_thresholds": {int(key): float(value) for key, value in self.expansion_thresholds.items()},
            "rng_state": self._rng_state(),
            "config": self.cfg.dump() if hasattr(self.cfg, "dump") else str(self.cfg),
        }
        target = directory / f"session_{int(completed_session):02d}.pth"
        temp_target = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temp_target)
        os.replace(temp_target, target)
        if bool(_cfg_value(self._checkpoint_cfg(), "SAVE_EVERY_SESSION", True)):
            latest = directory / "latest.pth"
            temp_latest = latest.with_suffix(latest.suffix + ".tmp")
            torch.save(payload, temp_latest)
            os.replace(temp_latest, latest)
        print(f"=> Saved checkpoint: {target}")
        return target

    # ------------------------------------------------------------------
    # Base MoE and descriptors
    # ------------------------------------------------------------------
    def build_base_optimizer(self):
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        base_lr = _cfg_value(optim_cfg, "LR_BASE_MOE", _cfg_value(optim_cfg, "LR_MOE", 1e-4))
        wd = _cfg_value(optim_cfg, "WEIGHT_DECAY", 1e-4)
        moe_params = []
        for name, param in self.model_without_dp.named_parameters():
            if not param.requires_grad or "moe_adapter" not in name:
                continue
            # Descriptors are fitted in a separate reconstruction-only stage.
            if ".descriptors." in name:
                continue
            moe_params.append(param)
        if not moe_params:
            raise RuntimeError("No trainable Base visual MoE parameters were found.")
        return optim.AdamW([{ "params": moe_params, "lr": base_lr }], weight_decay=wd,
                           betas=tuple(_cfg_value(optim_cfg, "BETAS", (0.9, 0.999))))

    def _train_descriptor(self, adapter, expert_id, values, epochs, sample_weights=None,
                          calibration_values=None):
        """Fit the SEMA linear AE on the full local representation distribution."""
        descriptor = adapter.descriptors[int(expert_id)]
        descriptor.float()
        values = values.detach().float()
        weights = (torch.ones(values.size(0), device=values.device) if sample_weights is None
                   else torch.as_tensor(sample_weights, device=values.device).float().reshape(-1))
        if weights.numel() != values.size(0) or not bool((weights.sum() > 0).item()):
            raise ValueError("AE sample weights must align with features and have positive mass.")
        if values.ndim != 2 or values.size(0) < 3:
            raise ValueError("AE fitting requires at least three CLS representations.")
        descriptor.set_center(values, weights)
        for parameter in descriptor.parameters():
            parameter.requires_grad = True
        descriptor.train()
        optimizer = optim.AdamW(
            descriptor.parameters(),
            lr=float(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "LR_DESCRIPTOR", 1e-3)),
            weight_decay=0.0,
        )
        epochs = max(1, int(epochs))
        for _ in range(epochs):
            permutation = torch.randperm(values.size(0), device=values.device)
            for indices in permutation.split(max(3, min(256, values.size(0)))):
                optimizer.zero_grad(set_to_none=True)
                reconstruction = descriptor(values[indices])
                per_sample = (reconstruction - values[indices]).square().mean(dim=-1)
                batch_weights = weights[indices]
                reconstruction_loss = (
                    (per_sample * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
                )
                loss = reconstruction_loss + 1e-3 * descriptor.orthogonality_loss()
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite AE reconstruction loss.")
                loss.backward()
                optimizer.step()
        descriptor.eval()
        with torch.no_grad():
            reference_values = (
                values if calibration_values is None
                else torch.as_tensor(calibration_values, device=values.device).float()
            )
            if reference_values.ndim != 2 or reference_values.size(0) < 3:
                raise ValueError("AE error calibration requires at least three held-out representations.")
            errors = descriptor.reconstruction_error(reference_values)
            if sample_weights is not None and calibration_values is None:
                probabilities = weights.clamp_min(0)
                if not bool((probabilities.sum() > 0).item()):
                    raise ValueError("AE responsibility distribution has zero mass.")
                reference_count = min(adapter.descriptor_buffer_size, errors.numel())
                selected = torch.multinomial(
                    probabilities / probabilities.sum(), reference_count, replacement=True)
                errors = errors[selected]
            adapter.update_descriptor_stats(errors, expert_id)
        for parameter in descriptor.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        if not descriptor.is_ready():
            raise RuntimeError("AE descriptor did not reach its minimum reference-buffer size.")
        return float(loss.item())

    def _train_base_descriptors(self, extract_loader):
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "BASE_DESCRIPTOR_EPOCHS", 8))
        print(f"\n========== Fit Base SEMA Representation Descriptors ({epochs} epochs) ==========")
        feature_map = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        self._base_calibration_features = {}
        for block_idx, adapter in self._moe_adapters():
            item = feature_map[block_idx]
            values = item["cls_in"].to(self.device)
            labels = item["labels"].to(self.device)
            source_ids = item["source_ids"].to(self.device)
            # Hold out one stable source per class. The AE is fitted without
            # these examples; only held-out errors calibrate its reference.
            calibration_mask = torch.zeros(labels.numel(), dtype=torch.bool, device=self.device)
            for class_id in torch.unique(labels, sorted=True):
                indices = torch.nonzero(labels.eq(class_id), as_tuple=False).flatten()
                if indices.numel() > 1:
                    order = torch.argsort(source_ids[indices], stable=True)
                    calibration_mask[indices[order[0]]] = True
            fit_mask = ~calibration_mask
            if int(calibration_mask.sum()) < 3 or int(fit_mask.sum()) < 3:
                # Tiny synthetic fixtures cannot support independent fitting
                # and calibration; production Base protocols should.
                fit_mask = torch.ones_like(calibration_mask)
                calibration_mask = torch.zeros_like(calibration_mask)
            fit_values = values[fit_mask]
            calibration_values = values[calibration_mask] if calibration_mask.any() else None
            self._base_calibration_features[int(block_idx)] = {
                "cls_in": (values[calibration_mask] if calibration_mask.any() else values).detach().cpu(),
                "labels": (labels[calibration_mask] if calibration_mask.any() else labels).detach().cpu(),
            }
            with torch.no_grad():
                _, responsibilities = adapter.router(fit_values)
            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(
                    adapter, expert_id, fit_values, epochs,
                    responsibilities[:, expert_id],
                    calibration_values=calibration_values,
                )
                descriptor = adapter.descriptors[expert_id]
                print(f"=> [SEMA-AE][B{block_idx}][E{expert_id}] loss={loss:.6f} "
                      f"reference_n={int(descriptor.error_count.item())} "
                      f"mean={descriptor.error_mean.item():.6f} std={descriptor.error_std.item():.6f}")
            adapter.assert_descriptors_ready()
            adapter.freeze_all()


    @torch.no_grad()
    def _calibrate_expansion_thresholds(self, extract_loader):
        """Calibrate per-layer known-class false-expansion thresholds on Base holdout sources."""
        alpha = float(getattr(self._visual_moe_cfg(), "EXPANSION_FALSE_POSITIVE_RATE", 0.05))
        if not 0.0 < alpha < 1.0:
            raise ValueError("EXPANSION_FALSE_POSITIVE_RATE must be in (0, 1).")
        if not getattr(self, "_base_calibration_features", None):
            raise RuntimeError("Base AE holdout features must be prepared before threshold calibration.")
        thresholds = {}
        for block_idx, adapter in self._moe_adapters():
            item = self._base_calibration_features[int(block_idx)]
            values, labels = item["cls_in"].to(self.device), item["labels"].to(self.device)
            if values.size(0) < 3:
                raise RuntimeError(f"Block {block_idx} has fewer than three held-out Base representations.")
            sample_best = adapter.descriptor_scores(values).min(dim=1).values
            class_scores = torch.stack([
                sample_best[labels.eq(class_id)].median()
                for class_id in torch.unique(labels, sorted=True)
            ])
            if class_scores.numel() < 2 or not torch.isfinite(class_scores).all():
                raise RuntimeError(f"Block {block_idx} has insufficient finite Base classes for threshold calibration.")
            threshold = torch.quantile(class_scores.float(), 1.0 - alpha)
            thresholds[int(block_idx)] = float(threshold.item())
            print(f"=> [ExpansionCalibration][B{block_idx}] alpha={alpha:.3f} threshold={threshold.item():.4f}")
        if set(thresholds) != {int(block) for block, _ in self._moe_adapters()}:
            raise RuntimeError("Expansion threshold calibration did not cover every MoE block.")
        self.expansion_thresholds = thresholds


    def _collect_class_anchors(self, loader):
        """Compact per-class, per-layer Gaussian anchors; no old images are retained."""
        feature_map = self.model_without_dp.collect_blockwise_cls_features(loader)
        anchors = {}
        for block_idx, item in feature_map.items():
            values = item["cls_in"].float()
            labels = item["labels"].long()
            means, variances, class_ids = [], [], []
            for class_id in torch.unique(labels, sorted=True):
                group = values[labels.eq(class_id)]
                means.append(group.mean(0))
                variances.append(group.var(0, unbiased=False).clamp_min(1e-6))
                class_ids.append(int(class_id))
            anchors[int(block_idx)] = {
                "class_ids": torch.tensor(class_ids, dtype=torch.long),
                "mean": torch.stack(means).cpu().half(),
                "var": torch.stack(variances).cpu().half(),
            }
        return anchors

    def _all_old_anchors(self, old_states, block_idx):
        means, variances = [], []
        for state in old_states:
            bank = state.get("anchor_by_block", {}).get(int(block_idx))
            if bank is not None:
                means.append(torch.as_tensor(bank["mean"], device=self.device).float())
                variances.append(torch.as_tensor(bank["var"], device=self.device).float())
        if not means:
            return None, None
        return torch.cat(means, dim=0), torch.cat(variances, dim=0)

    def _visual_intra_modal_calibration(self, visual, reference, old_states):
        """Calibrate in the frozen Base space, then transport the adapted support shift."""
        visual = F.normalize(visual.float(), dim=-1)
        reference = F.normalize(reference.float(), dim=-1)
        if visual.shape != reference.shape:
            raise ValueError("Adapted and reference visual prototypes must have identical shapes.")
        if not old_states:
            return visual
        base_visual = F.normalize(
            torch.as_tensor(old_states[0]["reference_image_proto"], device=self.device).float(), dim=-1
        )
        tau = float(getattr(self.cfg.TRAINER.BiMC, "TAU", 16.0))
        weights = torch.softmax(tau * reference @ base_visual.t(), dim=-1)
        guided_reference = F.normalize(weights @ base_visual, dim=-1)
        lambda_i = (
            float(self.cfg.TRAINER.BiMC.LAMBDA_I)
            if bool(getattr(self.cfg.TRAINER.BiMC, "VISION_CALIBRATION", True)) else 0.0
        )
        calibrated_reference = F.normalize(
            (1.0 - lambda_i) * reference + lambda_i * guided_reference, dim=-1
        )
        return F.normalize(calibrated_reference + visual - reference, dim=-1)

    def _text_intra_modal_calibration(self, text, description):
        """Fuse prompt and generated-description prototypes within text modality."""
        lambda_t = (
            float(self.cfg.TRAINER.BiMC.LAMBDA_T)
            if bool(getattr(self.cfg.TRAINER.BiMC, "TEXT_CALIBRATION", True)) else 0.0
        )
        return F.normalize(
            (1.0 - lambda_t) * F.normalize(text.float(), dim=-1)
            + lambda_t * F.normalize(description.float(), dim=-1),
            dim=-1,
        )

    def _inter_modal_fusion(self, visual, semantic):
        """Fuse calibrated visual and text prototypes in the shared CLIP space."""
        beta = float(getattr(self.cfg.TRAINER.BiMC, "INTER_MODAL_WEIGHT", 0.5))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("INTER_MODAL_WEIGHT must be in [0, 1].")
        return F.normalize(
            (1.0 - beta) * F.normalize(visual.float(), dim=-1)
            + beta * F.normalize(semantic.float(), dim=-1),
            dim=-1,
        )

    def _build_incremental_bank(self, loader, class_index, text, description, old_states):
        """Build a global visual bank with query-exclusive novel prototypes."""
        if not old_states:
            raise ValueError("Incremental visual classification requires the Base classifier state.")
        cache = self.model_without_dp.build_leave_one_out_prototypes(loader, class_index)
        if not bool(cache["loo_valid"].all().item()):
            raise ValueError("Every incremental training source needs another same-class source for LOO.")

        classes = cache["classes"].to(self.device)
        labels = cache["labels"].to(self.device)
        sources = cache["source_ids"].to(self.device)
        features = cache["features"].to(self.device).float()
        full_visual = F.normalize(torch.stack([
            features[labels.eq(cls)].mean(0) for cls in classes
        ]), dim=-1)
        loo_visual = cache["loo_visual_prototypes"].to(self.device).float()
        reference_features = cache["reference_features"].to(self.device).float()
        reference_full = F.normalize(torch.stack([
            reference_features[labels.eq(cls)].mean(0) for cls in classes
        ]), dim=-1)
        loo_reference = cache["loo_reference_prototypes"].to(self.device).float()

        semantic = self._text_intra_modal_calibration(text, description)
        requested_ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long).reshape(-1)
        matches = classes[:, None].eq(requested_ids[None, :])
        if not matches.sum(dim=1).eq(1).all():
            raise ValueError("Incremental semantic and visual class ids do not align.")
        semantic = semantic[matches.long().argmax(dim=1)]

        novel_visual_full = self._visual_intra_modal_calibration(
            full_visual, reference_full, old_states
        )
        novel_visual_loo = torch.stack([
            self._visual_intra_modal_calibration(current, reference, old_states)
            for current, reference in zip(loo_visual, loo_reference)
        ])
        novel_full = self._inter_modal_fusion(novel_visual_full, semantic)
        novel_loo = torch.stack([
            self._inter_modal_fusion(row, semantic) for row in novel_visual_loo
        ])

        old_visual, old_ids = [], []
        for state in old_states:
            old_visual.append(F.normalize(
                torch.as_tensor(state.get("final_proto", state["image_proto"]), device=self.device).float(), dim=-1
            ))
            ids = state.get("class_ids", state.get("class_index"))
            old_ids.append(torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1))
        old_visual = torch.cat(old_visual, dim=0)
        old_ids = torch.cat(old_ids, dim=0)
        global_full = torch.cat([old_visual, novel_full], dim=0)
        global_ids = torch.cat([old_ids, classes], dim=0)
        if torch.unique(global_ids).numel() != global_ids.numel():
            raise ValueError("Global visual prototype classifier contains duplicate class ids.")
        if global_full.size(0) != global_ids.numel():
            raise RuntimeError("Global visual prototypes and class ids lost row alignment.")
        return {
            "classes": classes, "labels": labels, "source_ids": sources,
            "novel_full": novel_full, "novel_loo": novel_loo,
            "global_full": global_full, "global_ids": global_ids,
            "old_class_count": old_ids.numel(),
        }

    def _global_incremental_logits(self, images, labels, source_ids, bank, scale):
        if torch.is_grad_enabled():
            features, _ = self.model_without_dp.extract_img_feature_train(images)
        else:
            features = self.model_without_dp.extract_img_feature(images)
        features = F.normalize(features.float(), dim=-1)
        prototypes = bank["global_full"].unsqueeze(0).expand(images.size(0), -1, -1).clone()
        prototypes[:, int(bank["old_class_count"]):] = bank["novel_full"].unsqueeze(0)
        if source_ids is not None:
            source_ids = torch.as_tensor(source_ids, device=images.device).reshape(-1)
            match = source_ids[:, None].eq(bank["source_ids"][None, :])
            if not bool(match.any(dim=1).all().item()):
                raise KeyError("Training query source is absent from the LOO support bank.")
            rows = match.long().argmax(dim=1)
            own = self._targets_to_columns(labels, {"class_ids": bank["global_ids"]})
            novel_loo = bank["novel_loo"][rows]
            novel_columns = own - int(bank["old_class_count"])
            if (novel_columns < 0).any():
                raise RuntimeError("Incremental query target unexpectedly maps to an old class.")
            prototypes[torch.arange(images.size(0), device=images.device), own] = novel_loo[
                torch.arange(images.size(0), device=images.device), novel_columns
            ]
        logits = float(scale) * torch.einsum("bd,bcd->bc", features, prototypes)
        return logits, self._targets_to_columns(labels, {"class_ids": bank["global_ids"]})

    def train_base_task(self, task_stat, train_loader, extract_loader):
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        loss_cfg = self.cfg.TRAINER.BiMC.LOSS
        epochs = int(_cfg_value(optim_cfg, "BASE_EPOCHS", 50))
        ce_weight = float(_cfg_value(loss_cfg, "CE_WEIGHT", 1.0))
        kd_weight = float(_cfg_value(loss_cfg, "KD_WEIGHT", 0.1))
        lb_weight = float(_cfg_value(loss_cfg, "LB_WEIGHT", 0.01))
        num_experts = int(_cfg_value(self._visual_moe_cfg(), "NUM_BASE_EXPERTS", 4))
        optimizer = self.build_base_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        print(f"\n========== Start Unified Base MoE Training ({epochs} epochs) ==========")
        for epoch in range(epochs):
            self.model.train()
            totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "lb": 0.0, "n": 0}
            for batch in tqdm(train_loader, desc=f"Base Epoch {epoch + 1}/{epochs}"):
                images, labels = self.parse_batch(batch)
                targets = self._targets_to_columns(labels, task_stat)
                optimizer.zero_grad(set_to_none=True)
                logits, img_feat, teacher_feat, moe_aux = self.model_without_dp.forward_train(
                    images, task_stat, compute_teacher=True
                )
                loss_ce = F.cross_entropy(logits, targets)
                loss_kd = 1.0 - F.cosine_similarity(img_feat, teacher_feat, dim=-1).mean()
                loss_lb = compute_load_balance_loss(moe_aux, num_experts).to(images.device)
                loss = ce_weight * loss_ce + kd_weight * loss_kd + lb_weight * loss_lb
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_without_dp.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                totals["loss"] += float(loss.item())
                totals["ce"] += float(loss_ce.item())
                totals["kd"] += float(loss_kd.item())
                totals["lb"] += float(loss_lb.item())
                totals["n"] += 1
            scheduler.step()
            n = max(1, totals["n"])
            print(f"=> [Base][{epoch+1}/{epochs}] loss={totals['loss']/n:.4f} "
                  f"ce={totals['ce']/n:.4f} kd={totals['kd']/n:.4f} lb={totals['lb']/n:.4f}")
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._train_base_descriptors(extract_loader)
        self._calibrate_expansion_thresholds(extract_loader)

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, old_states):
        """Transactional wrapper: a failed session cannot leave partial experts."""
        snapshots = []
        for block_idx, adapter in self._moe_adapters():
            snapshots.append((
                int(block_idx), adapter.export_topology(),
                {key: value.detach().clone() for key, value in adapter.state_dict().items()},
            ))
        try:
            return self._train_incremental_task_impl(
                task_id, train_loader, extract_loader, class_index,
                current_class_name, old_states,
            )
        except Exception:
            # Restore every layer, not just the layer where the exception arose.
            for block_idx, topology, state in snapshots:
                adapter = self._adapter_by_block(block_idx)
                adapter.rebuild_topology(topology)
                adapter.load_state_dict(state, strict=True)
                adapter.freeze_all()
            raise

    def _train_incremental_task_impl(self, task_id, train_loader, extract_loader, class_index,
                                      current_class_name, old_states):
        """Expand only where AE coverage fails; train new experts on uncovered classes."""
        text, description = self._text_prototypes(current_class_name, class_index)
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        epochs = int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)
        scale = float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        fallback_threshold = float(getattr(self._visual_moe_cfg(), "EXPANSION_Z_THRESHOLD", 1.0))
        ae_epochs = int(_cfg_value(optim_cfg, "INCREMENTAL_DESCRIPTOR_EPOCHS", 5))
        decisions, expanded, histories = [], [], {}
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        # Train shallow-to-deep: upstream experts alter downstream adapted inputs.\n        # Expansion decisions still use the independent frozen-reference stream.
        for block_idx in self.model_without_dp.expandable_blocks():
            features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
            threshold = float(getattr(self, "expansion_thresholds", {}).get(int(block_idx), fallback_threshold))
            decision = self.model_without_dp.plan_layer_expansion(
                block_idx, features, class_index, threshold)
            decisions.append({
                "block_idx": int(block_idx), "expand": bool(decision.expand),
                "class_min_z": dict(decision.class_scores),
                "uncovered_classes": list(decision.uncovered_classes),
            })
            if not decision.expand:
                continue

            uncovered_classes = torch.as_tensor(
                decision.uncovered_classes, device=self.device, dtype=torch.long
            ).reshape(-1)
            if uncovered_classes.numel() == 0:
                raise RuntimeError("AE requested expansion without an uncovered class.")

            adapter = self._adapter_by_block(block_idx)
            expert_id = self.model_without_dp.add_expert(block_idx, task_id)
            anchor_mean, anchor_var = self._all_old_anchors(old_states, block_idx)
            optimizer = optim.AdamW([
                {"params": adapter.experts[-1].parameters(),
                 "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                {"params": adapter.router.columns[-1].parameters(),
                 "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)},
            ], weight_decay=float(optim_cfg.WEIGHT_DECAY))
            loss_history, selected_samples = [], 0

            for epoch in range(epochs):
                bank = self._build_incremental_bank(
                    extract_loader, class_index, text, description, old_states)
                self.model.train()
                total, count = 0.0, 0
                for batch in train_loader:
                    all_images, all_labels = self.parse_batch(batch)
                    all_source_ids = self._batch_source_ids(
                        batch, all_labels.numel(), all_labels.device
                    )
                    selected_mask = torch.isin(all_labels, uncovered_classes)
                    images = all_images[selected_mask]
                    labels = all_labels[selected_mask]
                    source_ids = all_source_ids[selected_mask]

                    optimizer.zero_grad(set_to_none=True)
                    # Route responsibility is supervised on every current-session
                    # class: uncovered classes are positive, covered classes negative.
                    _, route_aux = self.model_without_dp.extract_img_feature_train(all_images)
                    aux_by_block = {
                        block: aux for (block, _), aux in zip(self._moe_adapters(), route_aux)
                    }
                    route_logits = aux_by_block[int(block_idx)]["logits"][:, -1]
                    route_targets = selected_mask.to(dtype=route_logits.dtype)
                    route_responsibility = F.binary_cross_entropy_with_logits(
                        route_logits, route_targets
                    )
                    route_sparse = torch.sigmoid(route_logits).mean()
                    if labels.numel() > 0:
                        logits, targets = self._global_incremental_logits(
                            images, labels, source_ids, bank, scale)
                        classification = F.cross_entropy(logits, targets)
                    else:
                        classification = route_responsibility.new_zeros(())
                    old_suppression = adapter.old_anchor_suppression_loss(
                        anchor_mean, anchor_var,
                        residual_weight=float(getattr(
                            self.cfg.TRAINER.BiMC.INCREMENTAL, "OLD_RESIDUAL_WEIGHT", 1.0)))
                    adapter_penalty = torch.stack([
                        parameter.float().square().mean()
                        for parameter in adapter.experts[-1].parameters()
                    ]).mean()
                    incremental_cfg = self.cfg.TRAINER.BiMC.INCREMENTAL
                    loss = (
                        classification
                        + float(getattr(incremental_cfg, "ROUTE_WEIGHT", 1.0)) * route_responsibility
                        + float(getattr(incremental_cfg, "OLD_SUPPRESSION_WEIGHT", 1.0)) * old_suppression
                        + float(getattr(incremental_cfg, "SPARSE_WEIGHT", 0.01)) * route_sparse
                        + float(getattr(incremental_cfg, "ADAPTER_WEIGHT", 1e-4)) * adapter_penalty
                    )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("Non-finite global visual-prototype incremental loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.newest_parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.item()) * max(1, all_labels.numel())
                    count += all_labels.numel()
                    selected_samples += labels.numel()

                if count == 0:
                    raise RuntimeError(
                        f"Block {block_idx} expanded for {uncovered_classes.tolist()} "
                        "but no matching training samples were selected."
                    )
                loss_history.append(total / count)
            adapter.freeze_all()

            # Fit the new AE on exactly the CLS distribution handled by its expert.
            new_features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
            local_item = new_features[block_idx]
            local_mask = torch.isin(local_item["labels"], uncovered_classes.detach().cpu())
            local_values = local_item["cls_in"][local_mask].to(self.device)
            local_labels = local_item["labels"][local_mask].to(self.device)
            local_sources = local_item["source_ids"][local_mask].to(self.device)
            calibration_mask = torch.zeros(
                local_labels.numel(), dtype=torch.bool, device=self.device
            )
            for class_id in uncovered_classes:
                indices = torch.nonzero(local_labels.eq(class_id), as_tuple=False).flatten()
                if indices.numel() > 1:
                    order = torch.argsort(local_sources[indices], stable=True)
                    calibration_mask[indices[order[0]]] = True
            fit_mask = ~calibration_mask
            if int(calibration_mask.sum()) < 3 or int(fit_mask.sum()) < 3:
                fit_mask = torch.ones_like(calibration_mask)
                calibration_mask = torch.zeros_like(calibration_mask)
            ae_fit_values = local_values[fit_mask]
            ae_calibration_values = (
                local_values[calibration_mask] if calibration_mask.any() else None
            )
            ae_loss = self._train_descriptor(
                adapter, expert_id, ae_fit_values, ae_epochs,
                calibration_values=ae_calibration_values,
            )
            adapter.freeze_all()
            histories[block_idx] = {
                "expert_id": int(expert_id), "loss": loss_history, "ae_loss": ae_loss,
                "uncovered_classes": uncovered_classes.detach().cpu().tolist(),
                "selected_samples": int(selected_samples),
                "old_anchor_gate_mean": adapter.old_anchor_response(anchor_mean, anchor_var),
            }
            expanded.append({
                "block_idx": int(block_idx), "expert_id": int(expert_id),
                "uncovered_classes": uncovered_classes.detach().cpu().tolist(),
            })

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        return {"trained": bool(expanded), "decisions": decisions, "expanded": expanded,
                "training": histories, "encoder_version": 0}

    def _text_prototypes(self, class_names, class_index):
        model = self.model_without_dp
        text, _ = model.inference_text_feature(class_names, model.template, int(class_index[0]))
        _, _, description = model.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH, int(class_index[0]))
        return text.detach(), description.detach()


    def _build_and_refine_task_state(self, current_class_name, extract_loader, class_index,
                                     old_states):
        """Build a Base-guided visual state without visual-text classifier fusion."""
        state = self.model_without_dp.build_task_statistics(
            current_class_name, extract_loader, class_index,
            calibrate_novel_vision_proto=False,
        )
        text = state["text_features"].to(self.device).float()
        desc = state["description_proto"].to(self.device).float()
        semantic = self._text_intra_modal_calibration(text, desc)
        visual = self._visual_intra_modal_calibration(
            state["image_proto"].to(self.device),
            state["reference_image_proto"].to(self.device),
            old_states,
        )
        state["image_proto"] = visual.detach()
        state["final_proto"] = self._inter_modal_fusion(visual, semantic).detach()
        state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
        state["class_index"] = state["class_ids"]
        state["encoder_version"] = 0
        state["anchor_by_block"] = self._collect_class_anchors(extract_loader)
        return state

    def run(self):
        configured_end = int(_cfg_value(self.cfg.TRAINER.BiMC, "END_SESSION", -1))
        end_task = self.data_manager.num_tasks - 1 if configured_end < 0 else min(
            configured_end, self.data_manager.num_tasks - 1)
        print(f"Start unified FSCIL training/evaluation on sessions [0, {end_task}]")
        state_dict_list = list(self._resume_state_dict_list)
        start_task = self._resume_completed_session + 1
        use_moe = bool(getattr(self.cfg.TRAINER.BiMC, "VISUAL_MOE", None)
                       and self._visual_moe_cfg().ENABLE)
        if start_task > end_task:
            print("=> Resume checkpoint already contains all configured FSCIL sessions.")
            return

        for task_id in range(start_task, end_task + 1):
            self.model.eval()
            adaptation, plan_summary = None, None
            class_index = self.data_manager.class_index_in_task[task_id]
            class_names = np.array(self.data_manager.class_names)[class_index].tolist()
            extract_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="test", accumulate_past=False)
            train_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="train", accumulate_past=False)
            old_states = list(state_dict_list)
            print(f"\n=> [Session {task_id}] support_manifest={self._support_manifest_hash()}")
            if use_moe:
                self._log_expert_topology(f"session{task_id}-before")

            if task_id == 0:
                state = self.model_without_dp.build_task_statistics(
                    class_names, extract_loader, class_index,
                    calibrate_novel_vision_proto=False)
                state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
                state["class_index"] = state["class_ids"]
                if use_moe:
                    self.train_base_task(state, train_loader, extract_loader)
                    self.model.eval()
                    state = self.model_without_dp.build_task_statistics(
                        class_names, extract_loader, class_index,
                        calibrate_novel_vision_proto=False)
                    state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
                    state["class_index"] = state["class_ids"]
                text = state["text_features"].to(self.device).float()
                description = state["description_proto"].to(self.device).float()
                semantic = self._text_intra_modal_calibration(text, description)
                state["final_proto"] = self._inter_modal_fusion(
                    state["image_proto"].to(self.device), semantic
                ).detach()
                state["encoder_version"] = 0
                state["creation_session"] = 0
                state["anchor_by_block"] = self._collect_class_anchors(extract_loader)
            else:
                if use_moe:
                    self._routing_diagnostics(extract_loader, f"session{task_id}-before",
                                              current_task=task_id)
                    adaptation = self.train_incremental_task(
                        task_id, train_loader, extract_loader, class_index,
                        class_names, old_states)
                    plan_summary = self._cpu_copy(adaptation["decisions"])
                self.model.eval()
                state = self._build_and_refine_task_state(
                    class_names, extract_loader, class_index, old_states)
                state["creation_session"] = int(task_id)

            state_dict_list.append(state)
            # Task offsets are deliberately absent from prediction. Keep a
            # zero-valued compatibility field for old checkpoint readers.
            merged = self.merge_dicts(state_dict_list)
            final_routing = {}
            if use_moe:
                final_routing = self._routing_diagnostics(
                    extract_loader, f"session{task_id}-final", current_task=task_id)
                self._log_expert_topology(f"session{task_id}-final")
            start_time = time.perf_counter()
            acc = self.inference_task_bilevel(task_id, merged)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            print(f"+++++++++++ session {task_id}, time: {elapsed:.3f}s ++++++++++++++++")
            task_history = self.task_acc_list + [acc["task_acc"]]
            forgetting = []
            for old_task in range(task_id + 1):
                best = max(float(task_history[session][old_task])
                           for session in range(old_task, task_id + 1))
                current = float(task_history[task_id][old_task])
                forgetting.append(best - current)
            acc["performance_drop"] = float(np.mean(forgetting)) if forgetting else 0.0
            self.acc_list.append(round(acc["mean_acc"], 3))
            acc["average_accuracy"] = float(np.mean(self.acc_list))
            print(f"=> Session [{task_id}], AA={acc['average_accuracy']:.3f}, "
                  f"Current={acc['mean_acc']:.3f}, Base={acc['base_avg_acc']:.2f}, "
                  f"Novel={acc['inc_avg_acc']:.2f}, H={acc['harmonic_acc']:.2f}, "
                  f"PD={acc['performance_drop']:.3f}")
            self.task_acc_list.append(acc["task_acc"])
            self.session_diagnostics.append({
                "session": int(task_id),
                "support_manifest_hash": self._support_manifest_hash(),
                "expansion": self._cpu_copy(plan_summary),
                "adaptation": self._cpu_copy(adaptation),
                "routing": self._cpu_copy(final_routing),
                "expert_counts": {int(b): int(a.num_experts) for b, a in self._moe_adapters()},
                "classifier": self._cpu_copy(acc),
                "evaluation_seconds": float(elapsed),
            })
            self.save_checkpoint(task_id, state_dict_list)

        print(f"\nFinal session accuracy: {self.acc_list}")
        print("Session-wise class accuracy:")
        for index, values in enumerate(self.task_acc_list):
            print(f"session {index:2d}: {values}")

    @torch.no_grad()
    def inference_task_bilevel(self, task_id, state_dict):
        """One current encoder, one feature space, one global calibrated prototype bank."""
        prototypes = F.normalize(state_dict["final_proto"].to(self.device).float(), dim=-1)
        class_ids = torch.as_tensor(state_dict["class_ids"], device=self.device, dtype=torch.long)
        test_loader = self.data_manager.get_dataloader(task_id, source="test", mode="test")
        all_logits, all_targets = [], []
        for batch in tqdm(test_loader, desc=f"Global Prototype Eval {task_id}"):
            images, labels = self.parse_batch(batch)
            features = F.normalize(self.model_without_dp.extract_img_feature(images).float(), dim=-1)
            all_logits.append(100.0 * features @ prototypes.t())
            all_targets.append(labels)
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        result = self.evaluator.calc_accuracy(logits, targets, task_id, class_ids=class_ids)
        result["error_flow"] = self.evaluator.error_flow(
            logits, targets, task_id, class_ids=class_ids)
        result["prototype_classifier"] = {
            "num_classes": int(class_ids.numel()),
            "feature_space_versions": 1,
            "global": True,
        }
        return result

    def parse_batch(self, batch):
        return batch["image"].to(self.device), batch["label"].to(self.device, dtype=torch.long)
