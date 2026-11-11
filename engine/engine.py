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
from models.bimc import BiMC
from utils.evaluator import AccuracyEvaluator


def compute_load_balance_loss(moe_aux_list):
    """Soft importance/load regularization for the base-session sparse router."""
    if not moe_aux_list:
        return torch.tensor(0.0)
    losses = []
    for aux in moe_aux_list:
        probabilities = aux["gate_probabilities"].float()
        indices = aux["top_k_indices"]
        expert_count = probabilities.size(-1)
        assignments = F.one_hot(indices, num_classes=expert_count).float()
        load = assignments.sum(dim=1).mean(dim=0) / max(1, indices.size(1))
        importance = probabilities.mean(dim=0)
        losses.append(expert_count * torch.sum(load * importance))
    return torch.stack(losses).mean()


def _cfg_value(node, name, default):
    return getattr(node, name, default) if node is not None else default


class Runner:
    """FSCIL with one shared feature space and incrementally expanded sparse MoE."""

    CHECKPOINT_SCHEMA = 13

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg)
        # The evaluator must use the dataset manager's global class-to-session
        # partition so metrics and error-flow analysis map logits to FSCIL labels.
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)
        self.device = cfg.DEVICE.DEVICE_NAME
        self.model = BiMC(cfg, self.data_manager.template, self.device)
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.model_without_dp = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.acc_list = []
        self.task_acc_list = []
        self.session_diagnostics = []
        self._resume_completed_session = -1
        self._resume_state_dict_list = []
        self._load_resume_checkpoint_if_requested()

    # ------------------------------------------------------------------
    # Shared model and class-bank helpers
    # ------------------------------------------------------------------
    def _moe_adapters(self):
        result = []
        for block_idx, block in enumerate(
            self.model_without_dp.clip_model.visual.transformer.resblocks
        ):
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

    def merge_dicts(self, state_list):
        """Merge class rows and layer statistics by global class id."""
        if not state_list:
            raise ValueError("Cannot merge an empty class-prototype bank.")
        per_state_ids = []
        for state in state_list:
            ids = state.get("class_ids", state.get("class_index"))
            if ids is None:
                raise KeyError("Every session state must contain global class_ids.")
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
            per_state_ids.append(ids)
        all_ids = torch.cat(per_state_ids)
        if all_ids.unique().numel() != all_ids.numel():
            raise ValueError("Session states contain duplicate global class ids.")
        order = all_ids.argsort()
        merged = {"class_ids": all_ids[order], "class_index": all_ids[order]}

        row_keys = (
            "text_features", "description_proto", "raw_image_proto",
            "image_proto", "fused_proto",
        )
        for key in row_keys:
            if not all(key in state for state in state_list):
                raise KeyError(f"Session prototype bank is missing {key}.")
            rows = []
            for state, ids in zip(state_list, per_state_ids):
                value = torch.as_tensor(state[key], device=self.device)
                if value.ndim != 2 or value.size(0) != ids.numel():
                    raise ValueError(f"{key} rows do not align with their class_ids.")
                rows.append(value)
            merged[key] = torch.cat(rows, dim=0)[order]

        block_sets = [set(state.get("layer_stats", {}).keys()) for state in state_list]
        if not block_sets or any(block_set != block_sets[0] for block_set in block_sets[1:]):
            raise ValueError("All sessions must store layer statistics for the same MoE blocks.")
        merged_stats = {}
        for block_key in sorted(block_sets[0], key=int):
            means, variances, counts, stat_ids = [], [], [], []
            for state, ids in zip(state_list, per_state_ids):
                stats = state["layer_stats"][block_key]
                local_ids = torch.as_tensor(stats["class_ids"], dtype=torch.long).reshape(-1)
                if not torch.equal(local_ids.cpu(), ids.cpu()):
                    raise ValueError(f"Layer {block_key} statistics do not align with class_ids.")
                means.append(torch.as_tensor(stats["mean"], device=self.device).float())
                variances.append(torch.as_tensor(stats["variance"], device=self.device).float())
                counts.append(torch.as_tensor(stats["count"], device=self.device).long())
                stat_ids.append(local_ids.to(self.device))
            merged_stats[block_key] = {
                "class_ids": torch.cat(stat_ids)[order],
                "mean": torch.cat(means, dim=0)[order],
                "variance": torch.cat(variances, dim=0)[order],
                "count": torch.cat(counts, dim=0)[order],
            }
        merged["layer_stats"] = merged_stats

        raw_block_sets = [
            set(state.get("raw_layer_prototypes", {}).keys()) for state in state_list
        ]
        if not raw_block_sets or any(
            block_set != raw_block_sets[0] for block_set in raw_block_sets[1:]
        ):
            raise ValueError("All sessions must store raw frozen-backbone prototypes for the same MoE blocks.")
        merged_raw = {}
        for block_key in sorted(raw_block_sets[0], key=int):
            rows, raw_ids = [], []
            for state, ids in zip(state_list, per_state_ids):
                item = state["raw_layer_prototypes"][block_key]
                local_ids = torch.as_tensor(item["class_ids"], dtype=torch.long).reshape(-1)
                if not torch.equal(local_ids.cpu(), ids.cpu()):
                    raise ValueError(f"Raw layer {block_key} prototypes do not align with class_ids.")
                value = torch.as_tensor(item["prototypes"], device=self.device).float()
                if value.ndim != 2 or value.size(0) != ids.numel():
                    raise ValueError(f"Raw layer {block_key} prototype rows are invalid.")
                rows.append(value)
                raw_ids.append(local_ids.to(self.device))
            merged_raw[block_key] = {
                "class_ids": torch.cat(raw_ids)[order],
                "prototypes": torch.cat(rows, dim=0)[order],
            }
        merged["raw_layer_prototypes"] = merged_raw

        thresholds = state_list[0].get("expansion_thresholds")
        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError("The base session must provide fixed per-layer expansion thresholds.")
        normalized_thresholds = {
            str(key): float(value) for key, value in thresholds.items()
        }
        for state in state_list[1:]:
            other = state.get("expansion_thresholds")
            if other is not None and {
                str(key): float(value) for key, value in other.items()
            } != normalized_thresholds:
                raise ValueError("Expansion thresholds must remain fixed after the base session.")
        merged["expansion_thresholds"] = normalized_thresholds
        return merged

    @staticmethod
    def _targets_to_columns(labels, class_state):
        class_ids = torch.as_tensor(
            class_state["class_ids"], device=labels.device, dtype=torch.long
        ).reshape(-1)
        if class_ids.numel() == 0 or class_ids.unique().numel() != class_ids.numel():
            raise ValueError("Classifier class_ids must be non-empty and unique.")
        matches = labels.reshape(-1, 1).eq(class_ids.reshape(1, -1))
        valid = matches.sum(dim=1).eq(1)
        if not valid.all():
            missing = labels[~valid].unique().tolist()
            raise ValueError(f"Classifier bank is missing target class ids: {missing}")
        return matches.to(dtype=torch.long).argmax(dim=1)

    @staticmethod
    def _batch_source_ids(batch, expected_size, device):
        raw = None
        if isinstance(batch, dict):
            raw = batch.get("source_ids", batch.get("source_id", batch.get("index")))
        if raw is None:
            raise KeyError("Incremental training requires stable support source_ids for LOO.")
        source_ids = torch.as_tensor(raw, device=device).reshape(-1)
        if source_ids.numel() != expected_size:
            raise ValueError("Batch source_ids must align with the image batch.")
        return source_ids

    def _support_manifest_hash(self):
        manifest = self.data_manager.get_support_manifest()
        payload = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _log_expert_topology(self, tag):
        rows = [
            f"B{block}:experts={adapter.num_experts} "
            f"birth_sessions={adapter.expert_birth_sessions} top_k={adapter.top_k}"
            for block, adapter in self._moe_adapters()
        ]
        if rows:
            print(f"=> [Experts][{tag}] " + " | ".join(rows))

    @torch.no_grad()
    def _routing_diagnostics(self, loader, tag):
        adapters = dict(self._moe_adapters())
        if not adapters:
            return {}
        totals = {
            block: {
                "selected": torch.zeros(adapter.num_experts, dtype=torch.float64),
                "weight_sum": torch.zeros(adapter.num_experts, dtype=torch.float64),
                "count": 0,
            }
            for block, adapter in adapters.items()
        }
        was_training = self.model.training
        self.model.eval()
        for batch in loader:
            images, _ = self.parse_batch(batch)
            _, aux_list = self.model_without_dp.extract_img_feature_train(images)
            for (block_idx, adapter), aux in zip(self._moe_adapters(), aux_list):
                indices = aux["top_k_indices"].detach().cpu()
                weights = aux["route_weights"].detach().float().cpu()
                item = totals[block_idx]
                item["selected"] += F.one_hot(
                    indices, num_classes=adapter.num_experts
                ).sum(dim=(0, 1)).double()
                item["weight_sum"] += weights.double().sum(dim=0)
                item["count"] += indices.size(0)
        self.model.train(was_training)
        output = {}
        for block_idx, item in totals.items():
            denominator = max(1, item["count"] * self._adapter_by_block(block_idx).top_k)
            selected_rate = item["selected"] / denominator
            mean_weight = item["weight_sum"] / max(1, item["count"])
            print(
                f"=> [Router][{tag}][B{block_idx}] "
                + " ".join(
                    f"E{index}:selected={selected_rate[index]:.3f},"
                    f"weight={mean_weight[index]:.3f}"
                    for index in range(len(selected_rate))
                )
            )
            output[block_idx] = {
                "selected_rate": selected_rate,
                "mean_weight": mean_weight,
            }
        return output

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------
    def _checkpoint_enabled(self):
        return bool(_cfg_value(self._checkpoint_cfg(), "ENABLE", False))

    def _checkpoint_dir(self):
        return Path(_cfg_value(self._checkpoint_cfg(), "DIR", "checkpoints/unified_dynamic_moe"))

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
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Resume checkpoint must contain a dictionary payload.")
        required = (
            "schema_version", "completed_session", "model_state",
            "moe_topology", "task_states", "support_manifest",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
        schema = int(payload["schema_version"])
        if schema != self.CHECKPOINT_SCHEMA:
            raise ValueError(
                f"Checkpoint schema {schema} is incompatible with the unified sparse-MoE "
                f"architecture (schema {self.CHECKPOINT_SCHEMA}). Restart from Session 0; "
                "versioned/APC checkpoints cannot be migrated without changing model semantics."
            )
        completed = int(payload["completed_session"])
        states = payload["task_states"]
        if not isinstance(states, list) or completed != len(states) - 1:
            raise ValueError("Checkpoint completed_session and task_states are inconsistent.")
        if not isinstance(payload["model_state"], dict):
            raise TypeError("Checkpoint model_state must be a state_dict mapping.")
        topology = payload["moe_topology"]
        if not isinstance(topology, dict):
            raise TypeError("Checkpoint moe_topology must be a mapping.")
        self.model_without_dp.rebuild_moe_topology(topology)
        self.model_without_dp.load_state_dict(payload["model_state"], strict=True)
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        manifest = payload["support_manifest"]
        if manifest is None:
            raise ValueError("Checkpoint support_manifest must not be null.")
        self.data_manager.load_support_manifest(manifest)
        base_prototype = payload.get("base_vision_prototype")
        self.model_without_dp.base_vision_prototype = (
            None if base_prototype is None else base_prototype.to(self.device)
        )
        self._resume_state_dict_list = states
        self.acc_list = list(payload.get("acc_list", []))
        self.task_acc_list = list(payload.get("task_acc_list", []))
        self.session_diagnostics = list(payload.get("session_diagnostics", []))
        self._resume_completed_session = completed
        if len(self.acc_list) != completed + 1 or len(self.task_acc_list) != completed + 1:
            raise ValueError("Checkpoint accuracy history does not match completed sessions.")
        if self.session_diagnostics and len(self.session_diagnostics) != completed + 1:
            raise ValueError("Checkpoint diagnostics do not match completed sessions.")
        self._restore_rng_state(payload.get("rng_state"))
        print(f"=> Resumed unified sparse-MoE session {completed} from {path}")

    def save_checkpoint(self, completed_session, state_list):
        if not self._checkpoint_enabled():
            return None
        directory = self._checkpoint_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.CHECKPOINT_SCHEMA,
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": self.model_without_dp.export_moe_topology(),
            "task_states": self._cpu_copy(state_list),
            "base_vision_prototype": self._cpu_copy(
                self.model_without_dp.base_vision_prototype
            ),
            "support_manifest": self.data_manager.get_support_manifest(),
            "acc_list": list(self.acc_list),
            "task_acc_list": copy.deepcopy(self.task_acc_list),
            "session_diagnostics": self._cpu_copy(self.session_diagnostics),
            "rng_state": self._rng_state(),
            "config": self.cfg.dump() if hasattr(self.cfg, "dump") else str(self.cfg),
        }
        target = directory / f"session_{int(completed_session):02d}.pth"
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        if bool(_cfg_value(self._checkpoint_cfg(), "SAVE_EVERY_SESSION", True)):
            latest = directory / "latest.pth"
            temporary_latest = latest.with_suffix(latest.suffix + ".tmp")
            torch.save(payload, temporary_latest)
            os.replace(temporary_latest, latest)
        print(f"=> Saved unified checkpoint: {target}")
        return target

    # ------------------------------------------------------------------
    # Base-session learning
    # ------------------------------------------------------------------
    def build_base_optimizer(self):
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        parameters = [
            parameter for name, parameter in self.model_without_dp.named_parameters()
            if parameter.requires_grad and "moe_adapter" in name
        ]
        if not parameters:
            raise RuntimeError("No trainable base MoE parameters were found.")
        return optim.AdamW(
            [{"params": parameters, "lr": float(optim_cfg.LR_BASE_MOE)}],
            weight_decay=float(optim_cfg.WEIGHT_DECAY),
            betas=tuple(optim_cfg.BETAS),
        )

    def train_base_task(self, task_stat, train_loader):
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        loss_cfg = self.cfg.TRAINER.BiMC.LOSS
        epochs = int(optim_cfg.BASE_EPOCHS)
        optimizer = self.build_base_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        print(f"\n========== Base-session MoE training ({epochs} epochs) ==========")
        for epoch in range(epochs):
            self.model.train()
            totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "balance": 0.0, "n": 0}
            progress = tqdm(train_loader, desc=f"Base Epoch {epoch + 1}/{epochs}")
            for batch in progress:
                images, labels = self.parse_batch(batch)
                targets = self._targets_to_columns(labels, task_stat)
                optimizer.zero_grad(set_to_none=True)
                logits, image_features, teacher_features, moe_aux = (
                    self.model_without_dp.forward_train(
                        images, task_stat, self.cfg.DATASET.BETA, compute_teacher=True
                    )
                )
                loss_ce = F.cross_entropy(logits, targets)
                loss_kd = 1.0 - F.cosine_similarity(
                    image_features.float(), teacher_features.float(), dim=-1
                ).mean()
                loss_balance = compute_load_balance_loss(moe_aux).to(images.device)
                loss = (
                    float(loss_cfg.CE_WEIGHT) * loss_ce
                    + float(loss_cfg.LB_WEIGHT) * loss_balance
                    + float(loss_cfg.KD_WEIGHT) * loss_kd
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite base-session MoE loss.")
                loss.backward()
                trainable = [
                    parameter for parameter in self.model_without_dp.parameters()
                    if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                totals["loss"] += float(loss.item())
                totals["ce"] += float(loss_ce.item())
                totals["kd"] += float(loss_kd.item())
                totals["balance"] += float(loss_balance.item())
                totals["n"] += 1
                progress.set_postfix({"loss": f"{loss.item():.3f}", "ce": f"{loss_ce.item():.3f}"})
            scheduler.step()
            divisor = max(1, totals["n"])
            print(
                "=> [Base][Epoch {}/{}] loss={:.4f} ce={:.4f} kd={:.4f} balance={:.4f}".format(
                    epoch + 1, epochs, totals["loss"] / divisor, totals["ce"] / divisor,
                    totals["kd"] / divisor, totals["balance"] / divisor,
                )
            )
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

    # ------------------------------------------------------------------
    # Few-shot risk estimation and unified expansion
    # ------------------------------------------------------------------
    def _text_prototypes(self, class_names, class_index):
        model = self.model_without_dp
        text, _ = model.inference_text_feature(
            class_names, model.template, int(class_index[0])
        )
        _, _, description = model.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH, int(class_index[0])
        )
        return text.detach(), description.detach()

    def _support_folds(self, train_loader, extract_loader):
        train_data, extract_data = train_loader.dataset, extract_loader.dataset
        if not np.array_equal(
            np.asarray(train_data.source_ids), np.asarray(extract_data.source_ids)
        ):
            raise RuntimeError("Train and extraction loaders must share source order.")
        labels = np.asarray(extract_data.labels)
        groups = [np.flatnonzero(labels == class_id) for class_id in np.unique(labels)]
        folds = min(len(group) for group in groups)
        if folds < 3:
            return []
        result = []
        for fold in range(folds):
            held_out = sorted(int(group[fold]) for group in groups)
            fitting = sorted(set(range(len(labels))) - set(held_out))

            def make_loader(dataset, indices, shuffle):
                return DataLoader(
                    Subset(dataset, indices),
                    batch_size=train_loader.batch_size,
                    shuffle=shuffle,
                    num_workers=train_loader.num_workers,
                    drop_last=False,
                    pin_memory=True,
                )

            result.append((
                make_loader(train_data, fitting, True),
                make_loader(extract_data, fitting, False),
                make_loader(extract_data, held_out, False),
            ))
        return result

    @torch.no_grad()
    @torch.no_grad()
    def _expansion_candidates(self, extract_loader, class_index, old_state):
        """Use frozen-backbone layer confusion to nominate layers for risk testing."""
        current = self.model_without_dp.build_raw_layer_prototypes(
            extract_loader, class_index
        )
        old = old_state.get("raw_layer_prototypes")
        thresholds = old_state.get("expansion_thresholds")
        if not isinstance(old, dict) or not isinstance(thresholds, dict):
            raise KeyError("Historical raw layer prototypes and base expansion thresholds are required.")
        candidate_blocks, diagnostics = [], {}
        for block_idx in self.model_without_dp.expandable_blocks():
            key = str(block_idx)
            if key not in current or key not in old or key not in thresholds:
                raise KeyError(f"Expansion controller is missing frozen-backbone statistics for block {block_idx}.")
            current_proto = F.normalize(current[key]["prototypes"].to(self.device).float(), dim=-1)
            old_proto = F.normalize(old[key]["prototypes"].to(self.device).float(), dim=-1)
            similarities = current_proto @ old_proto.t()
            max_similarity, nearest_index = similarities.max(dim=-1)
            threshold = float(thresholds[key])
            triggered = max_similarity > threshold
            current_ids = current[key]["class_ids"].to(self.device).long()
            triggered_ids = current_ids[triggered]
            diagnostics[block_idx] = {
                "threshold": threshold,
                "max_similarity": float(max_similarity.max().item()),
                "class_max_similarity": max_similarity.detach().cpu(),
                "nearest_old_class_ids": old[key]["class_ids"][nearest_index.cpu()],
                "triggered_class_ids": triggered_ids.detach().cpu(),
                "triggered": bool(triggered.any().item()),
            }
            print(
                f"=> [ExpansionTrigger][B{block_idx}] max_similarity="
                f"{diagnostics[block_idx]['max_similarity']:.5f} "
                f"threshold={threshold:.5f} triggered={int(triggered.sum().item())}/{len(triggered)}"
            )
            if bool(triggered.any().item()):
                candidate_blocks.append(int(block_idx))
        return candidate_blocks, diagnostics

    def _support_bank(self, loader, class_index, text_features, description_proto,
                      old_state):
        """Build query-exclusive novel prototypes and append the historical class bank."""
        model = self.model_without_dp
        current = model.build_leave_one_out_prototypes(loader, class_index)
        if not bool(current["loo_valid"].all().item()):
            raise ValueError("Every support query needs another source in its own class.")
        classes = current["classes"].to(self.device)
        labels = current["labels"].to(self.device)
        features = current["features"].to(self.device)
        visual_full = torch.stack([
            F.normalize(features[labels.eq(class_id)].mean(dim=0), dim=-1)
            for class_id in classes
        ])
        visual_loo = current["loo_visual_prototypes"].to(self.device)
        if bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION):
            if model.base_vision_prototype is None:
                raise RuntimeError("Base visual prototypes are unavailable for FSCIL calibration.")
            visual_full = model.soft_calibration(model.base_vision_prototype, visual_full)
            visual_loo = model.soft_calibration(model.base_vision_prototype, visual_loo)

        lambda_t = (
            float(self.cfg.TRAINER.BiMC.LAMBDA_T)
            if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        )
        text = F.normalize(
            (1.0 - lambda_t) * text_features.to(self.device)
            + lambda_t * description_proto.to(self.device),
            dim=-1,
        )
        beta = float(self.cfg.DATASET.BETA)
        fused_full = F.normalize(
            beta * text + (1.0 - beta) * visual_full, dim=-1
        )
        fused_loo = F.normalize(
            beta * text.unsqueeze(0) + (1.0 - beta) * visual_loo, dim=-1
        )
        return {
            "classes": classes,
            "source_ids": current["source_ids"].to(self.device),
            "fused_full": fused_full,
            "fused_loo": fused_loo,
        }

    def _incremental_logits(self, images, labels, source_ids, support_bank,
                            old_state, scale, leave_one_out=True):
        """Score current queries against every old and current class in one bank."""
        if leave_one_out:
            if source_ids is None:
                raise ValueError("LOO scoring requires query source_ids.")
            query_sources = torch.as_tensor(source_ids, device=images.device).reshape(-1)
            support_sources = support_bank["source_ids"].to(images.device).reshape(-1)
            matches = query_sources[:, None].eq(support_sources[None, :])
            if not bool(matches.any(dim=1).all().item()):
                raise KeyError("A training query is absent from the support LOO bank.")
            rows = matches.long().argmax(dim=1)
            current_prototypes = support_bank["fused_loo"][rows]
        else:
            current_prototypes = support_bank["fused_full"].unsqueeze(0).expand(
                images.size(0), -1, -1
            )

        history_prototypes = F.normalize(
            old_state["fused_proto"].to(images.device).float(), dim=-1
        )
        history_batch = history_prototypes.unsqueeze(0).expand(images.size(0), -1, -1)
        all_prototypes = torch.cat([
            history_batch, F.normalize(current_prototypes.float(), dim=-1)
        ], dim=1)
        all_class_ids = torch.cat([
            old_state["class_ids"].to(images.device).long(),
            support_bank["classes"].to(images.device).long(),
        ])
        features, moe_aux = self.model_without_dp.extract_img_feature_train(images)
        features = F.normalize(features.float(), dim=-1)
        logits = float(scale) * torch.einsum("bd,bcd->bc", features, all_prototypes)
        targets = self._targets_to_columns(labels, {"class_ids": all_class_ids})
        return logits, targets, moe_aux

    def _route_regularization(self, moe_aux, block_indices, old_state):
        visual_cfg = self._visual_moe_cfg()
        margin = float(_cfg_value(visual_cfg, "ROUTE_MARGIN", 0.1))
        uncertainty = float(_cfg_value(visual_cfg, "ROUTE_UNCERTAINTY", 1.0))
        activation_weight = float(_cfg_value(visual_cfg, "ROUTE_ACTIVATION_WEIGHT", 1.0))
        preservation_weight = float(_cfg_value(visual_cfg, "ROUTE_PRESERVATION_WEIGHT", 1.0))
        aux_by_block = dict(zip(self.model_without_dp.expandable_blocks(), moe_aux))
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        for block_idx in block_indices:
            if block_idx not in aux_by_block:
                raise RuntimeError(f"Block {block_idx} is missing MoE route diagnostics.")
            stats = old_state["layer_stats"].get(str(block_idx))
            if stats is None:
                raise KeyError(f"Historical layer statistics are missing for block {block_idx}.")
            activation, preservation = self._adapter_by_block(block_idx).route_objective(
                aux_by_block[block_idx]["cls_in"],
                stats["mean"].to(self.device),
                stats["variance"].to(self.device),
                margin=margin,
                uncertainty=uncertainty,
            )
            total = total + activation_weight * activation + preservation_weight * preservation
        return total

    def _train_candidate_set(self, task_id, block_indices, train_loader, extract_loader,
                             class_index, text, description, old_state):
        """Train newest experts and router columns against a refreshed all-class LOO bank."""
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        incremental_cfg = self.cfg.TRAINER.BiMC.INCREMENTAL
        visual_cfg = self._visual_moe_cfg()
        route_inputs = self.model_without_dp.collect_blockwise_cls_features(
            extract_loader, disable_moe=False
        )
        expert_parameters, router_parameters, optimizer_groups = [], [], []
        route_initialization = {}
        for block_idx in block_indices:
            adapter = self._adapter_by_block(block_idx)
            adapter.set_newest_trainable()
            stats = old_state["layer_stats"].get(str(block_idx))
            if stats is None or block_idx not in route_inputs:
                raise KeyError(f"Current or historical route features are missing for block {block_idx}.")
            route_initialization[block_idx] = adapter.initialize_new_route(
                route_inputs[block_idx]["cls_in"],
                stats["mean"].to(self.device),
                stats["variance"].to(self.device),
                margin=float(_cfg_value(visual_cfg, "ROUTE_MARGIN", 0.1)),
                uncertainty=float(_cfg_value(visual_cfg, "ROUTE_UNCERTAINTY", 1.0)),
                init_gap=float(_cfg_value(visual_cfg, "ROUTE_INIT_GAP", 0.02)),
                target_gap=float(_cfg_value(visual_cfg, "ROUTE_INIT_TARGET_GAP", 0.5)),
                max_weight_norm=float(_cfg_value(visual_cfg, "ROUTE_INIT_MAX_WEIGHT_NORM", 2.0)),
            )
            expert_params = list(adapter.experts[-1].parameters())
            router_params = list(adapter.router.columns[-1].parameters())
            expert_parameters.extend(expert_params)
            router_parameters.extend(router_params)
            optimizer_groups.extend([
                {"params": expert_params, "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                {"params": router_params, "lr": float(_cfg_value(
                    optim_cfg, "LR_INCREMENTAL_ROUTER", optim_cfg.LR_INCREMENTAL_EXPERT
                ))},
            ])
            print(
                f"=> [RouteInit][Task {task_id}][B{block_idx}] "
                f"bias={route_initialization[block_idx]['bias']:.5f} "
                f"current_gap={route_initialization[block_idx]['current_min_gap']:.5f} "
                f"history_gap={route_initialization[block_idx]['history_min_robust_gap']:.5f} "
                f"feasible={route_initialization[block_idx]['activation_feasible_at_init']}"
            )
        parameters = expert_parameters + router_parameters
        if not parameters:
            raise RuntimeError("Candidate training has no newly created expert parameters.")
        optimizer = optim.AdamW(
            optimizer_groups, weight_decay=float(optim_cfg.WEIGHT_DECAY)
        )
        scale = float(incremental_cfg.LOGIT_SCALE)
        history = []
        for epoch in range(int(incremental_cfg.EPOCHS)):
            self.model.train()
            total_loss, total_ce, total_route, count = 0.0, 0.0, 0.0, 0
            progress = tqdm(
                train_loader,
                desc=f"Task {task_id} Experts {list(block_indices)} Epoch {epoch + 1}",
                leave=False,
            )
            for batch in progress:
                # Rebuild LOO prototypes after every update so the classifier
                # always reflects the current shared visual encoder.
                support_bank = self._support_bank(
                    extract_loader, class_index, text, description, old_state
                )
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                optimizer.zero_grad(set_to_none=True)
                logits, targets, moe_aux = self._incremental_logits(
                    images, labels, source_ids, support_bank, old_state, scale,
                    leave_one_out=True,
                )
                loss_ce = F.cross_entropy(logits, targets)
                loss_route = self._route_regularization(
                    moe_aux, block_indices, old_state
                )
                loss = loss_ce + loss_route
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite all-class incremental objective.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                batch_count = labels.numel()
                total_loss += float(loss.item()) * batch_count
                total_ce += float(loss_ce.item()) * batch_count
                total_route += float(loss_route.item()) * batch_count
                count += batch_count
                progress.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "all_cls_ce": f"{loss_ce.item():.3f}",
                    "route": f"{loss_route.item():.3f}",
                })
            history.append({
                "epoch": epoch + 1,
                "loss": total_loss / max(1, count),
                "all_class_ce": total_ce / max(1, count),
                "route_loss": total_route / max(1, count),
            })
        for block_idx in block_indices:
            self._adapter_by_block(block_idx).freeze_all()
        return {"epochs": history, "route_initialization": route_initialization}

    @torch.no_grad()
    def _held_out_losses(self, validation_loader, support_loader, class_index,
                         text, description, old_state):
        self.model.eval()
        support_bank = self._support_bank(
            support_loader, class_index, text, description, old_state
        )
        losses = []
        scale = float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        for batch in validation_loader:
            images, labels = self.parse_batch(batch)
            logits, targets, _ = self._incremental_logits(
                images, labels, None, support_bank, old_state, scale,
                leave_one_out=False,
            )
            losses.append(F.cross_entropy(logits, targets, reduction="none").cpu().numpy())
        return np.concatenate(losses) if losses else np.empty(0, dtype=np.float64)

    @torch.no_grad()
    def _held_out_loss(self, validation_loader, support_loader, class_index,
                       text, description, old_state):
        losses = self._held_out_losses(
            validation_loader, support_loader, class_index, text, description, old_state
        )
        return float(losses.mean()) if losses.size else float("inf")

    def _candidate_preservation(self, block_indices, old_state):
        visual_cfg = self._visual_moe_cfg()
        margin = float(_cfg_value(visual_cfg, "ROUTE_MARGIN", 0.1))
        uncertainty = float(_cfg_value(visual_cfg, "ROUTE_UNCERTAINTY", 1.0))
        violations = {}
        for block_idx in block_indices:
            stats = old_state["layer_stats"][str(block_idx)]
            violation = self._adapter_by_block(block_idx).route_preservation_loss(
                stats["mean"].to(self.device),
                stats["variance"].to(self.device),
                margin=margin,
                uncertainty=uncertainty,
            )
            violations[int(block_idx)] = float(violation.detach().item())
        return violations

    @torch.no_grad()
    def _candidate_route_usage(self, loader, block_indices):
        """Measure whether each new expert is selected and contributes nonzero residual."""
        was_training = self.model.training
        self.model.eval()
        totals = {
            int(block_idx): {"selected": 0, "assignments": 0, "weight": 0.0,
                             "residual": 0.0, "samples": 0}
            for block_idx in block_indices
        }
        try:
            for batch in loader:
                images, _ = self.parse_batch(batch)
                _, aux_list = self.model_without_dp.extract_img_feature_train(images)
                aux_by_block = dict(zip(self.model_without_dp.expandable_blocks(), aux_list))
                for block_idx in block_indices:
                    adapter = self._adapter_by_block(block_idx)
                    aux = aux_by_block[block_idx]
                    expert_id = adapter.num_experts - 1
                    indices = aux["top_k_indices"]
                    weights = aux["route_weights"][:, expert_id]
                    item = totals[int(block_idx)]
                    item["selected"] += int(indices.eq(expert_id).sum().item())
                    item["assignments"] += int(indices.numel())
                    item["weight"] += float(weights.sum().item())
                    item["residual"] += float(
                        aux["expert_residual_norms"][expert_id].item()
                    ) * images.size(0)
                    item["samples"] += images.size(0)
        finally:
            self.model.train(was_training)
        result = {}
        for block_idx, item in totals.items():
            selected_rate = item["selected"] / max(1, item["assignments"])
            mean_weight = item["weight"] / max(1, item["samples"])
            residual_norm = item["residual"] / max(1, item["samples"])
            result[block_idx] = {
                "selected_rate": float(selected_rate),
                "mean_weight": float(mean_weight),
                "residual_norm": float(residual_norm),
                "active": bool(item["selected"] > 0 and residual_norm > 1e-8),
            }
        return result

    @staticmethod
    def _gain_summary(gains):
        values = np.asarray(gains, dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            return float("-inf"), float("inf")
        standard_error = (
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
        return float(values.mean()), standard_error

    def _evaluate_candidate_set(self, task_id, block_indices, folds, class_index,
                                text, description, old_state):
        """Cross-fit a temporary candidate set and return per-query all-class losses."""
        fold_losses = []
        tolerance = float(_cfg_value(
            self._visual_moe_cfg(), "ROUTE_PRESERVATION_TOLERANCE", 0.05
        ))
        for fit_train, fit_extract, validation in folds:
            added_blocks = []
            try:
                for block_idx in block_indices:
                    self.model_without_dp.add_expert(block_idx, task_id)
                    added_blocks.append(block_idx)
                self._train_candidate_set(
                    task_id, block_indices, fit_train, fit_extract, class_index,
                    text, description, old_state,
                )
                route_usage = self._candidate_route_usage(fit_extract, block_indices)
                validation_losses = self._held_out_losses(
                    validation, fit_extract, class_index, text, description, old_state
                )
                preservation = self._candidate_preservation(block_indices, old_state)
                invalid_route = any(not item["active"] for item in route_usage.values())
                invalid_preservation = any(
                    value > tolerance for value in preservation.values()
                )
                if invalid_route or invalid_preservation:
                    validation_losses = np.full_like(validation_losses, float("inf"))
                fold_losses.append(validation_losses)
            finally:
                for block_idx in reversed(added_blocks):
                    self._adapter_by_block(block_idx).remove_newest_expert()
        return np.concatenate(fold_losses) if fold_losses else np.empty(0, dtype=np.float64)

    def train_incremental_task(self, task_id, train_loader, extract_loader,
                               class_index, current_class_names, old_state):
        """Gate expansion by raw-layer confusion, then verify it by cross-fitted risk."""
        text, description = self._text_prototypes(current_class_names, class_index)
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        candidate_blocks, expansion_signals = self._expansion_candidates(
            extract_loader, class_index, old_state
        )
        if not candidate_blocks:
            return {
                "trained": False, "reason": "no_layer_confusion",
                "selected_blocks": [], "decisions": [],
                "expansion_signals": expansion_signals,
            }

        folds = self._support_folds(train_loader, extract_loader)
        if not folds:
            print("=> [Expansion] insufficient independent support sources for cross-fitted risk.")
            return {
                "trained": False, "reason": "insufficient_support_folds",
                "selected_blocks": [], "decisions": [],
                "expansion_signals": expansion_signals,
            }

        baseline_losses = np.concatenate([
            self._held_out_losses(
                validation, fit_extract, class_index, text, description, old_state
            )
            for _, fit_extract, validation in folds
        ])
        single_results, decisions = {}, []
        confidence_z = float(_cfg_value(
            self._visual_moe_cfg(), "EXPANSION_CONFIDENCE_Z", 1.0
        ))
        minimum_gain = float(_cfg_value(
            self._visual_moe_cfg(), "EXPANSION_MIN_GAIN", 1e-3
        ))
        for block_idx in candidate_blocks:
            candidate_losses = self._evaluate_candidate_set(
                task_id, [block_idx], folds, class_index, text, description, old_state
            )
            gains = baseline_losses - candidate_losses
            mean_gain, gain_se = self._gain_summary(gains)
            lower_bound = mean_gain - confidence_z * gain_se
            single_results[block_idx] = {
                "losses": candidate_losses, "gain": mean_gain,
                "gain_se": gain_se, "lower_bound": lower_bound,
            }
            decisions.append({
                "blocks": [int(block_idx)], "gain": mean_gain,
                "gain_se": gain_se, "lower_confidence_bound": lower_bound,
            })
            print(
                f"=> [ExpansionCandidate][Task {task_id}][B{block_idx}] "
                f"all-class gain={mean_gain:.5f} se={gain_se:.5f} "
                f"LCB={lower_bound:.5f}"
            )

        ordered_blocks = sorted(
            candidate_blocks,
            key=lambda block: single_results[block]["lower_bound"],
            reverse=True,
        )
        selected_blocks, selected_losses = [], baseline_losses
        for block_idx in ordered_blocks:
            candidate_set = selected_blocks + [block_idx]
            candidate_losses = (
                single_results[block_idx]["losses"] if not selected_blocks
                else self._evaluate_candidate_set(
                    task_id, candidate_set, folds, class_index, text, description, old_state
                )
            )
            gains = selected_losses - candidate_losses
            mean_gain, gain_se = self._gain_summary(gains)
            lower_bound = mean_gain - confidence_z * gain_se
            accepted = bool(np.isfinite(lower_bound) and lower_bound > minimum_gain)
            decisions.append({
                "blocks": [int(value) for value in candidate_set],
                "conditional_gain": mean_gain,
                "conditional_gain_se": gain_se,
                "conditional_lower_bound": lower_bound,
                "accepted": accepted,
            })
            if accepted:
                selected_blocks.append(block_idx)
                selected_losses = candidate_losses
                print(
                    f"=> [ExpansionAccepted][Task {task_id}] blocks={selected_blocks} "
                    f"conditional_gain={mean_gain:.5f} LCB={lower_bound:.5f}"
                )
            else:
                print(
                    f"=> [ExpansionRejected][Task {task_id}] blocks={candidate_set} "
                    f"conditional_gain={mean_gain:.5f} LCB={lower_bound:.5f}"
                )

        if not selected_blocks:
            return {
                "trained": False, "reason": "no_positive_cross_fitted_gain",
                "selected_blocks": [], "decisions": decisions,
                "baseline_loo_loss": float(baseline_losses.mean()),
                "expansion_signals": expansion_signals,
            }

        added_blocks = []
        try:
            for block_idx in selected_blocks:
                self.model_without_dp.add_expert(block_idx, task_id)
                added_blocks.append(block_idx)
            final_training = self._train_candidate_set(
                task_id, selected_blocks, train_loader, extract_loader,
                class_index, text, description, old_state,
            )
            route_usage = self._candidate_route_usage(extract_loader, selected_blocks)
            if any(not item["active"] for item in route_usage.values()):
                raise RuntimeError(f"Final training left a selected expert inactive: {route_usage}")
            preservation = self._candidate_preservation(selected_blocks, old_state)
            tolerance = float(_cfg_value(
                self._visual_moe_cfg(), "ROUTE_PRESERVATION_TOLERANCE", 0.05
            ))
            if any(value > tolerance for value in preservation.values()):
                raise RuntimeError(
                    "Final full-support training violated historical route preservation: "
                    f"{preservation}. Refusing to commit an unstable expert set."
                )
        except Exception:
            for block_idx in reversed(added_blocks):
                self._adapter_by_block(block_idx).remove_newest_expert()
            raise

        expanded = [
            {
                "block_idx": int(block_idx),
                "expert_id": int(self._adapter_by_block(block_idx).num_experts - 1),
            }
            for block_idx in selected_blocks
        ]
        return {
            "trained": True,
            "selected_blocks": [int(value) for value in selected_blocks],
            "expanded": expanded,
            "decisions": decisions,
            "expansion_signals": expansion_signals,
            "baseline_loo_loss": float(baseline_losses.mean()),
            "selected_loo_loss": float(selected_losses.mean()),
            "route_preservation": preservation,
            "route_usage": route_usage,
            "training": final_training,
        }

    # ------------------------------------------------------------------
    # Unified inference and complete FSCIL run
    # ------------------------------------------------------------------
    # Unified inference and complete FSCIL run
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference_all_seen_classes(self, task_id, classifier_state):
        self.model.eval()
        beta = float(self.cfg.DATASET.BETA)
        lambda_t = (
            float(self.cfg.TRAINER.BiMC.LAMBDA_T)
            if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        )
        image_proto = F.normalize(classifier_state["image_proto"].to(self.device).float(), dim=-1)
        text_features = classifier_state["text_features"].to(self.device).float()
        description = classifier_state["description_proto"].to(self.device).float()
        text_proto = F.normalize(
            (1.0 - lambda_t) * text_features + lambda_t * description, dim=-1
        )
        fused_proto = F.normalize(classifier_state["fused_proto"].to(self.device).float(), dim=-1)
        prototype_banks = {
            "visual": image_proto,
            "text": text_proto,
            "fused": fused_proto,
        }
        test_loader = self.data_manager.get_dataloader(
            task_id, source="test", mode="test"
        )
        logits_by_name = {name: [] for name in prototype_banks}
        all_targets = []
        for batch in tqdm(test_loader, desc=f"Eval Session {task_id}"):
            images, labels = self.parse_batch(batch)
            features = F.normalize(
                self.model_without_dp.extract_img_feature(images).float(), dim=-1
            )
            for name, prototypes in prototype_banks.items():
                logits_by_name[name].append(100.0 * features @ prototypes.t())
            all_targets.append(labels)
        targets = torch.cat(all_targets)
        variants = {
            name: self.evaluator.calc_accuracy(
                torch.cat(values), targets, task_id,
                class_ids=classifier_state["class_ids"],
            )
            for name, values in logits_by_name.items()
        }
        result = dict(variants["fused"])
        result["classifier_variants"] = variants
        result["error_flow"] = self.evaluator.error_flow(
            torch.cat(logits_by_name["fused"]), targets, task_id,
            class_ids=classifier_state["class_ids"],
        )
        print(
            "=> [UnifiedClassifier] visual={:.2f} text={:.2f} fused={:.2f}".format(
                variants["visual"]["mean_acc"],
                variants["text"]["mean_acc"],
                variants["fused"]["mean_acc"],
            )
        )
        flow = result["error_flow"]["named_rates"]
        print(
            "=> [Flow] base->current_novel={:.3f} old_novel->current_novel={:.3f} "
            "current_novel->old_novel={:.3f}".format(
                flow.get("base->current_novel", 0.0),
                flow.get("old_novel->current_novel", 0.0),
                flow.get("current_novel->old_novel", 0.0),
            )
        )
        return result

    def run(self):
        final_task = self.data_manager.num_tasks - 1
        state_list = list(self._resume_state_dict_list)
        start_task = self._resume_completed_session + 1
        visual_cfg = getattr(self.cfg.TRAINER.BiMC, "VISUAL_MOE", None)
        use_moe = visual_cfg is not None and bool(getattr(visual_cfg, "ENABLE", False))
        if start_task > final_task:
            print("=> Resume checkpoint already contains all configured FSCIL sessions.")
            return

        for task_id in range(start_task, final_task + 1):
            self.model.eval()
            class_index = self.data_manager.class_index_in_task[task_id]
            class_names = np.asarray(self.data_manager.class_names)[class_index].tolist()
            extract_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="test", accumulate_past=False
            )
            train_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="train", accumulate_past=False
            )
            print(
                f"\n=> [Session {task_id}] support_manifest={self._support_manifest_hash()}"
            )
            if use_moe:
                self._log_expert_topology(f"session{task_id}-before")

            adaptation = None
            if task_id == 0:
                task_state = self.model_without_dp.build_task_statistics(
                    class_names, extract_loader, class_index,
                    calibrate_novel_vision_proto=False,
                    is_base_session=True,
                )
                if use_moe:
                    self.train_base_task(task_state, train_loader)
                    self.model.eval()
                    task_state = self.model_without_dp.build_task_statistics(
                        class_names, extract_loader, class_index,
                        calibrate_novel_vision_proto=False,
                        is_base_session=True,
                    )
            else:
                old_state = self.merge_dicts(state_list)
                if use_moe:
                    adaptation = self.train_incremental_task(
                        task_id, train_loader, extract_loader, class_index,
                        class_names, old_state,
                    )
                self.model.eval()
                task_state = self.model_without_dp.build_task_statistics(
                    class_names, extract_loader, class_index,
                    calibrate_novel_vision_proto=bool(
                        self.cfg.TRAINER.BiMC.VISION_CALIBRATION
                    ),
                    is_base_session=False,
                )

            task_state["creation_session"] = int(task_id)
            state_list.append(task_state)
            merged_state = self.merge_dicts(state_list)
            routing = {}
            if use_moe:
                routing = self._routing_diagnostics(
                    extract_loader, f"session{task_id}-final"
                )
                self._log_expert_topology(f"session{task_id}-final")

            start_time = time.perf_counter()
            accuracy = self.inference_all_seen_classes(task_id, merged_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            print(
                f"=> Session [{task_id}], Acc: {accuracy['mean_acc']:.3f}, "
                f"Base: {accuracy['base_avg_acc']:.2f}, "
                f"Novel: {accuracy['inc_avg_acc']:.2f}, "
                f"H: {accuracy['harmonic_acc']:.2f}, time: {elapsed:.3f}s"
            )
            self.acc_list.append(round(accuracy["mean_acc"], 3))
            self.task_acc_list.append(accuracy["task_acc"])
            self.session_diagnostics.append({
                "session": int(task_id),
                "support_manifest_hash": self._support_manifest_hash(),
                "adaptation": self._cpu_copy(adaptation),
                "routing": self._cpu_copy(routing),
                "expert_counts": {
                    int(block): int(adapter.num_experts)
                    for block, adapter in self._moe_adapters()
                },
                "classifier": self._cpu_copy(accuracy),
                "evaluation_seconds": float(elapsed),
            })
            self.save_checkpoint(task_id, state_list)

        print(f"\nSession accuracies: {self.acc_list}")
        if self.acc_list:
            average_accuracy = float(np.mean(self.acc_list))
            performance_drop = float(self.acc_list[0] - self.acc_list[-1])
            print(
                f"=> [FSCIL Summary] AA={average_accuracy:.3f} "
                f"PD={performance_drop:.3f} pp Final={self.acc_list[-1]:.3f}"
            )
        for index, task_accuracy in enumerate(self.task_acc_list):
            print(f"Session {index:2d} class-group accuracy: {task_accuracy}")

    def parse_batch(self, batch):
        return (
            batch["image"].to(self.device),
            batch["label"].to(self.device, dtype=torch.long),
        )
