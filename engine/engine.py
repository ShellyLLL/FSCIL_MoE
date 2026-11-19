import copy
import hashlib
import json
import math
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
    """FSCIL runner with Session-Q90 RD detection, dynamic experts and semantic visual prototypes."""

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
        self.base_calibration_source_ids = []
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

    def _assert_moe_frozen(self, context):
        for block_idx, adapter in self._moe_adapters():
            trainable = [
                name for name, parameter in adapter.named_parameters()
                if parameter.requires_grad
            ]
            if trainable or adapter.training:
                raise RuntimeError(
                    f"MoE block {block_idx} is not fully frozen at {context}: "
                    f"trainable={trainable}, training_mode={adapter.training}."
                )

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

    def _expansion_policy(self):
        """Checkpointed contract for Base-calibrated Session-Q90/Q95 expansion."""
        moe_cfg = self._visual_moe_cfg()
        model_cfg = _cfg_value(self.cfg, "MODEL", None)
        backbone_cfg = _cfg_value(model_cfg, "BACKBONE", None)
        trainer_cfg = self.cfg.TRAINER.BiMC
        optim_cfg = _cfg_value(trainer_cfg, "OPTIM", None)
        incremental_cfg = _cfg_value(trainer_cfg, "INCREMENTAL", None)
        loss_cfg = _cfg_value(trainer_cfg, "LOSS", None)
        dataset_cfg = _cfg_value(self.cfg, "DATASET", None)
        return {
            "version": 8,
            "decision_unit": "incremental_session",
            "sample_score": "minimum_standardized_reconstruction_error_over_current_deployed_RDs",
            "threshold_refresh": "recalibrate_base_pseudo_sessions_after_every_expert_pool_change",
            "session_score": "session_q90_of_sample_scores",
            "decision": "session_q90_greater_than_base_holdout_q95_for_same_support_size",
            "threshold_quantile": float(_cfg_value(moe_cfg, "EXPANSION_THRESHOLD_QUANTILE", 0.95)),
            "calibration_groups": int(_cfg_value(moe_cfg, "EXPANSION_CALIBRATION_GROUPS", 256)),
            "expert_training_scope": "all_current_session_samples",
            "descriptor_training_scope": "source_disjoint_fit_partition_of_current_session_reference_cls",
            "descriptor_calibration_scope": "source_disjoint_heldout_partition_of_current_session_reference_cls",
            "router_learning_signal": "global_CE_plus_old_route_and_expert_residual_PD",
            "visual_moe_enabled": bool(_cfg_value(moe_cfg, "ENABLE", False)),
            "insert_blocks": [int(value) for value in _cfg_value(moe_cfg, "INSERT_BLOCKS", [])],
            "num_base_experts": int(_cfg_value(moe_cfg, "NUM_BASE_EXPERTS", 4)),
            "base_reduction": int(_cfg_value(moe_cfg, "BASE_REDUCTION", 4)),
            "incremental_reduction": int(_cfg_value(moe_cfg, "INCREMENTAL_REDUCTION", 8)),
            "descriptor_dim": int(_cfg_value(moe_cfg, "DESCRIPTOR_DIM", 64)),
            "descriptor_eps": float(_cfg_value(moe_cfg, "DESCRIPTOR_EPS", 1e-4)),
            "descriptor_buffer_size": int(_cfg_value(moe_cfg, "DESCRIPTOR_BUFFER_SIZE", 500)),
            "descriptor_initialization": "nearest_session_q90_historical_rd_plus_zero_up_low_rank_residual",
            "descriptor_residual_rank": int(_cfg_value(moe_cfg, "DESCRIPTOR_RESIDUAL_RANK", 16)),
            "descriptor_denoise_scale": float(_cfg_value(moe_cfg, "DESCRIPTOR_DENOISE_SCALE", 0.05)),
            "descriptor_residual_weight": float(_cfg_value(moe_cfg, "DESCRIPTOR_RESIDUAL_WEIGHT", 1e-3)),
            "incremental_gate_bias_init": float(_cfg_value(moe_cfg, "INCREMENTAL_GATE_BIAS_INIT", -1.0)),
            "incremental_dropout": float(_cfg_value(moe_cfg, "INCREMENTAL_DROPOUT", 0.1)),
            "base_scale_init": float(_cfg_value(moe_cfg, "BASE_SCALE_INIT", 1.0)),
            "backbone_name": str(_cfg_value(backbone_cfg, "NAME", "unknown")),
            "precision": str(_cfg_value(trainer_cfg, "PREC", "fp32")),
            "freeze_visual_backbone": bool(_cfg_value(trainer_cfg, "FREEZE_VISUAL_BACKBONE", True)),
            "freeze_text_encoder": bool(_cfg_value(trainer_cfg, "FREEZE_TEXT_ENCODER", True)),
            "semantic_temperature": float(_cfg_value(moe_cfg, "SEMANTIC_TEMPERATURE", 0.07)),
            "semantic_calibration_max_alpha": float(_cfg_value(moe_cfg, "SEMANTIC_CALIBRATION_MAX_ALPHA", 0.5)),
            "prototype_reliability_k": float(_cfg_value(moe_cfg, "PROTOTYPE_RELIABILITY_K", 5.0)),
            "incremental_training": {
                "epochs": int(_cfg_value(incremental_cfg, "EPOCHS", 6)),
                "logit_scale": float(_cfg_value(incremental_cfg, "LOGIT_SCALE", 100.0)),
                "old_suppression_weight": float(_cfg_value(incremental_cfg, "OLD_SUPPRESSION_WEIGHT", 1.0)),
                "rd_weight": float(_cfg_value(incremental_cfg, "RD_WEIGHT", 1.0)),
            },
            "optimizer": {
                "base_epochs": int(_cfg_value(optim_cfg, "BASE_EPOCHS", 50)),
                "base_descriptor_epochs": int(_cfg_value(optim_cfg, "BASE_DESCRIPTOR_EPOCHS", 8)),
                "lr_base_moe": float(_cfg_value(optim_cfg, "LR_BASE_MOE", 1e-4)),
                "lr_incremental_expert": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_EXPERT", 3e-4)),
                "lr_incremental_router": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_ROUTER", 3e-4)),
                "lr_descriptor": float(_cfg_value(optim_cfg, "LR_DESCRIPTOR", 1e-3)),
                "weight_decay": float(_cfg_value(optim_cfg, "WEIGHT_DECAY", 1e-4)),
                "betas": list(_cfg_value(optim_cfg, "BETAS", (0.9, 0.999))),
            },
            "loss": {
                "ce_weight": float(_cfg_value(loss_cfg, "CE_WEIGHT", 1.0)),
                "load_balance_weight": float(_cfg_value(loss_cfg, "LB_WEIGHT", 0.01)),
                "kd_weight": float(_cfg_value(loss_cfg, "KD_WEIGHT", 0.5)),
            },
            "data_protocol": {
                "seed": int(_cfg_value(self.cfg, "SEED", 1)),
                "num_init_classes": int(_cfg_value(dataset_cfg, "NUM_INIT_CLS", -1)),
                "num_incremental_classes": int(_cfg_value(dataset_cfg, "NUM_INC_CLS", -1)),
                "base_shot": int(_cfg_value(dataset_cfg, "NUM_BASE_SHOT", -1)),
                "incremental_shot": int(_cfg_value(dataset_cfg, "NUM_INC_SHOT", -1)),
                "support_manifest_seed": int(_cfg_value(dataset_cfg, "SUPPORT_MANIFEST_SEED", -1)),
            },
        }

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
        candidate_keys = ("description_proto", "text_features", "image_proto")
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
        if "image_proto" not in result:
            raise KeyError("Classifier state is missing calibrated visual prototypes.")
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
            "support_manifest", "moe_topology", "expansion_policy", "expansion_thresholds",
            "base_calibration_source_ids",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
        schema = int(payload["schema_version"])
        if schema != 28:
            raise ValueError(
                "This branch uses dynamic-pool Base-recalibrated Session-Q90 expansion checkpoint schema (28); "
                "older conformal/median expansion checkpoints are incompatible."
            )
        checkpoint_policy = payload.get("expansion_policy")
        if checkpoint_policy != self._expansion_policy():
            raise ValueError(
                "Checkpoint expansion policy does not match the configured Session-level "
                f"contract: {checkpoint_policy} versus {self._expansion_policy()}."
            )
        if not isinstance(payload["model_state"], dict):
            raise TypeError("Resume checkpoint model_state must be a state_dict mapping.")
        if not isinstance(payload["task_states"], list):
            raise TypeError("Resume checkpoint task_states must be a list.")
        diagnostics = payload.get("session_diagnostics", [])
        if not isinstance(diagnostics, list):
            raise TypeError("Resume checkpoint session_diagnostics must be a list.")
        completed_session = int(payload["completed_session"])
        if completed_session < 0 or completed_session >= self.data_manager.num_tasks:
            raise ValueError("Checkpoint completed_session is outside the configured FSCIL task range.")
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
        expected_topology_blocks = {
            str(block) for block in self.model_without_dp.expandable_blocks()
        }
        if set(topology) != expected_topology_blocks:
            raise ValueError(
                "Checkpoint MoE topology blocks do not match configured insertion blocks: "
                f"{set(topology)} versus {expected_topology_blocks}."
            )
        thresholds = payload.get("expansion_thresholds")
        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError("Checkpoint is missing Base-calibrated Session expansion thresholds.")
        self.expansion_thresholds = {
            int(block): {int(size): float(value) for size, value in by_size.items()}
            for block, by_size in thresholds.items()
        }
        self.base_calibration_source_ids = [
            int(value) for value in payload["base_calibration_source_ids"]
        ]
        if (not self.base_calibration_source_ids
                or len(set(self.base_calibration_source_ids)) != len(self.base_calibration_source_ids)):
            raise ValueError("Checkpoint Base calibration source ids must be non-empty and unique.")
        if any(
            size <= 0 or not np.isfinite(value)
            for by_size in self.expansion_thresholds.values()
            for size, value in by_size.items()
        ):
            raise ValueError("Checkpoint contains invalid Session expansion thresholds.")
        self.model_without_dp.rebuild_moe_topology(topology)
        self.model_without_dp.load_state_dict(payload["model_state"], strict=True)
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._assert_moe_frozen("checkpoint restore")
        checkpoint_class_ids = []
        for session_index, state in enumerate(payload["task_states"]):
            required_state = ("class_ids", "image_proto", "reference_image_proto",
                              "text_features", "description_proto", "support_count", "anchor_by_block")
            forbidden_state = ("anchor_features", "anchor_labels", "anchor_source_ids")
            retained_raw = [key for key in forbidden_state if key in state]
            if retained_raw:
                raise ValueError(
                    f"Checkpoint task state {session_index} stores forbidden historical sample features: {retained_raw}"
                )
            missing_state = [key for key in required_state if key not in state]
            if missing_state:
                raise ValueError(
                    f"Checkpoint task state {session_index} is missing calibrated fields: {missing_state}"
                )
            class_ids = torch.as_tensor(state["class_ids"], dtype=torch.long).reshape(-1)
            expected_ids = torch.as_tensor(
                self.data_manager.class_index_in_task[session_index], dtype=torch.long
            ).reshape(-1)
            if not torch.equal(class_ids.sort().values, expected_ids.sort().values):
                raise ValueError(
                    f"Checkpoint session {session_index} class ids do not match the configured FSCIL split."
                )
            checkpoint_class_ids.append(class_ids)
            support_count = torch.as_tensor(state["support_count"], dtype=torch.long).reshape(-1)
            if support_count.numel() != class_ids.numel() or (support_count <= 0).any():
                raise ValueError(f"Checkpoint session {session_index} has invalid per-class support counts.")
            for key in ("image_proto", "reference_image_proto", "text_features",
                        "description_proto"):
                prototype = torch.as_tensor(state[key]).float()
                if prototype.ndim != 2 or prototype.size(0) != class_ids.numel():
                    raise ValueError(
                        f"Checkpoint task state {session_index} has invalid {key}/class-id alignment."
                    )
                if key != "image_proto" and prototype.shape != torch.as_tensor(
                    state["image_proto"]
                ).shape:
                    raise ValueError(
                        f"Checkpoint task state {session_index} has inconsistent prototype dimensions."
                    )
                if not torch.isfinite(prototype).all():
                    raise ValueError(
                        f"Checkpoint task state {session_index} has non-finite {key}."
                    )
            anchor_by_block = state["anchor_by_block"]
            if not isinstance(anchor_by_block, dict) or not anchor_by_block:
                raise ValueError(f"Checkpoint task state {session_index} has no per-layer old-class anchors.")
            anchor_blocks = {int(block) for block in anchor_by_block}
            if bool(self._visual_moe_cfg().ENABLE) and anchor_blocks != {int(block) for block in expected_topology_blocks}:
                raise ValueError(
                    f"Checkpoint task state {session_index} anchor blocks do not match MoE topology."
                )
            for block_idx, bank in anchor_by_block.items():
                anchor_fields = ("class_ids", "route_mean", "route_var", "expert_mean", "expert_var")
                missing_anchor = [key for key in anchor_fields if key not in bank]
                if missing_anchor:
                    raise ValueError(
                        f"Checkpoint task state {session_index} block {block_idx} "
                        f"is missing anchor fields: {missing_anchor}"
                    )
                anchor_ids = torch.as_tensor(bank["class_ids"], dtype=torch.long).reshape(-1)
                if not torch.equal(anchor_ids.sort().values, class_ids.sort().values):
                    raise ValueError(
                        f"Checkpoint task state {session_index} block {block_idx} "
                        "anchor class ids do not match classifier class ids."
                    )
                route_mean = torch.as_tensor(bank["route_mean"]).float()
                route_var = torch.as_tensor(bank["route_var"]).float()
                expert_mean = torch.as_tensor(bank["expert_mean"]).float()
                expert_var = torch.as_tensor(bank["expert_var"]).float()
                expected_anchor_dim = self._adapter_by_block(int(block_idx)).d_model
                if (route_mean.ndim != 2 or route_var.shape != route_mean.shape
                        or route_mean.size(0) != class_ids.numel()
                        or route_mean.size(1) != expected_anchor_dim):
                    raise ValueError(
                        f"Checkpoint task state {session_index} block {block_idx} "
                        "has misaligned reference-route anchor tensors."
                    )
                if (expert_mean.shape != route_mean.shape or expert_var.shape != route_var.shape):
                    raise ValueError(
                        f"Checkpoint task state {session_index} block {block_idx} has misaligned expert-input anchors."
                    )
                for tensor in (route_mean, route_var, expert_mean, expert_var):
                    if not torch.isfinite(tensor).all():
                        raise ValueError(
                            f"Checkpoint task state {session_index} block {block_idx} "
                            "contains non-finite anchors."
                        )
                if (route_var < 0).any() or (expert_var < 0).any():
                    raise ValueError(
                        f"Checkpoint task state {session_index} block {block_idx} "
                        "contains negative anchor variances."
                    )
        all_checkpoint_ids = torch.cat(checkpoint_class_ids)
        if torch.unique(all_checkpoint_ids).numel() != all_checkpoint_ids.numel():
            raise ValueError("Checkpoint task states contain duplicate global class ids.")
        manifest = payload.get("support_manifest")
        if manifest is None:
            raise ValueError("Resume checkpoint support_manifest must not be null.")
        self.data_manager.load_support_manifest(manifest)
        expected_threshold_blocks = {int(block) for block in self.model_without_dp.expandable_blocks()}
        if set(self.expansion_thresholds) != expected_threshold_blocks:
            raise ValueError(
                "Checkpoint Session threshold blocks do not match configured MoE insertion blocks."
            )
        expected_support_sizes = set(self._incremental_support_sizes())
        for block_idx, by_size in self.expansion_thresholds.items():
            if set(by_size) != expected_support_sizes:
                raise ValueError(
                    f"Checkpoint block {block_idx} Session thresholds do not cover exact support sizes: "
                    f"{set(by_size)} versus {expected_support_sizes}."
                )
        base_proto = payload.get("base_enhanced_visual_prototype")
        if base_proto is None:
            raise ValueError("Checkpoint is missing the frozen Base reference visual prototype bank.")
        base_proto = torch.as_tensor(base_proto).float()
        expected_base = torch.as_tensor(
            payload["task_states"][0]["image_proto"]
        ).float()
        if base_proto.shape != expected_base.shape or not torch.isfinite(base_proto).all():
            raise ValueError("Checkpoint Base enhanced visual prototype bank has invalid shape or values.")
        if not torch.allclose(base_proto.cpu(), expected_base.cpu(), rtol=1e-5, atol=1e-6):
            raise ValueError("Checkpoint Base reference prototype bank disagrees with Base task state.")
        self.model_without_dp.base_enhanced_visual_prototype = base_proto.to(self.device)
        for block_idx, adapter in self._moe_adapters():
            adapter.assert_descriptors_ready()
            births = list(adapter.expert_birth_sessions)
            if len(births) != adapter.num_experts or any(
                int(session) < 0 or int(session) > completed_session for session in births
            ):
                raise ValueError(f"Checkpoint block {block_idx} has invalid expert birth-session metadata.")
            if any(int(session) != 0 for session in births[:adapter.base_expert_count]):
                raise ValueError(f"Checkpoint block {block_idx} has non-Base birth metadata in its Base pool.")
            incremental_births = births[adapter.base_expert_count:]
            if (len(incremental_births) != len(set(incremental_births))
                    or incremental_births != sorted(incremental_births)):
                raise ValueError(
                    f"Checkpoint block {block_idx} has duplicate or non-monotonic expansion birth sessions."
                )
        acc_list = payload.get("acc_list")
        task_acc_list = payload.get("task_acc_list")
        expected_sessions = completed_session + 1
        if (not isinstance(acc_list, list) or len(acc_list) != expected_sessions
                or not all(np.isfinite(float(value)) for value in acc_list)):
            raise ValueError("Checkpoint acc_list must contain one finite AA entry per completed session.")
        if (not isinstance(task_acc_list, list) or len(task_acc_list) != expected_sessions):
            raise ValueError("Checkpoint task_acc_list must contain one entry per completed session.")
        for session_index, values in enumerate(task_acc_list):
            if (not isinstance(values, (list, tuple))
                    or len(values) != session_index + 1
                    or not all(np.isfinite(float(value)) for value in values)):
                raise ValueError(
                    f"Checkpoint task accuracy history for session {session_index} is incomplete or non-finite."
                )
        self._resume_state_dict_list = payload.get("task_states", [])
        self.acc_list = list(acc_list)
        self.task_acc_list = [list(values) for values in task_acc_list]
        self.session_diagnostics = diagnostics
        # Thresholds were validated and installed before restoring support data.
        self._resume_completed_session = completed_session
        self._restore_rng_state(payload.get("rng_state"))
        print(f"=> Resumed session {self._resume_completed_session} from {path}")

    def save_checkpoint(self, completed_session, state_dict_list):
        if not self._checkpoint_enabled():
            return None
        self._assert_moe_frozen("checkpoint save")
        directory = self._checkpoint_dir()
        directory.mkdir(parents=True, exist_ok=True)
        topology = self.model_without_dp.export_moe_topology()
        payload = {
            "schema_version": 28,
            "completed_session": int(completed_session),
            "base_calibration_source_ids": list(self.base_calibration_source_ids),
            "expansion_thresholds": self._cpu_copy(self.expansion_thresholds),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
            "task_states": self._cpu_copy(state_dict_list),
            "base_enhanced_visual_prototype": self._cpu_copy(self.model_without_dp.base_enhanced_visual_prototype),
            "support_manifest": self.data_manager.get_support_manifest(),
            "acc_list": list(self.acc_list),
            "task_acc_list": copy.deepcopy(self.task_acc_list),
            "session_diagnostics": self._cpu_copy(self.session_diagnostics),
            "expansion_policy": self._expansion_policy(),
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

    def _train_descriptor(
        self, adapter, expert_id, values, epochs, sample_weights=None,
        calibration_values=None, calibration_weights=None,
    ):
        """Fit a Base RD or a denoising low-rank residual on a frozen historical RD anchor."""
        descriptor = adapter.descriptors[int(expert_id)]
        descriptor.float()
        values = values.detach().float()
        if values.ndim != 2 or values.size(0) < 3 or not torch.isfinite(values).all():
            raise ValueError("AE fitting requires at least three finite CLS representations.")
        weights = (
            torch.ones(values.size(0), device=values.device)
            if sample_weights is None
            else torch.as_tensor(sample_weights, device=values.device).float().reshape(-1)
        )
        if (weights.numel() != values.size(0) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not bool((weights.sum() > 0).item())):
            raise ValueError("AE sample weights must align with features and have positive finite mass.")

        if descriptor.is_anchored:
            if not bool(adapter.base_feature_stats_ready.item()):
                raise RuntimeError("Anchored RD training requires frozen Base feature statistics.")
            denoise_scale = float(_cfg_value(
                self._visual_moe_cfg(), "DESCRIPTOR_DENOISE_SCALE", 0.05
            ))
            residual_weight = float(_cfg_value(
                self._visual_moe_cfg(), "DESCRIPTOR_RESIDUAL_WEIGHT", 1e-3
            ))
            if (not np.isfinite(denoise_scale) or denoise_scale < 0
                    or not np.isfinite(residual_weight) or residual_weight < 0):
                raise ValueError("RD denoising and residual regularization weights must be finite and non-negative.")
            noise_std = adapter.base_feature_std.to(values.device).reshape(1, -1) * denoise_scale
        else:
            descriptor.set_center(values, weights)
            denoise_scale, residual_weight = 0.0, 0.0
            noise_std = None

        descriptor.set_trainable(True)
        descriptor.train()
        trainable_parameters = descriptor.trainable_parameters()
        if not trainable_parameters:
            raise RuntimeError("RD training has no trainable descriptor parameters.")
        optimizer = optim.AdamW(
            trainable_parameters,
            lr=float(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "LR_DESCRIPTOR", 1e-3)),
            weight_decay=0.0,
        )
        epochs = max(1, int(epochs))
        for _ in range(epochs):
            permutation = torch.randperm(values.size(0), device=values.device)
            for indices in permutation.split(max(3, min(256, values.size(0)))):
                optimizer.zero_grad(set_to_none=True)
                clean = values[indices]
                noisy = clean if noise_std is None else clean + torch.randn_like(clean) * noise_std
                reconstruction = descriptor(noisy)
                per_sample = (reconstruction - clean).square().mean(dim=-1)
                batch_weights = weights[indices]
                reconstruction_loss = (
                    (per_sample * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
                )
                orthogonality = descriptor.orthogonality_loss()
                residual_penalty = (
                    descriptor.residual_output(clean).square().mean()
                    if descriptor.is_anchored else clean.new_zeros(())
                )
                loss = (
                    reconstruction_loss
                    + 1e-3 * orthogonality
                    + residual_weight * residual_penalty
                )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite AE reconstruction loss.")
                loss.backward()
                optimizer.step()
                descriptor.retract_basis_()
        descriptor.eval()
        with torch.no_grad():
            if calibration_values is None:
                raise ValueError("AE reference statistics require independent held-out representations.")
            reference_values = torch.as_tensor(
                calibration_values, device=values.device
            ).float()
            minimum_calibration = 2
            if (reference_values.ndim != 2 or reference_values.size(0) < minimum_calibration
                    or not torch.isfinite(reference_values).all()):
                raise ValueError(
                    f"AE error calibration requires at least {minimum_calibration} "
                    "finite independent held-out representations."
                )
            errors = descriptor.reconstruction_error(reference_values)
            calibration_weights = torch.ones_like(errors) if calibration_weights is None else torch.as_tensor(
                calibration_weights, device=errors.device, dtype=torch.float32
            ).reshape(-1)
            if calibration_weights.numel() != errors.numel():
                raise ValueError("AE calibration responsibilities must align with held-out sources.")
            descriptor.update_reference(errors, calibration_weights)
            if int(descriptor.error_count.item()) < 2:
                raise RuntimeError("AE descriptors require at least two independent calibration errors.")
        descriptor.set_trainable(False)
        if not descriptor.is_ready():
            raise RuntimeError("AE descriptor did not reach a valid independent reference-statistics state.")
        return float(loss.item())

    @staticmethod
    def _make_calibration_mask(
        labels, source_ids, min_calibration_count=3,
        min_fit_per_class=3, min_calibration_per_class=1,
    ):
        """Hold out independent source identities with a per-class minimum."""
        labels = torch.as_tensor(labels).long().reshape(-1)
        source_ids = torch.as_tensor(source_ids, device=labels.device).reshape(-1)
        if labels.numel() == 0 or source_ids.numel() != labels.numel():
            raise ValueError("AE calibration labels and source ids must be non-empty and aligned.")
        for source_id in torch.unique(source_ids, sorted=True):
            source_labels = torch.unique(labels[source_ids.eq(source_id)])
            if source_labels.numel() != 1:
                raise ValueError(
                    f"AE calibration source id {int(source_id)} is associated with multiple classes."
                )
        min_calibration_count = int(min_calibration_count)
        min_fit_per_class = int(min_fit_per_class)
        min_calibration_per_class = int(min_calibration_per_class)
        if min_calibration_count < 1 or min_fit_per_class < 3 or min_calibration_per_class < 1:
            raise ValueError("AE calibration requires positive holdout counts and at least three fit sources per class.")
        classes = torch.unique(labels, sorted=True)
        per_class_sources, holdout_counts = {}, {}
        for class_id in classes.tolist():
            class_sources = torch.unique(source_ids[labels.eq(class_id)], sorted=True)
            capacity = int(class_sources.numel()) - min_fit_per_class
            if capacity < min_calibration_per_class:
                raise ValueError(
                    f"Class {class_id} has {class_sources.numel()} independent sources; "
                    f"requires at least {min_fit_per_class + min_calibration_per_class} "
                    "for disjoint fitting and per-class calibration."
                )
            ordered = sorted(
                (int(source_id) for source_id in class_sources.tolist()),
                key=lambda source_id: hashlib.sha256(
                    f"fscil-ae-holdout:{int(class_id)}:{source_id}".encode("utf-8")
                ).digest(),
            )
            per_class_sources[int(class_id)] = torch.as_tensor(
                ordered, device=class_sources.device, dtype=class_sources.dtype
            )
            holdout_counts[int(class_id)] = min_calibration_per_class
        heldout_total = sum(holdout_counts.values())
        while heldout_total < min_calibration_count:
            progressed = False
            for class_id in classes.tolist():
                capacity = int(per_class_sources[int(class_id)].numel()) - min_fit_per_class
                if holdout_counts[int(class_id)] < capacity:
                    holdout_counts[int(class_id)] += 1
                    heldout_total += 1
                    progressed = True
                    if heldout_total >= min_calibration_count:
                        break
            if not progressed:
                raise ValueError("Insufficient independent support sources for requested AE calibration holdout.")
        mask = torch.zeros(labels.numel(), dtype=torch.bool, device=labels.device)
        for class_id in classes.tolist():
            selected = per_class_sources[int(class_id)][:holdout_counts[int(class_id)]]
            mask |= labels.eq(int(class_id)) & torch.isin(source_ids, selected)
            fit_sources = torch.unique(source_ids[labels.eq(int(class_id)) & ~mask], sorted=True)
            if fit_sources.numel() < min_fit_per_class:
                raise RuntimeError(f"AE calibration split left too few fitting sources for class {class_id}.")
        return mask

    def _incremental_support_sizes(self):
        """Return exact total support counts for every configured incremental Session."""
        sizes = set()
        if hasattr(self, "data_manager") and hasattr(self.data_manager, "get_support_manifest"):
            manifest = self.data_manager.get_support_manifest()
            for session_key, class_map in manifest.get("sessions", {}).items():
                if int(session_key) == 0:
                    continue
                size = sum(len(source_ids) for source_ids in class_map.values())
                if size > 0:
                    sizes.add(int(size))
        if not sizes:
            dataset_cfg = self.cfg.DATASET
            configured = _cfg_value(dataset_cfg, "SESSION_SUPPORT_SIZES", [])
            sizes.update(int(value) for value in configured)
            if not sizes:
                sizes.add(int(dataset_cfg.NUM_INC_CLS) * int(dataset_cfg.NUM_INC_SHOT))
        return sorted(sizes)

    @torch.no_grad()
    def _calibrate_session_expansion_thresholds(
        self, adapter, heldout_values, heldout_labels, support_sizes, block_idx
    ):
        """Calibrate one Session-Q95 threshold per support-set size from Base holdout CLS."""
        moe_cfg = self._visual_moe_cfg()
        quantile = float(_cfg_value(moe_cfg, "EXPANSION_THRESHOLD_QUANTILE", 0.95))
        groups = int(_cfg_value(moe_cfg, "EXPANSION_CALIBRATION_GROUPS", 256))
        if not 0.0 < quantile < 1.0 or groups < 20:
            raise ValueError("Session threshold calibration requires Q in (0,1) and at least 20 groups.")
        heldout_values = torch.as_tensor(heldout_values, device=self.device).float()
        if (heldout_values.ndim != 2 or heldout_values.size(0) < 2
                or not torch.isfinite(heldout_values).all()):
            raise ValueError(f"Block {block_idx} Base holdout is insufficient for Session calibration.")
        if not support_sizes:
            raise ValueError("No incremental support-set sizes were supplied for Base threshold calibration.")
        heldout_labels = torch.as_tensor(heldout_labels, device=heldout_values.device).long().reshape(-1)
        if heldout_labels.numel() != heldout_values.size(0):
            raise ValueError("Base holdout labels must align with reference features.")
        classes = torch.unique(heldout_labels, sorted=True)
        configured_classes = int(self.cfg.DATASET.NUM_INC_CLS)
        if configured_classes < 1 or classes.numel() < configured_classes:
            raise ValueError("Base holdout has fewer classes than an incremental Session.")
        class_rows = {
            int(class_id): torch.nonzero(heldout_labels.eq(class_id), as_tuple=False).flatten().cpu()
            for class_id in classes.tolist()
        }
        scores = adapter.descriptor_scores(heldout_values, require_ready=True).min(dim=1).values
        cpu_scores = scores.detach().cpu()
        generator = torch.Generator(device="cpu").manual_seed(
            int(_cfg_value(moe_cfg, "EXPANSION_CALIBRATION_SEED", 20261119)) + int(block_idx)
        )
        thresholds = {}
        for size in support_sizes:
            size = int(size)
            quotient, remainder = divmod(size, configured_classes)
            per_class_counts = [quotient + (1 if index < remainder else 0)
                                for index in range(configured_classes)]
            if configured_classes > len(class_rows):
                raise ValueError("Base holdout has fewer classes than an incremental Session.")
            if any(
                count > min(class_rows[class_id].numel() for class_id in class_rows)
                for count in per_class_counts if count > 0
            ):
                raise ValueError(
                    f"Base holdout cannot form a class-balanced Session of {size} samples "
                    f"across {configured_classes} classes without within-Session replacement."
                )
            group_q90 = []
            class_ids = list(class_rows)
            for _ in range(groups):
                selected_class_indices = torch.randperm(
                    len(class_ids), generator=generator
                )[:configured_classes].tolist()
                selected_classes = [class_ids[index] for index in selected_class_indices]
                sampled_rows = []
                for class_position, class_id in enumerate(selected_classes):
                    count = per_class_counts[class_position]
                    if count:
                        rows = class_rows[class_id]
                        order = torch.randperm(rows.numel(), generator=generator)[:count]
                        sampled_rows.extend(rows[order].tolist())
                indices = torch.as_tensor(sampled_rows, dtype=torch.long)
                group_q90.append(torch.quantile(cpu_scores[indices], 0.90))
            thresholds[int(size)] = float(torch.quantile(
                torch.stack(group_q90), quantile
            ).item())
        print(
            f"=> [Session-Threshold][B{block_idx}] "
            + " ".join(f"N={size}:Q95={value:.6f}" for size, value in sorted(thresholds.items()))
        )
        return thresholds

    def _train_base_descriptors(self, extract_loader):
        """Fit Base RDs and calibrate support-size-specific Session thresholds.

        Each descriptor is fitted on Base reference CLS inputs weighted by its
        deployed Base-Softmax responsibility. A disjoint source-level holdout
        estimates standardized reconstruction-error statistics and supplies
        Base pseudo-Sessions for the Q95 threshold calibration.
        """
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "BASE_DESCRIPTOR_EPOCHS", 8))
        print(f"\n========== Fit Base AE Distribution Descriptors ({epochs} epochs) ==========")
        feature_map = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        support_sizes = self._incremental_support_sizes()
        self.expansion_thresholds = {}
        for block_idx, adapter in self._moe_adapters():
            item = feature_map[block_idx]
            values = item["cls_in"].to(self.device)
            labels = item["labels"].to(self.device)
            source_ids = item["source_ids"].to(self.device)
            adapter.set_base_feature_stats(values)
            # Independent error-reference sources are held out from AE fitting.
            # Every Base class contributes to both partitions.
            base_class_count = int(torch.unique(labels, sorted=True).numel())
            per_class_session_capacity = max(3, max(support_sizes))
            reference_mask = self._make_calibration_mask(
                labels, source_ids,
                min_calibration_count=per_class_session_capacity * base_class_count,
                min_fit_per_class=5,
                min_calibration_per_class=per_class_session_capacity,
            )
            fit_mask = ~reference_mask
            current_holdout_ids = [int(value) for value in source_ids[reference_mask].tolist()]
            existing_holdout_ids = getattr(self, "base_calibration_source_ids", [])
            if not existing_holdout_ids:
                self.base_calibration_source_ids = current_holdout_ids
            elif existing_holdout_ids != current_holdout_ids:
                raise RuntimeError("Base calibration source identities changed across MoE blocks.")
            fit_values = values[fit_mask]
            reference_values = values[reference_mask]
            fit_labels = labels[fit_mask]
            reference_labels = labels[reference_mask]
            expected_classes = sorted(int(value) for value in torch.unique(
                labels, sorted=True
            ).tolist())
            if (sorted(int(value) for value in torch.unique(
                    fit_labels, sorted=True).tolist()) != expected_classes
                    or sorted(int(value) for value in torch.unique(
                        reference_labels, sorted=True
                    ).tolist()) != expected_classes):
                raise RuntimeError(
                    f"Block {block_idx} Base AE fit/reference partitions must cover every Base class."
                )
            with torch.no_grad():
                fit_responsibilities = torch.stack([
                    adapter.expert_responsibility(fit_values, expert_id)
                    for expert_id in range(adapter.num_experts)
                ], dim=-1)
            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(
                    adapter, expert_id, fit_values, epochs,
                    fit_responsibilities[:, expert_id],
                    calibration_values=reference_values,
                )
                descriptor = adapter.descriptors[expert_id]
                if int(descriptor.error_count.item()) < 2:
                    raise RuntimeError(
                        "Base descriptor requires at least two independent reconstruction-error references."
                    )
                print(
                    f"=> [AE-Descriptor][B{block_idx}][E{expert_id}] "
                    f"loss={loss:.6f} reference_n={int(descriptor.error_count.item())} "
                    f"mean={descriptor.error_mean.item():.6f} "
                    f"std={descriptor.error_std.item():.6f}"
                )
            adapter.assert_descriptors_ready()
            adapter.freeze_all()
            self.expansion_thresholds[int(block_idx)] = self._calibrate_session_expansion_thresholds(
                adapter, reference_values, reference_labels, support_sizes, block_idx
            )

    @torch.no_grad()
    def _refresh_block_expansion_threshold(self, block_idx):
        """Recalibrate the changed deployed RD pool on the frozen Base holdout.

        The Base source identities are fixed at Base fitting and checkpointed.
        Whenever an expert is added, the minimum-over-pool statistic changes;
        therefore the old threshold is invalid and must be recomputed using
        the same Base holdout and the complete current descriptor pool.
        """
        if not self.base_calibration_source_ids:
            raise RuntimeError("Cannot refresh dynamic-pool thresholds without the fixed Base calibration sources.")
        if not hasattr(self, "data_manager") or not hasattr(
            self.data_manager, "get_base_source_dataloader"
        ):
            # Lightweight isolated unit fixtures have no dataset manager.
            # Production Runner always owns one and must take the calibrated path.
            if hasattr(self, "data_manager"):
                raise RuntimeError("DatasetManager lacks the Base calibration source loader.")
            return
        loader = self.data_manager.get_base_source_dataloader(
            self.base_calibration_source_ids
        )
        features = self.model_without_dp.collect_blockwise_cls_features(loader)
        item = features[int(block_idx)]
        adapter = self._adapter_by_block(block_idx)
        sizes = sorted(self.expansion_thresholds.get(int(block_idx), {}).keys())
        if not sizes:
            sizes = self._incremental_support_sizes()
        refreshed = self._calibrate_session_expansion_thresholds(
            adapter, item["cls_in"], item["labels"], sizes, int(block_idx)
        )
        self.expansion_thresholds[int(block_idx)] = refreshed
        print(
            f"=> [Session-Threshold][B{int(block_idx)}] refreshed for "
            f"{adapter.num_experts} deployed experts using fixed Base holdout "
            f"(n={len(self.base_calibration_source_ids)})."
        )

    def _collect_class_anchors(self, loader):
        """Store compact per-class anchors in both stable route and adapted expert-input spaces."""
        feature_map = self.model_without_dp.collect_blockwise_cls_features(loader)
        anchors = {}
        for block_idx, item in feature_map.items():
            route_values = item["cls_in"].float()
            labels = item["labels"].long()
            route_means, route_variances = [], []
            class_ids = []
            for class_id in torch.unique(labels, sorted=True):
                mask = labels.eq(class_id)
                route_group = route_values[mask]
                route_means.append(route_group.mean(0))
                route_variances.append(route_group.var(0, unbiased=False).clamp_min(1e-6))
                class_ids.append(int(class_id))
            route_mean = torch.stack(route_means).cpu().half()
            route_var = torch.stack(route_variances).cpu().half()
            anchors[int(block_idx)] = {
                "class_ids": torch.tensor(class_ids, dtype=torch.long),
                "route_mean": route_mean,
                "route_var": route_var,
                "expert_mean": torch.stack([
                    item["expert_in"].float()[labels.eq(class_id)].mean(0)
                    for class_id in torch.unique(labels, sorted=True)
                ]).cpu().half(),
                "expert_var": torch.stack([
                    item["expert_in"].float()[labels.eq(class_id)].var(0, unbiased=False).clamp_min(1e-6)
                    for class_id in torch.unique(labels, sorted=True)
                ]).cpu().half(),
                # Keep the old names as read-only aliases for diagnostics.
                "mean": route_mean,
                "var": route_var,
            }
        return anchors

    def _all_old_anchor_sets(self, old_states, block_idx):
        route_means, route_variances, expert_means, expert_variances = [], [], [], []
        for state in old_states:
            bank = state.get("anchor_by_block", {}).get(int(block_idx))
            if bank is None:
                continue
            route_mean = bank.get("route_mean", bank.get("mean"))
            route_var = bank.get("route_var", bank.get("var"))
            if any(value is None for value in (route_mean, route_var)):
                raise ValueError(f"Block {block_idx} old-class anchors must include reference route statistics.")
            route_means.append(torch.as_tensor(route_mean, device=self.device).float())
            route_variances.append(torch.as_tensor(route_var, device=self.device).float())
            expert_mean, expert_var = bank.get("expert_mean"), bank.get("expert_var")
            if expert_mean is None or expert_var is None:
                raise ValueError(f"Block {block_idx} old-class anchors lack adapted expert-input statistics.")
            expert_means.append(torch.as_tensor(expert_mean, device=self.device).float())
            expert_variances.append(torch.as_tensor(expert_var, device=self.device).float())
        if not route_means:
            return {
                "route_mean": None, "route_var": None,
                "expert_mean": None, "expert_var": None,
            }
        return {
            "route_mean": torch.cat(route_means, dim=0),
            "route_var": torch.cat(route_variances, dim=0),
            "expert_mean": torch.cat(expert_means, dim=0),
            "expert_var": torch.cat(expert_variances, dim=0),
        }

    def _all_old_anchors(self, old_states, block_idx):
        anchors = self._all_old_anchor_sets(old_states, block_idx)
        return anchors["route_mean"], anchors["route_var"]

    def _semantic_guided_visual_prototype_calibration(
        self, visual, semantic, support_count, old_states,
    ):
        """Calibrate novel visual prototypes using Base text semantics only.

        The description embedding retrieves related Base adapted visual
        prototypes. Text features do not form a classifier and are never
        fused into visual prototype vectors.
        """
        visual = F.normalize(torch.as_tensor(visual, device=self.device).float(), dim=-1)
        semantic = F.normalize(torch.as_tensor(semantic, device=self.device).float(), dim=-1)
        support_count = torch.as_tensor(
            support_count, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if visual.ndim != 2 or semantic.shape != visual.shape:
            raise ValueError("Novel visual and description-semantic prototypes must align.")
        if support_count.numel() != visual.size(0) or (support_count <= 0).any():
            raise ValueError("Per-class support counts must be positive and align with novel prototypes.")
        if not torch.isfinite(visual).all() or not torch.isfinite(semantic).all():
            raise ValueError("Novel visual and semantic prototypes must be finite.")
        if not old_states:
            diagnostics = {
                "alpha": torch.zeros(visual.size(0), device=self.device),
                "semantic_confidence": torch.zeros(visual.size(0), device=self.device),
                "visual_prototype_reliability": torch.ones(visual.size(0), device=self.device),
            }
            return visual, diagnostics

        base_state = old_states[0]
        base_visual = F.normalize(
            torch.as_tensor(base_state["image_proto"], device=self.device).float(), dim=-1
        )
        base_semantic = F.normalize(
            torch.as_tensor(base_state["description_proto"], device=self.device).float(), dim=-1
        )
        if (base_visual.ndim != 2 or base_semantic.shape != base_visual.shape
                or base_visual.size(0) == 0 or base_visual.size(1) != visual.size(1)):
            raise ValueError("Base enhanced visual and description-semantic banks must align.")

        moe_cfg = self._visual_moe_cfg()
        temperature = float(getattr(moe_cfg, "SEMANTIC_TEMPERATURE", 0.07))
        alpha_max = float(getattr(moe_cfg, "SEMANTIC_CALIBRATION_MAX_ALPHA", 0.5))
        reliability_k = float(getattr(moe_cfg, "PROTOTYPE_RELIABILITY_K", 5.0))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("SEMANTIC_TEMPERATURE must be finite and positive.")
        if not np.isfinite(alpha_max) or not 0.0 <= alpha_max <= 1.0:
            raise ValueError("SEMANTIC_CALIBRATION_MAX_ALPHA must be in [0, 1].")
        if not np.isfinite(reliability_k) or reliability_k <= 0:
            raise ValueError("PROTOTYPE_RELIABILITY_K must be finite and positive.")

        # q_c = T(Description(c)); semantic similarity alone retrieves Base
        # visual directions. No visual-neighbor weighting or text intra-modal
        # prompt/description fusion is performed.
        semantic_weights = torch.softmax((semantic @ base_semantic.t()) / temperature, dim=-1)
        guided_visual = F.normalize(semantic_weights @ base_visual, dim=-1)
        if base_visual.size(0) == 1:
            semantic_confidence = torch.ones(visual.size(0), device=self.device)
        else:
            entropy = -(
                semantic_weights.clamp_min(1e-8)
                * semantic_weights.clamp_min(1e-8).log()
            ).sum(dim=-1)
            semantic_confidence = (
                1.0 - entropy / math.log(float(base_visual.size(0)))
            ).clamp(0.0, 1.0)
        visual_reliability = support_count / (support_count + reliability_k)
        alpha = alpha_max * semantic_confidence * (1.0 - visual_reliability)
        calibrated = F.normalize(
            (1.0 - alpha.unsqueeze(-1)) * visual
            + alpha.unsqueeze(-1) * guided_visual,
            dim=-1,
        )
        if not torch.isfinite(calibrated).all() or (calibrated.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Semantic-guided visual prototype calibration produced invalid vectors.")
        diagnostics = {
            "alpha": alpha.detach(),
            "semantic_confidence": semantic_confidence.detach(),
            "visual_prototype_reliability": visual_reliability.detach(),
        }
        return calibrated, diagnostics

    def _build_incremental_bank(self, loader, class_index, old_states):
        """Build an uncalibrated, query-exclusive bank for expert training.

        Semantic prototype calibration is deferred until new experts have
        finished training and are frozen. The training objective therefore
        uses actual support visual prototypes rather than a post-training
        semantic correction.
        """
        if not old_states:
            raise ValueError("Incremental visual classification requires the Base classifier state.")
        cache = self.model_without_dp.build_leave_one_out_prototypes(loader, class_index)
        if not bool(cache["loo_valid"].all().item()):
            raise ValueError("Every incremental training source needs another same-class source for LOO.")
        classes = cache["classes"].to(self.device)
        labels = cache["labels"].to(self.device)
        sources = cache["source_ids"].to(self.device)
        features = cache["features"].to(self.device).float()
        novel_full = F.normalize(torch.stack([
            features[labels.eq(cls)].mean(0) for cls in classes
        ]), dim=-1)
        novel_loo = cache["loo_visual_prototypes"].to(self.device).float()

        old_visual, old_ids = [], []
        for state in old_states:
            old_visual.append(F.normalize(
                torch.as_tensor(state["image_proto"], device=self.device).float(), dim=-1
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
        lb_weight = float(_cfg_value(loss_cfg, "LB_WEIGHT", 0.01))
        num_experts = int(_cfg_value(self._visual_moe_cfg(), "NUM_BASE_EXPERTS", 4))
        optimizer = self.build_base_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        print(f"\n========== Start Unified Base MoE Training ({epochs} epochs) ==========")
        for epoch in range(epochs):
            self.model.train()
            totals = {"loss": 0.0, "ce": 0.0, "lb": 0.0, "n": 0}
            for batch in tqdm(train_loader, desc=f"Base Epoch {epoch + 1}/{epochs}"):
                images, labels = self.parse_batch(batch)
                targets = self._targets_to_columns(labels, task_stat)
                optimizer.zero_grad(set_to_none=True)
                logits, img_feat, moe_aux = self.model_without_dp.forward_train(images, task_stat)
                loss_ce = F.cross_entropy(logits, targets)
                loss_lb = compute_load_balance_loss(moe_aux, num_experts).to(images.device)
                loss = ce_weight * loss_ce + lb_weight * loss_lb
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_without_dp.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                totals["loss"] += float(loss.item())
                totals["ce"] += float(loss_ce.item())
                totals["lb"] += float(loss_lb.item())
                totals["n"] += 1
            scheduler.step()
            n = max(1, totals["n"])
            print(f"=> [Base][{epoch+1}/{epochs}] loss={totals['loss']/n:.4f} "
                  f"ce={totals['ce']/n:.4f} lb={totals['lb']/n:.4f}")
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._train_base_descriptors(extract_loader)

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
        """Session-level Q90 expansion; each new expert and RD models the full Session."""
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        incremental_cfg = self.cfg.TRAINER.BiMC.INCREMENTAL
        epochs = int(incremental_cfg.EPOCHS)
        scale = float(incremental_cfg.LOGIT_SCALE)
        # Validate the Session's class coverage and source-exclusive LOO
        # support before threshold lookup or any expert topology mutation.
        self._build_incremental_bank(extract_loader, class_index, old_states)
        if not self.expansion_thresholds:
            raise RuntimeError("Base Session-Q95 expansion thresholds have not been calibrated.")

        decisions, expanded, histories = [], [], {}
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        # The frozen reference stream is invariant to adapter residuals. This
        # full-Session scan supplies both expansion evidence and paired AE data.
        expansion_features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        expected_classes = sorted(int(value) for value in class_index)
        if not expansion_features:
            raise RuntimeError("Session-level expansion received no blockwise reference features.")
        reference_labels = sorted(int(value) for value in torch.unique(
            next(iter(expansion_features.values()))["labels"], sorted=True
        ).tolist())
        if reference_labels != expected_classes:
            raise RuntimeError(
                f"Reference support classes {reference_labels} do not match Session classes "
                f"{expected_classes}."
            )

        # Functional experts are trained shallow-to-deep because upstream
        # adapters change the adapted inputs seen by downstream expert functions.
        for block_idx in self.model_without_dp.expandable_blocks():
            decision = self.model_without_dp.plan_layer_expansion(
                block_idx,
                expansion_features,
                class_index,
                self.expansion_thresholds.get(int(block_idx), {}).get(
                    int(expansion_features[int(block_idx)]["cls_in"].size(0))
                ),
            )
            decisions.append({
                "block_idx": int(block_idx),
                "expand": bool(decision.expand),
                "session_q90": float(decision.session_q90),
                "session_q95_threshold": float(decision.threshold),
                "support_count": int(decision.support_count),
            })
            if not decision.expand:
                continue

            adapter = self._adapter_by_block(block_idx)
            local_item = expansion_features[int(block_idx)]
            local_values = local_item["cls_in"].to(self.device).float()
            local_labels = local_item["labels"].to(self.device).long()
            local_sources = local_item["source_ids"].to(self.device)
            # Source-level split: descriptor optimization and its error-scale
            # calibration are disjoint, while CE and routing still use every
            # Session support source.
            class_source_counts = [
                int(torch.unique(local_sources[local_labels.eq(class_id)]).numel())
                for class_id in torch.unique(local_labels, sorted=True).tolist()
            ]
            calibration_per_class = max(
                2, int(math.ceil(0.40 * min(class_source_counts)))
            )
            calibration_mask = self._make_calibration_mask(
                local_labels, local_sources,
                min_calibration_count=calibration_per_class * int(torch.unique(local_labels).numel()),
                min_fit_per_class=3,
                min_calibration_per_class=calibration_per_class,
            )
            descriptor_fit_mask = ~calibration_mask
            descriptor_calibration_mask = calibration_mask
            descriptor_fit_values = local_values[descriptor_fit_mask]
            descriptor_calibration_values = local_values[descriptor_calibration_mask]
            if (not descriptor_fit_values.numel() or not descriptor_calibration_values.numel()
                    or set(local_labels[descriptor_fit_mask].tolist()) != set(expected_classes)
                    or set(local_labels[descriptor_calibration_mask].tolist()) != set(expected_classes)):
                raise ValueError("New RD requires source-disjoint fit and calibration sources from every Session class.")
            anchor_expert_id, anchor_q90 = adapter.select_descriptor_anchor(local_values)
            expert_id = self.model_without_dp.add_expert(
                block_idx, task_id, anchor_expert_id=anchor_expert_id
            )
            anchor_sets = self._all_old_anchor_sets(old_states, block_idx)
            anchor_mean, anchor_var = anchor_sets["route_mean"], anchor_sets["route_var"]
            expert_anchor_mean, expert_anchor_var = (
                anchor_sets["expert_mean"], anchor_sets["expert_var"]
            )
            descriptor = adapter.descriptors[expert_id]
            descriptor.set_trainable(True)
            descriptor.train()
            descriptor_noise_scale = float(_cfg_value(
                self._visual_moe_cfg(), "DESCRIPTOR_DENOISE_SCALE", 0.05
            ))
            descriptor_residual_weight = float(_cfg_value(
                self._visual_moe_cfg(), "DESCRIPTOR_RESIDUAL_WEIGHT", 1e-3
            ))
            rd_weight = float(_cfg_value(incremental_cfg, "RD_WEIGHT", 1.0))
            if (not np.isfinite(descriptor_noise_scale) or descriptor_noise_scale < 0
                    or not np.isfinite(descriptor_residual_weight) or descriptor_residual_weight < 0
                    or not np.isfinite(rd_weight) or rd_weight < 0):
                raise ValueError("RD noise, residual and objective weights must be finite and non-negative.")
            noise_std = adapter.base_feature_std.to(self.device).reshape(1, -1) * descriptor_noise_scale
            source_to_row = {
                int(source_id): row for row, source_id in enumerate(local_sources.tolist())
            }
            fit_source_ids = set(local_sources[descriptor_fit_mask].detach().cpu().tolist())
            optimizer = optim.AdamW([
                {"params": adapter.experts[-1].parameters(),
                 "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                {"params": adapter.router.columns[-1].parameters(),
                 "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)},
                {"params": descriptor.trainable_parameters(),
                 "lr": float(_cfg_value(optim_cfg, "LR_DESCRIPTOR", 1e-3))},
            ], weight_decay=float(optim_cfg.WEIGHT_DECAY))

            loss_history, ce_history, rd_history, pd_history = [], [], [], []
            classification_samples_total = 0
            samples_per_epoch = []
            observed_train_classes = set()
            for epoch in range(epochs):
                bank = self._build_incremental_bank(
                    extract_loader, class_index, old_states)
                self.model.train()
                # Frozen historical experts stay in eval mode; only the
                # newest residual adapter uses its configured dropout.
                for _, layer_adapter in self._moe_adapters():
                    layer_adapter.eval()
                adapter.experts[-1].train()
                descriptor.train()
                total, ce_total, rd_total, pd_total, count = 0.0, 0.0, 0.0, 0.0, 0
                epoch_samples = 0
                for batch in train_loader:
                    all_images, all_labels = self.parse_batch(batch)
                    all_source_ids = self._batch_source_ids(
                        batch, all_labels.numel(), all_labels.device
                    )
                    if not torch.isin(
                        all_labels, torch.as_tensor(class_index, device=all_labels.device)
                    ).all():
                        raise ValueError(
                            f"Block {block_idx} training loader contains labels outside "
                            "the current Session."
                        )
                    observed_train_classes.update(int(value) for value in all_labels.unique().tolist())

                    optimizer.zero_grad(set_to_none=True)
                    # Every current-Session example contributes to global
                    # classification; the new router is learned through CE and PD.
                    logits, targets = self._global_incremental_logits(
                        all_images, all_labels, all_source_ids, bank, scale
                    )
                    classification = F.cross_entropy(logits, targets)
                    old_suppression = adapter.old_feature_preservation_loss(
                        anchor_mean, anchor_var,
                        expert_anchor_mean, expert_anchor_var,
                    )
                    try:
                        rd_rows = torch.as_tensor(
                            [source_to_row[int(source_id)] for source_id in all_source_ids.tolist()],
                            device=self.device, dtype=torch.long,
                        )
                    except KeyError as error:
                        raise KeyError(
                            f"Training source {error.args[0]} is absent from the Session RD feature cache."
                        ) from error
                    rd_fit_mask = torch.as_tensor(
                        [int(source_id) in fit_source_ids for source_id in all_source_ids.tolist()],
                        device=self.device, dtype=torch.bool,
                    )
                    rd_clean = local_values[rd_rows[rd_fit_mask]]
                    if rd_clean.numel():
                        rd_noisy = rd_clean + torch.randn_like(rd_clean) * noise_std
                        rd_reconstruction = descriptor(rd_noisy)
                        rd_reconstruction_loss = F.mse_loss(rd_reconstruction, rd_clean)
                        rd_residual_loss = descriptor.residual_output(rd_clean).square().mean()
                        rd_loss = rd_reconstruction_loss + descriptor_residual_weight * rd_residual_loss
                    else:
                        rd_loss = classification.new_zeros(())
                    pd_weight = float(getattr(incremental_cfg, "OLD_SUPPRESSION_WEIGHT", 1.0))
                    loss = classification + pd_weight * old_suppression + rd_weight * rd_loss
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            "Non-finite Session-level global visual-prototype incremental loss."
                        )
                    loss.backward()
                    # CE/PD gradients update the new functional expert/router;
                    # RD gradients update only the descriptor residual. Clip
                    # these objective groups independently so large CE/PD
                    # gradients cannot suppress the distribution learner.
                    torch.nn.utils.clip_grad_norm_(
                        adapter.newest_parameters(include_descriptor=False), 1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        descriptor.trainable_parameters(), 1.0
                    )
                    optimizer.step()
                    batch_size = int(all_labels.numel())
                    total += float(loss.item()) * batch_size
                    ce_total += float(classification.item()) * batch_size
                    rd_total += float(rd_loss.item()) * batch_size
                    pd_total += float(old_suppression.item()) * batch_size
                    count += batch_size
                    epoch_samples += batch_size
                    classification_samples_total += batch_size

                if count == 0:
                    raise RuntimeError(f"Block {block_idx} expanded but the current task loader was empty.")
                if sorted(observed_train_classes) != expected_classes:
                    raise RuntimeError(
                        f"Block {block_idx} training loader classes {sorted(observed_train_classes)} "
                        f"do not cover the current Session classes {expected_classes}."
                    )
                samples_per_epoch.append(int(epoch_samples))
                loss_history.append(total / count)
                ce_history.append(ce_total / count)
                rd_history.append(rd_total / count)
                pd_history.append(pd_total / count)

            # The paired RD is optimized only on the fit partition. Its
            # standardized error scale is calibrated on disjoint source IDs.
            descriptor.eval()
            with torch.no_grad():
                session_errors = descriptor.reconstruction_error(
                    descriptor_calibration_values
                )
                descriptor.update_reference(session_errors)
            adapter.freeze_all()
            # Adding an RD changes min_k z_{i,k}; the previous Base threshold
            # is no longer calibrated for this expert pool.
            self._refresh_block_expansion_threshold(block_idx)
            ae_loss = float(rd_history[-1])
            fit_class_ids = sorted(int(value) for value in torch.unique(
                local_labels[descriptor_fit_mask], sorted=True
            ).tolist())
            calibration_class_ids = sorted(int(value) for value in torch.unique(
                local_labels[descriptor_calibration_mask], sorted=True
            ).tolist())

            histories[block_idx] = {
                "expert_id": int(expert_id),
                "descriptor_anchor_id": int(anchor_expert_id),
                "descriptor_anchor_q90": [float(value) for value in anchor_q90.detach().cpu().tolist()],
                "loss": loss_history,
                "ce_loss": ce_history,
                "rd_loss": rd_history,
                "pd_loss": pd_history,
                "rd_loss_final": ae_loss,
                "session_q90": float(decision.session_q90),
                "session_q95_threshold": float(decision.threshold),
                "classification_samples_per_epoch": samples_per_epoch,
                "classification_samples_total": int(classification_samples_total),
                "rd_training_samples": int(descriptor_fit_values.size(0)),
                "rd_reference_samples": int(descriptor_calibration_values.size(0)),
                "rd_fit_source_ids": sorted(int(value) for value in local_sources[descriptor_fit_mask].tolist()),
                "rd_calibration_source_ids": sorted(int(value) for value in local_sources[descriptor_calibration_mask].tolist()),
                "rd_effective_reference_count": float(
                    adapter.descriptors[expert_id].effective_error_count.item()
                ),
                "rd_session_classes": calibration_class_ids,
                "old_anchor_gate_mean": adapter.old_anchor_response(anchor_mean, anchor_var),
                "old_anchor_residual_norm": adapter.old_anchor_residual_response(
                    anchor_mean, anchor_var, expert_anchor_mean, expert_anchor_var
                ),
            }
            expanded.append({
                "block_idx": int(block_idx),
                "expert_id": int(expert_id),
                "descriptor_anchor_id": int(anchor_expert_id),
                "descriptor_anchor_q90": [float(value) for value in anchor_q90.detach().cpu().tolist()],
                "session_q90": float(decision.session_q90),
                "session_q95_threshold": float(decision.threshold),
            })

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._assert_moe_frozen("incremental Session completion")
        return {
            "trained": bool(expanded),
            "decisions": decisions,
            "expanded": expanded,
            "training": histories,
            "encoder_version": 0,
        }

    def _build_and_refine_task_state(self, current_class_name, extract_loader, class_index,
                                     old_states):
        """Build a novel visual prototype, then apply the single semantic-guided calibration."""
        state = self.model_without_dp.build_task_statistics(
            current_class_name, extract_loader, class_index,
        )
        semantic = state["description_proto"].to(self.device).float()
        visual, calibration = self._semantic_guided_visual_prototype_calibration(
            state["image_proto"].to(self.device), semantic, state["support_count"], old_states,
        )
        state["image_proto"] = visual.detach()
        state["calibration_alpha"] = calibration["alpha"].detach().cpu()
        state["semantic_confidence"] = calibration["semantic_confidence"].detach().cpu()
        state["visual_prototype_reliability"] = calibration["visual_prototype_reliability"].detach().cpu()
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
                    )
                state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
                state["class_index"] = state["class_ids"]
                if use_moe:
                    self.train_base_task(state, train_loader, extract_loader)
                    self.model.eval()
                    state = self.model_without_dp.build_task_statistics(
                        class_names, extract_loader, class_index)
                    state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
                    state["class_index"] = state["class_ids"]
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
        prototypes = F.normalize(state_dict["image_proto"].to(self.device).float(), dim=-1)
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
