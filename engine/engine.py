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
    """FSCIL runner with class-balanced RD detection, dynamic experts and calibrated visual prototypes."""

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
        self.base_reference_source_ids = []
        self.base_reference_cls_features = {}
        self.base_reference_prefix_cache = None
        self.base_reference_feature_prototypes = None
        self.base_reference_class_ids = None
        self.base_reference_accuracy_baseline = None
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
        """Checkpointed contract for all-support RD fitting and cached Base-reference expansion."""
        moe_cfg = self._visual_moe_cfg()
        model_cfg = _cfg_value(self.cfg, "MODEL", None)
        backbone_cfg = _cfg_value(model_cfg, "BACKBONE", None)
        trainer_cfg = self.cfg.TRAINER.BiMC
        optim_cfg = _cfg_value(trainer_cfg, "OPTIM", None)
        incremental_cfg = _cfg_value(trainer_cfg, "INCREMENTAL", None)
        loss_cfg = _cfg_value(trainer_cfg, "LOSS", None)
        dataset_cfg = _cfg_value(self.cfg, "DATASET", None)
        return {
            "version": 17,
            "decision_unit": "incremental_session_per_layer",
            "sample_score": "minimum_standardized_reconstruction_error_over_current_deployed_RDs",
            "threshold_refresh": "recalibrate_from_all_accepted_session_reference_cls_after_every_successful_session",
            "session_score": "pooled_q90_of_all_session_sample_scores",
            "decision": "pooled_session_q90_greater_than_base_pseudo_session_q95_for_same_support_size",
            "prototype_training_calibration": "same_base_semantic_retrieval_for_full_and_source_exclusive_loo_visual_prototypes",
            "reuse_policy": "freeze_existing_experts_descriptors_and_router_without_parameter_updates",
            "feature_space_diagnostic": "fixed_historical_final_embedding_anchor_distillation_plus_base_reference_feature_drift_acceptance",
            "threshold_quantile": float(_cfg_value(moe_cfg, "EXPANSION_THRESHOLD_QUANTILE", 0.95)),
            "calibration_groups": int(_cfg_value(moe_cfg, "EXPANSION_CALIBRATION_GROUPS", 256)),
            "expert_training_scope": "all_current_session_samples_jointly_across_all_expanded_layers",
            "expert_input_space": "frozen_reference_vit_tokens_for_all_expert_generations",
            "gradient_clipping": "joint_all_expanded_functional_parameters_and_all_new_descriptors_norm_1.0",
            "base_descriptor_training_scope": "all_base_reference_cls_weighted_by_base_router_responsibility",
            "descriptor_training_scope": "all_current_session_augmented_cls_views",
            "descriptor_calibration_scope": "all_current_session_deterministic_reference_cls_weighted_by_deployed_expert_responsibility",
            "base_threshold_reference_scope": "deterministic_class_balanced_cached_base_reference_cls",
            "base_feature_diagnostic_scope": "cached_frozen_reference_prefix_replay_no_incremental_base_image_reload",
            "threshold_calibration_scope": "base_reference_plus_all_historical_and_current_incremental_support_cls",
            "feature_space_acceptance": "minimum_class_prototype_cosine_and_fixed_base_reference_accuracy_drop",
            "router_learning_signal": "joint_global_CE_projected_against_final_feature_and_local_residual_preservation",
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
                "feature_space_preservation_weight": float(_cfg_value(incremental_cfg, "FEATURE_SPACE_PRESERVATION_WEIGHT", 1.0)),
                "local_residual_preservation_weight": float(_cfg_value(incremental_cfg, "LOCAL_RESIDUAL_PRESERVATION_WEIGHT", 1.0)),
                "feature_anchor_batch_size": int(_cfg_value(moe_cfg, "FEATURE_SPACE_ANCHOR_BATCH_SIZE", 16)),
                "feature_anchors_per_class": int(_cfg_value(moe_cfg, "FEATURE_SPACE_ANCHORS_PER_CLASS", 2)),
                "feature_anchor_batch_size": int(_cfg_value(moe_cfg, "FEATURE_SPACE_ANCHOR_BATCH_SIZE", 16)),
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
    def _project_conflicting_gradients(primary_gradients, preservation_gradients, eps=1e-24):
        """Project a primary gradient away from a conflicting preservation gradient.

        This is the single-constraint GEM update: if <g_primary, g_preserve> < 0,
        remove the component of the primary gradient that would increase the
        preservation objective to first order. The projection is applied
        independently to each functional parameter group (expert and router).
        """
        primary_gradients = tuple(primary_gradients)
        preservation_gradients = tuple(preservation_gradients)
        if len(primary_gradients) != len(preservation_gradients):
            raise ValueError("Primary and preservation gradient groups must align.")

        reference = next(
            (value for value in primary_gradients + preservation_gradients if value is not None),
            None,
        )
        if reference is None:
            return primary_gradients, {
                "applied": False, "projection_coefficient": 0.0,
                "ce_pd_dot_before": 0.0, "ce_pd_dot_after": 0.0,
                "ce_gradient_norm": 0.0, "projected_ce_gradient_norm": 0.0,
                "preservation_gradient_norm": 0.0,
                "ce_pd_cosine_before": None, "ce_pd_cosine_after": None,
            }

        dot = torch.zeros((), device=reference.device, dtype=torch.float64)
        primary_norm_sq = torch.zeros_like(dot)
        preservation_norm_sq = torch.zeros_like(dot)
        for primary, preservation in zip(primary_gradients, preservation_gradients):
            if primary is not None:
                primary64 = primary.detach().to(dtype=torch.float64)
                primary_norm_sq += primary64.square().sum()
            if preservation is not None:
                preservation64 = preservation.detach().to(dtype=torch.float64)
                preservation_norm_sq += preservation64.square().sum()
            if primary is not None and preservation is not None:
                dot += (
                    primary.detach().to(dtype=torch.float64)
                    * preservation.detach().to(dtype=torch.float64)
                ).sum()

        applied = bool((dot < 0).item() and (preservation_norm_sq > float(eps)).item())
        coefficient = float((dot / preservation_norm_sq).item()) if applied else 0.0
        projected = []
        for primary, preservation in zip(primary_gradients, preservation_gradients):
            if primary is None:
                projected.append(None)
            elif applied and preservation is not None:
                projected.append(primary - coefficient * preservation)
            else:
                projected.append(primary)

        projected_dot = torch.zeros_like(dot)
        projected_norm_sq = torch.zeros_like(dot)
        for primary, preservation in zip(projected, preservation_gradients):
            if primary is not None:
                projected_norm_sq += primary.detach().to(dtype=torch.float64).square().sum()
            if primary is not None and preservation is not None:
                projected_dot += (
                    primary.detach().to(dtype=torch.float64)
                    * preservation.detach().to(dtype=torch.float64)
                ).sum()

        primary_norm = torch.sqrt(primary_norm_sq)
        preservation_norm = torch.sqrt(preservation_norm_sq)
        projected_norm = torch.sqrt(projected_norm_sq)
        cosine_before = (
            float((dot / (primary_norm * preservation_norm)).item())
            if bool((primary_norm > 0).item() and (preservation_norm > 0).item())
            else None
        )
        cosine_after = (
            float((projected_dot / (projected_norm * preservation_norm)).item())
            if bool((projected_norm > 0).item() and (preservation_norm > 0).item())
            else None
        )
        if any(
            value is not None and not bool(torch.isfinite(value).all().item())
            for value in projected
        ):
            raise FloatingPointError("Gradient conflict projection produced non-finite gradients.")
        return tuple(projected), {
            "applied": applied,
            "projection_coefficient": coefficient,
            "ce_pd_dot_before": float(dot.item()),
            "ce_pd_dot_after": float(projected_dot.item()),
            "ce_gradient_norm": float(primary_norm.item()),
            "projected_ce_gradient_norm": float(projected_norm.item()),
            "preservation_gradient_norm": float(preservation_norm.item()),
            "ce_pd_cosine_before": cosine_before,
            "ce_pd_cosine_after": cosine_after,
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
            "base_reference_source_ids", "base_reference_cls_features",
            "base_reference_prefix_cache", "base_reference_feature_prototypes", "base_reference_class_ids",
            "base_reference_accuracy_baseline",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
        schema = int(payload["schema_version"])
        if schema != 37:
            raise ValueError(
                "This branch uses all-support dual-view RD fitting, known-domain Session-Q90 threshold "
                "calibration and cosine/accuracy feature-space acceptance checkpoint schema (37); "
                "older expansion-policy checkpoints are incompatible."
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
        self.base_reference_source_ids = [
            int(value) for value in payload["base_reference_source_ids"]
        ]
        if (not self.base_reference_source_ids
                or len(set(self.base_reference_source_ids)) != len(self.base_reference_source_ids)):
            raise ValueError("Checkpoint Base reference source ids must be non-empty and unique.")
        raw_reference_cache = payload.get("base_reference_cls_features")
        if not isinstance(raw_reference_cache, dict):
            raise ValueError("Checkpoint is missing cached Base reference CLS features.")
        expected_reference_blocks = {int(value) for value in expected_topology_blocks}
        normalized_reference_cache = {}
        expected_base_ids = torch.as_tensor(
            payload["task_states"][0].get("class_ids", payload["task_states"][0].get("class_index")),
            dtype=torch.long,
        ).reshape(-1)
        for block_idx in expected_reference_blocks:
            cache = raw_reference_cache.get(block_idx, raw_reference_cache.get(str(block_idx)))
            if not isinstance(cache, dict):
                raise ValueError(f"Checkpoint is missing Base reference CLS cache for block {block_idx}.")
            cls_in = torch.as_tensor(cache.get("cls_in")).float()
            labels = torch.as_tensor(cache.get("labels"), dtype=torch.long).reshape(-1)
            source_ids = torch.as_tensor(cache.get("source_ids"), dtype=torch.long).reshape(-1)
            expected_sources = torch.as_tensor(self.base_reference_source_ids, dtype=torch.long)
            adapter = self._adapter_by_block(block_idx)
            if (cls_in.ndim != 2 or cls_in.size(1) != adapter.d_model
                    or cls_in.size(0) == 0 or labels.numel() != cls_in.size(0)
                    or source_ids.numel() != cls_in.size(0)
                    or not torch.isfinite(cls_in).all()
                    or torch.unique(source_ids).numel() != source_ids.numel()
                    or not torch.equal(source_ids, expected_sources)
                    or not torch.equal(torch.unique(labels, sorted=True), expected_base_ids.sort().values)):
                raise ValueError(
                    f"Checkpoint Base reference CLS cache for block {block_idx} is malformed or misaligned."
                )
            normalized_reference_cache[block_idx] = {
                "cls_in": cls_in.cpu().half(),
                "labels": labels.cpu(),
                "source_ids": source_ids.cpu(),
            }
        if set(normalized_reference_cache) != expected_reference_blocks:
            raise ValueError("Checkpoint Base reference CLS cache blocks do not match the MoE topology.")
        self.base_reference_cls_features = normalized_reference_cache
        raw_prefix_cache = payload.get("base_reference_prefix_cache")
        if raw_prefix_cache is None:
            if callable(getattr(getattr(self, "data_manager", None), "get_base_source_dataloader", None)):
                raise ValueError("Production checkpoint is missing cached Base reference prefix tokens.")
            self.base_reference_prefix_cache = None
        else:
            if not isinstance(raw_prefix_cache, dict):
                raise ValueError("Checkpoint Base reference prefix cache must be a mapping.")
            prefix_tokens = torch.as_tensor(raw_prefix_cache.get("prefix_tokens")).float()
            prefix_labels = torch.as_tensor(raw_prefix_cache.get("labels"), dtype=torch.long).reshape(-1)
            prefix_sources = torch.as_tensor(raw_prefix_cache.get("source_ids"), dtype=torch.long).reshape(-1)
            expected_sources = torch.as_tensor(self.base_reference_source_ids, dtype=torch.long)
            if (prefix_tokens.ndim != 3 or prefix_tokens.size(0) == 0
                    or prefix_labels.numel() != prefix_tokens.size(0)
                    or prefix_sources.numel() != prefix_tokens.size(0)
                    or not torch.isfinite(prefix_tokens).all()
                    or torch.unique(prefix_sources).numel() != prefix_sources.numel()
                    or not torch.equal(prefix_sources, expected_sources)
                    or not torch.equal(torch.unique(prefix_labels, sorted=True), expected_base_ids.sort().values)):
                raise ValueError("Checkpoint Base reference prefix cache is malformed or misaligned.")
            self.base_reference_prefix_cache = {
                "prefix_tokens": prefix_tokens.cpu().half(),
                "labels": prefix_labels.cpu(),
                "source_ids": prefix_sources.cpu(),
            }
        if any(
            size <= 0 or not np.isfinite(value)
            for by_size in self.expansion_thresholds.values()
            for size, value in by_size.items()
        ):
            raise ValueError("Checkpoint contains invalid Session expansion thresholds.")

        holdout_prototypes = payload["base_reference_feature_prototypes"]
        holdout_class_ids = payload["base_reference_class_ids"]
        if (holdout_prototypes is None) != (holdout_class_ids is None):
            raise ValueError("Checkpoint Base reference feature prototypes and class ids must be jointly present.")
        if holdout_prototypes is not None:
            holdout_prototypes = torch.as_tensor(holdout_prototypes).float()
            holdout_class_ids = torch.as_tensor(holdout_class_ids, dtype=torch.long).reshape(-1)
            expected_base_ids = torch.as_tensor(
                payload["task_states"][0].get("class_ids", payload["task_states"][0].get("class_index")),
                dtype=torch.long,
            ).reshape(-1)
            if (holdout_prototypes.ndim != 2
                    or holdout_prototypes.size(0) != holdout_class_ids.numel()
                    or holdout_prototypes.size(1) != int(self.model_without_dp.clip_model.visual.output_dim)
                    or not torch.isfinite(holdout_prototypes).all()
                    or not torch.equal(holdout_class_ids.sort().values, expected_base_ids.sort().values)):
                raise ValueError("Checkpoint Base reference feature anchors are malformed or misaligned.")
            self.base_reference_feature_prototypes = F.normalize(holdout_prototypes, dim=-1).cpu().half()
            self.base_reference_class_ids = holdout_class_ids.cpu()
            baseline_accuracy = payload.get("base_reference_accuracy_baseline")
            if (baseline_accuracy is None or not np.isfinite(float(baseline_accuracy))
                    or not 0.0 <= float(baseline_accuracy) <= 100.0):
                raise ValueError("Checkpoint Base reference accuracy baseline is missing or invalid.")
            self.base_reference_accuracy_baseline = float(baseline_accuracy)
        else:
            if callable(getattr(getattr(self, "data_manager", None), "get_base_source_dataloader", None)):
                raise ValueError("Production checkpoint is missing Base reference feature-space anchors.")
            self.base_reference_feature_prototypes = None
            self.base_reference_class_ids = None
            self.base_reference_accuracy_baseline = None
        self.model_without_dp.rebuild_moe_topology(topology)
        self.model_without_dp.load_state_dict(payload["model_state"], strict=True)
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._assert_moe_frozen("checkpoint restore")
        checkpoint_class_ids = []
        for session_index, state in enumerate(payload["task_states"]):
            required_state = ("class_ids", "image_proto", "reference_image_proto",
                              "text_features", "description_proto", "support_count",
                              "anchor_by_block", "final_feature_anchors",
                              "reference_cls_by_block")
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
            reference_cls_by_block = state["reference_cls_by_block"]
            if not isinstance(reference_cls_by_block, dict) or not reference_cls_by_block:
                raise ValueError(f"Checkpoint task state {session_index} has no reference CLS cache.")
            expected_reference_blocks = {int(value) for value in expected_topology_blocks}
            if {int(value) for value in reference_cls_by_block} != expected_reference_blocks:
                raise ValueError(f"Checkpoint task state {session_index} reference CLS blocks do not match MoE topology.")
            for block_idx, cache in reference_cls_by_block.items():
                cls_in = torch.as_tensor(cache.get("cls_in")).float()
                labels = torch.as_tensor(cache.get("labels"), dtype=torch.long).reshape(-1)
                source_ids = torch.as_tensor(cache.get("source_ids"), dtype=torch.long).reshape(-1)
                adapter = self._adapter_by_block(int(block_idx))
                if (cls_in.ndim != 2 or cls_in.size(1) != adapter.d_model
                        or cls_in.size(0) == 0 or labels.numel() != cls_in.size(0)
                        or source_ids.numel() != cls_in.size(0)
                        or torch.unique(source_ids).numel() != source_ids.numel()
                        or not torch.isfinite(cls_in).all()
                        or not torch.equal(torch.unique(labels, sorted=True), class_ids.sort().values)):
                    raise ValueError(
                        f"Checkpoint task state {session_index} block {block_idx} reference CLS cache is malformed."
                    )
            feature_anchors = state["final_feature_anchors"]
            if not isinstance(feature_anchors, dict):
                raise ValueError(f"Checkpoint task state {session_index} has no final feature-space anchors.")
            prefix_tokens = torch.as_tensor(feature_anchors.get("prefix_tokens"))
            final_features = torch.as_tensor(feature_anchors.get("final_features"))
            anchor_sources = torch.as_tensor(feature_anchors.get("source_ids"), dtype=torch.long).reshape(-1)
            anchor_labels = torch.as_tensor(feature_anchors.get("labels"), dtype=torch.long).reshape(-1)
            anchor_class_ids = torch.as_tensor(feature_anchors.get("class_ids"), dtype=torch.long).reshape(-1)
            if (prefix_tokens.ndim != 3 or final_features.ndim != 2
                    or prefix_tokens.size(0) != final_features.size(0)
                    or final_features.size(0) != anchor_sources.numel()
                    or anchor_labels.numel() != anchor_sources.numel()
                    or not torch.equal(torch.unique(anchor_labels, sorted=True), anchor_class_ids.sort().values)
                    or final_features.size(1) != int(torch.as_tensor(state["image_proto"]).size(1))
                    or final_features.size(0) == 0
                    or not torch.isfinite(prefix_tokens.float()).all()
                    or not torch.isfinite(final_features.float()).all()
                    or torch.unique(anchor_sources).numel() != anchor_sources.numel()):
                raise ValueError(f"Checkpoint task state {session_index} has malformed final feature anchors.")
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
            "schema_version": 37,
            "completed_session": int(completed_session),
            "base_reference_source_ids": list(self.base_reference_source_ids),
            "base_reference_cls_features": self._cpu_copy(self.base_reference_cls_features),
            "base_reference_prefix_cache": self._cpu_copy(getattr(self, "base_reference_prefix_cache", None)),
            "base_reference_feature_prototypes": self._cpu_copy(getattr(self, "base_reference_feature_prototypes", None)),
            "base_reference_class_ids": self._cpu_copy(getattr(self, "base_reference_class_ids", None)),
            "base_reference_accuracy_baseline": self._cpu_copy(getattr(self, "base_reference_accuracy_baseline", None)),
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
        """Fit a Base RD or anchored residual; reference errors may reuse support sources."""
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
            reference_values = (
                values if calibration_values is None
                else torch.as_tensor(calibration_values, device=values.device).float()
            )
            minimum_reference = 2
            if (reference_values.ndim != 2 or reference_values.size(0) < minimum_reference
                    or not torch.isfinite(reference_values).all()):
                raise ValueError(
                    f"AE error calibration requires at least {minimum_reference} "
                    "finite reference representations."
                )
            errors = descriptor.reconstruction_error(reference_values)
            if calibration_weights is None:
                reference_weights = (
                    weights if calibration_values is None else torch.ones_like(errors)
                )
            else:
                reference_weights = torch.as_tensor(
                    calibration_weights, device=errors.device, dtype=torch.float32
                ).reshape(-1)
            if reference_weights.numel() != errors.numel():
                raise ValueError("AE calibration responsibilities must align with reference sources.")
            descriptor.update_reference(errors, reference_weights)
            if int(descriptor.error_count.item()) < 2:
                raise RuntimeError("AE descriptors require at least two reference errors.")
        descriptor.set_trainable(False)
        if not descriptor.is_ready():
            raise RuntimeError("AE descriptor did not reach a valid independent reference-statistics state.")
        return float(loss.item())

    @staticmethod
    def _select_class_balanced_reference_mask(
        labels, source_ids, per_class_count=None, total_count=None,
    ):
        """Select deterministic class-balanced reference rows without excluding them from RD fitting."""
        labels = torch.as_tensor(labels).long().reshape(-1)
        source_ids = torch.as_tensor(source_ids, device=labels.device).long().reshape(-1)
        if labels.numel() == 0 or source_ids.numel() != labels.numel():
            raise ValueError("Base reference labels and source ids must be non-empty and aligned.")
        for source_id in torch.unique(source_ids, sorted=True):
            source_labels = torch.unique(labels[source_ids.eq(source_id)])
            if source_labels.numel() != 1:
                raise ValueError(
                    f"Base reference source id {int(source_id)} is associated with multiple classes."
                )
        if torch.unique(source_ids).numel() != source_ids.numel():
            raise ValueError("Base reference source ids must identify unique feature rows.")
        if (per_class_count is None) == (total_count is None):
            raise ValueError("Specify exactly one of per_class_count or total_count.")
        classes = torch.unique(labels, sorted=True)
        if per_class_count is not None:
            count_by_class = {int(class_id): int(per_class_count) for class_id in classes.tolist()}
            if int(per_class_count) < 1:
                raise ValueError("Per-class Base reference count must be positive.")
        else:
            requested = min(int(total_count), int(labels.numel()))
            if requested < int(classes.numel()):
                raise ValueError("Base reference total count must cover every represented class.")
            quotient, remainder = divmod(requested, int(classes.numel()))
            count_by_class = {
                int(class_id): quotient + (1 if index < remainder else 0)
                for index, class_id in enumerate(classes.tolist())
            }
        mask = torch.zeros(labels.numel(), dtype=torch.bool, device=labels.device)
        for class_id in classes.tolist():
            class_sources = torch.unique(source_ids[labels.eq(int(class_id))], sorted=True)
            count = count_by_class[int(class_id)]
            if class_sources.numel() < count:
                raise ValueError(
                    f"Base class {int(class_id)} has {class_sources.numel()} sources, "
                    f"but the reference protocol requires {count}."
                )
            ordered = sorted(
                (int(value) for value in class_sources.detach().cpu().tolist()),
                key=lambda source_id: hashlib.sha256(
                    f"fscil-ae-reference:{int(class_id)}:{source_id}".encode("utf-8")
                ).digest(),
            )
            selected = torch.as_tensor(
                ordered[:count], device=source_ids.device, dtype=source_ids.dtype
            )
            mask |= labels.eq(int(class_id)) & torch.isin(source_ids, selected)
        if int(mask.sum().item()) != sum(count_by_class.values()):
            raise RuntimeError("Class-balanced Base reference selection produced an invalid row count.")
        return mask

    @staticmethod
    def _base_reference_per_class(support_sizes, session_class_count):
        """Convert total Session support counts into per-class Base reference capacity.

        Threshold pseudo-Sessions contain support_size examples distributed
        across session_class_count classes. The Base descriptor split must
        reserve the largest per-class count, not the total Session count, from
        every Base class.
        """
        sizes = [int(value) for value in support_sizes]
        class_count = int(session_class_count)
        if not sizes or any(value < 1 for value in sizes):
            raise ValueError("Base Session calibration requires positive support-set sizes.")
        if class_count < 1:
            raise ValueError("Base Session calibration requires a positive Session class count.")
        return max(3, max((size + class_count - 1) // class_count for size in sizes))

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
        """Calibrate one Session-Q95 threshold per support-set size from Base reference CLS."""
        moe_cfg = self._visual_moe_cfg()
        quantile = float(_cfg_value(moe_cfg, "EXPANSION_THRESHOLD_QUANTILE", 0.95))
        groups = int(_cfg_value(moe_cfg, "EXPANSION_CALIBRATION_GROUPS", 256))
        if not 0.0 < quantile < 1.0 or groups < 20:
            raise ValueError("Session threshold calibration requires Q in (0,1) and at least 20 groups.")
        heldout_values = torch.as_tensor(heldout_values, device=self.device).float()
        if (heldout_values.ndim != 2 or heldout_values.size(0) < 2
                or not torch.isfinite(heldout_values).all()):
            raise ValueError(f"Block {block_idx} Base reference is insufficient for Session calibration.")
        if not support_sizes:
            raise ValueError("No incremental support-set sizes were supplied for Base threshold calibration.")
        heldout_labels = torch.as_tensor(heldout_labels, device=heldout_values.device).long().reshape(-1)
        if heldout_labels.numel() != heldout_values.size(0):
            raise ValueError("Base reference labels must align with reference features.")
        classes = torch.unique(heldout_labels, sorted=True)
        configured_classes = int(self.cfg.DATASET.NUM_INC_CLS)
        if configured_classes < 1 or classes.numel() < configured_classes:
            raise ValueError("Base reference has fewer classes than an incremental Session.")
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
            required_per_class = max(per_class_counts)
            eligible_classes = [
                class_id for class_id in class_rows
                if class_rows[class_id].numel() >= required_per_class
            ]
            if configured_classes > len(eligible_classes):
                raise ValueError(
                    f"Known reference cache has only {len(eligible_classes)} classes with "
                    f"at least {required_per_class} sources; cannot calibrate {configured_classes}-class Sessions."
                )
            group_scores = []
            for _ in range(groups):
                selected = torch.randperm(len(eligible_classes), generator=generator)[:configured_classes].tolist()
                selected_classes = [eligible_classes[index] for index in selected]
                session_rows = []
                for position, class_id in enumerate(selected_classes):
                    count = per_class_counts[position]
                    if count:
                        rows = class_rows[class_id]
                        order = torch.randperm(rows.numel(), generator=generator)[:count]
                        session_rows.append(cpu_scores[rows[order]])
                if len(session_rows) != configured_classes:
                    raise RuntimeError("Base pseudo-Session failed to sample every class.")
                pooled_scores = torch.cat(session_rows, dim=0)
                group_scores.append(torch.quantile(pooled_scores, 0.90))
            thresholds[int(size)] = float(torch.quantile(torch.stack(group_scores), quantile).item())
        print(
            f"=> [Session-Threshold][B{block_idx}] "
            + " ".join(f"N={size}:Q95={value:.6f}" for size, value in sorted(thresholds.items()))
        )
        return thresholds

    def _train_base_descriptors(self, train_loader, extract_loader):
        """Fit Base RDs on all augmented Base views and calibrate on all deterministic reference views."""
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "BASE_DESCRIPTOR_EPOCHS", 8))
        print(f"\n========== Fit Base AE Distribution Descriptors ({epochs} epochs) ==========")
        fit_feature_map = self.model_without_dp.collect_blockwise_cls_features(train_loader)
        reference_feature_map = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        if set(fit_feature_map) != set(reference_feature_map):
            raise RuntimeError("Base RD fit/reference feature views have different MoE blocks.")
        support_sizes = self._incremental_support_sizes()
        self.expansion_thresholds = {}
        self.base_reference_cls_features = {}
        for block_idx, adapter in self._moe_adapters():
            fit_item = fit_feature_map[block_idx]
            reference_item = reference_feature_map[block_idx]
            fit_values = fit_item["cls_in"].to(self.device).float()
            fit_labels = fit_item["labels"].to(self.device).long()
            fit_source_ids = fit_item["source_ids"].to(self.device).long()
            values = reference_item["cls_in"].to(self.device).float()
            labels = reference_item["labels"].to(self.device).long()
            source_ids = reference_item["source_ids"].to(self.device).long()
            if (set(fit_source_ids.tolist()) != set(source_ids.tolist())
                    or len(set(fit_source_ids.tolist())) != fit_source_ids.numel()
                    or len(set(source_ids.tolist())) != source_ids.numel()):
                raise RuntimeError(f"Block {block_idx} Base RD fit/reference views lost source identity alignment.")
            fit_label_by_source = {
                int(source): int(label)
                for source, label in zip(fit_source_ids.tolist(), fit_labels.tolist())
            }
            if any(fit_label_by_source.get(int(source)) != int(label)
                   for source, label in zip(source_ids.tolist(), labels.tolist())):
                raise RuntimeError(f"Block {block_idx} Base RD fit/reference views disagree on source labels.")
            adapter.set_base_feature_stats(fit_values)

            session_class_count = int(self.cfg.DATASET.NUM_INC_CLS)
            threshold_reference_per_class = self._base_reference_per_class(
                support_sizes, session_class_count
            )
            threshold_mask = self._select_class_balanced_reference_mask(
                labels, source_ids, per_class_count=threshold_reference_per_class
            )
            error_reference_count = min(
                int(adapter.descriptors[0].buffer_size), int(values.size(0))
            )
            error_reference_mask = self._select_class_balanced_reference_mask(
                labels, source_ids, total_count=error_reference_count
            )
            threshold_source_ids = [int(value) for value in source_ids[threshold_mask].tolist()]
            existing_source_ids = getattr(self, "base_reference_source_ids", [])
            if not existing_source_ids:
                self.base_reference_source_ids = threshold_source_ids
            elif existing_source_ids != threshold_source_ids:
                raise RuntimeError("Base reference source identities changed across MoE blocks.")

            threshold_values = values[threshold_mask]
            threshold_labels = labels[threshold_mask]
            self.base_reference_cls_features[int(block_idx)] = {
                "cls_in": threshold_values.detach().cpu().half(),
                "labels": threshold_labels.detach().cpu(),
                "source_ids": source_ids[threshold_mask].detach().cpu(),
            }
            error_reference_values = values[error_reference_mask]
            with torch.no_grad():
                fit_responsibilities = torch.stack([
                    adapter.expert_responsibility(fit_values, expert_id)
                    for expert_id in range(adapter.num_experts)
                ], dim=-1)
                reference_responsibilities = torch.stack([
                    adapter.expert_responsibility(error_reference_values, expert_id)
                    for expert_id in range(adapter.num_experts)
                ], dim=-1)

            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(
                    adapter, expert_id, fit_values, epochs,
                    fit_responsibilities[:, expert_id],
                    calibration_values=error_reference_values,
                    calibration_weights=reference_responsibilities[:, expert_id],
                )
                descriptor = adapter.descriptors[expert_id]
                if int(descriptor.error_count.item()) < 2:
                    raise RuntimeError("Base descriptor requires at least two weighted reference errors.")
                print(
                    f"=> [AE-Descriptor][B{block_idx}][E{expert_id}] "
                    f"loss={loss:.6f} fit_n={int(values.size(0))} "
                    f"error_reference_n={int(descriptor.error_count.item())} "
                    f"effective_n={descriptor.effective_error_count.item():.2f} "
                    f"mean={descriptor.error_mean.item():.6f} "
                    f"std={descriptor.error_std.item():.6f}"
                )
            adapter.assert_descriptors_ready()
            adapter.freeze_all()
            self.expansion_thresholds[int(block_idx)] = self._calibrate_session_expansion_thresholds(
                adapter, threshold_values, threshold_labels, support_sizes, block_idx
            )

        self._capture_base_reference_feature_anchors()

    @torch.no_grad()
    def _capture_base_reference_feature_anchors(self):
        """Cache frozen-reference ViT prefix tokens and Base feature anchors once at Base completion."""
        data_manager = getattr(self, "data_manager", None)
        loader_factory = getattr(data_manager, "get_base_source_dataloader", None)
        if not callable(loader_factory):
            return None
        if not self.base_reference_source_ids:
            raise RuntimeError("Cannot capture Base feature anchors before reference source identities are fixed.")
        loader = loader_factory(self.base_reference_source_ids)
        self.model.eval()
        prefix_rows, label_rows, source_rows = [], [], []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
            prefix = self.model_without_dp.extract_reference_prefix_tokens(images)
            prefix_rows.append(prefix.detach().cpu())
            label_rows.append(labels.detach().cpu().long())
            source_rows.append(source_ids.detach().cpu().long())
        if not prefix_rows:
            raise RuntimeError("Base reference prefix capture received an empty loader.")
        prefix_tokens = torch.cat(prefix_rows)
        labels = torch.cat(label_rows).long()
        source_ids = torch.cat(source_rows).long()
        expected_sources = torch.as_tensor(self.base_reference_source_ids, dtype=torch.long)
        if (not torch.equal(source_ids, expected_sources)
                or torch.unique(source_ids).numel() != source_ids.numel()
                or not torch.equal(torch.unique(labels, sorted=True),
                                   torch.as_tensor(
                                       self.base_reference_cls_features[next(iter(self.base_reference_cls_features))]["labels"]
                                   ).unique(sorted=True))):
            raise RuntimeError("Base reference prefix tokens lost source identity or ordering.")
        features = F.normalize(
            self.model_without_dp.extract_img_feature_from_reference_prefix(
                prefix_tokens.to(self.device)
            ).float(), dim=-1
        )
        class_ids = torch.unique(labels, sorted=True).to(self.device)
        prototypes = torch.stack([
            F.normalize(features[labels.to(self.device).eq(class_id)].mean(0), dim=-1)
            for class_id in class_ids
        ])
        if prototypes.ndim != 2 or prototypes.size(0) != class_ids.numel() or not torch.isfinite(prototypes).all():
            raise RuntimeError("Base reference feature-anchor capture produced invalid class prototypes.")
        self.base_reference_prefix_cache = {
            "prefix_tokens": prefix_tokens.contiguous().half(),
            "labels": labels.cpu(),
            "source_ids": source_ids.cpu(),
        }
        self.base_reference_feature_prototypes = prototypes.detach().cpu().half()
        self.base_reference_class_ids = class_ids.detach().cpu()
        print(
            f"=> [Feature-Space][Base-anchor] classes={class_ids.numel()} "
            f"reference_sources={len(self.base_reference_source_ids)} "
            f"cached_prefix_shape={tuple(prefix_tokens.shape)}"
        )
        return True

    @torch.no_grad()
    def _capture_base_reference_accuracy_anchor(self, base_state):
        """Measure fixed Base-reference accuracy before any incremental Session."""
        cache = getattr(self, "base_reference_prefix_cache", None)
        if cache is None:
            return None
        prefix_tokens = torch.as_tensor(cache["prefix_tokens"], device=self.device).float()
        labels = torch.as_tensor(cache["labels"], device=self.device, dtype=torch.long)
        base_ids = torch.as_tensor(base_state["class_ids"], device=self.device, dtype=torch.long).reshape(-1)
        prototypes = F.normalize(torch.as_tensor(base_state["image_proto"], device=self.device).float(), dim=-1)
        features = F.normalize(
            self.model_without_dp.extract_img_feature_from_reference_prefix(prefix_tokens).float(),
            dim=-1,
        )
        matches = labels[:, None].eq(base_ids[None, :])
        if not matches.any(dim=1).all():
            raise RuntimeError("Base reference accuracy anchor contains labels absent from Base prototypes.")
        targets = matches.long().argmax(-1)
        predictions = (features @ prototypes.t()).argmax(-1)
        accuracy = float(predictions.eq(targets).float().mean().item() * 100.0)
        if not np.isfinite(accuracy):
            raise FloatingPointError("Base reference accuracy anchor is non-finite.")
        self.base_reference_accuracy_baseline = accuracy
        print(f"=> [Feature-Space][Base-anchor] reference_accuracy={accuracy:.2f}%")
        return accuracy

    @torch.no_grad()
    def _base_reference_feature_diagnostics(self, base_state):
        """Measure Base drift by replaying cached reference prefixes; never reload Base images."""
        cache = getattr(self, "base_reference_prefix_cache", None)
        baseline = getattr(self, "base_reference_feature_prototypes", None)
        baseline_ids = getattr(self, "base_reference_class_ids", None)
        baseline_accuracy = getattr(self, "base_reference_accuracy_baseline", None)
        if cache is None or baseline is None or baseline_ids is None or baseline_accuracy is None:
            return {"available": False}
        prefix_tokens = torch.as_tensor(
            cache["prefix_tokens"], device=self.device
        ).float()
        labels = torch.as_tensor(cache["labels"], device=self.device, dtype=torch.long)
        source_ids = torch.as_tensor(cache["source_ids"], dtype=torch.long)
        if (prefix_tokens.ndim != 3 or prefix_tokens.size(0) == 0
                or labels.numel() != prefix_tokens.size(0)
                or source_ids.numel() != prefix_tokens.size(0)
                or not torch.equal(source_ids, torch.as_tensor(self.base_reference_source_ids, dtype=torch.long))):
            raise RuntimeError("Cached Base reference prefix tokens are malformed or misaligned.")
        self.model.eval()
        features = F.normalize(
            self.model_without_dp.extract_img_feature_from_reference_prefix(prefix_tokens).float(),
            dim=-1,
        )
        current_ids = torch.unique(labels, sorted=True)
        baseline_ids = torch.as_tensor(baseline_ids, device=self.device, dtype=torch.long).reshape(-1)
        if not torch.equal(current_ids.to(self.device), baseline_ids.sort().values):
            raise RuntimeError("Current Base reference classes differ from captured feature-space anchors.")
        current = torch.stack([
            F.normalize(features[labels.to(self.device).eq(class_id)].mean(0), dim=-1)
            for class_id in baseline_ids
        ])
        reference = F.normalize(torch.as_tensor(baseline, device=self.device).float(), dim=-1)
        alignment = F.cosine_similarity(reference, current, dim=-1)
        base_ids = torch.as_tensor(base_state["class_ids"], device=self.device, dtype=torch.long).reshape(-1)
        prototypes = F.normalize(torch.as_tensor(base_state["image_proto"], device=self.device).float(), dim=-1)
        matches = labels[:, None].eq(base_ids[None, :])
        if not matches.any(dim=1).all():
            raise RuntimeError("Base reference contains labels absent from the Base classifier bank.")
        targets = matches.long().argmax(-1)
        predictions = (F.normalize(features.float(), dim=-1) @ prototypes.t()).argmax(-1)
        predicted_ids = base_ids[predictions]
        class_acc = {
            int(class_id): float(
                predicted_ids[labels.to(self.device).eq(int(class_id))]
                .eq(int(class_id)).float().mean().item() * 100.0
            )
            for class_id in base_ids.tolist()
        }
        target_cosine = float(_cfg_value(self._visual_moe_cfg(), "FEATURE_SPACE_MIN_COSINE", 0.98))
        max_accuracy_drop = float(_cfg_value(
            self._visual_moe_cfg(), "FEATURE_SPACE_MAX_ACCURACY_DROP", 0.0
        ))
        if not np.isfinite(max_accuracy_drop) or max_accuracy_drop < 0:
            raise ValueError("FEATURE_SPACE_MAX_ACCURACY_DROP must be finite and non-negative.")
        current_accuracy = float(
            predictions.eq(targets.to(self.device)).float().mean().item() * 100.0
        )
        accuracy_drop = max(0.0, float(baseline_accuracy) - current_accuracy)
        within_accuracy_target = bool(accuracy_drop <= max_accuracy_drop)
        result = {
            "available": True,
            "num_classes": int(baseline_ids.numel()),
            "num_sources": int(labels.numel()),
            "prototype_cosine_mean": float(alignment.mean().item()),
            "prototype_cosine_min": float(alignment.min().item()),
            "prototype_cosine_per_class": {
                int(k): float(v) for k, v in zip(baseline_ids.tolist(), alignment.tolist())
            },
            "base_reference_accuracy_baseline": float(baseline_accuracy),
            "base_reference_accuracy": current_accuracy,
            "base_reference_accuracy_drop": accuracy_drop,
            "max_accuracy_drop": max_accuracy_drop,
            "base_reference_class_accuracy": class_acc,
            "minimum_cosine_target": target_cosine,
            "within_cosine_target": bool(float(alignment.min().item()) >= target_cosine),
            "within_accuracy_target": within_accuracy_target,
            "within_target": bool(float(alignment.min().item()) >= target_cosine) and within_accuracy_target,
        }
        print(
            f"=> [Feature-Space][Session] cosine_mean={result['prototype_cosine_mean']:.6f} "
            f"cosine_min={result['prototype_cosine_min']:.6f} "
            f"base_reference_acc={result['base_reference_accuracy']:.2f}% "
            f"accuracy_drop={result['base_reference_accuracy_drop']:.2f}pp "
            f"within_target={result['within_target']}"
        )
        return result

    @torch.no_grad()
    def _refresh_block_expansion_threshold(
        self, block_idx, current_reference_features=None, old_states=None,
    ):
        """Recalibrate a changed pool over Base and every previously accepted Session domain."""
        feature_items = []
        base_cache = getattr(self, "base_reference_cls_features", {}).get(int(block_idx))
        if base_cache is not None:
            feature_items.append(base_cache)
        for state in (old_states or []):
            cache_by_block = state.get("reference_cls_by_block", {})
            cache = cache_by_block.get(int(block_idx), cache_by_block.get(str(block_idx)))
            if cache is not None:
                feature_items.append(cache)
        if current_reference_features is not None:
            current = current_reference_features.get(int(block_idx))
            if current is not None:
                feature_items.append(current)
        if not feature_items:
            raise RuntimeError(
                f"Block {int(block_idx)} has no cached known-domain CLS features for threshold refresh."
            )

        # Base cache and Base task-state cache may overlap. Deduplicate by stable
        # source identity while rejecting inconsistent labels or feature rows.
        rows_by_source = {}
        for item in feature_items:
            values = torch.as_tensor(item["cls_in"]).float()
            labels = torch.as_tensor(item["labels"]).long().reshape(-1)
            sources = torch.as_tensor(item["source_ids"]).long().reshape(-1)
            if (values.ndim != 2 or values.size(0) == 0
                    or values.size(0) != labels.numel() or labels.numel() != sources.numel()
                    or not torch.isfinite(values).all()):
                raise ValueError("Known-domain threshold CLS cache contains malformed rows.")
            for row, source in enumerate(sources.tolist()):
                source = int(source)
                label = int(labels[row])
                if source in rows_by_source:
                    old_label, old_value = rows_by_source[source]
                    if old_label != label or not torch.allclose(
                        old_value.float(), values[row].float(), rtol=1e-3, atol=1e-3
                    ):
                        raise ValueError(
                            f"Known-domain threshold caches disagree for source {source}."
                        )
                    continue
                rows_by_source[source] = (label, values[row].detach().cpu())
        ordered_sources = sorted(rows_by_source)
        values = torch.stack([rows_by_source[source][1] for source in ordered_sources])
        labels = torch.as_tensor(
            [rows_by_source[source][0] for source in ordered_sources], dtype=torch.long
        )
        adapter = self._adapter_by_block(block_idx)
        sizes = sorted(self.expansion_thresholds.get(int(block_idx), {}).keys())
        if not sizes:
            sizes = self._incremental_support_sizes()
        refreshed = self._calibrate_session_expansion_thresholds(
            adapter, values.to(self.device), labels.to(self.device), sizes, int(block_idx)
        )
        self.expansion_thresholds[int(block_idx)] = refreshed
        print(
            f"=> [Session-Threshold][B{int(block_idx)}] refreshed for {adapter.num_experts} experts "
            f"from {len(feature_items)} accepted Session caches / {values.size(0)} unique reference rows."
        )

    @torch.no_grad()
    def _collect_final_feature_anchors(self, loader, session_id):
        """Cache a deterministic, class-balanced set of frozen-prefix tokens and final embeddings."""
        visual_cfg = self._visual_moe_cfg()
        per_class = int(_cfg_value(visual_cfg, "FEATURE_SPACE_ANCHORS_PER_CLASS", 2))
        if per_class < 1:
            raise ValueError("FEATURE_SPACE_ANCHORS_PER_CLASS must be positive.")
        prefix_rows, feature_rows, label_rows, source_rows = [], [], [], []
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                prefix = self.model_without_dp.extract_reference_prefix_tokens(images)
                features, _ = self.model_without_dp.extract_img_feature_train(images)
                prefix_rows.append(prefix.detach().cpu())
                feature_rows.append(F.normalize(features.detach().float(), dim=-1).cpu())
                label_rows.append(labels.detach().cpu())
                source_rows.append(source_ids.detach().cpu())
        finally:
            self.model.train(was_training)
        if not prefix_rows:
            raise RuntimeError("Cannot create final feature-space anchors from an empty support loader.")
        prefixes = torch.cat(prefix_rows)
        features = torch.cat(feature_rows)
        labels = torch.cat(label_rows).long()
        sources = torch.cat(source_rows).long()
        if sources.numel() != torch.unique(sources).numel():
            raise ValueError("Feature-space anchor sources must be unique within a Session.")
        selected = []
        for class_id in torch.unique(labels, sorted=True).tolist():
            rows = torch.nonzero(labels.eq(int(class_id)), as_tuple=False).flatten().tolist()
            ordered_rows = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"fscil-feature-anchor:{int(session_id)}:{int(class_id)}:{int(sources[row])}".encode("utf-8")
                ).digest(),
            )
            selected.extend(ordered_rows[:per_class])
        selected = torch.as_tensor(selected, dtype=torch.long)
        return {
            "session_id": int(session_id),
            "class_ids": torch.unique(labels, sorted=True).cpu(),
            "labels": labels[selected].clone(),
            "source_ids": sources[selected].clone(),
            "prefix_tokens": prefixes[selected].contiguous().half(),
            "final_features": features[selected].contiguous().half(),
            "anchor_count": int(selected.numel()),
        }

    def _historical_feature_anchor_batches(self, old_states):
        anchors = [state["final_feature_anchors"] for state in old_states
                   if state.get("final_feature_anchors") is not None]
        if not anchors:
            raise RuntimeError("Incremental feature-space preservation requires historical final-feature anchors.")
        prefixes = torch.cat([torch.as_tensor(item["prefix_tokens"]) for item in anchors], dim=0)
        targets = torch.cat([torch.as_tensor(item["final_features"]) for item in anchors], dim=0)
        if prefixes.size(0) != targets.size(0) or prefixes.size(0) == 0:
            raise ValueError("Historical feature-anchor prefixes and targets are empty or misaligned.")
        batch_size = int(_cfg_value(self._visual_moe_cfg(), "FEATURE_SPACE_ANCHOR_BATCH_SIZE", 16))
        if batch_size < 1:
            raise ValueError("FEATURE_SPACE_ANCHOR_BATCH_SIZE must be positive.")
        return prefixes.to(self.device), targets.to(self.device), batch_size

    def _final_feature_anchor_loss(self, old_prefix_tokens, old_final_features):
        """Distill the deployed final visual embedding; gradients reach only current new adapters."""
        current = self.model_without_dp.extract_img_feature_from_reference_prefix(old_prefix_tokens)
        current = F.normalize(current.float(), dim=-1)
        target = F.normalize(old_final_features.detach().to(self.device).float(), dim=-1)
        if current.shape != target.shape or not torch.isfinite(current).all():
            raise RuntimeError("Final feature-space replay produced invalid or misaligned embeddings.")
        return (1.0 - F.cosine_similarity(current, target, dim=-1)).mean()

    @torch.no_grad()
    def _collect_reference_cls_cache(self, loader):
        """Cache exact frozen-reference CLS rows for future known-domain threshold calibration."""
        feature_map = self.model_without_dp.collect_blockwise_cls_features(loader)
        cache = {}
        for block_idx, item in feature_map.items():
            cls_in = torch.as_tensor(item["cls_in"]).detach().cpu().half()
            labels = torch.as_tensor(item["labels"]).detach().cpu().long()
            source_ids = torch.as_tensor(item["source_ids"]).detach().cpu().long()
            if (cls_in.ndim != 2 or cls_in.size(0) == 0
                    or labels.numel() != cls_in.size(0)
                    or source_ids.numel() != cls_in.size(0)
                    or torch.unique(source_ids).numel() != source_ids.numel()
                    or not torch.isfinite(cls_in.float()).all()):
                raise ValueError(f"Block {int(block_idx)} reference CLS cache is malformed.")
            cache[int(block_idx)] = {
                "cls_in": cls_in, "labels": labels, "source_ids": source_ids,
            }
        if not cache:
            raise RuntimeError("Cannot cache reference CLS rows without configured MoE blocks.")
        return cache

    def _collect_class_anchors(self, loader):
        """Store compact per-class anchors in the shared frozen-reference input space."""
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

    def _build_session_semantic_prototype(self, class_names, class_index):
        """Build one class-aligned description bank shared by training and inference."""
        class_ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long).reshape(-1)
        if callable(getattr(self, "_text_prototypes", None)):
            _, semantic = self._text_prototypes(class_names, class_ids)
        elif hasattr(self, "data_manager"):
            encoder = getattr(self.model_without_dp, "inference_all_description_feature", None)
            gpt_path = _cfg_value(self.cfg.DATASET, "GPT_PATH", "")
            if not callable(encoder) or not gpt_path:
                raise RuntimeError("Semantic calibration requires description encoder and DATASET.GPT_PATH.")
            _, _, semantic = encoder(class_names, gpt_path, int(class_ids[0].item()))
        else:
            return None
        semantic = F.normalize(torch.as_tensor(semantic, device=self.device).float(), dim=-1)
        if semantic.ndim != 2 or semantic.size(0) != class_ids.numel() or not torch.isfinite(semantic).all():
            raise ValueError("Current Session semantic prototypes must be finite and class-aligned.")
        return semantic.detach()

    def _semantic_guided_visual_prototype_calibration(self, visual, semantic, support_count, old_states):
        """Apply the same Base-semantic calibration to full and query-exclusive LOO prototypes."""
        visual = F.normalize(torch.as_tensor(visual, device=self.device).float(), dim=-1)
        semantic = F.normalize(torch.as_tensor(semantic, device=self.device).float(), dim=-1)
        support_count = torch.as_tensor(support_count, device=self.device, dtype=torch.float32)
        if visual.ndim not in (2, 3) or semantic.ndim != 2:
            raise ValueError("Visual prototypes must be [C,D] or [N,C,D], semantic prototypes [C,D].")
        classes, dim = visual.shape[-2], visual.shape[-1]
        if semantic.shape != (classes, dim):
            raise ValueError("Novel visual and semantic prototypes must align.")
        expected_shape = (classes,) if visual.ndim == 2 else (visual.size(0), classes)
        if support_count.shape != expected_shape or (support_count <= 0).any():
            raise ValueError("Support counts must be positive and aligned with visual prototype rows.")
        if not torch.isfinite(visual).all() or not torch.isfinite(semantic).all():
            raise ValueError("Novel visual and semantic prototypes must be finite.")
        if not old_states:
            return visual, {"alpha": torch.zeros(expected_shape, device=self.device),
                            "semantic_confidence": torch.zeros(classes, device=self.device),
                            "visual_prototype_reliability": torch.ones(expected_shape, device=self.device)}
        base = old_states[0]
        base_visual = F.normalize(torch.as_tensor(base["image_proto"], device=self.device).float(), dim=-1)
        base_semantic = F.normalize(torch.as_tensor(base["description_proto"], device=self.device).float(), dim=-1)
        if base_visual.ndim != 2 or base_semantic.shape != base_visual.shape or base_visual.size(1) != dim:
            raise ValueError("Base visual and semantic prototype banks must align.")
        cfg = self._visual_moe_cfg()
        temperature = float(getattr(cfg, "SEMANTIC_TEMPERATURE", 0.07))
        alpha_max = float(getattr(cfg, "SEMANTIC_CALIBRATION_MAX_ALPHA", 0.5))
        reliability_k = float(getattr(cfg, "PROTOTYPE_RELIABILITY_K", 5.0))
        if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(alpha_max) or not 0 <= alpha_max <= 1 or not np.isfinite(reliability_k) or reliability_k <= 0:
            raise ValueError("Semantic calibration configuration is invalid.")
        weights = torch.softmax((semantic @ base_semantic.t()) / temperature, dim=-1)
        guided = F.normalize(weights @ base_visual, dim=-1)
        if base_visual.size(0) == 1:
            confidence = torch.ones(classes, device=self.device)
        else:
            entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
            confidence = (1.0 - entropy / math.log(float(base_visual.size(0)))).clamp(0, 1)
        reliability = support_count / (support_count + reliability_k)
        alpha = alpha_max * confidence * (1.0 - reliability)
        if visual.ndim == 3:
            # Reliability already has [query, class] shape; only the shared
            # semantic retrieval bank needs a leading query dimension.
            guided = guided.unsqueeze(0)
        calibrated = F.normalize((1.0 - alpha.unsqueeze(-1)) * visual + alpha.unsqueeze(-1) * guided, dim=-1)
        if not torch.isfinite(calibrated).all() or (calibrated.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Semantic-guided calibration produced invalid prototypes.")
        return calibrated, {"alpha": alpha.detach(), "semantic_confidence": confidence.detach(),
                            "visual_prototype_reliability": reliability.detach()}

    def _build_incremental_bank(self, loader, class_index, old_states, semantic_proto=None):
        """Build full and source-exclusive LOO visual banks with identical calibration."""
        if not old_states:
            raise ValueError("Incremental classification requires the Base classifier state.")
        cache = self.model_without_dp.build_leave_one_out_prototypes(loader, class_index)
        if not bool(cache["loo_valid"].all().item()):
            raise ValueError("Every incremental source requires another same-class source for LOO.")
        classes, labels = cache["classes"].to(self.device), cache["labels"].to(self.device)
        sources, features = cache["source_ids"].to(self.device), cache["features"].to(self.device).float()
        full_raw = F.normalize(torch.stack([features[labels.eq(cls)].mean(0) for cls in classes]), dim=-1)
        loo_raw = F.normalize(cache["loo_visual_prototypes"].to(self.device).float(), dim=-1)
        full_counts = torch.as_tensor([torch.unique(sources[labels.eq(cls)]).numel() for cls in classes.tolist()],
                                      device=self.device, dtype=torch.float32)
        loo_counts = torch.tensor([
            [int(torch.unique(sources[labels.eq(cls) & sources.ne(source_id)]).numel())
             for cls in classes.tolist()]
            for source_id in sources.tolist()
        ], device=self.device, dtype=torch.float32)
        if semantic_proto is not None:
            semantic_ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long).reshape(-1)
            semantic_proto = torch.as_tensor(semantic_proto, device=self.device).float()
            if semantic_proto.ndim != 2 or semantic_proto.size(0) != semantic_ids.numel():
                raise ValueError("Session semantic prototypes must align with class ids.")
            row_map = {int(cid): i for i, cid in enumerate(semantic_ids.tolist())}
            if set(row_map) != set(int(cid) for cid in classes.tolist()):
                raise ValueError("Semantic class ids do not match the LOO visual bank.")
            semantic = torch.stack([semantic_proto[row_map[int(cid)]] for cid in classes.tolist()])
            novel_full, full_diag = self._semantic_guided_visual_prototype_calibration(full_raw, semantic, full_counts, old_states)
            novel_loo, loo_diag = self._semantic_guided_visual_prototype_calibration(loo_raw, semantic, loo_counts, old_states)
        else:
            novel_full, novel_loo, full_diag, loo_diag = full_raw, loo_raw, None, None
        old_visual, old_ids = [], []
        for state in old_states:
            old_visual.append(F.normalize(torch.as_tensor(state["image_proto"], device=self.device).float(), dim=-1))
            old_ids.append(torch.as_tensor(state.get("class_ids", state.get("class_index")),
                                           device=self.device, dtype=torch.long).reshape(-1))
        old_visual, old_ids = torch.cat(old_visual), torch.cat(old_ids)
        global_full, global_ids = torch.cat([old_visual, novel_full]), torch.cat([old_ids, classes])
        if torch.unique(global_ids).numel() != global_ids.numel() or global_full.size(0) != global_ids.numel():
            raise ValueError("Global visual prototype classifier has duplicate or misaligned class ids.")
        return {"classes": classes, "features": features, "labels": labels, "source_ids": sources,
                "novel_full": novel_full, "novel_loo": novel_loo,
                "novel_full_raw": full_raw, "novel_loo_raw": loo_raw,
                "novel_support_count": full_counts, "novel_loo_support_count": loo_counts,
                "semantic_calibration_full": full_diag, "semantic_calibration_loo": loo_diag,
                "global_full": global_full, "global_ids": global_ids, "old_class_count": old_ids.numel()}

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
        self._train_base_descriptors(train_loader, extract_loader)

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, old_states):
        """Transactional wrapper: a failed session cannot leave partial experts."""
        snapshots = []
        threshold_snapshot = copy.deepcopy(self.expansion_thresholds)
        rng_snapshot = self._rng_state()
        training_modes = {id(module): bool(module.training) for module in self.model.modules()}
        for block_idx, adapter in self._moe_adapters():
            snapshots.append((
                int(block_idx), adapter.export_topology(),
                {key: value.detach().clone() for key, value in adapter.state_dict().items()},
            ))
        try:
            semantic_proto = self._build_session_semantic_prototype(current_class_name, class_index)
            result = self._train_incremental_task_impl(
                task_id, train_loader, extract_loader, class_index,
                current_class_name, old_states, semantic_proto,
            )
            # A successful optimizer run is not sufficient: the deployed
            # encoder must remain inside the Base feature-space trust region.
            feature_check = self._base_reference_feature_diagnostics(old_states[0])
            if feature_check.get("available", False) and not feature_check.get("within_target", False):
                raise RuntimeError(
                    "Session rejected: Base reference feature-space alignment fell below "
                    f"{feature_check.get('minimum_cosine_target')}; observed "
                    f"{feature_check.get('prototype_cosine_min')}."
                )
            if hasattr(self, "data_manager") and callable(
                getattr(self.data_manager, "get_base_source_dataloader", None)
            ) and not feature_check.get("available", False):
                raise RuntimeError(
                    "Session rejected: production feature-space preservation check is unavailable."
                )
            # Prototype construction and compact feature-anchor capture are
            # part of the same Session transaction, not post-commit side effects.
            result["task_state"] = self._build_and_refine_task_state(
                current_class_name, extract_loader, class_index, old_states,
                semantic_proto=semantic_proto, session_id=int(task_id),
            )
            result["feature_space_acceptance"] = self._cpu_copy(feature_check)
            return result
        except Exception:
            # Restore every layer, not just the layer where the exception arose.
            for block_idx, topology, state in snapshots:
                adapter = self._adapter_by_block(block_idx)
                adapter.rebuild_topology(topology)
                adapter.load_state_dict(state, strict=True)
                adapter.freeze_all()
            # Thresholds are part of the deployed expert-pool state. A failed
            # Session must roll back their recalibration together with topology.
            self.expansion_thresholds = threshold_snapshot
            self._restore_rng_state(rng_snapshot)
            for module in self.model.modules():
                module.training = training_modes.get(id(module), False)
            raise

    def _train_reused_router(self, block_idx, reference_features):
        """Reuse an adequate deployed expert pool without mutating historical parameters."""
        adapter = self._adapter_by_block(block_idx)
        adapter.freeze_all()
        values = torch.as_tensor(reference_features["cls_in"], device=self.device).float()
        if values.ndim != 2 or values.size(0) == 0 or not torch.isfinite(values).all():
            raise ValueError(f"Block {block_idx} REUSE requires finite frozen-reference support features.")
        with torch.no_grad():
            _, weights = adapter.routing_outputs(values)
        history = {
            "mode": "reuse_frozen", "parameter_updates": 0,
            "classification_samples_per_epoch": [], "classification_samples_total": 0,
            "rd_training_samples": 0, "rd_reference_samples": 0,
            "route_anchor_count": int(values.size(0)),
            "route_weight_mean": weights.mean(0).detach().cpu(),
            "expert_count": int(adapter.num_experts),
        }
        print(f"=> [Expert-Reuse][B{block_idx}] experts={adapter.num_experts} "
              f"support={values.size(0)} parameter_updates=0")
        return history

    def _train_incremental_task_impl(self, task_id, train_loader, extract_loader, class_index,
                                      current_class_name, old_states, semantic_proto=None):
        """Plan every layer first, then jointly optimize all expanded modules in one Session."""
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        incremental_cfg = self.cfg.TRAINER.BiMC.INCREMENTAL
        epochs = int(incremental_cfg.EPOCHS)
        scale = float(incremental_cfg.LOGIT_SCALE)
        if epochs < 1:
            raise ValueError("Incremental Session epochs must be positive.")
        if not self.expansion_thresholds:
            raise RuntimeError("Base Session-Q95 expansion thresholds have not been calibrated.")
        self._build_incremental_bank(extract_loader, class_index, old_states, semantic_proto)
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        # One immutable Session snapshot drives every layer's decision.
        expansion_features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        rd_fit_features = self.model_without_dp.collect_blockwise_cls_features(train_loader)
        if set(expansion_features) != set(rd_fit_features):
            raise RuntimeError("Incremental RD fit/reference feature views have different MoE blocks.")
        expected_classes = sorted(int(value) for value in class_index)
        if not expansion_features:
            raise RuntimeError("Session-level expansion received no blockwise reference features.")
        reference_labels = sorted(int(value) for value in torch.unique(
            next(iter(expansion_features.values()))["labels"], sorted=True
        ).tolist())
        if reference_labels != expected_classes:
            raise RuntimeError(
                f"Reference support classes {reference_labels} do not match Session classes {expected_classes}."
            )

        decisions, expanded, reused, contexts = [], [], [], {}
        # Phase 1: decide for all layers without mutating any layer topology.
        for block_idx in self.model_without_dp.expandable_blocks():
            item = expansion_features[int(block_idx)]
            support_size = int(item["cls_in"].size(0))
            threshold = self.expansion_thresholds.get(int(block_idx), {}).get(support_size)
            decision = self.model_without_dp.plan_layer_expansion(
                block_idx, expansion_features, class_index, threshold
            )
            decisions.append({
                "block_idx": int(block_idx), "expand": bool(decision.expand),
                "session_q90": float(decision.session_q90),
                "session_q95_threshold": float(decision.threshold),
                "support_count": int(decision.support_count),
            })
            if not decision.expand:
                adapter = self._adapter_by_block(block_idx)
                with torch.no_grad():
                    _, weights = adapter.routing_outputs(item["cls_in"].to(self.device).float())
                reused.append({"block_idx": int(block_idx), "mode": "reuse_frozen",
                               "expert_count": int(adapter.num_experts)})
                contexts[int(block_idx)] = {
                    "mode": "reuse_frozen", "adapter": adapter,
                    "history": {
                        "mode": "reuse_frozen", "parameter_updates": 0,
                        "classification_samples_per_epoch": [], "classification_samples_total": 0,
                        "rd_training_samples": 0, "rd_reference_samples": 0,
                        "route_anchor_count": int(item["cls_in"].size(0)),
                        "route_weight_mean": weights.mean(0).detach().cpu(),
                        "expert_count": int(adapter.num_experts),
                    },
                }
            else:
                contexts[int(block_idx)] = {"mode": "expand", "adapter": self._adapter_by_block(block_idx),
                                            "item": item, "decision": decision}

        # Phase 2: instantiate all new experts/RDs before the first optimizer step.
        for block_idx, context in contexts.items():
            if context["mode"] != "expand":
                continue
            adapter = context["adapter"]
            item = context["item"]
            values = item["cls_in"].to(self.device).float()
            labels = item["labels"].to(self.device).long()
            sources = item["source_ids"].to(self.device).long()
            fit_item = rd_fit_features[int(block_idx)]
            fit_values = fit_item["cls_in"].to(self.device).float()
            fit_labels = fit_item["labels"].to(self.device).long()
            fit_sources = fit_item["source_ids"].to(self.device).long()
            if (sources.numel() != values.size(0) or fit_sources.numel() != fit_values.size(0)
                    or torch.unique(sources).numel() != sources.numel()
                    or torch.unique(fit_sources).numel() != fit_sources.numel()
                    or set(sources.tolist()) != set(fit_sources.tolist())
                    or sorted(int(value) for value in torch.unique(labels, sorted=True).tolist()) != expected_classes
                    or sorted(int(value) for value in torch.unique(fit_labels, sorted=True).tolist()) != expected_classes):
                raise ValueError("New RD fit/reference views must cover the same stable Session support sources.")
            fit_label_by_source = {
                int(source): int(label)
                for source, label in zip(fit_sources.tolist(), fit_labels.tolist())
            }
            if any(fit_label_by_source.get(int(source)) != int(label)
                   for source, label in zip(sources.tolist(), labels.tolist())):
                raise ValueError("New RD fit/reference views disagree on source labels.")
            # Every Session source is used in both its augmented RD fitting view
            # and deterministic reference view; no support source is held out.
            reference_source_ids = set(int(value) for value in sources.detach().cpu().tolist())
            anchor_expert_id, anchor_q90 = adapter.select_descriptor_anchor(values)
            expert_id = self.model_without_dp.add_expert(
                block_idx, task_id, anchor_expert_id=anchor_expert_id
            )
            descriptor = adapter.descriptors[expert_id]
            descriptor.set_trainable(True)
            descriptor.train()
            anchor_sets = self._all_old_anchor_sets(old_states, block_idx)
            if any(anchor_sets[key] is None for key in ("route_mean", "route_var", "expert_mean", "expert_var")):
                raise RuntimeError(f"Block {block_idx} expansion lacks historical route/expert feature anchors.")
            context.update({
                "expert_id": int(expert_id), "descriptor": descriptor,
                "values": fit_values, "labels": fit_labels, "sources": fit_sources,
                "calibration_values": values, "calibration_labels": labels,
                "reference_source_ids": reference_source_ids,
                "source_to_row": {int(source): row for row, source in enumerate(fit_sources.tolist())},
                "anchor_q90": anchor_q90.detach().cpu(),
                "anchor_sets": anchor_sets,
                "history": {
                    "expert_id": int(expert_id), "descriptor_anchor_id": int(anchor_expert_id),
                    "descriptor_anchor_q90": [float(value) for value in anchor_q90.detach().cpu().tolist()],
                    "loss": [], "ce_loss": [], "rd_loss": [], "feature_space_loss": [],
                    "local_preservation_loss": [], "gradient_projection": [],
                    "classification_samples_per_epoch": [], "classification_samples_total": 0,
                    "rd_training_samples": int(values.size(0)),
                    "rd_reference_samples": int(values.size(0)),
                    "rd_training_source_ids": sorted(reference_source_ids),
                    "rd_reference_source_ids": sorted(reference_source_ids),
                    "rd_session_classes": sorted(int(value) for value in torch.unique(labels).tolist()),
                },
            })

        expanded_contexts = [context for context in contexts.values() if context["mode"] == "expand"]
        if not expanded_contexts:
            for _, adapter in self._moe_adapters():
                adapter.freeze_all()
            # Even when every layer reuses its pool, the accepted Session adds
            # a known feature domain for the next Session's novelty threshold.
            for block_idx in self.model_without_dp.expandable_blocks():
                self._refresh_block_expansion_threshold(
                    block_idx, current_reference_features=expansion_features, old_states=old_states
                )
            return {
                "trained": False, "decisions": decisions, "expanded": [],
                "reused": reused, "training": {block: value["history"] for block, value in contexts.items()},
                "semantic_prototype": None if semantic_proto is None else semantic_proto.detach(),
                "encoder_version": 0,
            }

        # One optimizer owns every newly created functional module and descriptor.
        parameter_groups, functional_parameters, descriptor_parameters = [], [], []
        for context in expanded_contexts:
            adapter = context["adapter"]
            expert_parameters = list(adapter.experts[-1].parameters())
            router_parameters = list(adapter.router.columns[-1].parameters())
            rd_parameters = list(context["descriptor"].trainable_parameters())
            functional_parameters.extend(expert_parameters + router_parameters)
            descriptor_parameters.extend(rd_parameters)
            parameter_groups.extend([
                {"params": expert_parameters, "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                {"params": router_parameters, "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)},
                {"params": rd_parameters, "lr": float(_cfg_value(optim_cfg, "LR_DESCRIPTOR", 1e-3))},
            ])
        optimizer = optim.AdamW(parameter_groups, weight_decay=float(optim_cfg.WEIGHT_DECAY))
        rd_weight = float(_cfg_value(incremental_cfg, "RD_WEIGHT", 1.0))
        feature_weight = float(_cfg_value(incremental_cfg, "FEATURE_SPACE_PRESERVATION_WEIGHT", 1.0))
        local_weight = float(_cfg_value(incremental_cfg, "LOCAL_RESIDUAL_PRESERVATION_WEIGHT", 1.0))
        if any(not np.isfinite(value) or value < 0 for value in (rd_weight, feature_weight, local_weight)):
            raise ValueError("Session objective weights must be finite and non-negative.")

        old_prefix, old_final, anchor_batch_size = self._historical_feature_anchor_batches(old_states)
        loss_projection_history = []
        for epoch in range(epochs):
            bank = self._build_incremental_bank(extract_loader, class_index, old_states, semantic_proto)
            self.model.train()
            for _, layer_adapter in self._moe_adapters():
                layer_adapter.eval()
            for context in expanded_contexts:
                context["adapter"].experts[-1].train()
                context["descriptor"].train()
            epoch_totals = {"ce": 0.0, "feature": 0.0, "local": 0.0, "rd": 0.0, "n": 0}
            observed_classes, projection_rows = set(), []
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                if not torch.isin(labels, torch.as_tensor(class_index, device=labels.device)).all():
                    raise ValueError("Training loader contains labels outside the current Session.")
                observed_classes.update(int(value) for value in labels.unique().tolist())
                optimizer.zero_grad(set_to_none=True)
                logits, targets = self._global_incremental_logits(images, labels, source_ids, bank, scale)
                classification = F.cross_entropy(logits, targets)

                # Replay historical frozen-prefix tokens through the complete current adapter stack.
                feature_losses = []
                for offset in range(0, old_prefix.size(0), anchor_batch_size):
                    stop = min(offset + anchor_batch_size, old_prefix.size(0))
                    feature_losses.append(self._final_feature_anchor_loss(
                        old_prefix[offset:stop], old_final[offset:stop]
                    ))
                feature_preservation = torch.stack(feature_losses).mean()

                local_losses, rd_losses = [], []
                for context in expanded_contexts:
                    adapter = context["adapter"]
                    anchors = context["anchor_sets"]
                    local_losses.append(adapter.old_feature_preservation_loss(
                        anchors["route_mean"], anchors["route_var"],
                        anchors["expert_mean"], anchors["expert_var"],
                    ))
                    source_to_row = context["source_to_row"]
                    try:
                        rows = torch.as_tensor(
                            [source_to_row[int(source)] for source in source_ids.tolist()],
                            device=self.device, dtype=torch.long,
                        )
                    except KeyError as error:
                        raise KeyError(f"Training source {error.args[0]} is absent from the Session reference cache.") from error
                    clean = context["values"][rows]
                    noise_scale = float(_cfg_value(self._visual_moe_cfg(), "DESCRIPTOR_DENOISE_SCALE", 0.05))
                    residual_weight = float(_cfg_value(self._visual_moe_cfg(), "DESCRIPTOR_RESIDUAL_WEIGHT", 1e-3))
                    noise = context["adapter"].base_feature_std.to(self.device).reshape(1, -1) * noise_scale
                    if clean.numel():
                        noisy = clean + torch.randn_like(clean) * noise
                        reconstruction = F.mse_loss(context["descriptor"](noisy), clean)
                        residual = context["descriptor"].residual_output(clean).square().mean()
                        rd_losses.append(reconstruction + residual_weight * residual)
                    else:
                        rd_losses.append(classification.new_zeros(()))
                local_preservation = torch.stack(local_losses).mean()
                rd_loss = torch.stack(rd_losses).mean()
                preservation = feature_weight * feature_preservation + local_weight * local_preservation
                total_loss = classification + preservation + rd_weight * rd_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("Unified Session objective became non-finite.")

                # Project the single global CE gradient against the complete historical-preservation gradient.
                ce_grad = torch.autograd.grad(
                    classification, functional_parameters, retain_graph=True, allow_unused=True
                )
                preservation_grad = torch.autograd.grad(
                    preservation, functional_parameters, retain_graph=True, allow_unused=True
                )
                projected_ce, projection_diag = self._project_conflicting_gradients(
                    ce_grad, preservation_grad
                )
                for parameter, ce_value, preserve_value in zip(
                    functional_parameters, projected_ce, preservation_grad
                ):
                    if ce_value is None and preserve_value is None:
                        parameter.grad = None
                    else:
                        combined = torch.zeros_like(parameter) if ce_value is None else ce_value
                        if preserve_value is not None:
                            combined = combined + preserve_value
                        parameter.grad = combined.detach()
                rd_grad = torch.autograd.grad(
                    rd_weight * rd_loss, descriptor_parameters, allow_unused=True
                )
                for parameter, gradient in zip(descriptor_parameters, rd_grad):
                    parameter.grad = None if gradient is None else gradient.detach()
                torch.nn.utils.clip_grad_norm_(functional_parameters, 1.0)
                torch.nn.utils.clip_grad_norm_(descriptor_parameters, 1.0)
                optimizer.step()

                batch_size = int(labels.numel())
                epoch_totals["ce"] += float(classification.item()) * batch_size
                epoch_totals["feature"] += float(feature_preservation.item()) * batch_size
                epoch_totals["local"] += float(local_preservation.item()) * batch_size
                epoch_totals["rd"] += float(rd_loss.item()) * batch_size
                epoch_totals["n"] += batch_size
                projection_rows.append(projection_diag)
                for context in expanded_contexts:
                    context["history"]["classification_samples_total"] += batch_size
            if epoch_totals["n"] == 0 or sorted(observed_classes) != expected_classes:
                raise RuntimeError("Joint Session loader was empty or failed to cover all Session classes.")
            n = epoch_totals["n"]
            for context in expanded_contexts:
                history = context["history"]
                history["ce_loss"].append(epoch_totals["ce"] / n)
                history["feature_space_loss"].append(epoch_totals["feature"] / n)
                history["local_preservation_loss"].append(epoch_totals["local"] / n)
                history["rd_loss"].append(epoch_totals["rd"] / n)
                history["loss"].append(
                    epoch_totals["ce"] / n + epoch_totals["feature"] / n +
                    epoch_totals["local"] / n + rd_weight * epoch_totals["rd"] / n
                )
                history["classification_samples_per_epoch"].append(n)
            loss_projection_history.append(projection_diag)

        # Calibrate all newly trained RDs only after the shared functional training has finished.
        for context in expanded_contexts:
            descriptor = context["descriptor"]
            descriptor.eval()
            with torch.no_grad():
                reference_values = context["calibration_values"]
                errors = descriptor.reconstruction_error(reference_values)
                responsibility = context["adapter"].expert_responsibility(
                    reference_values, context["expert_id"]
                ).detach()
                descriptor.update_reference(errors, responsibility)
                context['adapter'].assert_descriptors_ready()
                reference_scores = descriptor.standardized_score(reference_values)
                context["history"]["rd_effective_reference_count"] = float(
                    descriptor.effective_error_count.item()
                )
                context["history"]["rd_reference_q90"] = float(
                    torch.quantile(reference_scores, 0.90).item()
                )
                context["history"]["rd_reference_responsibility_mean"] = float(
                    responsibility.mean().item()
                )
                context["history"]["rd_fit_reference_view_cosine"] = float(
                    F.cosine_similarity(
                        F.normalize(context["values"], dim=-1),
                        F.normalize(reference_values, dim=-1),
                        dim=-1,
                    ).mean().item()
                )
                context["history"]["session_q90"] = float(context["decision"].session_q90)
                context["history"]["session_q95_threshold"] = float(context["decision"].threshold)
                context["history"]["old_anchor_gate_mean"] = context["adapter"].old_anchor_response(
                    context["anchor_sets"]["route_mean"], context["anchor_sets"]["route_var"]
                )
                context["history"]["old_anchor_residual_norm"] = context["adapter"].old_anchor_residual_response(
                    context["anchor_sets"]["route_mean"], context["anchor_sets"]["route_var"],
                    context["anchor_sets"]["expert_mean"], context["anchor_sets"]["expert_var"]
                )
            context["adapter"].freeze_all()
        # Recalibrate thresholds after every new RD has joined its layer's deployed pool.
        for block_idx, context in contexts.items():
            if context["mode"] == "expand":
                expanded.append({
                    "block_idx": int(block_idx), "expert_id": int(context["expert_id"]),
                    "descriptor_anchor_id": int(context["adapter"].descriptors[-1].anchor_expert_id),
                    "session_q90": float(context["decision"].session_q90),
                    "session_q95_threshold": float(context["decision"].threshold),
                })
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        # Recalibrate every layer after all new RDs have valid error statistics.
        # Reused-only layers also learn from this accepted Session's known domain.
        for block_idx in self.model_without_dp.expandable_blocks():
            self._refresh_block_expansion_threshold(
                block_idx, current_reference_features=expansion_features, old_states=old_states
            )
        self._assert_moe_frozen("joint incremental Session completion")
        histories = {block: context["history"] for block, context in contexts.items()}
        for block, history in histories.items():
            if history.get("mode") == "reuse_frozen":
                continue
            history["gradient_projection"] = loss_projection_history
        return {
            "trained": bool(expanded), "decisions": decisions, "expanded": expanded,
            "reused": reused, "training": histories,
            "semantic_prototype": None if semantic_proto is None else semantic_proto.detach(),
            "encoder_version": 0,
        }

    def _build_and_refine_task_state(self, current_class_name, extract_loader, class_index,
                                     old_states, semantic_proto=None, session_id=0):
        """Build a novel visual prototype, then apply the single semantic-guided calibration."""
        state = self.model_without_dp.build_task_statistics(
            current_class_name, extract_loader, class_index,
        )
        if semantic_proto is not None:
            state["description_proto"] = F.normalize(torch.as_tensor(semantic_proto, device=self.device).float(), dim=-1)
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
        state["creation_session"] = int(session_id)
        state["anchor_by_block"] = self._collect_class_anchors(extract_loader)
        state["reference_cls_by_block"] = self._collect_reference_cls_cache(extract_loader)
        state["final_feature_anchors"] = self._collect_final_feature_anchors(
            extract_loader, int(session_id)
        )
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
                if use_moe:
                    state["reference_cls_by_block"] = self._cpu_copy(self.base_reference_cls_features)
                    self._capture_base_reference_accuracy_anchor(state)
                    state["final_feature_anchors"] = self._collect_final_feature_anchors(extract_loader, 0)
            else:
                if use_moe:
                    self._routing_diagnostics(extract_loader, f"session{task_id}-before",
                                              current_task=task_id)
                    adaptation = self.train_incremental_task(
                        task_id, train_loader, extract_loader, class_index,
                        class_names, old_states)
                    plan_summary = self._cpu_copy(adaptation["decisions"])
                self.model.eval()
                state = adaptation.pop("task_state")

            state_dict_list.append(state)
            feature_space_diagnostics = self._base_reference_feature_diagnostics(state_dict_list[0])
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
                "feature_space": self._cpu_copy(feature_space_diagnostics),
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
