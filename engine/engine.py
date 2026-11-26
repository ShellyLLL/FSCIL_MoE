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
    @staticmethod
    def _tensor_tree_nbytes(value):
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(Runner._tensor_tree_nbytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(Runner._tensor_tree_nbytes(item) for item in value)
        return 0

    def _feature_memory_summary(self, state_dict_list):
        """Report persistent feature-anchor/cache memory in bytes and MiB."""
        base_prefix = self._tensor_tree_nbytes(
            None if self.base_reference_prefix_cache is None
            else self.base_reference_prefix_cache.get("prefix_tokens")
        )
        final_anchor_prefix = sum(
            self._tensor_tree_nbytes(state.get("final_feature_anchors", {}).get("prefix_tokens"))
            for state in state_dict_list
        )
        final_anchor_targets = sum(
            self._tensor_tree_nbytes(state.get("final_feature_anchors", {}).get("final_features"))
            for state in state_dict_list
        )
        reference_cls = sum(
            self._tensor_tree_nbytes(cache.get("cls_in"))
            for state in state_dict_list
            for cache in state.get("reference_cls_by_block", {}).values()
        )
        total = base_prefix + final_anchor_prefix + final_anchor_targets + reference_cls
        return {
            "base_reference_prefix_bytes": base_prefix,
            "historical_prefix_anchor_bytes": final_anchor_prefix,
            "historical_final_feature_bytes": final_anchor_targets,
            "known_domain_cls_cache_bytes": reference_cls,
            "total_feature_cache_bytes": total,
            "total_feature_cache_mib": float(total / (1024.0 ** 2)),
        }

    def _snapshot_session_transaction(self, state_dict_list):
        """Capture every mutable FSCIL state component before a Session starts."""
        adapter_states = []
        for block_idx, adapter in self._moe_adapters():
            adapter_states.append((
                int(block_idx),
                adapter.export_topology(),
                {key: value.detach().clone() for key, value in adapter.state_dict().items()},
            ))
        base_prototype = getattr(self.model_without_dp, "base_enhanced_visual_prototype", None)
        return {
            "adapters": adapter_states,
            "base_reference_source_ids": list(self.base_reference_source_ids),
            # These reference caches are immutable after Base completion; retain
            # their objects instead of duplicating potentially large CPU tensors.
            "base_reference_cls_features": self.base_reference_cls_features,
            "base_reference_prefix_cache": self.base_reference_prefix_cache,
            "base_reference_feature_prototypes": self.base_reference_feature_prototypes,
            "base_reference_class_ids": self.base_reference_class_ids,
            "base_reference_accuracy_baseline": self.base_reference_accuracy_baseline,
            "base_enhanced_visual_prototype": (
                None if base_prototype is None else base_prototype.detach().clone()
            ),
            "state_dict_list": list(state_dict_list),
            "acc_list": list(self.acc_list),
            "task_acc_list": list(self.task_acc_list),
            "session_diagnostics": list(self.session_diagnostics),
            "rng_state": self._rng_state(),
            "training_modes": {id(module): bool(module.training) for module in self.model.modules()},
        }

    def _restore_session_transaction(self, snapshot, state_dict_list):
        """Roll back model, references, metrics, RNG and module modes as one Session unit."""
        for block_idx, topology, state in snapshot["adapters"]:
            adapter = self._adapter_by_block(block_idx)
            adapter.rebuild_topology(topology)
            adapter.load_state_dict(state, strict=True)
            adapter.freeze_all()
        self.base_reference_source_ids = list(snapshot["base_reference_source_ids"])
        self.base_reference_cls_features = snapshot["base_reference_cls_features"]
        self.base_reference_prefix_cache = snapshot["base_reference_prefix_cache"]
        self.base_reference_feature_prototypes = snapshot["base_reference_feature_prototypes"]
        self.base_reference_class_ids = snapshot["base_reference_class_ids"]
        self.base_reference_accuracy_baseline = snapshot["base_reference_accuracy_baseline"]
        prototype = snapshot["base_enhanced_visual_prototype"]
        if prototype is not None and hasattr(self.model_without_dp, "base_enhanced_visual_prototype"):
            self.model_without_dp.base_enhanced_visual_prototype.copy_(prototype)
        state_dict_list[:] = snapshot["state_dict_list"]
        self.acc_list = list(snapshot["acc_list"])
        self.task_acc_list = list(snapshot["task_acc_list"])
        self.session_diagnostics = list(snapshot["session_diagnostics"])
        self._restore_rng_state(snapshot["rng_state"])
        for module in self.model.modules():
            module.training = snapshot["training_modes"].get(id(module), False)

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
        """Versioned contract for sequential adapted-stream AE-driven expansion."""
        moe = self._visual_moe_cfg()
        inc = self.cfg.TRAINER.BiMC.INCREMENTAL
        opt = self.cfg.TRAINER.BiMC.OPTIM
        data = _cfg_value(self.cfg, "DATASET", None)
        return {
            "version": 21,
            "decision": "sequential_layerwise_expand_iff_minimum_historical_rd_session_mean_z_gt_1",
            "threshold": 1.0,
            "session_statistic": "mean_unweighted_standardized_reconstruction_error_per_historical_rd",
            "feature_stream": "adapted_hidden_state_after_completed_shallower_blocks",
            "migration": "same_argmin_rd_initializes_expert_and_anchored_rd",
            "expert_initialization": "exact_historical_parameter_copy_and_hidden_width_inheritance",
            "regularization": "l2_sp_to_frozen_migration_snapshot",
            "reuse": "all_existing_experts_and_router_columns_frozen_and_active",
            "rd_fit": "separate_reconstruction_only_on_current_layer_augmented_session_inputs",
            "rd_calibration": "unweighted_deterministic_current_session_reference_errors",
            "router": "base_softmax_plus_incremental_independent_sigmoid_on_adapted_layer_input",
            "feature_preservation": "frozen_prefix_replay_cosine_distillation_plus_hard_acceptance_gate",
            "historical_prototypes": "immutable_original_support_prototypes",
            "prototype_calibration": "topk_base_semantic_retrieval_with_base_support_loo_alpha_selection",
            "semantic_alpha_selection": "deterministic_leave_one_source_out_base_support_grid_search_no_test_labels",
            "visual_moe_enabled": bool(_cfg_value(moe, "ENABLE", False)),
            "insert_blocks": [int(v) for v in _cfg_value(moe, "INSERT_BLOCKS", [])],
            "num_base_experts": int(_cfg_value(moe, "NUM_BASE_EXPERTS", 4)),
            "base_reduction": int(_cfg_value(moe, "BASE_REDUCTION", 4)),
            "descriptor_dim": int(_cfg_value(moe, "DESCRIPTOR_DIM", 64)),
            "descriptor_eps": float(_cfg_value(moe, "DESCRIPTOR_EPS", 1e-4)),
            "descriptor_buffer_size": int(_cfg_value(moe, "DESCRIPTOR_BUFFER_SIZE", 500)),
            "descriptor_residual_rank": int(_cfg_value(moe, "DESCRIPTOR_RESIDUAL_RANK", 16)),
            "descriptor_denoise_scale": float(_cfg_value(moe, "DESCRIPTOR_DENOISE_SCALE", 0.05)),
            "descriptor_residual_weight": float(_cfg_value(moe, "DESCRIPTOR_RESIDUAL_WEIGHT", 1e-3)),
            "incremental_gate_bias_init": float(_cfg_value(moe, "INCREMENTAL_GATE_BIAS_INIT", -4.0)),
            "incremental_dropout": float(_cfg_value(moe, "INCREMENTAL_DROPOUT", 0.1)),
            "semantic_temperature": float(_cfg_value(moe, "SEMANTIC_TEMPERATURE", 0.07)),
            "semantic_retrieval_top_k": int(_cfg_value(moe, "SEMANTIC_RETRIEVAL_TOP_K", 5)),
            "semantic_calibration_max_alpha": float(_cfg_value(moe, "SEMANTIC_CALIBRATION_MAX_ALPHA", 0.5)),
            "prototype_reliability_k": float(_cfg_value(moe, "PROTOTYPE_RELIABILITY_K", 5.0)),
            "feature_space_acceptance": {
                "minimum_class_prototype_cosine": float(_cfg_value(moe, "FEATURE_SPACE_MIN_COSINE", 0.98)),
                "maximum_base_reference_accuracy_drop_pp": float(
                    _cfg_value(moe, "FEATURE_SPACE_MAX_ACCURACY_DROP", 0.0)
                ),
                "anchors_per_class": int(_cfg_value(moe, "FEATURE_SPACE_ANCHORS_PER_CLASS", 2)),
                "anchor_batch_size": int(_cfg_value(moe, "FEATURE_SPACE_ANCHOR_BATCH_SIZE", 16)),
            },
            "incremental": {
                "epochs": int(_cfg_value(inc, "EPOCHS", 6)),
                "logit_scale": float(_cfg_value(inc, "LOGIT_SCALE", 100.0)),
                "l2sp_weight": float(_cfg_value(inc, "L2SP_WEIGHT", 1e-4)),
                "feature_space_preservation_weight": float(
                    _cfg_value(inc, "FEATURE_SPACE_PRESERVATION_WEIGHT", 1.0)
                ),
            },
            "optimizer": {
                "base_epochs": int(_cfg_value(opt, "BASE_EPOCHS", 50)),
                "base_descriptor_epochs": int(_cfg_value(opt, "BASE_DESCRIPTOR_EPOCHS", 8)),
                "incremental_descriptor_epochs": int(_cfg_value(opt, "INCREMENTAL_DESCRIPTOR_EPOCHS", 8)),
                "lr_base_moe": float(_cfg_value(opt, "LR_BASE_MOE", 1e-4)),
                "lr_incremental_expert": float(_cfg_value(opt, "LR_INCREMENTAL_EXPERT", 3e-4)),
                "lr_incremental_router": float(_cfg_value(opt, "LR_INCREMENTAL_ROUTER", 3e-4)),
                "lr_descriptor": float(_cfg_value(opt, "LR_DESCRIPTOR", 1e-3)),
                "weight_decay": float(_cfg_value(opt, "WEIGHT_DECAY", 1e-4)),
            },
            "data": {
                "seed": int(_cfg_value(self.cfg, "SEED", 1)),
                "num_init_classes": int(_cfg_value(data, "NUM_INIT_CLS", -1)),
                "num_incremental_classes": int(_cfg_value(data, "NUM_INC_CLS", -1)),
                "base_shot": int(_cfg_value(data, "NUM_BASE_SHOT", -1)),
                "incremental_shot": int(_cfg_value(data, "NUM_INC_SHOT", -1)),
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

    @staticmethod
    def _validate_checkpoint_source_rows(session_index, source_to_class, source_ids, labels, source_kind):
        """Validate cached rows against the complete session support manifest.

        Reference CLS caches may intentionally be a subset (the Base RD
        calibration holdout), while final-feature anchors are selected from
        the complete Session support set. They must both belong to the same
        manifest, but need not be subsets of one another.
        """
        source_ids = torch.as_tensor(source_ids, dtype=torch.long).reshape(-1)
        labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
        if (source_ids.numel() == 0 or source_ids.numel() != labels.numel()
                or torch.unique(source_ids).numel() != source_ids.numel()):
            raise ValueError(
                f"Checkpoint task state {session_index} {source_kind} source ids/labels are malformed."
            )
        source_values = [int(value) for value in source_ids.tolist()]
        label_values = [int(value) for value in labels.tolist()]
        missing = [value for value in source_values if value not in source_to_class]
        if missing:
            raise ValueError(
                f"Checkpoint task state {session_index} {source_kind} contains source ids "
                f"outside its support manifest: {missing[:8]}"
            )
        mismatched = [
            (source_id, label, int(source_to_class[source_id]))
            for source_id, label in zip(source_values, label_values)
            if int(source_to_class[source_id]) != label
        ]
        if mismatched:
            raise ValueError(
                f"Checkpoint task state {session_index} {source_kind} labels disagree "
                f"with its support manifest: {mismatched[:8]}"
            )
        return set(source_values)

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
            "support_manifest", "moe_topology", "expansion_policy",
            "base_reference_source_ids", "base_reference_cls_features",
            "base_reference_prefix_cache", "base_reference_feature_prototypes", "base_reference_class_ids",
            "base_reference_accuracy_baseline",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
        schema = int(payload["schema_version"])
        if schema != 40:
            raise ValueError("Checkpoint predates immutable historical prototypes and replay-constrained sequential expansion (schema 40).")
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
        manifest = payload.get("support_manifest")
        if manifest is None:
            raise ValueError("Resume checkpoint support_manifest must not be null.")
        # Validate the checkpoint's complete support protocol before using it
        # to verify cached reference and feature-anchor source identities.
        self.data_manager._validate_support_manifest(manifest)
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
        checkpoint_support_sources = set()
        checkpoint_anchor_sources = set()
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
            if session_index == 0:
                if "semantic_calibration_alpha_max" not in state:
                    raise ValueError("Checkpoint Base state is missing its LOO-calibrated semantic alpha.")
                calibrated_alpha = float(state["semantic_calibration_alpha_max"])
                if not np.isfinite(calibrated_alpha) or not 0.0 <= calibrated_alpha <= 1.0:
                    raise ValueError("Checkpoint Base semantic calibration alpha is invalid.")
            class_ids = torch.as_tensor(state["class_ids"], dtype=torch.long).reshape(-1)
            expected_ids = torch.as_tensor(
                self.data_manager.class_index_in_task[session_index], dtype=torch.long
            ).reshape(-1)
            if not torch.equal(class_ids.sort().values, expected_ids.sort().values):
                raise ValueError(
                    f"Checkpoint session {session_index} class ids do not match the configured FSCIL split."
                )
            checkpoint_class_ids.append(class_ids)
            session_manifest = manifest["sessions"][str(session_index)]
            expected_manifest_classes = {str(int(value)) for value in class_ids.tolist()}
            if set(session_manifest) != expected_manifest_classes:
                raise ValueError(
                    f"Checkpoint task state {session_index} support-manifest classes "
                    "do not match its classifier classes."
                )
            session_support_source_to_class = {}
            for class_key, raw_source_ids in session_manifest.items():
                class_id = int(class_key)
                for raw_source_id in raw_source_ids:
                    source_id = int(raw_source_id)
                    if source_id in session_support_source_to_class:
                        raise ValueError(
                            f"Checkpoint session {session_index} support manifest reuses source {source_id}."
                        )
                    session_support_source_to_class[source_id] = class_id
            session_support_sources = set(session_support_source_to_class)
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
            session_reference_sources = None
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
                self._validate_checkpoint_source_rows(
                    session_index, session_support_source_to_class,
                    source_ids, labels, "reference CLS cache",
                )
                if session_reference_sources is None:
                    session_reference_sources = set(int(value) for value in source_ids.tolist())
                elif session_reference_sources != set(int(value) for value in source_ids.tolist()):
                    raise ValueError(
                        f"Checkpoint task state {session_index} reference CLS blocks disagree on source identity."
                    )
            if not session_reference_sources:
                raise ValueError(f"Checkpoint task state {session_index} has an empty reference CLS source set.")
            if checkpoint_support_sources.intersection(session_support_sources):
                raise ValueError(
                    f"Checkpoint task state {session_index} reuses support sources from an earlier Session."
                )
            checkpoint_support_sources.update(session_support_sources)
            feature_anchors = state["final_feature_anchors"]
            if not isinstance(feature_anchors, dict):
                raise ValueError(f"Checkpoint task state {session_index} has no final feature-space anchors.")
            prefix_tokens = torch.as_tensor(feature_anchors.get("prefix_tokens"))
            final_features = torch.as_tensor(feature_anchors.get("final_features"))
            anchor_sources = torch.as_tensor(feature_anchors.get("source_ids"), dtype=torch.long).reshape(-1)
            anchor_labels = torch.as_tensor(feature_anchors.get("labels"), dtype=torch.long).reshape(-1)
            anchor_class_ids = torch.as_tensor(feature_anchors.get("class_ids"), dtype=torch.long).reshape(-1)
            expected_anchor_session = int(feature_anchors.get("session_id", -1))
            if (prefix_tokens.ndim != 3 or final_features.ndim != 2
                    or prefix_tokens.size(0) != final_features.size(0)
                    or final_features.size(0) != anchor_sources.numel()
                    or anchor_labels.numel() != anchor_sources.numel()
                    or not torch.equal(torch.unique(anchor_labels, sorted=True), anchor_class_ids.sort().values)
                    or not torch.equal(anchor_class_ids.sort().values, class_ids.sort().values)
                    or expected_anchor_session != int(session_index)
                    or final_features.size(1) != int(torch.as_tensor(state["image_proto"]).size(1))
                    or final_features.size(0) == 0
                    or not torch.isfinite(prefix_tokens.float()).all()
                    or not torch.isfinite(final_features.float()).all()
                    or torch.unique(anchor_sources).numel() != anchor_sources.numel()):
                raise ValueError(f"Checkpoint task state {session_index} has malformed final feature anchors.")
            anchor_source_set = self._validate_checkpoint_source_rows(
                session_index, session_support_source_to_class,
                anchor_sources, anchor_labels, "final feature anchors",
            )
            if checkpoint_anchor_sources.intersection(anchor_source_set):
                raise ValueError(
                    f"Checkpoint task state {session_index} reuses historical feature-anchor sources."
                )
            checkpoint_anchor_sources.update(anchor_source_set)
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
        self.data_manager.load_support_manifest(manifest)
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
            "schema_version": 40,
            "completed_session": int(completed_session),
            "base_reference_source_ids": list(self.base_reference_source_ids),
            "base_reference_cls_features": self._cpu_copy(self.base_reference_cls_features),
            "base_reference_prefix_cache": self._cpu_copy(getattr(self, "base_reference_prefix_cache", None)),
            "base_reference_feature_prototypes": self._cpu_copy(getattr(self, "base_reference_feature_prototypes", None)),
            "base_reference_class_ids": self._cpu_copy(getattr(self, "base_reference_class_ids", None)),
            "base_reference_accuracy_baseline": self._cpu_copy(getattr(self, "base_reference_accuracy_baseline", None)),
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
        publish_targets = [target]
        if bool(_cfg_value(self._checkpoint_cfg(), "SAVE_EVERY_SESSION", True)):
            publish_targets.append(directory / "latest.pth")
        temp_paths = [path.with_suffix(path.suffix + ".tmp") for path in publish_targets]
        backup_paths = [path.with_suffix(path.suffix + ".bak") for path in publish_targets]
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()
        for backup_path in backup_paths:
            if backup_path.exists():
                backup_path.unlink()
        try:
            for temp_path in temp_paths:
                torch.save(payload, temp_path)
            backed_up = []
            published = []
            try:
                for path, backup_path in zip(publish_targets, backup_paths):
                    if path.exists():
                        os.replace(path, backup_path)
                        backed_up.append((path, backup_path))
                for temp_path, path in zip(temp_paths, publish_targets):
                    os.replace(temp_path, path)
                    published.append(path)
            except Exception:
                for path in published:
                    if path.exists():
                        path.unlink()
                for path, backup_path in reversed(backed_up):
                    if backup_path.exists():
                        if path.exists():
                            path.unlink()
                        os.replace(backup_path, path)
                raise
            else:
                for _, backup_path in backed_up:
                    if backup_path.exists():
                        try:
                            backup_path.unlink()
                        except OSError:
                            pass
        finally:
            for temp_path in temp_paths:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
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

    @classmethod
    def _source_exclusive_rd_split(
        cls, fit_labels, fit_source_ids, reference_labels, reference_source_ids,
        per_class_reference=1,
    ):
        """Split RD fitting and error calibration by source identity, not augmentation view."""
        fit_labels = torch.as_tensor(fit_labels).long().reshape(-1)
        fit_source_ids = torch.as_tensor(fit_source_ids, device=fit_labels.device).long().reshape(-1)
        reference_labels = torch.as_tensor(reference_labels, device=fit_labels.device).long().reshape(-1)
        reference_source_ids = torch.as_tensor(reference_source_ids, device=fit_labels.device).long().reshape(-1)
        if (fit_labels.numel() == 0 or reference_labels.numel() == 0
                or fit_source_ids.numel() != fit_labels.numel()
                or reference_source_ids.numel() != reference_labels.numel()):
            raise ValueError("RD fit/reference labels and source ids must be non-empty and aligned.")
        if (torch.unique(fit_source_ids).numel() != fit_source_ids.numel()
                or torch.unique(reference_source_ids).numel() != reference_source_ids.numel()):
            raise ValueError("RD fit/reference loaders must expose one row per support source.")
        if set(fit_source_ids.tolist()) != set(reference_source_ids.tolist()):
            raise ValueError("RD fit/reference loaders must cover the same support sources.")
        fit_label_map = {int(source): int(label) for source, label in zip(fit_source_ids.tolist(), fit_labels.tolist())}
        if any(fit_label_map.get(int(source)) != int(label)
               for source, label in zip(reference_source_ids.tolist(), reference_labels.tolist())):
            raise ValueError("RD fit/reference labels disagree by source id.")
        reference_mask = cls._select_class_balanced_reference_mask(
            reference_labels, reference_source_ids, per_class_count=int(per_class_reference),
        )
        held_out_sources = reference_source_ids[reference_mask]
        fit_mask = ~torch.isin(fit_source_ids, held_out_sources)
        fit_classes = torch.unique(fit_labels[fit_mask], sorted=True)
        expected_classes = torch.unique(reference_labels, sorted=True)
        if not torch.equal(fit_classes, expected_classes):
            missing = expected_classes[~torch.isin(expected_classes, fit_classes)].tolist()
            raise ValueError(f"RD source-exclusive fitting split leaves classes without fit samples: {missing}.")
        if int(fit_mask.sum().item()) < 3:
            raise ValueError("RD source-exclusive fitting split requires at least three non-calibration sources.")
        if torch.isin(fit_source_ids[fit_mask], held_out_sources).any():
            raise RuntimeError("RD fit/calibration source split is not disjoint.")
        return fit_mask, reference_mask

    def _train_base_descriptors(self, train_loader, extract_loader):
        """Fit Base RDs on augmented CLS and calibrate immutable reconstruction statistics."""
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "BASE_DESCRIPTOR_EPOCHS", 8))
        fit_map = self.model_without_dp.collect_blockwise_cls_features(train_loader)
        ref_map = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        if set(fit_map) != set(ref_map):
            raise RuntimeError("Base RD fit/reference block views differ.")
        self.base_reference_cls_features, self.base_reference_source_ids = {}, []
        for block_idx, adapter in self._moe_adapters():
            fit, ref = fit_map[block_idx], ref_map[block_idx]
            fit_x, fit_y = fit["cls_in"].to(self.device).float(), fit["labels"].to(self.device).long()
            fit_sid = fit["source_ids"].to(self.device).long()
            ref_x, ref_y = ref["cls_in"].to(self.device).float(), ref["labels"].to(self.device).long()
            ref_sid = ref["source_ids"].to(self.device).long()
            if (set(fit_sid.tolist()) != set(ref_sid.tolist())
                    or torch.unique(fit_sid).numel() != fit_sid.numel()
                    or torch.unique(ref_sid).numel() != ref_sid.numel()):
                raise RuntimeError(f"Block {block_idx} Base fit/reference source ids are misaligned.")
            label_map = {int(i): int(y) for i,y in zip(fit_sid.tolist(),fit_y.tolist())}
            if any(label_map.get(int(i)) != int(y) for i,y in zip(ref_sid.tolist(),ref_y.tolist())):
                raise RuntimeError(f"Block {block_idx} Base fit/reference labels disagree.")
            classes = torch.unique(ref_y, sorted=True)
            reference_per_class = max(
                1, min(5, int(adapter.descriptors[0].buffer_size) // int(classes.numel()))
            )
            fit_mask, mask = self._source_exclusive_rd_split(
                fit_y, fit_sid, ref_y, ref_sid, per_class_reference=reference_per_class,
            )
            fit_values = fit_x[fit_mask]
            adapter.set_base_feature_stats(fit_values)
            selected = [int(v) for v in ref_sid[mask].tolist()]
            if not self.base_reference_source_ids:
                self.base_reference_source_ids = selected
            elif self.base_reference_source_ids != selected:
                raise RuntimeError("Base reference source ids differ across MoE blocks.")
            ref_values = ref_x[mask]
            self.base_reference_cls_features[int(block_idx)] = {
                "cls_in": ref_values.detach().cpu().half(),
                "labels": ref_y[mask].detach().cpu(),
                "source_ids": ref_sid[mask].detach().cpu(),
            }
            with torch.no_grad():
                fit_resp = torch.stack([
                    adapter.expert_responsibility(fit_values, i)
                    for i in range(adapter.num_experts)
                ], dim=-1)
            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(
                    adapter, expert_id, fit_values, epochs, fit_resp[:, expert_id],
                    calibration_values=ref_values, calibration_weights=None,
                )
                descriptor = adapter.descriptors[expert_id]
                if not descriptor.is_ready() or int(descriptor.error_count.item()) < 2:
                    raise RuntimeError("Base RD failed to calibrate reconstruction statistics.")
                print(f"=> [AE-Descriptor][B{block_idx}][E{expert_id}] loss={loss:.6f} "
                      f"mean={descriptor.error_mean.item():.6f} std={descriptor.error_std.item():.6f}")
            adapter.assert_descriptors_ready()
            adapter.freeze_all()
        if not self.base_reference_source_ids:
            raise RuntimeError("Base RD calibration produced no reference sources.")
        self._capture_base_reference_feature_anchors()

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
    @torch.no_grad()
    def _historical_feature_anchor_diagnostics(self, historical_states):
        """Measure every accepted Session's stored class anchors in the current encoder."""
        if not historical_states:
            return {"available": False, "reason": "no_historical_states"}
        self.model.eval()
        class_cosines, sample_cosines = [], []
        per_session = {}
        seen_classes, seen_sources = set(), set()
        for state_index, state in enumerate(historical_states):
            anchors = state.get("final_feature_anchors")
            if not isinstance(anchors, dict):
                return {
                    "available": False,
                    "reason": "missing_final_feature_anchors",
                    "missing_state_index": int(state_index),
                }
            session_id = int(anchors.get("session_id", state.get("creation_session", state_index)))
            class_ids = torch.as_tensor(
                anchors.get("class_ids"), dtype=torch.long
            ).reshape(-1)
            labels = torch.as_tensor(anchors.get("labels"), dtype=torch.long).reshape(-1)
            source_ids = torch.as_tensor(anchors.get("source_ids"), dtype=torch.long).reshape(-1)
            prefixes = torch.as_tensor(anchors.get("prefix_tokens")).float()
            targets = torch.as_tensor(anchors.get("final_features")).float()
            count = int(prefixes.size(0)) if prefixes.ndim == 3 else -1
            if (prefixes.ndim != 3 or count == 0 or targets.ndim != 2
                    or targets.size(0) != count or labels.numel() != count
                    or source_ids.numel() != count or class_ids.numel() == 0
                    or not torch.isfinite(prefixes).all() or not torch.isfinite(targets).all()
                    or torch.unique(source_ids).numel() != source_ids.numel()
                    or not set(torch.unique(labels).tolist()).issubset(set(class_ids.tolist()))):
                raise ValueError(
                    f"Historical feature anchors for Session {session_id} are malformed or misaligned."
                )
            duplicate_classes = seen_classes.intersection(int(value) for value in class_ids.tolist())
            duplicate_sources = seen_sources.intersection(int(value) for value in source_ids.tolist())
            if duplicate_classes:
                raise ValueError(
                    f"Historical feature anchors repeat classes across Sessions: {sorted(duplicate_classes)}."
                )
            if duplicate_sources:
                raise ValueError(
                    f"Historical feature anchors repeat source ids across Sessions: {sorted(duplicate_sources)}."
                )
            seen_classes.update(int(value) for value in class_ids.tolist())
            seen_sources.update(int(value) for value in source_ids.tolist())

            current = self.model_without_dp.extract_img_feature_from_reference_prefix(
                prefixes.to(self.device)
            )
            current = F.normalize(current.float(), dim=-1)
            target = F.normalize(targets.to(self.device), dim=-1)
            if current.shape != target.shape or not torch.isfinite(current).all():
                raise RuntimeError(
                    f"Historical feature replay for Session {session_id} produced invalid embeddings."
                )
            sample_alignment = F.cosine_similarity(current, target, dim=-1)
            session_class_cosines = {}
            for class_id in sorted(int(value) for value in class_ids.tolist()):
                mask = labels.to(self.device).eq(class_id)
                if not bool(mask.any().item()):
                    raise ValueError(
                        f"Historical feature anchors for Session {session_id} omit class {class_id}."
                    )
                current_prototype = F.normalize(current[mask].mean(0), dim=-1)
                target_prototype = F.normalize(target[mask].mean(0), dim=-1)
                cosine = float(F.cosine_similarity(
                    current_prototype.unsqueeze(0), target_prototype.unsqueeze(0), dim=-1
                ).item())
                session_class_cosines[int(class_id)] = cosine
                class_cosines.append(cosine)
            sample_cosines.extend(float(value) for value in sample_alignment.detach().cpu().tolist())
            per_session[int(session_id)] = {
                "num_classes": int(len(session_class_cosines)),
                "num_anchors": int(count),
                "class_prototype_cosine_mean": float(np.mean(list(session_class_cosines.values()))),
                "class_prototype_cosine_min": float(min(session_class_cosines.values())),
                "class_prototype_cosine_per_class": session_class_cosines,
                "sample_cosine_mean": float(sample_alignment.mean().item()),
                "sample_cosine_min": float(sample_alignment.min().item()),
            }
        if not class_cosines:
            return {"available": False, "reason": "empty_historical_anchor_bank"}
        target_cosine = float(_cfg_value(
            self._visual_moe_cfg(), "FEATURE_SPACE_MIN_COSINE", 0.98
        ))
        if not np.isfinite(target_cosine) or not -1.0 <= target_cosine <= 1.0:
            raise ValueError("FEATURE_SPACE_MIN_COSINE must be finite and in [-1, 1].")
        result = {
            "available": True,
            "num_sessions": int(len(per_session)),
            "num_classes": int(len(class_cosines)),
            "num_anchors": int(len(sample_cosines)),
            "class_prototype_cosine_mean": float(np.mean(class_cosines)),
            "class_prototype_cosine_min": float(min(class_cosines)),
            "sample_cosine_mean": float(np.mean(sample_cosines)),
            "sample_cosine_min": float(min(sample_cosines)),
            "minimum_cosine_target": target_cosine,
            "within_target": bool(min(class_cosines) >= target_cosine),
            "per_session": per_session,
        }
        print(
            f"=> [Historical-Feature-Space] sessions={result['num_sessions']} "
            f"classes={result['num_classes']} anchors={result['num_anchors']} "
            f"class_cosine_mean={result['class_prototype_cosine_mean']:.6f} "
            f"class_cosine_min={result['class_prototype_cosine_min']:.6f} "
            f"sample_cosine_mean={result['sample_cosine_mean']:.6f} "
            f"within_target={result['within_target']}"
        )
        return result

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

    @staticmethod
    def _l2sp_anchor_loss(module, anchor_parameters):
        current = dict(module.named_parameters())
        if not current or set(current) != set(anchor_parameters):
            raise ValueError("L2-SP anchors must exactly match migrated expert parameters.")
        terms = []
        for name, parameter in current.items():
            anchor = torch.as_tensor(anchor_parameters[name], device=parameter.device, dtype=torch.float32)
            if anchor.shape != parameter.shape or not torch.isfinite(anchor).all():
                raise ValueError(f"Invalid L2-SP anchor tensor: {name}")
            terms.append((parameter.float() - anchor).square().sum())
        loss = 0.5 * torch.stack(terms).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("L2-SP loss is non-finite.")
        return loss

    @torch.no_grad()
    def _collect_reference_cls_cache(self, loader):
        """Cache exact adapted layer-input CLS rows for future known-domain diagnostics."""
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
        """Store compact per-class anchors in each deployed adapted layer-input space."""
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

    def _semantic_retrieval_prior(
        self, semantic, base_semantic, base_visual, exclude_indices=None,
    ):
        """Retrieve a sparse Top-K Base visual prior from semantic similarities."""
        semantic = F.normalize(torch.as_tensor(semantic, device=self.device).float(), dim=-1)
        base_semantic = F.normalize(torch.as_tensor(base_semantic, device=self.device).float(), dim=-1)
        base_visual = F.normalize(torch.as_tensor(base_visual, device=self.device).float(), dim=-1)
        if (semantic.ndim != 2 or base_semantic.ndim != 2 or base_visual.ndim != 2
                or base_semantic.shape != base_visual.shape
                or semantic.size(1) != base_semantic.size(1)
                or not torch.isfinite(semantic).all()
                or not torch.isfinite(base_semantic).all()
                or not torch.isfinite(base_visual).all()):
            raise ValueError("Semantic retrieval requires finite, dimension-aligned prototype matrices.")
        temperature = float(_cfg_value(self._visual_moe_cfg(), "SEMANTIC_TEMPERATURE", 0.07))
        configured_top_k = int(_cfg_value(self._visual_moe_cfg(), "SEMANTIC_RETRIEVAL_TOP_K", 5))
        if not np.isfinite(temperature) or temperature <= 0 or configured_top_k < 1:
            raise ValueError("Semantic retrieval temperature and Top-K must be positive.")
        similarity = semantic @ base_semantic.t()
        if exclude_indices is not None:
            excluded = torch.as_tensor(exclude_indices, device=self.device, dtype=torch.long).reshape(-1)
            if excluded.numel() != semantic.size(0):
                raise ValueError("Semantic retrieval exclusion indices must align with query classes.")
            if (excluded < 0).any() or (excluded >= base_semantic.size(0)).any():
                raise ValueError("Semantic retrieval exclusion index is outside the Base bank.")
            similarity[torch.arange(semantic.size(0), device=self.device), excluded] = -torch.inf
            available = base_semantic.size(0) - 1
        else:
            available = base_semantic.size(0)
        if available < 1:
            raise ValueError("Semantic retrieval requires at least one eligible Base class.")
        top_k = min(configured_top_k, available)
        top_values, top_indices = torch.topk(similarity, k=top_k, dim=-1)
        weights = torch.softmax(top_values / temperature, dim=-1)
        retrieved = base_visual[top_indices]
        guided = F.normalize((weights.unsqueeze(-1) * retrieved).sum(dim=1), dim=-1)
        if top_k == 1:
            confidence = torch.ones(semantic.size(0), device=self.device)
        else:
            entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
            confidence = (1.0 - entropy / math.log(float(top_k))).clamp(0.0, 1.0)
        return guided, confidence, top_indices, weights

    @torch.no_grad()
    def _calibrate_semantic_alpha(self, base_state, loader):
        """Select semantic calibration strength by Base-support leave-one-source-out validation.

        This uses only the fixed Base training support and never reads the test
        split. Each query is classified against a visual prototype that excludes
        its own source; the semantic prior for a Base class also excludes that
        class from its retrieval bank to simulate a genuinely novel category.
        """
        configured_max = float(_cfg_value(
            self._visual_moe_cfg(), "SEMANTIC_CALIBRATION_MAX_ALPHA", 0.5
        ))
        if not np.isfinite(configured_max) or not 0.0 <= configured_max <= 1.0:
            raise ValueError("SEMANTIC_CALIBRATION_MAX_ALPHA must be finite and in [0, 1].")
        class_ids = torch.as_tensor(
            base_state.get("class_ids", base_state.get("class_index")),
            device=self.device, dtype=torch.long,
        ).reshape(-1)
        if class_ids.numel() < 2:
            return 0.0, {
                "available": False, "reason": "fewer_than_two_base_classes",
                "selected_alpha_max": 0.0, "validation_accuracy": None,
            }
        cache = self.model_without_dp.build_leave_one_out_prototypes(loader, class_ids)
        classes = torch.as_tensor(cache["classes"], device=self.device, dtype=torch.long).reshape(-1)
        labels = torch.as_tensor(cache["labels"], device=self.device, dtype=torch.long).reshape(-1)
        source_ids = torch.as_tensor(cache["source_ids"], device=self.device).reshape(-1)
        query_features = F.normalize(torch.as_tensor(cache["features"], device=self.device).float(), dim=-1)
        loo_visual = F.normalize(
            torch.as_tensor(cache["loo_visual_prototypes"], device=self.device).float(), dim=-1
        )
        if (not bool(torch.as_tensor(cache["loo_valid"]).all().item())
                or query_features.ndim != 2 or loo_visual.ndim != 3
                or loo_visual.size(0) != query_features.size(0)
                or loo_visual.size(1) != classes.numel()
                or labels.numel() != query_features.size(0)
                or source_ids.numel() != labels.numel()
                or torch.unique(source_ids).numel() != source_ids.numel()
                or not torch.isfinite(query_features).all()
                or not torch.isfinite(loo_visual).all()):
            raise ValueError("Base semantic-alpha validation requires aligned, source-exclusive LOO support rows.")
        if not torch.equal(classes.sort().values, class_ids.sort().values):
            raise ValueError("Base LOO classes do not match the Base classifier class ids.")
        state_ids = class_ids.tolist()
        row_map = {int(class_id): row for row, class_id in enumerate(state_ids)}
        order = torch.as_tensor(
            [row_map[int(class_id)] for class_id in classes.tolist()],
            device=self.device, dtype=torch.long,
        )
        base_visual = F.normalize(
            torch.as_tensor(base_state["image_proto"], device=self.device).float()[order], dim=-1
        )
        base_semantic = F.normalize(
            torch.as_tensor(base_state["description_proto"], device=self.device).float()[order], dim=-1
        )
        if base_visual.shape != base_semantic.shape or base_visual.size(0) != classes.numel():
            raise ValueError("Base visual and semantic banks do not align for alpha validation.")
        guided, confidence, _, _ = self._semantic_retrieval_prior(
            base_semantic, base_semantic, base_visual,
            exclude_indices=torch.arange(classes.numel(), device=self.device),
        )
        support_counts = torch.stack([labels.eq(class_id).sum() for class_id in classes]).float()
        loo_counts = support_counts.unsqueeze(0) - labels.unsqueeze(1).eq(classes.unsqueeze(0)).float()
        if (loo_counts <= 0).any():
            raise ValueError("Base semantic-alpha validation requires at least two support sources per class.")
        reliability_k = float(_cfg_value(self._visual_moe_cfg(), "PROTOTYPE_RELIABILITY_K", 5.0))
        if not np.isfinite(reliability_k) or reliability_k <= 0:
            raise ValueError("PROTOTYPE_RELIABILITY_K must be finite and positive.")
        alpha_factor = confidence.unsqueeze(0) * (
            1.0 - loo_counts / (loo_counts + reliability_k)
        )
        matches = labels.unsqueeze(1).eq(classes.unsqueeze(0))
        if not matches.any(dim=1).all():
            raise ValueError("Base LOO labels do not map to the Base classifier.")
        targets = matches.long().argmax(dim=1)
        best_alpha, best_accuracy = 0.0, -1.0
        candidate_rows = []
        for candidate in torch.linspace(0.0, configured_max, steps=11, device=self.device):
            alpha = candidate * alpha_factor
            calibrated = F.normalize(
                (1.0 - alpha.unsqueeze(-1)) * loo_visual
                + alpha.unsqueeze(-1) * guided.unsqueeze(0),
                dim=-1,
            )
            logits = torch.einsum("nd,ncd->nc", query_features, calibrated)
            accuracy = float(logits.argmax(dim=-1).eq(targets).float().mean().item())
            candidate_value = float(candidate.item())
            candidate_rows.append({"alpha_max": candidate_value, "accuracy": accuracy})
            # Iterating upward and updating only on strict improvement makes
            # ties prefer the more conservative, smaller calibration strength.
            if accuracy > best_accuracy + 1e-12:
                best_alpha, best_accuracy = candidate_value, accuracy
        diagnostics = {
            "available": True,
            "method": "base_support_leave_one_source_out",
            "selected_alpha_max": float(best_alpha),
            "validation_accuracy": float(best_accuracy),
            "query_count": int(query_features.size(0)),
            "base_class_count": int(classes.numel()),
            "retrieval_top_k": min(
                int(_cfg_value(self._visual_moe_cfg(), "SEMANTIC_RETRIEVAL_TOP_K", 5)),
                max(1, int(classes.numel()) - 1),
            ),
            "candidate_results": candidate_rows,
        }
        print(
            f"=> [Semantic-Calibration][Base-LOO] alpha_max={best_alpha:.3f} "
            f"accuracy={best_accuracy * 100.0:.2f}% queries={query_features.size(0)}"
        )
        return float(best_alpha), diagnostics

    def _semantic_guided_visual_prototype_calibration(self, visual, semantic, support_count, old_states):
        """Apply the same sparse Top-K Base-semantic calibration to full and LOO prototypes."""
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
            return visual, {
                "alpha": torch.zeros(expected_shape, device=self.device),
                "semantic_confidence": torch.zeros(classes, device=self.device),
                "visual_prototype_reliability": torch.ones(expected_shape, device=self.device),
            }
        base = old_states[0]
        base_visual = F.normalize(torch.as_tensor(base["image_proto"], device=self.device).float(), dim=-1)
        base_semantic = F.normalize(torch.as_tensor(base["description_proto"], device=self.device).float(), dim=-1)
        if base_visual.ndim != 2 or base_semantic.shape != base_visual.shape or base_visual.size(1) != dim:
            raise ValueError("Base visual and semantic prototype banks must align.")
        guided, confidence, retrieval_indices, retrieval_weights = self._semantic_retrieval_prior(
            semantic, base_semantic, base_visual
        )
        cfg = self._visual_moe_cfg()
        alpha_max = float(base.get(
            "semantic_calibration_alpha_max",
            getattr(cfg, "SEMANTIC_CALIBRATION_MAX_ALPHA", 0.5),
        ))
        reliability_k = float(getattr(cfg, "PROTOTYPE_RELIABILITY_K", 5.0))
        if (not np.isfinite(alpha_max) or not 0.0 <= alpha_max <= 1.0
                or not np.isfinite(reliability_k) or reliability_k <= 0):
            raise ValueError("Semantic calibration alpha or prototype reliability configuration is invalid.")
        reliability = support_count / (support_count + reliability_k)
        alpha = alpha_max * confidence * (1.0 - reliability)
        if visual.ndim == 3:
            guided = guided.unsqueeze(0)
        calibrated = F.normalize(
            (1.0 - alpha.unsqueeze(-1)) * visual + alpha.unsqueeze(-1) * guided,
            dim=-1,
        )
        if not torch.isfinite(calibrated).all() or (calibrated.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Semantic-guided calibration produced invalid prototypes.")
        return calibrated, {
            "alpha": alpha.detach(),
            "semantic_confidence": confidence.detach(),
            "visual_prototype_reliability": reliability.detach(),
            "retrieval_indices": retrieval_indices.detach(),
            "retrieval_weights": retrieval_weights.detach(),
        }

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

    def _global_incremental_logits(
        self, images, labels, source_ids, bank, scale, expected_encoder_step=None,
    ):
        if expected_encoder_step is not None:
            observed_step = bank.get("encoder_step")
            if observed_step is None or int(observed_step) != int(expected_encoder_step):
                raise RuntimeError(
                    "Incremental prototype bank is stale: "
                    f"bank_step={observed_step}, query_step={expected_encoder_step}."
                )
        if torch.is_grad_enabled():
            features, _ = self.model_without_dp.extract_img_feature_train(images)
        else:
            features = self.model_without_dp.extract_img_feature(images)
        features = F.normalize(features.float(), dim=-1)
        prototypes = bank["global_full"].unsqueeze(0).expand(images.size(0), -1, -1).clone()
        prototypes[:, int(bank["old_class_count"]):] = bank["novel_full"].unsqueeze(0)
        if source_ids is not None:
            source_ids = torch.as_tensor(source_ids, device=images.device).reshape(-1)
            if source_ids.numel() != images.size(0):
                raise ValueError("Training query source ids must align with the image batch.")
            support_source_ids = torch.as_tensor(
                bank["source_ids"], device=images.device
            ).reshape(-1)
            match = source_ids[:, None].eq(support_source_ids[None, :])
            if not bool(match.any(dim=1).all().item()):
                raise KeyError("Training query source is absent from the LOO support bank.")
            rows = match.long().argmax(dim=1)
            support_labels = torch.as_tensor(
                bank["labels"], device=labels.device, dtype=labels.dtype
            ).reshape(-1)[rows]
            if not torch.equal(support_labels, labels.reshape(-1)):
                raise ValueError(
                    "Training query source ids and labels disagree with the LOO support bank."
                )
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
            historical_feature_check = self._historical_feature_anchor_diagnostics(old_states)
            production_data_protocol = hasattr(self, "data_manager") and callable(
                getattr(self.data_manager, "get_base_source_dataloader", None)
            )
            if production_data_protocol and not feature_check.get("available", False):
                raise RuntimeError(
                    "Session diagnostics failed: production Base feature-space measurement is unavailable."
                )
            if production_data_protocol and not historical_feature_check.get("available", False):
                raise RuntimeError(
                    "Session diagnostics failed: historical class feature-anchor measurement is unavailable: "
                    f"{historical_feature_check.get('reason', 'unknown')}."
                )
            if production_data_protocol and not feature_check.get("within_target", False):
                raise RuntimeError(
                    "Session rejected: Base feature-space acceptance failed: "
                    f"cosine_min={feature_check.get('prototype_cosine_min')}, "
                    f"accuracy_drop={feature_check.get('base_reference_accuracy_drop')} pp, "
                    f"targets=({feature_check.get('minimum_cosine_target')}, "
                    f"{feature_check.get('max_accuracy_drop')} pp)."
                )
            if production_data_protocol and not historical_feature_check.get("within_target", False):
                raise RuntimeError(
                    "Session rejected: historical feature-anchor drift exceeded the configured cosine target: "
                    f"minimum={historical_feature_check.get('class_prototype_cosine_min')}, "
                    f"target={historical_feature_check.get('minimum_cosine_target')}."
                )
            # Historical visual prototypes remain immutable. Only the current
            # Session's novel prototypes are built and semantically calibrated.
            # Prototype construction and compact feature-anchor capture are
            # part of the same Session transaction, not post-commit side effects.
            result["task_state"] = self._build_and_refine_task_state(
                current_class_name, extract_loader, class_index, old_states,
                semantic_proto=semantic_proto, session_id=int(task_id),
            )
            result["feature_space_acceptance"] = self._cpu_copy(feature_check)
            result["historical_feature_space_acceptance"] = self._cpu_copy(historical_feature_check)
            return result
        except Exception:
            # Restore every layer, not just the layer where the exception arose.
            for block_idx, topology, state in snapshots:
                adapter = self._adapter_by_block(block_idx)
                adapter.rebuild_topology(topology)
                adapter.load_state_dict(state, strict=True)
                adapter.freeze_all()
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
        """Process insertion blocks shallow-to-deep with an isolated train/freeze cycle per layer."""
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        inc_cfg = self.cfg.TRAINER.BiMC.INCREMENTAL
        epochs = int(_cfg_value(inc_cfg, "EPOCHS", 6))
        scale = float(_cfg_value(inc_cfg, "LOGIT_SCALE", 100.0))
        descriptor_epochs = int(_cfg_value(
            optim_cfg, "INCREMENTAL_DESCRIPTOR_EPOCHS",
            _cfg_value(optim_cfg, "BASE_DESCRIPTOR_EPOCHS", 8),
        ))
        l2sp_weight = float(_cfg_value(inc_cfg, "L2SP_WEIGHT", 1e-4))
        feature_preservation_weight = float(_cfg_value(
            inc_cfg, "FEATURE_SPACE_PRESERVATION_WEIGHT", 1.0
        ))
        weight_decay = float(_cfg_value(optim_cfg, "WEIGHT_DECAY", 1e-4))
        if epochs < 1 or descriptor_epochs < 1:
            raise ValueError("Incremental expert and independent RD epochs must be positive.")
        if any(not np.isfinite(value) or value < 0 for value in (
            l2sp_weight, feature_preservation_weight, weight_decay
        )):
            raise ValueError(
                "Incremental L2-SP, feature preservation, and weight decay must be finite and non-negative."
            )
        if feature_preservation_weight > 0:
            old_prefix_tokens, old_final_features, anchor_batch_size = (
                self._historical_feature_anchor_batches(old_states)
            )
            if old_prefix_tokens.size(0) == 0:
                raise RuntimeError("Feature-space preservation requires non-empty historical anchors.")
        else:
            old_prefix_tokens = old_final_features = None
            anchor_batch_size = 0

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        # Historical visual prototypes are immutable. Feature-space replay
        # constrains newly trained experts; acceptance diagnostics reject drift.
        expected_classes = sorted(int(value) for value in class_index)
        decisions, expanded, reused, training = [], [], [], {}
        processed_blocks = []
        block_order = sorted(int(value) for value in self.model_without_dp.expandable_blocks())
        if not block_order:
            raise RuntimeError("Sequential AE-driven expansion requires at least one installed MoE block.")

        for layer_position, block_idx in enumerate(block_order):
            # Recompute both views after every shallower layer has completed
            # its REUSE/EXPAND decision and, when needed, expert optimization.
            # Consequently this block never plans from a stale all-layer snapshot.
            ref_map = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
            fit_map = self.model_without_dp.collect_blockwise_cls_features(train_loader)
            if set(ref_map) != set(fit_map) or block_idx not in ref_map:
                raise RuntimeError(
                    f"Sequential block {block_idx} has inconsistent augmented/reference feature maps."
                )
            ref_item, fit_item = ref_map[block_idx], fit_map[block_idx]
            ref_x = torch.as_tensor(ref_item["cls_in"], device=self.device).float()
            ref_y = torch.as_tensor(ref_item["labels"], device=self.device, dtype=torch.long).reshape(-1)
            ref_source_ids = torch.as_tensor(ref_item["source_ids"], device=self.device).reshape(-1)
            fit_x = torch.as_tensor(fit_item["cls_in"], device=self.device).float()
            fit_y = torch.as_tensor(fit_item["labels"], device=self.device, dtype=torch.long).reshape(-1)
            fit_source_ids = torch.as_tensor(fit_item["source_ids"], device=self.device).reshape(-1)
            observed_classes = sorted(int(value) for value in torch.unique(ref_y, sorted=True).tolist())
            if observed_classes != expected_classes:
                raise RuntimeError(
                    f"Block {block_idx} reference classes {observed_classes} do not match {expected_classes}."
                )
            if (ref_x.ndim != 2 or fit_x.ndim != 2 or ref_x.size(1) != fit_x.size(1)
                    or ref_x.size(0) == 0 or fit_x.size(0) == 0
                    or ref_y.numel() != ref_x.size(0) or fit_y.numel() != fit_x.size(0)
                    or ref_source_ids.numel() != ref_x.size(0)
                    or fit_source_ids.numel() != fit_x.size(0)
                    or torch.unique(ref_source_ids).numel() != ref_source_ids.numel()
                    or torch.unique(fit_source_ids).numel() != fit_source_ids.numel()
                    or not torch.isfinite(ref_x).all() or not torch.isfinite(fit_x).all()):
                raise ValueError(f"Block {block_idx} has empty, malformed, or non-finite Session layer inputs.")
            if set(ref_source_ids.tolist()) != set(fit_source_ids.tolist()):
                raise ValueError(f"Block {block_idx} RD fit/reference views do not share source identities.")
            fit_label_by_source = {
                int(source): int(label)
                for source, label in zip(fit_source_ids.tolist(), fit_y.tolist())
            }
            if any(
                fit_label_by_source.get(int(source)) != int(label)
                for source, label in zip(ref_source_ids.tolist(), ref_y.tolist())
            ):
                raise ValueError(f"Block {block_idx} RD fit/reference labels disagree by source id.")

            # RD fitting and reference-error calibration must use disjoint
            # support identities. Different augmentations of the same image
            # are not independent calibration samples.
            fit_mask, rd_reference_mask = self._source_exclusive_rd_split(
                fit_y, fit_source_ids, ref_y, ref_source_ids, per_class_reference=1,
            )
            rd_fit_x = fit_x[fit_mask]
            rd_fit_sources = fit_source_ids[fit_mask]
            rd_reference_x = ref_x[rd_reference_mask]
            rd_reference_sources = ref_source_ids[rd_reference_mask]
            if set(rd_fit_sources.tolist()) & set(rd_reference_sources.tolist()):
                raise RuntimeError(f"Block {block_idx} RD fit and calibration sources overlap.")

            decision = self.model_without_dp.plan_layer_expansion(block_idx, ref_map, class_index)
            decisions.append({
                "block_idx": block_idx,
                "layer_position": int(layer_position),
                "processed_blocks_before": list(processed_blocks),
                "expand": bool(decision.expand),
                "session_mean_z": float(decision.session_mean_z),
                "threshold": float(decision.threshold),
                "anchor_expert_id": int(decision.anchor_expert_id),
                "support_count": int(decision.support_count),
                "class_diagnostics": self._cpu_copy(decision.class_diagnostics or {}),
            })
            print(
                f"=> [Sequential-Expansion][B{block_idx}] position={layer_position + 1}/{len(block_order)} "
                f"completed_before={processed_blocks} decision={'EXPAND' if decision.expand else 'REUSE'}"
            )

            adapter = self._adapter_by_block(block_idx)
            if not decision.expand:
                history = self._train_reused_router(block_idx, ref_item)
                training[block_idx] = history
                reused.append({
                    "block_idx": block_idx,
                    "mode": "reuse_frozen",
                    "expert_count": int(adapter.num_experts),
                })
                processed_blocks.append(block_idx)
                continue

            anchor_id = int(decision.anchor_expert_id)
            expert_id = self.model_without_dp.add_expert(
                block_idx, task_id, anchor_expert_id=anchor_id
            )
            anchor_parameters = dict(adapter.experts[anchor_id].named_parameters())
            new_parameters = dict(adapter.experts[expert_id].named_parameters())
            if (set(anchor_parameters) != set(new_parameters) or any(
                anchor_parameters[name].shape != new_parameters[name].shape
                or not torch.equal(
                    anchor_parameters[name].detach(), new_parameters[name].detach()
                )
                for name in anchor_parameters
            )):
                raise RuntimeError("New expert failed exact nearest-historical parameter migration.")
            l2sp_anchor = {
                name: parameter.detach().clone()
                for name, parameter in new_parameters.items()
            }

            # RD fitting and error calibration are an independent stage. The
            # current block input is fixed because all shallower blocks are
            # already frozen; the new block's own residual is not part of its
            # input distribution.
            rd_loss = self._train_descriptor(
                adapter, expert_id, rd_fit_x, descriptor_epochs,
                sample_weights=None,
                calibration_values=rd_reference_x,
                calibration_weights=None,
            )
            descriptor = adapter.descriptors[expert_id]
            with torch.no_grad():
                rd_reference_mean_z = float(descriptor.standardized_score(rd_reference_x).mean().item())
                rd_effective_count = float(descriptor.effective_error_count.item())

            adapter.set_newest_trainable(descriptor=False)
            expert_parameters = list(adapter.experts[expert_id].parameters())
            router_parameters = list(adapter.router.columns[expert_id].parameters())
            if (not expert_parameters or not router_parameters
                    or any(not parameter.requires_grad for parameter in expert_parameters + router_parameters)
                    or any(parameter.requires_grad for parameter in descriptor.parameters())):
                raise RuntimeError(
                    f"Block {block_idx} must train only its newest expert and independent router after RD fitting."
                )
            optimizer = optim.AdamW(
                [
                    {"params": expert_parameters,
                     "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_EXPERT", 3e-4))},
                    {"params": router_parameters,
                     "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_ROUTER", 3e-4))},
                ],
                weight_decay=weight_decay,
                betas=tuple(_cfg_value(optim_cfg, "BETAS", (0.9, 0.999))),
            )
            history = {
                "mode": "expand_sequential",
                "expert_id": int(expert_id),
                "expert_anchor_id": anchor_id,
                "descriptor_anchor_id": int(descriptor.anchor_expert_id),
                "session_mean_z": float(decision.session_mean_z),
                "rd_training_loss": float(rd_loss),
                "rd_training_samples": int(rd_fit_x.size(0)),
                "rd_reference_samples": int(rd_reference_x.size(0)),
                "rd_reference_weighting": "uniform_source_exclusive_class_balanced",
                "rd_reference_mean_z": rd_reference_mean_z,
                "rd_effective_reference_count": rd_effective_count,
                "rd_training_source_ids": sorted(int(value) for value in rd_fit_sources.tolist()),
                "rd_reference_source_ids": sorted(int(value) for value in rd_reference_sources.tolist()),
                "rd_fit_reference_source_overlap": 0,
                "rd_fit_source_count_by_class": {
                    int(class_id): int(fit_y[fit_mask].eq(int(class_id)).sum().item())
                    for class_id in torch.unique(fit_y, sorted=True).tolist()
                },
                "rd_reference_source_count_by_class": {
                    int(class_id): int(ref_y[rd_reference_mask].eq(int(class_id)).sum().item())
                    for class_id in torch.unique(ref_y, sorted=True).tolist()
                },
                "loss": [],
                "ce_loss": [],
                "l2sp_loss": [],
                "feature_space_loss": [],
                "classification_samples_per_epoch": [],
                "classification_samples_total": 0,
            }

            history["prototype_refresh_policy"] = (
                "rebuild_source_exclusive_support_bank_before_every_optimizer_update"
            )
            history["prototype_feature_mode"] = "deterministic_eval_with_query_gradients"
            history["prototype_refreshes_per_epoch"] = []
            history["prototype_encoder_steps_per_epoch"] = []
            optimizer_update_step = 0

            for epoch in range(epochs):
                # Prototype-based classification requires a deterministic
                # feature function shared by support and query paths. eval()
                # disables expert dropout but does not disable autograd or
                # optimizer updates on the newest expert/router parameters.
                self.model.eval()
                totals = {"ce": 0.0, "l2sp": 0.0, "feature": 0.0, "count": 0}
                observed_train_classes = set()
                epoch_refreshes = 0
                epoch_encoder_steps = []
                for batch in train_loader:
                    images, labels = self.parse_batch(batch)
                    source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                    if not torch.isin(
                        labels, torch.as_tensor(class_index, device=labels.device)
                    ).all():
                        raise ValueError("Training loader contains labels outside the current Session.")
                    observed_train_classes.update(int(value) for value in labels.unique().tolist())

                    # Recompute support features after every preceding optimizer
                    # update. The bank and this query therefore use the exact
                    # same parameter version; an epoch-level cache is stale as
                    # soon as its first optimizer step changes the new expert.
                    self.model.eval()
                    bank = self._build_incremental_bank(
                        extract_loader, class_index, old_states, semantic_proto
                    )
                    bank["encoder_step"] = int(optimizer_update_step)
                    epoch_encoder_steps.append(int(optimizer_update_step))
                    epoch_refreshes += 1
                    self.model.eval()

                    optimizer.zero_grad(set_to_none=True)
                    logits, targets = self._global_incremental_logits(
                        images, labels, source_ids, bank, scale,
                        expected_encoder_step=optimizer_update_step,
                    )
                    ce_loss = F.cross_entropy(logits, targets)
                    l2sp_loss = self._l2sp_anchor_loss(
                        adapter.experts[expert_id], l2sp_anchor
                    )
                    feature_loss = ce_loss.new_zeros(())
                    if feature_preservation_weight > 0:
                        anchor_count = int(old_prefix_tokens.size(0))
                        selected_count = min(int(anchor_batch_size), anchor_count)
                        selected = torch.randperm(
                            anchor_count, device=self.device
                        )[:selected_count]
                        # Keep the same deterministic feature mode used by
                        # support-bank construction and query classification.
                        feature_loss = self._final_feature_anchor_loss(
                            old_prefix_tokens.index_select(0, selected),
                            old_final_features.index_select(0, selected),
                        )
                    total_loss = (
                        ce_loss
                        + l2sp_weight * l2sp_loss
                        + feature_preservation_weight * feature_loss
                    )
                    if not bool(torch.isfinite(total_loss).item()):
                        raise FloatingPointError(
                            f"Block {block_idx} incremental expert objective is non-finite."
                        )
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        expert_parameters + router_parameters, 1.0
                    )
                    optimizer.step()
                    optimizer_update_step += 1
                    count = int(labels.numel())
                    totals["ce"] += float(ce_loss.item()) * count
                    totals["l2sp"] += float(l2sp_loss.item()) * count
                    totals["feature"] += float(feature_loss.item()) * count
                    totals["count"] += count
                    history["classification_samples_total"] += count
                if (totals["count"] == 0
                        or sorted(observed_train_classes) != expected_classes):
                    raise RuntimeError(
                        f"Block {block_idx} incremental loader is empty or misses Session classes."
                    )
                denominator = float(totals["count"])
                history["ce_loss"].append(totals["ce"] / denominator)
                history["l2sp_loss"].append(totals["l2sp"] / denominator)
                history["feature_space_loss"].append(totals["feature"] / denominator)
                history["loss"].append(
                    totals["ce"] / denominator
                    + l2sp_weight * totals["l2sp"] / denominator
                    + feature_preservation_weight * totals["feature"] / denominator
                )
                history["classification_samples_per_epoch"].append(totals["count"])
                history["prototype_refreshes_per_epoch"].append(int(epoch_refreshes))
                history["prototype_encoder_steps_per_epoch"].append(epoch_encoder_steps)
                print(
                    f"=> [Sequential-Expert][B{block_idx}][{epoch + 1}/{epochs}] "
                    f"ce={history['ce_loss'][-1]:.5f} "
                    f"l2sp={history['l2sp_loss'][-1]:.6f} "
                    f"feature={history['feature_space_loss'][-1]:.6f} "
                    f"prototype_refreshes={epoch_refreshes}"
                )

            with torch.no_grad():
                history["l2sp_parameter_distance"] = float(torch.sqrt(
                    2.0 * self._l2sp_anchor_loss(adapter.experts[expert_id], l2sp_anchor)
                ).item())
            adapter.freeze_all()
            self._assert_moe_frozen(f"after sequential block {block_idx}")
            training[block_idx] = history
            expanded.append({
                "block_idx": block_idx,
                "expert_id": int(expert_id),
                "expert_anchor_id": anchor_id,
                "descriptor_anchor_id": int(descriptor.anchor_expert_id),
                "session_mean_z": float(decision.session_mean_z),
                "threshold": float(decision.threshold),
            })
            processed_blocks.append(block_idx)

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._assert_moe_frozen("incremental Session completion")
        return {
            "trained": bool(expanded),
            "decisions": decisions,
            "expanded": expanded,
            "reused": reused,
            "training": training,
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

            session_snapshot = self._snapshot_session_transaction(state_dict_list)
            try:
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
                        alpha_max, alpha_diagnostics = self._calibrate_semantic_alpha(
                            state, extract_loader
                        )
                        state["semantic_calibration_alpha_max"] = float(alpha_max)
                        state["semantic_calibration_validation"] = self._cpu_copy(alpha_diagnostics)
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
                feature_space_diagnostics["historical_anchors"] = self._historical_feature_anchor_diagnostics(
                    state_dict_list
                )
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
                    "feature_memory": self._feature_memory_summary(state_dict_list),
                    "evaluation_seconds": float(elapsed),
                })
                self.save_checkpoint(task_id, state_dict_list)
            except Exception:
                self._restore_session_transaction(session_snapshot, state_dict_list)
                raise

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
