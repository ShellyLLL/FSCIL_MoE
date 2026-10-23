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
from models.bimc import BiMC, build_loo_cache_from_embeddings, fit_reference_feature_map
from utils.evaluator import AccuracyEvaluator, compute_continual_metrics


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
    """FSCIL runner with predictive-risk expansion and versioned prototypes."""

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
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)
        self._resume_completed_session = -1
        self._resume_state_dict_list = []
        # The reference model is fixed after the base session; every session's
        # visual prototypes and query features are mapped into that same space.
        self._base_reference_router_states = None
        self._reference_space_id = "base_reference_v1"
        feature_dim = int(self.model_without_dp.clip_model.visual.output_dim)
        self._active_reference_feature_map = torch.eye(
            feature_dim, device=self.device, dtype=torch.float32
        )
        self._active_reference_alignment_info = {
            "fit_cosine_error": 0.0, "fit_samples": 0, "ridge": 0.0
        }
        self._reference_model_fingerprint = None
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
        space_ids = [state.get("prototype_space_id") for state in dict_list]
        if any(space_id is not None for space_id in space_ids):
            if any(space_id is None for space_id in space_ids) or len(set(space_ids)) != 1:
                raise ValueError(
                    "Cannot merge prototypes created under different or unknown feature-space references."
                )
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
        if space_ids and space_ids[0] is not None:
            result["prototype_space_id"] = space_ids[0]
        fingerprints = [state.get("reference_model_fingerprint") for state in dict_list]
        if any(value is not None for value in fingerprints):
            if any(value is None for value in fingerprints) or len(set(fingerprints)) != 1:
                raise ValueError(
                    "Cannot merge prototypes from different or unknown reference-model fingerprints."
                )
            active_fingerprint = getattr(self, "_reference_model_fingerprint", None)
            if active_fingerprint is not None and fingerprints[0] != active_fingerprint:
                raise ValueError("Prototype reference-model fingerprint does not match the active model.")
            result["reference_model_fingerprint"] = fingerprints[0]
        candidate_keys = (
            "description_proto", "text_features", "image_proto",
            "initial_fused_proto", "fused_proto", "anchor_fused_proto",
        )
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
        if "image_proto" not in result or "text_features" not in result:
            raise KeyError("Classifier state is missing image/text prototypes.")
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

    def _capture_base_reference_state(self):
        """Freeze post-base router columns and fingerprint the reference model."""
        self._base_reference_router_states = {
            str(block_idx): [
                self._cpu_copy(column.state_dict())
                for column in adapter.router.columns[:adapter.base_expert_count]
            ]
            for block_idx, adapter in self._moe_adapters()
        }
        self._reference_space_id = "base_reference_v1"
        self._reference_model_fingerprint = self._compute_reference_model_fingerprint()
        self._active_reference_feature_map = self._identity_reference_feature_map()
        self._active_reference_alignment_info = {
            "fit_cosine_error": 0.0, "fit_samples": 0, "ridge": 0.0
        }

    def _identity_reference_feature_map(self):
        feature_dim = int(self.model_without_dp.clip_model.visual.output_dim)
        return torch.eye(feature_dim, device=self.device, dtype=torch.float32)

    @staticmethod
    def _reference_alignment_hash(feature_map):
        value = torch.as_tensor(feature_map).detach().cpu().float().contiguous()
        return hashlib.sha256(value.numpy().tobytes()).hexdigest()

    def _compute_reference_model_fingerprint(self):
        """Hash only the immutable visual path used to produce reference features."""
        visual_state = self.model_without_dp.clip_model.visual.state_dict()
        base_counts = {
            int(block_idx): int(adapter.base_expert_count)
            for block_idx, adapter in self._moe_adapters()
        }
        digest = hashlib.sha256()

        def update_tensor(name, value):
            tensor = torch.as_tensor(value).detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

        for name, value in sorted(visual_state.items()):
            if ".moe_adapter.router.columns." in name:
                continue
            family = None
            for candidate in ("experts", "descriptors"):
                marker = f".moe_adapter.{candidate}."
                if marker in name:
                    family = (candidate, marker)
                    break
            if family is not None:
                candidate, marker = family
                try:
                    block_text = name.split(".resblocks.", 1)[1].split(".", 1)[0]
                    block_idx = int(block_text)
                    expert_idx = int(name.split(marker, 1)[1].split(".", 1)[0])
                except (IndexError, ValueError):
                    raise RuntimeError(f"Cannot identify base-reference tensor: {name}")
                if expert_idx >= base_counts.get(block_idx, 0):
                    continue
            update_tensor(name, value)

        for block_text, columns in sorted(
            (getattr(self, "_base_reference_router_states", None) or {}).items(),
            key=lambda item: int(item[0]),
        ):
            for column_idx, state in enumerate(columns):
                for name, value in sorted(state.items()):
                    update_tensor(
                        f"reference_router.block{block_text}.column{column_idx}.{name}",
                        value,
                    )
        return digest.hexdigest()

    def _reference_alignment_limit(self):
        incremental = getattr(self.cfg.TRAINER.BiMC, "INCREMENTAL", None)
        return float(getattr(incremental, "MAX_REFERENCE_ALIGNMENT_ERROR", 0.20))

    def _reference_alignment_allowed(self, values):
        limit = self._reference_alignment_limit()
        if limit < 0:
            return True
        return all(np.isfinite(float(value)) and float(value) <= limit for value in values)

    def _reference_map_ridge(self):
        incremental = getattr(self.cfg.TRAINER.BiMC, "INCREMENTAL", None)
        return float(getattr(incremental, "REFERENCE_MAP_RIDGE", 1.0))

    @torch.no_grad()
    def _collect_reference_feature_pairs(self, loader, reference_version):
        """Collect current/reference feature pairs in stable support-source order."""
        current_features, reference_features, source_ids = [], [], []
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in loader:
                images, labels = self.parse_batch(batch)
                ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                current = F.normalize(
                    self.model_without_dp.extract_img_feature(images).float(), dim=-1
                )
                reference = self._reference_features(images, reference_version)
                current_features.append(current.detach())
                reference_features.append(reference.detach())
                source_ids.append(ids.detach())
        finally:
            self.model.train(was_training)
        if not current_features:
            raise ValueError("Cannot fit reference alignment from an empty support loader.")
        ids = torch.cat(source_ids, dim=0)
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("Support source ids must be unique when fitting reference alignment.")
        return torch.cat(current_features, dim=0), torch.cat(reference_features, dim=0), ids

    @torch.no_grad()
    def _fit_reference_feature_map_from_loader(self, loader, reference_version):
        current, reference, source_ids = self._collect_reference_feature_pairs(
            loader, reference_version
        )
        feature_map, metrics = fit_reference_feature_map(
            current, reference, ridge=self._reference_map_ridge()
        )
        metrics["source_count"] = int(source_ids.numel())
        metrics["map_hash"] = self._reference_alignment_hash(feature_map)
        return feature_map.to(self.device), metrics

    def _snapshot_moe_state(self):
        """Capture topology and adapter tensors for transactional candidate updates."""
        return {
            "topology": copy.deepcopy(self.model_without_dp.export_moe_topology()),
            "adapter_states": {
                str(block_idx): self._cpu_copy(adapter.state_dict())
                for block_idx, adapter in self._moe_adapters()
            },
        }

    def _restore_moe_state(self, snapshot):
        """Restore the exact pre-candidate MoE state, including expert topology."""
        self.model_without_dp.rebuild_moe_topology(snapshot["topology"])
        for block_idx, adapter in self._moe_adapters():
            state = snapshot["adapter_states"].get(str(block_idx))
            if state is None:
                raise RuntimeError(f"Transactional MoE snapshot misses block {block_idx}.")
            adapter.load_state_dict(state, strict=True)
            adapter.freeze_all()

    def _reference_drift_limit(self):
        return float(getattr(
            self.cfg.TRAINER.BiMC.INCREMENTAL, "MAX_REFERENCE_DRIFT", -1.0
        ))

    def _reference_drift_allowed(self, values):
        limit = self._reference_drift_limit()
        if limit < 0:
            return True
        return all(np.isfinite(float(value)) and float(value) <= limit for value in values)

    @torch.no_grad()
    def _loader_reference_drift(self, loader, current_version, reference_version):
        """Measure mean cosine drift from the fixed base reference on a loader."""
        if not self._moe_adapters():
            return 0.0
        was_training = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        try:
            for batch in loader:
                images, labels = self.parse_batch(batch)
                drift = self._representation_drift(images, current_version, reference_version)
                total += float(drift.item()) * labels.numel()
                count += labels.numel()
        finally:
            self.model.train(was_training)
        return total / max(count, 1)

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
            "support_manifest",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Resume checkpoint is missing required fields: {missing}")
        schema = int(payload["schema_version"])
        if schema == 6 and int(payload["completed_session"]) == 0:
            # The base mixture is unchanged; only its erroneously reconstructed
            # visual prototype needs replacing with the saved true base mean.
            base_proto = payload.get("base_vision_prototype")
            if not isinstance(base_proto, torch.Tensor):
                raise ValueError("Base-only checkpoint lacks its true visual prototypes.")
            payload["task_states"][0] = dict(payload["task_states"][0])
            payload["task_states"][0]["image_proto"] = base_proto
            payload["task_states"][0]["encoder_version"] = 0
        elif schema not in (7, 8, 9, 10, 11, 12):
            raise ValueError("Checkpoint topology is incompatible; resume from Session 0.")
        if schema in (6, 7, 8, 9, 10, 11) and int(payload["completed_session"]) != 0:
            raise ValueError(
                "Incremental checkpoints predating validated reference-feature maps "
                "cannot be resumed safely; resume from a Session 0 checkpoint or restart from Session 0."
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
        saved_reference = payload.get("base_reference_router_states")
        if schema >= 11 and self._moe_adapters() and saved_reference is None:
            raise ValueError("Checkpoint is missing its fixed base-reference router state.")
        if saved_reference is None:
            # Safe for base-only legacy checkpoints: the loaded model is still the post-base reference.
            self._capture_base_reference_state()
        else:
            self._base_reference_router_states = self._cpu_copy(saved_reference)
            self._reference_space_id = str(payload.get("reference_space_id", "base_reference_v1"))
            computed_fingerprint = self._compute_reference_model_fingerprint()
            saved_fingerprint = payload.get("reference_model_fingerprint")
            if schema >= 12 and self._moe_adapters() and not saved_fingerprint:
                raise ValueError("Checkpoint is missing its reference-model fingerprint.")
            if saved_fingerprint is not None and saved_fingerprint != computed_fingerprint:
                raise ValueError(
                    "Checkpoint reference-model fingerprint does not match its frozen base reference."
                )
            self._reference_model_fingerprint = computed_fingerprint
        if schema >= 12:
            saved_map = payload.get("active_reference_feature_map")
            if not isinstance(saved_map, torch.Tensor):
                raise ValueError("Checkpoint is missing its active reference-feature map.")
            expected_shape = tuple(self._identity_reference_feature_map().shape)
            if tuple(saved_map.shape) != expected_shape or not torch.isfinite(saved_map).all():
                raise ValueError("Checkpoint reference-feature map has an invalid shape or non-finite values.")
            saved_map_hash = payload.get("active_reference_feature_map_hash")
            if saved_map_hash != self._reference_alignment_hash(saved_map):
                raise ValueError("Checkpoint reference-feature map hash mismatch.")
            self._active_reference_feature_map = saved_map.to(self.device).float()
            self._active_reference_alignment_info = payload.get(
                "active_reference_alignment_info",
                {"fit_cosine_error": 0.0, "fit_samples": 0, "ridge": 0.0},
            )
        else:
            self._active_reference_feature_map = self._identity_reference_feature_map()
            self._active_reference_alignment_info = {
                "fit_cosine_error": 0.0, "fit_samples": 0, "ridge": 0.0
            }
        manifest = payload.get("support_manifest")
        if manifest is None:
            raise ValueError("Resume checkpoint support_manifest must not be null.")
        self.data_manager.load_support_manifest(manifest)
        base_proto = payload.get("base_vision_prototype")
        self.model_without_dp.base_vision_prototype = (
            None if base_proto is None else base_proto.to(self.device)
        )
        if schema in (6, 7):
            base_loader = self.data_manager.get_dataloader(
                0, source="train", mode="test", accumulate_past=False)
            features, labels, _ = self.model_without_dp.inference_all_img_feature(
                base_loader, max_session=0)
            base_state = payload["task_states"][0]
            base_state["anchor_fused_proto"] = base_state["fused_proto"]
            base_state["anchor_features"] = features.detach().cpu().half()
            base_state["anchor_labels"] = labels.detach().cpu()
            sources = getattr(getattr(base_loader, "dataset", None), "source_ids", None)
            base_state["anchor_source_ids"] = (torch.arange(len(labels)) if sources is None else
                                               torch.as_tensor(sources, dtype=torch.long).cpu())
            base_state["creation_session"] = 0
        self._resume_state_dict_list = payload.get("task_states", [])
        for state in self._resume_state_dict_list:
            if "prototype_space_id" not in state:
                state["prototype_space_id"] = self._reference_space_id
                state["reference_session"] = 0
                state["prototype_space_legacy"] = True
            state_fingerprint = state.get("reference_model_fingerprint")
            if state_fingerprint is None and completed_session == 0 and schema < 12:
                state["reference_model_fingerprint"] = self._reference_model_fingerprint
                state["reference_alignment_hash"] = self._reference_alignment_hash(
                    self._identity_reference_feature_map()
                )
                state["reference_alignment_error"] = 0.0
            elif state_fingerprint != self._reference_model_fingerprint:
                raise ValueError(
                    "Checkpoint task prototype reference fingerprint is missing or inconsistent; "
                    "restart from Session 0."
                )
        # Legacy task-gate offsets are ignored: inference uses one global bank.
        self.acc_list = payload.get("acc_list", [])
        self.task_acc_list = payload.get("task_acc_list", [])
        self.session_diagnostics = diagnostics
        self._resume_completed_session = completed_session
        if hasattr(self.model_without_dp, "restore_fused_history"):
            self.model_without_dp.restore_fused_history(self._resume_state_dict_list)
        self._restore_rng_state(payload.get("rng_state"))
        print(f"=> Resumed session {self._resume_completed_session} from {path}")

    def save_checkpoint(self, completed_session, state_dict_list):
        if not self._checkpoint_enabled():
            return None
        directory = self._checkpoint_dir()
        directory.mkdir(parents=True, exist_ok=True)
        topology = self.model_without_dp.export_moe_topology()
        if self._reference_model_fingerprint is None:
            self._base_reference_router_states = self._base_reference_router_states or {}
            self._reference_model_fingerprint = self._compute_reference_model_fingerprint()
            for state in state_dict_list:
                state["reference_model_fingerprint"] = self._reference_model_fingerprint
                state.setdefault(
                    "reference_alignment_hash",
                    self._reference_alignment_hash(self._active_reference_feature_map),
                )
                state.setdefault("reference_alignment_error", 0.0)
        payload = {
            "schema_version": 12,
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
            "base_reference_router_states": self._cpu_copy(self._base_reference_router_states or {}),
            "reference_space_id": self._reference_space_id,
            "reference_model_fingerprint": self._reference_model_fingerprint,
            "active_reference_feature_map": self._cpu_copy(self._active_reference_feature_map),
            "active_reference_feature_map_hash": self._reference_alignment_hash(
                self._active_reference_feature_map
            ),
            "active_reference_alignment_info": self._cpu_copy(
                self._active_reference_alignment_info
            ),
            "task_states": self._cpu_copy(state_dict_list),
            "base_vision_prototype": self._cpu_copy(self.model_without_dp.base_vision_prototype),
            "support_manifest": self.data_manager.get_support_manifest(),
            "acc_list": list(self.acc_list),
            "task_acc_list": copy.deepcopy(self.task_acc_list),
            "session_diagnostics": self._cpu_copy(self.session_diagnostics),
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

    def _train_descriptor(self, adapter, expert_id, values, responsibilities, epochs):
        descriptor = adapter.descriptors[int(expert_id)]
        descriptor.float()
        values, responsibilities = values.detach(), responsibilities.detach().float()
        count = values.size(0)
        if count == 0:
            raise RuntimeError("descriptor training received no features.")
        order = torch.randperm(count, device=values.device)
        split = max(1, min(count - 1, int(round(count * 0.8)))) if count > 1 else 1
        train_idx = order[:split]
        calibration_idx = order[split:] if split < count else order
        for parameter in descriptor.parameters():
            parameter.requires_grad = True
        descriptor.train()
        optimizer = optim.AdamW(
            descriptor.parameters(),
            lr=float(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "LR_DESCRIPTOR", 1e-3)),
            weight_decay=0.0,
        )
        for _ in range(max(1, int(epochs))):
            optimizer.zero_grad(set_to_none=True)
            errors = descriptor.reconstruction_error(values[train_idx])
            weights = responsibilities[train_idx]
            loss = (weights * errors).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
        descriptor.eval()
        with torch.no_grad():
            errors = descriptor.reconstruction_error(values[calibration_idx])
            adapter.update_descriptor_stats(
                errors, expert_id, responsibilities[calibration_idx],
                fit_sample_count=train_idx.numel(),
            )
        for parameter in descriptor.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        return float(loss.item())

    def _train_base_descriptors(self, extract_loader):
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "BASE_DESCRIPTOR_EPOCHS", 8))
        features_by_block = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        print(f"\n========== Train Base Representation Descriptors ({epochs} epochs) ==========")
        for block_idx, adapter in self._moe_adapters():
            values = features_by_block[block_idx]["cls_in"].to(self.device)
            with torch.no_grad():
                _, responsibilities = adapter.router(values)
            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(
                    adapter, expert_id, values, responsibilities[:, expert_id], epochs
                )
                descriptor = adapter.descriptors[expert_id]
                print(
                    f"=> [Descriptor][B{block_idx}][E{expert_id}] loss={loss:.6f} "
                    f"mean={descriptor.mean.item():.6f} std={descriptor.std.item():.6f}"
                )
            adapter.assert_descriptors_ready()
            adapter.freeze_all()

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

        print(f"\n========== Start Base Training Task 0 (Epochs: {epochs}) ==========")
        for epoch in range(epochs):
            self.model.train()
            totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "lb": 0.0, "n": 0}
            pbar = tqdm(train_loader, desc=f"Base Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                images, labels = self.parse_batch(batch)
                targets = self._targets_to_columns(labels, task_stat)
                optimizer.zero_grad(set_to_none=True)
                logits, img_feat, teacher_feat, moe_aux = self.model_without_dp.forward_train(
                    images, task_stat, self.cfg.DATASET.BETA, compute_teacher=True
                )
                loss_ce = F.cross_entropy(logits, targets)
                loss_kd = 1.0 - F.cosine_similarity(img_feat, teacher_feat, dim=-1).mean()
                loss_lb = compute_load_balance_loss(moe_aux, num_experts).to(images.device)
                loss = ce_weight * loss_ce + lb_weight * loss_lb + kd_weight * loss_kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model_without_dp.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                totals["loss"] += float(loss.item()); totals["ce"] += float(loss_ce.item())
                totals["kd"] += float(loss_kd.item()); totals["lb"] += float(loss_lb.item()); totals["n"] += 1
                pbar.set_postfix({"Loss": f"{loss.item():.3f}", "CE": f"{loss_ce.item():.3f}"})
            scheduler.step()
            divisor = max(1, totals["n"])
            print(
                "=> [Base][Epoch {}/{}] loss={:.4f} ce={:.4f} kd={:.4f} lb={:.4f}".format(
                    epoch + 1, epochs, totals["loss"] / divisor, totals["ce"] / divisor,
                    totals["kd"] / divisor, totals["lb"] / divisor,
                )
            )

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        self._train_base_descriptors(extract_loader)

    # ------------------------------------------------------------------
    # Support-risk-guided expansion
    # ------------------------------------------------------------------
    def _current_encoder_version(self):
        return max((birth for _, adapter in self._moe_adapters()
                    for birth in adapter.expert_birth_sessions), default=0)

    @torch.no_grad()
    def _historical_prototype_bank(self):
        states = getattr(self, "_resume_state_dict_list", [])
        if not states: return None, None
        ids, prototypes = [], []
        for state in states:
            state_space = state.get(
                "prototype_space_id", getattr(self, "_reference_space_id", "base_reference_v1")
            )
            if state_space != self._reference_space_id:
                raise ValueError(
                    f"Historical prototype space {state_space!r} does not match "
                    f"the active reference {self._reference_space_id!r}."
                )
            active_fingerprint = getattr(self, "_reference_model_fingerprint", None)
            state_fingerprint = state.get("reference_model_fingerprint")
            if active_fingerprint is not None and state_fingerprint != active_fingerprint:
                raise ValueError(
                    "Historical prototype reference-model fingerprint is missing or incompatible."
                )
            class_ids = torch.as_tensor(state.get("class_ids", state.get("class_index")), device=self.device, dtype=torch.long).reshape(-1)
            fused = state.get("fused_proto")
            if fused is None: fused = self.model_without_dp.fuse_prototypes(state)
            fused = F.normalize(torch.as_tensor(fused, device=self.device).float(), dim=-1)
            if fused.ndim != 2 or fused.size(0) != class_ids.numel(): raise ValueError("Historical prototypes and ids do not align.")
            ids.append(class_ids); prototypes.append(fused)
        ids, prototypes = torch.cat(ids), torch.cat(prototypes)
        if torch.unique(ids).numel() != ids.numel(): raise ValueError("Duplicate historical class ids.")
        order = ids.argsort()
        return ids[order], prototypes[order]

    @torch.no_grad()
    def _support_bank(self, loader, class_index, text_features, description_proto, version, calibrate=True):
        """Build query-exclusive prototypes after mapping current features to reference space."""
        model = self.model_without_dp
        current_features, reference_features, source_ids = self._collect_reference_feature_pairs(
            loader, version
        )
        device = torch.device(self.device)
        labels = []
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in loader:
                _, batch_labels = self.parse_batch(batch)
                labels.append(batch_labels.detach())
        finally:
            self.model.train(was_training)
        labels = torch.cat(labels, dim=0).to(device).long()
        requested = torch.as_tensor(class_index, device=device, dtype=torch.long).reshape(-1)
        classes = torch.unique(requested, sorted=True)
        actual_classes = torch.unique(labels, sorted=True)
        if not torch.equal(classes, actual_classes):
            raise ValueError("Support labels do not match the requested incremental class ids.")
        feature_map, alignment = fit_reference_feature_map(
            current_features, reference_features, ridge=self._reference_map_ridge()
        )
        feature_map = feature_map.to(device)
        features = F.normalize(current_features.to(device).float() @ feature_map, dim=-1)
        current = build_loo_cache_from_embeddings(
            features.detach().cpu(), labels.detach().cpu(), source_ids.detach().cpu(),
            class_index=classes.detach().cpu(),
        )
        if not bool(current["loo_valid"].all().item()):
            raise ValueError("Mapped support cache contains a query without a same-class peer.")
        current["reference_feature_map"] = feature_map.detach()
        current["reference_alignment_fit_error"] = float(alignment["fit_cosine_error"])
        current["reference_alignment_fit_samples"] = int(alignment["fit_samples"])
        current["reference_alignment_ridge"] = float(alignment["ridge"])
        current["reference_alignment_hash"] = self._reference_alignment_hash(feature_map)

        features = current["features"].to(device).float()
        labels = current["labels"].to(device).long()
        classes = current["classes"].to(device).long()
        matches = classes[:, None].eq(requested[None, :])
        if not matches.sum(1).eq(1).all() or not matches.sum(0).eq(1).all():
            raise ValueError("Support class ids mismatch.")
        order_text = matches.long().argmax(1)
        text_features = torch.as_tensor(text_features, device=device).float()[order_text]
        description_proto = torch.as_tensor(description_proto, device=device).float()[order_text]
        visual_full = F.normalize(
            torch.stack([features[labels.eq(class_id)].mean(0) for class_id in classes]), dim=-1
        )
        visual_loo = current["loo_visual_prototypes"].to(device).float()
        if calibrate:
            visual_full = model.calibrate_incremental_visual(visual_full)
            visual_loo = model.calibrate_incremental_visual(
                visual_loo.reshape(-1, visual_loo.size(-1))
            ).reshape_as(visual_loo)
        lam = float(self.cfg.TRAINER.BiMC.LAMBDA_T) if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        text = F.normalize((1 - lam) * text_features + lam * description_proto, dim=-1)
        beta = float(self.cfg.DATASET.BETA)
        new_full = F.normalize(beta * text + (1 - beta) * visual_full, dim=-1)
        new_loo = F.normalize(beta * text.unsqueeze(0) + (1 - beta) * visual_loo, dim=-1)
        old_ids, old_proto = self._historical_prototype_bank()
        if old_ids is None:
            all_ids, full_bank, loo_bank = classes, new_full, new_loo
        else:
            if torch.isin(old_ids, classes).any():
                raise ValueError("Historical/current class ids overlap.")
            all_ids = torch.cat([old_ids, classes])
            order = all_ids.argsort()
            full_bank = torch.cat([old_proto, new_full], dim=0)[order]
            loo_bank = torch.cat([
                old_proto.unsqueeze(0).expand(new_loo.size(0), -1, -1), new_loo
            ], dim=1)[:, order, :]
        current["classes"] = all_ids.sort().values
        current["fused_full"], current["fused_loo"] = full_bank, loo_bank
        return current

    def _incremental_logits(self, images, labels, source_ids, support_bank,
                            version, scale, leave_one_out=True):
        """Rank novel classes as they are ranked in the deployed task classifier."""
        if leave_one_out:
            cache_sources = torch.as_tensor(support_bank["source_ids"], device=source_ids.device)
            matches = source_ids.reshape(-1, 1).eq(cache_sources.reshape(1, -1))
            if not bool(matches.any(dim=1).all().item()):
                raise KeyError("Training source is absent from its leave-one-out bank.")
            rows = matches.long().argmax(dim=1)
            new_prototypes = support_bank["fused_loo"][rows]
        else:
            new_prototypes = support_bank["fused_full"].unsqueeze(0).expand(images.size(0), -1, -1)
        if torch.is_grad_enabled():
            # Training/validation use the current complete model. The version
            # argument is retained only for backward-compatible call sites.
            features, _ = self.model_without_dp.extract_img_feature_train(images)
        else:
            # Validation and deployment both use the complete current expert pool.
            features = self.model_without_dp.extract_img_feature(images)
        features = F.normalize(features.float(), dim=-1)
        feature_map = support_bank.get("reference_feature_map")
        if feature_map is not None:
            features = F.normalize(
                features @ feature_map.to(device=features.device, dtype=features.dtype), dim=-1
            )
        logits = scale * torch.einsum("bd,bcd->bc", features, new_prototypes.float())
        return logits, self._targets_to_columns(labels, {"class_ids": support_bank["classes"]})

    def _text_prototypes(self, class_names, class_index):
        model = self.model_without_dp
        text, _ = model.inference_text_feature(class_names, model.template, int(class_index[0]))
        _, _, description = model.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH, int(class_index[0]))
        return text.detach(), description.detach()

    def _support_folds(self, train_loader, extract_loader):
        train_data, extract_data = train_loader.dataset, extract_loader.dataset
        if not np.array_equal(train_data.source_ids, extract_data.source_ids):
            raise RuntimeError("Train and extraction loaders must share source order.")
        labels = np.asarray(extract_data.labels)
        groups = [np.flatnonzero(labels == cls) for cls in np.unique(labels)]
        folds = min(len(group) for group in groups)
        # A held-out query and a source-exclusive training prototype require
        # at least three distinct sources per class.
        if folds < 3:
            return []
        result = []
        for fold in range(folds):
            held_out = sorted(int(group[fold]) for group in groups)
            fitting = sorted(set(range(len(labels))) - set(held_out))
            def loader(dataset, indices, shuffle):
                return DataLoader(Subset(dataset, indices), batch_size=train_loader.batch_size,
                                  shuffle=shuffle, num_workers=train_loader.num_workers,
                                  drop_last=False, pin_memory=True)
            result.append((loader(train_data, fitting, True),
                           loader(extract_data, fitting, False),
                           loader(extract_data, held_out, False)))
        return result

    @torch.no_grad()
    def _reference_features(self, images, previous_version):
        """Encode with the immutable post-base router state when available."""
        adapters = self._moe_adapters()
        base_snapshots = getattr(self, "_base_reference_router_states", None)
        if base_snapshots:
            live = {}
            try:
                for block, adapter in adapters:
                    reference_columns = base_snapshots.get(str(block), base_snapshots.get(block))
                    if reference_columns is None or len(reference_columns) != adapter.base_expert_count:
                        raise RuntimeError(
                            f"Fixed reference router state is incomplete for block {block}."
                        )
                    live[block] = [
                        copy.deepcopy(column.state_dict())
                        for column in adapter.router.columns[:adapter.base_expert_count]
                    ]
                    for column, state in zip(adapter.router.columns, reference_columns):
                        column.load_state_dict(state)
                return F.normalize(
                    self.model_without_dp.extract_img_feature(images, max_session=0).float(),
                    dim=-1,
                )
            finally:
                for block, adapter in adapters:
                    for column, state in zip(
                        adapter.router.columns[:adapter.base_expert_count], live.get(block, [])
                    ):
                        column.load_state_dict(state)

        # Compatibility fallback for callers that have not completed the base session.
        snapshots = getattr(self, "_candidate_reference_router_states", None)
        if not snapshots:
            return F.normalize(
                self.model_without_dp.extract_img_feature(images, max_session=previous_version).float(),
                dim=-1,
            )
        live = {}
        try:
            for block, adapter in adapters:
                live[block] = [copy.deepcopy(col.state_dict()) for col in adapter.router.columns]
                for col, state in zip(adapter.router.columns, snapshots.get(block, [])):
                    col.load_state_dict(state)
            return F.normalize(
                self.model_without_dp.extract_img_feature(images, max_session=previous_version).float(),
                dim=-1,
            )
        finally:
            for block, adapter in adapters:
                for col, state in zip(adapter.router.columns, live.get(block, [])):
                    col.load_state_dict(state)

    def _representation_drift(self, images, current_version, previous_version):
        # Before the base reference is captured, equal versions denote the same
        # function and should produce an exact zero without requiring a model stub.
        if current_version == previous_version and not getattr(
            self, "_base_reference_router_states", None
        ):
            return images.new_zeros(())
        previous = self._reference_features(images, previous_version)
        current, _ = self.model_without_dp.extract_img_feature_train(images, max_session=current_version)
        current = F.normalize(current.float(), dim=-1)
        return (1.0-(current*previous).sum(dim=-1)).mean().clamp_min(0.0)

    def _train_candidate(self, task_id, block_idx, train_loader, extract_loader,
                         class_index, text, description, calibrate):
        adapter = self._adapter_by_block(block_idx)
        adapter.set_newest_trainable()
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        optimizer = optim.AdamW([
            {"params": adapter.experts[-1].parameters(),
             "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
            {"params": adapter.router.columns[-1].parameters(),
             "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)},
        ], weight_decay=float(optim_cfg.WEIGHT_DECAY))
        scale = float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        history = []
        for epoch in range(int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)):
            bank = self._support_bank(extract_loader, class_index, text, description,
                                      task_id, calibrate)
            self.model.train()
            total_loss, count = 0.0, 0
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                optimizer.zero_grad(set_to_none=True)
                # Compute drift first: _representation_drift temporarily swaps router
                # state_dicts to evaluate the frozen reference model. Do this before
                # building the classification graph, otherwise load_state_dict mutates
                # router parameters saved for backward and triggers a version-counter error.
                stability_weight = float(getattr(self.cfg.TRAINER.BiMC.INCREMENTAL,
                                                 "STABILITY_WEIGHT", 0.0))
                stability_loss = None
                if stability_weight > 0.0:
                    stability_loss = self._representation_drift(
                        images, task_id, self._candidate_previous_version)
                logits, targets = self._incremental_logits(
                    images, labels, source_ids, bank, task_id, scale)
                loss = F.cross_entropy(logits, targets)
                if stability_loss is not None:
                    loss = loss + stability_weight * stability_loss
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite fused incremental loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.newest_parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * labels.numel()
                count += labels.numel()
            history.append({"epoch": epoch + 1, "loss": total_loss / max(count, 1)})
        adapter.freeze_all()
        return history

    def _train_router_only(self, task_id, train_loader, extract_loader, class_index,
                           text, description, calibrate):
        """Adapt router columns only; experts and the pretrained ViT stay frozen."""
        adapters = self._moe_adapters()
        parameters = []
        for _, adapter in adapters:
            adapter.set_router_trainable()
            parameters.extend(adapter.router.parameters())
        if not parameters:
            raise RuntimeError("Router-only adaptation requested without MoE adapters.")

        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        optimizer = optim.AdamW(
            [{"params": parameters,
              "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)}],
            weight_decay=float(optim_cfg.WEIGHT_DECAY),
        )
        scale = float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        history = []
        for epoch in range(int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)):
            bank = self._support_bank(
                extract_loader, class_index, text, description, task_id, calibrate
            )
            self.model.train()
            total_loss, count = 0.0, 0
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                optimizer.zero_grad(set_to_none=True)
                # _representation_drift restores reference router parameters in-place;
                # run it before constructing the classification autograd graph.
                stability = float(getattr(self.cfg.TRAINER.BiMC.INCREMENTAL, "STABILITY_WEIGHT", 0.0))
                stability_loss = None
                if stability > 0:
                    stability_loss = self._representation_drift(
                        images, task_id, self._candidate_previous_version
                    )
                logits, targets = self._incremental_logits(
                    images, labels, source_ids, bank, task_id, scale
                )
                loss = F.cross_entropy(logits, targets)
                if stability_loss is not None:
                    loss = loss + stability * stability_loss
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite router-only incremental loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * labels.numel()
                count += labels.numel()
            history.append({"epoch": epoch + 1, "loss": total_loss / max(count, 1)})
        for _, adapter in adapters:
            adapter.freeze_all()
        return history

    @torch.no_grad()
    def _held_out_loss(self, validation_loader, support_loader, class_index, text,
                       description, version, calibrate, previous_version=None,
                       return_metrics=False):
        """Score every candidate with the same classification-plus-drift objective."""
        self.model.eval()
        bank = self._support_bank(
            support_loader, class_index, text, description, version, calibrate
        )
        classification_total, drift_total, alignment_total, count = 0.0, 0.0, 0.0, 0
        stability_weight = float(getattr(
            self.cfg.TRAINER.BiMC.INCREMENTAL, "STABILITY_WEIGHT", 0.0
        ))
        feature_map = bank.get("reference_feature_map")
        for batch in validation_loader:
            images, labels = self.parse_batch(batch)
            logits, targets = self._incremental_logits(
                images, labels, None, bank, version,
                float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE), leave_one_out=False
            )
            classification_total += float(
                F.cross_entropy(logits, targets, reduction="sum").item()
            )
            if previous_version is not None:
                drift = self._representation_drift(images, version, previous_version)
                drift_total += float(drift.item()) * labels.numel()
            if feature_map is not None:
                current_features = F.normalize(
                    self.model_without_dp.extract_img_feature(images).float(), dim=-1
                )
                aligned_features = F.normalize(
                    current_features @ feature_map.to(
                        device=current_features.device, dtype=current_features.dtype
                    ), dim=-1
                )
                reference_features = self._reference_features(images, version)
                alignment_error = (
                    1.0 - (aligned_features * reference_features).sum(dim=-1)
                ).clamp_min(0.0)
                alignment_total += float(alignment_error.sum().item())
            count += labels.numel()
        count = max(count, 1)
        classification_loss = classification_total / count
        reference_drift = drift_total / count
        reference_alignment_error = alignment_total / count if feature_map is not None else 0.0
        alignment_passed = (
            self._reference_alignment_allowed([reference_alignment_error])
            if feature_map is not None else True
        )
        objective_loss = classification_loss + stability_weight * reference_drift
        metrics = {
            "loss": float(objective_loss),
            "classification_loss": float(classification_loss),
            "reference_drift": float(reference_drift),
            "reference_alignment_error": float(reference_alignment_error),
            "reference_alignment_limit": self._reference_alignment_limit(),
            "alignment_passed": bool(alignment_passed),
            "samples": int(count),
        }
        return metrics if return_metrics else objective_loss

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, enable_visual_calibration):
        """Run a transactional update; rejection or exceptions restore the exact MoE state."""
        snapshot = self._snapshot_moe_state()
        previous_map = self._active_reference_feature_map.detach().clone()
        previous_alignment_info = copy.deepcopy(self._active_reference_alignment_info)
        try:
            result = self._train_incremental_task_impl(
                task_id, train_loader, extract_loader, class_index,
                current_class_name, enable_visual_calibration,
            )
            accepted = bool(result.get("trained") or result.get("router_adapted"))
            if accepted:
                feature_map, alignment = self._fit_reference_feature_map_from_loader(
                    extract_loader, int(task_id)
                )
                if not self._reference_alignment_allowed([alignment["fit_cosine_error"]]):
                    self._restore_moe_state(snapshot)
                    self._active_reference_feature_map = previous_map
                    self._active_reference_alignment_info = previous_alignment_info
                    for decision in result.get("decisions", []):
                        if decision.get("accepted"):
                            decision["accepted"] = False
                            decision["postfit_reject_reason"] = "full_support_alignment_error_limit"
                    result["trained"] = False
                    result["router_adapted"] = False
                    result["expanded"] = []
                    result["reason"] = "full_support_reference_alignment_error_limit"
                    accepted = False
                else:
                    self._active_reference_feature_map = feature_map.detach()
                    self._active_reference_alignment_info = dict(alignment)
                    result["reference_alignment"] = dict(alignment)
            if not accepted:
                self._restore_moe_state(snapshot)
                self._active_reference_feature_map = previous_map
                self._active_reference_alignment_info = previous_alignment_info
            result["state_transaction"] = {
                "accepted": accepted,
                "rolled_back": not accepted,
                "reference_space_id": self._reference_space_id,
                "reference_model_fingerprint": self._reference_model_fingerprint,
                "reference_alignment_hash": self._reference_alignment_hash(
                    self._active_reference_feature_map
                ),
            }
            return result
        except Exception:
            self._restore_moe_state(snapshot)
            self._active_reference_feature_map = previous_map
            self._active_reference_alignment_info = previous_alignment_info
            raise

    def _train_incremental_task_impl(self, task_id, train_loader, extract_loader, class_index,
                                     current_class_name, enable_visual_calibration):
        text, description = self._text_prototypes(current_class_name, class_index)
        folds = self._support_folds(train_loader, extract_loader)
        previous_version = self._current_encoder_version()
        self._candidate_previous_version = previous_version
        for _, adapter in self._moe_adapters(): adapter.freeze_all()
        self._candidate_reference_router_states = {
            block: [copy.deepcopy(col.state_dict()) for col in adapter.router.columns]
            for block, adapter in self._moe_adapters()
        }
        if not folds:
            return {"trained": False, "router_adapted": False, "decisions": [],
                    "expanded": [], "encoder_version": previous_version,
                    "reason": "insufficient independent support folds"}

        null_metrics = [
            self._held_out_loss(
                val, fit_extract, class_index, text, description,
                previous_version, enable_visual_calibration,
                previous_version=previous_version, return_metrics=True,
            )
            for _, fit_extract, val in folds
        ]
        null_losses = [metric["loss"] for metric in null_metrics]

        # First test the lower-capacity option: update router columns only.
        initial_router_states = {
            block_idx: [copy.deepcopy(column.state_dict()) for column in adapter.router.columns]
            for block_idx, adapter in self._moe_adapters()
        }
        router_metrics = []
        for fit_train, fit_extract, validation in folds:
            for block_idx, adapter in self._moe_adapters():
                for column, state in zip(adapter.router.columns, initial_router_states[block_idx]):
                    column.load_state_dict(state)
                adapter.freeze_all()
            self._train_router_only(
                task_id, fit_train, fit_extract, class_index, text, description,
                enable_visual_calibration,
            )
            router_metrics.append(self._held_out_loss(
                validation, fit_extract, class_index, text, description,
                previous_version, enable_visual_calibration,
                previous_version=previous_version, return_metrics=True,
            ))
        router_losses = [metric["loss"] for metric in router_metrics]
        router_drifts = [metric["reference_drift"] for metric in router_metrics]
        router_alignment_errors = [metric["reference_alignment_error"] for metric in router_metrics]
        router_alignment_passed = all(metric["alignment_passed"] for metric in router_metrics)
        router_gains = np.asarray(null_losses) - np.asarray(router_losses)
        router_gain = float(np.mean(router_gains))
        router_gain_se = float(
            np.std(router_gains, ddof=1) / np.sqrt(len(router_gains))
        ) if len(router_gains) > 1 else float("inf")
        router_drift_max = float(max(router_drifts, default=0.0))
        router_drift_passed = self._reference_drift_allowed(router_drifts)
        drift_limit = self._reference_drift_limit()
        print(f"=> [RouterOnly][Task {task_id}] gain={router_gain:.5f} "
              f"se={router_gain_se:.5f} reference_drift_max={router_drift_max:.5f} "
              f"drift_passed={router_drift_passed} "
              f"alignment_error_max={max(router_alignment_errors, default=0.0):.5f} "
              f"alignment_passed={router_alignment_passed}")

        # Restore the exact pre-candidate state before deciding whether to keep it.
        for block_idx, adapter in self._moe_adapters():
            for column, state in zip(adapter.router.columns, initial_router_states[block_idx]):
                column.load_state_dict(state)
            adapter.freeze_all()
        decisions = [{
            "mode": "router_only",
            "null_loss": float(np.mean(null_losses)),
            "candidate_loss": float(np.mean(router_losses)),
            "null_classification_loss": float(np.mean([m["classification_loss"] for m in null_metrics])),
            "candidate_classification_loss": float(np.mean([m["classification_loss"] for m in router_metrics])),
            "gain": router_gain,
            "gain_se": router_gain_se,
            "reference_drift_max": router_drift_max,
            "reference_drift_limit": drift_limit,
            "drift_passed": bool(router_drift_passed),
            "reference_alignment_error_max": float(max(router_alignment_errors, default=0.0)),
            "reference_alignment_limit": self._reference_alignment_limit(),
            "alignment_passed": bool(router_alignment_passed),
            "accepted": False,
        }]
        if router_gain > router_gain_se and router_drift_passed and router_alignment_passed:
            history = self._train_router_only(
                task_id, train_loader, extract_loader, class_index, text,
                description, enable_visual_calibration,
            )
            postfit_drift = self._loader_reference_drift(
                extract_loader, task_id, previous_version
            )
            decisions[0]["postfit_reference_drift"] = float(postfit_drift)
            if self._reference_drift_allowed([postfit_drift]):
                decisions[0]["accepted"] = True
                print(f"=> [Accepted][Task {task_id}] router-only adaptation; no expert added.")
                return {"trained": False, "router_adapted": True,
                        "router_epochs": history, "decisions": decisions, "expanded": [],
                        "encoder_version": previous_version}
            decisions[0]["reject_reason"] = "postfit_reference_drift_limit"
            for block_idx, adapter in self._moe_adapters():
                for column, state in zip(adapter.router.columns, initial_router_states[block_idx]):
                    column.load_state_dict(state)
                adapter.freeze_all()
        else:
            if router_gain <= router_gain_se:
                decisions[0]["reject_reason"] = "insufficient_validation_gain"
            elif not router_drift_passed:
                decisions[0]["reject_reason"] = "reference_drift_limit"
            else:
                decisions[0]["reject_reason"] = "reference_alignment_error_limit"

        # Descriptor coverage gates candidate layers before predictive validation.
        coverage_features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
        expansion_candidates = []
        for block_idx in self.model_without_dp.expandable_blocks():
            coverage = self.model_without_dp.plan_layer_expansion(
                block_idx, extract_loader, class_index, block_features=coverage_features.get(block_idx))
            decisions.append({"mode":"descriptor_coverage","block_idx":int(block_idx),
                "expand":bool(coverage.expand),"class_scores":dict(coverage.class_scores),
                "uncovered_classes":list(coverage.uncovered_classes)})
            if coverage.expand: expansion_candidates.append(block_idx)
        if not expansion_candidates:
            for _, adapter in self._moe_adapters(): adapter.freeze_all()
            return {"trained":False,"router_adapted":False,"decisions":decisions,"expanded":[],
                    "encoder_version":previous_version,"reason":"existing descriptors cover support distribution"}
        route_mass = float(_cfg_value(self._visual_moe_cfg(),"INITIAL_EXPERT_ROUTE_MASS",0.01))
        for block_idx in expansion_candidates:
            adapter = self._adapter_by_block(block_idx)
            self.model_without_dp.add_expert(block_idx, task_id,
                reference_features=coverage_features[block_idx]["cls_in"].to(self.device),
                target_route_mass=route_mass)
            initial_expert = copy.deepcopy(adapter.experts[-1].state_dict())
            initial_router = copy.deepcopy(adapter.router.columns[-1].state_dict())
            candidate_metrics = []
            for fit_train, fit_extract, validation in folds:
                adapter.experts[-1].load_state_dict(initial_expert)
                adapter.router.columns[-1].load_state_dict(initial_router)
                self._train_candidate(task_id, block_idx, fit_train, fit_extract,
                                      class_index, text, description,
                                      enable_visual_calibration)
                candidate_metrics.append(self._held_out_loss(
                    validation, fit_extract, class_index, text, description,
                    task_id, enable_visual_calibration,
                    previous_version=previous_version, return_metrics=True,
                ))
            adapter.remove_newest_expert()
            candidate_losses = [metric["loss"] for metric in candidate_metrics]
            candidate_drifts = [metric["reference_drift"] for metric in candidate_metrics]
            candidate_alignment_errors = [
                metric["reference_alignment_error"] for metric in candidate_metrics
            ]
            alignment_passed = all(metric["alignment_passed"] for metric in candidate_metrics)
            gains = np.asarray(null_losses) - np.asarray(candidate_losses)
            drift_passed = self._reference_drift_allowed(candidate_drifts)
            decisions.append({
                "mode": "expert_expansion",
                "block_idx": block_idx,
                "null_loss": float(np.mean(null_losses)),
                "candidate_loss": float(np.mean(candidate_losses)),
                "null_classification_loss": float(np.mean([m["classification_loss"] for m in null_metrics])),
                "candidate_classification_loss": float(np.mean([m["classification_loss"] for m in candidate_metrics])),
                "gain": float(np.mean(gains)),
                "gain_se": float(np.std(gains, ddof=1) / np.sqrt(len(gains))),
                "reference_drift_max": float(max(candidate_drifts, default=0.0)),
                "reference_drift_limit": self._reference_drift_limit(),
                "drift_passed": bool(drift_passed),
                "reference_alignment_error_max": float(max(candidate_alignment_errors, default=0.0)),
                "reference_alignment_limit": self._reference_alignment_limit(),
                "alignment_passed": bool(alignment_passed),
                "accepted": False,
            })
            print(f"=> [Candidate][Task {task_id}][B{block_idx}] "
                  f"gain={decisions[-1]['gain']:.5f} se={decisions[-1]['gain_se']:.5f}")

        eligible = sorted(
            [
                row for row in decisions
                if row.get("mode") == "expert_expansion"
                and row["gain"] > row["gain_se"]
                and row["drift_passed"]
                and row["alignment_passed"]
            ],
            key=lambda row: row["gain"],
            reverse=True,
        )
        if not eligible:
            for _, adapter in self._moe_adapters():
                adapter.freeze_all()
            return {
                "trained": False, "router_adapted": False,
                "decisions": decisions, "expanded": [],
                "encoder_version": previous_version,
                "reason": "no expansion candidate passed gain and reference-drift constraints",
            }

        # Rank candidates by cross-validated gain, then guard the full-support fit.
        for candidate in eligible:
            block_idx = int(candidate["block_idx"])
            adapter = self._adapter_by_block(block_idx)
            expert_id = self.model_without_dp.add_expert(
                block_idx, task_id,
                reference_features=coverage_features[block_idx]["cls_in"].to(self.device),
                target_route_mass=route_mass,
            )
            epochs = self._train_candidate(
                task_id, block_idx, train_loader, extract_loader, class_index,
                text, description, enable_visual_calibration,
            )
            desc_features = self.model_without_dp.collect_blockwise_cls_features(
                extract_loader
            )[block_idx]["cls_in"].to(self.device)
            with torch.no_grad():
                _, responsibilities = adapter.router(desc_features)
            desc_loss = self._train_descriptor(
                adapter, expert_id, desc_features, responsibilities[:, expert_id],
                int(_cfg_value(
                    self.cfg.TRAINER.BiMC.OPTIM, "INCREMENTAL_DESCRIPTOR_EPOCHS", 5
                )),
            )
            adapter.freeze_all()
            if not adapter.descriptors[expert_id].is_ready():
                raise RuntimeError("New expert descriptor is not calibrated.")
            postfit_drift = self._loader_reference_drift(
                extract_loader, task_id, previous_version
            )
            candidate["postfit_reference_drift"] = float(postfit_drift)
            if not self._reference_drift_allowed([postfit_drift]):
                candidate["accepted"] = False
                candidate["postfit_reject_reason"] = "postfit_reference_drift_limit"
                adapter.remove_newest_expert()
                continue
            candidate["accepted"] = True
            return {
                "trained": True, "router_adapted": False,
                "decisions": decisions,
                "expanded": [{
                    "block_idx": block_idx, "expert_id": expert_id,
                    "epochs": epochs, "descriptor_loss": float(desc_loss),
                }],
                "encoder_version": task_id,
            }

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        return {
            "trained": False, "router_adapted": False,
            "decisions": decisions, "expanded": [],
            "encoder_version": previous_version,
            "reason": "all cross-validated expansion candidates failed the post-fit drift guard",
        }

    @torch.no_grad()
    def _rebuild_unified_task_states(self, through_task, previous_states=None):
        """Append current-session prototypes without reopening historical support images."""
        self.model.eval(); through_task=int(through_task)
        if through_task == 0:
            self.model_without_dp.base_vision_prototype=None
            self.model_without_dp._fused_history_prototypes=None
            self.model_without_dp._fused_history_class_ids=None
            states=[]
        else:
            states=list(previous_states if previous_states is not None else getattr(self,"_resume_state_dict_list",[]))
            if len(states)!=through_task: raise RuntimeError(f"Expected {through_task} historical states, got {len(states)}.")
            if self.model_without_dp.base_vision_prototype is None: raise RuntimeError("Base visual prototypes missing; resume from Session 0.")
        ids=self.data_manager.class_index_in_task[through_task]
        names=np.array(self.data_manager.class_names)[ids].tolist()
        loader=self.data_manager.get_dataloader(through_task,source="train",mode="test",accumulate_past=False)
        state = self.model_without_dp.build_task_statistics(
            names, loader, ids,
            calibrate_novel_vision_proto=(
                through_task > 0 and bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
            ),
            encoder_version=None,
            image_feature_map=self._active_reference_feature_map,
        )
        state["class_ids"] = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        state["creation_session"] = through_task
        state["encoder_version"] = self._current_encoder_version()
        # model_revision captures router-only updates too; encoder_version only tracks expert birth.
        state["model_revision"] = through_task
        state["prototype_space_id"] = self._reference_space_id
        state["reference_session"] = 0
        state["reference_alignment_hash"] = self._reference_alignment_hash(
            self._active_reference_feature_map
        )
        state["reference_alignment_error"] = float(
            self._active_reference_alignment_info.get("fit_cosine_error", 0.0)
        )
        state["reference_alignment_ridge"] = float(
            self._active_reference_alignment_info.get("ridge", 0.0)
        )
        if self._reference_model_fingerprint is not None:
            state["reference_model_fingerprint"] = self._reference_model_fingerprint
        state["prototype_refresh_mode"] = "current_session_support_no_replay"
        state["reference_drift"] = float(
            self._loader_reference_drift(loader, through_task, through_task)
        )
        return states + [state]

    def run(self):
        configured_end = int(_cfg_value(self.cfg.TRAINER.BiMC, "END_SESSION", -1))
        end_task = self.data_manager.num_tasks - 1 if configured_end < 0 else min(
            configured_end, self.data_manager.num_tasks - 1
        )
        print(f"Start unified FSCIL inference on all tasks: [0, {end_task}]")
        state_dict_list = list(self._resume_state_dict_list)
        start_task = self._resume_completed_session + 1
        use_moe = bool(
            getattr(self.cfg.TRAINER.BiMC, "VISUAL_MOE", None)
            and self._visual_moe_cfg().ENABLE
        )
        if start_task > end_task:
            print("=> Resume checkpoint already contains all configured FSCIL sessions.")
            return

        for task_id in range(start_task, end_task + 1):
            self.model.eval()
            plan_summary, adaptation, final_routing = None, None, {}
            class_index = self.data_manager.class_index_in_task[task_id]
            current_class_name = np.array(self.data_manager.class_names)[class_index].tolist()
            extract_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="test", accumulate_past=False
            )
            train_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="train", accumulate_past=False
            )
            print(f"\n=> [Session {task_id}] support_manifest={self._support_manifest_hash()}")
            if use_moe:
                self._log_expert_topology(f"task{task_id}-before")

            if task_id == 0:
                base_task_state = self.model_without_dp.build_task_statistics(
                    current_class_name, extract_loader, class_index,
                    calibrate_novel_vision_proto=False, encoder_version=None,
                )
                base_task_state["class_ids"] = torch.as_tensor(
                    class_index, device=self.device, dtype=torch.long
                )
                if use_moe:
                    self.train_base_task(base_task_state, train_loader, extract_loader)
                    # Capture the reference only after base experts/router are trained.
                    self._capture_base_reference_state()
                adaptation = {
                    "trained": bool(use_moe), "router_adapted": False,
                    "decisions": [], "expanded": [], "encoder_version": 0,
                    "mode": "base_session",
                }
            else:
                enable_visual_calibration = bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                if use_moe:
                    self._routing_diagnostics(
                        extract_loader, f"task{task_id}-before", current_task=task_id
                    )
                    adaptation = self.train_incremental_task(
                        task_id, train_loader, extract_loader, class_index,
                        current_class_name, enable_visual_calibration,
                    )
                    plan_summary = self._cpu_copy(adaptation.get("decisions", []))
                else:
                    adaptation = {
                        "trained": False, "router_adapted": False,
                        "decisions": [], "expanded": [],
                        "encoder_version": 0, "mode": "frozen_bimc",
                    }

            # Refresh prototypes for every seen class after any accepted update.
            self.model.eval()
            state_dict_list = self._rebuild_unified_task_states(task_id, previous_states=state_dict_list)
            self._resume_state_dict_list = state_dict_list
            merged_state = self.merge_dicts(state_dict_list)

            if use_moe:
                final_routing = self._routing_diagnostics(
                    extract_loader, f"task{task_id}-final", current_task=task_id
                )
                self._log_expert_topology(f"task{task_id}-final")

            start_time = time.perf_counter()
            acc = self.inference_task_bilevel(task_id, merged_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            print(f"+++++++++++ task {task_id}, time: {elapsed:.3f}s ++++++++++++++++")
            print(
                f"=> Task [{task_id}], Acc: {acc['mean_acc']:.3f}, "
                f"Base: {acc['base_avg_acc']:.2f}, Novel: {acc['inc_avg_acc']:.2f}, "
                f"H: {acc['harmonic_acc']:.2f}"
            )
            self.acc_list.append(round(acc["mean_acc"], 3))
            self.task_acc_list.append(acc["task_acc"])
            self.session_diagnostics.append({
                "session": int(task_id),
                "support_manifest_hash": self._support_manifest_hash(),
                "plan": plan_summary,
                "adaptation": self._cpu_copy(adaptation),
                "routing": self._cpu_copy(final_routing),
                "expert_counts": {
                    int(block_idx): int(adapter.num_experts)
                    for block_idx, adapter in self._moe_adapters()
                },
                "classifier": self._cpu_copy(acc),
                "evaluation_seconds": float(elapsed),
                "classifier_mode": "single_current_model_global_prototype_bank",
            })
            self.save_checkpoint(task_id, state_dict_list)

        print(f"\nFinal acc: {self.acc_list}")
        metrics = compute_continual_metrics(self.acc_list, self.task_acc_list)
        print(f"=> AA (mean session accuracy, %): {metrics['aa']:.3f}")
        print(
            "=> PD (initial overall acc - final overall acc, pp): "
            f"{metrics['pd']:.3f} "
            f"(initial={metrics['initial_acc']:.3f}, final={metrics['final_acc']:.3f})"
        )
        print(
            "=> Final Novel average accuracy (%): "
            f"{metrics['final_novel_avg_acc']:.3f}"
        )
        # Keep the former old-task forgetting statistic as a diagnostic only.
        print(
            "=> Diagnostic old-task average accuracy drop (pp): "
            f"{metrics['old_task_avg_drop']:.3f}; "
            f"per-task={metrics['old_task_drops']}"
        )
        print("Task-wise acc:")
        for index, task_acc in enumerate(self.task_acc_list):
            print(f"task {index:2d}, acc: {task_acc}")

    @torch.no_grad()
    def inference_task_bilevel(self, task_id, state_dict):
        """Classify all seen classes from one current-model feature extraction."""
        beta = float(self.cfg.DATASET.BETA)
        lambda_t = (
            float(self.cfg.TRAINER.BiMC.LAMBDA_T)
            if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        )
        image_proto = F.normalize(state_dict["image_proto"].to(self.device).float(), dim=-1)
        text_features = state_dict["text_features"].to(self.device).float()
        description_proto = state_dict["description_proto"].to(self.device).float()
        text_proto = F.normalize(
            (1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1
        )
        bimc_proto = F.normalize(
            beta * text_proto + (1.0 - beta) * image_proto, dim=-1
        )
        stored_refined = state_dict.get("fused_proto")
        refined_proto = F.normalize(
            bimc_proto if stored_refined is None else stored_refined.to(self.device).float(),
            dim=-1,
        )
        prototype_banks = {
            "visual": image_proto,
            "text": text_proto,
            "bimc": bimc_proto,
            "refined": refined_proto,
        }
        test_loader = self.data_manager.get_dataloader(task_id, source="test", mode="test")
        all_logits = {name: [] for name in prototype_banks}
        all_targets = []
        for batch in tqdm(test_loader, desc=f"Eval Task {task_id}"):
            images, targets = self.parse_batch(batch)
            # Exactly one feature extraction through the current complete model.
            features = F.normalize(
                self.model_without_dp.extract_img_feature(images).float(), dim=-1
            )
            feature_map = getattr(self, "_active_reference_feature_map", None)
            if feature_map is not None:
                features = F.normalize(
                    features @ feature_map.to(
                        device=features.device, dtype=features.dtype
                    ), dim=-1
                )
            for name, prototypes in prototype_banks.items()
                all_logits[name].append(
                    100.0 * features @ F.normalize(prototypes, dim=-1).t()
                )
            all_targets.append(targets)

        logits_by_variant = {
            name: torch.cat(values, dim=0) for name, values in all_logits.items()
        }
        targets = torch.cat(all_targets, dim=0)
        variants = {
            name: self.evaluator.calc_accuracy(
                logits, targets, task_id, class_ids=state_dict["class_ids"]
            )
            for name, logits in logits_by_variant.items()
        }
        result = dict(variants["refined"])
        result["classifier_variants"] = variants
        result["error_flow"] = self.evaluator.error_flow(
            logits_by_variant["refined"], targets, task_id,
            class_ids=state_dict["class_ids"],
        )
        print(
            "=> [Unified classifier] visual={:.2f} text={:.2f} BiMC={:.2f} refined={:.2f}".format(
                variants["visual"]["mean_acc"], variants["text"]["mean_acc"],
                variants["bimc"]["mean_acc"], variants["refined"]["mean_acc"],
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

    def parse_batch(self, batch):
        return batch["image"].to(self.device), batch["label"].to(self.device, dtype=torch.long)
