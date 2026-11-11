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
        self.task_gate_biases = []
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
        result["encoder_versions"] = torch.cat([
            torch.full_like(ids, int(state.get("encoder_version", 0)))
            for ids, state in zip(per_state_ids, dict_list)
        ])[order]
        result["task_ids"] = torch.cat([
            torch.full_like(ids, int(state.get("creation_session", index)))
            for index, (ids, state) in enumerate(zip(per_state_ids, dict_list))
        ])[order]
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
        elif schema not in (7, 8, 9):
            raise ValueError("Checkpoint topology is incompatible; resume from Session 0.")
        if schema in (7, 8) and int(payload["completed_session"]) != 0:
            raise ValueError("The calibrated gate and task-local objective need a base-only "
                             "checkpoint or a schema 9 checkpoint.")
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
        self.task_gate_biases = ([0.0] if schema in (6, 7, 8) else
                                 list(payload["task_gate_biases"]))
        if len(self.task_gate_biases) != completed_session + 1:
            raise ValueError("Checkpoint task gate biases do not match completed sessions.")
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
        payload = {
            "schema_version": 9,
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
            "task_states": self._cpu_copy(state_dict_list),
            "task_gate_biases": list(self.task_gate_biases),
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

    def _train_descriptor(self, adapter, expert_id, values, epochs):
        """Fit the SEMA linear AE on the full local representation distribution."""
        descriptor = adapter.descriptors[int(expert_id)]
        descriptor.float()
        values = values.detach().float()
        if values.ndim != 2 or values.size(0) < 3:
            raise ValueError("AE fitting requires at least three CLS representations.")
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
                loss = F.mse_loss(reconstruction, values[indices])
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite AE reconstruction loss.")
                loss.backward()
                optimizer.step()
        descriptor.eval()
        with torch.no_grad():
            errors = descriptor.reconstruction_error(values)
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
        for block_idx, adapter in self._moe_adapters():
            # Recollect per layer after previous descriptors are frozen; the
            # CLS input is exactly the local representation seen by that layer.
            features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
            values = features[block_idx]["cls_in"].to(self.device)
            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(adapter, expert_id, values, epochs)
                descriptor = adapter.descriptors[expert_id]
                print(f"=> [SEMA-AE][B{block_idx}][E{expert_id}] loss={loss:.6f} "
                      f"reference_n={int(descriptor.error_count.item())} "
                      f"mean={descriptor.error_mean.item():.6f} std={descriptor.error_std.item():.6f}")
            adapter.assert_descriptors_ready()
            adapter.freeze_all()

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

    def _semantic_visual_calibration(self, visual, semantic, old_states):
        """Use semantic similarity to transfer reliable old visual directions."""
        visual = F.normalize(visual.float(), dim=-1)
        semantic = F.normalize(semantic.float(), dim=-1)
        if not old_states:
            return visual
        old_visual = torch.cat([
            F.normalize(torch.as_tensor(state["image_proto"], device=self.device).float(), dim=-1)
            for state in old_states
        ], dim=0)
        old_semantic = []
        lambda_t = (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
                    if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
        for state in old_states:
            text = torch.as_tensor(state["text_features"], device=self.device).float()
            desc = torch.as_tensor(state.get("description_proto", text), device=self.device).float()
            old_semantic.append(F.normalize((1.0 - lambda_t) * text + lambda_t * desc, dim=-1))
        old_semantic = torch.cat(old_semantic, dim=0)
        tau = float(getattr(self.cfg.TRAINER.BiMC, "TAU", 16.0))
        weights = torch.softmax(tau * semantic @ old_semantic.t(), dim=-1)
        transferred = F.normalize(weights @ old_visual, dim=-1)
        strength = float(getattr(self.cfg.TRAINER.BiMC, "SEMANTIC_VISUAL_CALIBRATION", 0.1))
        return F.normalize((1.0 - strength) * visual + strength * transferred, dim=-1)

    def _build_incremental_bank(self, loader, class_index, text, description, old_states):
        """Build source-exclusive novel prototypes and one global old+new bank."""
        cache = self.model_without_dp.build_leave_one_out_prototypes(loader, class_index)
        classes = cache["classes"].to(self.device)
        labels = cache["labels"].to(self.device)
        sources = cache["source_ids"].to(self.device)
        features = cache["features"].to(self.device).float()
        full_visual = F.normalize(torch.stack([
            features[labels.eq(cls)].mean(0) for cls in classes
        ]), dim=-1)
        loo_visual = cache["loo_visual_prototypes"].to(self.device).float()
        semantic = F.normalize(
            (1.0 - (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
                    if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)) * text.float()
            + (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
               if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0) * description.float(),
            dim=-1,
        )
        full_calibrated = self._semantic_visual_calibration(full_visual, semantic, old_states)
        loo_calibrated = torch.stack([
            self._semantic_visual_calibration(row, semantic, old_states)
            for row in loo_visual
        ])
        beta = float(self.cfg.DATASET.BETA)
        novel_full = F.normalize(beta * semantic + (1.0 - beta) * full_calibrated, dim=-1)
        novel_loo = F.normalize(beta * semantic.unsqueeze(0) +
                                (1.0 - beta) * loo_calibrated, dim=-1)
        old_fused = []
        old_ids = []
        for state in old_states:
            old_fused.append(F.normalize(torch.as_tensor(
                state["fused_proto"], device=self.device).float(), dim=-1))
            old_ids.append(torch.as_tensor(state["class_ids"], device=self.device).long())
        old_fused = torch.cat(old_fused, dim=0) if old_fused else novel_full.new_empty((0, novel_full.size(-1)))
        old_ids = torch.cat(old_ids, dim=0) if old_ids else classes.new_empty(0)
        global_full = torch.cat([old_fused, novel_full], dim=0)
        global_ids = torch.cat([old_ids, classes], dim=0)
        if torch.unique(global_ids).numel() != global_ids.numel():
            raise ValueError("Global prototype classifier contains duplicate class ids.")
        return {
            "classes": classes, "labels": labels, "source_ids": sources,
            "visual_full": full_calibrated, "visual_loo": loo_calibrated,
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
            prototypes[torch.arange(images.size(0), device=images.device)[:, None],
                       (int(bank["old_class_count"]) + torch.arange(
                           bank["classes"].numel(), device=images.device))[None, :]] = bank["novel_full"][None, :, :]
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
                    images, task_stat, self.cfg.DATASET.BETA, compute_teacher=True
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

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, old_states):
        """SEMA-ordered layerwise expansion followed by global-prototype training."""
        text, description = self._text_prototypes(current_class_name, class_index)
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        epochs = int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)
        scale = float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        threshold = float(getattr(self._visual_moe_cfg(), "EXPANSION_Z_THRESHOLD", 1.0))
        ae_epochs = int(_cfg_value(optim_cfg, "INCREMENTAL_DESCRIPTOR_EPOCHS", 5))
        decisions, expanded, histories = [], [], {}
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        # Scan shallow-to-deep. Recollect features after each accepted layer
        # because an upstream expansion changes all downstream representations.
        for block_idx in self.model_without_dp.expandable_blocks():
            features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
            decision = self.model_without_dp.plan_layer_expansion(
                block_idx, features, class_index, threshold)
            decisions.append({
                "block_idx": int(block_idx), "expand": bool(decision.expand),
                "descriptor_scores": dict(decision.class_scores),
                "uncovered_classes": list(decision.uncovered_classes),
            })
            if not decision.expand:
                continue

            adapter = self._adapter_by_block(block_idx)
            expert_id = self.model_without_dp.add_expert(block_idx, task_id)
            adapter.set_newest_trainable()
            anchor_mean, anchor_var = self._all_old_anchors(old_states, block_idx)
            optimizer = optim.AdamW([
                {"params": adapter.experts[-1].parameters(),
                 "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                {"params": adapter.router.columns[-1].parameters(),
                 "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)},
            ], weight_decay=float(optim_cfg.WEIGHT_DECAY))
            loss_history = []
            for epoch in range(epochs):
                bank = self._build_incremental_bank(
                    extract_loader, class_index, text, description, old_states)
                self.model.train()
                total, count = 0.0, 0
                for batch in train_loader:
                    images, labels = self.parse_batch(batch)
                    source_ids = self._batch_source_ids(batch, labels.numel(), labels.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, targets = self._global_incremental_logits(
                        images, labels, source_ids, bank, scale)
                    classification = F.cross_entropy(logits, targets)
                    old_suppression = adapter.old_anchor_suppression_loss(
                        anchor_mean, anchor_var,
                        residual_weight=float(getattr(
                            self.cfg.TRAINER.BiMC.INCREMENTAL, "OLD_RESIDUAL_WEIGHT", 1.0)))
                    gate_penalty = torch.sigmoid(adapter.router.columns[-1](
                        adapter.layer_norm(adapter.experts[-1].c_fc.weight.new_zeros(
                            (1, adapter.d_model))))).square().mean() * 0.0
                    loss = (classification
                            + float(getattr(self.cfg.TRAINER.BiMC.INCREMENTAL,
                                            "OLD_SUPPRESSION_WEIGHT", 1.0)) * old_suppression
                            + gate_penalty)
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("Non-finite global prototype incremental loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.newest_parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.item()) * labels.numel()
                    count += labels.numel()
                loss_history.append(total / max(1, count))
            adapter.freeze_all()

            # The descriptor is paired with the newly learned expert and is
            # fitted only after that expert's local representation is final.
            new_features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)
            local_values = new_features[block_idx]["cls_in"].to(self.device)
            ae_loss = self._train_descriptor(adapter, expert_id, local_values, ae_epochs)
            adapter.freeze_all()
            histories[block_idx] = {
                "expert_id": int(expert_id), "loss": loss_history, "ae_loss": ae_loss,
                "old_anchor_gate_mean": adapter.old_anchor_response(anchor_mean, anchor_var),
            }
            expanded.append({"block_idx": int(block_idx), "expert_id": int(expert_id)})

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

    def _representation_drift(self, images, current_version, previous_version):
        """Differentiable cosine drift on current support images only.

        The teacher is the frozen previous encoder version. This introduces no
        rehearsal images or changes to historical experts and prototypes.
        """
        if current_version == previous_version:
            return images.new_zeros((), dtype=torch.float32)
        with torch.no_grad():
            previous = F.normalize(self.model_without_dp.extract_img_feature(
                images, max_session=previous_version).float(), dim=-1)
        current, _ = self.model_without_dp.extract_img_feature_train(
            images, max_session=current_version)
        current = F.normalize(current.float(), dim=-1)
        return (1.0 - (current * previous).sum(dim=-1)).mean().clamp_min(0.0)

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
                logits, targets = self._incremental_logits(
                    images, labels, source_ids, bank, task_id, scale)
                loss = F.cross_entropy(logits, targets)
                stability_weight = float(getattr(self.cfg.TRAINER.BiMC.INCREMENTAL,
                                                 "STABILITY_WEIGHT", 0.0))
                if stability_weight > 0.0:
                    loss = loss + stability_weight * self._representation_drift(
                        images, task_id, self._candidate_previous_version)
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

    @torch.no_grad()
    def _held_out_loss(self, validation_loader, support_loader, class_index, text,
                       description, version, calibrate, previous_version=None):
        self.model.eval()
        bank = self._support_bank(support_loader, class_index, text, description,
                                  version, calibrate)
        total, count = 0.0, 0
        for batch in validation_loader:
            images, labels = self.parse_batch(batch)
            logits, targets = self._incremental_logits(
                images, labels, None, bank, version,
                float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE), leave_one_out=False)
            value = F.cross_entropy(logits, targets, reduction="sum")
            stability_weight = float(getattr(self.cfg.TRAINER.BiMC.INCREMENTAL,
                                             "STABILITY_WEIGHT", 0.0))
            if (stability_weight > 0.0 and previous_version is not None
                    and version != previous_version):
                with torch.no_grad():
                    before = F.normalize(self.model_without_dp.extract_img_feature(
                        images, max_session=previous_version).float(), dim=-1)
                    after = F.normalize(self.model_without_dp.extract_img_feature(
                        images, max_session=version).float(), dim=-1)
                    drift = (1.0 - (after * before).sum(dim=-1)).clamp_min(0.0).sum()
                    value = value + stability_weight * drift
            total += float(value.item())
            count += labels.numel()
        return total / max(count, 1)

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, enable_visual_calibration):
        text, description = self._text_prototypes(current_class_name, class_index)
        folds = self._support_folds(train_loader, extract_loader)
        previous_version = self._current_encoder_version()
        self._candidate_previous_version = previous_version
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        if not folds:
            return {"trained": False, "decisions": [], "expanded": [],
                    "encoder_version": previous_version}
        null_losses = [self._held_out_loss(val, fit_extract, class_index, text, description,
                                           previous_version, enable_visual_calibration)
                       for _, fit_extract, val in folds]
        decisions = []
        for block_idx in self.model_without_dp.expandable_blocks():
            adapter = self._adapter_by_block(block_idx)
            self.model_without_dp.add_expert(block_idx, task_id)
            initial_expert = copy.deepcopy(adapter.experts[-1].state_dict())
            initial_router = copy.deepcopy(adapter.router.columns[-1].state_dict())
            candidate_losses = []
            for fit_train, fit_extract, validation in folds:
                adapter.experts[-1].load_state_dict(initial_expert)
                adapter.router.columns[-1].load_state_dict(initial_router)
                self._train_candidate(task_id, block_idx, fit_train, fit_extract,
                                      class_index, text, description,
                                      enable_visual_calibration)
                candidate_losses.append(self._held_out_loss(
                    validation, fit_extract, class_index, text, description,
                    task_id, enable_visual_calibration, previous_version=previous_version))
            adapter.remove_newest_expert()
            gains = np.asarray(null_losses) - np.asarray(candidate_losses)
            decisions.append({"block_idx": block_idx, "null_loss": float(np.mean(null_losses)),
                              "candidate_loss": float(np.mean(candidate_losses)),
                              "gain": float(np.mean(gains)),
                              "gain_se": float(np.std(gains, ddof=1) / np.sqrt(len(gains)))})
            print(f"=> [Candidate][Task {task_id}][B{block_idx}] "
                  f"gain={decisions[-1]['gain']:.5f} se={decisions[-1]['gain_se']:.5f}")
        best = max(decisions, key=lambda row: row["gain"], default=None)
        # One-standard-error selection favours the null model on five-shot noise.
        if best is None or best["gain"] <= best["gain_se"]:
            return {"trained": False, "decisions": decisions, "expanded": [],
                    "encoder_version": previous_version}
        block_idx = best["block_idx"]
        adapter = self._adapter_by_block(block_idx)
        expert_id = self.model_without_dp.add_expert(block_idx, task_id)
        epochs = self._train_candidate(task_id, block_idx, train_loader, extract_loader,
                                       class_index, text, description,
                                       enable_visual_calibration)
        print(f"=> [Accepted][Task {task_id}][B{block_idx}] expert={expert_id} "
              f"validation_gain={best['gain']:.5f}")
        return {"trained": True, "decisions": decisions,
                "expanded": [{"block_idx": block_idx, "expert_id": expert_id,
                              "epochs": epochs}], "encoder_version": task_id}

    def _build_and_refine_task_state(self, current_class_name, extract_loader, class_index,
                                     old_states):
        state = self.model_without_dp.build_task_statistics(
            current_class_name, extract_loader, class_index,
            calibrate_novel_vision_proto=False,
        )
        text = state["text_features"].to(self.device).float()
        desc = state["description_proto"].to(self.device).float()
        lambda_t = (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
                    if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
        semantic = F.normalize((1.0 - lambda_t) * text + lambda_t * desc, dim=-1)
        visual = self._semantic_visual_calibration(
            state["image_proto"].to(self.device), semantic, old_states)
        beta = float(self.cfg.DATASET.BETA)
        fused = F.normalize(beta * semantic + (1.0 - beta) * visual, dim=-1)
        state["image_proto"] = visual.detach()
        state["fused_proto"] = fused.detach()
        state["initial_fused_proto"] = fused.detach()
        state["semantic_proto"] = semantic.detach()
        state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
        state["class_index"] = state["class_ids"]
        state["encoder_version"] = 0
        state["anchor_by_block"] = self._collect_class_anchors(extract_loader)
        return state

    def _anchor_task_scores(self, features, state, query_loo=None, query_labels=None):
        """Comparable task evidence from the single frozen base encoder."""
        prototypes = F.normalize(state["anchor_fused_proto"].to(features.device).float(), dim=-1)
        queries = F.normalize(features.float(), dim=-1)
        scores = 100.0 * queries @ prototypes.T
        task_ids = state["task_ids"].to(features.device)
        if query_loo is not None:
            if query_labels is None or query_loo.shape != queries.shape:
                raise ValueError("Source-exclusive prototypes need one label and weight per query.")
            columns = self._targets_to_columns(query_labels, state)
            rows = torch.arange(len(queries), device=features.device)
            scores[rows, columns] = 100.0 * (queries * F.normalize(
                query_loo.to(features.device).float(), dim=-1)).sum(dim=-1)
        tasks = torch.unique(task_ids, sorted=True)
        return torch.stack([torch.logsumexp(scores[:, task_ids.eq(task)], dim=1)
                            for task in tasks], dim=1).double()

    def _anchor_leave_one_out(self, task_state):
        """Recompute only each query's own class, excluding its source."""
        features = F.normalize(task_state["anchor_features"].to(self.device).float(), dim=-1)
        labels = task_state["anchor_labels"].to(self.device)
        sources = torch.as_tensor(task_state.get(
            "anchor_source_ids", torch.arange(len(labels))), device=self.device)
        classes = self._targets_to_columns(labels, task_state)
        _, source_rows = torch.unique(torch.stack([classes, sources], dim=1),
                                      dim=0, return_inverse=True)
        n_classes = len(task_state["class_ids"])
        class_sums = features.new_zeros((n_classes, features.size(1))).index_add_(0, classes, features)
        source_sums = features.new_zeros((int(source_rows.max()) + 1, features.size(1)))
        source_sums.index_add_(0, source_rows, features)
        class_counts = torch.bincount(classes, minlength=n_classes)
        source_counts = torch.bincount(source_rows)
        if not bool((class_counts[classes] > source_counts[source_rows]).all().item()):
            raise ValueError("Task gate calibration needs at least two sources per class.")
        visual = F.normalize(class_sums[classes] - source_sums[source_rows], dim=-1)
        if (int(task_state["creation_session"]) > 0 and
                self.cfg.TRAINER.BiMC.VISION_CALIBRATION):
            visual = self.model_without_dp.soft_calibration(
                self.model_without_dp.base_vision_prototype, visual)
        weight = (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
                  if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
        text = F.normalize((1.0 - weight) * task_state["text_features"] +
                           weight * task_state["description_proto"], dim=-1).to(self.device)
        beta = float(self.cfg.DATASET.BETA)
        return F.normalize(beta * text[classes] + (1.0 - beta) * visual, dim=-1)

    def _calibrate_task_gate(self, task_id, states):
        """Fit all task offsets to class-balanced, source-exclusive anchor evidence."""
        if task_id == 0:
            self.task_gate_biases = [0.0]
            return {"new_task_bias": 0.0}
        if len(self.task_gate_biases) != task_id:
            raise ValueError("Historical task gate biases are missing.")
        merged = self.merge_dicts(states)
        with torch.no_grad():
            evidences, labels, task_targets, counts = [], [], [], []
            for state in states:
                features = state["anchor_features"].to(self.device).float()
                targets = state["anchor_labels"].to(self.device)
                evidences.append(self._anchor_task_scores(
                    features, merged, self._anchor_leave_one_out(state), targets))
                labels.append(targets)
                task_targets.append(torch.full_like(targets, int(state["creation_session"])))
                counts.append(len(targets))
            evidence = torch.cat(evidences)
            class_labels = torch.cat(labels)
            tasks = torch.cat(task_targets)
            columns = self._targets_to_columns(class_labels, merged)
            class_counts = torch.bincount(columns, minlength=len(merged["class_ids"]))
            sample_weights = class_counts[columns].reciprocal().double()

        offsets = nn.Parameter(evidence.new_tensor([*self.task_gate_biases[1:], 0.0]))
        optimizer = optim.LBFGS([offsets], lr=1.0, max_iter=200,
                                tolerance_grad=1e-9, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = evidence + torch.cat([offsets.new_zeros(1), offsets])[None, :]
            loss = (F.cross_entropy(logits, tasks, reduction="none") * sample_weights).sum()
            loss = loss / sample_weights.sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        if not bool(torch.isfinite(offsets).all().item()):
            raise FloatingPointError("Non-finite calibrated task offsets.")
        self.task_gate_biases = [0.0, *offsets.detach().cpu().tolist()]
        with torch.no_grad():
            logits = evidence + evidence.new_tensor(self.task_gate_biases)
            correct = logits.argmax(dim=1).eq(tasks)
            old_count = sum(counts[:-1])
            result = {"new_task_bias": self.task_gate_biases[-1],
                      "balanced_loo_nll": float((F.cross_entropy(
                          logits, tasks, reduction="none") * sample_weights).sum() /
                          sample_weights.sum()),
                      "novel_loo_task_recall": float(correct[old_count:].float().mean()),
                      "old_loo_task_recall": float(correct[:old_count].float().mean())}
        print(f"=> [AnchorGate][Task {task_id}] "
              f"novel_loo_task_recall={result['novel_loo_task_recall']:.3f} "
              f"old_loo_task_recall={result['old_loo_task_recall']:.3f} "
              f"balanced_loo_nll={result['balanced_loo_nll']:.3f}")
        return result

    def _retained_scores(self, anchor_features, state, versioned_logits):
        """Task evidence from the anchor; expert logits rank classes within a task."""
        task_ids = state["task_ids"].to(versioned_logits.device)
        evidence = self._anchor_task_scores(anchor_features, state)
        evidence += evidence.new_tensor(self.task_gate_biases)
        result = torch.empty_like(versioned_logits, dtype=torch.float64)
        for task in torch.unique(task_ids, sorted=True).tolist():
            mask = task_ids.eq(task)
            local = versioned_logits[:, mask].double()
            result[:, mask] = evidence[:, task:task + 1] + local - local.max(dim=1, keepdim=True).values
        return result

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
            self.task_gate_biases = [0.0] * len(state_dict_list)
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
            print(f"=> Session [{task_id}], AA={acc['mean_acc']:.3f}, "
                  f"Base={acc['base_avg_acc']:.2f}, Novel={acc['inc_avg_acc']:.2f}, "
                  f"H={acc['harmonic_acc']:.2f}, PD={acc.get('performance_drop', 0.0):.3f}")
            self.acc_list.append(round(acc["mean_acc"], 3))
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
        prototypes = F.normalize(state_dict["fused_proto"].to(self.device).float(), dim=-1)
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
