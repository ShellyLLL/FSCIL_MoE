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
from tqdm import tqdm

from datasets.data_manager import DatasetManager
from models.bimc import BiMC
from utils.evaluator import AccuracyEvaluator


def compute_load_balance_loss(moe_aux_list, num_experts):
    """Base-session load balance over the immutable four-way base router."""
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
    """FSCIL runner with descriptor-driven, layer-wise MoE expansion."""

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
        self.historical_anchor_bank = {}
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
        candidate_keys = (
            "description_proto", "text_features", "image_proto",
            "initial_fused_proto", "fused_proto",
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
            block: {"weight_sum": torch.zeros(adapter.base_expert_count),
                    "top1": torch.zeros(adapter.base_expert_count),
                    "dynamic": torch.zeros(adapter.num_incremental_experts() + 1),
                    "count": 0}
            for block, adapter in adapters.items()
        }
        self.model.eval()
        for batch in loader:
            images, labels = self.parse_batch(batch)
            _, aux_list = self.model_without_dp.extract_img_feature_train(images)
            for (block_idx, adapter), aux in zip(self._moe_adapters(), aux_list):
                weights = aux["route_weights"].detach().float().cpu()
                top1 = weights.argmax(dim=-1)
                item = totals[block_idx]
                item["weight_sum"] += weights.sum(dim=0)
                item["top1"] += F.one_hot(top1, adapter.base_expert_count).sum(dim=0)
                dynamic = aux["incremental_route_indices"].detach().cpu()
                dynamic = torch.where(
                    dynamic.lt(0), torch.zeros_like(dynamic),
                    dynamic - adapter.base_expert_count + 1,
                )
                item["dynamic"] += F.one_hot(
                    dynamic, adapter.num_incremental_experts() + 1
                ).sum(dim=0)
                item["count"] += weights.size(0)
        output = {}
        for block_idx, item in totals.items():
            count = max(1, item["count"])
            means, usage = item["weight_sum"] / count, item["top1"] / count
            adapter = adapters[block_idx]
            dynamic_usage = item["dynamic"] / count
            base_rows = [
                f"base-e{expert_id} mean={means[expert_id]:.3f} top1={usage[expert_id]:.3f}"
                for expert_id in range(adapter.base_expert_count)
            ]
            dynamic_rows = [f"NULL={dynamic_usage[0]:.3f}"]
            dynamic_rows.extend(
                f"inc-e{adapter.base_expert_count + index}={dynamic_usage[index + 1]:.3f}"
                for index in range(adapter.num_incremental_experts())
            )
            print(f"=> [Router][{tag}][B{block_idx}] " + " | ".join(base_rows + dynamic_rows))
            output[block_idx] = {
                "base_mean_weight": means, "base_top1_usage": usage,
                "incremental_usage_with_null": dynamic_usage,
            }
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
        if int(payload["schema_version"]) != 7:
            raise ValueError(
                "Checkpoint schema is incompatible with separated routing v2; "
                "retrain from Session 0."
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
        self.historical_anchor_bank = payload.get("historical_anchor_bank", {})
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
            "schema_version": 7,
            "implementation_version": "2.0.0-rc1",
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
            "task_states": self._cpu_copy(state_dict_list),
            "base_vision_prototype": self._cpu_copy(self.model_without_dp.base_vision_prototype),
            "support_manifest": self.data_manager.get_support_manifest(),
            "acc_list": list(self.acc_list),
            "task_acc_list": copy.deepcopy(self.task_acc_list),
            "session_diagnostics": self._cpu_copy(self.session_diagnostics),
            "historical_anchor_bank": self._cpu_copy(self.historical_anchor_bank),
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

    def _collect_support_views(self, task_id, view_ids):
        """Collect and concatenate fixed source-aligned views for every MoE block."""
        merged = {}
        for view_id in view_ids:
            loader = self.data_manager.get_support_view_dataloader(
                task_id, int(view_id), mode="train"
            )
            current = self.model_without_dp.collect_blockwise_cls_features(loader)
            for block_idx, item in current.items():
                target = merged.setdefault(block_idx, {key: [] for key in item})
                for key, value in item.items():
                    target[key].append(value)
        return {
            block_idx: {key: torch.cat(values, dim=0) for key, values in item.items()}
            for block_idx, item in merged.items()
        }

    def _train_descriptor(self, adapter, expert_id, fit_values, fit_responsibilities,
                          calibration_values, calibration_responsibilities, epochs):
        descriptor = adapter.descriptors[int(expert_id)]
        descriptor.float()
        fit_values = fit_values.detach()
        fit_responsibilities = fit_responsibilities.detach().float()
        calibration_values = calibration_values.detach()
        calibration_responsibilities = calibration_responsibilities.detach().float()
        if fit_values.size(0) == 0 or calibration_values.size(0) == 0:
            raise RuntimeError("descriptor fit and calibration views must both be non-empty.")
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
            errors = descriptor.reconstruction_error(fit_values)
            weights = fit_responsibilities
            loss = (weights * errors).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
        descriptor.eval()
        with torch.no_grad():
            errors = descriptor.reconstruction_error(calibration_values)
            adapter.update_descriptor_stats(
                errors, expert_id, calibration_responsibilities,
                fit_sample_count=fit_values.size(0),
            )
        for parameter in descriptor.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        return float(loss.item())

    def _train_base_descriptors(self, task_id):
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "BASE_DESCRIPTOR_EPOCHS", 8))
        moe_cfg = self._visual_moe_cfg()
        fit_by_block = self._collect_support_views(task_id, moe_cfg.TRAIN_VIEW_IDS)
        calibration_by_block = self._collect_support_views(task_id, moe_cfg.CALIBRATION_VIEW_IDS)
        print(f"\n========== Train Base Representation Descriptors ({epochs} epochs) ==========")
        for block_idx, adapter in self._moe_adapters():
            fit_values = fit_by_block[block_idx]["cls_in"].to(self.device)
            calibration_values = calibration_by_block[block_idx]["cls_in"].to(self.device)
            with torch.no_grad():
                _, fit_responsibilities = adapter.router(fit_values)
                _, calibration_responsibilities = adapter.router(calibration_values)
            for expert_id in range(adapter.num_experts):
                loss = self._train_descriptor(
                    adapter, expert_id,
                    fit_values, fit_responsibilities[:, expert_id],
                    calibration_values, calibration_responsibilities[:, expert_id], epochs,
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
        self._train_base_descriptors(task_id=0)

    # ------------------------------------------------------------------
    # Descriptor-driven layer-wise expansion
    # ------------------------------------------------------------------
    def _incremental_loo_logits(self, images, labels, source_ids, loo_assessment):
        img_feat, moe_aux = self.model_without_dp.extract_img_feature_train(images)
        img_feat_norm = F.normalize(img_feat, dim=-1)
        cache_sources = torch.as_tensor(loo_assessment["source_ids"]).reshape(-1)
        query_sources = source_ids.detach().cpu().reshape(-1).to(cache_sources.dtype)
        matches = query_sources[:, None].eq(cache_sources[None, :])
        if not matches.any(dim=1).all():
            missing = query_sources[~matches.any(dim=1)].unique().tolist()
            raise KeyError(f"query source ids are absent from LOO cache: {missing}")
        rows = matches.long().argmax(dim=1)
        banks = loo_assessment["prototype_banks"][rows].to(images.device, img_feat_norm.dtype)
        logit_scale = float(_cfg_value(getattr(self.cfg.TRAINER.BiMC, "INCREMENTAL", None),
                                       "LOGIT_SCALE", 25.0))
        logits = logit_scale * torch.einsum("bd,bcd->bc", img_feat_norm, F.normalize(banks, dim=-1))
        class_ids = loo_assessment["prototype_class_ids"].to(images.device)
        targets = self._targets_to_columns(labels, {"class_ids": class_ids})
        return logits, targets, moe_aux

    @staticmethod
    def _historical_snapshot(adapter):
        return {
            name: value.detach().cpu().clone()
            for name, value in adapter.state_dict().items()
        }

    @staticmethod
    def _historical_state_unchanged(adapter, snapshot):
        current = adapter.state_dict()
        for name, before in snapshot.items():
            after = current.get(name)
            if after is None or not torch.equal(after.detach().cpu(), before):
                return False, name
        return True, None

    def train_descriptor_for_expert(self, block_idx, fit_features, calibration_features):
        adapter = self._adapter_by_block(block_idx)
        fit_values = fit_features[block_idx]["cls_in"].to(self.device)
        calibration_values = calibration_features[block_idx]["cls_in"].to(self.device)
        fit_responsibilities = torch.ones(fit_values.size(0), device=self.device)
        calibration_responsibilities = torch.ones(calibration_values.size(0), device=self.device)
        expert_id = adapter.num_experts - 1
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "INCREMENTAL_DESCRIPTOR_EPOCHS", 5))
        loss = self._train_descriptor(
            adapter, expert_id,
            fit_values, fit_responsibilities,
            calibration_values, calibration_responsibilities, epochs,
        )
        descriptor = adapter.descriptors[expert_id]
        print(
            f"=> [Descriptor][B{block_idx}][E{expert_id}] loss={loss:.6f} "
            f"mean={descriptor.mean.item():.6f} std={descriptor.std.item():.6f}"
        )
        adapter.assert_descriptors_ready()

    def _historical_anchor_features(self, block_idx):
        item = self.historical_anchor_bank.get(int(block_idx), {})
        return torch.as_tensor(item.get("features", torch.empty(0))).reshape(
            -1, self._adapter_by_block(block_idx).d_model
        )

    @torch.no_grad()
    def _classifier_support_metrics(self, loader, classifier_state):
        """Evaluate old-class fixed support without using the test split."""
        prototypes = self.model_without_dp.fuse_prototypes(classifier_state)
        class_ids = torch.as_tensor(
            classifier_state["class_ids"], device=self.device, dtype=torch.long
        )
        per_class_limit = int(_cfg_value(
            self.cfg.TRAINER.BiMC.INCREMENTAL, "HISTORY_EVAL_SAMPLES_PER_CLASS", 20
        ))
        seen = {int(class_id): 0 for class_id in class_ids.detach().cpu().tolist()}
        correct, count, margin_sum = 0, 0, 0.0
        self.model.eval()
        for batch in loader:
            images, labels = self.parse_batch(batch)
            keep = torch.zeros(labels.numel(), device=labels.device, dtype=torch.bool)
            for index, class_id in enumerate(labels.detach().cpu().tolist()):
                if seen[int(class_id)] < per_class_limit:
                    keep[index] = True
                    seen[int(class_id)] += 1
            if not keep.any():
                continue
            images, labels = images[keep], labels[keep]
            features = F.normalize(self.model_without_dp.extract_img_feature(images), dim=-1)
            logits = features @ prototypes.t()
            targets = self._targets_to_columns(labels, {"class_ids": class_ids})
            positive = logits.gather(1, targets[:, None]).squeeze(1)
            competing = logits.masked_fill(
                F.one_hot(targets, num_classes=class_ids.numel()).bool(), float("-inf")
            ).max(dim=1).values
            competing = torch.where(torch.isfinite(competing), competing, positive.new_full(positive.shape, -1.0))
            correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
            count += labels.numel()
            margin_sum += float((positive - competing).sum().item())
            if all(value >= per_class_limit for value in seen.values()):
                break
        return {"accuracy": correct / max(1, count), "mean_margin": margin_sum / max(1, count)}

    @torch.no_grad()
    def _historical_route_preservation(self, before_adapter, after_adapter, block_idx):
        anchors = self._historical_anchor_features(block_idx)
        if anchors.numel() == 0:
            return 1.0
        values = anchors.to(self.device)
        before_adapter.eval()
        after_adapter.eval()
        before = before_adapter.incremental_route_indices(values)
        after = after_adapter.incremental_route_indices(values)
        return float(before.eq(after).float().mean().item())

    @torch.no_grad()
    def _update_historical_anchors(self, loader):
        """Store class-centroid layer inputs; raw historical images are not retained."""
        features = self.model_without_dp.collect_blockwise_cls_features(loader)
        for block_idx, item in features.items():
            rows, labels = [], []
            for class_id in torch.unique(item["labels"], sorted=True):
                rows.append(item["cls_in"][item["labels"].eq(class_id)].mean(dim=0))
                labels.append(class_id)
            incoming = {"features": torch.stack(rows).cpu(), "labels": torch.stack(labels).cpu()}
            previous = self.historical_anchor_bank.get(int(block_idx))
            if previous is None:
                self.historical_anchor_bank[int(block_idx)] = incoming
            else:
                self.historical_anchor_bank[int(block_idx)] = {
                    "features": torch.cat([previous["features"], incoming["features"]], dim=0),
                    "labels": torch.cat([previous["labels"], incoming["labels"]], dim=0),
                }

    def train_expanded_module(self, task_id, block_idx, train_loader, extract_loader,
                              class_names, class_index, old_state,
                              enable_visual_calibration):
        adapter = self._adapter_by_block(block_idx)
        for _, item in self._moe_adapters():
            item.freeze_all()
        adapter.set_newest_trainable(descriptor=False)
        adapter.set_force_newest(True)
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        dynamic_id = adapter.num_incremental_experts() - 1
        optimizer = optim.AdamW([
            {"params": adapter.experts[-1].parameters(),
             "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_EXPERT", 3e-4))},
            {"params": adapter.incremental_gates[dynamic_id].parameters(),
             "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_ROUTER", 3e-4))},
            {"params": [adapter.incremental_scales[dynamic_id]],
             "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_EXPERT", 3e-4))},
        ], weight_decay=float(_cfg_value(optim_cfg, "WEIGHT_DECAY", 1e-4)))
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.INCREMENTAL, "EPOCHS", 6))
        loss_cfg = self.cfg.TRAINER.BiMC.LOSS
        ce_weight = float(_cfg_value(loss_cfg, "CE_WEIGHT", 1.0))
        gate_weight = float(_cfg_value(loss_cfg, "GATE_WEIGHT", 0.1))
        history_weight = float(_cfg_value(loss_cfg, "HISTORY_INVARIANCE_WEIGHT", 0.1))
        history_anchors = self._historical_anchor_features(block_idx).to(self.device)
        logs = []
        for epoch in range(epochs):
            self.model.train()
            loo_assessment = self.model_without_dp.assess_incremental_loo(
                class_names, extract_loader, class_index, old_state,
                enable_visual_calibration,
            )
            total_loss, total_correct, total_count = 0.0, 0, 0
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(
                    batch, labels.numel(), labels.device
                )
                optimizer.zero_grad(set_to_none=True)
                logits, targets, moe_aux = self._incremental_loo_logits(
                    images, labels, source_ids, loo_assessment
                )
                aux = dict(zip((idx for idx, _ in self._moe_adapters()), moe_aux))[block_idx]
                positive_gate = F.softplus(-aux["incremental_gate_logits"][:, dynamic_id]).mean()
                if history_anchors.numel() > 0:
                    historical_gate = adapter.incremental_gates[dynamic_id](history_anchors).float()
                    history_gate = F.softplus(historical_gate + adapter.route_score_margin).mean()
                else:
                    history_gate = positive_gate.new_zeros(())
                loss_ce = F.cross_entropy(logits, targets)
                loss = ce_weight * loss_ce + gate_weight * positive_gate + history_weight * history_gate
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("non-finite incremental classification loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.newest_parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * labels.numel()
                total_correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
                total_count += labels.numel()
            with torch.no_grad():
                mean_gate = float(torch.sigmoid(
                    adapter.incremental_gates[dynamic_id](
                        loo_assessment["features"].to(self.device)
                    )
                ).mean().item())
            row = {"epoch": epoch + 1, "loss": total_loss / max(1, total_count),
                   "loo_acc": total_correct / max(1, total_count),
                   "new_gate_probability": mean_gate}
            logs.append(row)
            print(
                f"=> [ExpandTrain][Task {task_id}][B{block_idx}] epoch={epoch + 1} "
                f"loss={row['loss']:.6f} loo_acc={row['loo_acc']:.3f} "
                f"new_gate_probability={mean_gate:.3f}"
            )
        adapter.set_force_newest(False)
        return logs

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, prev_state_dict_list,
                               enable_visual_calibration):
        old_state = self.merge_dicts(prev_state_dict_list) if prev_state_dict_list else None
        decisions, expanded = [], []
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        pre = self.model_without_dp.assess_incremental_loo(
            current_class_name, extract_loader, class_index, old_state,
            enable_visual_calibration,
        )
        incremental_cfg = self.cfg.TRAINER.BiMC.INCREMENTAL
        expansion_mode = str(_cfg_value(self._visual_moe_cfg(), "EXPANSION_MODE", "auto")).lower()
        no_deficit = (
            pre["accuracy"] >= float(_cfg_value(incremental_cfg, "LOO_ACC_THRESHOLD", 0.70))
            and pre["mean_margin"] >= float(_cfg_value(incremental_cfg, "LOO_MARGIN_THRESHOLD", 0.0))
        )
        if expansion_mode == "none" or no_deficit or pre["hard_count"] == 0:
            reason = "disabled" if expansion_mode == "none" else "LOO deficit absent"
            print(f"=> [Expand][Task {task_id}] REUSE ({reason})")
            return {
                "trained": False, "accepted": False, "reason": reason,
                "pre_loo": {"accuracy": pre["accuracy"], "mean_margin": pre["mean_margin"],
                            "hard_count": pre["hard_count"]},
                "decisions": [], "expanded": [],
            }

        moe_cfg = self._visual_moe_cfg()
        fit_features = self._collect_support_views(task_id, moe_cfg.TRAIN_VIEW_IDS)
        calibration_features = self._collect_support_views(task_id, moe_cfg.CALIBRATION_VIEW_IDS)
        for block_idx in sorted(self.model_without_dp.expandable_blocks()):
            decision = self.model_without_dp.plan_layer_expansion(
                block_idx, fit_features, class_index, pre["hard_source_ids"]
            )
            decisions.append({
                "block_idx": decision.block_idx, "expand": decision.expand,
                "class_scores": decision.class_scores,
                "uncovered_classes": decision.uncovered_classes,
                "hard_source_count": decision.hard_source_count,
            })
        eligible = [item for item in decisions if item["expand"]]
        if not eligible:
            return {
                "trained": False, "accepted": False, "reason": "descriptor coverage sufficient",
                "pre_loo": {"accuracy": pre["accuracy"], "mean_margin": pre["mean_margin"],
                            "hard_count": pre["hard_count"]},
                "decisions": decisions, "expanded": [],
            }
        threshold = float(moe_cfg.EXPANSION_Z_THRESHOLD)
        selected = max(
            eligible,
            key=lambda item: (max(item["class_scores"].values()) - threshold, -item["block_idx"]),
        )
        block_idx = selected["block_idx"]
        adapter = self._adapter_by_block(block_idx)
        historical_loader = self.data_manager.get_dataloader(
            task_id - 1, source="train", mode="test", accumulate_past=True
        )
        pre_history = self._classifier_support_metrics(historical_loader, old_state)
        before_adapter = copy.deepcopy(adapter).to(self.device)
        historical = self._historical_snapshot(adapter)
        before = adapter.num_experts
        try:
            expert_id = self.model_without_dp.add_expert(block_idx, task_id)
            print(
                f"=> [ExpandCandidate][Task {task_id}][B{block_idx}] expert={expert_id} "
                f"num_experts: {before} -> {adapter.num_experts}"
            )
            self.train_descriptor_for_expert(block_idx, fit_features, calibration_features)
            logs = self.train_expanded_module(
                task_id, block_idx, train_loader, extract_loader, current_class_name,
                class_index, old_state, enable_visual_calibration,
            )
            adapter.set_force_newest(False)
            adapter.freeze_all()
            post = self.model_without_dp.assess_incremental_loo(
                current_class_name, extract_loader, class_index, old_state,
                enable_visual_calibration,
            )
            post_history = self._classifier_support_metrics(historical_loader, old_state)
            unchanged, changed = self._historical_state_unchanged(adapter, historical)
            if not unchanged:
                raise RuntimeError(f"historical parameter changed at B{block_idx}: {changed}")
            preservation = self._historical_route_preservation(before_adapter, adapter, block_idx)
            min_margin_gain = float(_cfg_value(incremental_cfg, "MIN_NEW_MARGIN_GAIN", 0.0))
            min_preservation = float(_cfg_value(
                incremental_cfg, "MIN_HISTORY_ROUTE_PRESERVATION", 1.0
            ))
            min_history_acc_delta = float(_cfg_value(
                incremental_cfg, "MIN_HISTORY_ACCURACY_DELTA", 0.0
            ))
            min_history_margin_delta = float(_cfg_value(
                incremental_cfg, "MIN_HISTORY_MARGIN_DELTA", -1e-3
            ))
            new_improved = (
                post["accuracy"] > pre["accuracy"] + 1e-8
                or (abs(post["accuracy"] - pre["accuracy"]) <= 1e-8
                    and post["mean_margin"] > pre["mean_margin"] + min_margin_gain)
            )
            margin_weight = float(_cfg_value(incremental_cfg, "MARGIN_OBJECTIVE_WEIGHT", 0.05))
            pre_quality = max(0.0, min(1.0, pre["accuracy"] + margin_weight * pre["mean_margin"]))
            post_quality = max(0.0, min(1.0, post["accuracy"] + margin_weight * post["mean_margin"]))
            history_preserved = (
                post_history["accuracy"] >= pre_history["accuracy"] + min_history_acc_delta - 1e-8
                and post_history["mean_margin"] >= (
                    pre_history["mean_margin"] + min_history_margin_delta
                )
            )
            pre_h = 2.0 * pre_history["accuracy"] * pre_quality / max(
                1e-8, pre_history["accuracy"] + pre_quality
            )
            post_h = 2.0 * post_history["accuracy"] * post_quality / max(
                1e-8, post_history["accuracy"] + post_quality
            )
            accepted = (
                new_improved and history_preserved
                and preservation >= min_preservation and post_h > pre_h + 1e-8
            )
            reason = "accepted" if accepted else "candidate did not improve protected LOO objective"
            if not accepted:
                self.model_without_dp.clip_model.visual.transformer.resblocks[
                    int(block_idx)
                ].moe_adapter = before_adapter
                before_adapter.freeze_all()
                print(f"=> [ExpandRollback][Task {task_id}][B{block_idx}] {reason}")
            else:
                print(
                    f"=> [ExpandCommit][Task {task_id}][B{block_idx}] "
                    f"LOO {pre['accuracy']:.3f}->{post['accuracy']:.3f}, "
                    f"history {pre_history['accuracy']:.3f}->{post_history['accuracy']:.3f}, "
                    f"history_route_preservation={preservation:.3f}"
                )
                expanded.append({"block_idx": block_idx, "expert_id": expert_id, "epochs": logs})
        except Exception:
            self.model_without_dp.clip_model.visual.transformer.resblocks[
                int(block_idx)
            ].moe_adapter = before_adapter
            before_adapter.freeze_all()
            raise
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        return {
            "trained": True, "accepted": bool(expanded), "reason": reason,
            "selected_block": int(block_idx),
            "pre_loo": {"accuracy": pre["accuracy"], "mean_margin": pre["mean_margin"],
                        "hard_count": pre["hard_count"]},
            "post_loo": {"accuracy": post["accuracy"], "mean_margin": post["mean_margin"],
                         "hard_count": post["hard_count"]},
            "history_route_preservation": preservation,
            "history_support": {"pre": pre_history, "post": post_history},
            "protected_harmonic": {"pre": pre_h, "post": post_h},
            "decisions": decisions, "expanded": expanded,
        }

    def _build_and_refine_task_state(self, current_class_name, extract_loader, class_index,
                                     old_state, enable_visual_calibration):
        state = self.model_without_dp.build_task_statistics(
            current_class_name, extract_loader, class_index,
            calibrate_novel_vision_proto=enable_visual_calibration,
        )
        state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
        if old_state is not None and hasattr(self.model_without_dp, "refine_new_fused_prototypes"):
            state = self.model_without_dp.refine_new_fused_prototypes(old_state, state, extract_loader)
        return state

    def run(self):
        configured_end = int(_cfg_value(self.cfg.TRAINER.BiMC, "END_SESSION", -1))
        end_task = self.data_manager.num_tasks - 1 if configured_end < 0 else min(
            configured_end, self.data_manager.num_tasks - 1
        )
        print(f"Start inferencing on all tasks: [0, {end_task}]")
        state_dict_list = list(self._resume_state_dict_list)
        start_task = self._resume_completed_session + 1
        use_moe = bool(getattr(self.cfg.TRAINER.BiMC, "VISUAL_MOE", None) and self._visual_moe_cfg().ENABLE)
        if start_task > end_task:
            print("=> Resume checkpoint already contains all configured FSCIL sessions.")
            return

        for task_id in range(start_task, end_task + 1):
            self.model.eval()
            plan_summary, adaptation = None, None
            final_routing = {}
            class_index = self.data_manager.class_index_in_task[task_id]
            current_class_name = np.array(self.data_manager.class_names)[class_index].tolist()
            extract_loader = self.data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
            train_loader = self.data_manager.get_dataloader(task_id, source="train", mode="train", accumulate_past=False)
            print(f"\n=> [Session {task_id}] support_manifest={self._support_manifest_hash()}")
            if use_moe:
                self._log_expert_topology(f"task{task_id}-before")

            if task_id == 0:
                task_stat = self.model_without_dp.build_task_statistics(
                    current_class_name, extract_loader, class_index, calibrate_novel_vision_proto=False
                )
                task_stat["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
                if use_moe:
                    self.train_base_task(task_stat, train_loader, extract_loader)
                    self.model.eval()
                    task_stat = self.model_without_dp.build_task_statistics(
                        current_class_name, extract_loader, class_index, calibrate_novel_vision_proto=False
                    )
                    task_stat["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
                state_dict_list.append(task_stat)
                state_dict_list[-1]["creation_session"] = int(task_id)
            else:
                old_state = self.merge_dicts(state_dict_list)
                enable_visual_calibration = bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                if use_moe:
                    self._routing_diagnostics(extract_loader, f"task{task_id}-before", current_task=task_id)
                    adaptation = self.train_incremental_task(
                        task_id, train_loader, extract_loader, class_index,
                        current_class_name, state_dict_list, enable_visual_calibration,
                    )
                    plan_summary = self._cpu_copy(adaptation["decisions"])
                self.model.eval()
                task_stat = self._build_and_refine_task_state(
                    current_class_name, extract_loader, class_index, old_state, enable_visual_calibration
                )
                state_dict_list.append(task_stat)
                state_dict_list[-1]["creation_session"] = int(task_id)

            merged_state = self.merge_dicts(state_dict_list)
            if use_moe:
                final_routing = self._routing_diagnostics(
                    extract_loader, f"task{task_id}-final", current_task=task_id
                )
                self._log_expert_topology(f"task{task_id}-final")
                self._update_historical_anchors(extract_loader)
            start_time = time.perf_counter()
            acc = self.inference_task_bilevel(task_id, merged_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            suffix = f", time: {elapsed:.3f}s"
            print(f"+++++++++++ task {task_id}{suffix} ++++++++++++++++")
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
            })
            self.save_checkpoint(task_id, state_dict_list)

        print(f"\nFinal acc: {self.acc_list}")
        print("Task-wise acc:")
        for index, task_acc in enumerate(self.task_acc_list):
            print(f"task {index:2d}, acc: {task_acc}")

    @torch.no_grad()
    def inference_task_bilevel(self, task_id, state_dict):
        """Evaluate visual/text/BiMC/refined banks from one image forward pass."""
        beta = self.cfg.DATASET.BETA
        lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        image_proto = F.normalize(state_dict["image_proto"].to(self.device), dim=-1)
        text_features = state_dict["text_features"].to(self.device)
        description_proto = state_dict["description_proto"].to(self.device)
        text_proto = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
        bimc_proto = F.normalize(beta * text_proto + (1.0 - beta) * image_proto, dim=-1)
        stored_refined = state_dict.get("fused_proto")
        refined_proto = F.normalize(
            bimc_proto if stored_refined is None else stored_refined.to(self.device), dim=-1
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
            data, targets = self.parse_batch(batch)
            image_features = F.normalize(self.model_without_dp.extract_img_feature(data), dim=-1)
            for name, prototypes in prototype_banks.items():
                all_logits[name].append(100.0 * image_features @ prototypes.t())
            all_targets.append(targets)
        logits_by_variant = {name: torch.cat(values, dim=0) for name, values in all_logits.items()}
        targets = torch.cat(all_targets, dim=0)
        variants = {
            name: self.evaluator.calc_accuracy(logits, targets, task_id, class_ids=state_dict["class_ids"])
            for name, logits in logits_by_variant.items()
        }
        result = dict(variants["refined"])
        result["classifier_variants"] = variants
        logits = logits_by_variant["refined"]
        result["error_flow"] = self.evaluator.error_flow(
            logits, targets, task_id, class_ids=state_dict["class_ids"]
        )
        print(
            "=> [Classifier] visual={:.2f} text={:.2f} bimc={:.2f} refined={:.2f}".format(
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
