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
from utils.metrics import session_metrics


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
        paths = []
        for state, ids in zip(dict_list, per_state_ids):
            if "feature_path" not in state:
                raise ValueError("Task state lacks its immutable feature path; retrain Session 0.")
            paths.extend([dict(state["feature_path"]) for _ in range(ids.numel())])
        result["feature_paths"] = [paths[i] for i in order.tolist()]
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
            block: {"weight_sum": torch.zeros(adapter.num_experts), "top1": torch.zeros(adapter.num_experts),
                    "count": 0, "classes": [set() for _ in range(adapter.num_experts)]}
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
                item["top1"] += F.one_hot(top1, adapter.num_experts).sum(dim=0)
                item["count"] += weights.size(0)
                for expert_id in range(adapter.num_experts):
                    chosen = labels[top1.to(labels.device).eq(expert_id)]
                    item["classes"][expert_id].update(int(value) for value in chosen.cpu().tolist())
        output = {}
        for block_idx, item in totals.items():
            count = max(1, item["count"])
            means, usage = item["weight_sum"] / count, item["top1"] / count
            rows = []
            adapter = adapters[block_idx]
            for expert_id in range(adapter.num_experts):
                born = adapter.expert_birth_sessions[expert_id]
                role = "new" if current_task is not None and born == int(current_task) else "history"
                rows.append(
                    f"e{expert_id}({role},birth={born}) mean_weight={means[expert_id]:.3f} "
                    f"top1={usage[expert_id]:.3f} classes={len(item['classes'][expert_id])}"
                )
            print(f"=> [Router][{tag}][B{block_idx}] " + " | ".join(rows))
            output[block_idx] = {"mean_weight": means, "top1_usage": usage}
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
            raise ValueError("Checkpoint topology is incompatible; retrain from Session 0.")
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
        if len(self.acc_list) != completed_session + 1:
            raise ValueError("Checkpoint accuracy history is incomplete.")
        for state in self._resume_state_dict_list:
            with self.model_without_dp.use_feature_path(state["feature_path"]):
                pass
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
            "schema_version": 7,
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
            "task_states": self._cpu_copy(state_dict_list),
            "base_vision_prototype": self._cpu_copy(self.model_without_dp.base_vision_prototype),
            "support_manifest": self.data_manager.get_support_manifest(),
            "acc_list": list(self.acc_list),
            "metrics": session_metrics(self.acc_list),
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
    # Descriptor-driven layer-wise expansion
    # ------------------------------------------------------------------
    def _build_incremental_loo_cache(self, extract_loader, class_index):
        return self.model_without_dp.build_leave_one_out_prototypes(extract_loader, class_index)

    def _incremental_loo_logits(self, images, labels, source_ids, support_cache, old_state):
        model = self.model_without_dp
        features, aux = model.extract_img_feature_train(images)
        features = F.normalize(features, dim=-1)
        sources = support_cache["source_ids"].cpu()
        matches = source_ids.cpu().reshape(-1, 1).eq(sources.reshape(1, -1))
        if not matches.any(dim=1).all():
            raise KeyError("Query source is absent from the support cache.")
        rows = matches.long().argmax(dim=1)
        if not support_cache["loo_valid"][rows].all():
            raise ValueError("Incremental LOO requires at least two distinct sources per class.")
        visual = support_cache["loo_visual_prototypes"][rows].to(features)
        visual = model.calibrate_visual(visual)
        text = support_cache["calibrated_text"].to(features)
        beta = self.cfg.DATASET.BETA
        fused = F.normalize(beta * text.unsqueeze(0) + (1.0 - beta) * visual, dim=-1)
        current_logits = torch.einsum("bd,bcd->bc", features, fused)
        with torch.no_grad():
            old_logits = (model.classifier_scores(images, old_state) if old_state is not None
                          else features.new_empty((features.size(0), 0)))
        scale = float(self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        logits = scale * torch.cat([old_logits, current_logits], dim=1)
        old_ids = model._state_class_ids(old_state, images.device)
        current_ids = support_cache["classes"].to(images.device)
        targets = self._targets_to_columns(labels, {"class_ids": torch.cat([old_ids, current_ids])})
        return logits, targets, aux

    def _current_support_cache(self, loader, class_index, calibrated_text):
        cache = self._build_incremental_loo_cache(loader, class_index)
        if not cache["loo_valid"].all():
            raise ValueError("Expansion requires at least two distinct support sources per class.")
        cache["calibrated_text"] = calibrated_text.detach()
        return cache

    @torch.no_grad()
    def _support_objective(self, loader, cache, old_state):
        self.model.eval()
        loss, count, correct, deficit = 0.0, 0, 0, {}
        for batch in loader:
            images, labels = self.parse_batch(batch)
            sources = self._batch_source_ids(batch, labels.numel(), labels.device)
            logits, targets, _ = self._incremental_loo_logits(images, labels, sources, cache, old_state)
            loss += float(F.cross_entropy(logits, targets, reduction="sum"))
            correct += int(logits.argmax(1).eq(targets).sum())
            count += labels.numel()
            other = logits.clone()
            other.scatter_(1, targets[:, None], -torch.inf)
            margins = logits.gather(1, targets[:, None]).squeeze(1) - other.max(1).values
            for source, margin in zip(sources.tolist(), margins.tolist()):
                deficit[int(source)] = max(0.0, -margin)
        return {"loss": loss / count, "accuracy": correct / count, "deficit": deficit}

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

    def train_descriptor_for_expert(self, block_idx, extract_loader):
        adapter = self._adapter_by_block(block_idx)
        features = self.model_without_dp.collect_blockwise_cls_features(extract_loader)[block_idx]
        values = features["cls_in"].to(self.device)
        with torch.no_grad():
            _, responsibilities = adapter.router(values)
        expert_id = adapter.num_experts - 1
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.OPTIM, "INCREMENTAL_DESCRIPTOR_EPOCHS", 5))
        loss = self._train_descriptor(
            adapter, expert_id, values, responsibilities[:, expert_id], epochs
        )
        descriptor = adapter.descriptors[expert_id]
        print(
            f"=> [Descriptor][B{block_idx}][E{expert_id}] loss={loss:.6f} "
            f"mean={descriptor.mean.item():.6f} std={descriptor.std.item():.6f}"
        )
        adapter.assert_descriptors_ready()

    def train_expanded_module(self, task_id, block_idx, train_loader, extract_loader,
                              class_index, calibrated_text, old_state):
        adapter = self._adapter_by_block(block_idx)
        for _, item in self._moe_adapters():
            item.freeze_all()
        adapter.set_newest_trainable(descriptor=False)
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        optimizer = optim.AdamW([
            {"params": adapter.experts[-1].parameters(),
             "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_EXPERT", 3e-4))},
            {"params": adapter.router.columns[-1].parameters(),
             "lr": float(_cfg_value(optim_cfg, "LR_INCREMENTAL_ROUTER", 3e-4))},
        ], weight_decay=float(_cfg_value(optim_cfg, "WEIGHT_DECAY", 1e-4)))
        epochs = int(_cfg_value(self.cfg.TRAINER.BiMC.INCREMENTAL, "EPOCHS", 6))
        logs = []
        for epoch in range(epochs):
            # Stop-gradient support bank refreshed in the current representation.
            support_cache = self._current_support_cache(extract_loader, class_index, calibrated_text)
            self.model.train()
            total_loss, total_correct, total_count = 0.0, 0, 0
            for batch in train_loader:
                images, labels = self.parse_batch(batch)
                source_ids = self._batch_source_ids(
                    batch, labels.numel(), labels.device
                )
                optimizer.zero_grad(set_to_none=True)
                logits, targets, _ = self._incremental_loo_logits(
                    images, labels, source_ids, support_cache, old_state
                )
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("non-finite incremental classification loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.newest_parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * labels.numel()
                total_correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
                total_count += labels.numel()
            with torch.no_grad():
                weights = []
                for batch in train_loader:
                    images, _ = self.parse_batch(batch)
                    _, aux_list = self.model_without_dp.extract_img_feature_train(images)
                    aux = dict(zip((idx for idx, _ in self._moe_adapters()), aux_list))[block_idx]
                    weights.append(aux["route_weights"][:, -1].detach())
                mean_weight = float(torch.cat(weights).mean().item())
            row = {"epoch": epoch + 1, "loss": total_loss / max(1, total_count),
                   "loo_acc": total_correct / max(1, total_count),
                   "new_router_mean_weight": mean_weight}
            logs.append(row)
            print(
                f"=> [ExpandTrain][Task {task_id}][B{block_idx}] epoch={epoch + 1} "
                f"loss={row['loss']:.6f} loo_acc={row['loo_acc']:.3f} "
                f"new_router_mean_weight={mean_weight:.3f}"
            )
        return logs

    def train_incremental_task(self, task_id, train_loader, extract_loader, class_index,
                               current_class_name, prev_state_dict_list,
                               enable_visual_calibration):
        del enable_visual_calibration
        model = self.model_without_dp
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        if self._visual_moe_cfg().EXPANSION_MODE == "none":
            return {"trained": False, "decisions": [], "expanded": []}
        old_state = self.merge_dicts(prev_state_dict_list) if prev_state_dict_list else None
        text, _ = model.inference_text_feature(current_class_name, model.template, class_index[0])
        _, _, description = model.inference_all_description_feature(
            current_class_name, self.cfg.DATASET.GPT_PATH, class_index[0])
        # DataManager uses ascending global class IDs; explicitly align LOO rows.
        order = torch.as_tensor(class_index, device=text.device).argsort()
        weight = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        calibrated_text = F.normalize((1-weight)*text + weight*description, dim=-1)[order]
        cache = self._current_support_cache(extract_loader, class_index, calibrated_text)
        before_score = self._support_objective(extract_loader, cache, old_state)
        decisions = [model.plan_layer_expansion(i, extract_loader, class_index,
                                               deficits=before_score["deficit"])
                     for i in model.expandable_blocks()]
        eligible = [d for d in decisions if d.expand]
        summary = [dict(block_idx=d.block_idx, expand=d.expand,
                        class_scores=d.class_scores, uncovered_classes=d.uncovered_classes)
                   for d in decisions]
        if not eligible:
            return {"trained": False, "decisions": summary, "expanded": []}
        # One candidate per session is an explicit capacity budget, not an accuracy target.
        decision = max(eligible, key=lambda d: max(d.class_scores.values()))
        block_idx = decision.block_idx
        adapter = self._adapter_by_block(block_idx)
        historical = self._historical_snapshot(adapter)
        topology = adapter.export_topology()
        rng = self._rng_state()
        accepted = False
        try:
            expert_id = model.add_expert(block_idx, task_id)
            logs = self.train_expanded_module(task_id, block_idx, train_loader, extract_loader,
                                              class_index, calibrated_text, old_state)
            adapter.freeze_all()
            fresh_cache = self._current_support_cache(extract_loader, class_index, calibrated_text)
            after_score = self._support_objective(extract_loader, fresh_cache, old_state)
            # This is a support fit diagnostic, NOT held-out generalization evidence.
            accepted = (np.isfinite(after_score["loss"])
                        and after_score["loss"] < before_score["loss"]
                        and after_score["accuracy"] >= before_score["accuracy"])
            if accepted:
                self.train_descriptor_for_expert(block_idx, extract_loader)
                unchanged, changed = self._historical_state_unchanged(adapter, historical)
                if not unchanged:
                    accepted = False
                    raise RuntimeError(f"Historical parameter changed at B{block_idx}: {changed}")
        except BaseException:
            accepted = False
            raise
        finally:
            if not accepted:
                adapter.rebuild_topology(topology)
                adapter.load_state_dict(historical, strict=True)
                self._restore_rng_state(rng)
            for _, item in self._moe_adapters():
                item.freeze_all()
        print(f"=> [Candidate] B{block_idx} accepted={accepted} "
              f"support_LOO_CE={before_score['loss']:.6f}->{after_score['loss']:.6f}")
        return {"trained": accepted, "decisions": summary,
                "expanded": ([{"block_idx": block_idx, "expert_id": expert_id, "epochs": logs}]
                             if accepted else []),
                "candidate": {"accepted": accepted, "before": before_score, "after": after_score}}

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
                state_dict_list[-1]["feature_path"] = self.model_without_dp.feature_path()
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
                state_dict_list[-1]["feature_path"] = self.model_without_dp.feature_path()

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
            suffix = f", time: {elapsed:.3f}s"
            print(f"+++++++++++ task {task_id}{suffix} ++++++++++++++++")
            print(
                f"=> Task [{task_id}], Acc: {acc['mean_acc']:.3f}, "
                f"Base: {acc['base_avg_acc']:.2f}, Novel: {acc['inc_avg_acc']:.2f}, "
                f"H: {acc['harmonic_acc']:.2f}"
            )
            self.acc_list.append(acc["mean_acc"])
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
        metrics = session_metrics(self.acc_list)
        scope = "full" if end_task == self.data_manager.num_tasks - 1 else "partial"
        print(f"=> [{scope}] AA={metrics['AA']:.4f}, PD={metrics['PD']:.4f} percentage points")
        print("Task-wise acc:")
        for index, task_acc in enumerate(self.task_acc_list):
            print(f"task {index:2d}, acc: {task_acc}")

    @torch.no_grad()
    def inference_task_bilevel(self, task_id, state_dict):
        """Evaluate all banks, sharing image forwards among identical feature paths."""
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
            # Reuse each historical-path forward across all diagnostic banks.
            groups = {}
            for column, path in enumerate(state_dict["feature_paths"]):
                groups.setdefault(tuple(sorted(path.items())), []).append(column)
            batch_logits = {name: image_proto.new_empty((data.size(0), image_proto.size(0)))
                            for name in prototype_banks}
            for key, columns in groups.items():
                with self.model_without_dp.use_feature_path(dict(key)):
                    features = F.normalize(self.model_without_dp.extract_img_feature(data), dim=-1)
                for name, prototypes in prototype_banks.items():
                    batch_logits[name][:, columns] = 100.0 * features @ prototypes[columns].t()
            for name in prototype_banks:
                all_logits[name].append(batch_logits[name])
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

