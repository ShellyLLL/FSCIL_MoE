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
        elif schema not in (7, 8, 9, 10):
            raise ValueError("Checkpoint topology is incompatible; resume from Session 0.")
        if schema in (6, 7, 8, 9) and int(payload["completed_session"]) != 0:
            raise ValueError(
                "Versioned-classifier checkpoints cannot be resumed with unified routing; "
                "resume from a base-only checkpoint or start from Session 0."
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
        # Legacy task-gate offsets are ignored: inference uses one global bank.
        self.task_gate_biases = [0.0] * (completed_session + 1)
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
            "schema_version": 10,
            "completed_session": int(completed_session),
            "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
            "moe_topology": topology,
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
    def _support_bank(self, loader, class_index, text_features, description_proto,
                      version, calibrate=True):
        """Build both leave-one-source-out and full-shot deployment classifiers."""
        model = self.model_without_dp
        current = model.build_leave_one_out_prototypes(loader, class_index, max_session=None)
        reference = (current if version == 0 else
                     model.build_leave_one_out_prototypes(loader, class_index, max_session=0))
        if not torch.equal(current["source_ids"], reference["source_ids"]):
            raise RuntimeError("Current and reference support source orders differ.")
        labels, classes = current["labels"], current["classes"]
        current_features, reference_features = current["features"], reference["features"]
        visual_full = F.normalize(torch.stack([
            current_features[labels.eq(cls)].mean(0) for cls in classes
        ]), dim=-1)
        reference_full = F.normalize(torch.stack([
            reference_features[labels.eq(cls)].mean(0) for cls in classes
        ]), dim=-1)
        visual_loo = current["loo_visual_prototypes"]
        if calibrate:
            visual_full = model.calibrate_incremental_visual(visual_full.to(self.device),
                                                               reference_full.to(self.device))
            visual_loo = model.calibrate_incremental_visual(visual_loo.to(self.device),
                                                              reference["loo_visual_prototypes"].to(self.device))
        lambda_t = (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
                    if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
        text = F.normalize((1.0 - lambda_t) * text_features
                           + lambda_t * description_proto, dim=-1)
        beta = float(self.cfg.DATASET.BETA)
        current["visual_full"] = visual_full.to(self.device)
        current["visual_loo"] = visual_loo.to(self.device)
        current["fused_full"] = F.normalize(beta * text + (1.0 - beta) * current["visual_full"], dim=-1)
        current["fused_loo"] = F.normalize(beta * text[None, :, :] +
                                            (1.0 - beta) * current["visual_loo"], dim=-1)
        if not bool(current["loo_valid"].all().item()):
            raise ValueError("Every training query needs another source in its class.")
        return current

    def _versioned_scores(self, images, prototypes, versions, scale):
        """Score each class using the encoder which created its prototype."""
        versions = torch.as_tensor(versions, device=images.device, dtype=torch.long)
        output = images.new_empty((images.size(0), prototypes.size(0)))
        for version in torch.unique(versions).tolist():
            mask = versions.eq(version)
            with torch.no_grad():
                features = self.model_without_dp.extract_img_feature(images, max_session=version)
            part = scale * F.normalize(features.float(), dim=-1) @ F.normalize(
                prototypes[mask].to(images.device).float(), dim=-1).t()
            output[:, mask] = part
        return output

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
                logits, targets = self._incremental_logits(
                    images, labels, source_ids, bank, task_id, scale
                )
                loss = F.cross_entropy(logits, targets)
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
            return {"trained": False, "router_adapted": False, "decisions": [],
                    "expanded": [], "encoder_version": previous_version,
                    "reason": "insufficient independent support folds"}

        null_losses = [
            self._held_out_loss(val, fit_extract, class_index, text, description,
                                previous_version, enable_visual_calibration)
            for _, fit_extract, val in folds
        ]

        # First test the lower-capacity option: update router columns only.
        initial_router_states = {
            block_idx: copy.deepcopy(adapter.router.state_dict())
            for block_idx, adapter in self._moe_adapters()
        }
        router_losses = []
        for fit_train, fit_extract, validation in folds:
            for block_idx, adapter in self._moe_adapters():
                adapter.router.load_state_dict(initial_router_states[block_idx])
                adapter.freeze_all()
            self._train_router_only(
                task_id, fit_train, fit_extract, class_index, text, description,
                enable_visual_calibration,
            )
            router_losses.append(self._held_out_loss(
                validation, fit_extract, class_index, text, description,
                previous_version, enable_visual_calibration,
            ))
        router_gains = np.asarray(null_losses) - np.asarray(router_losses)
        router_gain = float(np.mean(router_gains))
        router_gain_se = float(
            np.std(router_gains, ddof=1) / np.sqrt(len(router_gains))
        ) if len(router_gains) > 1 else float("inf")
        print(f"=> [RouterOnly][Task {task_id}] gain={router_gain:.5f} "
              f"se={router_gain_se:.5f}")

        # Restore the exact pre-candidate state before accepting/rejecting it.
        for block_idx, adapter in self._moe_adapters():
            adapter.router.load_state_dict(initial_router_states[block_idx])
            adapter.freeze_all()
        decisions = [{
            "mode": "router_only",
            "null_loss": float(np.mean(null_losses)),
            "candidate_loss": float(np.mean(router_losses)),
            "gain": router_gain,
            "gain_se": router_gain_se,
        }]
        if router_gain > router_gain_se:
            history = self._train_router_only(
                task_id, train_loader, extract_loader, class_index, text,
                description, enable_visual_calibration,
            )
            print(f"=> [Accepted][Task {task_id}] router-only adaptation; no expert added.")
            return {"trained": False, "router_adapted": True,
                    "router_epochs": history, "decisions": decisions, "expanded": [],
                    "encoder_version": previous_version}

        # Routing alone did not pass held-out validation. Test temporary expert
        # additions with historical experts frozen and active in the shared Softmax.
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
                    task_id, enable_visual_calibration,
                    previous_version=previous_version,
                ))
            adapter.remove_newest_expert()
            gains = np.asarray(null_losses) - np.asarray(candidate_losses)
            decisions.append({
                "mode": "expert_expansion",
                "block_idx": block_idx,
                "null_loss": float(np.mean(null_losses)),
                "candidate_loss": float(np.mean(candidate_losses)),
                "gain": float(np.mean(gains)),
                "gain_se": float(np.std(gains, ddof=1) / np.sqrt(len(gains))),
            })
            print(f"=> [Candidate][Task {task_id}][B{block_idx}] "
                  f"gain={decisions[-1]['gain']:.5f} se={decisions[-1]['gain_se']:.5f}")

        candidates = [row for row in decisions if row.get("mode") == "expert_expansion"]
        best = max(candidates, key=lambda row: row["gain"], default=None)
        if best is None or best["gain"] <= best["gain_se"]:
            for _, adapter in self._moe_adapters():
                adapter.freeze_all()
            return {"trained": False, "router_adapted": False,
                    "decisions": decisions, "expanded": [],
                    "encoder_version": previous_version,
                    "reason": "neither router adaptation nor expansion passed validation"}

        block_idx = best["block_idx"]
        adapter = self._adapter_by_block(block_idx)
        expert_id = self.model_without_dp.add_expert(block_idx, task_id)
        epochs = self._train_candidate(task_id, block_idx, train_loader, extract_loader,
                                       class_index, text, description,
                                       enable_visual_calibration)
        print(f"=> [Accepted][Task {task_id}][B{block_idx}] expert={expert_id} "
              f"validation_gain={best['gain']:.5f}")
        return {"trained": True, "router_adapted": False, "decisions": decisions,
                "expanded": [{"block_idx": block_idx, "expert_id": expert_id,
                              "epochs": epochs}], "encoder_version": task_id}

    def _build_and_refine_task_state(self, current_class_name, extract_loader, class_index,
                                     old_state, enable_visual_calibration, encoder_version):
        refinement = getattr(self.cfg.TRAINER.BiMC, "PROTOTYPE_REFINEMENT", None)
        if refinement is not None and bool(getattr(refinement, "ENABLE", False)):
            raise ValueError("PROTOTYPE_REFINEMENT is incompatible with versioned classifiers.")
        state = self.model_without_dp.build_task_statistics(
            current_class_name, extract_loader, class_index,
            calibrate_novel_vision_proto=enable_visual_calibration,
            encoder_version=encoder_version,
        )
        state["class_ids"] = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
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

    @torch.no_grad()
    def _rebuild_unified_task_states(self, through_task):
        """Re-encode every seen class with the current complete model.

        Text/description prototypes remain in CLIP's semantic space. Visual
        support prototypes are refreshed after every session, so no class is
        evaluated with an encoder version from its creation session.
        """
        self.model.eval()
        self.model_without_dp.base_vision_prototype = None
        self.model_without_dp._fused_history_prototypes = None
        self.model_without_dp._fused_history_class_ids = None
        states = []
        for session_id in range(int(through_task) + 1):
            class_index = self.data_manager.class_index_in_task[session_id]
            class_names = np.array(self.data_manager.class_names)[class_index].tolist()
            support_loader = self.data_manager.get_dataloader(
                session_id, source="train", mode="test", accumulate_past=False
            )
            state = self.model_without_dp.build_task_statistics(
                class_names,
                support_loader,
                class_index,
                calibrate_novel_vision_proto=(
                    session_id > 0 and bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                ),
                encoder_version=None,
            )
            state["class_ids"] = torch.as_tensor(
                class_index, device=self.device, dtype=torch.long
            )
            state["creation_session"] = int(session_id)
            # Legacy metadata only; inference never branches on this value.
            state["encoder_version"] = 0
            states.append(state)
        return states

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
            state_dict_list = self._rebuild_unified_task_states(task_id)
            self._resume_state_dict_list = state_dict_list
            self.task_gate_biases = [0.0] * (task_id + 1)
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
        # AA: mean of all-session mean accuracies (percentage points).
        aa = float(np.mean(self.acc_list)) if self.acc_list else 0.0
        # PD: average drop on each earlier task from its introduction session
        # to the final session. The final task is excluded because it has no
        # subsequent session in which forgetting can occur.
        if len(self.task_acc_list) > 1:
            final_task_acc = self.task_acc_list[-1]
            drops = [
                float(self.task_acc_list[task_id][task_id]) -
                float(final_task_acc[task_id])
                for task_id in range(len(self.task_acc_list) - 1)
            ]
            pd = float(np.mean(drops)) if drops else 0.0
        else:
            drops, pd = [], 0.0
        print(f"=> AA (mean session accuracy): {aa:.3f}")
        print(f"=> PD (mean old-task accuracy drop, pp): {pd:.3f}; per-task={drops}")
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
            for name, prototypes in prototype_banks.items():
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
