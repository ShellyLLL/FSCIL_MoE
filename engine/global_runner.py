"""Image-level entry point for fixed-feature, global-risk FSCIL.

The original runner remains available as a reproducible legacy ablation.
Schema 10 checkpoints include the fixed feature replay memory and every route.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from tqdm import tqdm

from engine.engine import Runner
from engine.feature_risk import FeatureRiskTrainer, RiskConfig
from models.global_moe import GlobalFeatureMoE, FeatureRoute, fused_prototypes, unit


class GlobalRiskRunner(Runner):
    def __init__(self, cfg):
        self.feature_head = None
        self.feature_memories = []
        self.risk_config = self.make_risk_config(cfg)
        self._validate_settings(cfg)
        super().__init__(cfg)

    @staticmethod
    def make_risk_config(cfg):
        b = cfg.TRAINER.BiMC
        g = b.GLOBAL_MOE
        return RiskConfig(epochs=g.EPOCHS, lr=g.LR, weight_decay=g.WEIGHT_DECAY,
                          batch_size=g.BATCH_SIZE, folds=g.FOLDS, rank=g.RANK,
                          max_experts=g.MAX_EXPERTS, residual_weight=g.RESIDUAL_WEIGHT,
                          min_gain=g.MIN_GAIN, old_risk_tolerance=g.OLD_RISK_TOLERANCE,
                          parameter_cost=g.PARAMETER_COST, beta=cfg.DATASET.BETA,
                          visual_weight=b.LAMBDA_I if b.VISION_CALIBRATION else 0.0,
                          visual_tau=b.TAU, seed=cfg.SEED, expand=g.EXPAND)

    @staticmethod
    def _validate_settings(cfg):
        b = cfg.TRAINER.BiMC
        if not b.FREEZE_VISUAL_BACKBONE or not b.FREEZE_TEXT_ENCODER:
            raise ValueError("Global feature MoE requires a frozen CLIP backbone")
        if b.FUSED_CONFLICT_STRENGTH != 0 or b.PROTOTYPE_REFINEMENT.ENABLE:
            raise ValueError("Global feature MoE uses one fused classifier; disable legacy refinement")
        if b.PREC not in ("fp32", "amp"):
            raise ValueError("Use fp32/amp for reproducible float32 cached features")
        if b.TEXT_CALIBRATION and not 0 <= b.LAMBDA_T <= 1:
            raise ValueError("Text calibration weight must lie in [0,1]")

    def _protocol(self):
        cfg = self.cfg
        return {"risk": asdict(self.risk_config),
                "scale": float(cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE),
                "backbone": cfg.MODEL.BACKBONE.NAME,
                "dataset": cfg.DATASET.NAME,
                "class_names": list(self.data_manager.class_names),
                "class_order": [list(map(int, c)) for c in self.data_manager.class_index_in_task],
                "text_weight": (cfg.TRAINER.BiMC.LAMBDA_T
                                if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0),
                "description_sha256": hashlib.sha256(Path(cfg.DATASET.GPT_PATH).read_bytes()).hexdigest()}

    def _freeze_encoder(self):
        self.model_without_dp.requires_grad_(False)
        self.model.eval()

    def _train_base_descriptors(self, extract_loader):
        # Descriptor novelty does not participate in the global-risk decision.
        # Avoid training unused modules while retaining checkpoint compatibility.
        return None

    @torch.no_grad()
    def _extract_memory(self, task_id, loader, text=None):
        self.model.eval()
        features, labels, sources = [], [], []
        for batch in loader:
            images, target = self.parse_batch(batch)
            features.append(unit(self.model_without_dp.extract_img_feature(images, max_session=0)))
            labels.append(target)
            sources.append(self._batch_source_ids(batch, len(target), target.device).long())
        ids = torch.as_tensor(self.data_manager.class_index_in_task[task_id],
                              device=self.device, dtype=torch.long)
        if text is None:
            names = [self.data_manager.class_names[i] for i in ids.tolist()]
            template, description = self._text_prototypes(names, ids.tolist())
            weight = (self.cfg.TRAINER.BiMC.LAMBDA_T
                      if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
            text = unit((1 - weight) * template + weight * description)
        cache = {"features": torch.cat(features), "labels": torch.cat(labels),
                 "sources": torch.cat(sources), "class_ids": ids, "text": unit(text)}
        self._check_memory(cache)
        return cache

    @staticmethod
    def _check_memory(cache):
        required = {"features", "labels", "sources", "class_ids", "text"}
        if set(cache) != required:
            raise ValueError("Incomplete feature memory")
        x, y, sid, ids, text = (cache[k] for k in ("features", "labels", "sources", "class_ids", "text"))
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(sid):
            raise ValueError("Misaligned feature memory")
        if text.shape != (len(ids), x.shape[1]) or not torch.equal(torch.unique(y, sorted=True), ids.sort().values):
            raise ValueError("Feature memory class/text mismatch")
        if not torch.isfinite(x).all() or not torch.isfinite(text).all():
            raise ValueError("Non-finite feature memory")
        if len(torch.unique(ids)) != len(ids):
            raise ValueError("Duplicate memory class IDs")

    def _initialize_base(self, cache, prototypes=None):
        self.feature_head = GlobalFeatureMoE(
            cache["features"].shape[-1], self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE).to(self.device)
        self.risk_config.validate(self.feature_head.dim)
        route = FeatureRoute(self.feature_head.dim).to(self.device)
        if prototypes is None:
            prototypes = fused_prototypes(route, [], cache, cache["text"], cache,
                                           self.risk_config.beta, 0.0, self.risk_config.visual_tau)
        self.feature_head.bind(route, cache["class_ids"], prototypes)
        self.feature_memories = [cache]
        self._freeze_encoder()

    def _load_resume_checkpoint_if_requested(self):
        resume = self.cfg.TRAINER.BiMC.CHECKPOINT.RESUME
        if not resume:
            return
        try:
            payload = torch.load(resume, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(resume, map_location="cpu")
        schema = payload.get("schema_version")
        completed = payload.get("completed_session", -1)
        explicit = self.cfg.DATASET.SUPPORT_MANIFEST_PATH
        if explicit and Path(explicit).is_file():
            if json.loads(Path(explicit).read_text()) != payload["support_manifest"]:
                raise ValueError("Explicit support manifest differs from checkpoint")
        if schema in (7, 8, 9) and completed == 0:
            if Path(resume).resolve().parent == self._checkpoint_dir().resolve():
                raise ValueError("Use a different output directory to preserve the legacy base checkpoint")
            # Load the exact trained base encoder using the legacy migration.
            super()._load_resume_checkpoint_if_requested()
            loader = self.data_manager.get_dataloader(0, source="train", mode="test", accumulate_past=False)
            cache = self._extract_memory(0, loader)
            # Recompute fusion with the requested calibration config, then
            # re-evaluate S0; never mix rounded legacy accuracy into the new AA.
            self._initialize_base(cache)
            result = self.evaluate_global(0)
            self.acc_list, self.task_acc_list = [result["mean_acc"]], [result["task_acc"]]
            self.session_diagnostics = [{"session": 0, "migration": f"base-schema-{schema}",
                                         "classifier": result}]
            self._resume_state_dict_list = []
            self.save_checkpoint(0, [])
            return
        if schema != 10:
            raise ValueError("Global MoE requires schema 10 or a base-only schema 7/8/9 checkpoint; "
                             "legacy incremental checkpoints cannot be converted")
        if payload["protocol"] != self._protocol():
            raise ValueError("Checkpoint protocol/configuration differs; use matching settings")
        memories = payload["feature_memories"]
        if completed != len(memories) - 1 or completed < 0:
            raise ValueError("Checkpoint session/memory mismatch")
        for key in ("acc_list", "task_acc_list", "session_diagnostics"):
            if len(payload[key]) != len(memories):
                raise ValueError(f"Checkpoint session/{key} mismatch")
        self.data_manager.load_support_manifest(payload["support_manifest"])
        self.model_without_dp.rebuild_moe_topology(payload["moe_topology"])
        self.model_without_dp.load_state_dict(payload["model_state"], strict=True)
        self.feature_head = GlobalFeatureMoE.restore(payload["feature_head"], self.device)
        if self.feature_head.scale != self.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE:
            raise ValueError("Checkpoint head temperature differs from protocol")
        if len(self.feature_head.banks) != len(memories):
            raise ValueError("Checkpoint classifier/memory mismatch")
        self.feature_memories = [self._move_state_to_device(m, self.device) for m in memories]
        for i, memory in enumerate(self.feature_memories):
            self._check_memory(memory)
            expected = torch.as_tensor(self.data_manager.class_index_in_task[i], device=self.device)
            if not torch.equal(memory["class_ids"], expected) or not torch.equal(
                    memory["class_ids"], self.feature_head.banks[i].class_ids):
                raise ValueError("Checkpoint class binding differs from dataset")
        self.acc_list = payload["acc_list"]
        self.task_acc_list = payload["task_acc_list"]
        self.session_diagnostics = payload["session_diagnostics"]
        self._resume_completed_session = completed
        self._freeze_encoder()
        self._restore_rng_state(payload["rng_state"])
        print(f"=> Resumed global-risk session {completed} from {resume}")

    def save_checkpoint(self, completed_session, state_dict_list=None):
        if not self._checkpoint_enabled():
            return None
        directory = self._checkpoint_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 10, "completed_session": completed_session,
                   "protocol": self._protocol(), "config": self.cfg.dump(),
                   "model_state": self._cpu_copy(self.model_without_dp.state_dict()),
                   "moe_topology": self.model_without_dp.export_moe_topology(),
                   "feature_head": self.feature_head.export(),
                   "feature_memories": self._cpu_copy(self.feature_memories),
                   "support_manifest": self.data_manager.get_support_manifest(),
                   "acc_list": list(self.acc_list), "task_acc_list": self.task_acc_list,
                   "session_diagnostics": self.session_diagnostics, "rng_state": self._rng_state()}
        target = directory / f"session_{completed_session:02d}.pth"
        paths = [directory / "latest.pth"]
        if self.cfg.TRAINER.BiMC.CHECKPOINT.SAVE_EVERY_SESSION:
            paths.append(target)
        for path in paths:
            temp = path.with_suffix(".pth.tmp")
            torch.save(payload, temp)
            os.replace(temp, path)
        print(f"=> Saved global-risk checkpoint: {paths[-1]}")
        return paths[-1]

    @torch.no_grad()
    def evaluate_global(self, task_id):
        loader = self.data_manager.get_dataloader(task_id, source="test", mode="test")
        logits, labels = [], []
        self.model.eval()
        self.feature_head.eval()
        for batch in tqdm(loader, desc=f"Global Eval {task_id}"):
            images, target = self.parse_batch(batch)
            z = self.model_without_dp.extract_img_feature(images, max_session=0)
            logits.append(self.feature_head(unit(z)))
            labels.append(target)
        scores, targets = torch.cat(logits), torch.cat(labels)
        result = self.evaluator.calc_accuracy(scores, targets, task_id, self.feature_head.class_ids)
        # Legacy display fields are rounded; compute experiment metrics from counts.
        predictions = self.feature_head.class_ids[scores.argmax(1)]
        correct = predictions.eq(targets)
        result["mean_acc"] = 100.0 * int(correct.sum()) / len(targets)
        result["task_acc"] = []
        for classes in self.data_manager.class_index_in_task[:task_id + 1]:
            mask = torch.isin(targets, torch.as_tensor(classes, device=targets.device))
            result["task_acc"].append(100.0 * int(correct[mask].sum()) / int(mask.sum()))
        result["base_avg_acc"] = result["task_acc"][0]
        result["inc_avg_acc"] = (sum(result["task_acc"][1:]) / task_id if task_id else 0.0)
        base, novel = result["base_avg_acc"], result["inc_avg_acc"]
        result["harmonic_acc"] = 2 * base * novel / (base + novel) if base + novel else 0.0
        result["macro_task_acc"] = sum(result["task_acc"]) / len(result["task_acc"])
        result["error_flow"] = self.evaluator.error_flow(scores, targets, task_id, self.feature_head.class_ids)
        return result

    def run(self):
        end = self.cfg.TRAINER.BiMC.END_SESSION
        end = self.data_manager.num_tasks - 1 if end < 0 else min(end, self.data_manager.num_tasks - 1)
        for task_id in range(self._resume_completed_session + 1, end + 1):
            start = time.perf_counter()
            loader = self.data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
            print(f"=> [Session {task_id}] support_manifest={self._support_manifest_hash()}")
            if task_id == 0:
                names = [self.data_manager.class_names[i] for i in self.data_manager.class_index_in_task[0]]
                state = self.model_without_dp.build_task_statistics(
                    names, loader, self.data_manager.class_index_in_task[0], calibrate_novel_vision_proto=False)
                if self.cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE:
                    train = self.data_manager.get_dataloader(0, source="train", mode="train", accumulate_past=False)
                    self.train_base_task(state, train, loader)
                cache = self._extract_memory(0, loader)
                self._initialize_base(cache)
                decision = {"selected": "base", "num_experts": 0}
            else:
                cache = self._extract_memory(task_id, loader)
                decision = FeatureRiskTrainer(self.feature_head, self.risk_config).add_session(
                    cache, self.feature_memories)
                self.feature_memories.append(cache)
            print(f"=> [GlobalRisk][Task {task_id}] {json.dumps(decision)}")
            eval_start = time.perf_counter()
            result = self.evaluate_global(task_id)
            evaluation_seconds = time.perf_counter() - eval_start
            self.acc_list.append(result["mean_acc"])
            self.task_acc_list.append(result["task_acc"])
            aa = sum(self.acc_list) / len(self.acc_list)
            pd = self.acc_list[0] - self.acc_list[-1]
            memory_bytes = sum(v.numel() * v.element_size() for m in self.feature_memories for v in m.values())
            self.session_diagnostics.append({"session": task_id, "adaptation": decision,
                "classifier": result, "AA": aa, "PD": pd, "feature_memory_bytes": memory_bytes,
                "head_parameters": sum(p.numel() for p in self.feature_head.parameters()),
                "evaluation_seconds": evaluation_seconds, "seconds": time.perf_counter() - start})
            print(f"=> Task [{task_id}], Acc: {result['mean_acc']:.4f}, "
                  f"Base: {result['base_avg_acc']:.4f}, Novel: {result['inc_avg_acc']:.4f}, "
                  f"AA: {aa:.4f}, PD: {pd:.4f}, feature_memory_bytes={memory_bytes}")
            self._resume_completed_session = task_id
            self.save_checkpoint(task_id)
        print(f"Final acc: {self.acc_list}")
        print(f"Task-wise acc: {self.task_acc_list}")
