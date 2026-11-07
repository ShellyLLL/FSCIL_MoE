"""Single-classifier FSCIL training and APC-guided dynamic MoE expansion."""
import copy
import json
import os
import random
import time
from statistics import NormalDist
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets.data_manager import DatasetManager
from models.bimc import BiMC
from utils.evaluator import AccuracyEvaluator


def _get(node, name, default):
    return getattr(node, name, default) if node is not None else default


class Runner:
    """All sessions share one current encoder and one all-seen BiMC cosine classifier."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE.DEVICE_NAME
        self.data_manager = DatasetManager(cfg)
        self.model = BiMC(cfg, self.data_manager.template, self.device)
        self.acc_list, self.task_acc_list, self.session_diagnostics = [], [], []
        self.states = []
        self.apc_state = None
        self.start_session = 0
        self._load_checkpoint()

    def parse_batch(self, batch):
        return self.model.parse_batch(batch)

    def _adapters(self):
        return self.model.adapters()

    @staticmethod
    def _source(batch):
        value = batch.get("source_ids", batch.get("source_id", batch.get("index")))
        if value is None:
            raise KeyError("FSCIL training and validation require stable source IDs.")
        return torch.as_tensor(value).long().reshape(-1)

    @staticmethod
    def _merge_states(states, device):
        rows = []
        for state in states:
            for i, class_id in enumerate(torch.as_tensor(state["class_ids"]).tolist()):
                rows.append((int(class_id), state, i))
        rows.sort(key=lambda row: row[0])
        ordered_ids = [r[0] for r in rows]
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("Global prototype bank contains duplicate class IDs.")
        ids = torch.tensor(ordered_ids, dtype=torch.long, device=device)
        for state_key in ("image_proto", "text_features", "description_proto", "fused_proto"):
            values = [torch.as_tensor(state[state_key])[i] for _, state, i in rows]
            if not values:
                raise ValueError("Cannot build an empty global classifier.")
            # Every prototype must share the current global embedding dimension.
            if len({tuple(v.shape) for v in values}) != 1:
                raise ValueError(f"Global prototype field {state_key} has inconsistent dimensions.")
            rows_value = torch.stack(values).to(device)
            if not torch.isfinite(rows_value).all():
                raise FloatingPointError(f"Non-finite values in global {state_key}.")
            if state_key == "fused_proto":
                fused = rows_value
            elif state_key == "image_proto":
                image = rows_value
            elif state_key == "text_features":
                text = rows_value
            else:
                desc = rows_value
        return {
            "class_ids": ids,
            "image_proto": image,
            "text_features": text,
            "description_proto": desc,
            "fused_proto": F.normalize(fused.float(), dim=-1),
        }

    def _columns(self, labels, class_ids):
        ids = torch.as_tensor(class_ids, device=labels.device, dtype=torch.long)
        match = labels.reshape(-1, 1).eq(ids.reshape(1, -1))
        if not match.sum(1).eq(1).all():
            raise ValueError("Each target must occur exactly once in the global class bank.")
        return match.long().argmax(1)

    def _support_folds(self, dataset, n_splits=None):
        labels = np.asarray(dataset.labels)
        sources = np.asarray(dataset.source_ids)
        if labels.shape != sources.shape:
            raise ValueError("Support labels and source IDs do not align.")
        by_class = {int(c): np.unique(sources[labels == c]) for c in np.unique(labels)}
        n = min(map(len, by_class.values()))
        if n < 3:
            return []
        folds = []
        for k in range(n):
            held = {c: s[k] for c, s in by_class.items()}
            val_idx = [i for i, (y, sid) in enumerate(zip(labels, sources)) if held[int(y)] == sid]
            fit_idx = sorted(set(range(len(labels))) - set(val_idx))
            if not val_idx or not fit_idx:
                raise RuntimeError("Empty source-exclusive CV fold.")
            folds.append((fit_idx, val_idx))
        return folds

    def _loader(self, dataset, indices, batch_size, shuffle):
        return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.cfg.DATALOADER.NUM_WORKERS, drop_last=False)

    @torch.no_grad()
    def _extract_support(self, loader):
        feats, labels, sources = [], [], []
        self.model.eval()
        for batch in loader:
            images, y = self.parse_batch(batch)
            feats.append(F.normalize(self.model.extract_img_feature(images).float(), dim=-1))
            labels.append(y)
            sources.append(self._source(batch).to(self.device))
        return torch.cat(feats), torch.cat(labels), torch.cat(sources)

    @torch.no_grad()
    def _class_prototypes(self, features, labels, class_ids, exclude_sources=None, query_sources=None):
        class_ids = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
        sources = None if exclude_sources is None else torch.as_tensor(
            exclude_sources, device=features.device
        ).reshape(-1)
        if query_sources is None:
            result = []
            for cls in class_ids:
                selected = features[labels.eq(cls)]
                if selected.numel() == 0:
                    raise ValueError(f"No features for class {int(cls)}.")
                result.append(F.normalize(selected.mean(0), dim=-1))
            return torch.stack(result)
        banks = []
        for sid, query_label in zip(query_sources, exclude_sources):
            prototypes = []
            for cls in class_ids:
                mask = labels.eq(cls)
                if int(cls) == int(query_label):
                    mask = mask & sources.ne(sid)
                selected = features[mask]
                if selected.numel() == 0:
                    raise ValueError("A query has no source-exclusive support prototype.")
                prototypes.append(F.normalize(selected.mean(0), dim=-1))
            banks.append(torch.stack(prototypes))
        return torch.stack(banks)

    def _text(self, class_ids):
        names = np.asarray(self.data_manager.class_names)[np.asarray(class_ids, dtype=np.int64)].tolist()
        text = self.model.inference_text_feature(names, self.data_manager.template)
        desc_list, desc = self.model.inference_all_description_feature(names, self.cfg.DATASET.GPT_PATH)
        return text, desc, desc_list

    def _fuse(self, visual, text, desc):
        lam_t = float(self.cfg.TRAINER.BiMC.LAMBDA_T) if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        text_cal = F.normalize((1-lam_t)*text + lam_t*desc, dim=-1)
        beta = float(self.cfg.DATASET.BETA)
        return F.normalize(beta*text_cal + (1-beta)*visual, dim=-1)

    def _build_state(self, class_ids, loader, old_states, calibrate_visual):
        feat, labels, sources = self._extract_support(loader)
        visual = self._class_prototypes(feat, labels, class_ids)
        if old_states and calibrate_visual:
            visual = self.model.calibrate_incremental_visual(visual)
        names = np.asarray(self.data_manager.class_names)[np.asarray(class_ids, dtype=np.int64)].tolist()
        text = self.model.inference_text_feature(names, self.data_manager.template)
        desc_list, desc = self.model.inference_all_description_feature(names, self.cfg.DATASET.GPT_PATH)
        fused = self._fuse(visual, text, desc)
        raw_layers = self.model.collect_blockwise_cls_features(loader)
        raw_prototypes = {}
        for block, item in raw_layers.items():
            vals, ys = item["raw_reference_cls"], item["labels"]
            rows = []
            for cls in class_ids:
                v = vals[ys.eq(int(cls))]
                if not len(v):
                    raise ValueError(f"Missing raw reference features for class {cls}, block {block}.")
                rows.append(F.normalize(v.float().mean(0), dim=-1))
            raw_prototypes[int(block)] = torch.stack(rows).cpu()
        return {
            "class_ids": torch.as_tensor(class_ids, dtype=torch.long),
            "image_proto": visual.detach().cpu(),
            "text_features": text.detach().cpu(),
            "description_proto": desc.detach().cpu(),
            "description_features": [x.detach().cpu() for x in desc_list],
            "fused_proto": fused.detach().cpu(),
            "raw_layer_prototypes": raw_prototypes,
            "creation_session": len(old_states),
        }

    def _raw_history(self, states, block):
        ids, protos = [], []
        for state in states:
            raw = state.get("raw_layer_prototypes", {}).get(int(block))
            if raw is None:
                raise RuntimeError(f"Historical raw prototype missing for MoE block {block}.")
            ids.extend(torch.as_tensor(state["class_ids"]).tolist())
            protos.extend(torch.as_tensor(raw).unbind(0))
        return torch.tensor(ids, dtype=torch.long), torch.stack(protos).to(self.device)

    def _calibrate_apc(self, base_state):
        thresholds = {}
        for block, item in base_state["raw_layer_prototypes"].items():
            proto = F.normalize(item.float(), dim=-1)
            similarity = proto @ proto.t()
            similarity.fill_diagonal_(-1.0)
            nearest = similarity.max(dim=1).values
            alpha = float(self.cfg.TRAINER.BiMC.APC.ALPHA)
            thresholds[int(block)] = float(torch.quantile(nearest, 1.0-alpha).item())
        self.apc_state = {"thresholds": thresholds, "schema": 1}

    def _candidate_layers(self, new_state, old_states):
        result = []
        for block, threshold in self.apc_state["thresholds"].items():
            new_proto = F.normalize(new_state["raw_layer_prototypes"][block].to(self.device).float(), dim=-1)
            old_ids, old_proto = self._raw_history(old_states, block)
            confusion = (new_proto @ old_proto.t()).max().item()
            result.append({"block": block, "confusion": float(confusion),
                           "threshold": float(threshold), "candidate": confusion > threshold})
        return result

    def _candidate_train(self, task_id, selected_blocks, fit_loader, fit_extract_loader,
                         old_states, class_ids, text, desc):
        for _, adapter in self._adapters():
            adapter.freeze_all()
        for block in selected_blocks:
            self.model.add_expert(block, task_id)
        optim_cfg = self.cfg.TRAINER.BiMC.OPTIM
        params = []
        for block in selected_blocks:
            adapter = dict(self._adapters())[block]
            params += [{"params": list(adapter.experts[-1].parameters()),
                        "lr": float(optim_cfg.LR_INCREMENTAL_EXPERT)},
                       {"params": list(adapter.incremental_routers[-1].parameters()),
                        "lr": float(optim_cfg.LR_INCREMENTAL_ROUTER)}]
        if not params:
            return []
        optimizer = optim.AdamW(params, weight_decay=float(optim_cfg.WEIGHT_DECAY))
        epochs = int(self.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)
        fit_feat, fit_labels, fit_sources = self._extract_support(fit_extract_loader)
        # Historical classifier rows are immutable; only current-session rows
        # are recomputed from the candidate encoder and query-exclusive support.
        old_bank = self._merge_states(old_states, self.device)
        old_proto = old_bank["fused_proto"]
        lambda_t = float(self.cfg.TRAINER.BiMC.LAMBDA_T) if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        new_text = F.normalize((1-lambda_t)*text + lambda_t*desc, dim=-1)
        beta = float(self.cfg.DATASET.BETA)
        history_raw = {b: self._raw_history(old_states, b)[1] for b in selected_blocks}
        logs = []
        for epoch in range(epochs):
            self.model.train()
            total = 0.0
            for batch in fit_loader:
                images, labels = self.parse_batch(batch)
                sid = self._source(batch).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                query_feat, _ = self.model.extract_img_feature_train(images)
                query_feat = F.normalize(query_feat.float(), dim=-1)
                # LOO prototypes are rebuilt from the current candidate weights.
                current_support, current_labels, current_sources = self._extract_support(fit_extract_loader)
                self.model.train()
                proto_rows = []
                for qlabel, qsid in zip(labels, sid):
                    class_rows = []
                    for cls in class_ids:
                        mask = current_labels.eq(int(cls))
                        if int(cls) == int(qlabel):
                            mask = mask & current_sources.ne(qsid)
                        chosen = current_support[mask]
                        if not chosen.numel():
                            raise ValueError("Incremental classification needs at least two sources per class.")
                        class_rows.append(F.normalize(chosen.mean(0), dim=-1))
                    proto_rows.append(torch.stack(class_rows))
                visual_loo = torch.stack(proto_rows)
                if self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
                    visual_loo = self.model.calibrate_incremental_visual(
                        visual_loo.reshape(-1, visual_loo.size(-1))
                    ).reshape_as(visual_loo)
                fused_loo = F.normalize(beta*new_text[None] + (1-beta)*visual_loo, dim=-1)
                full_bank = torch.cat([old_proto[None].expand(len(labels), -1, -1), fused_loo], dim=1)
                logits = torch.einsum("nd,ncd->nc", query_feat, full_bank)
                all_ids = torch.cat([old_bank["class_ids"], torch.as_tensor(class_ids, device=self.device)])
                target_ids = all_ids
                targets = self._columns(labels, target_ids)
                loss = F.cross_entropy(logits, targets)
                # Suppress each new expert on stored old-class raw reference prototypes.
                gate_terms = []
                for block in selected_blocks:
                    adapter = dict(self._adapters())[block]
                    gate_terms.append(adapter.newest_gate(history_raw[block]).square().mean())
                if gate_terms:
                    loss = loss + float(self.cfg.TRAINER.BiMC.PROTECTION.OLD_GATE_WEIGHT) * torch.stack(gate_terms).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite all-class incremental loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(labels)
            logs.append({"epoch": epoch+1, "loss": total/max(1,len(fit_extract_loader.dataset))})
        for _, adapter in self._adapters():
            adapter.freeze_all()
        return logs

    @torch.no_grad()
    def _evaluate_candidate(self, validation_loader, fit_extract_loader, old_states,
                            class_ids, text, desc):
        self.model.eval()
        feat, labels, sources = self._extract_support(fit_extract_loader)
        old_bank = self._merge_states(old_states, self.device)
        visual = self._class_prototypes(feat, labels, class_ids)
        if self.cfg.TRAINER.BiMC.VISION_CALIBRATION:
            visual = self.model.calibrate_incremental_visual(visual)
        new_fused = self._fuse(visual, text, desc)
        all_proto = torch.cat([old_bank["fused_proto"], new_fused], dim=0)
        all_ids = torch.cat([old_bank["class_ids"], torch.as_tensor(class_ids, device=self.device)])
        loss_sum, count = 0.0, 0
        for batch in validation_loader:
            images, y = self.parse_batch(batch)
            q = F.normalize(self.model.extract_img_feature(images).float(), dim=-1)
            logits = q @ all_proto.t()
            target = self._columns(y, all_ids)
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
            count += len(y)
        return loss_sum/max(1,count)

    def _select_and_train_expansion(self, task_id, train_loader, extract_loader,
                                    old_states, class_ids, class_names):
        """Evaluate one predeclared APC expansion policy with source-held-out CV.

        Each fold proposes layers from its fitting sources only, jointly trains
        all proposed layers, and evaluates the resulting all-seen-class risk on
        that fold's untouched sources. We do not intersect fold proposals or
        select a winning subset on the same validation folds.
        """
        if not self.apc_state:
            raise RuntimeError("Base APC thresholds are not initialized.")
        folds = self._support_folds(train_loader.dataset)
        if not folds:
            raise RuntimeError(
                "Source-exclusive expansion validation requires at least 3 shots per class."
            )
        text, desc, _ = self._text(class_ids)
        fold_candidate_sets = []
        fold_baseline_losses = []
        fold_expanded_losses = []
        fold_gains = []

        for fit_idx, val_idx in folds:
            fit_train = self._loader(
                train_loader.dataset, fit_idx, train_loader.batch_size, True
            )
            fit_extract = self._loader(
                extract_loader.dataset, fit_idx, extract_loader.batch_size, False
            )
            val_loader = self._loader(
                extract_loader.dataset, val_idx, extract_loader.batch_size, False
            )

            # APC proposal is derived only from this fold's fitting sources.
            fold_state = self._build_state(class_ids, fit_extract, old_states, False)
            fold_candidates = sorted(
                row["block"] for row in self._candidate_layers(fold_state, old_states)
                if row["candidate"]
            )
            fold_candidate_sets.append(fold_candidates)

            topology_before = self.model.export_moe_topology()
            state_before = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            try:
                baseline_loss = self._evaluate_candidate(
                    val_loader, fit_extract, old_states, class_ids, text, desc
                )
                if fold_candidates:
                    self._candidate_train(
                        task_id, fold_candidates, fit_train, fit_extract,
                        old_states, class_ids, text, desc
                    )
                    expanded_loss = self._evaluate_candidate(
                        val_loader, fit_extract, old_states, class_ids, text, desc
                    )
                else:
                    # The policy predicts no expansion in this fold.
                    expanded_loss = baseline_loss
                gain = float(baseline_loss - expanded_loss)
                fold_baseline_losses.append(float(baseline_loss))
                fold_expanded_losses.append(float(expanded_loss))
                fold_gains.append(gain)
            finally:
                self.model.rebuild_moe_topology(topology_before)
                self.model.load_state_dict(state_before, strict=True)
                for _, adapter in self._adapters():
                    adapter.freeze_all()

        # Re-apply the same fixed APC proposal rule to all current support
        # sources for the final fit. Validation folds estimate the policy;
        # they do not choose individual layers or subsets.
        full_state = self._build_state(class_ids, extract_loader, old_states, False)
        final_candidates = sorted(
            row["block"] for row in self._candidate_layers(full_state, old_states)
            if row["candidate"]
        )

        gains = np.asarray(fold_gains, dtype=np.float64)
        mean_gain = float(gains.mean()) if len(gains) else 0.0
        if len(gains) > 1:
            # Corrected resampled-t uncertainty accounts conservatively for
            # overlap between fitting folds (test/train ratio = 1/(K-1)).
            sample_variance = float(gains.var(ddof=1))
            test_train_ratio = 1.0 / max(1, len(gains) - 1)
            corrected_se = float(np.sqrt((1.0 / len(gains) + test_train_ratio)
                                         * sample_variance))
            from scipy.stats import t as student_t
            significance = float(
                _get(self.cfg.TRAINER.BiMC.APC, "SIGNIFICANCE", 0.05)
            )
            critical = float(student_t.ppf(1.0 - significance, df=len(gains) - 1))
            required_gain = critical * corrected_se
        else:
            corrected_se = float("inf")
            critical = float("inf")
            required_gain = float("inf")

        diagnostics = {
            "fold_candidate_layers": fold_candidate_sets,
            "fold_baseline_nll": fold_baseline_losses,
            "fold_expanded_nll": fold_expanded_losses,
            "fold_risk_gains": fold_gains,
            "mean_risk_gain": mean_gain,
            "corrected_standard_error": corrected_se,
            "critical_value": critical,
            "required_risk_gain": required_gain,
            "final_candidate_layers": final_candidates,
        }

        if not any(fold_candidate_sets):
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": "APC proposed no layer in any fitting fold",
                **diagnostics, "logs": [],
            }
        if not final_candidates:
            return {
                "expanded": [], "candidate_layers": [],
                "reason": "APC proposed no layer on the full support set",
                **diagnostics, "logs": [],
            }
        if not np.isfinite(mean_gain) or mean_gain <= required_gain:
            return {
                "expanded": [], "candidate_layers": final_candidates,
                "reason": "Held-out full-class risk improvement did not clear the uncertainty bound",
                **diagnostics, "logs": [],
            }

        topology_before = self.model.export_moe_topology()
        state_before = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        try:
            logs = self._candidate_train(
                task_id, final_candidates, train_loader, extract_loader,
                old_states, class_ids, text, desc
            )
        except Exception:
            self.model.rebuild_moe_topology(topology_before)
            self.model.load_state_dict(state_before, strict=True)
            for _, adapter in self._adapters():
                adapter.freeze_all()
            raise
        return {
            "expanded": list(final_candidates),
            "candidate_layers": final_candidates,
            "reason": "Held-out full-class risk improvement accepted",
            **diagnostics, "logs": logs,
        }

    def _base_train(self, state, train_loader, extract_loader):
        for _, adapter in self._adapters():
            adapter.set_base_trainable()
        params = [p for _, a in self._adapters() for p in a.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=float(self.cfg.TRAINER.BiMC.OPTIM.LR_BASE_MOE),
                                weight_decay=float(self.cfg.TRAINER.BiMC.OPTIM.WEIGHT_DECAY))
        epochs = int(self.cfg.TRAINER.BiMC.OPTIM.BASE_EPOCHS)
        class_ids = state["class_ids"].to(self.device)
        text, desc = state["text_features"].to(self.device), state["description_proto"].to(self.device)
        for epoch in range(epochs):
            self.model.eval()
            feat, labels, _ = self._extract_support(extract_loader)
            visual = self._class_prototypes(feat, labels, class_ids)
            prototypes = self._fuse(visual, text, desc)
            self.model.train()
            for batch in train_loader:
                images, y = self.parse_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                logits, _, _ = self.model.forward_train(images, prototypes)
                loss = F.cross_entropy(logits, self._columns(y, class_ids))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            print(f"=> Base epoch {epoch+1}/{epochs} complete")
        for _, adapter in self._adapters():
            adapter.freeze_all()

    @torch.no_grad()
    def _evaluate_session(self, task_id, global_state):
        loader = self.data_manager.get_dataloader(task_id, source="test", mode="test")
        prototypes = global_state["fused_proto"].to(self.device)
        ids = global_state["class_ids"].to(self.device)
        logits_all, labels_all = [], []
        self.model.eval()
        for batch in loader:
            images, labels = self.parse_batch(batch)
            feat = F.normalize(self.model.extract_img_feature(images).float(), dim=-1)
            logits_all.append(feat @ prototypes.t())
            labels_all.append(labels)
        logits, labels = torch.cat(logits_all), torch.cat(labels_all)
        return self.evaluator.calc_accuracy(logits, labels, task_id, class_ids=ids)

    def _save_checkpoint(self, completed_session):
        cfg = self.cfg.TRAINER.BiMC.CHECKPOINT
        if not bool(_get(cfg, "ENABLE", True)):
            return
        path = Path(_get(cfg, "DIR", "checkpoints/fscil_moe"))
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 20,
            "completed_session": int(completed_session),
            "model_state": {k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            "moe_topology": self.model.export_moe_topology(),
            "states": self.states,
            "apc_state": self.apc_state,
            "acc_list": self.acc_list,
            "task_acc_list": self.task_acc_list,
            "support_manifest": self.data_manager.get_support_manifest(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        target = path / f"session_{completed_session:02d}.pth"
        temp = target.with_suffix(".tmp")
        torch.save(payload, temp)
        os.replace(temp, target)
        latest = path / "latest.pth"
        latest_tmp = latest.with_suffix(".tmp")
        torch.save(payload, latest_tmp)
        os.replace(latest_tmp, latest)

    def _load_checkpoint(self):
        resume = _get(self.cfg.TRAINER.BiMC.CHECKPOINT, "RESUME", "")
        if not resume:
            return
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if int(payload.get("schema_version", -1)) != 20:
            raise ValueError("This unified architecture requires checkpoint schema 20; restart from Session 0.")
        self.model.rebuild_moe_topology(payload["moe_topology"])
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.states = payload["states"]
        if not self.states:
            raise ValueError("Unified checkpoint has no base-session prototype state.")
        # Restore the exact post-base-training anchor used by every later
        # incremental visual calibration step.
        self.model.set_base_visual_prototype(self.states[0]["image_proto"])
        self.apc_state = payload["apc_state"]
        self.acc_list = payload["acc_list"]
        self.task_acc_list = payload["task_acc_list"]
        self.data_manager.load_support_manifest(payload["support_manifest"])
        self.start_session = int(payload["completed_session"]) + 1
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])

    def run(self):
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)
        end = self.data_manager.num_tasks - 1
        configured = int(self.cfg.TRAINER.BiMC.END_SESSION)
        if configured >= 0:
            end = min(end, configured)
        for task_id in range(self.start_session, end+1):
            # Session 0 does not run incremental expansion, but the common
            # reporting path below still records the (empty) expansion result.
            adaptation = {"expanded": []}
            classes = self.data_manager.class_index_in_task[task_id]
            names = np.asarray(self.data_manager.class_names)[classes].tolist()
            train = self.data_manager.get_dataloader(task_id, "train", mode="train", accumulate_past=False)
            extract = self.data_manager.get_dataloader(task_id, "train", mode="test", accumulate_past=False)
            if task_id == 0:
                state = self._build_state(classes, extract, [], False)
                if self._adapters():
                    self._base_train(state, train, extract)
                    # Rebuild prototypes after base expert training. This is
                    # the final base feature space and therefore the only
                    # valid anchor for all incremental visual calibration.
                    state = self._build_state(classes, extract, [], False)
                self.model.set_base_visual_prototype(state["image_proto"])
                self.states.append(state)
                if self._adapters():
                    self._calibrate_apc(state)
            else:
                adaptation = self._select_and_train_expansion(
                    task_id, train, extract, self.states, classes, names
                ) if self._adapters() else {"expanded": []}
                state = self._build_state(
                    classes, extract, self.states,
                    bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                )
                self.states.append(state)
            global_state = self._merge_states(self.states, self.device)
            metrics = self._evaluate_session(task_id, global_state)
            task_acc = metrics["task_acc"]
            pd = 0.0
            if task_id > 0:
                drops = []
                for old_task in range(task_id):
                    best = max(self.task_acc_list[s][old_task] for s in range(old_task, task_id))
                    drops.append(max(0.0, best-task_acc[old_task]))
                pd = float(np.mean(drops)) if drops else 0.0
            self.acc_list.append(metrics["mean_acc"])
            self.task_acc_list.append(task_acc)
            self._save_checkpoint(task_id)
            print(f"=> Session {task_id}: Acc={metrics['mean_acc']:.3f}, AA={metrics['macro_task_acc']:.3f}, PD={pd:.3f}, expanded={adaptation.get('expanded', [])}")
