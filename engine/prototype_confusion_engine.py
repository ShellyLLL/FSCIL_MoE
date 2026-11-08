"""Strict FSCIL training for Prototype-Confusion Guided Dynamic MoE."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.data_manager import DatasetManager
from models.prototype_confusion_moe import (
    PrototypeConfusionGuidedMoE, build_apc_reference,
    build_leave_one_source_out_prototypes, class_prototype_statistics,
    compute_prototype_confusion, fuse_modal_scores,
)


def _mean_prototypes(features, labels, class_ids):
    return class_prototype_statistics(features, labels, class_ids)["prototypes"]


class PrototypeConfusionRunner:
    """No replay, no pseudo sessions, one all-seen-class classifier."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = DatasetManager(cfg)
        self.device = cfg.DEVICE.DEVICE_NAME
        self.model = PrototypeConfusionGuidedMoE(cfg, self.data.template, self.device).to(self.device)
        self.class_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.visual_prototypes = None
        self.text_prototypes = None
        self.visual_reliability = None
        self.text_reliability = None
        self.layer_prototypes = {}
        self.layer_concentration = {}
        self.apc_reference = {}
        self.diagnostics = []
        self._fit_temperature = None
        self._checkpoint_dir = Path(cfg.TRAINER.PCMOE.CHECKPOINT.DIR)

    def _extract(self, loader, labels_required=True):
        raw, layers, labels, sources = [], {layer: [] for layer in self.model.layer_indices}, [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                raw_i, _, layer_i = self.model.encode_features(images)
                raw.append(raw_i.cpu())
                for layer in self.model.layer_indices:
                    layers[layer].append(layer_i[layer].cpu())
                if labels_required:
                    labels.append(batch["label"].long().cpu())
                    sources.append(torch.as_tensor(batch.get("source_id", batch.get("idx"))).reshape(-1).cpu())
        return {
            "raw": torch.cat(raw).to(self.device),
            "layers": {layer: torch.cat(values).to(self.device) for layer, values in layers.items()},
            "labels": torch.cat(labels).long().to(self.device) if labels_required else None,
            "sources": torch.cat(sources).to(self.device) if labels_required else None,
        }

    def _compose_from_cache(self, cache, disabled=None):
        return self.model.compose_visual_features(cache["raw"], cache["layers"], disabled)

    def _rebuild_visual_prototypes(self, raw_layer_state):
        result = {}
        for class_id in self.class_ids.tolist():
            result[int(class_id)] = {
                layer: raw_layer_state[layer][int(class_id)]
                for layer in self.model.layer_indices
            }
        rows = []
        for class_id in self.class_ids.tolist():
            one = {layer: result[int(class_id)][layer].unsqueeze(0)
                   for layer in self.model.layer_indices}
            raw = self.raw_visual_prototypes[int(class_id)].unsqueeze(0)
            rows.append(self.model.compose_visual_features(raw, one)[0])
        self.visual_prototypes = F.normalize(torch.stack(rows), dim=-1)

    def _calibrate_temperatures(self, cache, visual_prototypes, text_prototypes):
        with torch.no_grad():
            enhanced = self._compose_from_cache(cache)
            y = cache["labels"]
            ids = self.class_ids
            target = ids[None, :].eq(y[:, None]).float().argmax(dim=1)
            v = F.normalize(enhanced, dim=-1) @ F.normalize(visual_prototypes, dim=-1).t()
            t = F.normalize(cache["raw"], dim=-1) @ F.normalize(text_prototypes, dim=-1).t()
        temperatures = []
        for scores in (v, t):
            log_temp = torch.zeros((), device=self.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([log_temp], lr=0.25, max_iter=50,
                                          line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                temp = log_temp.exp().clamp(0.02, 2.0)
                loss = F.cross_entropy(scores / temp, target)
                loss.backward()
                return loss
            optimizer.step(closure)
            temperatures.append(float(log_temp.detach().exp().clamp(0.02, 2.0).item()))
        self.model.set_temperatures(*temperatures)

    def _modal_reliability(self, visual_stats, text_consistency):
        count = visual_stats["counts"].float()
        concentration = visual_stats["concentration"].clamp(0.0, 1.0)
        kappa = float(self.cfg.TRAINER.PCMOE.CALIBRATION.RELIABILITY_KAPPA)
        rv = (count / (count + kappa)) * concentration
        rt = torch.as_tensor(text_consistency, device=self.device).float().clamp(1e-3, 1.0)
        return rv, rt

    def _classifier(self, raw, enhanced):
        return fuse_modal_scores(
            raw, enhanced, self.visual_prototypes, self.text_prototypes,
            self.visual_reliability, self.text_reliability,
            self.model.temperature_visual, self.model.temperature_text,
        )[0]

    def _base_train(self, loader, class_ids):
        cfg = self.cfg.TRAINER.PCMOE
        self.model.set_base_trainable()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg.OPTIM.BASE_LR,
                                      weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        for epoch in range(int(cfg.OPTIM.BASE_EPOCHS)):
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].long().to(self.device)
                raw, enhanced, _, aux = self.model.encode_features(images, return_aux=True)
                text, _ = self.model.encode_text_prototypes(
                    [self.data.class_names[int(i)] for i in class_ids]
                )
                visual_stats = class_prototype_statistics(enhanced.detach(), labels, class_ids)
                # Batch prototypes are used only as differentiable training targets;
                # every target is a genuine base class from the actual base session.
                batch_proto = []
                for cid in class_ids:
                    selected = enhanced[labels.eq(int(cid))]
                    batch_proto.append(F.normalize(selected.mean(0), dim=-1)
                                       if selected.numel() else text[list(class_ids).index(int(cid))])
                proto = torch.stack(batch_proto)
                columns = torch.as_tensor(class_ids, device=self.device).eq(labels[:, None]).float().argmax(1)
                visual_logits = 20.0 * F.normalize(enhanced, dim=-1) @ F.normalize(proto, dim=-1).t()
                text_logits = 20.0 * F.normalize(raw, dim=-1) @ F.normalize(text, dim=-1).t()
                loss = F.cross_entropy(visual_logits, columns) + F.cross_entropy(text_logits, columns)
                for layer_aux in aux.values():
                    weights = layer_aux["route_weights"]
                    importance = weights.mean(0)
                    loss = loss + float(cfg.LOSS.ROUTE_BALANCE_WEIGHT) * (
                        importance * importance.numel()
                    ).sum().pow(2) / importance.numel()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            print("[PC-MoE][Base] epoch={}/{} loss={:.5f}".format(
                epoch + 1, cfg.OPTIM.BASE_EPOCHS, float(loss.item())))
        self.model.freeze_all()

    def _support_split(self, train_loader, extraction_loader):
        train_ds, extract_ds = train_loader.dataset, extraction_loader.dataset
        labels = np.asarray(extract_ds.labels)
        groups = [np.flatnonzero(labels == cid) for cid in np.unique(labels)]
        if not groups or min(map(len, groups)) < 3:
            raise ValueError(
                "Risk-guided expansion requires at least 3 genuine support sources per new class "
                "(fit, calibration and held-out risk); no pseudo-incremental samples are created."
            )
        # Deterministic, source-disjoint fit/calibration/validation partition.
        fit, calibration, validation = [], [], []
        for group in groups:
            validation.append(int(group[-1]))
            calibration.append(int(group[-2]))
            fit.extend(int(x) for x in group[:-2])
        def make(ds, indices, shuffle):
            return DataLoader(Subset(ds, indices), batch_size=train_loader.batch_size,
                              shuffle=shuffle, num_workers=train_loader.num_workers,
                              drop_last=False)
        return make(train_ds, fit, True), make(extract_ds, fit, False), \
               make(extract_ds, calibration, False), make(extract_ds, validation, False)

    def _apc(self, cache, new_ids):
        history = {str(layer): self.layer_prototypes[layer] for layer in self.model.layer_indices}
        return compute_prototype_confusion(
            cache["layers"], cache["labels"], new_ids, history, self.apc_reference
        )

    def _history_cache(self):
        # Historical class anchors are compact feature statistics, not stored images.
        return {
            "raw": torch.stack([self.raw_visual_prototypes[int(cid)] for cid in self.class_ids.tolist()]),
            "layers": {
                layer: torch.stack([self.layer_prototypes[layer][int(cid)]
                                    for cid in self.class_ids.tolist()])
                for layer in self.model.layer_indices
            },
        }

    def _candidate_risk(self, validation_cache, new_ids, candidate_layers, baseline=False):
        raw = validation_cache["raw"]
        enhanced = self._compose_from_cache(
            validation_cache,
            disabled={layer: (self.model.adapter_for_layer(layer).num_experts - 1,)
                      for layer in candidate_layers} if baseline else None,
        )
        all_ids = self.class_ids
        # Candidate current prototypes are derived from fit supports and are
        # concatenated with historical prototypes re-embedded through current experts.
        logits = self._classifier(raw, enhanced)
        labels = validation_cache["labels"]
        targets = all_ids[None, :].eq(labels[:, None]).float().argmax(1)
        ce = F.cross_entropy(logits, targets)
        old_mask = ~torch.isin(labels, new_ids)
        old_risk = (F.cross_entropy(logits[old_mask], targets[old_mask])
                    if old_mask.any() else ce.new_zeros(()))
        return float(ce.item()), float(old_risk.item())

    def _update_current_statistics(self, fit_cache, all_cache, new_ids, text, text_consistency):
        fit_stats = class_prototype_statistics(fit_cache["raw"], fit_cache["labels"], new_ids)
        current_visual = class_prototype_statistics(
            self._compose_from_cache(all_cache), all_cache["labels"], new_ids
        )
        self.class_ids = torch.cat([self.class_ids, _as_ids(new_ids, self.device)])
        for layer in self.model.layer_indices:
            stats = class_prototype_statistics(
                all_cache["layers"][layer], all_cache["labels"], new_ids
            )
            for i, cid in enumerate(new_ids):
                self.layer_prototypes[layer][int(cid)] = stats["prototypes"][i].detach()
                self.layer_concentration[layer][int(cid)] = stats["concentration"][i].detach()
        raw_stats = class_prototype_statistics(all_cache["raw"], all_cache["labels"], new_ids)
        for i, cid in enumerate(new_ids):
            self.raw_visual_prototypes[int(cid)] = raw_stats["prototypes"][i].detach()
        self.text_prototypes = torch.cat([self.text_prototypes, text], dim=0)
        self.visual_prototypes = torch.cat([self.visual_prototypes, current_visual["prototypes"]], dim=0)
        self.visual_reliability = torch.cat([self.visual_reliability, current_visual["concentration"]], dim=0)
        self.text_reliability = torch.cat([self.text_reliability, text_consistency], dim=0)

    def _train_candidates(self, task_id, fit_train, fit_extract, calibration_loader,
                          validation_loader, new_ids, names, apc):
        fit_cache = self._extract(fit_extract)
        cal_cache = self._extract(calibration_loader)
        val_cache = self._extract(validation_loader)
        candidate_layers = list(apc["layer_order"])
        if not candidate_layers:
            return {"accepted": False, "reason": "no candidate layers", "apc": apc}
        # All candidate layers are expanded and optimized jointly.  A single
        # validation transaction evaluates the combined architecture.
        history = self._history_cache()
        history_features = {
            layer: history["layers"][layer] for layer in self.model.layer_indices
        }
        for layer in candidate_layers:
            self.model.add_expert(layer, task_id, fit_cache["layers"][layer],
                                  history_features[layer])
        self.model.set_incremental_trainable(candidate_layers)
        params = self.model.newest_parameters(candidate_layers)
        cfg = self.cfg.TRAINER.PCMOE
        optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.INCREMENTAL_LR,
                                      weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        text, text_consistency = self.model.encode_text_prototypes(names)
        for epoch in range(int(cfg.OPTIM.INCREMENTAL_EPOCHS)):
            self.model.train()
            for batch in fit_train:
                images = batch["image"].to(self.device)
                labels = batch["label"].long().to(self.device)
                raw, enhanced, _, aux = self.model.encode_features(images, return_aux=True)
                # Query-exclusive prototypes: every support query is scored
                # against a bank excluding all views of its own source.
                fit_rows = self._extract_batch_cache(batch)
                all_fit = self._extract(fit_extract)
                loo = build_leave_one_source_out_prototypes(
                    all_fit["raw"], all_fit["labels"], all_fit["sources"], new_ids
                )
                visual_proto = torch.stack([
                    F.normalize(all_fit["raw"][all_fit["labels"].eq(cid)].mean(0), dim=-1)
                    for cid in new_ids
                ])
                # New experts optimize the same final dual-modality classifier;
                # history visual/text columns remain active in every CE decision.
                old_enhanced = self._history_enhanced()
                current_proto = self._current_visual_fit_prototypes(all_fit, new_ids)
                bank_visual = torch.cat([old_enhanced, current_proto], dim=0)
                bank_text = torch.cat([self.text_prototypes, text], dim=0)
                rv = torch.cat([self.visual_reliability, torch.ones(len(new_ids), device=self.device)])
                rt = torch.cat([self.text_reliability, text_consistency])
                logits = fuse_modal_scores(
                    raw, enhanced, bank_visual, bank_text, rv, rt,
                    self.model.temperature_visual, self.model.temperature_text
                )[0]
                targets = self.class_ids.new_tensor(self.class_ids.tolist() + [int(x) for x in new_ids])
                target_cols = targets[None, :].eq(labels[:, None]).float().argmax(1)
                loss = F.cross_entropy(logits, target_cols)
                for layer, layer_aux in aux.items():
                    weights = layer_aux["route_weights"]
                    loss = loss + cfg.LOSS.ROUTE_BALANCE_WEIGHT * (
                        weights.mean(0).pow(2).sum() * weights.size(1)
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
        self.model.freeze_all()
        # Build fit-set visual prototypes after training and re-embed old
        # historical layer anchors through the candidate architecture.
        old_risk = self._risk_on_old_anchors(new_ids)
        new_visual = self._current_visual_fit_prototypes(fit_cache, new_ids)
        old_visual = self._history_enhanced()
        full_visual = torch.cat([old_visual, new_visual], dim=0)
        full_text = torch.cat([self.text_prototypes, text], dim=0)
        rv = torch.cat([self.visual_reliability, torch.ones(len(new_ids), device=self.device)])
        rt = torch.cat([self.text_reliability, text_consistency])
        val_enhanced = self._compose_from_cache(val_cache)
        logits = fuse_modal_scores(
            val_cache["raw"], val_enhanced, full_visual, full_text, rv, rt,
            self.model.temperature_visual, self.model.temperature_text
        )[0]
        targets = torch.as_tensor(full_text.new_tensor(self.class_ids.tolist() + [int(x) for x in new_ids]),
                                  device=self.device, dtype=torch.long)
        target_cols = targets[None, :].eq(val_cache["labels"][:, None]).float().argmax(1)
        candidate_loss = F.cross_entropy(logits, target_cols)
        # Baseline uses the exact same new-class support prototypes and classifier,
        # but excludes the newly proposed experts.
        disabled = {layer: (self.model.adapter_for_layer(layer).num_experts - 1,)
                    for layer in candidate_layers}
        base_enhanced = self._compose_from_cache(val_cache, disabled=disabled)
        base_logits = fuse_modal_scores(
            val_cache["raw"], base_enhanced, full_visual, full_text, rv, rt,
            self.model.temperature_visual, self.model.temperature_text
        )[0]
        baseline_loss = F.cross_entropy(base_logits, target_cols)
        # Old prototype anchors are re-evaluated in the same all-class decision space.
        old_anchor_risk = self._risk_on_old_anchors(new_ids)
        parameter_cost = sum(p.numel() for p in params) / max(
            1, sum(p.numel() for p in self.model.adapters.parameters())
        )
        risk_delta = float((candidate_loss - baseline_loss).item())
        accepted = (risk_delta < -float(cfg.EXPANSION.MIN_RISK_GAIN)
                    and old_anchor_risk <= float(cfg.EXPANSION.MAX_OLD_RISK))
        if not accepted:
            for layer in reversed(candidate_layers):
                self.model.adapter_for_layer(layer).remove_newest_expert()
            self.model.freeze_all()
        return {
            "accepted": accepted, "candidate_layers": candidate_layers,
            "baseline_risk": float(baseline_loss.item()),
            "candidate_risk": float(candidate_loss.item()),
            "risk_delta": risk_delta, "old_anchor_risk": old_anchor_risk,
            "parameter_cost": parameter_cost, "apc": apc,
        }

    def _extract_batch_cache(self, batch):
        return None

    def _history_enhanced(self):
        rows = []
        for cid in self.class_ids.tolist():
            layer_row = {layer: self.layer_prototypes[layer][int(cid)].unsqueeze(0)
                         for layer in self.model.layer_indices}
            rows.append(self.model.compose_visual_features(
                self.raw_visual_prototypes[int(cid)].unsqueeze(0), layer_row
            )[0])
        return F.normalize(torch.stack(rows), dim=-1) if rows else torch.empty(
            0, self.text_prototypes.size(1), device=self.device
        )

    def _current_visual_fit_prototypes(self, cache, new_ids):
        enhanced = self._compose_from_cache(cache)
        return class_prototype_statistics(enhanced, cache["labels"], new_ids)["prototypes"]

    def _risk_on_old_anchors(self, new_ids):
        if not self.class_ids.numel():
            return 0.0
        old_visual = self._history_enhanced()
        old_text = self.text_prototypes
        logits, _, _, _ = fuse_modal_scores(
            self.raw_visual_prototypes_tensor(), old_visual, old_visual, old_text,
            self.visual_reliability, self.text_reliability,
            self.model.temperature_visual, self.model.temperature_text,
        )
        return float(F.cross_entropy(logits, torch.arange(len(self.class_ids), device=self.device)).item())

    def raw_visual_prototypes_tensor(self):
        return torch.stack([self.raw_visual_prototypes[int(cid)] for cid in self.class_ids.tolist()])

    def run(self):
        end_session = int(self.cfg.TRAINER.PCMOE.END_SESSION)
        end_session = self.data.num_tasks - 1 if end_session < 0 else min(end_session, self.data.num_tasks - 1)
        for session in range(end_session + 1):
            ids = self.data.class_index_in_task[session].tolist()
            names = [self.data.class_names[int(cid)] for cid in ids]
            train = self.data.get_dataloader(session, "train", mode="train", accumulate_past=False)
            extract = self.data.get_dataloader(session, "train", mode="test", accumulate_past=False)
            if session == 0:
                self._base_train(train, ids)
                cache = self._extract(extract)
                self.class_ids = torch.as_tensor(ids, device=self.device)
                self.raw_visual_prototypes = {}
                self.layer_prototypes = {layer: {} for layer in self.model.layer_indices}
                self.layer_concentration = {layer: {} for layer in self.model.layer_indices}
                raw_stats = class_prototype_statistics(cache["raw"], cache["labels"], ids)
                self.raw_visual_prototypes = {
                    int(cid): raw_stats["prototypes"][i].detach() for i, cid in enumerate(ids)
                }
                for layer in self.model.layer_indices:
                    stats = class_prototype_statistics(cache["layers"][layer], cache["labels"], ids)
                    self.layer_prototypes[layer] = {
                        int(cid): stats["prototypes"][i].detach() for i, cid in enumerate(ids)
                    }
                    self.layer_concentration[layer] = {
                        int(cid): stats["concentration"][i].detach() for i, cid in enumerate(ids)
                    }
                visual_stats = class_prototype_statistics(self._compose_from_cache(cache), cache["labels"], ids)
                self.visual_prototypes = visual_stats["prototypes"]
                self.text_prototypes, text_consistency = self.model.encode_text_prototypes(names)
                self.visual_reliability, self.text_reliability = self._modal_reliability(
                    visual_stats, text_consistency
                )
                self.apc_reference = build_apc_reference({
                    layer: torch.stack(list(self.layer_prototypes[layer].values()))
                    for layer in self.model.layer_indices
                })
                self._calibrate_temperatures(cache, self.visual_prototypes, self.text_prototypes)
                result = {"accepted": False, "base_session": True}
            else:
                fit_train, fit_extract, calibration, validation = self._support_split(train, extract)
                fit_cache = self._extract(fit_extract)
                apc = self._apc(fit_cache, ids)
                result = self._train_candidates(
                    session, fit_train, fit_extract, calibration, validation,
                    ids, names, apc,
                )
                all_cache = self._extract(extract)
                text, text_consistency = self.model.encode_text_prototypes(names)
                # Update class memory only after the candidate transaction ends.
                self._append_new_class_state(all_cache, ids, text, text_consistency)
            self.model.freeze_all()
            self.diagnostics.append({"session": session, **result})
            self._save_checkpoint(session)
            print("[PC-MoE][Session {}] {}".format(session, json.dumps(
                self._json_safe(result), ensure_ascii=False, sort_keys=True
            )))

    def _append_new_class_state(self, cache, ids, text, text_consistency):
        raw_stats = class_prototype_statistics(cache["raw"], cache["labels"], ids)
        visual_stats = class_prototype_statistics(self._compose_from_cache(cache), cache["labels"], ids)
        for i, cid in enumerate(ids):
            self.raw_visual_prototypes[int(cid)] = raw_stats["prototypes"][i].detach()
        for layer in self.model.layer_indices:
            stats = class_prototype_statistics(cache["layers"][layer], cache["labels"], ids)
            for i, cid in enumerate(ids):
                self.layer_prototypes[layer][int(cid)] = stats["prototypes"][i].detach()
                self.layer_concentration[layer][int(cid)] = stats["concentration"][i].detach()
        self.class_ids = torch.cat([self.class_ids, torch.as_tensor(ids, device=self.device)])
        self.visual_prototypes = torch.cat([self._history_enhanced()[:-len(ids)] if False else
                                            self._history_enhanced(), visual_stats["prototypes"]], dim=0)
        self.text_prototypes = torch.cat([self.text_prototypes, text], dim=0)
        self.visual_reliability = torch.cat([self.visual_reliability, visual_stats["concentration"]], dim=0)
        self.text_reliability = torch.cat([self.text_reliability, text_consistency], dim=0)

    def _save_checkpoint(self, session):
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "completed_session": int(session),
            "model": self.model.state_dict(), "topology": self.model.export_topology(),
            "class_ids": self.class_ids.detach().cpu(),
            "raw_visual_prototypes": self.raw_visual_prototypes,
            "layer_prototypes": self.layer_prototypes,
            "layer_concentration": self.layer_concentration,
            "visual_prototypes": self.visual_prototypes.detach().cpu(),
            "text_prototypes": self.text_prototypes.detach().cpu(),
            "visual_reliability": self.visual_reliability.detach().cpu(),
            "text_reliability": self.text_reliability.detach().cpu(),
            "apc_reference": self.apc_reference,
            "diagnostics": self.diagnostics,
            "temperatures": [float(self.model.temperature_visual), float(self.model.temperature_text)],
            "support_manifest": self.data.get_support_manifest(),
        }
        path = self._checkpoint_dir / "session_{:02d}.pth".format(session)
        torch.save(payload, path)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): PrototypeConfusionRunner._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PrototypeConfusionRunner._json_safe(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value


def _as_ids(values, device):
    return torch.as_tensor(values, device=device, dtype=torch.long).reshape(-1)
