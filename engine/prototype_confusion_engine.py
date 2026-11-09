"""Training, expansion transactions and all-class evaluation for PC-MoE FSCIL."""
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.data_manager import DatasetManager
from utils.evaluator import AccuracyEvaluator
from models.prototype_confusion_moe import (
    PrototypeConfusionMoE, balanced_risk, class_means, make_apc_reference,
    normalize, prototype_confusion, source_exclusive_means,
)


class PrototypeConfusionRunner:
    """Feature-memory FSCIL: no old images, no task oracle, one global classifier."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = DatasetManager(cfg)
        self.device = cfg.DEVICE.DEVICE_NAME
        self.model = PrototypeConfusionMoE(cfg, self.data.template, self.device).to(self.device)
        self.class_memory = {}
        self.class_ids = []
        self.apc_reference = None
        self.base_layer_prototypes = {}
        self.text_prototypes = {}
        self.description_prototypes = {}
        self.text_consistency = {}
        self.visual_temperature = float(cfg.TRAINER.PCMOE.CALIBRATION.INITIAL_TEMPERATURE_VISUAL)
        self.text_temperature = float(cfg.TRAINER.PCMOE.CALIBRATION.INITIAL_TEMPERATURE_TEXT)
        self.inter_modal_beta = float(cfg.TRAINER.PCMOE.CALIBRATION.INTER_MODAL_BETA)
        self.diagnostics = []
        self.base_accuracy = None
        self.start_session = 0
        self._load_resume()

    def _extract(self, loader):
        raw, labels, sources = [], [], []
        layers = {i: [] for i in self.model.layer_ids}
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                r, _, hidden = self.model.encode(images)
                raw.append(r.cpu())
                for layer in self.model.layer_ids:
                    layers[layer].append(hidden[layer].cpu())
                labels.append(batch["label"].long().cpu())
                sources.append(torch.as_tensor(batch.get("source_id", batch.get("idx"))).reshape(-1).cpu())
        if not raw:
            raise ValueError("Cannot extract an empty FSCIL support set.")
        return {
            "raw": torch.cat(raw), "labels": torch.cat(labels).long(),
            "sources": torch.cat(sources),
            "layers": {i: torch.cat(values) for i, values in layers.items()},
        }

    @staticmethod
    def _slice(cache, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return {
            "raw": cache["raw"][idx], "labels": cache["labels"][idx],
            "sources": cache["sources"][idx],
            "layers": {k: v[idx] for k, v in cache["layers"].items()},
        }

    def _split(self, cache, session):
        fit, calibration, risk = [], [], []
        for cid in torch.unique(cache["labels"], sorted=True).tolist():
            indices = torch.where(cache["labels"].eq(cid))[0].tolist()
            indices = sorted(indices, key=lambda i: int(cache["sources"][i]))
            rng = np.random.default_rng(int(self.cfg.SEED) * 100003 + int(session) * 997 + int(cid))
            rng.shuffle(indices)
            if len(indices) < 3:
                raise ValueError(
                    "PC-MoE risk-controlled expansion requires >=3 distinct support sources "
                    "per class (fit/calibration/risk)."
                )
            n_cal = max(1, int(round(0.2 * len(indices))))
            n_risk = max(1, int(round(0.2 * len(indices))))
            while len(indices) - n_cal - n_risk < 1:
                if n_cal > 1:
                    n_cal -= 1
                elif n_risk > 1:
                    n_risk -= 1
                else:
                    raise ValueError("Support split cannot retain a fitting sample.")
            risk.extend(indices[:n_risk])
            calibration.extend(indices[n_risk:n_risk + n_cal])
            fit.extend(indices[n_risk + n_cal:])
        return (self._slice(cache, fit), self._slice(cache, calibration),
                self._slice(cache, risk))

    def _memory_cache(self, field, class_ids=None):
        ids = sorted(self.class_ids if class_ids is None else class_ids)
        raw, labels, sources = [], [], []
        layers = {i: [] for i in self.model.layer_ids}
        for cid in ids:
            part = self.class_memory[cid][field]
            if part is None or part["raw"].numel() == 0:
                continue
            raw.append(part["raw"])
            labels.append(torch.full((part["raw"].size(0),), cid, dtype=torch.long))
            sources.append(part["sources"])
            for layer in self.model.layer_ids:
                layers[layer].append(part["layers"][layer])
        if not raw:
            return None
        return {
            "raw": torch.cat(raw).to(self.device),
            "labels": torch.cat(labels).to(self.device),
            "sources": torch.cat(sources).to(self.device),
            "layers": {i: torch.cat(v).to(self.device) for i, v in layers.items()},
        }

    def _store_classes(self, split_cache, field):
        for cid in torch.unique(split_cache["labels"], sorted=True).tolist():
            idx = torch.where(split_cache["labels"].eq(cid))[0]
            self.class_memory.setdefault(int(cid), {})[field] = {
                "raw": split_cache["raw"][idx].clone(),
                "sources": split_cache["sources"][idx].clone(),
                "layers": {i: split_cache["layers"][i][idx].clone()
                           for i in self.model.layer_ids},
            }

    def _all_text(self, ids=None):
        ids = sorted(self.class_ids if ids is None else ids)
        return torch.stack([self.text_prototypes[c] for c in ids]).to(self.device)

    def _all_descriptions(self, ids=None):
        ids = sorted(self.class_ids if ids is None else ids)
        return torch.stack([self.description_prototypes[c] for c in ids]).to(self.device)

    def _text_for_new(self, names, ids):
        text, consistency = self.model.text_features(names)
        # CuPL/LLM descriptions are optional source data, never fabricated.
        path = str(getattr(self.cfg.DATASET, "GPT_PATH", "") or "")
        descriptions = []
        if path and Path(path).is_file():
            with open(path, "r", encoding="utf-8") as handle:
                prompt_bank = {k.lower().replace("_", " "): v for k, v in json.load(handle).items()}
            import models.clip.clip as clip
            for name in names:
                prompts = prompt_bank.get(str(name).lower().replace("_", " "), [])
                if prompts:
                    tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                    values = normalize(self.model.clip_model.encode_text(tokens))
                    descriptions.append(normalize(values.mean(0)))
                else:
                    descriptions.append(text[len(descriptions)])
        else:
            descriptions = [text[i] for i in range(len(ids))]
        for i, cid in enumerate(ids):
            self.text_prototypes[int(cid)] = text[i].detach().cpu()
            self.description_prototypes[int(cid)] = descriptions[i].detach().cpu()
            self.text_consistency[int(cid)] = consistency[i].detach().cpu()

    def _raw_visual_and_layer_means(self, cache, class_ids):
        raw_proto, layer_proto = {}, {i: {} for i in self.model.layer_ids}
        for cid in class_ids:
            mask = cache["labels"].eq(int(cid))
            if not mask.any():
                raise ValueError("Missing fit support for class {}".format(cid))
            raw_proto[int(cid)] = normalize(cache["raw"][mask].mean(0))
            for layer in self.model.layer_ids:
                layer_proto[layer][int(cid)] = normalize(cache["layers"][layer][mask].mean(0))
        return raw_proto, layer_proto

    def _calibrate_visual_bank(self, prototypes):
        """Apply visual intra-modal calibration to [C,D] or [B,C,D] banks."""
        base_count = int(self.cfg.DATASET.NUM_INIT_CLS)
        if prototypes.size(-2) <= base_count:
            return normalize(prototypes)
        lam = float(self.cfg.TRAINER.PCMOE.CALIBRATION.LAMBDA_VISUAL)
        tau = float(self.cfg.TRAINER.PCMOE.CALIBRATION.VISUAL_TAU)
        base = normalize(prototypes[..., :base_count, :])
        novel = normalize(prototypes[..., base_count:, :])
        weights = torch.softmax(tau * torch.matmul(novel, base.transpose(-1, -2)), dim=-1)
        transported = normalize(torch.matmul(weights, base))
        calibrated = normalize((1.0 - lam) * novel + lam * transported)
        return torch.cat([normalize(prototypes[..., :base_count, :]), calibrated], dim=-2)

    def _prototypes(self, disabled=None):
        fit = self._memory_cache("fit")
        ids = sorted(self.class_ids)
        if fit is None:
            raise RuntimeError("Classifier prototype bank has no fitting features.")
        enhanced = self.model.compose(fit["raw"], fit["layers"], disabled_by_layer=disabled)
        visual, counts, concentration = [], [], []
        for cid in ids:
            values = enhanced[fit["labels"].eq(cid)]
            mean = values.mean(0)
            visual.append(normalize(mean))
            counts.append(values.size(0))
            concentration.append(mean.norm().clamp(0.0, 1.0))
        visual = torch.stack(visual)
        counts = torch.tensor(counts, device=self.device, dtype=torch.float32)
        concentration = torch.stack(concentration)
        # Intra-modal visual calibration: new visual means inherit the
        # discriminative geometry of the fixed base-class visual prototypes.
        visual = self._calibrate_visual_bank(visual)
        text = self._all_text(ids)
        descriptions = self._all_descriptions(ids)
        # Intra-modal text calibration (BiMC's description-to-class semantic
        # correction); if no description bank is supplied, the class text
        # prototype is retained and this term is an identity operation.
        lam_t = float(self.cfg.TRAINER.PCMOE.CALIBRATION.LAMBDA_TEXT)
        text = normalize((1.0 - lam_t) * text + lam_t * descriptions)
        kappa = float(self.cfg.TRAINER.PCMOE.CALIBRATION.RELIABILITY_KAPPA)
        visual_reliability = (counts / (counts + kappa)) * concentration
        return {
            "ids": torch.tensor(ids, device=self.device),
            "visual": visual, "text": text,
            "visual_reliability": visual_reliability,
            "text_reliability": torch.stack([
                self.text_consistency[cid].to(self.device) for cid in ids
            ]).clamp(1e-3, 1.0),
        }

    def _score(self, raw, enhanced, bank, visual_temp=None, text_temp=None):
        tv = max(1e-4, float(self.visual_temperature if visual_temp is None else visual_temp))
        tt = max(1e-4, float(self.text_temperature if text_temp is None else text_temp))
        visual_logits = normalize(enhanced) @ bank["visual"].t() / tv
        text_logits = normalize(raw) @ bank["text"].t() / tt
        rv = bank["visual_reliability"]
        rt = bank["text_reliability"]
        alpha = rv / (rv + rt).clamp_min(1e-6)
        # Inter-modal calibration is applied after both intra-modal
        # calibrations; these are the sole logits used for all-class decisions.
        beta = float(self.inter_modal_beta)
        fused = (1.0 - beta) * (alpha.unsqueeze(0) * visual_logits
                                + (1.0 - alpha.unsqueeze(0)) * text_logits) + beta * (
                                    0.5 * visual_logits + 0.5 * text_logits)
        return fused, visual_logits, text_logits

    def _calibration_features(self):
        return self._memory_cache("calibration")

    def _fit_temperatures(self, bank, disabled=None):
        cache = self._calibration_features()
        if cache is None:
            return self.visual_temperature, self.text_temperature
        with torch.no_grad():
            enhanced = self.model.compose(cache["raw"], cache["layers"], disabled_by_layer=disabled)
            visual = normalize(enhanced) @ bank["visual"].t()
            text = normalize(cache["raw"]) @ bank["text"].t()
            labels = cache["labels"]
            ids = bank["ids"]
            target_match = labels[:, None].eq(ids[None, :])
            valid = target_match.sum(1).eq(1)
            if not valid.any():
                return self.visual_temperature, self.text_temperature
            targets = target_match[valid].long().argmax(1)
            visual, text = visual[valid], text[valid]
        values = []
        for scores in (visual.detach(), text.detach()):
            log_temp = torch.tensor(
                math.log(max(1e-4, self.visual_temperature if len(values) == 0 else self.text_temperature)),
                device=self.device, requires_grad=True
            )
            opt = torch.optim.LBFGS([log_temp], lr=0.25, max_iter=40,
                                    line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                temp = log_temp.exp().clamp(
                    float(self.cfg.TRAINER.PCMOE.CALIBRATION.MIN_TEMPERATURE),
                    float(self.cfg.TRAINER.PCMOE.CALIBRATION.MAX_TEMPERATURE)
                )
                loss = F.cross_entropy(scores / temp, targets)
                loss.backward()
                return loss
            opt.step(closure)
            values.append(float(log_temp.detach().exp().clamp(
                self.cfg.TRAINER.PCMOE.CALIBRATION.MIN_TEMPERATURE,
                self.cfg.TRAINER.PCMOE.CALIBRATION.MAX_TEMPERATURE).item()))
        return values[0], values[1]

    def _query_cache(self, field):
        return self._memory_cache(field)

    def _risk(self, bank, field, disabled=None, temperatures=None):
        cache = self._query_cache(field)
        if cache is None:
            raise RuntimeError("Missing {} risk query features.".format(field))
        with torch.no_grad():
            enhanced = self.model.compose(cache["raw"], cache["layers"], disabled_by_layer=disabled)
            vt, tt = (self.visual_temperature, self.text_temperature) if temperatures is None else temperatures
            logits = self._score(cache["raw"], enhanced, bank, vt, tt)[0]
            return balanced_risk(logits, cache["labels"], bank["ids"])

    def _fit_bank_and_temperatures(self, disabled=None):
        bank = self._prototypes(disabled=disabled)
        temperatures = self._fit_temperatures(bank, disabled=disabled)
        return bank, temperatures

    def _train_base(self):
        cfg = self.cfg.TRAINER.PCMOE
        self.model.set_base_trainable()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg.OPTIM.BASE_LR,
                                      weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        ids = sorted(self.class_ids)
        fit = self._memory_cache("fit")
        text = self._all_text(ids)
        for epoch in range(int(cfg.OPTIM.BASE_EPOCHS)):
            self.model.train()
            enhanced, aux = self.model.compose(fit["raw"], fit["layers"], return_aux=True)
            # Class reliability is a calibration statistic, not a trainable
            # shortcut.  Detach it from the expert optimization graph.
            with torch.no_grad():
                counts = torch.stack([fit["labels"].eq(cid).sum() for cid in ids]).float()
                concentration = torch.stack([
                    enhanced.detach()[fit["labels"].eq(cid)].mean(0).norm().clamp(0.0, 1.0)
                    for cid in ids
                ])
                kappa = float(cfg.CALIBRATION.RELIABILITY_KAPPA)
                rv = counts / (counts + kappa) * concentration
                rt = torch.stack([self.text_consistency[cid].to(self.device)
                                  for cid in ids]).clamp(1e-3, 1.0)
                alpha = (rv / (rv + rt).clamp_min(1e-6)).detach()
                calibrated_text = normalize(
                    (1.0 - cfg.CALIBRATION.LAMBDA_TEXT) * text
                    + cfg.CALIBRATION.LAMBDA_TEXT * self._all_descriptions(ids)
                )
                target = torch.as_tensor(ids, device=self.device)[None, :].eq(
                    fit["labels"][:, None]).long().argmax(1)
            # Construct only a minibatch of source-exclusive prototype banks
            # at a time; the full [N,C,D] tensor is prohibitively large on CUB.
            batch_size = int(cfg.OPTIM.BASE_BATCH_SIZE)
            loss_sum = enhanced.sum() * 0.0
            for begin in range(0, enhanced.size(0), batch_size):
                end = min(begin + batch_size, enhanced.size(0))
                rows = torch.arange(begin, end, device=self.device)
                loo = source_exclusive_means(
                    enhanced, fit["labels"], fit["sources"], ids, query_indices=rows
                )
                loo = self._calibrate_visual_bank(loo)
                visual = torch.bmm(enhanced[begin:end].unsqueeze(1), loo.transpose(1, 2)).squeeze(1)
                textual = fit["raw"][begin:end] @ calibrated_text.t()
                alpha_rows = alpha.unsqueeze(0)
                logits = ((1.0 - self.inter_modal_beta) *
                          (alpha_rows * visual / self.visual_temperature
                           + (1.0 - alpha_rows) * textual / self.text_temperature)
                          + self.inter_modal_beta * 0.5 *
                          (visual / self.visual_temperature + textual / self.text_temperature))
                loss_sum = loss_sum + F.cross_entropy(logits, target[begin:end], reduction="sum")
            loss = loss_sum / enhanced.size(0)
            loss = F.cross_entropy(logits, target)
            balance = []
            for item in aux.values():
                importance = item["route_weights"].mean(0)
                balance.append(importance.pow(2).sum() * item["route_weights"].size(1))
            if balance:
                loss = loss + float(cfg.LOSS.ROUTE_BALANCE_WEIGHT) * torch.stack(balance).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            print("[PC-MoE][Base] epoch={}/{} loss={:.5f}".format(
                epoch + 1, cfg.OPTIM.BASE_EPOCHS, float(loss.item())))
        self.model.freeze_all()

    def _history_layer_means(self):
        fit = self._memory_cache("fit")
        result = {}
        for layer in self.model.layer_ids:
            result[layer] = {}
            for cid in sorted(self.class_ids):
                result[layer][cid] = normalize(fit["layers"][layer][fit["labels"].eq(cid)].mean(0))
        return result

    def _candidate_layers(self, apc):
        cfg = self.cfg.TRAINER.PCMOE.EXPANSION
        ordered = sorted(apc, key=lambda layer: apc[layer]["mean"], reverse=True)
        selected = [layer for layer in ordered
                    if apc[layer]["mean"] >= float(cfg.APC_MIN_PERCENTILE)]
        return selected[:int(cfg.MAX_LAYERS_PER_SESSION)]

    def _train_incremental(self, session, fit_new, new_ids, apc):
        cfg = self.cfg.TRAINER.PCMOE
        candidate_layers = self._candidate_layers(apc)
        new_ids = [int(x) for x in new_ids]
        if not candidate_layers:
            # No capacity request: still acquire the new classes with the
            # unchanged expert pool and calibrate the all-seen classifier.
            baseline_bank, baseline_t = self._fit_bank_and_temperatures()
            baseline_risk, per_class = self._risk(
                baseline_bank, "risk", None, baseline_t
            )
            self.visual_temperature, self.text_temperature = baseline_t
            self.model.set_temperatures(*baseline_t)
            return {
                "accepted": False, "reason": "APC found no layer above capacity threshold",
                "apc": apc, "candidate_layers": [],
                "baseline_all_class_risk": float(baseline_risk.item()),
                "baseline_per_class_risk": per_class,
            }
        old_ids = [int(x) for x in self.class_ids if int(x) not in new_ids]
        old_fit = self._memory_cache("fit", old_ids)
        all_fit = {
            "raw": torch.cat([old_fit["raw"], fit_new["raw"].to(self.device)]),
            "labels": torch.cat([old_fit["labels"], fit_new["labels"].to(self.device)]),
            "sources": torch.cat([old_fit["sources"], fit_new["sources"].to(self.device)]),
            "layers": {i: torch.cat([old_fit["layers"][i], fit_new["layers"][i].to(self.device)])
                       for i in self.model.layer_ids},
        }
        # Snapshot is the transaction boundary.  Any failure or rejected risk
        # restores both topology and exact parameter values.
        model_snapshot = copy.deepcopy(self.model.state_dict())
        topology_snapshot = copy.deepcopy(self.model.topology())
        memory_snapshot = copy.deepcopy(self.class_memory)
        class_ids_snapshot = list(self.class_ids)
        try:
            # New class fit/calibration/risk memories and text prototypes
            # are staged by run() before this transaction.  Class acquisition
            # is independent of whether extra model capacity is accepted.
            for layer in candidate_layers:
                self.model.add_expert(
                    layer, session, fit_new["layers"][layer].to(self.device),
                    old_fit["layers"][layer]
                )
            self.model.set_incremental_trainable(candidate_layers)
            parameters = []
            for layer in candidate_layers:
                parameters.extend(self.model.adapters[str(layer)].newest_parameters())
            optimizer = torch.optim.AdamW(parameters, lr=cfg.OPTIM.INCREMENTAL_LR,
                                          weight_decay=cfg.OPTIM.WEIGHT_DECAY)
            ids_all = sorted(old_ids + [int(x) for x in new_ids])
            text = torch.stack([self.text_prototypes[cid] for cid in ids_all]).to(self.device)
            descriptions = torch.stack([self.description_prototypes[cid] for cid in ids_all]).to(self.device)
            for epoch in range(int(cfg.OPTIM.INCREMENTAL_EPOCHS)):
                self.model.train()
                with torch.no_grad():
                    reliability_bank = self._prototypes()
                    alpha = (reliability_bank["visual_reliability"] /
                             (reliability_bank["visual_reliability"]
                              + reliability_bank["text_reliability"]).clamp_min(1e-6)).detach()
                enhanced_all, aux = self.model.compose(
                    all_fit["raw"], all_fit["layers"], return_aux=True
                )
                new_mask = torch.isin(all_fit["labels"], torch.as_tensor(new_ids, device=self.device))
                rows = torch.where(new_mask)[0]
                loo = source_exclusive_means(
                    enhanced_all, all_fit["labels"], all_fit["sources"], ids_all,
                    query_indices=rows
                )
                loo = self._calibrate_visual_bank(loo)
                visual_logits, text_logits, targets = [], [], []
                calibrated_text = normalize((1.0 - cfg.CALIBRATION.LAMBDA_TEXT) * text
                                             + cfg.CALIBRATION.LAMBDA_TEXT * descriptions)
                for local_row, row in enumerate(rows.tolist()):
                    visual_logits.append(enhanced_all[row:row + 1] @ loo[local_row].t()
                                          / self.visual_temperature)
                    text_logits.append(all_fit["raw"][row:row + 1] @ calibrated_text.t()
                                       / self.text_temperature)
                    targets.append(ids_all.index(int(all_fit["labels"][row].item())))
                visual_logits = torch.cat(visual_logits)
                text_logits = torch.cat(text_logits)
                logits = ((1.0 - self.inter_modal_beta) *
                          (alpha.unsqueeze(0) * visual_logits + (1.0 - alpha.unsqueeze(0)) * text_logits)
                          + self.inter_modal_beta * 0.5 * (visual_logits + text_logits))
                target = torch.tensor(targets, device=self.device, dtype=torch.long)
                loss = F.cross_entropy(logits, target)
                balance = [item["route_weights"].mean(0).pow(2).sum()
                           * item["route_weights"].size(1) for item in aux.values()]
                if balance:
                    loss = loss + float(cfg.LOSS.ROUTE_BALANCE_WEIGHT) * torch.stack(balance).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            self.model.freeze_all()
            # Temporarily expose new class fit/calibration/risk features so the
            # baseline and candidate are evaluated against identical all-class
            # data.  This is not a commit; rollback restores it if rejected.
            # Fit, calibration and risk features are staged by the caller.
            baseline_disabled = {
                layer: (self.model.adapters[str(layer)].num_experts - 1,)
                for layer in candidate_layers
            }
            baseline_bank, baseline_t = self._fit_bank_and_temperatures(baseline_disabled)
            candidate_bank, candidate_t = self._fit_bank_and_temperatures()
            baseline_risk, baseline_per_class = self._risk(
                baseline_bank, "risk", baseline_disabled, baseline_t
            )
            candidate_risk, candidate_per_class = self._risk(
                candidate_bank, "risk", None, candidate_t
            )
            old_per_class = [cid for cid in old_ids if cid in candidate_per_class]
            old_delta = max(
                (candidate_per_class[cid] - baseline_per_class[cid] for cid in old_per_class),
                default=0.0
            )
            risk_gain = float((baseline_risk - candidate_risk).item())
            accepted = (
                risk_gain >= float(cfg.EXPANSION.MIN_RISK_GAIN)
                and old_delta <= float(cfg.EXPANSION.MAX_OLD_CLASS_RISK_INCREASE)
            )
            result = {
                "accepted": accepted, "candidate_layers": candidate_layers,
                "baseline_all_class_risk": float(baseline_risk.item()),
                "candidate_all_class_risk": float(candidate_risk.item()),
                "risk_gain": risk_gain, "worst_old_class_risk_increase": float(old_delta),
                "baseline_per_class_risk": baseline_per_class,
                "candidate_per_class_risk": candidate_per_class,
                "apc": apc,
            }
            if accepted:
                self.visual_temperature, self.text_temperature = candidate_t
                self.model.set_temperatures(*candidate_t)
                return result
            # Reject only the added capacity.  The class-incremental task and
            # its support/text memories remain valid under the old experts.
            self.visual_temperature, self.text_temperature = baseline_t
            self.model.set_temperatures(*baseline_t)
            raise _RejectExpansion(result)
        except _RejectExpansion as rejected:
            self.model.rebuild_topology(topology_snapshot)
            self.model.load_state_dict(model_snapshot, strict=True)
            self.model.freeze_all()
            self.class_memory = memory_snapshot
            self.class_ids = class_ids_snapshot
            return rejected.result
        except Exception:
            self.model.rebuild_topology(topology_snapshot)
            self.model.load_state_dict(model_snapshot, strict=True)
            self.model.freeze_all()
            self.class_memory = memory_snapshot
            self.class_ids = class_ids_snapshot
            raise


    @torch.no_grad()
    def _evaluate_all_seen(self, session):
        """Evaluate the sole all-seen-class classifier; never use task IDs for prediction."""
        loader = self.data.get_dataloader(session, "test", mode="test", accumulate_past=False)
        bank = self._prototypes()
        all_logits, all_labels = [], []
        self.model.eval()
        for batch in loader:
            images = batch["image"].to(self.device)
            raw, enhanced, _ = self.model.encode(images)
            logits = self._score(raw, enhanced, bank)[0]
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].long().cpu())
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        evaluator = AccuracyEvaluator(self.data.class_index_in_task)
        metrics = evaluator.calc_accuracy(
            logits, labels, session, class_ids=bank["ids"].cpu()
        )
        base_ids = torch.as_tensor(
            self.data.class_index_in_task[0], dtype=torch.long
        )
        base_mask = torch.isin(labels, base_ids)
        pred_ids = bank["ids"].cpu()[logits.argmax(dim=1)]
        base_accuracy = float(pred_ids[base_mask].eq(labels[base_mask]).float().mean().item())
        if session == 0:
            self.base_accuracy = base_accuracy
        pd = (0.0 if self.base_accuracy is None else
              float(self.base_accuracy - base_accuracy))
        metrics.update({
            "AA": float(np.mean(metrics["task_acc"])) if metrics["task_acc"] else 0.0,
            "PD": pd,
            "base_class_accuracy": base_accuracy,
            "num_seen_classes": int(bank["ids"].numel()),
        })
        return metrics

    def run(self):
        end = int(self.cfg.TRAINER.PCMOE.END_SESSION)
        end = self.data.num_tasks - 1 if end < 0 else min(end, self.data.num_tasks - 1)
        for session in range(self.start_session, end + 1):
            ids = [int(x) for x in self.data.class_index_in_task[session].tolist()]
            names = [self.data.class_names[i] for i in ids]
            loader = self.data.get_dataloader(session, "train", mode="test", accumulate_past=False)
            cache = self._extract(loader)
            fit, calibration, risk = self._split(cache, session)
            if session == 0:
                self.class_ids = list(ids)
                self._store_classes(fit, "fit")
                self._store_classes(calibration, "calibration")
                self._store_classes(risk, "risk")
                self._text_for_new(names, ids)
                self._train_base()
                # APC reference is frozen from base class pairwise confusion
                # distributions at every configured adapter layer.
                self.base_layer_prototypes = self._history_layer_means()
                self.apc_reference = make_apc_reference({
                    layer: torch.stack([self.base_layer_prototypes[layer][cid]
                                        for cid in ids])
                    for layer in self.model.layer_ids
                })
                bank, temps = self._fit_bank_and_temperatures()
                self.visual_temperature, self.text_temperature = temps
                self.model.set_temperatures(*temps)
                result = {"base_session": True, "accepted": False}
            else:
                fit_before = self._memory_cache("fit")
                apc = prototype_confusion(
                    fit["layers"].to(self.device) if False else fit["layers"],
                    fit["labels"].to(self.device), ids,
                    {layer: torch.stack([self._history_layer_means()[layer][cid]
                                         for cid in sorted(self.class_ids)])
                     for layer in self.model.layer_ids},
                    self.apc_reference,
                )
                # New classes are always acquired.  APC controls only whether
                # additional experts are proposed; rejection must never erase
                # the novel classes from the global classifier.
                self._text_for_new(names, ids)
                self.class_ids.extend(ids)
                self._store_classes(fit, "fit")
                self._store_classes(calibration, "calibration")
                self._store_classes(risk, "risk")
                result = self._train_incremental(session, fit, ids, apc)
                # The class memory transaction commits regardless of capacity
                # acceptance.  A rejected candidate restores only model topology
                # and parameters to the old expert pool.
                self.base_layer_prototypes = self._history_layer_means()
            self.model.freeze_all()
            evaluation = self._evaluate_all_seen(session)
            result["evaluation"] = evaluation
            self.diagnostics.append({"session": session, **result})
            self._save(session)
            print("[PC-MoE][Session {}] {}".format(
                session, json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
            ))

    def _save(self, session):
        directory = Path(self.cfg.TRAINER.PCMOE.CHECKPOINT.DIR)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "completed_session": int(session),
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "topology": self.model.topology(),
            "class_ids": list(self.class_ids), "class_memory": self.class_memory,
            "text_prototypes": self.text_prototypes,
            "description_prototypes": self.description_prototypes,
            "text_consistency": self.text_consistency,
            "apc_reference": self.apc_reference,
            "base_layer_prototypes": self.base_layer_prototypes,
            "visual_temperature": self.visual_temperature,
            "text_temperature": self.text_temperature,
            "inter_modal_beta": self.inter_modal_beta,
            "diagnostics": self.diagnostics,
            "base_accuracy": self.base_accuracy,
            "support_manifest": self.data.get_support_manifest(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        target = directory / "session_{:02d}.pth".format(session)
        tmp = target.with_suffix(".pth.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)

    def _load_resume(self):
        path = str(self.cfg.TRAINER.PCMOE.CHECKPOINT.RESUME or "")
        if not path:
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported PC-MoE checkpoint schema.")
        self.model.rebuild_topology(payload["topology"])
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.freeze_all()
        self.data.load_support_manifest(payload["support_manifest"])
        self.class_ids = list(payload["class_ids"])
        self.class_memory = payload["class_memory"]
        self.text_prototypes = payload["text_prototypes"]
        self.description_prototypes = payload["description_prototypes"]
        self.text_consistency = payload["text_consistency"]
        self.apc_reference = payload["apc_reference"]
        self.base_layer_prototypes = payload["base_layer_prototypes"]
        self.visual_temperature = float(payload["visual_temperature"])
        self.text_temperature = float(payload["text_temperature"])
        self.model.set_temperatures(self.visual_temperature, self.text_temperature)
        self.inter_modal_beta = float(payload["inter_modal_beta"])
        self.diagnostics = list(payload["diagnostics"])
        self.base_accuracy = payload.get("base_accuracy")
        self.start_session = int(payload["completed_session"]) + 1
        rng = payload.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])


class _RejectExpansion(Exception):
    def __init__(self, result):
        self.result = result
