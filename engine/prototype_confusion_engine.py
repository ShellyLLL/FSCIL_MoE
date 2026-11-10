"""Training, expansion transactions and all-class evaluation for PC-MoE FSCIL."""
import copy
import math
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

APC_VERSION = "cardinality_matched_empirical_tail_by_v1"


def finalize_fscil_metrics(metrics, base_accuracy, is_final_session):
    """AA is macro task accuracy; Final Acc and PD are emitted only at final session."""
    result = dict(metrics)
    task_accuracies = result.get("task_acc", [])
    result["AA"] = float(np.mean(task_accuracies)) if task_accuracies else 0.0
    if is_final_session:
        if base_accuracy is None:
            raise RuntimeError("Cannot compute final PD without Session-0 base accuracy.")
        if "mean_acc" not in result or "base_class_accuracy" not in result:
            raise ValueError("Final evaluation is missing all-class or base-class accuracy.")
        result["Final Acc"] = float(result["mean_acc"])
        result["PD"] = float(base_accuracy - result["base_class_accuracy"])
    else:
        result.pop("Final Acc", None)
        result.pop("PD", None)
    return result


class PrototypeConfusionRunner:
    """Prototype-memory FSCIL: no old samples, no task oracle, one global classifier."""
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
        """Split support by source image, never by individual view/sample index."""
        fit, calibration, risk = [], [], []
        for cid in torch.unique(cache["labels"], sorted=True).tolist():
            indices = torch.where(cache["labels"].eq(cid))[0].tolist()
            source_to_indices = {}
            for index in indices:
                source_id = int(cache["sources"][index].item())
                source_to_indices.setdefault(source_id, []).append(index)
            source_ids = sorted(source_to_indices)
            if len(source_ids) < 4:
                raise ValueError(
                    "PC-MoE requires >=4 distinct source images per class "
                    "(>=2 fit sources, 1 calibration source, 1 risk source); "
                    "class {} has {}.".format(int(cid), len(source_ids))
                )
            rng = np.random.default_rng(
                int(self.cfg.SEED) * 100003 + int(session) * 997 + int(cid)
            )
            rng.shuffle(source_ids)
            n_cal = max(1, int(round(0.2 * len(source_ids))))
            n_risk = max(1, int(round(0.2 * len(source_ids))))
            while len(source_ids) - n_cal - n_risk < 2:
                if n_cal > 1:
                    n_cal -= 1
                elif n_risk > 1:
                    n_risk -= 1
                else:
                    raise ValueError("Support split cannot retain two fitting source images.")
            risk_sources = source_ids[:n_risk]
            calibration_sources = source_ids[n_risk:n_risk + n_cal]
            fit_sources = source_ids[n_risk + n_cal:]
            if len(fit_sources) < 2 or not calibration_sources or not risk_sources:
                raise RuntimeError("Source-disjoint support partition invariant was violated.")
            if (set(fit_sources) & set(calibration_sources)
                    or set(fit_sources) & set(risk_sources)
                    or set(calibration_sources) & set(risk_sources)):
                raise RuntimeError("A source image was assigned to multiple support partitions.")
            for source_id in fit_sources:
                fit.extend(source_to_indices[source_id])
            for source_id in calibration_sources:
                calibration.extend(source_to_indices[source_id])
            for source_id in risk_sources:
                risk.extend(source_to_indices[source_id])
        return (self._slice(cache, fit), self._slice(cache, calibration),
                self._slice(cache, risk))

    def _memory_cache(self, field, class_ids=None):
        """Assemble current-session samples plus one stored centroid per old class.

        Historical sample-level features are never persisted.  Prototype records
        contribute exactly one row and a synthetic negative source id, while
        current-session records retain their true source ids for LOO exclusion.
        """
        ids = sorted(self.class_ids if class_ids is None else class_ids)
        raw, labels, sources = [], [], []
        layers = {i: [] for i in self.model.layer_ids}
        class_counts = {}
        prototype_mask = []
        for position, cid in enumerate(ids):
            part = self.class_memory[cid].get(field)
            if part is None:
                continue
            kind = part.get("kind")
            values = part["raw"]
            if kind == "prototype":
                if values.ndim != 2 or values.size(0) != 1:
                    raise ValueError("Historical prototype memory must contain exactly one row per class.")
                source_values = torch.tensor([-(position + 1)], dtype=torch.long)
                class_counts[int(cid)] = int(part["count"])
                prototype_mask.append(torch.ones(1, dtype=torch.bool))
            elif kind == "samples":
                if values.ndim != 2 or values.size(0) == 0:
                    raise ValueError("Current-session sample memory must be a non-empty [N,D] tensor.")
                source_values = torch.as_tensor(part["sources"], dtype=torch.long).reshape(-1)
                if source_values.numel() != values.size(0) or source_values.lt(0).any():
                    raise ValueError("Current-session source ids must be non-negative and align with features.")
                class_counts[int(cid)] = int(values.size(0))
                prototype_mask.append(torch.zeros(values.size(0), dtype=torch.bool))
            else:
                raise ValueError(
                    "Unknown class-memory record type for class {} field {}.".format(cid, field)
                )
            raw.append(values)
            labels.append(torch.full((values.size(0),), int(cid), dtype=torch.long))
            sources.append(source_values)
            for layer in self.model.layer_ids:
                layer_values = part["layers"][layer]
                if layer_values.ndim != 2 or layer_values.size(0) != values.size(0):
                    raise ValueError("Layer memory shape does not match raw feature memory.")
                layers[layer].append(layer_values)
        if not raw:
            return None
        return {
            "raw": torch.cat(raw).to(self.device),
            "labels": torch.cat(labels).to(self.device),
            "sources": torch.cat(sources).to(self.device),
            "prototype_mask": torch.cat(prototype_mask).to(self.device),
            "class_counts": class_counts,
            "layers": {i: torch.cat(v).to(self.device) for i, v in layers.items()},
        }

    def _store_classes(self, split_cache, field):
        """Stage only the current session's support features until its transaction ends."""
        for cid in torch.unique(split_cache["labels"], sorted=True).tolist():
            idx = torch.where(split_cache["labels"].eq(cid))[0]
            self.class_memory.setdefault(int(cid), {})[field] = {
                "kind": "samples",
                "raw": split_cache["raw"][idx].clone(),
                "sources": split_cache["sources"][idx].clone(),
                "layers": {i: split_cache["layers"][i][idx].clone()
                           for i in self.model.layer_ids},
            }

    def _compress_class_memory(self):
        """Replace every completed class/session sample set with one centroid.

        This is the no-replay boundary: after compression, no per-example raw
        feature, hidden feature, source id, or route record survives in memory.
        """
        for cid, fields in self.class_memory.items():
            for field, part in list(fields.items()):
                if part is None or part.get("kind") == "prototype":
                    continue
                if part.get("kind") != "samples":
                    raise ValueError("Cannot compress unknown memory type for class {}.".format(cid))
                count = int(part["raw"].size(0))
                if count < 1:
                    raise ValueError("Cannot create an empty class prototype.")
                fields[field] = {
                    "kind": "prototype",
                    "raw": normalize(part["raw"].mean(0, keepdim=True)).cpu(),
                    "layers": {
                        layer: values.mean(0, keepdim=True).cpu()
                        for layer, values in part["layers"].items()
                    },
                    "count": count,
                }

    def _assert_prototype_only_memory(self):
        """Fail closed if a checkpoint would retain historical sample replay."""
        for cid, fields in self.class_memory.items():
            for field, part in fields.items():
                if part is None:
                    continue
                if part.get("kind") != "prototype" or "sources" in part:
                    raise RuntimeError(
                        "Checkpoint blocked: class {} field {} still contains sample-level memory.".format(
                            cid, field
                        )
                    )
                if part["raw"].ndim != 2 or part["raw"].size(0) != 1:
                    raise RuntimeError("Checkpoint blocked: memory must contain one raw centroid per class.")
                if any(value.ndim != 2 or value.size(0) != 1
                       for value in part["layers"].values()):
                    raise RuntimeError("Checkpoint blocked: memory must contain one layer centroid per class.")
                if int(part.get("count", 0)) < 1:
                    raise RuntimeError("Checkpoint blocked: prototype support count is invalid.")

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
            counts.append(fit["class_counts"][cid])
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

    @torch.no_grad()
    def _fit_inter_modal_beta(self, bank, temperatures):
        """Select one base-session inter-modal coefficient on held-out calibration data."""
        cache = self._calibration_features()
        if cache is None:
            return float(self.inter_modal_beta)
        enhanced = self.model.compose(cache["raw"], cache["layers"])
        visual = normalize(enhanced) @ bank["visual"].t() / float(temperatures[0])
        textual = normalize(cache["raw"]) @ bank["text"].t() / float(temperatures[1])
        ids = bank["ids"]
        match = cache["labels"][:, None].eq(ids[None, :])
        if not match.sum(1).eq(1).all():
            raise ValueError("Calibration labels must map exactly once into the all-class bank.")
        targets = match.long().argmax(1)
        rv, rt = bank["visual_reliability"], bank["text_reliability"]
        alpha = rv / (rv + rt).clamp_min(1e-6)
        best_beta, best_risk = None, float("inf")
        for beta in torch.linspace(0.0, 1.0, 21, device=self.device):
            logits = ((1.0 - beta) * (
                alpha.unsqueeze(0) * visual + (1.0 - alpha.unsqueeze(0)) * textual
            ) + beta * 0.5 * (visual + textual))
            risk, _ = balanced_risk(logits, cache["labels"], ids)
            value = float(risk.item())
            if value < best_risk:
                best_beta, best_risk = float(beta.item()), value
        return best_beta

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
        batch_size = int(cfg.OPTIM.BASE_BATCH_SIZE)
        for epoch in range(int(cfg.OPTIM.BASE_EPOCHS)):
            self.model.train()
            # Freeze a detached prototype snapshot for this epoch.  Queries
            # still backpropagate through the live MoE, while prototype banks
            # do not retain an N-by-C-by-D autograd graph.
            with torch.no_grad():
                enhanced_bank = self.model.compose(fit["raw"], fit["layers"])
                counts = torch.stack([fit["labels"].eq(cid).sum() for cid in ids]).float()
                concentration = torch.stack([
                    enhanced_bank[fit["labels"].eq(cid)].mean(0).norm().clamp(0.0, 1.0)
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
            epoch_loss = 0.0
            for begin in range(0, fit["raw"].size(0), batch_size):
                end = min(begin + batch_size, fit["raw"].size(0))
                rows = torch.arange(begin, end, device=self.device)
                batch_layers = {layer: values[begin:end]
                                for layer, values in fit["layers"].items()}
                enhanced, aux = self.model.compose(
                    fit["raw"][begin:end], batch_layers, return_aux=True
                )
                loo = source_exclusive_means(
                    enhanced_bank, fit["labels"], fit["sources"], ids, query_indices=rows
                )
                loo = self._calibrate_visual_bank(loo)
                visual = torch.bmm(enhanced.unsqueeze(1), loo.transpose(1, 2)).squeeze(1)
                textual = fit["raw"][begin:end] @ calibrated_text.t()
                logits = ((1.0 - self.inter_modal_beta) *
                          (alpha.unsqueeze(0) * visual / self.visual_temperature
                           + (1.0 - alpha.unsqueeze(0)) * textual / self.text_temperature)
                          + self.inter_modal_beta * 0.5 *
                          (visual / self.visual_temperature + textual / self.text_temperature))
                loss = F.cross_entropy(logits, target[begin:end])
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
                epoch_loss += float(loss.detach().item()) * (end - begin)
            print("[PC-MoE][Base] epoch={}/{} loss={:.5f}".format(
                epoch + 1, cfg.OPTIM.BASE_EPOCHS, epoch_loss / fit["raw"].size(0)))
        self.model.freeze_all()

    def _history_layer_means(self):
        """Return one normalized layer prototype per seen class."""
        fit = self._memory_cache("fit")
        result = {}
        for layer in self.model.layer_ids:
            result[layer] = {}
            for cid in sorted(self.class_ids):
                values = fit["layers"][layer][fit["labels"].eq(cid)]
                if values.size(0) == 0:
                    raise RuntimeError("Missing layer memory for class {}.".format(cid))
                # At the base session, current fit examples are still available
                # for base training; after session commit, each old class has
                # exactly one stored prototype row.
                result[layer][cid] = normalize(values.mean(0))
        return result

    @staticmethod
    def _apc_by_discoveries(apc, fdr_q):
        """Benjamini-Yekutieli correction for dependent layer/class tests."""
        tests = []
        for layer, stats in apc.items():
            for cid, value in stats.get("p_value_by_class", {}).items():
                p_value = float(value)
                if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                    raise ValueError("APC p-values must be finite and in [0,1].")
                tests.append((p_value, int(layer), int(cid)))
        if not tests:
            return {}
        tests.sort(key=lambda x: x[0])
        m = len(tests)
        harmonic = sum(1.0 / rank for rank in range(1, m + 1))
        cutoff = None
        for rank, (p_value, _, _) in enumerate(tests, start=1):
            if p_value <= float(fdr_q) * rank / (m * harmonic):
                cutoff = p_value
        if cutoff is None:
            return {}
        discoveries = {}
        for p_value, layer, cid in tests:
            if p_value <= cutoff:
                discoveries.setdefault(layer, []).append(cid)
        return {layer: sorted(classes) for layer, classes in discoveries.items()}

    def _candidate_layers(self, apc):
        cfg = self.cfg.TRAINER.PCMOE.EXPANSION
        discoveries = self._apc_by_discoveries(apc, float(cfg.APC_FDR_Q))
        ordered = sorted(discoveries, key=lambda layer: min(
            float(apc[layer]["p_value_by_class"][cid]) for cid in discoveries[layer]))
        return ordered[:int(cfg.MAX_LAYERS_PER_SESSION)]

    def _demand_classes_by_layer(self, apc, candidate_layers, fdr_q):
        discoveries = self._apc_by_discoveries(apc, float(fdr_q))
        demand = {int(layer): discoveries[int(layer)] for layer in candidate_layers
                  if int(layer) in discoveries}
        if len(demand) != len(candidate_layers):
            raise RuntimeError("Candidate layer has no statistically supported APC demand.")
        return demand

    def _train_incremental(self, session, fit_new, new_ids, apc):
        cfg = self.cfg.TRAINER.PCMOE
        candidate_layers = self._candidate_layers(apc)
        new_ids = [int(x) for x in new_ids]
        demand_classes = self._demand_classes_by_layer(
            apc, candidate_layers, cfg.EXPANSION.APC_FDR_Q
        ) if candidate_layers else {}
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
                "apc": apc, "candidate_layers": [], "demand_class_ids": {},
                "baseline_all_class_risk": float(baseline_risk.item()),
                "baseline_per_class_risk": per_class,
                "risk_protocol": "historical class-risk centroids plus current-session held-out risk supports; prototype-risk surrogate, not historical per-example empirical risk",
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
            route_cfg = cfg.LOSS
            for layer in candidate_layers:
                demand_ids = torch.as_tensor(
                    demand_classes[int(layer)], device=self.device, dtype=torch.long
                )
                demand_mask = torch.isin(fit_new["labels"].to(self.device), demand_ids)
                if not demand_mask.any():
                    raise RuntimeError("APC-demanded classes have no fitting support features.")
                self.model.add_expert(
                    layer, session, fit_new["layers"][layer].to(self.device)[demand_mask],
                    old_fit["layers"][layer],
                    route_margin=float(route_cfg.ROUTE_MARGIN)
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
                new_mask = torch.isin(all_fit["labels"], torch.as_tensor(new_ids, device=self.device))
                rows = torch.where(new_mask)[0]
                with torch.no_grad():
                    enhanced_bank = self.model.compose(all_fit["raw"], all_fit["layers"])
                    loo = source_exclusive_means(
                        enhanced_bank, all_fit["labels"], all_fit["sources"], ids_all,
                        query_indices=rows
                    )
                    loo = self._calibrate_visual_bank(loo)
                query_layers = {
                    layer: values[rows] for layer, values in all_fit["layers"].items()
                }
                enhanced_queries, aux = self.model.compose(
                    all_fit["raw"][rows], query_layers, return_aux=True
                )
                calibrated_text = normalize((1.0 - cfg.CALIBRATION.LAMBDA_TEXT) * text
                                             + cfg.CALIBRATION.LAMBDA_TEXT * descriptions)
                visual_logits = torch.bmm(
                    enhanced_queries.unsqueeze(1), loo.transpose(1, 2)
                ).squeeze(1) / self.visual_temperature
                text_logits = all_fit["raw"][rows] @ calibrated_text.t() / self.text_temperature
                targets = [
                    ids_all.index(int(all_fit["labels"][row].item()))
                    for row in rows.tolist()
                ]
                logits = ((1.0 - self.inter_modal_beta) *
                          (alpha.unsqueeze(0) * visual_logits + (1.0 - alpha.unsqueeze(0)) * text_logits)
                          + self.inter_modal_beta * 0.5 * (visual_logits + text_logits))
                target = torch.tensor(targets, device=self.device, dtype=torch.long)
                loss = F.cross_entropy(logits, target)

                # Incremental routing is not forced toward uniform expert use.
                # The new gate must cross the old Top-k boundary on novel
                # support while remaining below that boundary on every stored
                # historical class. Old experts and old gate columns stay frozen.
                route_new, route_old = [], []
                for layer in candidate_layers:
                    adapter = self.model.adapters[str(layer)]
                    demand_ids = torch.as_tensor(
                        demand_classes[int(layer)], device=self.device, dtype=torch.long
                    )
                    demand_mask = torch.isin(all_fit["labels"][rows], demand_ids)
                    protected_new_mask = ~demand_mask
                    protected_features = [old_fit["layers"][layer]]
                    if protected_new_mask.any():
                        protected_features.append(query_layers[layer][protected_new_mask])
                    terms = adapter.route_margin_loss(
                        query_layers[layer][demand_mask],
                        torch.cat(protected_features, dim=0),
                        margin=float(route_cfg.ROUTE_MARGIN)
                    )
                    route_new.append(terms["new"])
                    route_old.append(terms["historical"])
                if route_new:
                    loss = loss + float(route_cfg.NEW_ROUTE_WEIGHT) * torch.stack(route_new).mean()
                    loss = loss + float(route_cfg.HISTORICAL_ROUTE_WEIGHT) * torch.stack(route_old).mean()

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
            new_per_class = [cid for cid in new_ids if cid in candidate_per_class]
            old_delta = max(
                (candidate_per_class[cid] - baseline_per_class[cid] for cid in old_per_class),
                default=0.0
            )
            new_delta = max(
                (candidate_per_class[cid] - baseline_per_class[cid] for cid in new_per_class),
                default=0.0
            )
            risk_gain = float((baseline_risk - candidate_risk).item())
            demand_per_class = sorted({
                int(cid) for classes in demand_classes.values() for cid in classes
            })
            missing_demand = [cid for cid in demand_per_class
                              if cid not in baseline_per_class or cid not in candidate_per_class]
            if missing_demand:
                raise RuntimeError("Demand classes missing from paired risk evaluation: {}".format(
                    missing_demand))
            demand_risk_gain = float(np.mean([
                baseline_per_class[cid] - candidate_per_class[cid]
                for cid in demand_per_class
            ]))

            # Evaluate the *inference* router on source-disjoint risk examples.
            # The route-margin loss itself is never used as an acceptance proxy.
            route_checks = {}
            for layer in candidate_layers:
                adapter = self.model.adapters[str(layer)]
                old_queries = self._query_cache("risk")
                old_queries = {
                    key: value for key, value in old_queries.items()
                }
                old_mask = torch.isin(
                    old_queries["labels"],
                    torch.as_tensor(old_ids, device=self.device, dtype=torch.long)
                )
                new_mask = torch.isin(
                    old_queries["labels"],
                    torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
                )
                old_route = adapter.route_drift(old_queries["layers"][layer][old_mask])
                new_route = adapter.route_drift(old_queries["layers"][layer][new_mask])
                novel_labels = old_queries["labels"][new_mask]
                per_new_class_activation = {}
                demand_activation, non_demand_activation = {}, {}
                demanded = set(demand_classes[int(layer)])
                for cid in new_ids:
                    class_mask = novel_labels.eq(int(cid))
                    if not class_mask.any():
                        raise RuntimeError(
                            "Expansion risk set is missing novel class {}.".format(cid)
                        )
                    rate = float(
                        new_route["new_expert_selected"][class_mask].float().mean().item()
                    )
                    per_new_class_activation[int(cid)] = rate
                    if int(cid) in demanded:
                        demand_activation[int(cid)] = rate
                    else:
                        non_demand_activation[int(cid)] = rate
                route_checks[int(layer)] = {
                    "demand_class_ids": sorted(demanded),
                    "historical_mean_drift": float(old_route["drift"].mean().item()),
                    "historical_max_drift": float(old_route["drift"].max().item()),
                    "historical_intrusion_rate": float(
                        old_route["new_expert_selected"].float().mean().item()
                    ),
                    "novel_activation_rate": float(
                        new_route["new_expert_selected"].float().mean().item()
                    ),
                    "novel_activation_by_class": per_new_class_activation,
                    "demand_activation_by_class": demand_activation,
                    "non_demand_activation_by_class": non_demand_activation,
                    "novel_mean_weight": float(new_route["new_expert_weight"].mean().item()),
                }
            route_ok = all(
                item["historical_max_drift"] <= float(route_cfg.MAX_HISTORICAL_ROUTE_DRIFT)
                and item["historical_intrusion_rate"] <= float(route_cfg.MAX_HISTORICAL_INTRUSION_RATE)
                and all(
                    rate >= float(route_cfg.MIN_DEMAND_ACTIVATION_RATE)
                    for rate in item["demand_activation_by_class"].values()
                )
                and all(
                    rate <= float(route_cfg.MAX_NONDEMAND_ACTIVATION_RATE)
                    for rate in item["non_demand_activation_by_class"].values()
                )
                for item in route_checks.values()
            )
            accepted = (
                risk_gain >= float(cfg.EXPANSION.MIN_RISK_GAIN)
                and demand_risk_gain >= float(cfg.EXPANSION.MIN_DEMAND_RISK_GAIN)
                and old_delta <= float(cfg.EXPANSION.MAX_OLD_CLASS_RISK_INCREASE)
                and new_delta <= float(cfg.EXPANSION.MAX_NEW_CLASS_RISK_INCREASE)
                and route_ok
            )
            result = {
                "accepted": accepted, "candidate_layers": candidate_layers,
                "demand_class_ids": demand_classes,
                "baseline_all_class_risk": float(baseline_risk.item()),
                "candidate_all_class_risk": float(candidate_risk.item()),
                "risk_gain": risk_gain, "demand_class_risk_gain": demand_risk_gain,
                "demand_class_ids": demand_classes,
                "worst_old_class_risk_increase": float(old_delta),
                "worst_new_class_risk_increase": float(new_delta),
                "route_checks": route_checks, "route_constraints_passed": route_ok,
                "baseline_per_class_risk": baseline_per_class,
                "candidate_per_class_risk": candidate_per_class,
                "apc": apc,
                "risk_protocol": "historical class-risk centroids plus current-session held-out risk supports; not historical per-example empirical risk",
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
            self.model.set_temperatures(self.visual_temperature, self.text_temperature)
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
        logits, labels = torch.cat(all_logits), torch.cat(all_labels)
        class_ids = bank["ids"].cpu()
        target_match = labels[:, None].eq(class_ids[None, :])
        if not target_match.sum(1).eq(1).all():
            raise RuntimeError(
                "All-class classifier does not contain every evaluation label exactly once."
            )
        evaluator = AccuracyEvaluator(self.data.class_index_in_task)
        metrics = evaluator.calc_accuracy(logits, labels, session, class_ids=class_ids)
        base_ids = torch.as_tensor(self.data.class_index_in_task[0], dtype=torch.long)
        base_mask = torch.isin(labels, base_ids)
        if not base_mask.any():
            raise RuntimeError("All-seen evaluation set contains no base-class examples.")
        pred_ids = class_ids[logits.argmax(dim=1)]
        base_accuracy = float(
            100.0 * pred_ids[base_mask].eq(labels[base_mask]).float().mean().item()
        )
        if session == 0:
            self.base_accuracy = base_accuracy
        metrics.update({
            "base_class_accuracy": base_accuracy,
            "num_seen_classes": int(class_ids.numel()),
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
                self.inter_modal_beta = self._fit_inter_modal_beta(bank, temps)
                self.model.set_temperatures(*temps)
                result = {
                    "base_session": True, "accepted": False,
                    "base_calibrated_inter_modal_beta": self.inter_modal_beta,
                }
            else:
                apc = prototype_confusion(
                    {layer: values.to(self.device) for layer, values in fit["layers"].items()},
                    fit["labels"].to(self.device), ids,
                    {layer: torch.stack([self._history_layer_means()[layer][cid]
                                         for cid in sorted(self.class_ids)]).to(self.device)
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
            self._compress_class_memory()
            self._assert_prototype_only_memory()
            self.model.freeze_all()
            evaluation = self._evaluate_all_seen(session)
            evaluation = finalize_fscil_metrics(
                evaluation, self.base_accuracy,
                is_final_session=(session == self.data.num_tasks - 1)
            )
            result["evaluation"] = evaluation
            self.diagnostics.append({"session": session, **result})
            self._save(session)
            # Keep stdout focused on decisions and metrics. Full per-class risks,
            # tensor diagnostics and route details remain in the checkpoint diagnostics.
            accepted = bool(result.get("accepted", False))
            candidates = result.get("candidate_layers", [])
            if accepted:
                expansion_state = "ACCEPTED"
            elif candidates:
                expansion_state = "REJECTED"
            else:
                expansion_state = "NOT_TRIGGERED"

            print("[PC-MoE][Session {:02d}] expansion={} candidates={} reason={}".format(
                int(session), expansion_state, candidates,
                result.get("reason", "capacity transaction evaluated")
            ))

            apc_parts = []
            for layer, stats in sorted(result.get("apc", {}).items(), key=lambda item: int(item[0])):
                apc_parts.append("L{} confusion_mean={:.3f} min_p={:.5f} K={}".format(
                    int(layer), float(stats.get("mean", 0.0)),
                    min(stats.get("p_value_by_class", {0: 1.0}).values()),
                    int(stats.get("comparison_count", 0))
                ))
            if apc_parts:
                print("[PC-MoE][Session {:02d}] APC: {}".format(
                    int(session), " | ".join(apc_parts)
                ))

            risk_fields = (
                "baseline_all_class_risk", "candidate_all_class_risk", "risk_gain",
                "worst_old_class_risk_increase", "worst_new_class_risk_increase",
            )
            risk_parts = [
                "{}={:.4f}".format(key, float(result[key]))
                for key in risk_fields if key in result and result[key] is not None
            ]
            if risk_parts:
                print("[PC-MoE][Session {:02d}] Risk: {}".format(
                    int(session), " | ".join(risk_parts)
                ))

            metrics = result.get("evaluation", {})
            metric_parts = []
            for key in ("mean_acc", "AA", "base_avg_acc", "inc_avg_acc"):
                if key in metrics and metrics[key] is not None:
                    metric_parts.append("{}={:.2f}".format(key, float(metrics[key])))
            if "num_seen_classes" in metrics:
                metric_parts.append("seen_classes={}".format(int(metrics["num_seen_classes"])))
            if metric_parts:
                print("[PC-MoE][Session {:02d}] Eval: {}".format(
                    int(session), " | ".join(metric_parts)
                ))

        final_session = self.data.num_tasks - 1
        final_record = next(
            (item for item in reversed(self.diagnostics) if int(item["session"]) == final_session),
            None
        )
        if final_record is not None:
            final_metrics = final_record.get("evaluation", {})
            if not {"Final Acc", "AA", "PD"}.issubset(final_metrics):
                raise RuntimeError("Final-session diagnostics are missing Final Acc, AA or PD.")
            print(
                "[PC-MoE][Final] Final Acc={:.2f} AA={:.2f} PD={:.2f}pp "
                "(Base Acc={:.2f} -> Final Base Acc={:.2f})".format(
                    float(final_metrics["Final Acc"]), float(final_metrics["AA"]),
                    float(final_metrics["PD"]), float(self.base_accuracy),
                    float(final_metrics["base_class_accuracy"])
                )
            )
        elif self.diagnostics:
            last = self.diagnostics[-1]
            metrics = last.get("evaluation", {})
            print(
                "[PC-MoE][Partial] Session={:02d} Acc={:.2f} AA={:.2f}; "
                "PD deferred until the final real FSCIL session.".format(
                    int(last["session"]), float(metrics.get("mean_acc", 0.0)),
                    float(metrics.get("AA", 0.0))
                )
            )

    def _save(self, session):
        self._compress_class_memory()
        self._assert_prototype_only_memory()
        directory = Path(self.cfg.TRAINER.PCMOE.CHECKPOINT.DIR)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 5, "apc_version": APC_VERSION, "completed_session": int(session),
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
        if payload.get("schema_version") != 5 or payload.get("apc_version") != APC_VERSION:
            raise ValueError(
                "Unsupported PC-MoE checkpoint/APC schema. This version requires schema 5 "
                "with cardinality-matched empirical APC null distributions and prototype-only "
                "historical memory. Older checkpoints must be restarted from Session 0."
            )
        apc_reference = payload.get("apc_reference")
        if not isinstance(apc_reference, dict) or not apc_reference:
            raise ValueError("Checkpoint is missing the cardinality-corrected APC reference.")
        for layer, item in apc_reference.items():
            if not isinstance(item, dict):
                raise ValueError("Checkpoint APC reference for layer {} is malformed.".format(layer))
            pairwise = torch.as_tensor(item.get("pairwise_similarity", []))
            base_count = int(item.get("base_class_count", 0))
            if (base_count < 3 or tuple(pairwise.shape) != (base_count, base_count)
                    or not torch.isfinite(pairwise).all()
                    or not torch.allclose(pairwise, pairwise.t(), atol=1e-5, rtol=1e-5)
                    or not torch.allclose(pairwise.diag(), torch.ones(base_count), atol=1e-4)):
                raise ValueError("Checkpoint APC pairwise geometry for layer {} is inconsistent.".format(layer))
        memory = payload.get("class_memory", {})
        for cid, fields in memory.items():
            for field, part in fields.items():
                if part is None:
                    continue
                if (part.get("kind") != "prototype" or "sources" in part
                        or part["raw"].ndim != 2 or part["raw"].size(0) != 1
                        or any(value.ndim != 2 or value.size(0) != 1
                               for value in part["layers"].values())):
                    raise ValueError(
                        "Checkpoint contains sample-level historical memory for class {} field {}.".format(
                            cid, field
                        )
                    )
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
