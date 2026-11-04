"""Source-disjoint model selection under one all-seen-class scoring rule.

Validation is conditional on the already frozen history. Historical classifier
training cannot be undone, but its held-out rows never train a new candidate or
enter that candidate's calibration references. No test labels are used here.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from models.global_moe import (FeatureExpert, FeatureRoute, balanced_loss,
                               columns, fused_prototypes, unit)


@dataclass(frozen=True)
class RiskConfig:
    epochs: int = 6
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    folds: int = 5
    rank: int = 16
    max_experts: int = 10
    residual_weight: float = 0.1
    min_gain: float = 0.0
    old_risk_tolerance: float = 0.0
    parameter_cost: float = 0.01  # NLL cost per million newly stored parameters
    beta: float = 0.3
    visual_weight: float = 0.1
    visual_tau: float = 16.0
    seed: int = 1
    expand: bool = True

    def validate(self, dim):
        if self.epochs < 1 or self.batch_size < 1 or self.folds < 3:
            raise ValueError("Positive epochs/batch size and at least three folds required")
        if not 1 <= self.rank <= dim or self.max_experts < 0:
            raise ValueError("Invalid expert rank/budget")
        if not 0 <= self.beta <= 1 or not 0 <= self.visual_weight <= 1:
            raise ValueError("Calibration weights must lie in [0,1]")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid nonnegative finite setting: {key}")
        if self.lr == 0 or self.visual_tau == 0:
            raise ValueError("Learning rate and visual temperature must be positive")


def subset(cache, mask):
    return {k: v[mask] if k in ("features", "labels", "sources") else v
            for k, v in cache.items()}


def combine(caches):
    return {k: torch.cat([c[k] for c in caches])
            for k in ("features", "labels", "sources", "class_ids", "text")}


def source_folds(cache, folds, seed):
    """Partition original sources within every class, grouping all their views."""
    assignments = torch.empty_like(cache["labels"])
    generator = torch.Generator().manual_seed(seed)
    for cls in cache["class_ids"].tolist():
        rows = cache["labels"].eq(cls)
        sources = torch.unique(cache["sources"][rows], sorted=True)
        if len(sources) < folds:
            raise ValueError(f"Class {cls} needs at least {folds} distinct sources")
        order = torch.randperm(len(sources), generator=generator).to(sources.device)
        for i, source in enumerate(sources[order]):
            assignments[rows & cache["sources"].eq(source)] = i % folds
    return assignments


def select_candidate(rows, min_gain, old_tolerance):
    """Filter admissible candidates first; then compare their penalized risks.

    This is an empirical selection rule, not a confidence interval or a safety
    certificate. Fold overlap and candidate selection preclude a naive SE claim.
    """
    admissible = [r for r in rows if math.isfinite(r["gain"])
                  and math.isfinite(r["old_regret"])
                  and r["gain"] > min_gain and r["old_regret"] <= old_tolerance]
    return min(admissible, key=lambda r: (r["risk"], r["parameters"], r["name"])) if admissible else None


class FeatureRiskTrainer:
    def __init__(self, head, config):
        self.head, self.cfg = head, config
        config.validate(head.dim)

    def _make(self, name, device, seed):
        # Candidate initialization is identical across folds and final refit;
        # trial RNG consumption cannot perturb future sessions or checkpoint RNG.
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            extra = FeatureExpert(self.head.dim, self.cfg.rank).to(device) if name == "expand" else None
            ids = () if name == "identity" else tuple(range(len(self.head.experts) + int(extra is not None)))
            route = FeatureRoute(self.head.dim, ids).to(device)
        return route, extra

    def _pool(self, extra):
        return list(self.head.experts) + ([] if extra is None else [extra])

    def _prototypes(self, route, pool, support, base, labels=None, sources=None):
        c = self.cfg
        return fused_prototypes(route, pool, support, support["text"], base,
                                c.beta, c.visual_weight, c.visual_tau, labels, sources)

    def _logits(self, route, extra, query, support, base, old_logits, loo=False):
        pool = self._pool(extra)
        h = route(query["features"], pool)
        proto = self._prototypes(route, pool, support, base)
        novel = self.head.scale * h @ proto.T
        if loo:
            is_new = torch.isin(query["labels"], support["class_ids"])
            if bool(is_new.any()):
                loo_proto = self._prototypes(route, pool, support, base,
                                             query["labels"][is_new], query["sources"][is_new])
                novel = novel.clone()
                novel[is_new] = self.head.scale * torch.einsum("bd,bcd->bc", h[is_new], loo_proto)
        return torch.cat([old_logits, novel], dim=1), h

    def _fit(self, name, current, history, base, seed):
        route, extra = self._make(name, current["features"].device, seed)
        params = list(route.parameters()) + ([] if extra is None else list(extra.parameters()))
        if not params:
            return route, extra
        fitting = combine([history, current])
        ids = torch.cat([self.head.class_ids, current["class_ids"]])
        with torch.no_grad():
            old_logits = self.head(fitting["features"])
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        generator = torch.Generator().manual_seed(seed)
        # Use full-objective weights, not within-mini-batch class counts. Summing
        # weighted chunk gradients gives exactly the class-balanced risk gradient.
        targets = columns(fitting["labels"], ids)
        counts = torch.bincount(targets, minlength=len(ids))
        weights = counts[targets].float().reciprocal() / len(ids)
        for _ in range(self.cfg.epochs):
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(targets), generator=generator).to(targets.device)
            for indices in order.split(self.cfg.batch_size):
                query = subset(fitting, indices)
                logits, h = self._logits(route, extra, query, current, base,
                                         old_logits[indices], loo=True)
                loss = (F.cross_entropy(logits, targets[indices], reduction="none") * weights[indices]).sum()
                drift = 1 - (h * unit(query["features"])).sum(-1)
                loss = loss + self.cfg.residual_weight * (drift * weights[indices]).sum()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite all-class candidate loss")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Non-finite candidate parameters")
        return route.eval(), extra

    @torch.no_grad()
    def _risk(self, route, extra, query, support, base):
        ids = torch.cat([self.head.class_ids, support["class_ids"]])
        logits, _ = self._logits(route, extra, query, support, base,
                                 self.head(query["features"]))
        old = torch.isin(query["labels"], self.head.class_ids)
        return (float(balanced_loss(logits, query["labels"], ids)),
                float(balanced_loss(logits[old], query["labels"][old], ids)))

    def add_session(self, current, memories):
        """Select, refit, then atomically bind. Failed trials never mutate history."""
        if not memories or not self.head.banks:
            raise ValueError("Initialize a frozen base classifier before incrementing")
        if torch.isin(current["class_ids"], self.head.class_ids).any():
            raise ValueError("New session contains previously bound classes")
        history = combine(memories)
        base = memories[0]
        seed = self.cfg.seed + 1009 * len(self.head.banks)
        new_fold = source_folds(current, self.cfg.folds, seed)
        old_fold = source_folds(history, self.cfg.folds, seed + 1)
        base_count = len(base["labels"])
        names = (["reuse"] if len(self.head.experts) else [])
        if self.cfg.expand and len(self.head.experts) < self.cfg.max_experts:
            names.append("expand")
        records = {name: {"name": name, "fold_risk": [], "fold_old_risk": []}
                   for name in ["identity", *names]}
        for fold in range(self.cfg.folds):
            support = subset(current, new_fold.ne(fold))
            old_fit = subset(history, old_fold.ne(fold))
            reference = subset(base, old_fold[:base_count].ne(fold))
            validation = combine([subset(history, old_fold.eq(fold)),
                                  subset(current, new_fold.eq(fold))])
            for name, record in records.items():
                route, extra = self._fit(name, support, old_fit, reference, seed)
                risk, old_risk = self._risk(route, extra, validation, support, reference)
                record["fold_risk"].append(risk)
                record["fold_old_risk"].append(old_risk)
                record["parameters"] = sum(p.numel() for p in route.parameters()) + (
                    0 if extra is None else sum(p.numel() for p in extra.parameters()))
        for record in records.values():
            penalty = self.cfg.parameter_cost * record["parameters"] / 1e6
            record["risk"] = sum(record["fold_risk"]) / self.cfg.folds + penalty
            record["old_risk"] = sum(record["fold_old_risk"]) / self.cfg.folds
        baseline = records["identity"]
        for record in records.values():
            record["gain"] = baseline["risk"] - record["risk"]
            record["old_regret"] = record["old_risk"] - baseline["old_risk"]
        best = select_candidate([records[n] for n in names], self.cfg.min_gain,
                                self.cfg.old_risk_tolerance)
        chosen = "identity" if best is None else best["name"]
        route, extra = self._fit(chosen, current, history, base, seed)
        with torch.no_grad():
            proto = self._prototypes(route, self._pool(extra), current, base)
        # bind validates all inputs before mutating either registry.
        self.head.bind(route, current["class_ids"], proto, extra)
        return {"selected": chosen, "candidates": list(records.values()),
                "num_experts": len(self.head.experts), "selection_seed": seed,
                "protocol": "conditional-source-cross-validation"}
