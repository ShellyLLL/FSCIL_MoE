"""Source-exclusive FSCIL model selection with version-owned prototype classifiers.

No held-out test samples or past training images are consulted after Session 0.
All CV folds are separated by original source_id (not augmented image views).
The same reference/owner-topology score is used for planning, training and testing.
"""
from collections import defaultdict
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def topology_key(topology):
    return tuple(sorted((int(k), int(v)) for k, v in topology.items()))


def unit(x):
    return F.normalize(x.float(), dim=-1)


def class_prototypes(features, labels, classes, excluded=None):
    """Compute normalized class means with optional query-specific source exclusion."""
    result = []
    for cls in classes:
        mask = labels.eq(int(cls))
        if excluded is not None:
            mask = mask & ~excluded
        if not bool(mask.any()):
            raise ValueError("Every query needs at least one other support for its class")
        result.append(unit(unit(features[mask]).mean(dim=0)))
    return torch.stack(result)


def make_fused(visual, text, beta):
    return unit(float(beta) * unit(text) + (1.0 - float(beta)) * unit(visual))


class VersionedScorer:
    """Shared S4/S7 score semantics: reference and each class's owner topology."""

    def __init__(self, model, old_states, text_state, reference_topology, eta):
        self.model = model
        self.old_states = list(old_states)
        self.text_state = text_state
        self.reference_topology = dict(reference_topology)
        self.eta = float(eta)
        self.beta = float(model.cfg.DATASET.BETA)
        self.device = model.device
        self.new_ids = text_state["class_ids"].to(self.device)
        self.new_text = text_state["calibrated_text"].to(self.device)
        self.old_ids = torch.cat(
            [state["class_ids"].to(self.device) for state in self.old_states]
        ) if self.old_states else torch.empty(0, dtype=torch.long, device=self.device)
        self.class_ids = torch.cat([self.old_ids, self.new_ids])
        if torch.unique(self.class_ids).numel() != self.class_ids.numel():
            raise ValueError("Duplicate class ids in versioned classifier")
        self.base_prototypes = model.base_vision_prototype.to(self.device)

    def feature(self, images, topology, no_grad=True):
        if no_grad:
            with torch.no_grad():
                return unit(self.model.clip_model.encode_image(
                    images.to(self.device), active_counts=topology
                ))
        return unit(self.model.clip_model.encode_image(
            images.to(self.device), active_counts=topology
        ))

    def visual_calibrate(self, visual):
        if not self.model.cfg.TRAINER.BiMC.VISION_CALIBRATION:
            return unit(visual)
        base = unit(self.base_prototypes)
        weight = torch.softmax(
            float(self.model.cfg.TRAINER.BiMC.TAU) * unit(visual) @ base.t(), dim=-1
        )
        shift = unit(weight @ base)
        lam = float(self.model.cfg.TRAINER.BiMC.LAMBDA_I)
        return unit((1.0 - lam) * unit(visual) + lam * shift)

    @torch.no_grad()
    def support_bank(self, loader, owner_topology):
        ref, dyn, labels, sources = [], [], [], []
        for batch in loader:
            images = batch["image"].to(self.device)
            labels.append(batch["label"].to(self.device).long())
            sources.append(batch["source_id"].to(self.device).long())
            ref.append(self.feature(images, self.reference_topology))
            if topology_key(owner_topology) == topology_key(self.reference_topology):
                dyn.append(ref[-1])
            else:
                dyn.append(self.feature(images, owner_topology))
        bank = {
            "reference": torch.cat(ref), "dynamic": torch.cat(dyn),
            "labels": torch.cat(labels), "source_ids": torch.cat(sources),
        }
        if torch.unique(bank["source_ids"]).numel() != len(bank["source_ids"]):
            raise ValueError("Support bank must contain one deterministic view per source_id")
        if not torch.equal(torch.unique(bank["labels"]).sort()[0], self.new_ids.sort()[0]):
            raise ValueError("Support bank classes differ from the current session")
        return bank

    def current_fused(self, bank, exclude_source_ids=None):
        """Full-shot or source-exclusive banks; all vectors are detached."""
        labels = bank["labels"]
        if exclude_source_ids is None:
            ref = class_prototypes(bank["reference"], labels, self.new_ids)
            dyn = class_prototypes(bank["dynamic"], labels, self.new_ids)
        else:
            ref, dyn = [], []
            for sid in exclude_source_ids:
                exclude = bank["source_ids"].eq(sid)
                ref.append(class_prototypes(bank["reference"], labels, self.new_ids, exclude))
                dyn.append(class_prototypes(bank["dynamic"], labels, self.new_ids, exclude))
            ref, dyn = torch.stack(ref), torch.stack(dyn)
        reference = make_fused(self.visual_calibrate(ref), self.new_text, self.beta)
        dynamic = make_fused(dyn, self.new_text, self.beta)
        return reference, dynamic

    def logits(self, images, owner_topology, current_reference, current_dynamic,
               include_old=True, train_new=False, logit_scale=25.0):
        """One common label order for both training and validation.

        current_reference/dynamic: [C,D] or per-query [B,C,D]. Old state
        representations are immutable and evaluated through their owner paths.
        """
        image_ref = self.feature(images, self.reference_topology)
        image_dyn = (image_ref if topology_key(owner_topology) == topology_key(self.reference_topology)
                     else self.feature(images, owner_topology, no_grad=not train_new))
        def cosine(feature, prototypes):
            p = unit(prototypes.to(self.device))
            return (torch.einsum("bd,bcd->bc", feature, p)
                    if p.ndim == 3 else feature @ p.t())
        current = ((1.0 - self.eta) * cosine(image_ref, current_reference)
                   + self.eta * cosine(image_dyn, current_dynamic))
        history = []
        if include_old:
            # Cache each historical topology within this batch.
            historical_features = {topology_key(self.reference_topology): image_ref}
            for state in self.old_states:
                version = state["topology"]
                key = topology_key(version)
                if key not in historical_features:
                    historical_features[key] = self.feature(images, version)
                old_ref = state["ref_fused_proto"].to(self.device)
                old_own = state["fused_proto"].to(self.device)
                history.append(
                    (1.0 - self.eta) * cosine(image_ref, old_ref)
                    + self.eta * cosine(historical_features[key], old_own)
                )
        return float(logit_scale) * torch.cat([*history, current], dim=-1)

    def targets(self, labels):
        labels = labels.reshape(-1).to(self.device)
        match = labels[:, None].eq(self.class_ids[None, :])
        if not bool(match.sum(dim=1).eq(1).all()):
            raise ValueError("Each query must map to exactly one prototype")
        return match.long().argmax(dim=1)

    @torch.no_grad()
    def evaluate(self, loader, topology, bank, exclude_query=False):
        total_loss, total_right, total, margins = 0., 0, 0, []
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).long()
            exclude = batch["source_id"].to(self.device).long() if exclude_query else None
            ref, dyn = self.current_fused(bank, exclude)
            logits = self.logits(
                images, topology, ref, dyn,
                logit_scale=float(self.model.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
            )
            target = self.targets(labels)
            total_loss += F.cross_entropy(logits, target, reduction="sum").item()
            total_right += logits.argmax(-1).eq(target).sum().item()
            total += len(target)
            true = logits.gather(1, target[:, None]).squeeze(1)
            competing = logits.scatter(1, target[:, None], float("-inf")).max(dim=-1).values
            margins.extend(((true - competing) / float(self.model.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)).tolist())
        return {
            "ce": total_loss / max(1, total), "accuracy": total_right / max(1, total),
            "margins": margins,
        }


def source_stratified_folds(dataset, folds):
    """Deterministic class-stratified folds; each raw image belongs to one fold."""
    by_class = defaultdict(list)
    for i in range(len(dataset)):
        by_class[int(dataset.labels[i])].append((int(dataset.source_ids[i]), i))
    min_shot = min(len(v) for v in by_class.values())
    if min_shot < 3:
        raise ValueError("CV requires at least three distinct source images per class")
    folds = min(int(folds), min_shot)
    ordered = {c: [i for _, i in sorted(v)] for c, v in by_class.items()}
    for fold in range(folds):
        val = [row for rows in ordered.values() for rank, row in enumerate(rows)
               if rank % folds == fold]
        train = [row for rows in ordered.values() for rank, row in enumerate(rows)
                 if rank % folds != fold]
        yield sorted(train), sorted(val)


def subset_loader(loader, indices, shuffle=False):
    return DataLoader(
        Subset(loader.dataset, indices), batch_size=loader.batch_size,
        shuffle=shuffle, num_workers=loader.num_workers,
        pin_memory=loader.pin_memory, drop_last=False,
    )


class DemandExpansion:
    """Single, transactional model-selection operation per increment."""

    def __init__(self, runner, old_states, text_state):
        self.runner = runner
        self.model = runner.model_without_dp
        self.old_states = old_states
        self.text_state = text_state
        self.config = runner.cfg.TRAINER.BiMC.DEMAND
        self.logit_scale = float(runner.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE)
        self.reference = dict(self.model.reference_topology)
        self.eta = float(self.config.DYNAMIC_WEIGHT)

    def scorer(self):
        return VersionedScorer(
            self.model, self.old_states, self.text_state, self.reference, self.eta
        )

    @torch.no_grad()
    def layer_scores(self, loader, scorer, base_bank, prior):
        """Negative LOO margin times sample-wise minimum positive descriptor z."""
        metrics = scorer.evaluate(loader, prior, base_bank, exclude_query=True)
        margins = torch.tensor(metrics["margins"])
        difficulty = (-margins).clamp_min(0)
        scores = {}
        if not bool(difficulty.gt(0).any()):
            return scores
        by_version = {}
        for block, adapter in self.runner._moe_adapters():
            adapter.assert_descriptors_ready()
            expert_scores = []
            for expert_id, descriptor in enumerate(adapter.descriptors):
                version = adapter.descriptor_topologies[expert_id] or self.reference
                key = topology_key(version)
                if key not in by_version:
                    by_version[key] = self.model.collect_blockwise_cls_features(
                        loader, active_counts=version
                    )
                values = by_version[key][block]["cls_in"].to(self.model.device)
                expert_scores.append(descriptor.standardized_score(values).detach().cpu())
            z = torch.stack(expert_scores, dim=-1)
            coverage = z.clamp_min(0).min(dim=-1).values
            # Novel class confusion is the necessity signal. Descriptors only
            # rank candidate layers; a covered but nondiscriminative class is
            # still eligible for expansion if CV demonstrates a benefit.
            scores[block] = float(((1.0 + coverage) * difficulty).mean().item())
        return scores

    def train_candidate(self, block, train_loader, train_extract, old_topology,
                        epochs=None):
        """Train only the newest adapter/column with refreshed LOO banks."""
        adapter = self.runner._adapter_by_block(block)
        adapter.set_newest_trainable()
        scorer = self.scorer()
        options = self.runner.cfg.TRAINER.BiMC.OPTIM
        optimizer = torch.optim.AdamW([
            {"params": adapter.experts[-1].parameters(),
             "lr": float(options.LR_INCREMENTAL_EXPERT)},
            {"params": adapter.router.columns[-1].parameters(),
             "lr": float(options.LR_INCREMENTAL_ROUTER)},
        ], weight_decay=float(options.WEIGHT_DECAY))
        epochs = int(epochs or self.runner.cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS)
        for _ in range(epochs):
            # Refresh per-source prototypes after every parameter update epoch.
            self.model.eval()
            bank = scorer.support_bank(train_extract, self.model.current_topology())
            self.model.train()
            for batch in train_loader:
                images = batch["image"].to(self.model.device)
                labels = batch["label"].to(self.model.device).long()
                ids = batch["source_id"].to(self.model.device).long()
                ref, dyn = scorer.current_fused(bank, ids)
                optimizer.zero_grad(set_to_none=True)
                logits = scorer.logits(images, self.model.current_topology(),
                                       ref, dyn, train_new=True,
                                       logit_scale=self.logit_scale)
                loss = F.cross_entropy(logits, scorer.targets(labels))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite candidate CV loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.newest_parameters(), 1.0)
                optimizer.step()
        adapter.freeze_all()

    def select(self, task_id, train_loader, extract_loader):
        """CV compares reuse with each eligible layer; no commit during CV."""
        old_topology = self.model.current_topology()
        scorer = self.scorer()
        full = scorer.support_bank(extract_loader, old_topology)
        scores = self.layer_scores(extract_loader, scorer, full, old_topology)
        ranked = sorted((block for block, value in scores.items() if value > 0),
                        key=lambda block: (-scores[block], block))
        folds = list(source_stratified_folds(extract_loader.dataset, self.config.CV_FOLDS))
        train_dataset = train_loader.dataset
        extract_dataset = extract_loader.dataset
        if list(train_dataset.source_ids) != list(extract_dataset.source_ids):
            raise ValueError("CV train/extract loaders must share exactly the same source order")
        baselines = []
        for train_idx, val_idx in folds:
            tr = subset_loader(extract_loader, train_idx)
            val = subset_loader(extract_loader, val_idx)
            bank = scorer.support_bank(tr, old_topology)
            baselines.append(scorer.evaluate(val, old_topology, bank)["ce"])
        baseline = sum(baselines) / len(baselines)
        choices = []
        before_state = {block: self.runner._historical_snapshot(adapter)
                        for block, adapter in self.runner._moe_adapters()}
        for block in ranked:
            candidate_losses = []
            for train_idx, val_idx in folds:
                adapter = self.runner._adapter_by_block(block)
                self.model.add_expert(block, task_id)
                try:
                    self.train_candidate(
                        block, subset_loader(train_loader, train_idx, shuffle=True),
                        subset_loader(extract_loader, train_idx), old_topology
                    )
                    candidate_scorer = self.scorer()
                    # Validation prototypes use training sources only.
                    bank = candidate_scorer.support_bank(
                        subset_loader(extract_loader, train_idx),
                        self.model.current_topology()
                    )
                    val = subset_loader(extract_loader, val_idx)
                    candidate_losses.append(candidate_scorer.evaluate(
                        val, self.model.current_topology(), bank
                    )["ce"])
                finally:
                    adapter.discard_newest()
                ok, changed = self.runner._historical_state_unchanged(
                    adapter, before_state[block]
                )
                if not ok:
                    raise RuntimeError("CV modified historical modules: " + str(changed))
            # Two dense bottleneck projections and one independent routing column.
            d_model = self.runner._adapter_by_block(block).d_model
            reduction = int(self.runner.cfg.TRAINER.BiMC.VISUAL_MOE.INCREMENTAL_REDUCTION)
            width = max(1, d_model // reduction)
            trainable_count = 2 * d_model * width + width + 2 * d_model + 1
            base_count = sum(
                p.numel() for _, adapter in self.runner._moe_adapters()
                for expert in adapter.experts for p in expert.parameters()
            )
            complexity = math.log1p(trainable_count / max(1, base_count))
            cv_loss = sum(candidate_losses) / len(candidate_losses)
            objective = cv_loss + float(self.config.COMPLEXITY_WEIGHT) * complexity
            choices.append({
                "block": block, "cv_loss": cv_loss, "complexity": complexity,
                "objective": objective, "coverage_difficulty": scores[block],
            })
        accepted = min(choices, key=lambda x: (x["objective"], x["block"])) if choices else None
        if accepted is None or accepted["objective"] >= baseline:
            return {
                "trained": False, "expanded": [], "baseline_ce": baseline,
                "candidates": choices, "scores": scores,
            }
        block = accepted["block"]
        adapter = self.runner._adapter_by_block(block)
        self.model.add_expert(block, task_id)
        try:
            self.train_candidate(block, train_loader, extract_loader, old_topology)
            self.runner.train_descriptor_for_expert(block, extract_loader)
            adapter.set_descriptor_topology(adapter.num_experts - 1,
                                            self.model.current_topology())
            adapter.freeze_all()
        except Exception:
            adapter.discard_newest()
            raise
        return {
            "trained": True,
            "expanded": [{"block_idx": block, "expert_id": adapter.num_experts - 1}],
            "baseline_ce": baseline, "candidates": choices,
            "selected": accepted, "scores": scores,
        }
