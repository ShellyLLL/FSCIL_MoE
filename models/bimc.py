
import json
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.clip.clip as clip


@dataclass
class LayerExpansionDecision:
    block_idx: int
    expand: bool
    class_scores: Dict[int, float]
    uncovered_classes: List[int]


def compute_loo_classification_metrics(queries, per_query_prototypes, query_class_ids,
                                       prototype_class_ids):
    """Classify each query against its own query-exclusive prototype bank.

    Args:
        queries: ``[N, D]`` query representations.
        per_query_prototypes: ``[N, C, D]``.  For FSCIL support LOO this has a
            four-shot prototype for the query's own class and normal class
            prototypes for every other current/history class.
        query_class_ids: global labels, shape ``[N]``.
        prototype_class_ids: global labels for the C columns.

    The return value contains cosine margins rather than temperature-dependent
    logits, so the same value can be used for threshold calibration and early
    stopping.  Class ids, not prototype positions, determine the positive
    column.
    """
    if queries.ndim != 2 or per_query_prototypes.ndim != 3:
        raise ValueError("queries must be [N,D] and per_query_prototypes must be [N,C,D].")
    if queries.size(0) != per_query_prototypes.size(0) or queries.size(1) != per_query_prototypes.size(2):
        raise ValueError("query and per-query prototype dimensions must align.")
    ids = torch.as_tensor(prototype_class_ids, device=queries.device, dtype=torch.long).reshape(-1)
    labels = torch.as_tensor(query_class_ids, device=queries.device, dtype=torch.long).reshape(-1)
    if labels.numel() != queries.size(0) or ids.numel() != per_query_prototypes.size(1):
        raise ValueError("global class ids must align with query/prototype dimensions.")
    positive = labels[:, None].eq(ids[None, :])
    if not positive.any(dim=1).all():
        missing = labels[~positive.any(dim=1)].unique().tolist()
        raise ValueError(f"LOO prototype bank misses query class ids: {missing}")
    q = F.normalize(queries.float(), dim=-1)
    proto = F.normalize(per_query_prototypes.float().to(queries.device), dim=-1)
    logits = torch.einsum("nd,ncd->nc", q, proto)
    true_score = logits.masked_fill(~positive, float("-inf")).max(dim=1).values
    competitor = logits.masked_fill(positive, float("-inf")).max(dim=1).values
    competitor = torch.where(torch.isfinite(competitor), competitor, true_score.new_full(true_score.shape, -1.0))
    pred = ids[logits.argmax(dim=1)]
    return {
        "logits": logits,
        "predictions": pred,
        "accuracy": pred.eq(labels).float().mean(),
        "margins": true_score - competitor,
        "true_scores": true_score,
        "competitor_scores": competitor,
    }


def build_loo_cache_from_embeddings(features, labels, source_ids, class_index=None):
    """Build source-exclusive visual prototypes from already encoded features."""
    feature = F.normalize(torch.as_tensor(features).detach(), dim=-1)
    target = torch.as_tensor(labels, device=feature.device, dtype=torch.long).reshape(-1)
    source = torch.as_tensor(source_ids, device=feature.device).reshape(-1)
    if feature.ndim != 2 or feature.size(0) != target.numel() or source.numel() != target.numel():
        raise ValueError("features, labels and source_ids must align for visual LOO.")
    classes = (
        torch.unique(target, sorted=True)
        if class_index is None
        else torch.unique(torch.as_tensor(class_index, device=feature.device, dtype=torch.long), sorted=True)
    )
    rows_per_query = []
    for sid in source:
        rows = []
        for cls in classes:
            mask = target.eq(cls) & source.ne(sid)
            rows.append(feature[mask].mean(0) if mask.any() else feature.new_zeros(feature.size(1)))
        rows_per_query.append(F.normalize(torch.stack(rows), dim=-1))
    loo_valid = torch.stack([
        (target.eq(target[index]) & source.ne(source[index])).any()
        for index in range(source.numel())
    ])
    return {
        "features": feature,
        "labels": target,
        "source_ids": source,
        "classes": classes,
        "loo_visual_prototypes": torch.stack(rows_per_query),
        "loo_valid": loo_valid,
    }


def refine_conflict_aware_fused_prototypes(
    current_prototypes, current_class_ids, history_prototypes=None,
    history_class_ids=None, strength=0.25, margin=0.05,
):
    """Repel only current classes that collide with a different historical class.

    The update is deterministic, normalised and leaves novel classes unchanged
    when there is no historical competitor.  It is intentionally a standalone
    helper so callers can unit-test it without constructing CLIP.
    """
    current = F.normalize(current_prototypes, dim=-1)
    if history_prototypes is None or torch.as_tensor(history_prototypes).numel() == 0:
        return current
    hist = F.normalize(torch.as_tensor(history_prototypes, device=current.device, dtype=current.dtype), dim=-1)
    cids = torch.as_tensor(current_class_ids, device=current.device).reshape(-1)
    hids = torch.as_tensor(history_class_ids, device=current.device).reshape(-1)
    sim = current @ hist.t()
    sim = sim.masked_fill(cids[:, None].eq(hids[None, :]), float("-inf"))
    best, best_idx = sim.max(dim=1)
    active = torch.isfinite(best) & (best > float(margin))
    if active.any():
        repel = hist[best_idx[active]]
        weight = float(strength) * ((best[active] - float(margin)) / max(1e-6, 1.0 - float(margin))).clamp(0, 1)
        current = current.clone()
        current[active] = F.normalize(current[active] - weight[:, None] * repel, dim=-1)
    return current


def optimize_query_exclusive_fused_prototypes(
    loo_current_fused, query_features, query_class_ids, current_class_ids,
    history_fused=None, history_class_ids=None, epochs=30, lr=1e-2,
    old_protect_weight=1.0, anchor_weight=1.0, old_protect_margin=0.70,
    initial_global=None, return_metrics=False,
):
    """Optimize only current-class fused prototype residuals.

    ``loo_current_fused`` is ``[N, C_new, D]`` and is already query-exclusive.
    A single residual per new class is shared across all LOO banks, which keeps
    the update compact while avoiding the accidental full-shot self prototype
    often seen in FSCIL prototype refinement.
    """
    if loo_current_fused.ndim != 3 or query_features.ndim != 2:
        raise ValueError("loo_current_fused must be [N,C,D] and query_features [N,D].")
    n, c, d = loo_current_fused.shape
    if query_features.shape != (n, d):
        raise ValueError("query_features must align with LOO fused prototype dimensions.")
    current_ids = torch.as_tensor(current_class_ids, device=query_features.device, dtype=torch.long).reshape(-1)
    labels = torch.as_tensor(query_class_ids, device=query_features.device, dtype=torch.long).reshape(-1)
    if current_ids.numel() != c or labels.numel() != n:
        raise ValueError("current/query class ids do not align with prototype tensors.")

    base = F.normalize(loo_current_fused.detach().to(query_features.device).float(), dim=-1)
    if initial_global is None:
        initial = F.normalize(base.mean(dim=0), dim=-1)
    else:
        initial = F.normalize(
            torch.as_tensor(initial_global, device=query_features.device, dtype=base.dtype), dim=-1
        )
        if initial.shape != (c, d):
            raise ValueError("initial_global must have shape [num_current_classes, feature_dim].")
    if history_fused is None:
        history = base.new_empty((0, d))
        history_ids = current_ids.new_empty(0)
    else:
        history = F.normalize(torch.as_tensor(history_fused, device=base.device, dtype=base.dtype), dim=-1)
        history_ids = torch.as_tensor(history_class_ids, device=base.device, dtype=torch.long).reshape(-1)
        if history.size(0) != history_ids.numel():
            raise ValueError("history prototype ids must align with history prototypes.")

    delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))
    q = F.normalize(query_features.detach().to(base.device).float(), dim=-1)
    all_ids = torch.cat([history_ids, current_ids], dim=0)
    target_cols = torch.nonzero(labels[:, None].eq(all_ids[None, :]), as_tuple=False)
    if target_cols.size(0) != n:
        raise ValueError("every refinement query must have exactly one class column.")
    target_cols = target_cols[:, 1]

    @torch.no_grad()
    def loo_metrics_for(delta_value):
        current = F.normalize(base + delta_value[None, :, :], dim=-1)
        bank = torch.cat([history[None, :, :].expand(n, -1, -1), current], dim=1)
        return compute_loo_classification_metrics(q, bank, labels, all_ids)

    # The baseline is the actual task-level fused bank (when supplied), not
    # an average of LOO means.  This guarantees prototype refinement cannot
    # silently replace a working incremental classifier with a worse last
    # optimizer step merely because its regularised loss decreased.
    best_delta = torch.zeros_like(initial)
    best_metrics = loo_metrics_for(best_delta)
    best_accuracy = float(best_metrics["accuracy"].item())
    best_margin = float(best_metrics["margins"].mean().item())
    best_epoch = 0

    for epoch in range(max(0, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        current = F.normalize(base + delta[None, :, :], dim=-1)
        bank = torch.cat([history[None, :, :].expand(n, -1, -1), current], dim=1)
        logits = 10.0 * torch.einsum("nd,ncd->nc", q, bank)
        loss_loo = F.cross_entropy(logits, target_cols)

        # A low-temperature old classification term alone can still saturate
        # when a stored prototype self-similarity is 1.  Add an explicit cosine
        # margin against new prototypes so colliding new classes keep gradients.
        current_global = F.normalize(initial + delta, dim=-1)
        if history.numel():
            old_logits = 10.0 * history @ torch.cat([history, current_global], dim=0).t()
            old_targets = torch.arange(history.size(0), device=history.device)
            loss_old_ce = F.cross_entropy(old_logits, old_targets)
            collision = F.relu(history @ current_global.t() - float(old_protect_margin)).pow(2).mean()
            loss_old = loss_old_ce + collision
        else:
            loss_old = loss_loo.new_zeros(())
        loss_anchor = 1.0 - F.cosine_similarity(current_global, initial, dim=-1).mean()
        loss = loss_loo + float(old_protect_weight) * loss_old + float(anchor_weight) * loss_anchor
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        if not torch.isfinite(delta).all():
            with torch.no_grad():
                delta.copy_(best_delta)
            break
        candidate_metrics = loo_metrics_for(delta.detach())
        candidate_accuracy = float(candidate_metrics["accuracy"].item())
        candidate_margin = float(candidate_metrics["margins"].mean().item())
        if (candidate_accuracy > best_accuracy + 1e-8 or
                (abs(candidate_accuracy - best_accuracy) <= 1e-8
                 and candidate_margin > best_margin + 1e-8)):
            best_delta = delta.detach().clone()
            best_metrics = candidate_metrics
            best_accuracy, best_margin, best_epoch = candidate_accuracy, candidate_margin, epoch + 1

    result = F.normalize(initial + best_delta, dim=-1)
    if not return_metrics:
        return result
    return result, {
        "pre_loo_accuracy": float(loo_metrics_for(torch.zeros_like(initial))["accuracy"].item()),
        "pre_loo_margin": float(loo_metrics_for(torch.zeros_like(initial))["margins"].mean().item()),
        "post_loo_accuracy": best_accuracy,
        "post_loo_margin": best_margin,
        "selected_epoch": int(best_epoch),
    }


def loo_fused_prototypes_for_queries(support_cache, query_source_ids, old_state,
                                     class_names=None, beta=0.5, lambda_t=0.0):
    """Construct per-query all-class fused banks without support-source leakage.

    ``support_cache`` is produced by :meth:`build_leave_one_out_prototypes` and
    should additionally contain current ``text_features`` and optionally
    ``description_proto``. ``old_state`` may be an engine merged state dict;
    its ``image_proto``, ``text_features``, ``description_proto`` and
    ``class_index`` entries are appended as the historical bank. ``class_names``
    is accepted for engine-call compatibility and is not used for positional ids.
    """
    del class_names
    source = torch.as_tensor(query_source_ids).reshape(-1)
    support_sources = torch.as_tensor(support_cache["source_ids"]).reshape(-1)
    rows = []
    text = support_cache.get("text_features")
    desc = support_cache.get("description_proto", text)
    if text is None:
        raise ValueError("support_cache needs text_features to construct fused prototypes.")
    text = torch.as_tensor(text)
    desc = torch.as_tensor(desc, device=text.device, dtype=text.dtype)
    current_text = F.normalize((1.0 - float(lambda_t)) * text + float(lambda_t) * desc, dim=-1)
    old_img = old_state.get("image_proto") if old_state is not None else None
    old_explicit_fused = old_state.get("fused_proto") if old_state is not None else None
    if old_explicit_fused is not None:
        old_fused = F.normalize(torch.as_tensor(old_explicit_fused, device=text.device, dtype=text.dtype), dim=-1)
    elif old_img is not None:
        old_img = torch.as_tensor(old_img, device=text.device, dtype=text.dtype)
        old_text = torch.as_tensor(old_state.get("text_features", old_img), device=text.device, dtype=text.dtype)
        old_desc = torch.as_tensor(old_state.get("description_proto", old_text), device=text.device, dtype=text.dtype)
        old_fused = F.normalize(float(beta) * F.normalize((1-float(lambda_t))*old_text + float(lambda_t)*old_desc, dim=-1)
                                 + (1-float(beta)) * F.normalize(old_img, dim=-1), dim=-1)
    else:
        old_fused = text.new_empty((0, text.size(-1)))
    for sid in source:
        hit = torch.nonzero(support_sources.eq(sid), as_tuple=False).flatten()
        if hit.numel() == 0:
            raise KeyError("query source id is absent from support_cache")
        current = support_cache["loo_visual_prototypes"][hit[0]].to(text.device, text.dtype)
        fused = F.normalize(float(beta) * current_text + (1.0 - float(beta)) * current, dim=-1)
        rows.append(torch.cat([old_fused, fused], dim=0))
    return torch.stack(rows, dim=0)


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), cfg=cfg)
    return model


class BiMC(nn.Module):
    def __init__(self, cfg, template, device):
        super(BiMC, self).__init__()
        self.cfg = cfg
        self.device = device
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.BiMC.PREC in ["fp32", "amp"]:
            clip_model.float()
        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None
        self.base_vision_prototype = None
        self.reference_topology = None  # immutable after base-session training
        # Runtime cache for conflict-aware fused refinement.  Adapter coverage
        # banks remain the checkpointed source of truth for visual planning.
        self._fused_history_prototypes = None
        self._fused_history_class_ids = None
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if hasattr(self.cfg.TRAINER.BiMC, 'VISUAL_MOE') and self.cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE:
            for name, param in self.clip_model.visual.named_parameters():
                # 只训练显式插入的视觉 MoE Adapter。不要使用 "proj" 模糊匹配，
                # 否则会误解冻 attention projection、MLP projection 和 visual.proj。
                if "moe_adapter" in name:
                    # Descriptors are opened only by their reconstruction-only
                    # training stage; classification never owns their gradient.
                    param.requires_grad = ".descriptors." not in name
        trainable = [
            name for name, param in self.clip_model.named_parameters()
            if param.requires_grad
        ]
        trainable_numel = sum(
            param.numel() for param in self.clip_model.parameters() if param.requires_grad
        )
        total_numel = sum(param.numel() for param in self.clip_model.parameters())
        print(
            f"[Freeze] trainable tensors: {len(trainable)}, "
            f"parameters: {trainable_numel:,}/{total_numel:,}"
        )
        for name in trainable:
            print(f"  - {name}")

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.clip_model.eval()
            if hasattr(self.cfg.TRAINER.BiMC, 'VISUAL_MOE') and self.cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE:
                for block in self.clip_model.visual.transformer.resblocks:
                    if hasattr(block, 'moe_adapter') and block.moe_adapter is not None:
                        block.moe_adapter.train()
                        if not any(
                            param.requires_grad
                            for param in block.moe_adapter.router.parameters()
                        ):
                            block.moe_adapter.router.eval()
        return self

    @torch.no_grad()
    def collect_blockwise_cls_features(self, loader, active_counts=None):
        """Collect current-path Adapter-LayerNorm CLS features and labels."""
        block_features, fallback_source_id = {}, 0
        moe_blocks = [idx for idx, block in enumerate(self.clip_model.visual.transformer.resblocks)
                      if getattr(block, "moe_adapter", None) is not None]
        if not moe_blocks:
            return block_features
        was_training = self.training
        self.eval()
        try:
            for batch in loader:
                images, labels = self.parse_batch(batch)
                raw = batch.get("source_ids", batch.get("source_id", batch.get("index"))) if isinstance(batch, dict) else None
                source_ids = (torch.arange(fallback_source_id, fallback_source_id + labels.numel())
                              if raw is None else torch.as_tensor(raw).reshape(-1))
                fallback_source_id += labels.numel()
                if source_ids.numel() != labels.numel():
                    raise ValueError("support source ids must align with labels.")
                _, aux_list = self.clip_model.encode_image(
                    images, return_moe_aux=True, active_counts=active_counts
                )
                if len(aux_list) != len(moe_blocks):
                    raise RuntimeError("MoE auxiliary outputs do not match inserted blocks.")
                for block_idx, aux in zip(moe_blocks, aux_list):
                    cls_in = aux.get("cls_in")
                    if cls_in is None or cls_in.ndim != 2:
                        raise RuntimeError(f"Block {block_idx} has invalid CLS features.")
                    item = block_features.setdefault(block_idx, {
                        "cls_in": [], "patch_mean_in": [], "labels": [], "source_ids": []
                    })
                    item["cls_in"].append(cls_in.detach().cpu())
                    if aux.get("patch_mean_in") is not None:
                        item["patch_mean_in"].append(aux["patch_mean_in"].detach().cpu())
                    item["labels"].append(labels.detach().cpu())
                    item["source_ids"].append(source_ids.detach().cpu())
        finally:
            self.train(was_training)
        for item in block_features.values():
            item["cls_in"], item["labels"] = torch.cat(item["cls_in"]), torch.cat(item["labels"])
            item["source_ids"] = torch.cat(item["source_ids"])
            item["patch_mean_in"] = (torch.cat(item["patch_mean_in"]) if item["patch_mean_in"]
                                     else item["cls_in"].new_empty(0, item["cls_in"].size(-1)))
        return block_features

    @torch.no_grad()
    def build_leave_one_out_prototypes(self, loader, class_index=None, active_counts=None):
        """Collect a support cache with stable source ids and per-query LOO means.

        ``loo_visual_prototypes[i, j]`` is class ``j``'s support mean after
        removing all views having query ``i``'s source id.  This prevents a
        query from winning merely because its own augmented view is in support.
        """
        features, labels, source_ids = [], [], []
        next_id = 0
        was_training = self.training
        self.eval()
        try:
            for batch in loader:
                images, y = self.parse_batch(batch)
                raw = batch.get("source_ids", batch.get("source_id", batch.get("index"))) if isinstance(batch, dict) else None
                sid = torch.arange(next_id, next_id + y.numel()) if raw is None else torch.as_tensor(raw).reshape(-1)
                if sid.numel() != y.numel():
                    raise ValueError("support source ids must align with labels.")
                next_id += y.numel()
                features.append(F.normalize(
                    self.clip_model.encode_image(images, active_counts=active_counts), dim=-1
                ).detach().cpu())
                labels.append(y.detach().cpu())
                source_ids.append(sid.detach().cpu())
        finally:
            self.train(was_training)
        return build_loo_cache_from_embeddings(
            torch.cat(features), torch.cat(labels).long(), torch.cat(source_ids), class_index
        )

    @staticmethod
    def _state_class_ids(state, device=None):
        if not state:
            return torch.empty(0, dtype=torch.long, device=device)
        ids = state.get("class_ids", state.get("class_index", []))
        return torch.as_tensor(ids, dtype=torch.long, device=device).reshape(-1)

    def fuse_prototypes(self, state):
        """Return the classifier prototypes in a state, preserving class ids."""
        if state is None:
            return None
        fused = state.get("fused_proto")
        if fused is not None:
            return F.normalize(torch.as_tensor(fused, device=self.device), dim=-1)
        image = torch.as_tensor(state["image_proto"], device=self.device)
        text = torch.as_tensor(state["text_features"], device=self.device)
        description = torch.as_tensor(state.get("description_proto", text), device=self.device)
        lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        calibrated_text = F.normalize((1.0 - lambda_t) * text + lambda_t * description, dim=-1)
        return F.normalize(self.cfg.DATASET.BETA * calibrated_text + (1.0 - self.cfg.DATASET.BETA) * image, dim=-1)

    def current_topology(self):
        return {
            int(index): int(block.moe_adapter.num_experts)
            for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
            if getattr(block, "moe_adapter", None) is not None
        }

    @staticmethod
    def canonical_topology(topology):
        if topology is None:
            return None
        return {int(k): int(v) for k, v in topology.items()}

    def expandable_blocks(self):
        return [index for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
                if getattr(block, "moe_adapter", None) is not None]

    @torch.no_grad()
    def plan_layer_expansion(self, block_idx, loader, class_index, threshold=None):
        """Decide one layer solely from class-aggregated descriptor coverage."""
        block_idx = int(block_idx)
        threshold = float(getattr(
            self.cfg.TRAINER.BiMC.VISUAL_MOE, "EXPANSION_Z_THRESHOLD", 1.0
        ) if threshold is None else threshold)
        features = self.collect_blockwise_cls_features(loader).get(block_idx)
        if features is None:
            raise KeyError(f"Block {block_idx} did not produce support features.")
        adapter = self.clip_model.visual.transformer.resblocks[block_idx].moe_adapter
        adapter.assert_descriptors_ready()
        values, labels = features["cls_in"].to(self.device), features["labels"].to(self.device)
        scores = adapter.descriptor_scores(values)
        class_scores = {
            int(class_id.item()): float(scores[labels.eq(class_id)].mean(dim=0).min().item())
            for class_id in torch.unique(labels, sorted=True)
        }
        expected = sorted(int(value) for value in class_index)
        if sorted(class_scores) != expected:
            raise RuntimeError(f"Block {block_idx} received classes {sorted(class_scores)}, expected {expected}.")
        uncovered = [class_id for class_id, score in class_scores.items() if score > threshold]
        for class_id, score in class_scores.items():
            state = "uncovered" if class_id in uncovered else "covered"
            print(f"=> [Coverage][B{block_idx}] class={class_id} score={score:.4f} {state}")
        decision = LayerExpansionDecision(block_idx, bool(uncovered), class_scores, uncovered)
        print(f"=> [Coverage][B{block_idx}] decision={'EXPAND' if decision.expand else 'REUSE'}")
        return decision

    def add_expert(self, block_idx, session_id):
        adapter = self.clip_model.visual.transformer.resblocks[int(block_idx)].moe_adapter
        reduction = int(getattr(self.cfg.TRAINER.BiMC.VISUAL_MOE, "INCREMENTAL_REDUCTION", 8))
        return adapter.add_expert(session_id=session_id, reduction=reduction)

    def export_moe_topology(self):
        return {str(index): block.moe_adapter.export_topology()
                for index, block in enumerate(self.clip_model.visual.transformer.resblocks)
                if getattr(block, "moe_adapter", None) is not None}

    def rebuild_moe_topology(self, topology):
        topology = topology or {}
        for index, block in enumerate(self.clip_model.visual.transformer.resblocks):
            adapter = getattr(block, "moe_adapter", None)
            if adapter is not None:
                adapter.rebuild_topology(topology.get(str(index), topology.get(index)))

    def refine_new_fused_prototypes(self, old_classifier_state, task_stat, loader):
        """Apply query-exclusive residual refinement to only the current classes."""
        refinement = getattr(self.cfg.TRAINER.BiMC, "PROTOTYPE_REFINEMENT", None)
        if refinement is None or not bool(getattr(refinement, "ENABLE", False)):
            return task_stat
        current_ids = self._state_class_ids(task_stat, self.device)
        if current_ids.numel() == 0:
            current_ids = torch.as_tensor(task_stat["class_index"], device=self.device, dtype=torch.long)
        # LOO caches use globally sorted ids.  Preserve row/id alignment even
        # for an externally supplied non-contiguous or reordered task list.
        task_stat = dict(task_stat)
        state_ids = self._state_class_ids(task_stat, self.device)
        row_matches = current_ids[:, None].eq(state_ids[None, :])
        if row_matches.shape != (current_ids.numel(), state_ids.numel()) or not row_matches.sum(dim=1).eq(1).all():
            raise ValueError("Current task prototype rows must map one-to-one to global class ids.")
        row_order = row_matches.to(dtype=torch.long).argmax(dim=1)
        for key in (
            "description_proto", "text_features", "image_proto",
            "initial_fused_proto", "fused_proto",
        ):
            value = task_stat.get(key)
            if value is not None and torch.as_tensor(value).ndim >= 1:
                value = torch.as_tensor(value)
                if value.size(0) == state_ids.numel():
                    task_stat[key] = value[row_order.to(value.device)]
        task_stat["class_ids"] = current_ids
        task_stat["class_index"] = current_ids
        initial_global = task_stat.get("fused_proto")
        if initial_global is None:
            initial_global = self.fuse_prototypes(task_stat)
        cache = self.build_leave_one_out_prototypes(loader, current_ids.detach().cpu().tolist())
        cache["text_features"] = task_stat["text_features"].detach()
        cache["description_proto"] = task_stat["description_proto"].detach()
        lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        per_query = loo_fused_prototypes_for_queries(
            cache, cache["source_ids"], old_classifier_state or {},
            beta=self.cfg.DATASET.BETA, lambda_t=lambda_t,
        ).to(self.device)
        old_fused = self.fuse_prototypes(old_classifier_state)
        if old_fused is None:
            old_fused = per_query.new_empty((0, per_query.size(-1)))
        old_ids = self._state_class_ids(old_classifier_state, self.device)
        loo_current = per_query[:, old_fused.size(0):, :]
        refined, refinement_metrics = optimize_query_exclusive_fused_prototypes(
            loo_current,
            cache["features"].to(self.device), cache["labels"].to(self.device), current_ids,
            history_fused=old_fused, history_class_ids=old_ids,
            epochs=getattr(refinement, "EPOCHS", 30),
            lr=getattr(refinement, "LR", 1e-2),
            old_protect_weight=getattr(refinement, "OLD_PROTECT_WEIGHT", 1.0),
            anchor_weight=getattr(refinement, "ANCHOR_WEIGHT", 1.0),
            old_protect_margin=getattr(refinement, "OLD_PROTECT_MARGIN", 0.70),
            initial_global=initial_global,
            return_metrics=True,
        )
        refined = refined.to(task_stat["image_proto"].dtype)
        task_stat["initial_fused_proto"] = F.normalize(
            torch.as_tensor(initial_global, device=refined.device, dtype=refined.dtype), dim=-1
        ).detach()
        task_stat["fused_proto"] = refined
        task_stat["prototype_refinement"] = refinement_metrics
        print(
            "=> [PrototypeRefine] LOO acc {:.3f}->{:.3f}, margin {:.5f}->{:.5f}, selected_epoch={}".format(
                refinement_metrics["pre_loo_accuracy"], refinement_metrics["post_loo_accuracy"],
                refinement_metrics["pre_loo_margin"], refinement_metrics["post_loo_margin"],
                refinement_metrics["selected_epoch"],
            )
        )
        # ``build_task_statistics`` updates the legacy conflict cache before
        # this residual refinement runs.  Keep that cache coherent as well so
        # an optional non-zero legacy conflict setting never reintroduces the
        # pre-refinement current prototypes in the following session.
        history = getattr(self, "_fused_history_prototypes", None)
        history_ids = getattr(self, "_fused_history_class_ids", None)
        if history is None or history_ids is None:
            self._fused_history_prototypes = refined.detach()
            self._fused_history_class_ids = current_ids.detach()
        else:
            history_ids = history_ids.to(current_ids.device)
            keep = (history_ids[:, None] != current_ids[None, :]).all(dim=1)
            self._fused_history_prototypes = torch.cat([history[keep].to(refined.device), refined.detach()], dim=0)
            self._fused_history_class_ids = torch.cat([history_ids[keep], current_ids.detach()], dim=0)
        return task_stat

    @torch.no_grad()
    def restore_fused_history(self, state_dict_list):
        """Restore the optional legacy fused-history cache after checkpoint load."""
        if not state_dict_list:
            self._fused_history_prototypes = None
            self._fused_history_class_ids = None
            return
        fused, ids = [], []
        for state in state_dict_list:
            fused.append(self.fuse_prototypes(state).detach())
            ids.append(self._state_class_ids(state, self.device))
        self._fused_history_prototypes = torch.cat(fused, dim=0)
        self._fused_history_class_ids = torch.cat(ids, dim=0)

    @torch.no_grad()
    def extract_teacher_features(self, images):
        images = images.to(self.device)
        for block in self.clip_model.visual.transformer.resblocks:
            if hasattr(block, 'moe_adapter') and block.moe_adapter is not None:
                block._temp_moe = block.moe_adapter
                block.moe_adapter = None
        teacher_features = self.clip_model.encode_image(images)
        for block in self.clip_model.visual.transformer.resblocks:
            if hasattr(block, '_temp_moe'):
                block.moe_adapter = block._temp_moe
                del block._temp_moe
        return teacher_features

    def extract_img_feature_train(self, images, active_counts=None):
        return self.clip_model.encode_image(
            images.to(self.device), return_moe_aux=True, active_counts=active_counts
        )

    def forward_train(self, images, task_stat, beta, compute_teacher=False):
        img_feat, moe_aux = self.extract_img_feature_train(images)
        teacher_feat = self.extract_teacher_features(images) if compute_teacher else None
        img_feat_norm = F.normalize(img_feat, dim=-1)

        text_features = task_stat["text_features"].to(self.device)
        description_proto = task_stat["description_proto"].to(self.device)
        image_proto = task_stat["image_proto"].to(self.device)

        lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        calibrated_text_proto = F.normalize((1 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
        fused_proto = task_stat.get("fused_proto")
        if fused_proto is None:
            fused_proto = F.normalize(beta * calibrated_text_proto + (1 - beta) * image_proto, dim=-1)
        else:
            fused_proto = fused_proto.to(self.device)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * img_feat_norm @ fused_proto.t()
        return logits, img_feat, teacher_feat, moe_aux

    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        clip_weights, all_targets = [], []
        k = cls_begin_index
        for classname in class_names:
            targets = torch.full((len(template),), k, device=self.device)
            all_targets.append(targets)
            k += 1
            classname = classname.replace("_", " ").replace("-", " ")
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(self.device)
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings.mean(dim=0) / class_embeddings.mean(dim=0).norm())
        return F.normalize(torch.stack(clip_weights, dim=0), dim=-1), torch.cat(all_targets, dim=0)

    @torch.no_grad()
    def inference_all_img_feature(self, loader, active_counts=None):
        all_features, all_labels = [], []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = F.normalize(
                self.clip_model.encode_image(images, active_counts=active_counts), dim=-1
            )
            all_features.append(features)
            all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        prototypes = [all_features[all_labels == c].mean(dim=0) for c in torch.unique(all_labels)]
        return all_features, all_labels, F.normalize(torch.stack(prototypes, dim=0), dim=-1)

    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        description_embeddings, mean_embeddings, all_targets = [], [], []
        with open(gpt_path, "r", encoding="utf-8") as file:
            gpt_prompt_dict = {k.lower().replace("_", " "): v for k, v in json.load(file).items()}
        k = cls_begin_index
        for single_key in class_names:
            single_class_prompts = gpt_prompt_dict[single_key.lower().replace("_", " ")]
            all_targets.append(torch.full((len(single_class_prompts),), k, device=self.device))
            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts]).to(self.device)
            text_features = self.clip_model.encode_text(x_tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0, keepdim=True))
            description_embeddings.append(text_features)
        return torch.cat(description_embeddings, dim=0), torch.cat(all_targets, dim=0), F.normalize(torch.cat(mean_embeddings, dim=0), dim=-1)

    def soft_calibration(self, base_protos, cur_protos):
        shift_weight, tau = self.cfg.TRAINER.BiMC.LAMBDA_I, self.cfg.TRAINER.BiMC.TAU
        base_protos, cur_protos = F.normalize(base_protos, p=2, dim=-1), F.normalize(cur_protos, p=2, dim=-1)
        norm_weights = torch.softmax(torch.mm(cur_protos, base_protos.T) * tau, dim=1)
        delta_protos = F.normalize(torch.matmul(norm_weights, base_protos), p=2, dim=-1)
        return F.normalize((1 - shift_weight) * cur_protos + shift_weight * delta_protos, dim=-1)

    @torch.no_grad()
    def build_task_statistics(self, class_names, loader, class_index, calibrate_novel_vision_proto=False):
        cls_begin_index = class_index[0]
        text_features, _ = self.inference_text_feature(class_names, self.template, cls_begin_index)
        _, _, description_proto = self.inference_all_description_feature(class_names, self.cfg.DATASET.GPT_PATH, cls_begin_index)
        images_features, _, images_proto = self.inference_all_img_feature(loader)
        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
        else:
            self.base_vision_prototype = images_proto
        lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
        calibrated_text = F.normalize((1 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
        raw_fused = F.normalize(self.cfg.DATASET.BETA * calibrated_text + (1 - self.cfg.DATASET.BETA) * images_proto, dim=-1)
        history_fused = getattr(self, "_fused_history_prototypes", None)
        history_ids = getattr(self, "_fused_history_class_ids", None)
        strength = float(getattr(self.cfg.TRAINER.BiMC, "FUSED_CONFLICT_STRENGTH", 0.0))
        margin = float(getattr(self.cfg.TRAINER.BiMC, "FUSED_CONFLICT_MARGIN", 0.05))
        fused_proto = refine_conflict_aware_fused_prototypes(
            raw_fused, torch.as_tensor(class_index), history_fused, history_ids, strength, margin
        )
        # Re-express refined fusion as a visual prototype so legacy engine
        # merge/forward paths (which know only image/text/descriptor keys) use it.
        if self.cfg.DATASET.BETA < 1.0:
            images_proto = F.normalize((fused_proto - self.cfg.DATASET.BETA * calibrated_text) /
                                       max(1e-6, 1.0 - self.cfg.DATASET.BETA), dim=-1)
        ids = torch.as_tensor(class_index, device=fused_proto.device, dtype=torch.long)
        if history_fused is None:
            self._fused_history_prototypes, self._fused_history_class_ids = fused_proto.detach(), ids.detach()
        else:
            keep = (history_ids[:, None].to(ids.device) != ids[None, :]).all(dim=1)
            self._fused_history_prototypes = torch.cat([history_fused[keep], fused_proto.detach()], dim=0)
            self._fused_history_class_ids = torch.cat([history_ids[keep].to(ids.device), ids.detach()], dim=0)
        return {
            "description_proto": description_proto,
            "text_features": text_features,
            "image_proto": images_proto,
            "class_index": class_index,
            "class_ids": ids,
            "sample_cnt": len(images_features),
            "initial_fused_proto": raw_fused,
            "fused_proto": fused_proto,
        }

    @torch.no_grad()
    def compute_text_state(self, class_names, class_index):
        """S6 is topology-independent; reuse one text bank across CV folds."""
        ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
        # Existing text utilities expect the task's contiguous starting index.
        if ids.numel() == 0 or not torch.equal(ids, torch.arange(
                int(ids[0]), int(ids[0]) + ids.numel(), device=self.device)):
            raise ValueError("text-state class ids must be consecutive in task order")
        text, _ = self.inference_text_feature(class_names, self.template, int(ids[0]))
        _, _, descriptions = self.inference_all_description_feature(
            class_names, self.cfg.DATASET.GPT_PATH, int(ids[0])
        )
        text_lambda = (float(self.cfg.TRAINER.BiMC.LAMBDA_T)
                       if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
        calibrated = F.normalize((1.0 - text_lambda) * text + text_lambda * descriptions, dim=-1)
        return {"class_ids": ids, "text_features": text,
                "description_proto": descriptions, "calibrated_text": calibrated}

    @torch.no_grad()
    def build_versioned_task_state(self, loader, class_index, text_state, topology,
                                   reference_topology, visual_calibration=True):
        """S5-S7: maintain reference and owner-topology prototypes separately.

        Never reverse-engineer a visual prototype from a normalized fused vector:
        the latter loses its original norm. All saved prototypes have explicit
        feature-space ownership, so old samples need not be replayed.
        """
        topology = self.canonical_topology(topology)
        reference_topology = self.canonical_topology(reference_topology)
        ids = torch.as_tensor(class_index, device=self.device, dtype=torch.long)
        if not torch.equal(ids, text_state["class_ids"].to(self.device)):
            raise ValueError("text/prototype class ids disagree")
        _, own_labels, visual = self.inference_all_img_feature(loader, active_counts=topology)
        if topology == reference_topology:
            reference_raw = visual
        else:
            _, reference_labels, reference_raw = self.inference_all_img_feature(
                loader, active_counts=reference_topology
            )
            if not torch.equal(own_labels, reference_labels):
                raise ValueError("reference and owner class orders differ")
        if int(ids[0]) == 0:
            self.base_vision_prototype = reference_raw.detach().clone()
        reference_visual = reference_raw
        if int(ids[0]) != 0 and visual_calibration:
            if self.base_vision_prototype is None:
                raise RuntimeError("base reference prototypes must exist before an increment")
            reference_visual = self.soft_calibration(self.base_vision_prototype, reference_raw)
        text = text_state["calibrated_text"]
        beta = float(self.cfg.DATASET.BETA)
        ref_fused = F.normalize(beta * text + (1.0 - beta) * reference_visual, dim=-1)
        dynamic_fused = F.normalize(beta * text + (1.0 - beta) * visual, dim=-1)
        return {
            "class_ids": ids, "class_index": ids.detach().cpu().tolist(),
            "text_features": text_state["text_features"].detach().clone(),
            "description_proto": text_state["description_proto"].detach().clone(),
            "image_proto": visual.detach().clone(),
            "fused_proto": dynamic_fused.detach().clone(),
            "ref_image_proto": reference_visual.detach().clone(),
            "ref_fused_proto": ref_fused.detach().clone(),
            "topology": dict(topology),
        }

    @torch.no_grad()
    def forward_ours(self, images, image_proto, description_proto, text_features, beta, fused_proto=None):
        img_feat = F.normalize(self.extract_img_feature(images), dim=-1)
        if fused_proto is None:
            lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0
            calibrated_text_proto = F.normalize((1 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
            fused_proto = F.normalize(beta * calibrated_text_proto + (1 - beta) * image_proto, dim=-1)
        else:
            fused_proto = F.normalize(fused_proto.to(self.device), dim=-1)
        return 100.0 * img_feat @ fused_proto.t()

    @torch.no_grad()
    def extract_img_feature(self, images):
        return self.clip_model.encode_image(images.to(self.device))

    def parse_batch(self, batch):
        # NumPy-backed FSCIL datasets can yield int32 labels on Windows.  Keep
        # targets in PyTorch's required class-index dtype for CE and indexing.
        return batch["image"].to(self.device), batch["label"].to(self.device, dtype=torch.long)
