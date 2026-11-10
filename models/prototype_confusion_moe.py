"""Prototype-Confusion Guided Dynamic MoE-FSCIL.

Frozen CLIP ViT feature stream + parallel residual experts.  APC proposes
where capacity may be needed; only a held-out all-seen-class risk transaction
can commit an expert.  All experts in a layer share one Top-k Softmax router.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.clip.clip as clip


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def class_means(features, labels, class_ids):
    features = normalize(features)
    labels = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    class_ids = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
    means, counts = [], []
    for cid in class_ids:
        values = features[labels.eq(cid)]
        if values.numel() == 0:
            raise ValueError("No support features for class {}".format(int(cid)))
        means.append(normalize(values.mean(0)))
        counts.append(values.size(0))
    return torch.stack(means), torch.tensor(counts, device=features.device)


def source_exclusive_means(features, labels, source_ids, class_ids, query_indices=None):
    """Build query-specific class means after removing every view of its source.

    Class sufficient statistics are computed once. For each queried source,
    its per-class contribution is subtracted in one vectorized operation;
    this avoids materializing an N-by-C-by-D bank for the full support set.
    """
    features = normalize(features)
    labels = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    sources = torch.as_tensor(source_ids, device=features.device).reshape(-1)
    classes = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
    if features.size(0) != labels.numel() or labels.numel() != sources.numel():
        raise ValueError("Features, labels and source ids must have equal lengths.")
    if classes.numel() == 0 or classes.unique().numel() != classes.numel():
        raise ValueError("Class ids must be non-empty and unique.")
    membership = labels[:, None].eq(classes[None, :])
    if not membership.sum(1).eq(1).all():
        raise ValueError("Every support label must map to exactly one requested class.")
    class_index = membership.long().argmax(1)
    class_count = torch.bincount(class_index, minlength=classes.numel())
    class_sum = features.new_zeros(classes.numel(), features.size(-1))
    class_sum.index_add_(0, class_index, features)
    if query_indices is None:
        query_indices = torch.arange(features.size(0), device=features.device)
    else:
        query_indices = torch.as_tensor(query_indices, device=features.device, dtype=torch.long)
    if query_indices.numel() == 0:
        return features.new_empty((0, classes.numel(), features.size(-1)))
    query_sources = sources[query_indices]
    unique_sources, inverse = torch.unique(query_sources, sorted=False, return_inverse=True)
    banks_by_source = []
    for source in unique_sources:
        source_mask = sources.eq(source)
        source_indices = class_index[source_mask]
        source_sum = features.new_zeros(classes.numel(), features.size(-1))
        source_sum.index_add_(0, source_indices, features[source_mask])
        source_count = torch.bincount(source_indices, minlength=classes.numel())
        remaining = class_count - source_count
        if remaining.le(0).any():
            missing = classes[remaining.le(0)].tolist()
            raise ValueError("LOO bank empty for source {} in classes {}".format(int(source.item()), missing))
        means = (class_sum - source_sum) / remaining.to(features.dtype).unsqueeze(1)
        banks_by_source.append(normalize(means))
    return torch.stack(banks_by_source)[inverse]


def _cardinality_matched_null_maxima(pairwise_similarity, competitor_count,
                                          resamples=512, seed=0):
    """Empirical null maxima for exactly K competitors, sampled from base classes."""
    pairwise = torch.as_tensor(pairwise_similarity, dtype=torch.float32).cpu()
    if pairwise.ndim != 2 or pairwise.size(0) != pairwise.size(1):
        raise ValueError("APC pairwise reference must be a square matrix.")
    base_count = int(pairwise.size(0))
    if base_count < 3 or competitor_count < 1 or resamples < 32:
        raise ValueError("APC requires >=3 base classes, K>=1 and >=32 resamples.")
    if not torch.isfinite(pairwise).all():
        raise ValueError("APC pairwise reference contains non-finite values.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729 * int(competitor_count))
    maxima = []
    for anchor in range(base_count):
        competitors = torch.cat((pairwise[anchor, :anchor], pairwise[anchor, anchor + 1:]))
        if competitor_count <= base_count - 1:
            keys = torch.rand(int(resamples), base_count - 1, generator=generator)
            indices = keys.argsort(dim=1)[:, :int(competitor_count)]
        else:
            indices = torch.randint(0, base_count - 1,
                (int(resamples), int(competitor_count)), generator=generator)
        maxima.append(competitors[indices].max(dim=1).values)
    return torch.cat(maxima, dim=0).sort().values


def prototype_confusion(layer_features, labels, new_class_ids,
                        historical_layer_prototypes, apc_reference):
    """Class/layer bootstrap tail scores at matched competitor cardinality.

    Bootstrap repetitions approximate the empirical reference distribution; they
    are not independent class observations and are not inferential p-values.
    """
    output = {}
    for layer, values in layer_features.items():
        new_proto, _ = class_means(values, labels, new_class_ids)
        history = normalize(historical_layer_prototypes[int(layer)])
        if history.ndim != 2 or history.size(0) == 0:
            raise ValueError("APC requires non-empty historical class prototypes.")
        reference = apc_reference[int(layer)]
        if not isinstance(reference, dict) or not {
            "pairwise_similarity", "base_class_count"
        }.issubset(reference):
            raise ValueError("APC reference is incompatible; rebuild it from Session 0.")
        pairwise = torch.as_tensor(reference["pairwise_similarity"]).float().cpu()
        base_count = int(reference["base_class_count"])
        if pairwise.shape != (base_count, base_count) or base_count < 3:
            raise ValueError("APC pairwise reference shape/count is inconsistent.")
        similarities = new_proto @ history.t()
        if new_proto.size(0) > 1:
            within = new_proto @ new_proto.t()
            within.fill_diagonal_(float("-inf"))
            similarities = torch.cat([similarities, within], dim=1)
        nearest = similarities.max(dim=1).values.detach().float().cpu()
        k = int(history.size(0) + new_proto.size(0) - 1)
        null_maxima = _cardinality_matched_null_maxima(
            pairwise, k, resamples=512, seed=1701 + int(layer))
        p_values = torch.stack([
            (1.0 + (null_maxima >= score).sum().float()) /
            (1.0 + float(null_maxima.numel())) for score in nearest
        ])
        scores = 1.0 - p_values
        output[int(layer)] = {
            "class_ids": [int(cid) for cid in new_class_ids],
            "score_by_class": {int(cid): float(scores[i].item())
                               for i, cid in enumerate(new_class_ids)},
            "tail_fraction_by_class": {int(cid): float(p_values[i].item())
                                 for i, cid in enumerate(new_class_ids)},
            "scores": scores, "p_values": p_values,
            "mean": float(scores.mean().item()),
            "p90": float(torch.quantile(scores, 0.90).item()),
            "nearest_similarity": nearest,
            "null_max_mean": float(null_maxima.mean().item()),
            "null_max_std": float(null_maxima.std(unbiased=False).item()),
            "comparison_count": k, "null_sample_count": int(null_maxima.numel()),
        }
    return output


def make_apc_reference(layer_prototypes):
    """Freeze base-class pairwise geometry for per-session cardinality matching."""
    reference = {}
    for layer, prototypes in layer_prototypes.items():
        p = normalize(prototypes)
        if p.ndim != 2 or p.size(0) < 3:
            raise ValueError("APC base reference requires at least three class prototypes.")
        reference[int(layer)] = {
            "pairwise_similarity": (p @ p.t()).clamp(-1.0, 1.0).detach().cpu(),
            "base_class_count": int(p.size(0)),
        }
    return reference

def balanced_risk(logits, labels, class_ids):
    """Strict macro NLL: every requested class must be represented exactly in the bank."""
    logits = torch.as_tensor(logits).float()
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long).reshape(-1)
    class_ids = torch.as_tensor(class_ids, device=logits.device, dtype=torch.long).reshape(-1)
    if logits.ndim != 2 or logits.size(0) != labels.numel():
        raise ValueError("Risk logits must be [N,C] and align with query labels.")
    if class_ids.numel() == 0 or class_ids.unique().numel() != class_ids.numel():
        raise ValueError("Risk class ids must be non-empty and unique.")
    targets = labels[:, None].eq(class_ids[None, :])
    if not targets.sum(1).eq(1).all():
        raise ValueError("Risk classifier must contain every query label exactly once.")
    counts = targets.sum(0)
    if counts.eq(0).any():
        missing = class_ids[counts.eq(0)].tolist()
        raise ValueError("Balanced risk requires at least one query for every class: {}".format(missing))
    columns = targets.long().argmax(1)
    losses = F.cross_entropy(logits, columns, reduction="none")
    class_losses = torch.stack([
        losses[labels.eq(cid)].mean() for cid in class_ids
    ])
    return class_losses.mean(), {
        int(cid): float(class_losses[index].item())
        for index, cid in enumerate(class_ids)
    }


class BottleneckExpert(nn.Module):
    def __init__(self, input_dim, output_dim, reduction, zero_up=False):
        super().__init__()
        hidden = max(4, int(input_dim) // max(1, int(reduction)))
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, output_dim)
        if zero_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(self.norm(x))))


class UnifiedTopKRouter(nn.Module):
    """One append-only router; all old and new experts share the same Top-k Softmax."""
    def __init__(self, input_dim, count):
        super().__init__()
        self.input_dim = int(input_dim)
        self.columns = nn.ModuleList()
        for _ in range(int(count)):
            self.add_column()

    def add_column(self):
        column = nn.Linear(self.input_dim, 1)
        nn.init.normal_(column.weight, std=0.02)
        nn.init.zeros_(column.bias)
        self.columns.append(column)
        return self.columns[-1]

    def logits(self, x):
        if x.ndim != 2:
            raise ValueError("Unified router input must have shape [N,D].")
        if x.size(0) == 0:
            raise ValueError("Unified router does not accept an empty batch.")
        if not self.columns:
            raise RuntimeError("Unified router has no experts.")
        return torch.cat([column(x) for column in self.columns], dim=-1)

    def forward(self, x, top_k, excluded=()):
        logits = self.logits(x)
        k = int(top_k)
        if k < 1:
            raise ValueError("Top-k must be a positive integer.")
        masked = logits.clone()
        if excluded:
            excluded_ids = [int(i) for i in excluded]
            if len(set(excluded_ids)) != len(excluded_ids):
                raise ValueError("Excluded expert indices must be unique.")
            if any(i < 0 or i >= len(self.columns) for i in excluded_ids):
                raise IndexError("Excluded expert index is outside the router topology.")
            masked[:, excluded_ids] = float("-inf")
        available = torch.isfinite(masked).sum(dim=1)
        if not available.eq(available[0]).all():
            raise RuntimeError("Router availability must be consistent across a batch.")
        if int(available[0].item()) < k:
            raise ValueError(
                "Top-k={} exceeds the {} experts available after exclusions.".format(
                    k, int(available[0].item())
                )
            )
        selected_logits, selected_ids = masked.topk(k, dim=-1)
        weights = torch.softmax(selected_logits, dim=-1)
        dense = torch.zeros_like(logits).scatter(1, selected_ids, weights)
        return logits, dense, selected_ids


class DynamicLayerMoE(nn.Module):
    def __init__(self, input_dim, output_dim, base_experts, base_reduction,
                 incremental_reduction, top_k, scale_init):
        super().__init__()
        self.input_dim, self.output_dim = int(input_dim), int(output_dim)
        self.base_expert_count = int(base_experts)
        self.incremental_reduction = int(incremental_reduction)
        self.top_k = int(top_k)
        if self.base_expert_count < 1:
            raise ValueError("At least one base expert is required.")
        if self.top_k < 1 or self.top_k > self.base_expert_count:
            raise ValueError("Top-k must be between one and the number of base experts.")
        self.norm = nn.LayerNorm(input_dim)
        self.experts = nn.ModuleList([
            BottleneckExpert(input_dim, output_dim, base_reduction)
            for _ in range(self.base_expert_count)
        ])
        self.router = UnifiedTopKRouter(input_dim, self.base_expert_count)
        self.birth_sessions = [0] * self.base_expert_count

    @property
    def num_experts(self):
        return len(self.experts)

    def forward(self, features, excluded=()):
        x = self.norm(features.float())
        logits, weights, selected = self.router(x, self.top_k, excluded=excluded)
        output = x.new_zeros(x.size(0), self.output_dim)
        # Evaluate only selected experts.  Each expert has one globally shared
        # route column; no task-specific router or Sigmoid side path exists.
        for expert_id, expert in enumerate(self.experts):
            mask = weights[:, expert_id].gt(0)
            if mask.any():
                output[mask] += expert(x[mask]) * weights[mask, expert_id:expert_id + 1]
        return output, {
            "router_logits": logits, "route_weights": weights,
            "selected_experts": selected,
        }

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.eval()

    def add_expert(self, session_id, new_features, historical_features, route_margin=0.05):
        """Append a zero-output expert without changing any historical support route.

        The new gate starts strictly below the minimum historical Top-k boundary.
        Its trainable margin objective must then separate new support from the
        complete historical feature memory; no class/task mask is used at inference.
        """
        new_features = torch.as_tensor(new_features, device=self.norm.weight.device).float()
        historical_features = torch.as_tensor(
            historical_features, device=self.norm.weight.device
        ).float()
        margin = float(route_margin)
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("Routing margin must be finite and positive.")
        if new_features.ndim != 2 or new_features.size(0) == 0:
            raise ValueError("A new expert requires non-empty [N,D] new-class features.")
        if historical_features.ndim != 2 or historical_features.size(0) == 0:
            raise ValueError("A new expert requires non-empty [N,D] historical features.")
        if new_features.size(1) != self.input_dim or historical_features.size(1) != self.input_dim:
            raise ValueError("Expert initialization features have an incompatible width.")

        self.freeze_all()
        device = self.norm.weight.device
        expert = BottleneckExpert(
            self.input_dim, self.output_dim, self.incremental_reduction, zero_up=True
        ).to(device=device)
        self.experts.append(expert)
        column = self.router.add_column().to(device=device)
        self.birth_sessions.append(int(session_id))

        with torch.no_grad():
            old_x = self.norm(historical_features)
            old_logits = self.router.logits(old_x)[:, :-1]
            if old_logits.size(1) < self.top_k:
                raise RuntimeError("Historical expert pool cannot satisfy the configured Top-k.")
            old_boundary = old_logits.topk(self.top_k, dim=1).values[:, -1]
            # A constant gate below every stored historical boundary guarantees
            # exact old Top-k membership and Softmax weights at initialization.
            column.weight.zero_()
            column.bias.copy_(old_boundary.min() - margin)

        self.set_newest_trainable()
        return self.num_experts - 1

    def route_margin_loss(self, new_features, historical_features, margin=0.05):
        """Learn a new-expert decision boundary while protecting historical routes.

        New samples must cross the old-pool Top-k boundary; historical samples
        must remain below it. Since old router columns are frozen, these hinge
        terms provide a direct gradient to the new column even before the new
        expert is selected by the sparse forward pass.
        """
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Routing margin loss requires an incremental expert.")
        margin = float(margin)
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("Routing margin must be finite and positive.")
        device = self.norm.weight.device
        new_x = self.norm(torch.as_tensor(new_features, device=device).float())
        old_x = self.norm(torch.as_tensor(historical_features, device=device).float())
        if new_x.ndim != 2 or new_x.size(0) == 0:
            raise ValueError("New-route supervision requires non-empty new features.")
        if old_x.ndim != 2 or old_x.size(0) == 0:
            raise ValueError("Historical-route protection requires non-empty old features.")

        new_logits = self.router.logits(new_x)
        old_logits = self.router.logits(old_x)
        new_boundary = new_logits[:, :-1].topk(self.top_k, dim=1).values[:, -1].detach()
        old_boundary = old_logits[:, :-1].topk(self.top_k, dim=1).values[:, -1].detach()
        new_score = new_logits[:, -1]
        old_score = old_logits[:, -1]
        new_loss = F.relu(new_boundary + margin - new_score).mean()
        historical_loss = F.relu(old_score - old_boundary + margin).mean()
        return {
            "new": new_loss,
            "historical": historical_loss,
            "new_activation_rate": (new_score > new_boundary).float().mean().detach(),
            "historical_intrusion_rate": (old_score > old_boundary).float().mean().detach(),
        }

    @torch.no_grad()
    def route_drift(self, features):
        """Compare current routes with the exact old-pool Top-k Softmax routes."""
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Route drift requires an incremental expert.")
        x = self.norm(torch.as_tensor(features, device=self.norm.weight.device).float())
        logits, current_weights, _ = self.router(x, self.top_k)
        old_logits = logits[:, :-1]
        old_values, old_ids = old_logits.topk(self.top_k, dim=1)
        old_weights = torch.softmax(old_values, dim=1)
        old_reference = torch.zeros_like(old_logits).scatter(1, old_ids, old_weights)
        drift = (current_weights[:, :-1] - old_reference).abs().sum(dim=1)
        drift = drift + current_weights[:, -1]
        return {
            "drift": drift,
            "new_expert_weight": current_weights[:, -1],
            "new_expert_selected": current_weights[:, -1].gt(0),
        }

    def set_newest_trainable(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("No incremental expert exists.")
        self.freeze_all()
        for p in self.experts[-1].parameters():
            p.requires_grad_(True)
        for p in self.router.columns[-1].parameters():
            p.requires_grad_(True)
        self.train()

    def newest_parameters(self):
        return list(self.experts[-1].parameters()) + list(self.router.columns[-1].parameters())

    def remove_newest(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Cannot remove a base expert.")
        del self.experts[-1]
        del self.router.columns[-1]
        self.birth_sessions.pop()
        self.freeze_all()

    def topology(self):
        return {
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "base_expert_count": self.base_expert_count, "top_k": self.top_k,
            "routing_version": "global_topk_stability_margin_v1",
            "experts": [
                {"reduction": max(1, self.input_dim // expert.down.out_features),
                 "birth_session": self.birth_sessions[i]}
                for i, expert in enumerate(self.experts)
            ],
        }

    def rebuild(self, spec):
        if not spec:
            return
        if (int(spec["input_dim"]) != self.input_dim or
                int(spec["output_dim"]) != self.output_dim or
                int(spec["base_expert_count"]) != self.base_expert_count or
                int(spec["top_k"]) != self.top_k or
                spec.get("routing_version") != "global_topk_stability_margin_v1"):
            raise ValueError("Checkpoint MoE topology is incompatible with this configuration.")
        self.experts = nn.ModuleList()
        self.router = UnifiedTopKRouter(self.input_dim, 0)
        self.birth_sessions = []
        for i, item in enumerate(spec["experts"]):
            self.experts.append(BottleneckExpert(
                self.input_dim, self.output_dim, int(item["reduction"]),
                zero_up=(i >= self.base_expert_count)
            ))
            self.router.add_column()
            self.birth_sessions.append(int(item["birth_session"]))
        self.to(self.norm.weight.device)
        self.freeze_all()


class PrototypeConfusionMoE(nn.Module):
    def __init__(self, cfg, templates, device):
        super().__init__()
        self.cfg, self.device_name = cfg, str(device)
        backbone = cfg.MODEL.BACKBONE.NAME
        if backbone not in clip._MODELS:
            raise ValueError("PC-MoE requires a supported CLIP ViT backbone.")
        path = clip._download(clip._MODELS[backbone])
        try:
            scripted = torch.jit.load(path, map_location="cpu").eval()
            state = scripted.state_dict()
        except RuntimeError:
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(path, map_location="cpu")
        self.clip_model = clip.build_model(state, cfg=None).float().to(device).eval()
        if not hasattr(self.clip_model.visual, "transformer"):
            raise ValueError("PrototypeConfusionMoE requires a ViT visual encoder.")
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        cfg_moe = cfg.TRAINER.PCMOE.VISUAL_MOE
        blocks = len(self.clip_model.visual.transformer.resblocks)
        self.layer_ids = sorted({
            (int(i) + blocks) if int(i) < 0 else int(i)
            for i in cfg_moe.INSERT_BLOCKS
        })
        if not self.layer_ids or any(i < 0 or i >= blocks for i in self.layer_ids):
            raise ValueError("Invalid PC-MoE INSERT_BLOCKS.")
        width = self.clip_model.visual.transformer.width
        embed = self.clip_model.text_projection.shape[1]
        self.adapters = nn.ModuleDict({
            str(i): DynamicLayerMoE(
                width, embed, cfg_moe.NUM_BASE_EXPERTS, cfg_moe.BASE_REDUCTION,
                cfg_moe.INCREMENTAL_REDUCTION, cfg_moe.TOP_K, cfg_moe.BASE_SCALE_INIT
            ) for i in self.layer_ids
        })
        self.layer_scale = nn.ParameterDict({
            str(i): nn.Parameter(torch.tensor(float(cfg_moe.BASE_SCALE_INIT)))
            for i in self.layer_ids
        })
        cal = cfg.TRAINER.PCMOE.CALIBRATION
        self.register_buffer("temperature_visual", torch.tensor(float(cal.INITIAL_TEMPERATURE_VISUAL)))
        self.register_buffer("temperature_text", torch.tensor(float(cal.INITIAL_TEMPERATURE_TEXT)))
        self._captured = {}
        self._hooks = []
        for i in self.layer_ids:
            self._hooks.append(
                self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                    self._capture(i)
                )
            )
        self.freeze_all()

    def _capture(self, layer_id):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[int(layer_id)] = hidden[0].detach().float()
        return hook

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def _raw_and_layers(self, images):
        self._captured = {}
        with torch.no_grad():
            raw = normalize(self.clip_model.encode_image(images.to(
                device=self.clip_model.visual.conv1.weight.device,
                dtype=self.clip_model.dtype
            )))
        if set(self.layer_ids) != set(self._captured):
            raise RuntimeError("Frozen ViT capture hooks did not fire for every selected layer.")
        return raw, {i: self._captured[i] for i in self.layer_ids}

    def compose(self, raw, layer_features, return_aux=False, disabled_by_layer=None):
        enhanced = normalize(raw)
        aux = {}
        disabled_by_layer = disabled_by_layer or {}
        for i in self.layer_ids:
            residual, info = self.adapters[str(i)](
                layer_features[i], excluded=disabled_by_layer.get(i, disabled_by_layer.get(str(i), ()))
            )
            enhanced = enhanced + self.layer_scale[str(i)] * residual
            aux[i] = info
        enhanced = normalize(enhanced)
        return (enhanced, aux) if return_aux else enhanced

    def encode(self, images, return_aux=False):
        raw, layers = self._raw_and_layers(images)
        enhanced, aux = self.compose(raw, layers, True)
        return (raw, enhanced, layers, aux) if return_aux else (raw, enhanced, layers)

    @torch.no_grad()
    def text_features(self, class_names):
        templates = self.cfg.TRAINER.PCMOE.TEXT_TEMPLATES
        templates = list(templates) if templates else list(self._default_templates())
        outputs, consistency = [], []
        for name in class_names:
            name = str(name).replace("_", " ").replace("-", " ")
            tokens = clip.tokenize([t.format(name) for t in templates]).to(self.clip_model.token_embedding.weight.device)
            values = normalize(self.clip_model.encode_text(tokens))
            mean = values.mean(0)
            consistency.append(mean.norm().clamp(0.0, 1.0))
            outputs.append(normalize(mean))
        return torch.stack(outputs), torch.stack(consistency)

    @staticmethod
    def _default_templates():
        return ["a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}."]

    def set_base_trainable(self):
        for adapter in self.adapters.values():
            for p in adapter.parameters():
                p.requires_grad_(True)
        for p in self.layer_scale.values():
            p.requires_grad_(True)

    def set_incremental_trainable(self, layers):
        self.freeze_all()
        for layer in layers:
            self.adapters[str(layer)].set_newest_trainable()

    def freeze_all(self):
        for adapter in self.adapters.values():
            adapter.freeze_all()
        for p in self.layer_scale.values():
            p.requires_grad_(False)
            p.grad = None
        self.clip_model.eval()

    def add_expert(self, layer, session, new_features, historical_features, route_margin=0.05):
        return self.adapters[str(layer)].add_expert(
            session, new_features, historical_features, route_margin=route_margin
        )

    def topology(self):
        return {str(i): self.adapters[str(i)].topology() for i in self.layer_ids}

    def rebuild_topology(self, topology):
        for i in self.layer_ids:
            self.adapters[str(i)].rebuild(topology.get(str(i)))
        self.freeze_all()

    def set_temperatures(self, visual, text):
        cfg = self.cfg.TRAINER.PCMOE.CALIBRATION
        self.temperature_visual.fill_(max(cfg.MIN_TEMPERATURE, min(cfg.MAX_TEMPERATURE, float(visual))))
        self.temperature_text.fill_(max(cfg.MIN_TEMPERATURE, min(cfg.MAX_TEMPERATURE, float(text))))
