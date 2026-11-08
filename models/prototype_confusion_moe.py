"""Prototype-confusion guided dynamic MoE for strict FSCIL.

The CLIP ViT remains an immutable feature extractor. Layer experts operate on
captured hidden states in a parallel residual branch and never alter the ViT
token stream. All experts at a layer share one sparse Top-k Softmax router.
"""
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.clip.clip as clip


def _as_long(values, device):
    return torch.as_tensor(values, device=device, dtype=torch.long).reshape(-1)


def class_prototype_statistics(features, labels, class_ids):
    """Return normalized class means, concentration and support counts."""
    features = F.normalize(torch.as_tensor(features).float(), dim=-1)
    labels = _as_long(labels, features.device)
    class_ids = _as_long(class_ids, features.device)
    prototypes, concentration, counts = [], [], []
    for class_id in class_ids:
        selected = features[labels.eq(class_id)]
        if selected.numel() == 0:
            raise ValueError("A requested class has no support features: {}".format(int(class_id)))
        mean = selected.mean(dim=0)
        concentration.append(mean.norm().clamp(0.0, 1.0))
        prototypes.append(F.normalize(mean, dim=-1))
        counts.append(selected.size(0))
    return {
        "prototypes": torch.stack(prototypes),
        "concentration": torch.stack(concentration),
        "counts": torch.tensor(counts, device=features.device, dtype=torch.long),
    }


def build_leave_one_source_out_prototypes(features, labels, source_ids, class_ids):
    """Build per-query class prototype banks, excluding the entire query source."""
    features = F.normalize(torch.as_tensor(features).float(), dim=-1)
    labels = _as_long(labels, features.device)
    source_ids = torch.as_tensor(source_ids, device=features.device).reshape(-1)
    class_ids = _as_long(class_ids, features.device)
    if features.ndim != 2 or labels.numel() != features.size(0) or source_ids.numel() != features.size(0):
        raise ValueError("Features, labels and source ids must have matching leading dimensions.")
    rows = []
    for source_id in source_ids:
        class_rows = []
        keep_source = source_ids.ne(source_id)
        for class_id in class_ids:
            selected = features[keep_source & labels.eq(class_id)]
            if selected.numel() == 0:
                raise ValueError(
                    "Leave-one-source-out prototype is empty for class {}.".format(int(class_id))
                )
            class_rows.append(F.normalize(selected.mean(dim=0), dim=-1))
        rows.append(torch.stack(class_rows))
    return torch.stack(rows)


def build_apc_reference(layer_prototypes):
    """Base-session empirical distributions of nearest inter-class similarities."""
    reference = {}
    for layer, prototypes in layer_prototypes.items():
        prototypes = F.normalize(torch.as_tensor(prototypes).float(), dim=-1)
        if prototypes.size(0) < 2:
            raise ValueError("APC reference needs at least two base classes per layer.")
        similarities = prototypes @ prototypes.t()
        similarities.fill_diagonal_(float("-inf"))
        reference[str(layer)] = similarities.max(dim=1).values.sort().values.detach()
    return reference


def compute_prototype_confusion(layer_features, labels, new_class_ids,
                                history_layer_prototypes, reference_distributions):
    """Compute continuous layer-wise confusion percentiles; never hard-gate expansion."""
    result = {"layers": {}, "layer_order": []}
    labels = torch.as_tensor(labels, dtype=torch.long)
    new_class_ids = _as_long(new_class_ids, labels.device)
    for layer, features in layer_features.items():
        key = str(layer)
        current = class_prototype_statistics(features, labels, new_class_ids)["prototypes"]
        history = F.normalize(
            torch.as_tensor(history_layer_prototypes[key], device=current.device).float(), dim=-1
        )
        if history.numel() == 0:
            raise ValueError("Historical layer prototypes are required for incremental APC.")
        nearest_old = (current @ history.t()).max(dim=1).values
        nearest = nearest_old
        if current.size(0) > 1:
            within = current @ current.t()
            within.fill_diagonal_(float("-inf"))
            nearest = torch.maximum(nearest_old, within.max(dim=1).values)
        reference = torch.as_tensor(
            reference_distributions[key], device=current.device, dtype=torch.float32
        ).flatten().sort().values
        if reference.numel() == 0:
            percentiles = torch.ones_like(nearest)
        else:
            percentiles = torch.searchsorted(reference, nearest.contiguous(), right=True).float()
            percentiles = percentiles / float(reference.numel())
        class_rows = {}
        for index, class_id in enumerate(new_class_ids.tolist()):
            class_rows[int(class_id)] = {
                "nearest_similarity": float(nearest[index].item()),
                "confusion_percentile": float(percentiles[index].item()),
            }
        result["layers"][key] = {
            "mean_percentile": float(percentiles.mean().item()),
            "p90_percentile": float(torch.quantile(percentiles, 0.90).item()),
            "class_scores": class_rows,
        }
    result["layer_order"] = sorted(
        (int(key) for key in result["layers"]),
        key=lambda layer: result["layers"][str(layer)]["mean_percentile"],
        reverse=True,
    )
    return result


def fuse_modal_scores(raw_visual_features, enhanced_visual_features,
                      visual_prototypes, text_prototypes,
                      visual_reliability, text_reliability,
                      temperature_visual, temperature_text):
    """Single classifier decision: calibrated visual/text scores, then reliability fusion."""
    z0 = F.normalize(raw_visual_features.float(), dim=-1)
    zv = F.normalize(enhanced_visual_features.float(), dim=-1)
    visual_prototypes = F.normalize(visual_prototypes.float(), dim=-1)
    text_prototypes = F.normalize(text_prototypes.float(), dim=-1)
    if visual_prototypes.ndim == 2:
        visual_scores = zv @ visual_prototypes.t()
    elif visual_prototypes.ndim == 3:
        visual_scores = torch.einsum("bd,bcd->bc", zv, visual_prototypes)
    else:
        raise ValueError("Visual prototypes must have shape [C,D] or [B,C,D].")
    text_scores = z0 @ text_prototypes.t()
    tv = torch.as_tensor(temperature_visual, device=zv.device, dtype=zv.dtype).clamp_min(1e-4)
    tt = torch.as_tensor(temperature_text, device=zv.device, dtype=zv.dtype).clamp_min(1e-4)
    visual_logits = visual_scores / tv
    text_logits = text_scores / tt
    rv = torch.as_tensor(visual_reliability, device=zv.device, dtype=zv.dtype).clamp_min(0.0)
    rt = torch.as_tensor(text_reliability, device=zv.device, dtype=zv.dtype).clamp_min(0.0)
    alpha = rv / (rv + rt).clamp_min(1e-6)
    logits = alpha.unsqueeze(0) * visual_logits + (1.0 - alpha.unsqueeze(0)) * text_logits
    return logits, visual_logits, text_logits, alpha


class BottleneckExpert(nn.Module):
    def __init__(self, input_dim, output_dim, reduction=16, small_init=False):
        super().__init__()
        hidden = max(4, int(input_dim) // max(1, int(reduction)))
        self.norm = nn.LayerNorm(int(input_dim))
        self.down = nn.Linear(int(input_dim), hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, int(output_dim))
        if small_init:
            nn.init.normal_(self.up.weight, std=1e-3)
            nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return self.up(self.activation(self.down(self.norm(features))))


class UnifiedSparseRouter(nn.Module):
    """One append-only router; every old/new expert competes in one Top-k Softmax."""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.input_dim = int(input_dim)
        self.columns = nn.ModuleList()
        for _ in range(int(num_experts)):
            self.add_column()

    def add_column(self):
        column = nn.Linear(self.input_dim, 1)
        nn.init.normal_(column.weight, std=0.02)
        nn.init.zeros_(column.bias)
        self.columns.append(column)
        return len(self.columns) - 1

    def logits(self, features):
        if not self.columns:
            raise RuntimeError("The unified router has no experts.")
        return torch.cat([column(features) for column in self.columns], dim=-1)

    def forward(self, features, top_k=2, excluded=()):
        logits = self.logits(features)
        available = torch.ones(logits.size(1), device=logits.device, dtype=torch.bool)
        if excluded:
            excluded_ids = [int(value) for value in excluded]
            available[excluded_ids] = False
        if not bool(available.any().item()):
            raise RuntimeError("Router exclusion disabled every expert.")
        masked_logits = logits.masked_fill(~available.unsqueeze(0), float("-inf"))
        k = min(max(1, int(top_k)), int(available.sum().item()))
        selected = masked_logits.topk(k, dim=-1).indices
        selected_mask = torch.zeros_like(masked_logits, dtype=torch.bool).scatter_(1, selected, True)
        selected_mask = selected_mask & available.unsqueeze(0)
        weights = torch.softmax(masked_logits.masked_fill(~selected_mask, float("-inf")), dim=-1)
        return logits, weights


class DynamicVisualMoEAdapter(nn.Module):
    """Layer-local experts with a unified sparse router and append-only expansion."""
    def __init__(self, input_dim, output_dim, num_base_experts=4,
                 base_reduction=4, incremental_reduction=16, top_k=2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.base_expert_count = int(num_base_experts)
        self.incremental_reduction = int(incremental_reduction)
        self.top_k = int(top_k)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.experts = nn.ModuleList([
            BottleneckExpert(self.input_dim, self.output_dim, base_reduction)
            for _ in range(self.base_expert_count)
        ])
        self.router = UnifiedSparseRouter(self.input_dim, self.base_expert_count)
        self.birth_sessions = [0] * self.base_expert_count

    @property
    def num_experts(self):
        return len(self.experts)

    def route_weights(self, features, excluded=()):
        normalized = self.input_norm(features.float())
        return self.router(normalized, self.top_k, excluded)[1]

    @torch.no_grad()
    def add_expert(self, session_id, init_features, history_features):
        self.freeze_all()
        init_features = self.input_norm(torch.as_tensor(init_features, device=self.input_norm.weight.device).float())
        history_features = self.input_norm(torch.as_tensor(history_features, device=self.input_norm.weight.device).float())
        expert = BottleneckExpert(
            self.input_dim, self.output_dim, self.incremental_reduction, small_init=True
        ).to(device=self.input_norm.weight.device, dtype=self.input_norm.weight.dtype)
        self.experts.append(expert)
        self.router.add_column().to(device=self.input_norm.weight.device, dtype=self.input_norm.weight.dtype)
        self.birth_sessions.append(int(session_id))

        # Initialize the new column to enter Top-k on current support while
        # remaining below the old Top-k boundary on historical class anchors.
        column = self.router.columns[-1]
        if init_features.numel() and history_features.numel():
            mu_new = init_features.mean(dim=0)
            mu_old = history_features.mean(dim=0)
            direction = F.normalize(mu_new - mu_old, dim=0)
            old_current = self.router.logits(init_features)[:, :-1]
            old_history = self.router.logits(history_features)[:, :-1]
            k_current = min(self.top_k, old_current.size(1))
            k_history = min(self.top_k, old_history.size(1))
            threshold_current = old_current.topk(k_current, dim=-1).values[:, -1].mean()
            threshold_history = old_history.topk(k_history, dim=-1).values[:, -1].mean()
            separation = torch.dot(direction, mu_new - mu_old).clamp_min(1e-4)
            scale = ((threshold_current - threshold_history + 0.4) / separation).clamp(0.1, 10.0)
            column.weight.copy_((scale * direction).reshape(1, -1))
            column.bias.copy_(threshold_history - 0.2 - scale * torch.dot(direction, mu_old))
        self.set_newest_trainable()
        return self.num_experts - 1

    def forward(self, features, excluded=()):
        normalized = self.input_norm(features.float())
        logits, weights = self.router(normalized, self.top_k, excluded)
        expert_outputs = torch.stack([expert(normalized) for expert in self.experts], dim=1)
        contributions = expert_outputs * weights.unsqueeze(-1)
        residual = contributions.sum(dim=1)
        newest_contribution = contributions[:, -1].norm(dim=-1) if self.num_experts > self.base_expert_count else residual.new_zeros(residual.size(0))
        return residual, {
            "router_logits": logits,
            "route_weights": weights,
            "expert_outputs": expert_outputs,
            "newest_contribution_norm": newest_contribution,
        }

    def freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.eval()

    def set_newest_trainable(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("No incremental expert exists.")
        self.freeze_all()
        for parameter in self.experts[-1].parameters():
            parameter.requires_grad = True
        for parameter in self.router.columns[-1].parameters():
            parameter.requires_grad = True
        self.train()

    def newest_parameters(self):
        return list(self.experts[-1].parameters()) + list(self.router.columns[-1].parameters())

    def remove_newest_expert(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Cannot remove a base expert.")
        del self.experts[-1]
        del self.router.columns[-1]
        self.birth_sessions.pop()
        self.freeze_all()

    def export_topology(self):
        return {
            "base_expert_count": self.base_expert_count,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "experts": [
                {
                    "reduction": max(1, self.input_dim // expert.down.out_features),
                    "birth_session": int(self.birth_sessions[index]),
                }
                for index, expert in enumerate(self.experts)
            ],
        }

    def rebuild_topology(self, topology):
        if not topology:
            return
        if int(topology["input_dim"]) != self.input_dim or int(topology["output_dim"]) != self.output_dim:
            raise ValueError("Checkpoint expert dimensions do not match the configured backbone.")
        if int(topology["base_expert_count"]) != self.base_expert_count:
            raise ValueError("Checkpoint base expert count differs from configuration.")
        specs = list(topology["experts"])
        if len(specs) < self.base_expert_count:
            raise ValueError("Checkpoint topology is missing base experts.")
        self.experts = nn.ModuleList()
        self.router = UnifiedSparseRouter(self.input_dim, 0)
        self.birth_sessions = []
        for index, spec in enumerate(specs):
            reduction = int(spec["reduction"])
            self.experts.append(BottleneckExpert(
                self.input_dim, self.output_dim, reduction,
                small_init=(index >= self.base_expert_count),
            ))
            self.router.add_column()
            self.birth_sessions.append(int(spec["birth_session"]))
        self.freeze_all()


def _load_clip_model(cfg):
    backbone = cfg.MODEL.BACKBONE.NAME
    if backbone not in clip._MODELS:
        raise ValueError("Unsupported CLIP backbone: {}".format(backbone))
    path = clip._download(clip._MODELS[backbone])
    try:
        jit_model = torch.jit.load(path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state_dict = torch.load(path, map_location="cpu")
    model = clip.build_model(state_dict, cfg=None)
    model.float()
    return model.eval()


class PrototypeConfusionGuidedMoE(nn.Module):
    """Frozen CLIP plus a parallel, layer-wise dynamic residual MoE branch."""
    def __init__(self, cfg, template, device):
        super().__init__()
        self.cfg = cfg
        self.device_name = device
        self.template = list(template)
        self.clip_model = _load_clip_model(cfg).to(device)
        if not hasattr(self.clip_model.visual, "transformer"):
            raise ValueError("PrototypeConfusionGuidedMoE requires a ViT visual backbone.")
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        moe_cfg = cfg.TRAINER.PCMOE.VISUAL_MOE
        block_count = len(self.clip_model.visual.transformer.resblocks)
        self.layer_indices = sorted({
            (int(index) + block_count) if int(index) < 0 else int(index)
            for index in moe_cfg.INSERT_BLOCKS
        })
        if not self.layer_indices or any(index < 0 or index >= block_count for index in self.layer_indices):
            raise ValueError("VISUAL_MOE.INSERT_BLOCKS contains an invalid ViT block.")
        width = self.clip_model.visual.transformer.width
        embed_dim = self.clip_model.text_projection.shape[1]
        self.adapters = nn.ModuleDict({
            str(index): DynamicVisualMoEAdapter(
                width, embed_dim,
                num_base_experts=moe_cfg.NUM_BASE_EXPERTS,
                base_reduction=moe_cfg.BASE_REDUCTION,
                incremental_reduction=moe_cfg.INCREMENTAL_REDUCTION,
                top_k=moe_cfg.TOP_K,
            )
            for index in self.layer_indices
        })
        self.layer_scales = nn.ParameterDict({
            str(index): nn.Parameter(torch.tensor(float(moe_cfg.BASE_SCALE_INIT)))
            for index in self.layer_indices
        })
        self.register_buffer(
            "temperature_visual",
            torch.tensor(float(cfg.TRAINER.PCMOE.CALIBRATION.INITIAL_TEMPERATURE_VISUAL)),
        )
        self.register_buffer(
            "temperature_text",
            torch.tensor(float(cfg.TRAINER.PCMOE.CALIBRATION.INITIAL_TEMPERATURE_TEXT)),
        )
        self._captured_layer_features = {}
        self._hook_handles = []
        for index in self.layer_indices:
            block = self.clip_model.visual.transformer.resblocks[index]
            self._hook_handles.append(block.register_forward_hook(self._make_capture_hook(index)))
        self.freeze_all()

    def _make_capture_hook(self, layer_index):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise RuntimeError("ViT layer hook did not receive [tokens,batch,width] features.")
            self._captured_layer_features[int(layer_index)] = hidden[0].detach().float()
        return capture

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        if mode:
            for adapter in self.adapters.values():
                adapter.train()
        return self

    def _extract_frozen_features(self, images):
        self._captured_layer_features = {}
        images = images.to(device=self.clip_model.visual.conv1.weight.device, dtype=self.clip_model.dtype)
        with torch.no_grad():
            raw = self.clip_model.encode_image(images).float()
        missing = set(self.layer_indices) - set(self._captured_layer_features)
        if missing:
            raise RuntimeError("ViT feature hooks did not fire for blocks: {}".format(sorted(missing)))
        layers = {index: self._captured_layer_features[index] for index in self.layer_indices}
        return F.normalize(raw, dim=-1), layers

    def compose_visual_features(self, raw_visual_features, layer_features, disabled_by_layer=None,
                                return_aux=False):
        raw = F.normalize(torch.as_tensor(raw_visual_features).float(), dim=-1)
        enhanced = raw
        aux = {}
        disabled_by_layer = disabled_by_layer or {}
        for index in self.layer_indices:
            residual, layer_aux = self.adapters[str(index)](
                layer_features[index] if index in layer_features else layer_features[str(index)],
                excluded=disabled_by_layer.get(index, disabled_by_layer.get(str(index), ())),
            )
            enhanced = enhanced + self.layer_scales[str(index)].to(residual.dtype) * residual
            aux[index] = layer_aux
        enhanced = F.normalize(enhanced, dim=-1)
        return (enhanced, aux) if return_aux else enhanced

    def encode_features(self, images, disabled_by_layer=None, return_aux=False):
        raw, layers = self._extract_frozen_features(images)
        enhanced, aux = self.compose_visual_features(
            raw, layers, disabled_by_layer=disabled_by_layer, return_aux=True
        )
        if return_aux:
            return raw, enhanced, layers, aux
        return raw, enhanced, layers

    @torch.no_grad()
    def encode_text_prototypes(self, class_names):
        class_features, consistency = [], []
        templates = self.template or ["a photo of a {}."]
        for class_name in class_names:
            normalized_name = str(class_name).replace("_", " ").replace("-", " ")
            tokens = clip.tokenize([template.format(normalized_name) for template in templates]).to(
                self.clip_model.token_embedding.weight.device
            )
            embeddings = F.normalize(self.clip_model.encode_text(tokens).float(), dim=-1)
            mean = embeddings.mean(dim=0)
            consistency.append(mean.norm().clamp(0.0, 1.0))
            class_features.append(F.normalize(mean, dim=-1))
        return torch.stack(class_features), torch.stack(consistency)

    def set_base_trainable(self):
        for adapter in self.adapters.values():
            for parameter in adapter.parameters():
                parameter.requires_grad = True
        for parameter in self.layer_scales.values():
            parameter.requires_grad = True

    def set_incremental_trainable(self, candidate_layers):
        self.freeze_all()
        for index in candidate_layers:
            self.adapters[str(index)].set_newest_trainable()

    def freeze_all(self):
        for adapter in self.adapters.values():
            adapter.freeze_all()
        for parameter in self.layer_scales.values():
            parameter.requires_grad = False
            parameter.grad = None
        self.clip_model.eval()

    def add_expert(self, layer, session_id, init_features, history_features):
        return self.adapters[str(layer)].add_expert(session_id, init_features, history_features)

    def adapter_for_layer(self, layer):
        return self.adapters[str(layer)]

    def newest_parameters(self, layers):
        parameters = []
        for layer in layers:
            parameters.extend(self.adapters[str(layer)].newest_parameters())
        return parameters

    def export_topology(self):
        return {key: adapter.export_topology() for key, adapter in self.adapters.items()}

    def rebuild_topology(self, topology):
        for key, adapter in self.adapters.items():
            adapter.rebuild_topology(topology.get(key))
        self.freeze_all()

    @torch.no_grad()
    def set_temperatures(self, visual, text):
        minimum = float(self.cfg.TRAINER.PCMOE.CALIBRATION.MIN_TEMPERATURE)
        maximum = float(self.cfg.TRAINER.PCMOE.CALIBRATION.MAX_TEMPERATURE)
        self.temperature_visual.fill_(float(min(maximum, max(minimum, visual))))
        self.temperature_text.fill_(float(min(maximum, max(minimum, text))))
