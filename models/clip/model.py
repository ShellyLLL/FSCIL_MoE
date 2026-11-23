
from collections import OrderedDict
from typing import Tuple, Union
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class RepresentationDescriptor(nn.Module):
    """Centered tied-weight low-rank AE with standardized reconstruction-error statistics."""
    def __init__(self, d_model, descriptor_dim=64, eps=1e-4, buffer_size=500):
        super().__init__()
        self.d_model = int(d_model)
        self.descriptor_dim = max(1, min(int(descriptor_dim), self.d_model))
        self.eps, self.buffer_size = float(eps), int(buffer_size)
        if self.eps <= 0 or self.buffer_size < 3:
            raise ValueError("AE eps must be positive and buffer_size must be at least three.")
        basis, _ = torch.linalg.qr(torch.randn(self.d_model, self.descriptor_dim), mode="reduced")
        self.basis = nn.Parameter(basis)
        self.register_buffer("center", torch.zeros(self.d_model, dtype=torch.float32))
        self.residual_down = None
        self.residual_up = None
        self.residual_activation = QuickGELU()
        self.anchor_expert_id = -1
        self.residual_rank = 0
        self._anchor_ref = None
        self.register_buffer("error_history", torch.zeros(self.buffer_size, dtype=torch.float32))
        self.register_buffer("error_weight_history", torch.zeros(self.buffer_size, dtype=torch.float32))
        self.register_buffer("effective_error_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("error_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("error_cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("error_mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("error_std", torch.ones((), dtype=torch.float32))

    @property
    def is_anchored(self):
        return self._anchor_ref is not None

    def initialize_from_anchor(self, anchor, anchor_expert_id, residual_rank):
        """Reuse a frozen historical RD and add a zero-output low-rank residual."""
        anchor_expert_id = int(anchor_expert_id)
        if anchor is self or anchor.d_model != self.d_model or anchor.descriptor_dim != self.descriptor_dim:
            raise ValueError("RD anchor must be a distinct descriptor with matching dimensions.")
        if self.is_anchored:
            raise RuntimeError("A representation descriptor can only be anchored once.")
        rank = min(max(1, int(residual_rank)), self.d_model)
        with torch.no_grad():
            self.center.copy_(anchor.center)
            self.basis.copy_(anchor.basis)
        self.basis.requires_grad_(False)
        self.basis.grad = None
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        anchor.eval()
        self._anchor_ref = weakref.ref(anchor)
        self.anchor_expert_id = anchor_expert_id
        self.residual_rank = rank
        device, dtype = self.basis.device, self.basis.dtype
        self.residual_down = nn.Linear(self.d_model, rank, bias=False).to(device=device, dtype=dtype)
        self.residual_up = nn.Linear(rank, self.d_model, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_normal_(self.residual_down.weight, nonlinearity="relu")
        nn.init.zeros_(self.residual_up.weight)

    def trainable_parameters(self):
        if self.is_anchored:
            return list(self.residual_down.parameters()) + list(self.residual_up.parameters())
        return [self.basis]

    def set_trainable(self, enabled):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if enabled:
            for parameter in self.trainable_parameters():
                parameter.requires_grad_(True)

    def residual_output(self, x):
        if not self.is_anchored:
            return torch.zeros_like(x, dtype=torch.float32)
        centered = x.float() - self.center
        return self.residual_up(self.residual_activation(self.residual_down(centered)))

    @torch.no_grad()
    def set_center(self, values, weights=None):
        if self.is_anchored:
            raise RuntimeError("An anchored RD keeps its historical center frozen.")
        values = torch.as_tensor(values, device=self.center.device, dtype=torch.float32)
        if values.ndim != 2 or values.size(1) != self.d_model or not torch.isfinite(values).all():
            raise ValueError("AE center requires finite [samples, d_model] representations.")
        if weights is None:
            center = values.mean(dim=0)
        else:
            weights = torch.as_tensor(weights, device=values.device, dtype=torch.float32).reshape(-1)
            if (weights.numel() != values.size(0) or not torch.isfinite(weights).all()
                    or (weights < 0).any() or not bool((weights.sum() > 0).item())):
                raise ValueError("AE center weights must be finite, non-negative and have positive mass.")
            center = (values * weights[:, None]).sum(0) / weights.sum()
        self.center.copy_(center)

    @torch.no_grad()
    def retract_basis_(self):
        """Retract a trainable Base basis onto the Stiefel manifold."""
        if self.is_anchored:
            return
        q, r = torch.linalg.qr(self.basis.float(), mode="reduced")
        diagonal = torch.diagonal(r)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        self.basis.copy_((q * signs.unsqueeze(0)).to(dtype=self.basis.dtype))

    def forward(self, x):
        if self.is_anchored:
            anchor = self._anchor_ref()
            if anchor is None:
                raise RuntimeError("Historical RD anchor is no longer available.")
            return anchor(x) + self.residual_output(x)
        centered = x.float() - self.center
        basis = self.basis.float()
        return self.center + (centered @ basis) @ basis.t()

    def orthogonality_loss(self):
        if self.is_anchored:
            return self.basis.new_zeros(())
        basis = self.basis.float()
        gram = basis.t() @ basis
        identity = torch.eye(self.descriptor_dim, device=basis.device, dtype=basis.dtype)
        return (gram - identity).square().mean()

    def reconstruction_error(self, x):
        value = x.detach().float()
        if not torch.isfinite(value).all():
            raise FloatingPointError("AE input contains non-finite representations.")
        errors = (self.forward(value) - value).square().mean(dim=-1)
        if not torch.isfinite(errors).all():
            raise FloatingPointError("AE reconstruction errors are non-finite.")
        return errors

    @torch.no_grad()
    def update_reference(self, errors, weights=None):
        """Atomically update weighted error statistics from independent holdout errors."""
        errors = torch.as_tensor(
            errors, device=self.error_history.device, dtype=torch.float32
        ).reshape(-1)
        if errors.numel() < 2 or not torch.isfinite(errors).all():
            raise ValueError("AE reference update requires at least two finite reference errors.")
        weights = torch.ones_like(errors) if weights is None else torch.as_tensor(
            weights, device=errors.device, dtype=torch.float32
        ).reshape(-1)
        if (weights.numel() != errors.numel() or not torch.isfinite(weights).all()
                or (weights < 0).any() or not bool((weights.sum() > 0).item())):
            raise ValueError("AE calibration weights must align with errors and have positive finite mass.")

        # Construct the prospective ring-buffer state without mutating the live
        # descriptor. A failed effective-sample-size check must not leave a
        # partially updated calibration buffer behind.
        new_history = self.error_history.detach().clone()
        new_weight_history = self.error_weight_history.detach().clone()
        old_count = int(self.error_count.item())
        old_cursor = int(self.error_cursor.item())
        n = int(errors.numel())
        if n >= self.buffer_size:
            new_history.copy_(errors[-self.buffer_size:])
            new_weight_history.copy_(weights[-self.buffer_size:])
            new_count = self.buffer_size
            new_cursor = 0
        else:
            positions = (
                torch.arange(n, device=errors.device) + old_cursor
            ) % self.buffer_size
            new_history[positions] = errors
            new_weight_history[positions] = weights
            new_count = min(self.buffer_size, old_count + n)
            new_cursor = (old_cursor + n) % self.buffer_size

        values = new_history if new_count == self.buffer_size else new_history[:new_count]
        mass_weights = (
            new_weight_history if new_count == self.buffer_size
            else new_weight_history[:new_count]
        )
        mass = mass_weights.sum()
        squared_mass = mass_weights.square().sum()
        if (not bool((mass > 0).item())
                or not bool((squared_mass > 0).item())):
            raise ValueError("AE calibration responsibility mass must be strictly positive.")
        effective_n = mass.square() / squared_mass
        # Reliability-weighted unbiased variance is undefined when the
        # effective sample size is <= 1. Do not silently replace that case
        # with eps: it would manufacture arbitrarily large standardized scores.
        if not bool((effective_n > 1.0 + 1e-6).item()):
            raise ValueError(
                "AE calibration requires effective sample size greater than one; "
                f"received {float(effective_n.item()):.6f}."
            )
        mean = (values * mass_weights).sum() / mass
        variance = (mass_weights * (values - mean).square()).sum() / mass
        correction = effective_n / (effective_n - 1.0)
        raw_std = (variance * correction).sqrt()
        # A clamped standard deviation is not a calibrated scale. Accepting it
        # would turn tiny reconstruction-error differences into enormous z-scores.
        if (not torch.isfinite(mean) or not torch.isfinite(raw_std)
                or not bool((raw_std > self.eps).item())):
            raise ValueError(
                "AE reference-error distribution is degenerate: unbiased std "
                f"must exceed DESCRIPTOR_EPS ({self.eps:g}), got {float(raw_std.item()):.8g}."
            )
        std = raw_std
        if not bool((std > 0).item()):
            raise FloatingPointError("AE calibration produced invalid weighted error statistics.")

        self.error_history.copy_(new_history)
        self.error_weight_history.copy_(new_weight_history)
        self.error_count.fill_(new_count)
        self.error_cursor.fill_(new_cursor)
        self.error_mean.copy_(mean)
        self.error_std.copy_(std)
        self.effective_error_count.copy_(effective_n)
        return self.error_mean, self.error_std

    def standardized_score(self, x):
        if not self.is_ready():
            raise RuntimeError("AE descriptor has no valid independent weighted reference errors.")
        return (self.reconstruction_error(x) - self.error_mean) / self.error_std.clamp_min(self.eps)

    def is_ready(self):
        return (
            int(self.error_count.item()) >= 2
            and float(self.effective_error_count.item()) > 1.0 + 1e-6
            and bool(torch.isfinite(self.error_mean).item())
            and bool(torch.isfinite(self.error_std).item())
            and float(self.error_std.item()) > self.eps
        )


class ExpandableRouter(nn.Module):
    """Soft router with independently freezable output columns."""

    def __init__(self, dim, num_columns=0):
        super().__init__()
        self.dim, self.columns = int(dim), nn.ModuleList()
        for _ in range(int(num_columns)):
            self.add_column()

    def add_column(self, bias_init=0.0):
        column = nn.Linear(self.dim, 1)
        nn.init.normal_(column.weight, std=0.02)
        nn.init.constant_(column.bias, float(bias_init))
        self.columns.append(column)
        return len(self.columns) - 1

    def forward(self, x, num_columns=None):
        if not self.columns:
            raise RuntimeError("ExpandableRouter requires at least one column.")
        selected = self.columns if num_columns is None else self.columns[:int(num_columns)]
        logits = torch.cat([column(x) for column in selected], dim=-1)
        return logits, torch.softmax(logits, dim=-1)


class BottleneckExpert(nn.Module):
    def __init__(self, d_model, reduction=4, zero_init_up=False, dropout=0.0):
        super().__init__()
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Expert dropout must be in [0, 1).")
        hidden_dim = max(1, int(d_model) // max(1, int(reduction)))
        self.c_fc, self.gelu = nn.Linear(d_model, hidden_dim), QuickGELU()
        nn.init.kaiming_normal_(self.c_fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.c_fc.bias)
        self.dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(hidden_dim, d_model)
        if zero_init_up:
            nn.init.zeros_(self.c_proj.weight)
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        return self.c_proj(self.dropout(self.gelu(self.c_fc(x))))


class VisualMoEAdapter(nn.Module):
    """Unified expert pool, anchored distribution descriptors and expandable router."""

    def __init__(self, d_model, num_experts=4, reduction=4, residual_scale_init=1.0,
                 descriptor_dim=64, descriptor_eps=1e-4, descriptor_buffer_size=500,
                 incremental_gate_bias_init=-1.0, descriptor_residual_rank=16,
                 incremental_dropout=0.0):
        super().__init__()
        self.d_model, self.base_expert_count = int(d_model), int(num_experts)
        self.default_reduction, self.descriptor_dim = int(reduction), int(descriptor_dim)
        self.descriptor_eps = float(descriptor_eps)
        self.descriptor_buffer_size = int(descriptor_buffer_size)
        self.descriptor_residual_rank = max(1, min(int(descriptor_residual_rank), self.d_model))
        self.incremental_gate_bias_init = float(incremental_gate_bias_init)
        self.incremental_dropout = float(incremental_dropout)
        if not 0.0 <= self.incremental_dropout < 1.0:
            raise ValueError("INCREMENTAL_DROPOUT must be in [0, 1).")
        self.register_buffer("base_feature_std", torch.ones(self.d_model, dtype=torch.float32))
        self.register_buffer("base_feature_stats_ready", torch.zeros((), dtype=torch.bool))
        self.layer_norm = LayerNorm(self.d_model)
        self.residual_scale = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.experts = nn.ModuleList([BottleneckExpert(self.d_model, reduction)
                                      for _ in range(self.base_expert_count)])
        self.descriptors = nn.ModuleList([
            RepresentationDescriptor(self.d_model, descriptor_dim, descriptor_eps, descriptor_buffer_size)
            for _ in range(self.base_expert_count)
        ])
        self.router = ExpandableRouter(self.d_model, self.base_expert_count)
        self.expert_birth_sessions = [0] * self.base_expert_count
        # Runtime-only diagnostic mask. It is deliberately excluded from state_dict.
        self._inference_disabled_expert_ids = frozenset()

    @property
    def num_experts(self):
        return len(self.experts)

    def num_incremental_experts(self):
        return max(0, self.num_experts - self.base_expert_count)

    def assert_descriptors_ready(self):
        if len(self.descriptors) != self.num_experts:
            raise RuntimeError("every deployed expert must own one descriptor.")
        missing = [i for i, descriptor in enumerate(self.descriptors) if not descriptor.is_ready()]
        if missing:
            raise RuntimeError(f"deployed experts have uncalibrated descriptors: {missing}")

    def descriptor_scores(self, x, require_ready=True):
        if require_ready:
            self.assert_descriptors_ready()
        return torch.stack([descriptor.standardized_score(x) for descriptor in self.descriptors], dim=-1)

    def expansion_statistics(self, x):
        scores = self.descriptor_scores(x, require_ready=True)
        return scores, scores.mean(dim=0)

    @torch.no_grad()
    def set_base_feature_stats(self, values):
        values = torch.as_tensor(values, device=self.base_feature_std.device, dtype=torch.float32)
        if values.ndim != 2 or values.size(1) != self.d_model or values.size(0) < 2:
            raise ValueError("Base feature statistics require at least two [d_model] reference rows.")
        if not torch.isfinite(values).all():
            raise ValueError("Base feature statistics must be finite.")
        std = values.std(dim=0, unbiased=False).clamp_min(self.descriptor_eps)
        self.base_feature_std.copy_(std)
        self.base_feature_stats_ready.fill_(True)

    @torch.no_grad()
    def select_descriptor_anchor(self, session_features):
        """Select the historical RD with the lowest Session-level 90th-percentile z-score."""
        scores = self.descriptor_scores(session_features, require_ready=True).float()
        if scores.ndim != 2 or scores.size(0) == 0 or not torch.isfinite(scores).all():
            raise ValueError("RD anchor selection requires finite [session, experts] scores.")
        upper_scores = torch.quantile(scores, 0.90, dim=0)
        anchor_id = int(torch.argmin(upper_scores).item())
        return anchor_id, upper_scores.detach()

    @torch.no_grad()
    def update_descriptor_stats(self, errors, expert_id):
        return self.descriptors[int(expert_id)].update_reference(errors)

    def routing_outputs(self, route_features):
        """Return the exact base-softmax plus incremental-sigmoid gates used by forward."""
        if route_features.ndim != 2 or route_features.size(-1) != self.d_model:
            raise ValueError("Router inputs must have shape [batch, d_model].")
        if len(self.router.columns) != self.num_experts:
            raise RuntimeError("Router columns and deployed experts are not aligned.")
        base_logits, base_weights = self.router(route_features, self.base_expert_count)
        incremental_ids = list(range(self.base_expert_count, self.num_experts))
        gate_logits = [self.router.columns[index](route_features) for index in incremental_ids]
        incremental_gates = [torch.sigmoid(value) for value in gate_logits]
        logits = torch.cat([base_logits] + gate_logits, dim=-1) if gate_logits else base_logits
        weights = torch.cat([base_weights] + incremental_gates, dim=-1) if incremental_gates else base_weights
        return logits, weights

    def expert_responsibility(self, route_features, expert_id):
        """Return one expert's deployed responsibility, not a surrogate softmax."""
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= self.num_experts:
            raise IndexError(f"Expert id {expert_id} is outside the deployed expert pool.")
        if expert_id < self.base_expert_count:
            _, base_weights = self.router(route_features, self.base_expert_count)
            return base_weights[:, expert_id]
        return torch.sigmoid(self.router.columns[expert_id](route_features)).reshape(-1)

    @staticmethod
    def _anchor_points(mean, variance):
        mean = torch.as_tensor(mean).float()
        variance = torch.as_tensor(variance, device=mean.device).float()
        if mean.ndim != 2 or variance.shape != mean.shape:
            raise ValueError("Anchor means and variances must be aligned [classes, dim] matrices.")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError("Anchor statistics must be finite with non-negative variances.")
        variance = variance.clamp_min(1e-6)
        return torch.cat([mean, mean + variance.sqrt(), mean - variance.sqrt()], dim=0)

    def old_feature_preservation_loss(
        self, route_anchor_mean, route_anchor_var,
        expert_anchor_mean, expert_anchor_var,
    ):
        """Historical feature-space trust-region loss for the newest expert.

        Historical class means and diagonal variances are converted into
        balanced anchor points. The newest independent gate is discouraged
        from activating on old classes, while its actual gated residual is
        penalized relative to the anchor feature energy. Only the newest
        expert and router column receive gradients from this objective.
        """
        fields = (
            route_anchor_mean, route_anchor_var,
            expert_anchor_mean, expert_anchor_var,
        )
        if any(value is None for value in fields):
            raise ValueError("Feature-space preservation requires route and expert-input historical anchors.")
        route_anchors = self._anchor_points(route_anchor_mean, route_anchor_var).to(
            device=self.residual_scale.device
        )
        expert_anchors = self._anchor_points(expert_anchor_mean, expert_anchor_var).to(
            device=self.residual_scale.device
        )
        if route_anchors.shape != expert_anchors.shape:
            raise ValueError("Historical route and expert-input anchors must have aligned class rows and dimensions.")
        gate = torch.sigmoid(self.router.columns[-1](route_anchors))
        expert_output = self.experts[-1](expert_anchors.unsqueeze(0)).squeeze(0)
        gated_residual = self.residual_scale.to(expert_output.dtype) * gate * expert_output
        anchor_energy = expert_anchors.float().square().sum(dim=-1).clamp_min(1e-6)
        relative_residual_energy = gated_residual.float().square().sum(dim=-1) / anchor_energy
        loss = gate.float().square().mean() + relative_residual_energy.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Historical feature-space preservation loss is non-finite.")
        return loss

    @torch.no_grad()
    def snapshot_route_anchor_targets(self, route_anchor_mean, route_anchor_var):
        """Snapshot deployed routing responsibilities on compact historical anchors."""
        if route_anchor_mean is None or route_anchor_var is None:
            raise ValueError("Router preservation requires historical route anchors.")
        anchors = self._anchor_points(route_anchor_mean, route_anchor_var).to(
            device=self.residual_scale.device, dtype=torch.float32
        )
        _, responsibilities = self.routing_outputs(anchors)
        return anchors.detach(), responsibilities.detach()

    def route_preservation_loss(self, route_anchors, reference_responsibilities):
        """Preserve historical Base-softmax and incremental-sigmoid responsibilities."""
        route_anchors = torch.as_tensor(route_anchors, device=self.residual_scale.device).float()
        reference = torch.as_tensor(
            reference_responsibilities, device=route_anchors.device
        ).float().detach()
        if reference.shape != (route_anchors.size(0), self.num_experts):
            raise ValueError("Historical routing targets do not align with deployed experts.")
        _, current = self.routing_outputs(route_anchors)
        base_target = reference[:, :self.base_expert_count].clamp_min(1e-8)
        base_target = base_target / base_target.sum(-1, keepdim=True).clamp_min(1e-8)
        base_current = current[:, :self.base_expert_count].clamp_min(1e-8)
        loss = F.kl_div(base_current.log(), base_target, reduction="batchmean")
        if self.num_experts > self.base_expert_count:
            loss = loss + F.mse_loss(
                current[:, self.base_expert_count:],
                reference[:, self.base_expert_count:]
            )
        return loss

    @torch.no_grad()
    def old_anchor_response(self, anchor_mean, anchor_var):
        if anchor_mean is None or anchor_mean.numel() == 0:
            return 0.0
        anchors = self._anchor_points(anchor_mean, anchor_var).to(
            device=self.residual_scale.device
        )
        return float(torch.sigmoid(self.router.columns[-1](anchors)).mean().item())

    @torch.no_grad()
    def old_anchor_residual_response(
        self, route_anchor_mean, route_anchor_var,
        expert_anchor_mean, expert_anchor_var,
    ):
        route_anchors = self._anchor_points(route_anchor_mean, route_anchor_var).to(
            device=self.residual_scale.device
        )
        expert_anchors = self._anchor_points(expert_anchor_mean, expert_anchor_var).to(
            device=self.residual_scale.device
        )
        gate = torch.sigmoid(self.router.columns[-1](route_anchors))
        expert_output = self.experts[-1](expert_anchors.unsqueeze(0)).squeeze(0)
        residual = self.residual_scale.to(expert_output.dtype) * gate * expert_output
        return float(residual.float().norm(dim=-1).mean().item())

    def set_inference_expert_mask(self, disabled_expert_ids):
        """Disable selected expert residuals for evaluation-only ablations.

        Router logits and responsibilities are intentionally unchanged. This
        isolates each expert's residual contribution without retraining or
        changing the checkpointed model state.
        """
        if self.training:
            raise RuntimeError("Inference expert masks can only be set while the adapter is in eval mode.")
        if disabled_expert_ids is None:
            values = ()
        elif isinstance(disabled_expert_ids, int) and not isinstance(disabled_expert_ids, bool):
            values = (disabled_expert_ids,)
        else:
            try:
                values = tuple(disabled_expert_ids)
            except TypeError as error:
                raise TypeError("disabled_expert_ids must be an integer or an iterable of integers.") from error
        normalized = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Inference expert ids must be integers (booleans are not accepted).")
            expert_id = int(value)
            if expert_id < 0 or expert_id >= self.num_experts:
                raise ValueError(
                    f"Inference mask expert id {expert_id} is outside [0, {self.num_experts})."
                )
            normalized.add(expert_id)
        self._inference_disabled_expert_ids = frozenset(normalized)

    def clear_inference_expert_mask(self):
        """Restore the complete expert pool without changing model parameters."""
        self._inference_disabled_expert_ids = frozenset()

    def freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.eval()

    def set_newest_trainable(self, descriptor=False):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("there is no incremental expert to train.")
        self.freeze_all()
        for module in (self.experts[-1], self.router.columns[-1]):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if descriptor:
            self.descriptors[-1].set_trainable(True)
        self.train()

    def add_expert(self, session_id, reduction=None, gate_bias_init=None,
                   anchor_expert_id=None, descriptor_residual_rank=None, dropout=None):
        # Validate the complete expert/RD/router contract before changing the
        # live topology or freezing any currently trainable module.
        if isinstance(session_id, bool) or int(session_id) <= 0:
            raise ValueError("Incremental expert birth session must be a positive integer.")
        session_id = int(session_id)
        reduction = self.default_reduction if reduction is None else int(reduction)
        if reduction < 1:
            raise ValueError("Incremental expert reduction must be positive.")
        anchor_expert_id = 0 if anchor_expert_id is None else int(anchor_expert_id)
        if anchor_expert_id < -1 or anchor_expert_id >= self.num_experts:
            raise ValueError(
                f"RD anchor expert id {anchor_expert_id} is outside the historical pool."
            )
        residual_rank = (
            self.descriptor_residual_rank if descriptor_residual_rank is None
            else int(descriptor_residual_rank)
        )
        if anchor_expert_id == -1:
            if residual_rank != 0:
                raise ValueError("Independent RD initialization requires residual_rank=0.")
        else:
            if residual_rank < 1:
                raise ValueError("Anchored RD initialization requires a positive residual rank.")
            residual_rank = min(residual_rank, self.d_model)
        expert_dropout = self.incremental_dropout if dropout is None else float(dropout)
        gate_bias = self.incremental_gate_bias_init if gate_bias_init is None else float(gate_bias_init)
        if not np.isfinite(expert_dropout) or not 0.0 <= expert_dropout < 1.0:
            raise ValueError("Incremental expert dropout must be finite and in [0, 1).")
        if not np.isfinite(gate_bias):
            raise ValueError("Incremental router gate bias must be finite.")

        device, dtype = self.residual_scale.device, self.residual_scale.dtype
        new_expert = BottleneckExpert(
            self.d_model, reduction, zero_init_up=True, dropout=expert_dropout
        ).to(device=device, dtype=dtype)
        new_descriptor = RepresentationDescriptor(
            self.d_model, self.descriptor_dim, self.descriptor_eps, self.descriptor_buffer_size
        ).to(device=device)
        new_router_column = nn.Linear(self.d_model, 1).to(device=device, dtype=dtype)
        nn.init.normal_(new_router_column.weight, std=0.02)
        nn.init.constant_(new_router_column.bias, gate_bias)
        if anchor_expert_id >= 0:
            new_descriptor.initialize_from_anchor(
                self.descriptors[anchor_expert_id], anchor_expert_id, residual_rank
            )

        self.freeze_all()
        self.experts.append(new_expert)
        self.descriptors.append(new_descriptor)
        self.router.columns.append(new_router_column)
        self.expert_birth_sessions.append(session_id)
        self.set_newest_trainable()
        return self.num_experts - 1

    def newest_parameters(self, include_descriptor=False):
        parameters = list(self.experts[-1].parameters()) + list(self.router.columns[-1].parameters())
        if include_descriptor:
            parameters += self.descriptors[-1].trainable_parameters()
        return parameters

    def historical_named_parameters(self):
        newest = self.num_experts - 1
        prefixes = (f"experts.{newest}.", f"router.columns.{newest}.", f"descriptors.{newest}.")
        for name, parameter in self.named_parameters():
            if not name.startswith(prefixes):
                yield name, parameter

    def export_topology(self):
        return {
            "version": 4,
            "d_model": self.d_model,
            "base_expert_count": self.base_expert_count,
            "experts": [
                {
                    "reduction": max(1, self.d_model // expert.c_fc.out_features),
                    "dropout": float(expert.dropout.p),
                    "birth_session": int(self.expert_birth_sessions[index]),
                    "descriptor_anchor_id": int(self.descriptors[index].anchor_expert_id),
                    "descriptor_residual_rank": int(self.descriptors[index].residual_rank),
                }
                for index, expert in enumerate(self.experts)
            ],
        }

    def rebuild_topology(self, topology):
        if topology is None:
            topology = {
                "version": 4,
                "d_model": self.d_model,
                "base_expert_count": self.base_expert_count,
                "experts": [
                    {"reduction": self.default_reduction, "birth_session": 0,
                     "descriptor_anchor_id": -1, "descriptor_residual_rank": 0}
                    for _ in range(self.base_expert_count)
                ],
            }
        if not isinstance(topology, dict):
            raise ValueError("MoE topology must be a mapping.")
        if int(topology.get("version", 4)) != 4:
            raise ValueError("Unsupported MoE topology version.")
        if int(topology.get("d_model", self.d_model)) != self.d_model:
            raise ValueError("MoE topology d_model mismatch.")
        if int(topology.get("base_expert_count", self.base_expert_count)) != self.base_expert_count:
            raise ValueError("MoE topology base expert count mismatch.")
        specs = list(topology.get("experts", []))
        if len(specs) < self.base_expert_count:
            raise ValueError("MoE topology has fewer experts than the base pool.")
        validated = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError(f"MoE topology expert {index} must be a mapping.")
            reduction = spec.get("reduction", self.default_reduction)
            birth_session = spec.get("birth_session", 0)
            anchor_id = spec.get("descriptor_anchor_id", -1 if index < self.base_expert_count else 0)
            residual_rank = spec.get(
                "descriptor_residual_rank",
                0 if index < self.base_expert_count else self.descriptor_residual_rank,
            )
            dropout = spec.get("dropout", 0.0 if index < self.base_expert_count else self.incremental_dropout)
            if isinstance(reduction, bool) or int(reduction) < 1:
                raise ValueError(f"MoE topology expert {index} has invalid reduction.")
            if isinstance(birth_session, bool) or int(birth_session) < 0:
                raise ValueError(f"MoE topology expert {index} has invalid birth session.")
            if not 0.0 <= float(dropout) < 1.0:
                raise ValueError(f"MoE topology expert {index} has invalid dropout.")
            if index < self.base_expert_count:
                if float(dropout) != 0.0:
                    raise ValueError("Base experts must have dropout=0.0.")
                if int(birth_session) != 0:
                    raise ValueError("Base experts must have birth_session=0.")
                if int(anchor_id) != -1 or int(residual_rank) != 0:
                    raise ValueError("Base descriptors cannot have a historical RD anchor or residual branch.")
            else:
                if int(birth_session) <= 0:
                    raise ValueError("Incremental experts must have a positive birth session.")
                if int(anchor_id) == -1:
                    if isinstance(residual_rank, bool) or int(residual_rank) != 0:
                        raise ValueError(
                            f"Independent incremental descriptor {index} must have residual_rank=0."
                        )
                else:
                    if int(anchor_id) < 0 or int(anchor_id) >= index:
                        raise ValueError(
                            f"Incremental descriptor {index} must reference an earlier expert as its anchor."
                        )
                    if isinstance(residual_rank, bool) or int(residual_rank) < 1:
                        raise ValueError(f"Incremental descriptor {index} has invalid residual rank.")
            validated.append((int(reduction), int(birth_session), int(anchor_id), int(residual_rank), float(dropout)))

        # Build the complete topology only after every expert/anchor specification is valid.
        self.experts, self.descriptors = nn.ModuleList(), nn.ModuleList()
        self.router, self.expert_birth_sessions = ExpandableRouter(self.d_model), []
        self._inference_disabled_expert_ids = frozenset()
        for index, (reduction, birth_session, anchor_id, residual_rank, dropout) in enumerate(validated):
            self.experts.append(BottleneckExpert(self.d_model, reduction, dropout=dropout))
            descriptor = RepresentationDescriptor(
                self.d_model, self.descriptor_dim, self.descriptor_eps, self.descriptor_buffer_size
            )
            if anchor_id >= 0:
                descriptor.initialize_from_anchor(
                    self.descriptors[anchor_id], anchor_id, residual_rank
                )
            self.descriptors.append(descriptor)
            self.router.add_column()
            self.expert_birth_sessions.append(birth_session)
        device, dtype = self.residual_scale.device, self.residual_scale.dtype
        self.experts.to(device=device, dtype=dtype)
        self.router.to(device=device, dtype=dtype)
        self.descriptors.to(device=device, dtype=torch.float32)
        self.freeze_all()

    def remove_newest_expert(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Cannot remove a base expert.")
        # ModuleList.pop() is unavailable in older PyTorch releases.
        # Deleting the last index unregisters the temporary modules and
        # preserves the indices/state_dict keys of historical experts.
        del self.experts[-1]
        del self.descriptors[-1]
        del self.router.columns[-1]
        self.expert_birth_sessions.pop()
        self.freeze_all()

    def forward(self, x, route_input=None):
        if x.ndim != 3:
            raise ValueError("VisualMoEAdapter expects [tokens, batch, dim].")
        # Every expert and router column consumes the frozen-reference token
        # stream. The adapted stream is only the residual destination. This
        # keeps Base and incremental expert input distributions invariant to
        # upstream adapter updates, so all historical input anchors remain valid.
        if route_input is None:
            reference_tokens = self.layer_norm(x)
        elif route_input.ndim == 2 and route_input.shape == x[0].shape:
            # Backward-compatible CLS-only entry for direct adapter unit use.
            reference_tokens = self.layer_norm(x)
            route_features = self.layer_norm(route_input)
        else:
            if route_input.shape != x.shape:
                raise ValueError("Reference expert tokens must match adapted token shape.")
            reference_tokens = self.layer_norm(route_input)
        if route_input is None or route_input.ndim == 3:
            route_features = reference_tokens[0]
        logits, weights = self.routing_outputs(route_features)
        disabled = self._inference_disabled_expert_ids
        if disabled and self.training:
            raise RuntimeError("Inference expert masks must be cleared before training.")
        residual, norms, output_energies = torch.zeros_like(x), [], []
        for column, expert_id in enumerate(range(self.num_experts)):
            expert_output = self.experts[expert_id](reference_tokens)
            output_energies.append(expert_output.float().pow(2).mean())
            if expert_id in disabled:
                # Keep diagnostic tensor alignment while excluding only this
                # expert's residual. Its router gate is not renormalized.
                norms.append(expert_output.detach().float().new_zeros(()))
                continue
            weighted = expert_output * weights[:, column].unsqueeze(0).unsqueeze(-1)
            residual = residual + weighted
            norms.append(weighted.detach().float().pow(2).mean().sqrt())
        residual = self.residual_scale.to(residual.dtype) * residual
        return residual, {"logits": logits, "route_weights": weights,
                          "top_k_indices": weights.argmax(dim=-1, keepdim=True),
                          "cls_in": route_features,
                          "expert_in": reference_tokens[0],
                          "expert_residual_norms": torch.stack(norms),
                          "expert_output_energy": torch.stack(output_energies),
                          "token_residual": residual.detach()}


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.moe_adapter = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        moe_aux = None
        if self.moe_adapter is not None:
            moe_out, moe_aux = self.moe_adapter(x)
            x = x + moe_out
        return x, moe_aux

    def forward_dual(self, adapted_x: torch.Tensor, reference_x: torch.Tensor):
        """Advance adapted and frozen-reference streams through one ViT block."""
        # The reference stream is the original frozen CLIP path.  It never
        # receives adapter residuals and never builds a backbone autograd graph.
        with torch.no_grad():
            reference_x = reference_x + self.attention(self.ln_1(reference_x))
            reference_x = reference_x + self.mlp(self.ln_2(reference_x))

        adapted_x = adapted_x + self.attention(self.ln_1(adapted_x))
        adapted_x = adapted_x + self.mlp(self.ln_2(adapted_x))
        moe_aux = None
        if self.moe_adapter is not None:
            moe_out, moe_aux = self.moe_adapter(
                adapted_x, route_input=reference_x
            )
            adapted_x = adapted_x + moe_out
        return adapted_x, reference_x, moe_aux

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        moe_aux_list = []
        for block in self.resblocks:
            x, aux = block(x)
            if aux is not None: moe_aux_list.append(aux)
        return x, moe_aux_list

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def _embed_image_tokens(self, images):
        x = self.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x,
        ], dim=1)
        return self.ln_pre(x + self.positional_embedding.to(x.dtype))

    def first_moe_block(self):
        blocks = [
            index for index, block in enumerate(self.transformer.resblocks)
            if getattr(block, "moe_adapter", None) is not None
        ]
        if not blocks:
            raise RuntimeError("Reference-prefix replay requires at least one installed visual MoE adapter.")
        return min(blocks)

    @torch.no_grad()
    def reference_prefix_tokens(self, images):
        """Cache frozen-ViT tokens immediately before the first MoE insertion block."""
        start = self.first_moe_block()
        tokens = self._embed_image_tokens(images.type(self.conv1.weight.dtype)).permute(1, 0, 2)
        for block in self.transformer.resblocks[:start]:
            tokens, _ = block(tokens)
        return tokens.permute(1, 0, 2).detach()

    def forward_from_reference_prefix(self, prefix_tokens, return_moe_aux=False):
        """Differentiably replay cached frozen-prefix tokens through the current adapter pool."""
        start = self.first_moe_block()
        if prefix_tokens.ndim != 3 or prefix_tokens.size(-1) != self.conv1.out_channels:
            raise ValueError("Feature-anchor prefix tokens must have shape [batch, tokens, ViT width].")
        adapted = prefix_tokens.to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        ).permute(1, 0, 2)
        reference = adapted.detach()
        aux_list = []
        for block in self.transformer.resblocks[start:]:
            adapted, reference, aux = block.forward_dual(adapted, reference)
            if aux is not None:
                aux_list.append(aux)
        tokens = adapted.permute(1, 0, 2)
        features = self.ln_post(tokens[:, 0, :])
        if self.proj is not None:
            features = features @ self.proj
        if return_moe_aux:
            return features, aux_list
        return features

    def forward(self, x: torch.Tensor, all_layer_outputs=False,
                return_moe_aux=False, return_reference=False):
        x = self._embed_image_tokens(x)

        has_moe = any(block.moe_adapter is not None for block in self.transformer.resblocks)
        if has_moe:
            adapted_x = x.permute(1, 0, 2)
            reference_x = adapted_x.detach()
            moe_aux_list, adapted_outputs, reference_outputs = [], [], []
            for block in self.transformer.resblocks:
                adapted_x, reference_x, aux = block.forward_dual(adapted_x, reference_x)
                if aux is not None:
                    moe_aux_list.append(aux)
                if all_layer_outputs:
                    adapted_tokens = adapted_x.permute(1, 0, 2)
                    reference_tokens = reference_x.permute(1, 0, 2)
                    adapted_feature = self.ln_post(adapted_tokens[:, 0, :])
                    with torch.no_grad():
                        reference_feature = self.ln_post(reference_tokens[:, 0, :])
                        if self.proj is not None:
                            reference_feature = reference_feature @ self.proj
                    if self.proj is not None:
                        adapted_feature = adapted_feature @ self.proj
                    adapted_outputs.append(adapted_feature)
                    reference_outputs.append(reference_feature)

            if all_layer_outputs:
                if return_reference:
                    if return_moe_aux:
                        return adapted_outputs, reference_outputs, moe_aux_list
                    return adapted_outputs, reference_outputs
                if return_moe_aux:
                    return adapted_outputs, moe_aux_list
                return adapted_outputs

            adapted_tokens = adapted_x.permute(1, 0, 2)
            reference_tokens = reference_x.permute(1, 0, 2)
            adapted_feature = self.ln_post(adapted_tokens[:, 0, :])
            with torch.no_grad():
                reference_feature = self.ln_post(reference_tokens[:, 0, :])
                if self.proj is not None:
                    reference_feature = reference_feature @ self.proj
            if self.proj is not None:
                adapted_feature = adapted_feature @ self.proj
            if return_reference:
                if return_moe_aux:
                    return adapted_feature, reference_feature, moe_aux_list
                return adapted_feature, reference_feature
            if return_moe_aux:
                return adapted_feature, moe_aux_list
            return adapted_feature

        # Preserve the original single-pass CLIP path when no visual adapters
        # are installed; reference and adapted representations are identical.
        if not all_layer_outputs:
            tokens = x.permute(1, 0, 2)
            tokens, moe_aux_list = self.transformer(tokens)
            tokens = tokens.permute(1, 0, 2)
            feature = self.ln_post(tokens[:, 0, :])
            if self.proj is not None:
                feature = feature @ self.proj
            if return_reference:
                if return_moe_aux:
                    return feature, feature, moe_aux_list
                return feature, feature
            if return_moe_aux:
                return feature, moe_aux_list
            return feature

        tokens = x.permute(1, 0, 2)
        outputs, moe_aux_list = [], []
        for block in self.transformer.resblocks:
            tokens, aux = block(tokens)
            if aux is not None:
                moe_aux_list.append(aux)
            current = tokens.permute(1, 0, 2)
            current = self.ln_post(current[:, 0, :])
            if self.proj is not None:
                current = current @ self.proj
            outputs.append(current)
        if return_reference:
            if return_moe_aux:
                return outputs, outputs, moe_aux_list
            return outputs, outputs
        if return_moe_aux:
            return outputs, moe_aux_list
        return outputs

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"): nn.init.zeros_(param)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def extract_reference_prefix_tokens(self, images):
        if not isinstance(self.visual, VisionTransformer):
            raise TypeError("Feature-anchor replay is supported only by the ViT visual backbone.")
        return self.visual.reference_prefix_tokens(images.type(self.dtype))

    def extract_img_feature_from_reference_prefix(self, prefix_tokens, return_moe_aux=False):
        if not isinstance(self.visual, VisionTransformer):
            raise TypeError("Feature-anchor replay is supported only by the ViT visual backbone.")
        return self.visual.forward_from_reference_prefix(
            prefix_tokens, return_moe_aux=return_moe_aux
        )

    def encode_image(self, image, return_moe_aux=False, return_reference=False):
        if not isinstance(self.visual, VisionTransformer):
            feature = self.visual(image.type(self.dtype))
            if return_reference:
                if return_moe_aux:
                    return feature, feature, []
                return feature, feature
            return (feature, []) if return_moe_aux else feature
        return self.visual(
            image.type(self.dtype),
            return_moe_aux=return_moe_aux,
            return_reference=return_reference,
        )

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text

def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None: l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None: tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None: attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)

def build_model(state_dict: dict, cfg=None):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict: del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)

    if cfg is not None and hasattr(cfg, 'TRAINER') and cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE:
        moe_cfg = cfg.TRAINER.BiMC.VISUAL_MOE
        d_model = model.visual.transformer.width
        num_base_experts = getattr(
            moe_cfg,
            "NUM_BASE_EXPERTS",
            getattr(moe_cfg, "NUM_EXPERTS", 4),
        )
        base_reduction = getattr(
            moe_cfg,
            "BASE_REDUCTION",
            getattr(moe_cfg, "BOTTLENECK_RATIO", 4),
        )
        base_scale_init = getattr(
            moe_cfg,
            "BASE_SCALE_INIT",
            getattr(moe_cfg, "RESIDUAL_SCALE_INIT", 1.0),
        )
        descriptor_dim = getattr(moe_cfg, "DESCRIPTOR_DIM", 64)
        descriptor_eps = getattr(moe_cfg, "DESCRIPTOR_EPS", 1e-4)
        descriptor_buffer_size = getattr(moe_cfg, "DESCRIPTOR_BUFFER_SIZE", 500)
        descriptor_residual_rank = getattr(moe_cfg, "DESCRIPTOR_RESIDUAL_RANK", 16)
        incremental_gate_bias_init = getattr(moe_cfg, "INCREMENTAL_GATE_BIAS_INIT", -1.0)
        incremental_dropout = getattr(moe_cfg, "INCREMENTAL_DROPOUT", 0.1)
        for idx in moe_cfg.INSERT_BLOCKS:
            if idx < 0: idx = len(model.visual.transformer.resblocks) + idx
            block = model.visual.transformer.resblocks[idx]
            block.moe_adapter = VisualMoEAdapter(
                d_model=d_model,
                num_experts=num_base_experts,
                reduction=base_reduction,
                residual_scale_init=base_scale_init,
                descriptor_dim=descriptor_dim,
                descriptor_eps=descriptor_eps,
                descriptor_buffer_size=descriptor_buffer_size,
                incremental_gate_bias_init=incremental_gate_bias_init,
                descriptor_residual_rank=descriptor_residual_rank,
                incremental_dropout=incremental_dropout,
            )
        convert_weights(model)
    return model.eval()
