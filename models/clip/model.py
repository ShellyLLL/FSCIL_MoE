
from collections import OrderedDict
from typing import Tuple, Union

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


class ExpandableRouter(nn.Module):
    """A single expandable router whose columns compete in one sparse Top-K space."""

    def __init__(self, dim, num_columns=0):
        super().__init__()
        self.dim = int(dim)
        self.columns = nn.ModuleList()
        for _ in range(int(num_columns)):
            self.add_column()

    def add_column(self, init_bias=0.0, zero_weight=False):
        column = nn.Linear(self.dim, 1)
        nn.init.normal_(column.weight, std=0.02)
        nn.init.constant_(column.bias, float(init_bias))
        if zero_weight:
            nn.init.zeros_(column.weight)
        self.columns.append(column)
        return len(self.columns) - 1

    def logits(self, x, num_columns=None):
        if not self.columns:
            raise RuntimeError("ExpandableRouter requires at least one column.")
        selected = self.columns if num_columns is None else self.columns[:int(num_columns)]
        return torch.cat([column(x) for column in selected], dim=-1)

    def forward(self, x, top_k=2):
        logits = self.logits(x)
        probabilities = torch.softmax(logits, dim=-1)
        k = min(max(1, int(top_k)), logits.size(-1))
        selected_logits, selected_indices = torch.topk(logits, k=k, dim=-1)
        selected_weights = torch.softmax(selected_logits, dim=-1)
        sparse_weights = torch.zeros_like(logits).scatter(-1, selected_indices, selected_weights)
        return logits, probabilities, selected_indices, selected_weights, sparse_weights


class BottleneckExpert(nn.Module):
    """Lightweight residual expert with an optionally function-preserving output."""

    def __init__(self, d_model, reduction=4, zero_init_up=False):
        super().__init__()
        hidden_dim = max(1, int(d_model) // max(1, int(reduction)))
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(hidden_dim, d_model)
        if zero_init_up:
            nn.init.zeros_(self.c_proj.weight)
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class VisualMoEAdapter(nn.Module):
    """Expandable residual experts with one sparse, globally competitive router."""

    def __init__(self, d_model, num_experts=4, reduction=4, residual_scale_init=1.0,
                 top_k=2):
        super().__init__()
        if int(num_experts) < 2:
            raise ValueError("VisualMoEAdapter needs at least two base experts for Top-2 routing.")
        self.d_model = int(d_model)
        self.base_expert_count = int(num_experts)
        self.default_reduction = int(reduction)
        self.top_k = max(1, int(top_k))
        self.layer_norm = LayerNorm(self.d_model)
        self.residual_scale = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.experts = nn.ModuleList([
            BottleneckExpert(self.d_model, self.default_reduction)
            for _ in range(self.base_expert_count)
        ])
        self.router = ExpandableRouter(self.d_model, self.base_expert_count)
        self.expert_birth_sessions = [0] * self.base_expert_count

    @property
    def num_experts(self):
        return len(self.experts)

    def num_incremental_experts(self):
        return max(0, self.num_experts - self.base_expert_count)

    def freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.eval()

    def set_newest_trainable(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("There is no incremental expert to train.")
        self.freeze_all()
        for module in (self.experts[-1], self.router.columns[-1]):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.train()

    def add_expert(self, session_id, reduction=None):
        """Append a zero-output expert and a dormant router column.

        The new column starts below the existing routes. Its activation is
        learned from current-class support while historical class statistics
        explicitly constrain it to stay outside their Top-K routing regions.
        """
        self.freeze_all()
        reduction = self.default_reduction if reduction is None else int(reduction)
        device = self.residual_scale.device
        expert_dtype = next(self.experts[0].parameters()).dtype
        router_dtype = next(self.router.columns[0].parameters()).dtype
        self.experts.append(BottleneckExpert(
            self.d_model, reduction, zero_init_up=True
        ).to(device=device, dtype=expert_dtype))
        self.router.add_column(init_bias=-5.0, zero_weight=True)
        self.router.columns[-1].to(device=device, dtype=router_dtype)
        self.expert_birth_sessions.append(int(session_id))
        self.set_newest_trainable()
        return self.num_experts - 1

    def newest_parameters(self):
        return list(self.experts[-1].parameters()) + list(self.router.columns[-1].parameters())

    @torch.no_grad()
    def initialize_new_route(self, current_features, history_mean, history_variance,
                              margin=0.1, uncertainty=1.0, init_gap=0.02,
                              target_gap=0.5, max_weight_norm=2.0):
        """Initialize a dormant route near a support-derived, history-safe boundary."""
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Route initialization requires a newly appended expert.")
        current = torch.as_tensor(
            current_features, device=self.residual_scale.device, dtype=torch.float32
        ).reshape(-1, self.d_model)
        history = torch.as_tensor(
            history_mean, device=current.device, dtype=torch.float32
        ).reshape(-1, self.d_model)
        variance = torch.as_tensor(
            history_variance, device=current.device, dtype=torch.float32
        ).reshape_as(history).clamp_min(0.0)
        if current.size(0) == 0 or history.size(0) == 0:
            raise ValueError("Current and historical route features must be non-empty.")

        old_weights = torch.cat(
            [column.weight.float() for column in self.router.columns[:-1]], dim=0
        )
        old_biases = torch.cat(
            [column.bias.float() for column in self.router.columns[:-1]], dim=0
        )
        if old_weights.size(0) < 2:
            raise RuntimeError("History-safe route initialization requires at least two old experts.")

        direction = current.mean(dim=0) - history.mean(dim=0)
        direction_norm_sq = direction.square().sum()
        if float(direction_norm_sq.item()) <= 1e-12:
            weight = torch.zeros_like(direction)
        else:
            weight = direction * (float(target_gap) / direction_norm_sq)
            weight_norm = weight.norm()
            if float(weight_norm.item()) > float(max_weight_norm):
                weight = weight * (float(max_weight_norm) / weight_norm.clamp_min(1e-12))

        current_old_logits = F.linear(current, old_weights, old_biases)
        current_second_old = current_old_logits.topk(k=2, dim=-1).values[:, -1]
        current_no_bias = F.linear(current, weight.unsqueeze(0), None).squeeze(-1)
        current_bias_limit = (current_second_old - current_no_bias).min() - float(init_gap)

        history_old_logits = F.linear(history, old_weights, old_biases)
        history_top2 = history_old_logits.topk(k=2, dim=-1).indices
        selected_old_logits = history_old_logits.gather(1, history_top2)
        selected_old_weights = old_weights[history_top2]
        delta_weight = weight.view(1, 1, -1) - selected_old_weights
        difference_std = torch.sqrt(
            (variance[:, None, :] * delta_weight.square()).sum(dim=-1).clamp_min(0.0)
        )
        history_no_bias = F.linear(history, weight.unsqueeze(0), None).squeeze(-1)
        history_bias_limit = (
            selected_old_logits - history_no_bias[:, None]
            - float(margin) - float(uncertainty) * difference_std
        ).min()

        bias = torch.minimum(current_bias_limit, history_bias_limit)
        new_column = self.router.columns[-1]
        new_column.weight.copy_(weight.view_as(new_column.weight))
        new_column.bias.fill_(float(bias.item()))

        current_gap = current_second_old - F.linear(
            current, new_column.weight.float(), new_column.bias.float()
        ).squeeze(-1)
        history_gap = (
            selected_old_logits - F.linear(
                history, new_column.weight.float(), new_column.bias.float()
            ).squeeze(-1)[:, None]
            - float(uncertainty) * difference_std
        )
        activation_bias_needed = torch.quantile(
            current_second_old - current_no_bias, 0.5
        ) + float(margin)
        return {
            "bias": float(new_column.bias.detach().item()),
            "weight_norm": float(new_column.weight.detach().float().norm().item()),
            "current_min_gap": float(current_gap.min().item()),
            "history_min_robust_gap": float(history_gap.min().item()),
            "activation_feasible_at_init": bool(history_bias_limit > activation_bias_needed),
        }

    def historical_named_parameters(self):
        newest = self.num_experts - 1
        prefixes = (f"experts.{newest}.", f"router.columns.{newest}.")
        for name, parameter in self.named_parameters():
            if not name.startswith(prefixes):
                yield name, parameter

    def route_preservation_loss(self, history_mean, history_variance,
                                margin=0.1, uncertainty=1.0):
        """Penalize new-route activation over stored class-local feature statistics.

        The uncertainty term uses a diagonal-Gaussian approximation to historical
        support covariance; it is an empirical constraint, not a distribution-free guarantee.
        """
        if self.num_experts <= self.base_expert_count:
            return self.residual_scale.new_zeros((), dtype=torch.float32)
        means = torch.as_tensor(history_mean, device=self.residual_scale.device,
                                dtype=torch.float32).reshape(-1, self.d_model)
        if means.numel() == 0:
            return self.residual_scale.new_zeros((), dtype=torch.float32)
        variances = torch.as_tensor(history_variance, device=means.device,
                                    dtype=torch.float32).reshape_as(means).clamp_min(0.0)
        old_weights = torch.cat([
            column.weight.float() for column in self.router.columns[:-1]
        ], dim=0)
        old_biases = torch.cat([
            column.bias.float() for column in self.router.columns[:-1]
        ], dim=0)
        old_logits = F.linear(means, old_weights, old_biases)
        old_top2 = old_logits.topk(k=2, dim=-1).indices
        selected_weights = old_weights[old_top2]
        selected_biases = old_biases[old_top2]
        new_column = self.router.columns[-1]
        delta_weight = new_column.weight.float().view(1, 1, -1) - selected_weights
        delta_bias = new_column.bias.float().view(1, 1) - selected_biases
        mean_difference = (means[:, None, :] * delta_weight).sum(dim=-1) + delta_bias
        difference_variance = (
            variances[:, None, :] * delta_weight.square()
        ).sum(dim=-1).clamp_min(0.0)
        robust_upper = mean_difference + float(uncertainty) * torch.sqrt(
            difference_variance + 1e-8
        )
        return F.relu(robust_upper + float(margin)).mean()

    def route_objective(self, current_features, history_mean, history_variance,
                        margin=0.1, uncertainty=1.0):
        """Activate the new expert for current classes while preserving old Top-2 routes."""
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Route objectives require a newly appended expert.")
        features = current_features.float()
        old_weights = torch.cat([
            column.weight.float() for column in self.router.columns[:-1]
        ], dim=0)
        old_biases = torch.cat([
            column.bias.float() for column in self.router.columns[:-1]
        ], dim=0)
        old_logits = F.linear(features, old_weights, old_biases)
        second_old = old_logits.topk(k=2, dim=-1).values[:, -1]
        new_column = self.router.columns[-1]
        new_logits = F.linear(features, new_column.weight.float(),
                              new_column.bias.float()).squeeze(-1)
        activation_loss = F.relu(second_old + float(margin) - new_logits).mean()
        preservation_loss = self.route_preservation_loss(
            history_mean, history_variance, margin=margin, uncertainty=uncertainty
        )
        return activation_loss, preservation_loss

    def export_topology(self):
        return {
            "version": 2,
            "d_model": self.d_model,
            "base_expert_count": self.base_expert_count,
            "top_k": self.top_k,
            "experts": [
                {
                    "reduction": max(1, self.d_model // expert.c_fc.out_features),
                    "birth_session": int(self.expert_birth_sessions[index]),
                }
                for index, expert in enumerate(self.experts)
            ],
        }

    def rebuild_topology(self, topology):
        if topology is None:
            topology = {
                "version": 2,
                "d_model": self.d_model,
                "base_expert_count": self.base_expert_count,
                "top_k": self.top_k,
                "experts": [
                    {"reduction": self.default_reduction, "birth_session": 0}
                    for _ in range(self.base_expert_count)
                ],
            }
        if int(topology.get("version", -1)) != 2:
            raise ValueError("Unsupported MoE topology; rebuild from a unified Top-K checkpoint.")
        if int(topology.get("d_model", self.d_model)) != self.d_model:
            raise ValueError("MoE topology d_model mismatch.")
        if int(topology.get("base_expert_count", self.base_expert_count)) != self.base_expert_count:
            raise ValueError("MoE topology base expert count mismatch.")
        if int(topology.get("top_k", self.top_k)) != self.top_k:
            raise ValueError("MoE topology Top-K mismatch.")
        specs = list(topology.get("experts", []))
        if len(specs) < self.base_expert_count:
            raise ValueError("MoE topology has fewer experts than the base pool.")
        self.experts = nn.ModuleList()
        self.router = ExpandableRouter(self.d_model)
        self.expert_birth_sessions = []
        for index, spec in enumerate(specs):
            reduction = int(spec.get("reduction", self.default_reduction))
            self.experts.append(BottleneckExpert(self.d_model, reduction))
            self.router.add_column(
                init_bias=-5.0 if index >= self.base_expert_count else 0.0,
                zero_weight=index >= self.base_expert_count,
            )
            self.expert_birth_sessions.append(int(spec.get("birth_session", 0)))
        device = self.residual_scale.device
        expert_dtype = next(self.experts[0].parameters()).dtype
        router_dtype = next(self.router.columns[0].parameters()).dtype
        self.experts.to(device=device, dtype=expert_dtype)
        self.router.to(device=device, dtype=router_dtype)
        self.freeze_all()

    def remove_newest_expert(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Cannot remove a base expert.")
        del self.experts[-1]
        del self.router.columns[-1]
        self.expert_birth_sessions.pop()
        self.freeze_all()

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("VisualMoEAdapter expects [tokens, batch, dim].")
        # Keep the adapter path in FP32 even when the frozen CLIP backbone is FP16.
        x_norm = self.layer_norm(x.float())
        logits, probabilities, top_indices, top_weights, sparse_weights = self.router(
            x_norm[0], top_k=self.top_k
        )
        residual = torch.zeros_like(x_norm)
        residual_norms = x_norm.new_zeros(self.num_experts, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            selected = top_indices.eq(expert_id)
            rows = torch.nonzero(selected.any(dim=-1), as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            weights = (selected[rows].to(top_weights.dtype) * top_weights[rows]).sum(dim=-1)
            expert_input = x_norm.index_select(1, rows)
            expert_output = expert(expert_input)
            weighted_output = expert_output * weights.view(1, -1, 1)
            residual = residual.index_add(1, rows, weighted_output)
            residual_norms[expert_id] = weighted_output.detach().float().pow(2).mean().sqrt()
        residual = self.residual_scale.float() * residual
        return residual.to(dtype=x.dtype), {
            "logits": logits,
            "gate_probabilities": probabilities,
            "route_weights": sparse_weights,
            "top_k_indices": top_indices,
            "top_k_weights": top_weights,
            "cls_in": x_norm[0],
            "patch_mean_in": x_norm[1:].mean(dim=0),
            "expert_residual_norms": residual_norms,
        }


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

    def forward(self, x: torch.Tensor, disable_moe=False):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        moe_aux = None
        if self.moe_adapter is not None:
            if disable_moe:
                # Raw frozen-backbone block output before any adapter residual.
                moe_aux = {"cls_in": x[0], "patch_mean_in": x[1:].mean(dim=0)}
            else:
                moe_out, moe_aux = self.moe_adapter(x, )
                x = x + moe_out
        return x, moe_aux

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, disable_moe=False):
        moe_aux_list = []
        for block in self.resblocks:
            x, aux = block(x, disable_moe=disable_moe)
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

    def forward(self, x: torch.Tensor, all_layer_outputs=False, return_moe_aux=False,
                disable_moe=False):
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        if not all_layer_outputs:
            x = x.permute(1, 0, 2)  
            x, moe_aux_list = self.transformer(
                x, disable_moe=disable_moe
            )
            x = x.permute(1, 0, 2)  
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None: x = x @ self.proj
            if return_moe_aux: return x, moe_aux_list
            return x
        else:
            x = x.permute(1, 0, 2)  
            outputs, moe_aux_list = [], []
            for block in self.transformer.resblocks:
                x, aux = block(x, disable_moe=disable_moe)
                if aux is not None: moe_aux_list.append(aux)
                cur_output = x.permute(1, 0, 2)
                cur_output = self.ln_post(cur_output[:, 0, :])
                if self.proj is not None: cur_output = cur_output @ self.proj
                outputs.append(cur_output)
            if return_moe_aux: return outputs, moe_aux_list
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

    def encode_image(self, image, return_moe_aux=False, disable_moe=False):
        if not isinstance(self.visual, VisionTransformer):
            return self.visual(image.type(self.dtype))
        if return_moe_aux:
            return self.visual(image.type(self.dtype), return_moe_aux=True,
                               disable_moe=disable_moe)
        return self.visual(image.type(self.dtype), disable_moe=disable_moe)

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
        top_k = getattr(moe_cfg, "TOP_K", 2)
        for idx in moe_cfg.INSERT_BLOCKS:
            if idx < 0: idx = len(model.visual.transformer.resblocks) + idx
            block = model.visual.transformer.resblocks[idx]
            # Pretrained CLIP weights were converted before adapter insertion.
            # Keep all expert and router parameters in FP32 for stable few-shot updates.
            block.moe_adapter = VisualMoEAdapter(
                d_model=d_model,
                num_experts=num_base_experts,
                reduction=base_reduction,
                residual_scale_init=base_scale_init,
                top_k=top_k,
            ).float()
    return model.eval()
