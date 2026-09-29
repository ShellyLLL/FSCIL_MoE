
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


class RepresentationDescriptor(nn.Module):
    """FP32 reconstruction descriptor for one expert's local input distribution."""

    def __init__(self, d_model, descriptor_dim=64, eps=1e-4):
        super().__init__()
        self.d_model, self.eps = int(d_model), float(eps)
        self.descriptor_dim = max(1, min(int(descriptor_dim), self.d_model))
        self.encoder = nn.Linear(self.d_model, self.descriptor_dim)
        self.activation = QuickGELU()
        self.decoder = nn.Linear(self.descriptor_dim, self.d_model)
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("fit_sample_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibration_sample_count", torch.tensor(0, dtype=torch.long))

    def _reconstruct_fp32(self, x):
        encoded = F.linear(x.float(), self.encoder.weight.float(), self.encoder.bias.float())
        return F.linear(self.activation(encoded), self.decoder.weight.float(), self.decoder.bias.float())

    def forward(self, x):
        return self._reconstruct_fp32(x).to(dtype=x.dtype)

    def reconstruction_error(self, x):
        values = x.detach().float()
        return (self._reconstruct_fp32(values) - values).pow(2).mean(dim=-1)

    @torch.no_grad()
    def update_stats(self, errors, responsibilities=None, fit_sample_count=None):
        errors = torch.as_tensor(errors, device=self.mean.device, dtype=torch.float32).reshape(-1)
        finite = torch.isfinite(errors)
        errors = errors[finite]
        if errors.numel() == 0:
            raise ValueError("descriptor calibration requires finite errors.")
        weights = torch.ones_like(errors) if responsibilities is None else torch.as_tensor(
            responsibilities, device=self.mean.device, dtype=torch.float32
        ).reshape(-1)[finite].clamp_min(0)
        mass = weights.sum()
        if not bool(mass.gt(0).item()):
            raise ValueError("descriptor responsibilities require positive mass.")
        mean = (weights * errors).sum() / mass
        variance = (weights * (errors - mean).square()).sum() / mass
        self.mean.copy_(mean)
        self.std.copy_(variance.sqrt().clamp_min(max(self.eps, abs(float(mean.item())) * self.eps)))
        self.fit_sample_count.fill_(errors.numel() if fit_sample_count is None else int(fit_sample_count))
        self.calibration_sample_count.fill_(errors.numel())
        return self.mean, self.std

    def standardized_score(self, x):
        return (self.reconstruction_error(x) - self.mean) / self.std.clamp_min(self.eps)

    def is_ready(self):
        return int(self.fit_sample_count.item()) > 0 and int(self.calibration_sample_count.item()) > 0


class ExpandableRouter(nn.Module):
    """Soft router with independently freezable output columns."""

    def __init__(self, dim, num_columns=0):
        super().__init__()
        self.dim, self.columns = int(dim), nn.ModuleList()
        for _ in range(int(num_columns)):
            self.add_column()

    def add_column(self):
        column = nn.Linear(self.dim, 1)
        nn.init.normal_(column.weight, std=0.02)
        nn.init.zeros_(column.bias)
        self.columns.append(column)
        return len(self.columns) - 1

    def forward(self, x, active_count=None):
        """Normalize only over the columns present in the requested historical topology."""
        count = len(self.columns) if active_count is None else int(active_count)
        if not 1 <= count <= len(self.columns):
            raise ValueError("active_count must select an existing nonempty router prefix.")
        logits = torch.cat([column(x) for column in self.columns[:count]], dim=-1)
        return logits, torch.softmax(logits, dim=-1)


class BottleneckExpert(nn.Module):
    def __init__(self, d_model, reduction=4, zero_init_up=False):
        super().__init__()
        hidden_dim = max(1, int(d_model) // max(1, int(reduction)))
        self.c_fc, self.gelu = nn.Linear(d_model, hidden_dim), QuickGELU()
        self.c_proj = nn.Linear(hidden_dim, d_model)
        if zero_init_up:
            nn.init.zeros_(self.c_proj.weight)
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class VisualMoEAdapter(nn.Module):
    """Unified expert pool, descriptors and expandable soft router."""

    def __init__(self, d_model, num_experts=4, reduction=4, residual_scale_init=1.0,
                 descriptor_dim=64, descriptor_eps=1e-4):
        super().__init__()
        self.d_model, self.base_expert_count = int(d_model), int(num_experts)
        self.default_reduction, self.descriptor_dim = int(reduction), int(descriptor_dim)
        self.descriptor_eps = float(descriptor_eps)
        self.layer_norm = LayerNorm(self.d_model)
        self.residual_scale = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.experts = nn.ModuleList([BottleneckExpert(self.d_model, reduction)
                                      for _ in range(self.base_expert_count)])
        self.descriptors = nn.ModuleList([
            RepresentationDescriptor(self.d_model, descriptor_dim, descriptor_eps)
            for _ in range(self.base_expert_count)
        ])
        self.router = ExpandableRouter(self.d_model, self.base_expert_count)
        self.expert_birth_sessions = [0] * self.base_expert_count
        # Each descriptor is evaluated using the feature path on which it was fitted.
        self.descriptor_topologies = [None] * self.base_expert_count

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

    @torch.no_grad()
    def update_descriptor_stats(self, errors, expert_id, responsibilities=None, fit_sample_count=None):
        return self.descriptors[int(expert_id)].update_stats(
            errors, responsibilities=responsibilities, fit_sample_count=fit_sample_count
        )

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
            for parameter in self.descriptors[-1].parameters():
                parameter.requires_grad = True
        self.train()

    def add_expert(self, session_id, reduction=None):
        self.freeze_all()
        reduction = self.default_reduction if reduction is None else int(reduction)
        self.experts.append(BottleneckExpert(self.d_model, reduction, zero_init_up=True).to(
            device=self.residual_scale.device, dtype=self.residual_scale.dtype))
        self.descriptors.append(RepresentationDescriptor(
            self.d_model, self.descriptor_dim, self.descriptor_eps).to(device=self.residual_scale.device))
        self.router.add_column()
        self.router.columns[-1].to(device=self.residual_scale.device, dtype=self.residual_scale.dtype)
        self.expert_birth_sessions.append(int(session_id))
        self.descriptor_topologies.append(None)
        self.set_newest_trainable()
        return self.num_experts - 1

    def set_descriptor_topology(self, expert_id, topology):
        self.descriptor_topologies[int(expert_id)] = (
            None if topology is None else {int(k): int(v) for k, v in topology.items()}
        )

    def discard_newest(self):
        """Transactional rollback: previous modules, parameters and buffers are untouched."""
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("cannot discard a base expert")
        self.experts.pop(-1)
        self.descriptors.pop(-1)
        self.router.columns.pop(-1)
        self.expert_birth_sessions.pop()
        self.descriptor_topologies.pop()
        self.freeze_all()

    def newest_parameters(self, include_descriptor=False):
        parameters = list(self.experts[-1].parameters()) + list(self.router.columns[-1].parameters())
        if include_descriptor:
            parameters += list(self.descriptors[-1].parameters())
        return parameters

    def historical_named_parameters(self):
        newest = self.num_experts - 1
        prefixes = (f"experts.{newest}.", f"router.columns.{newest}.", f"descriptors.{newest}.")
        for name, parameter in self.named_parameters():
            if not name.startswith(prefixes):
                yield name, parameter

    def export_topology(self):
        return {"version": 1, "d_model": self.d_model, "base_expert_count": self.base_expert_count,
                "experts": [{"reduction": max(1, self.d_model // expert.c_fc.out_features),
                             "birth_session": int(self.expert_birth_sessions[index]),
                             "descriptor_topology": self.descriptor_topologies[index]}
                            for index, expert in enumerate(self.experts)]}

    def rebuild_topology(self, topology):
        if topology is None:
            topology = {"d_model": self.d_model, "base_expert_count": self.base_expert_count,
                        "experts": [{"reduction": self.default_reduction, "birth_session": 0}
                                    for _ in range(self.base_expert_count)]}
        if int(topology.get("d_model", self.d_model)) != self.d_model:
            raise ValueError("MoE topology d_model mismatch.")
        if int(topology.get("base_expert_count", self.base_expert_count)) != self.base_expert_count:
            raise ValueError("MoE topology base expert count mismatch.")
        specs = list(topology.get("experts", []))
        if len(specs) < self.base_expert_count:
            raise ValueError("MoE topology has fewer experts than the base pool.")
        self.experts, self.descriptors = nn.ModuleList(), nn.ModuleList()
        self.router, self.expert_birth_sessions = ExpandableRouter(self.d_model), []
        self.descriptor_topologies = []
        for spec in specs:
            reduction = int(spec.get("reduction", self.default_reduction))
            self.experts.append(BottleneckExpert(self.d_model, reduction))
            self.descriptors.append(RepresentationDescriptor(
                self.d_model, self.descriptor_dim, self.descriptor_eps))
            self.router.add_column()
            self.expert_birth_sessions.append(int(spec.get("birth_session", 0)))
            version = spec.get("descriptor_topology")
            self.descriptor_topologies.append(
                None if version is None else {int(k): int(v) for k, v in version.items()}
            )
        device, dtype = self.residual_scale.device, self.residual_scale.dtype
        self.experts.to(device=device, dtype=dtype)
        self.router.to(device=device, dtype=dtype)
        self.descriptors.to(device=device, dtype=torch.float32)
        self.freeze_all()

    def forward(self, x, active_count=None):
        if x.ndim != 3:
            raise ValueError("VisualMoEAdapter expects [tokens, batch, dim].")
        count = self.num_experts if active_count is None else int(active_count)
        x_norm = self.layer_norm(x)
        logits, weights = self.router(x_norm[0], active_count=count)
        residual, norms = torch.zeros_like(x_norm), []
        for expert_id, expert in enumerate(self.experts[:count]):
            weighted = expert(x_norm) * weights[:, expert_id].unsqueeze(0).unsqueeze(-1)
            residual = residual + weighted
            norms.append(weighted.detach().float().pow(2).mean().sqrt())
        residual = self.residual_scale.to(residual.dtype) * residual
        return residual, {"logits": logits, "route_weights": weights,
                          "top_k_indices": weights.argmax(dim=-1, keepdim=True),
                          "cls_in": x_norm[0], "patch_mean_in": x_norm[1:].mean(dim=0),
                          "expert_residual_norms": torch.stack(norms),
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

    def forward(self, x: torch.Tensor, active_count=None):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        moe_aux = None
        if self.moe_adapter is not None:
            moe_out, moe_aux = self.moe_adapter(x, active_count=active_count)
            x = x + moe_out
        return x, moe_aux

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, active_counts=None):
        moe_aux_list = []
        for index, block in enumerate(self.resblocks):
            count = None if active_counts is None else active_counts.get(index, active_counts.get(str(index)))
            x, aux = block(x, active_count=count)
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
                active_counts=None):
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        if not all_layer_outputs:
            x = x.permute(1, 0, 2)  
            x, moe_aux_list = self.transformer(x, active_counts=active_counts)
            x = x.permute(1, 0, 2)  
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None: x = x @ self.proj
            if return_moe_aux: return x, moe_aux_list
            return x
        else:
            x = x.permute(1, 0, 2)  
            outputs, moe_aux_list = [], []
            for index, block in enumerate(self.transformer.resblocks):
                count = None if active_counts is None else active_counts.get(index, active_counts.get(str(index)))
                x, aux = block(x, active_count=count)
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

    def encode_image(self, image, return_moe_aux=False, active_counts=None):
        return self.visual(
            image.type(self.dtype), return_moe_aux=return_moe_aux,
            active_counts=active_counts,
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
            )
        convert_weights(model)
    return model.eval()
