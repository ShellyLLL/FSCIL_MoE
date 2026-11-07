
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


class BottleneckExpert(nn.Module):
    """A residual bottleneck expert. Zero-up initialization preserves the old function."""
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
    """Base experts use one Softmax pool; every incremental expert has an independent Sigmoid gate.

    Routers consume the frozen-backbone reference CLS feature. Experts transform
    the enhanced hidden sequence. The two routing pools are never jointly
    normalized and no Top-K selection is applied.
    """
    def __init__(self, d_model, num_experts=4, reduction=4, incremental_reduction=None):
        super().__init__()
        self.d_model = int(d_model)
        self.base_expert_count = int(num_experts)
        self.default_reduction = int(reduction)
        self.incremental_reduction = int(
            self.default_reduction if incremental_reduction is None else incremental_reduction
        )
        if self.base_expert_count < 1:
            raise ValueError("At least one base expert is required.")
        self.layer_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.experts = nn.ModuleList([
            BottleneckExpert(self.d_model, self.default_reduction, zero_init_up=False)
            for _ in range(self.base_expert_count)
        ])
        self.base_router = nn.Linear(self.d_model, self.base_expert_count)
        self.incremental_routers = nn.ModuleList()
        self.expert_birth_sessions = [0] * self.base_expert_count

    @property
    def num_experts(self):
        return len(self.experts)

    def freeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        self.eval()

    def set_base_trainable(self):
        self.freeze_all()
        for parameter in self.base_router.parameters():
            parameter.requires_grad = True
        for expert in self.experts[:self.base_expert_count]:
            for parameter in expert.parameters():
                parameter.requires_grad = True
        self.train()

    def set_newest_trainable(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("There is no incremental expert to train.")
        self.freeze_all()
        for parameter in self.experts[-1].parameters():
            parameter.requires_grad = True
        for parameter in self.incremental_routers[-1].parameters():
            parameter.requires_grad = True
        self.train()

    def add_expert(self, session_id, reduction=None):
        self.freeze_all()
        reduction = self.incremental_reduction if reduction is None else int(reduction)
        reference = next(self.base_router.parameters())
        expert = BottleneckExpert(self.d_model, reduction, zero_init_up=True)
        router = nn.Linear(self.d_model, 1)
        # A low initial gate keeps the newly added residual conservative;
        # the classification and old-gate objectives subsequently learn it.
        nn.init.normal_(router.weight, std=0.01)
        nn.init.constant_(router.bias, -1.0)
        expert.to(device=reference.device, dtype=reference.dtype)
        router.to(device=reference.device, dtype=reference.dtype)
        self.experts.append(expert)
        self.incremental_routers.append(router)
        self.expert_birth_sessions.append(int(session_id))
        self.set_newest_trainable()
        return self.num_experts - 1

    def newest_parameters(self):
        if self.num_experts <= self.base_expert_count:
            return []
        return list(self.experts[-1].parameters()) + list(self.incremental_routers[-1].parameters())

    def newest_gate(self, raw_reference_cls):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("There is no incremental expert to inspect.")
        router = self.incremental_routers[-1]
        values = torch.as_tensor(raw_reference_cls, device=router.weight.device, dtype=router.weight.dtype)
        return torch.sigmoid(router(values)).squeeze(-1)

    def export_topology(self):
        return {
            "version": 3,
            "d_model": self.d_model,
            "base_expert_count": self.base_expert_count,
            "incremental_reduction": self.incremental_reduction,
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
                "d_model": self.d_model,
                "base_expert_count": self.base_expert_count,
                "incremental_reduction": self.incremental_reduction,
                "experts": [
                    {"reduction": self.default_reduction, "birth_session": 0}
                    for _ in range(self.base_expert_count)
                ],
            }
        if int(topology.get("d_model", self.d_model)) != self.d_model:
            raise ValueError("MoE topology d_model mismatch.")
        if int(topology.get("base_expert_count", self.base_expert_count)) != self.base_expert_count:
            raise ValueError("MoE topology base expert count mismatch.")
        self.incremental_reduction = int(topology.get(
            "incremental_reduction", self.incremental_reduction
        ))
        specs = list(topology.get("experts", []))
        if len(specs) < self.base_expert_count:
            raise ValueError("MoE topology has fewer experts than the base pool.")
        if len(specs[:self.base_expert_count]) != self.base_expert_count:
            raise ValueError("MoE topology base expert count is inconsistent.")
        reference = next(self.base_router.parameters())
        self.experts = nn.ModuleList()
        self.incremental_routers = nn.ModuleList()
        self.expert_birth_sessions = []
        for index, spec in enumerate(specs):
            reduction = int(spec.get("reduction", self.default_reduction))
            expert = BottleneckExpert(
                self.d_model, reduction,
                zero_init_up=(index >= self.base_expert_count),
            ).to(device=reference.device, dtype=reference.dtype)
            self.experts.append(expert)
            self.expert_birth_sessions.append(int(spec.get("birth_session", 0)))
            if index >= self.base_expert_count:
                router = nn.Linear(self.d_model, 1).to(
                    device=reference.device, dtype=reference.dtype
                )
                self.incremental_routers.append(router)
        self.freeze_all()

    def remove_newest_expert(self):
        if self.num_experts <= self.base_expert_count:
            raise RuntimeError("Cannot remove a base expert.")
        del self.experts[-1]
        del self.incremental_routers[-1]
        self.expert_birth_sessions.pop()
        self.freeze_all()

    def forward(self, x, raw_reference_cls):
        if x.ndim != 3:
            raise ValueError("VisualMoEAdapter expects [tokens, batch, dim].")
        if (raw_reference_cls is None or raw_reference_cls.ndim != 2
                or raw_reference_cls.shape != (x.size(1), self.d_model)):
            raise ValueError("raw_reference_cls must have shape [batch, d_model].")
        expert_input = self.layer_norm(x)
        route_input = raw_reference_cls.to(dtype=expert_input.dtype)
        base_weights = torch.softmax(self.base_router(route_input), dim=-1)
        residual = torch.zeros_like(expert_input)
        base_norms = []
        for expert_id in range(self.base_expert_count):
            component = self.experts[expert_id](expert_input)
            weight = base_weights[:, expert_id].view(1, -1, 1)
            weighted = component * weight
            residual = residual + weighted
            base_norms.append(weighted.detach().float().pow(2).mean().sqrt())

        incremental_gates = []
        incremental_norms = []
        for offset, router in enumerate(self.incremental_routers):
            gate = torch.sigmoid(router(route_input)).squeeze(-1)
            incremental_gates.append(gate)
            component = self.experts[self.base_expert_count + offset](expert_input)
            weighted = component * gate.view(1, -1, 1)
            residual = residual + weighted
            incremental_norms.append(weighted.detach().float().pow(2).mean().sqrt())

        gates = (
            torch.stack(incremental_gates, dim=-1)
            if incremental_gates else route_input.new_empty((route_input.size(0), 0))
        )
        return residual, {
            "raw_reference_cls": raw_reference_cls.detach(),
            "adapter_input_cls": x[0],
            "adapter_normalized_cls": expert_input[0],
            "adapter_output_cls": (expert_input + residual)[0],
            "base_route_weights": base_weights,
            "incremental_gates": gates,
            "expert_residual_norms": torch.stack(base_norms + incremental_norms),
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

    def forward_base(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

    def forward(self, x: torch.Tensor, raw_reference_cls=None):
        x = self.forward_base(x)
        aux = None
        if self.moe_adapter is not None:
            if raw_reference_cls is None:
                raise ValueError("An adapter block requires its frozen-backbone reference CLS feature.")
            residual, aux = self.moe_adapter(x, raw_reference_cls)
            x = x + residual
            aux["block_out"] = x[0]
        return x, aux


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, return_layer_outputs=False):
        # Reference stream is the exact frozen ViT path. Before the first
        # adapter, the enhanced stream is identical and can share this tensor.
        reference = x.detach()
        hidden = None
        has_adapter = False
        moe_aux_list, layer_outputs = [], []
        for block in self.resblocks:
            with torch.no_grad():
                reference = block.forward_base(reference)
            hidden = block.forward_base(hidden) if has_adapter else reference
            if block.moe_adapter is not None:
                residual, aux = block.moe_adapter(hidden, reference[0])
                hidden = hidden + residual
                aux["block_out"] = hidden[0]
                moe_aux_list.append(aux)
                has_adapter = True
            if return_layer_outputs:
                layer_outputs.append(hidden)
        return hidden, moe_aux_list, layer_outputs


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
                max_version=None, max_session=None):
        if max_version is not None or max_session is not None:
            raise ValueError("Versioned encoder execution is not part of the unified FSCIL model.")
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)
        hidden, moe_aux_list, layer_outputs = self.transformer(
            x, return_layer_outputs=all_layer_outputs
        )
        if all_layer_outputs:
            outputs = []
            for layer_hidden in layer_outputs:
                value = layer_hidden.permute(1, 0, 2)
                value = self.ln_post(value[:, 0, :])
                if self.proj is not None:
                    value = value @ self.proj
                outputs.append(value)
            if return_moe_aux:
                return outputs, moe_aux_list
            return outputs
        value = hidden.permute(1, 0, 2)
        value = self.ln_post(value[:, 0, :])
        if self.proj is not None:
            value = value @ self.proj
        if return_moe_aux:
            return value, moe_aux_list
        return value



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

    def encode_image(self, image, return_moe_aux=False, max_version=None, max_session=None):
        if max_version is not None or max_session is not None:
            raise ValueError("Versioned encoder execution is not part of the unified FSCIL model.")
        if not isinstance(self.visual, VisionTransformer):
            if return_moe_aux:
                raise ValueError("MoE auxiliary outputs are available only for ViT backbones.")
            return self.visual(image.type(self.dtype))
        return self.visual(
            image.type(self.dtype),
            return_moe_aux=return_moe_aux,
        )

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x, _, _ = self.transformer(x)
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
        num_base_experts = int(getattr(moe_cfg, "NUM_BASE_EXPERTS", 4))
        base_reduction = int(getattr(moe_cfg, "BASE_REDUCTION", 4))
        incremental_reduction = int(getattr(moe_cfg, "INCREMENTAL_REDUCTION", base_reduction))
        for idx in moe_cfg.INSERT_BLOCKS:
            if idx < 0:
                idx = len(model.visual.transformer.resblocks) + idx
            block = model.visual.transformer.resblocks[idx]
            if block.moe_adapter is not None:
                raise RuntimeError(f"MoE adapter already exists at visual block {idx}.")
            block.moe_adapter = VisualMoEAdapter(
                d_model=d_model,
                num_experts=num_base_experts,
                reduction=base_reduction,
                incremental_reduction=incremental_reduction,
            )
        convert_weights(model)
    return model.eval()

