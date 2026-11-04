"""Parallel adapters on a fixed encoder, with immutable class scoring functions.

This module deliberately has no image encoder, task gate, or mutable global
softmax over experts. A class bank owns its routing function and prototypes.
New banks can reuse frozen experts without changing any previous class score.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


def unit(x):
    return F.normalize(x.float(), dim=-1)


def columns(labels, class_ids):
    match = labels[:, None].eq(class_ids[None, :])
    if not bool(match.sum(1).eq(1).all()):
        raise ValueError("Each label must identify exactly one classifier column")
    return match.long().argmax(1)


def balanced_loss(logits, labels, class_ids):
    target = columns(labels, class_ids)
    counts = torch.bincount(target, minlength=len(class_ids))
    weights = counts[target].float().reciprocal()
    losses = F.cross_entropy(logits, target, reduction="none")
    return (weights * losses).sum() / weights.sum()


class FeatureExpert(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        if not 0 < rank <= dim:
            raise ValueError("Expert rank must be in [1, feature_dim]")
        self.rank = int(rank)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z):
        return self.up(F.gelu(self.down(z.float())))


class FeatureRoute(nn.Module):
    """A fixed-size mixture of identity and parallel residual experts.

    An index is global to the expert pool, not a session identifier. All experts
    see the same frozen input. Adding a pool entry cannot resize an old router.
    """
    def __init__(self, dim, expert_ids=()):
        super().__init__()
        self.expert_ids = tuple(int(e) for e in expert_ids)
        self.gate = nn.Linear(dim, len(self.expert_ids) + 1) if expert_ids else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, z, experts):
        z = unit(z)
        if self.gate is None:
            return z
        weights = self.gate(z).softmax(-1)
        residual = torch.zeros_like(z)
        for k, expert_id in enumerate(self.expert_ids):
            residual = residual + weights[:, k + 1:k + 2] * experts[expert_id](z)
        return unit(z + residual)


class ClassBank(nn.Module):
    def __init__(self, route, class_ids, prototypes):
        super().__init__()
        self.route = route
        self.register_buffer("class_ids", class_ids.detach().long().clone())
        self.register_buffer("prototypes", unit(prototypes.detach()).clone())
        self.requires_grad_(False)


class GlobalFeatureMoE(nn.Module):
    def __init__(self, dim, scale=100.0):
        super().__init__()
        if scale <= 0:
            raise ValueError("Logit scale must be positive")
        self.dim, self.scale = int(dim), float(scale)
        self.experts = nn.ModuleList()
        self.banks = nn.ModuleList()

    @property
    def class_ids(self):
        if not self.banks:
            return torch.empty(0, dtype=torch.long)
        return torch.cat([b.class_ids for b in self.banks])

    def forward(self, z):
        if not self.banks:
            return z.new_empty((len(z), 0), dtype=torch.float32)
        # Keep each historical matmul unchanged as the class set grows.
        return torch.cat([self.scale * b.route(z, self.experts) @ b.prototypes.T
                          for b in self.banks], dim=1)

    def bind(self, route, class_ids, prototypes, new_expert=None):
        if torch.isin(class_ids.cpu(), self.class_ids.cpu()).any():
            raise ValueError("Cannot rebind an existing class")
        pool_size = len(self.experts) + int(new_expert is not None)
        if any(i < 0 or i >= pool_size for i in route.expert_ids):
            raise ValueError("Route references a missing expert")
        if len(class_ids) != len(prototypes) or len(torch.unique(class_ids)) != len(class_ids):
            raise ValueError("Invalid class bank")
        if prototypes.shape[1] != self.dim or not torch.isfinite(prototypes).all():
            raise ValueError("Invalid prototypes")
        bank = ClassBank(copy.deepcopy(route), class_ids, prototypes)
        if new_expert is not None:
            self.experts.append(copy.deepcopy(new_expert).requires_grad_(False))
        self.banks.append(bank)
        self.eval()

    def export(self):
        return {"dim": self.dim, "scale": self.scale,
                "ranks": [e.rank for e in self.experts],
                "banks": [{"expert_ids": b.route.expert_ids,
                           "class_ids": b.class_ids.detach().cpu(),
                           "prototypes": b.prototypes.detach().cpu()} for b in self.banks],
                "state": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def restore(cls, payload, device="cpu"):
        model = cls(payload["dim"], payload["scale"]).to(device)
        for rank in payload["ranks"]:
            model.experts.append(FeatureExpert(model.dim, rank).to(device))
        for bank in payload["banks"]:
            model.bind(FeatureRoute(model.dim, bank["expert_ids"]).to(device),
                       bank["class_ids"].to(device), bank["prototypes"].to(device))
        model.load_state_dict(payload["state"], strict=True)
        model.requires_grad_(False)
        return model.eval()


def fused_prototypes(route, experts, support, text, base, beta, visual_weight,
                     visual_tau, query_labels=None, query_sources=None):
    """Differentiable full/leave-source-out prototypes in the query's space.

    support/base are feature caches, never stale adapted features. Source IDs
    identify original images; all views of a training query are excluded.
    A validation query must already be absent from support and base inputs.
    """
    ids = support["class_ids"]
    h = route(support["features"], experts)
    membership = support["labels"][:, None].eq(ids[None, :]).float()
    sums, counts = membership.T @ h, membership.sum(0)
    if bool((counts == 0).any()):
        raise ValueError("Every support class needs a sample")
    visual = unit(sums / counts[:, None])
    if query_labels is not None:
        same_source = query_sources[:, None].eq(support["sources"][None, :])
        same_class = query_labels[:, None].eq(support["labels"][None, :])
        excluded = (same_source & same_class).float()
        own = columns(query_labels, ids)
        remain = counts[own] - excluded.sum(1)
        if bool((remain <= 0).any()):
            raise ValueError("LOO needs another source in every query class")
        own_proto = unit((sums[own] - excluded @ h) / remain[:, None])
        visual = visual.unsqueeze(0).expand(len(query_labels), -1, -1).clone()
        visual[torch.arange(len(query_labels), device=h.device), own] = own_proto
    if visual_weight:
        # Transform actual cached base features, not base means in another space.
        hb = route(base["features"], experts)
        base_ids = torch.unique(base["labels"], sorted=True)
        m = base["labels"][:, None].eq(base_ids[None, :]).float()
        reference = unit((m.T @ hb) / m.sum(0)[:, None])
        weights = (visual @ reference.T * visual_tau).softmax(-1)
        visual = unit((1 - visual_weight) * visual + visual_weight * unit(weights @ reference))
    return unit(beta * unit(text) + (1 - beta) * visual)
