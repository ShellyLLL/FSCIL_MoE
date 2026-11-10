"""Prototype-confusion expansion calibrated only from real FSCIL data.

Session 0 supplies real base-class prototypes and a fixed per-layer nearest-
neighbour similarity quantile. Each incremental decision uses the actual
few-shot support of current classes and prototypes of previously seen classes.
No pseudo classes, synthetic sessions, resampling episodes, or support splits.
"""
import torch
import torch.nn.functional as F


def _normalize(features):
    return F.normalize(torch.as_tensor(features).float(), dim=-1, eps=1e-12)


def _class_prototypes(block_features, class_ids, expected_shot=None):
    features = _normalize(block_features["cls_in"])
    labels = torch.as_tensor(block_features["labels"], dtype=torch.long).reshape(-1)
    if features.ndim != 2 or features.size(0) != labels.numel():
        raise ValueError("APC features and labels must be aligned [N,D] tensors.")
    prototypes, counts = {}, {}
    for class_id in [int(value) for value in class_ids]:
        selected = features[labels.eq(class_id)]
        count = int(selected.size(0))
        if count == 0:
            raise ValueError(f"APC has no real support features for class {class_id}.")
        if expected_shot is not None and count != int(expected_shot):
            raise ValueError(
                f"APC expected {int(expected_shot)} real support samples for class "
                f"{class_id}, received {count} samples."
            )
        prototypes[class_id] = _normalize(selected.mean(dim=0)).detach().cpu()
        counts[class_id] = count
    return prototypes, counts


def build_few_shot_apc_reference(block_features, base_class_ids, quantile=0.95):
    """Calibrate fixed thresholds from actual Session-0 base class prototypes."""
    base_ids = [int(value) for value in base_class_ids]
    if len(base_ids) < 2 or len(set(base_ids)) != len(base_ids):
        raise ValueError("APC needs at least two unique base classes.")
    quantile = float(quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("APC quantile must be strictly between zero and one.")
    if not block_features:
        raise ValueError("APC cannot initialize without raw visual features.")
    layers = {}
    for block_idx, item in sorted(block_features.items(), key=lambda pair: int(pair[0])):
        prototypes, counts = _class_prototypes(item, base_ids)
        matrix = torch.stack([prototypes[class_id] for class_id in base_ids])
        similarities = matrix @ matrix.t()
        similarities.fill_diagonal_(float("-inf"))
        nearest = similarities.max(dim=1).values
        threshold = float(torch.quantile(nearest, quantile).item())
        layers[str(int(block_idx))] = {
            "prototypes": {str(class_id): prototypes[class_id] for class_id in base_ids},
            "threshold": threshold,
            "base_nearest_similarities": nearest.detach().cpu(),
            "base_nearest_mean": float(nearest.mean().item()),
            "base_nearest_std": float(nearest.std(unbiased=False).item()),
            "base_sample_counts": counts,
            "base_class_count": len(base_ids),
        }
    return {
        "version": 2,
        "feature_space": "frozen_pretrained_visual_encoder_without_moe",
        "quantile": quantile,
        "base_class_ids": base_ids,
        "seen_class_ids": list(base_ids),
        "layers": layers,
    }


def evaluate_few_shot_apc(block_features, apc_state, new_class_ids, expected_shot=5):
    """Compare actual current-session few-shot prototypes with prior seen classes."""
    new_ids = [int(value) for value in new_class_ids]
    if not new_ids or len(set(new_ids)) != len(new_ids):
        raise ValueError("APC needs non-empty, unique current-session class ids.")
    seen_ids = [int(value) for value in apc_state["seen_class_ids"]]
    if set(new_ids).intersection(seen_ids):
        raise ValueError("APC current classes overlap previously seen classes.")
    if set(block_features) != {int(key) for key in apc_state["layers"]}:
        raise ValueError("APC feature blocks do not match calibrated base blocks.")
    output = {
        "candidate_blocks": [], "layers": {},
        "new_class_ids": new_ids, "seen_class_ids": seen_ids,
        "support_shot": int(expected_shot), "quantile": float(apc_state["quantile"]),
    }
    for block_key, layer_state in sorted(
        apc_state["layers"].items(), key=lambda pair: int(pair[0])
    ):
        block_idx = int(block_key)
        new_prototypes, counts = _class_prototypes(
            block_features[block_idx], new_ids, expected_shot=expected_shot)
        history_ids = sorted(int(value) for value in layer_state["prototypes"])
        history = _normalize(torch.stack([
            torch.as_tensor(layer_state["prototypes"][str(class_id)])
            for class_id in history_ids
        ]))
        current = torch.stack([new_prototypes[class_id] for class_id in new_ids])
        similarities = current @ history.t()
        max_values, max_indices = similarities.max(dim=1)
        observed = float(max_values.max().item())
        threshold = float(layer_state["threshold"])
        candidate = observed > threshold
        output["layers"][block_key] = {
            "candidate": bool(candidate),
            "observed_max_confusion": observed,
            "threshold": threshold,
            "margin": observed - threshold,
            "nearest_seen_class_by_new_class": {
                new_id: history_ids[int(max_indices[index].item())]
                for index, new_id in enumerate(new_ids)
            },
            "confusion_by_new_class": {
                new_id: float(max_values[index].item())
                for index, new_id in enumerate(new_ids)
            },
            "support_counts": counts,
            "base_nearest_mean": float(layer_state["base_nearest_mean"]),
            "base_nearest_std": float(layer_state["base_nearest_std"]),
            "base_class_count": int(layer_state["base_class_count"]),
        }
        if candidate:
            output["candidate_blocks"].append(block_idx)
    output["candidate_blocks"].sort()
    return output


def update_few_shot_apc_memory(apc_state, real_support_features, new_class_ids,
                               expected_shot=5):
    """Commit only the actual support prototypes of a learned FSCIL session."""
    new_ids = [int(value) for value in new_class_ids]
    if not new_ids or len(set(new_ids)) != len(new_ids):
        raise ValueError("APC memory update needs non-empty unique class ids.")
    seen_ids = set(int(value) for value in apc_state["seen_class_ids"])
    if seen_ids.intersection(new_ids):
        raise ValueError("APC memory cannot commit a class more than once.")
    if set(real_support_features) != {int(key) for key in apc_state["layers"]}:
        raise ValueError("APC support blocks do not match calibrated base blocks.")
    additions = {}
    for block_key in apc_state["layers"]:
        block_idx = int(block_key)
        prototypes, _ = _class_prototypes(
            real_support_features[block_idx], new_ids, expected_shot=expected_shot)
        additions[block_key] = prototypes
    for block_key, prototypes in additions.items():
        layer_state = apc_state["layers"][block_key]
        for class_id, prototype in prototypes.items():
            layer_state["prototypes"][str(class_id)] = prototype
    apc_state["seen_class_ids"] = sorted(seen_ids.union(new_ids))
    return apc_state
