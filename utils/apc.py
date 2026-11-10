"""Few-shot calibrated APC expansion decisions for the Risk-Stable FSCIL runner.

The calibration simulates the exact support protocol used by incremental
candidate evaluation: an S-shot pseudo-novel class is cross-fitted into
(S-1)-shot query prototypes, while historical classes retain S-shot
prototypes. The null statistic is the maximum class confusion over the
current task, with competitor cardinality matched to the real session.
"""

import math
import torch
import torch.nn.functional as F


def _normalize(values):
    return F.normalize(torch.as_tensor(values).float(), dim=-1)


def _threshold_key(competitor_count, new_class_count):
    return f"{int(competitor_count)}:{int(new_class_count)}"


def _draw_competitors(pseudo_ids, base_count, competitor_count, generator):
    """Sample K base-class competitors without self-comparisons.

    Sampling is without replacement while K fits the base pool and with
    replacement only when the real FSCIL session has more competitors than
    the number of base classes available to estimate the null distribution.
    """
    episodes, new_count = pseudo_ids.shape
    available = int(base_count) - 1
    k = int(competitor_count)
    if k <= 0 or available <= 0:
        raise ValueError("APC needs a positive competitor count and at least two base classes.")
    if k <= available:
        keys = torch.rand(episodes, new_count, available, generator=generator)
        ranks = keys.argsort(dim=-1)[..., :k]
    else:
        ranks = torch.randint(0, available, (episodes, new_count, k), generator=generator)
    return ranks + ranks.ge(pseudo_ids.unsqueeze(-1)).long()


def _calibrate_layer(features, labels, class_ids, task_specs, shot, episodes,
                     alpha, layer_count, seed, device, batch_size=16):
    features = _normalize(features).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().reshape(-1)
    class_ids = [int(value) for value in class_ids]
    groups_cpu = [features[labels.eq(class_id)] for class_id in class_ids]
    if any(group.size(0) < shot for group in groups_cpu):
        counts = {class_id: int(group.size(0)) for class_id, group in zip(class_ids, groups_cpu)}
        raise ValueError(f"APC base calibration needs at least {shot} sources per class: {counts}")
    groups = [group.to(device) for group in groups_cpu]
    base_count = len(class_ids)
    if base_count < 3:
        raise ValueError("APC calibration requires at least three base classes.")

    fixed_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
    fixed_prototypes = []
    for group in groups_cpu:
        indices = torch.randperm(group.size(0), generator=fixed_generator)[:shot]
        fixed_prototypes.append(_normalize(group[indices].mean(dim=0)))
    fixed_prototypes = torch.stack(fixed_prototypes)

    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
    quantile = 1.0 - float(alpha) / max(1, int(layer_count))
    thresholds = {}
    calibration = {}
    specs = sorted({
        (int(item["competitor_count"]), int(item["new_class_count"]))
        for item in task_specs
    })

    for competitor_count, new_count in specs:
        if new_count < 1 or new_count >= base_count:
            raise ValueError(
                f"APC pseudo-new class count {new_count} must be in [1,{base_count - 1}]."
            )
        null_batches = []
        remaining = int(episodes)
        while remaining:
            current_batch = min(int(batch_size), remaining)
            pseudo_ids = torch.stack([
                torch.randperm(base_count, generator=generator)[:new_count]
                for _ in range(current_batch)
            ])
            sample_indices = [
                torch.stack([
                    torch.randperm(group.size(0), generator=generator)[:shot]
                    for _ in range(current_batch)
                ])
                for group in groups_cpu
            ]
            support = torch.stack([
                group[index.to(device)]
                for group, index in zip(groups, sample_indices)
            ], dim=1)
            pseudo_device = pseudo_ids.to(device)
            batch_rows = torch.arange(current_batch, device=device)[:, None]
            pseudo_support = support[batch_rows, pseudo_device]
            history_prototypes = _normalize(support.mean(dim=2))
            competitor_ids = _draw_competitors(
                pseudo_ids, base_count, competitor_count, generator
            ).to(device)
            fold_max_sum = torch.zeros(current_batch, new_count, device=device)

            for held_out in range(shot):
                query_prototypes = _normalize(
                    (pseudo_support.sum(dim=2) - pseudo_support[:, :, held_out, :])
                    / float(shot - 1)
                )
                candidate_prototypes = history_prototypes.clone()
                candidate_prototypes[batch_rows, pseudo_device] = query_prototypes
                similarities = torch.bmm(
                    query_prototypes, candidate_prototypes.transpose(1, 2)
                )
                selected = similarities.gather(2, competitor_ids).max(dim=-1).values
                fold_max_sum += selected

            episode_statistic = (fold_max_sum / float(shot)).max(dim=1).values
            null_batches.append(episode_statistic.detach().cpu())
            remaining -= current_batch

        null_values = torch.cat(null_batches)
        threshold = float(torch.quantile(null_values, quantile).item())
        key = _threshold_key(competitor_count, new_count)
        thresholds[key] = threshold
        calibration[key] = {
            "episodes": int(episodes),
            "null_count": int(null_values.numel()),
            "quantile": float(quantile),
            "threshold": threshold,
            "null_mean": float(null_values.mean().item()),
            "null_std": float(null_values.std(unbiased=False).item()),
        }

    return {
        "prototypes": {
            str(class_id): fixed_prototypes[index].detach().cpu()
            for index, class_id in enumerate(class_ids)
        },
        "thresholds": thresholds,
        "calibration": calibration,
    }


def build_few_shot_apc_reference(block_features, base_class_ids, task_specs,
                                 shot=5, episodes=512, alpha=0.05,
                                 seed=1701, device="cpu"):
    """Build 5-shot anchor prototypes and cross-fitted empirical APC thresholds.

    block_features maps each MoE block to the output of
    BiMC.collect_blockwise_cls_features. All features must be extracted
    with max_session=0 so every class is measured in one immutable anchor
    representation, regardless of later expert expansions.
    """
    shot, episodes = int(shot), int(episodes)
    alpha = float(alpha)
    if shot < 3:
        raise ValueError("Few-shot APC requires at least three support sources per class.")
    if episodes < 64:
        raise ValueError("Few-shot APC requires at least 64 calibration episodes.")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("APC alpha must be finite and strictly between zero and one.")
    if not block_features:
        raise ValueError("APC cannot be initialized without MoE block features.")
    if not task_specs:
        raise ValueError("APC needs at least one incremental task specification.")

    base_ids = [int(value) for value in base_class_ids]
    if len(base_ids) < 3 or len(set(base_ids)) != len(base_ids):
        raise ValueError("APC base class ids must contain at least three unique classes.")

    layers = {}
    layer_count = len(block_features)
    for block_idx, item in sorted(block_features.items(), key=lambda pair: int(pair[0])):
        if "cls_in" not in item or "labels" not in item:
            raise ValueError(f"APC block {block_idx} is missing CLS features or labels.")
        layers[str(int(block_idx))] = _calibrate_layer(
            item["cls_in"], item["labels"], base_ids, task_specs,
            shot, episodes, alpha, layer_count,
            int(seed) + 1009 * int(block_idx), device,
        )

    return {
        "version": 1,
        "anchor_encoder_version": 0,
        "shot": shot,
        "query_shot": shot - 1,
        "alpha": alpha,
        "episodes": episodes,
        "base_class_ids": base_ids,
        "layers": layers,
    }


def evaluate_few_shot_apc(fold_features, apc_state, new_class_ids, device="cpu"):
    """Return layer candidates using cross-fitted (S-1)-shot novel prototypes."""
    if not fold_features:
        raise ValueError("Few-shot APC needs at least one cross-fitting fold.")
    new_ids = [int(value) for value in new_class_ids]
    if not new_ids or len(set(new_ids)) != len(new_ids):
        raise ValueError("APC new class ids must be non-empty and unique.")
    shot = int(apc_state["shot"])
    if len(fold_features) != shot:
        raise ValueError(
            f"APC expects {shot} leave-one-out folds, received {len(fold_features)}."
        )
    output = {"candidate_blocks": [], "layers": {}, "shot": shot,
              "query_shot": shot - 1, "alpha": float(apc_state["alpha"])}

    for block_key, layer_state in sorted(apc_state["layers"].items(), key=lambda pair: int(pair[0])):
        block_idx = int(block_key)
        prototypes = layer_state["prototypes"]
        history_ids = sorted(int(value) for value in prototypes)
        if set(history_ids).intersection(new_ids):
            raise ValueError("APC memory already contains a current novel class.")
        history = _normalize(torch.stack([
            torch.as_tensor(prototypes[str(class_id)]) for class_id in history_ids
        ])).to(device)
        fold_scores = []
        for fold in fold_features:
            if block_idx not in fold:
                raise KeyError(f"APC fold is missing MoE block {block_idx}.")
            item = fold[block_idx]
            features = _normalize(item["cls_in"]).to(device)
            labels = torch.as_tensor(item["labels"], dtype=torch.long, device=device)
            new_prototypes = []
            for class_id in new_ids:
                class_features = features[labels.eq(class_id)]
                if class_features.size(0) != shot - 1:
                    raise ValueError(
                        f"APC fold has {class_features.size(0)} samples for class {class_id}; "
                        f"expected {shot - 1}."
                    )
                new_prototypes.append(_normalize(class_features.mean(dim=0)))
            new_prototypes = torch.stack(new_prototypes)
            all_prototypes = torch.cat([history, new_prototypes], dim=0)
            similarities = new_prototypes @ all_prototypes.t()
            rows = torch.arange(len(new_ids), device=device)
            similarities[rows, len(history_ids) + rows] = float("-inf")
            fold_scores.append(similarities.max(dim=1).values)

        per_class = torch.stack(fold_scores).mean(dim=0)
        competitor_count = len(history_ids) + len(new_ids) - 1
        key = _threshold_key(competitor_count, len(new_ids))
        if key not in layer_state["thresholds"]:
            raise KeyError(
                f"APC has no calibrated threshold for block {block_idx}, K={competitor_count}, "
                f"new_classes={len(new_ids)}. Rebuild from a Session 0 checkpoint."
            )
        threshold = float(layer_state["thresholds"][key])
        observed = float(per_class.max().item())
        candidate = observed > threshold
        output["layers"][block_key] = {
            "candidate": bool(candidate),
            "observed_max_confusion": observed,
            "threshold": threshold,
            "margin": observed - threshold,
            "competitor_count": competitor_count,
            "new_class_count": len(new_ids),
            "mean_confusion_by_class": {
                class_id: float(per_class[index].item())
                for index, class_id in enumerate(new_ids)
            },
            "calibration": dict(layer_state["calibration"][key]),
        }
        if candidate:
            output["candidate_blocks"].append(block_idx)

    output["candidate_blocks"].sort()
    return output


def update_few_shot_apc_memory(apc_state, full_support_features, new_class_ids):
    """Append one 5-shot anchor prototype per acquired class and MoE block."""
    new_ids = [int(value) for value in new_class_ids]
    if not new_ids or len(set(new_ids)) != len(new_ids):
        raise ValueError("APC memory update needs non-empty unique class ids.")
    shot = int(apc_state["shot"])
    for block_key, layer_state in apc_state["layers"].items():
        block_idx = int(block_key)
        if block_idx not in full_support_features:
            raise KeyError(f"Full support features are missing MoE block {block_idx}.")
        item = full_support_features[block_idx]
        features = _normalize(item["cls_in"]).cpu()
        labels = torch.as_tensor(item["labels"], dtype=torch.long).cpu()
        for class_id in new_ids:
            if str(class_id) in layer_state["prototypes"]:
                raise ValueError(f"APC class {class_id} was already committed to memory.")
            values = features[labels.eq(class_id)]
            if values.size(0) != shot:
                raise ValueError(
                    f"APC memory requires exactly {shot} support sources for class {class_id}; "
                    f"received {values.size(0)}."
                )
            layer_state["prototypes"][str(class_id)] = _normalize(values.mean(dim=0)).detach().cpu()
    return apc_state
