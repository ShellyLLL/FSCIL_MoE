"""Post-hoc Session-1 diagnostics for the unified FSCIL MoE architecture.

These utilities are inference/analysis only: they do not update model parameters,
optimizer state, checkpoints, or the production training objective.
"""
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


RESPONSE_METRICS = (
    "gate_logit", "gate", "residual_norm", "patch_residual_norm",
    "token_residual_norm", "relative_residual",
    "relative_patch_residual", "relative_token_residual",
)


def metric_summary(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return {
            "count": 0, "mean": None, "std": None, "median": None,
            "q05": None, "q25": None, "q75": None, "q90": None, "q95": None,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(values.median()),
        "q05": float(torch.quantile(values, 0.05)),
        "q25": float(torch.quantile(values, 0.25)),
        "q75": float(torch.quantile(values, 0.75)),
        "q90": float(torch.quantile(values, 0.90)),
        "q95": float(torch.quantile(values, 0.95)),
    }


def binary_auc(positive_mask, scores):
    """Mann-Whitney ROC-AUC with average ranks for ties; no extra dependency."""
    positive = np.asarray(positive_mask, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positive.size != scores.size or positive.size == 0 or not np.isfinite(scores).all():
        return None
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _set_masks(runner, disabled):
    disabled = {int(block): set(map(int, ids)) for block, ids in disabled.items()}
    for block, adapter in runner._moe_adapters():
        adapter.eval()
        adapter.set_inference_expert_mask(disabled.get(block, set()))


@contextmanager
def preserve_diagnostic_state(runner):
    """Restore model tensors, gradients, modes, masks and RNG after diagnostics."""
    model = runner.model
    modules = list(model.modules())
    parameters = list(model.parameters())
    adapters = [adapter for _, adapter in runner._moe_adapters()]

    module_modes = [(module, bool(module.training)) for module in modules]
    parameter_state = [
        (
            parameter,
            bool(parameter.requires_grad),
            None if parameter.grad is None else parameter.grad.detach().cpu().clone(),
        )
        for parameter in parameters
    ]
    model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    adapter_masks = [
        (
            adapter,
            frozenset(getattr(adapter, "_inference_disabled_expert_ids", frozenset())),
        )
        for adapter in adapters
    ]

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        yield
    finally:
        # Restore the complete model state, not merely the adapter subset.
        # This protects all parameters and persistent buffers against accidental
        # writes introduced by future diagnostic extensions.
        with torch.no_grad():
            model.load_state_dict(model_state, strict=True)
            for adapter, mask in adapter_masks:
                adapter._inference_disabled_expert_ids = mask

            for parameter, requires_grad, gradient in parameter_state:
                parameter.requires_grad_(requires_grad)
                parameter.grad = (
                    None if gradient is None
                    else gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
                )

            # Restore mixed train/eval states exactly; model.train(flag) would
            # flatten intentionally mixed adapter/dropout modes.
            for module, training in module_modes:
                module.training = training

        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def collect_features_and_expert_stats(
    runner, loader, current_session, base_ids, novel_ids
):
    """Capture actual full-token outputs of newly born experts with forward hooks.

    Labels are used only after the model forward pass for post-hoc grouping.
    No expert is re-run on a CLS-only surrogate and no model behavior is changed.
    """
    model = runner.model_without_dp
    feature_rows, label_rows = [], []
    base_ids = torch.as_tensor(base_ids, dtype=torch.long).detach().cpu().reshape(-1)
    novel_ids = torch.as_tensor(novel_ids, dtype=torch.long).detach().cpu().reshape(-1)

    stats, raw_stats, captured, hook_handles = {}, {}, {}, []
    adapters = runner._moe_adapters()
    for block, adapter in adapters:
        born_ids = [
            index for index, birth in enumerate(adapter.expert_birth_sessions)
            if int(birth) == int(current_session) and index >= adapter.base_expert_count
        ]
        for expert_id in born_ids:
            key = (int(block), int(expert_id))
            stats[key] = {
                group: {metric: [] for metric in RESPONSE_METRICS}
                for group in ("all", "base", "novel")
            }
            raw_stats[key] = {metric: [] for metric in RESPONSE_METRICS}
            raw_stats[key]["labels"] = []

            def capture_expert_output(module, inputs, output, capture_key=key):
                if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                    raise RuntimeError("Expert diagnostic hook received unsupported input/output.")
                captured[capture_key] = (
                    inputs[0].detach().float(), output.detach().float()
                )

            hook_handles.append(
                adapter.experts[expert_id].register_forward_hook(capture_expert_output)
            )

    try:
        model.eval()
        for batch in loader:
            captured.clear()
            images, labels = runner.parse_batch(batch)
            features, aux_list = model.extract_img_feature_train(images)
            labels_cpu = labels.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            if features.size(0) != labels_cpu.numel():
                raise RuntimeError("Feature and label batch sizes do not align.")

            base_mask = torch.isin(labels_cpu, base_ids)
            novel_mask = torch.isin(labels_cpu, novel_ids)
            if bool((base_mask & novel_mask).any()):
                raise RuntimeError("Base and Novel diagnostic class IDs overlap.")
            if bool((~(base_mask | novel_mask)).any()):
                unknown = labels_cpu[~(base_mask | novel_mask)].unique().tolist()
                raise RuntimeError(f"Test loader contains unassigned diagnostic labels: {unknown}.")

            group_masks = {
                "all": torch.ones_like(base_mask, dtype=torch.bool),
                "base": base_mask,
                "novel": novel_mask,
            }
            feature_rows.append(F.normalize(features.float(), dim=-1).cpu())
            label_rows.append(labels_cpu)
            if len(adapters) != len(aux_list):
                raise RuntimeError("MoE adapter order does not match auxiliary outputs.")

            for (block, adapter), aux in zip(adapters, aux_list):
                if aux is None:
                    raise RuntimeError(f"Missing MoE diagnostics for transformer block {block}.")
                route_weights = aux["route_weights"].float()
                route_logits = aux["logits"].float()
                born_ids = [
                    index for index, birth in enumerate(adapter.expert_birth_sessions)
                    if int(birth) == int(current_session) and index >= adapter.base_expert_count
                ]
                for expert_id in born_ids:
                    key = (int(block), int(expert_id))
                    if key not in captured:
                        raise RuntimeError(f"Expert hook did not capture B{block}_E{expert_id}.")
                    expert_input, expert_output = captured[key]
                    if expert_input.ndim != 3 or expert_output.shape != expert_input.shape:
                        raise RuntimeError(
                            f"B{block}_E{expert_id} tensors must align as [tokens,batch,dim]."
                        )

                    gate = route_weights[:, expert_id]
                    gate_logit = route_logits[:, expert_id]
                    contribution = (
                        adapter.residual_scale.float()
                        * expert_output
                        * gate.reshape(1, -1, 1)
                    )
                    # The router gate is retained, but a masked expert contributes exactly zero.
                    if expert_id in getattr(adapter, "_inference_disabled_expert_ids", frozenset()):
                        contribution = torch.zeros_like(contribution)

                    cls_residual = torch.linalg.vector_norm(contribution[0], dim=-1)
                    cls_input = torch.linalg.vector_norm(expert_input[0], dim=-1).clamp_min(1e-8)
                    if contribution.size(0) > 1:
                        patch_residual = contribution[1:].square().mean(dim=(0, 2)).sqrt()
                        patch_input = expert_input[1:].square().mean(dim=(0, 2)).sqrt().clamp_min(1e-8)
                    else:
                        patch_residual = cls_residual.new_zeros(cls_residual.shape)
                        patch_input = cls_input.new_ones(cls_input.shape)
                    token_residual = contribution.square().mean(dim=(0, 2)).sqrt()
                    token_input = expert_input.square().mean(dim=(0, 2)).sqrt().clamp_min(1e-8)

                    values = {
                        "gate_logit": gate_logit.detach().cpu(),
                        "gate": gate.detach().cpu(),
                        "residual_norm": cls_residual.detach().cpu(),
                        "patch_residual_norm": patch_residual.detach().cpu(),
                        "token_residual_norm": token_residual.detach().cpu(),
                        "relative_residual": (cls_residual / cls_input).detach().cpu(),
                        "relative_patch_residual": (patch_residual / patch_input).detach().cpu(),
                        "relative_token_residual": (token_residual / token_input).detach().cpu(),
                    }
                    raw_stats[key]["labels"].append(labels_cpu)
                    for metric_name, value in values.items():
                        raw_stats[key][metric_name].append(value)
                    for group, mask in group_masks.items():
                        for metric_name, value in values.items():
                            stats[key][group][metric_name].append(value[mask])
    finally:
        for handle in hook_handles:
            handle.remove()

    if not feature_rows:
        raise RuntimeError("The Session-1 test loader produced no samples.")
    features, labels = torch.cat(feature_rows), torch.cat(label_rows)
    summarized, raw_summarized = {}, {}
    for (block, expert_id), groups in stats.items():
        expert_name = f"B{block}_E{expert_id}"
        summarized[expert_name] = {}
        raw_summarized[expert_name] = {
            metric: torch.cat(chunks) if chunks else torch.empty(0)
            for metric, chunks in raw_stats[(block, expert_id)].items()
            if metric != "labels"
        }
        raw_summarized[expert_name]["labels"] = torch.cat(
            raw_stats[(block, expert_id)]["labels"]
        )
        for group, group_metrics in groups.items():
            values = {
                metric: metric_summary(torch.cat(chunks) if chunks else torch.empty(0))
                for metric, chunks in group_metrics.items()
            }
            values["sample_count"] = int(
                sum(chunk.numel() for chunk in group_metrics["gate"])
            )
            summarized[expert_name][group] = values
    return features, labels, summarized, raw_summarized


def response_posthoc(raw_stats, labels, predictions, base_ids, novel_ids):
    """Per-class response, correctness-conditioned response and discrimination."""
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().reshape(-1)
    predictions = torch.as_tensor(predictions, dtype=torch.long).cpu().reshape(-1)
    base_ids = torch.as_tensor(base_ids, dtype=torch.long).cpu().reshape(-1)
    novel_ids = torch.as_tensor(novel_ids, dtype=torch.long).cpu().reshape(-1)
    if labels.numel() != predictions.numel():
        raise ValueError("Predictions and labels must align for response diagnostics.")
    base_mask, novel_mask = torch.isin(labels, base_ids), torch.isin(labels, novel_ids)
    correct_mask = predictions.eq(labels)
    by_class, discrimination, correctness = {}, {}, {}
    class_ids = torch.unique(labels, sorted=True)

    for expert_name, raw in raw_stats.items():
        expert_classes, expert_correctness, expert_auc = {}, {}, {}
        for class_id in class_ids.tolist():
            mask = labels.eq(int(class_id))
            expert_classes[str(int(class_id))] = {
                "sample_count": int(mask.sum()),
                **{metric: metric_summary(raw[metric][mask]) for metric in RESPONSE_METRICS},
            }
        for group_name, mask in (("correct", correct_mask), ("incorrect", ~correct_mask)):
            expert_correctness[group_name] = {
                "sample_count": int(mask.sum()),
                **{metric: metric_summary(raw[metric][mask]) for metric in RESPONSE_METRICS},
            }

        for metric in ("gate", "residual_norm", "patch_residual_norm", "token_residual_norm"):
            scores = raw[metric].numpy()
            auc_row = {"base_vs_novel_auc": binary_auc(novel_mask.numpy(), scores)}
            novel_class_auc, base_class_auc, pair_auc = [], [], []
            for class_id in novel_ids.tolist():
                mask = base_mask | labels.eq(int(class_id))
                value = binary_auc(labels[mask].eq(int(class_id)).numpy(), scores[mask])
                if value is not None:
                    novel_class_auc.append(value)
            for class_id in base_ids.tolist():
                mask = novel_mask | labels.eq(int(class_id))
                value = binary_auc(torch.isin(labels[mask], novel_ids).numpy(), scores[mask])
                if value is not None:
                    base_class_auc.append(value)
            for novel_id in novel_ids.tolist():
                for base_id in base_ids.tolist():
                    mask = labels.eq(int(novel_id)) | labels.eq(int(base_id))
                    value = binary_auc(labels[mask].eq(int(novel_id)).numpy(), scores[mask])
                    if value is not None:
                        pair_auc.append(value)
            auc_row["novel_class_macro_auc"] = (
                float(np.mean(novel_class_auc)) if novel_class_auc else None
            )
            auc_row["base_class_macro_auc"] = (
                float(np.mean(base_class_auc)) if base_class_auc else None
            )
            auc_row["class_pair_macro_auc"] = float(np.mean(pair_auc)) if pair_auc else None
            expert_auc[metric] = auc_row

        by_class[expert_name] = expert_classes
        correctness[expert_name] = expert_correctness
        discrimination[expert_name] = expert_auc
    return by_class, discrimination, correctness


def classification_diagnostics(logits, labels, class_ids, base_ids, novel_ids):
    labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device).reshape(-1)
    class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device).reshape(-1)
    base_ids = torch.as_tensor(base_ids, dtype=torch.long, device=logits.device).reshape(-1)
    novel_ids = torch.as_tensor(novel_ids, dtype=torch.long, device=logits.device).reshape(-1)
    if logits.ndim != 2 or logits.shape != (labels.numel(), class_ids.numel()):
        raise ValueError("Global logits, labels and class ids are not aligned.")
    matches = labels[:, None].eq(class_ids[None, :])
    if not matches.sum(dim=1).eq(1).all():
        raise ValueError("Every diagnostic label must map to exactly one global logit column.")
    target_columns = matches.long().argmax(dim=1)
    predictions = class_ids[logits.argmax(dim=1)]
    true_logits = logits.gather(1, target_columns[:, None]).squeeze(1)
    competing_logits = logits.clone()
    competing_logits.scatter_(1, target_columns[:, None], float("-inf"))
    margins = true_logits - competing_logits.max(dim=1).values

    ordered_ids = [int(value) for value in class_ids.detach().cpu().tolist()]
    row_for_id = {class_id: index for index, class_id in enumerate(ordered_ids)}
    confusion = np.zeros((len(ordered_ids), len(ordered_ids)), dtype=np.int64)
    for true_id, predicted_id in zip(
        labels.detach().cpu().tolist(), predictions.detach().cpu().tolist()
    ):
        confusion[row_for_id[int(true_id)], row_for_id[int(predicted_id)]] += 1

    per_class = {}
    for class_id in ordered_ids:
        mask = labels.eq(class_id)
        per_class[str(class_id)] = {
            "sample_count": int(mask.sum().item()),
            "accuracy": float(predictions[mask].eq(labels[mask]).float().mean().item())
            if bool(mask.any()) else None,
            "margin": metric_summary(margins[mask]),
        }
    error_flow = {}
    for row_index, true_id in enumerate(ordered_ids):
        errors = {
            str(ordered_ids[column_index]): int(confusion[row_index, column_index])
            for column_index in range(len(ordered_ids))
            if column_index != row_index and confusion[row_index, column_index] > 0
        }
        if errors:
            error_flow[str(true_id)] = errors

    return {
        "per_class": per_class,
        "confusion_class_ids": ordered_ids,
        "confusion_matrix": confusion.tolist(),
        "error_flow": error_flow,
        "margin_all": metric_summary(margins),
        "margin_base": metric_summary(margins[torch.isin(labels, base_ids)]),
        "margin_novel": metric_summary(margins[torch.isin(labels, novel_ids)]),
    }, predictions.detach().cpu()


def prototype_generalization_diagnostics(
    support_features, support_labels, support_loo_prototypes,
    test_features, test_labels, global_prototypes, class_ids,
    old_class_count, base_ids, novel_ids, logit_scale,
):
    """Compare support resubstitution, source-exclusive LOO and held-out test.

    The same raw visual prototype bank is used for support resubstitution and
    test. For each support query, every novel prototype is replaced by the
    source-exclusive LOO prototype built without that query's source id.
    """
    device = global_prototypes.device
    support_features = F.normalize(
        torch.as_tensor(support_features, device=device).float(), dim=-1
    )
    support_labels = torch.as_tensor(
        support_labels, device=device, dtype=torch.long
    ).reshape(-1)
    support_loo_prototypes = torch.as_tensor(
        support_loo_prototypes, device=device
    ).float()
    test_features = F.normalize(
        torch.as_tensor(test_features, device=device).float(), dim=-1
    )
    test_labels = torch.as_tensor(
        test_labels, device=device, dtype=torch.long
    ).reshape(-1)
    global_prototypes = F.normalize(
        torch.as_tensor(global_prototypes, device=device).float(), dim=-1
    )
    class_ids = torch.as_tensor(class_ids, device=device, dtype=torch.long).reshape(-1)
    base_ids = torch.as_tensor(base_ids, device=device, dtype=torch.long).reshape(-1)
    novel_ids = torch.as_tensor(novel_ids, device=device, dtype=torch.long).reshape(-1)

    if global_prototypes.ndim != 2 or global_prototypes.size(0) != class_ids.numel():
        raise ValueError("Global visual prototypes and class ids must align.")
    if torch.unique(class_ids).numel() != class_ids.numel():
        raise ValueError("Global class ids must be unique.")
    if support_features.ndim != 2 or support_features.size(0) != support_labels.numel():
        raise ValueError("Support features and labels must align.")
    if test_features.ndim != 2 or test_features.size(0) != test_labels.numel():
        raise ValueError("Test features and labels must align.")
    if support_features.size(1) != global_prototypes.size(1) or test_features.size(1) != global_prototypes.size(1):
        raise ValueError("Support, test and prototype feature dimensions must agree.")
    if not 0 <= int(old_class_count) <= class_ids.numel():
        raise ValueError("old_class_count is outside the global classifier.")
    novel_count = class_ids.numel() - int(old_class_count)
    if support_loo_prototypes.shape != (
        support_features.size(0), novel_count, global_prototypes.size(1)
    ):
        raise ValueError("Support LOO prototypes must have shape [support, novel_classes, dim].")
    if not torch.isfinite(support_features).all() or not torch.isfinite(test_features).all():
        raise ValueError("Support and test features must be finite.")
    if not torch.isfinite(global_prototypes).all() or not torch.isfinite(support_loo_prototypes).all():
        raise ValueError("Visual prototypes must be finite.")
    if not torch.isin(support_labels, novel_ids).all():
        raise ValueError("Support LOO diagnostics expect current-session novel labels only.")
    if not torch.isin(test_labels, class_ids).all():
        raise ValueError("Test labels contain classes outside the global classifier.")
    if not torch.isin(base_ids, class_ids).all() or not torch.isin(novel_ids, class_ids).all():
        raise ValueError("Base/Novel IDs must belong to the global classifier.")

    support_count = support_features.size(0)
    support_resub_logits = float(logit_scale) * support_features @ global_prototypes.t()
    support_prototypes = torch.cat(
        [
            global_prototypes[:int(old_class_count)].unsqueeze(0).expand(
                support_count, -1, -1
            ),
            F.normalize(support_loo_prototypes, dim=-1),
        ],
        dim=1,
    )
    support_loo_logits = float(logit_scale) * torch.einsum(
        "bd,bcd->bc", support_features, support_prototypes
    )
    test_logits = float(logit_scale) * test_features @ global_prototypes.t()

    resub_diag, resub_predictions = classification_diagnostics(
        support_resub_logits, support_labels, class_ids, base_ids, novel_ids
    )
    loo_diag, loo_predictions = classification_diagnostics(
        support_loo_logits, support_labels, class_ids, base_ids, novel_ids
    )
    test_diag, test_predictions = classification_diagnostics(
        test_logits, test_labels, class_ids, base_ids, novel_ids
    )

    support_novel_mask = torch.isin(support_labels, novel_ids)
    test_base_mask = torch.isin(test_labels, base_ids)
    test_novel_mask = torch.isin(test_labels, novel_ids)

    def accuracy(predictions, labels, mask):
        if not bool(mask.any()):
            return None
        return float(predictions[mask].eq(labels[mask]).float().mean().item())

    support_loo_novel = accuracy(loo_predictions, support_labels, support_novel_mask)
    test_novel = accuracy(test_predictions, test_labels, test_novel_mask)
    per_class_gap = {}
    for class_id in novel_ids.detach().cpu().tolist():
        support_mask = support_labels.eq(int(class_id))
        test_mask = test_labels.eq(int(class_id))
        support_acc = accuracy(loo_predictions, support_labels, support_mask)
        test_acc = accuracy(test_predictions, test_labels, test_mask)
        per_class_gap[str(int(class_id))] = {
            "support_loo_accuracy": support_acc,
            "test_accuracy": test_acc,
            "support_minus_test": (
                None if support_acc is None or test_acc is None
                else float(support_acc - test_acc)
            ),
        }

    return {
        "protocol": (
            "Raw visual global prototypes; support queries use source-exclusive LOO "
            "novel prototypes, while test queries use the same full support-derived "
            "prototype bank. Semantic prototype calibration is intentionally excluded."
        ),
        "support_sample_count": int(support_count),
        "test_sample_count": int(test_labels.numel()),
        "support_resubstitution": {
            "novel_accuracy": accuracy(
                resub_predictions, support_labels, support_novel_mask
            ),
            "classification": resub_diag,
        },
        "support_loo": {
            "novel_accuracy": support_loo_novel,
            "classification": loo_diag,
        },
        "test": {
            "base_accuracy": accuracy(test_predictions, test_labels, test_base_mask),
            "novel_accuracy": test_novel,
            "classification": test_diag,
        },
        "novel_support_loo_minus_test_accuracy": (
            None if support_loo_novel is None or test_novel is None
            else float(support_loo_novel - test_novel)
        ),
        "novel_per_class_gap": per_class_gap,
        "interpretation_limit": (
            "Support sources participated in incremental optimization. LOO removes "
            "the query source from prototype construction, but does not make the "
            "support query an independent validation example. Interpret the "
            "support-test gap as an overfitting indicator, not a standalone proof."
        ),
    }


@torch.no_grad()
def collect_support_generalization_diagnostics(
    runner, support_loader, test_loader, class_index, old_states,
    base_ids, novel_ids, logit_scale,
):
    """Measure source-exclusive support LOO and held-out test accuracy."""
    with preserve_diagnostic_state(runner):
        runner.model.eval()
        _set_masks(runner, {})
        bank = runner._build_incremental_bank(support_loader, class_index, old_states)
        if not bool(bank["loo_valid"].all().item()):
            raise RuntimeError("Support generalization diagnostics require valid source-exclusive LOO.")
        test_features, test_labels, _, _ = runner.model_without_dp.inference_all_img_feature(
            test_loader
        )
        return prototype_generalization_diagnostics(
            support_features=bank["features"],
            support_labels=bank["labels"],
            support_loo_prototypes=bank["novel_loo"],
            test_features=test_features,
            test_labels=test_labels,
            global_prototypes=bank["global_full"],
            class_ids=bank["global_ids"],
            old_class_count=int(bank["old_class_count"]),
            base_ids=base_ids,
            novel_ids=novel_ids,
            logit_scale=logit_scale,
        )


@torch.no_grad()
def collect_historical_anchor_responses(runner, old_states, expanded):
    """Evaluate the three-point per-class moment proxies used by the PD loss."""
    result = {}
    for block, expert_ids in sorted(expanded.items()):
        adapter = runner._adapter_by_block(block)
        adapter.eval()
        for expert_id in expert_ids:
            expert_name = f"B{block}_E{expert_id}"
            per_class = {}
            for state in old_states:
                anchors_by_block = state.get("anchor_by_block", {})
                bank = anchors_by_block.get(int(block), anchors_by_block.get(str(block)))
                if bank is None:
                    raise KeyError(f"Historical state lacks anchor statistics for block {block}.")
                class_ids = torch.as_tensor(bank["class_ids"], dtype=torch.long).reshape(-1)
                route_points = adapter._anchor_points(
                    bank["route_mean"], bank["route_var"]
                ).to(runner.device)
                expert_points = adapter._anchor_points(
                    bank["expert_mean"], bank["expert_var"]
                ).to(runner.device)
                if route_points.shape != expert_points.shape:
                    raise ValueError("Historical route/expert moment proxies are not aligned.")
                logits = adapter.router.columns[expert_id](route_points).reshape(-1)
                gates = torch.sigmoid(logits)
                outputs = adapter.experts[expert_id](expert_points.unsqueeze(0)).squeeze(0)
                residual = adapter.residual_scale.float() * gates.unsqueeze(-1) * outputs.float()
                residual_norm = torch.linalg.vector_norm(residual, dim=-1)
                relative = residual_norm / torch.linalg.vector_norm(
                    expert_points.float(), dim=-1
                ).clamp_min(1e-8)
                count = class_ids.numel()
                values = {
                    "gate_logit": logits.reshape(3, count).transpose(0, 1),
                    "gate": gates.reshape(3, count).transpose(0, 1),
                    "residual_norm": residual_norm.reshape(3, count).transpose(0, 1),
                    "relative_residual": relative.reshape(3, count).transpose(0, 1),
                }
                for row, class_id in enumerate(class_ids.tolist()):
                    entry = per_class.setdefault(int(class_id), {key: [] for key in values})
                    for key, matrix in values.items():
                        entry[key].append(matrix[row].detach().cpu())
            result[expert_name] = {}
            for class_id, values in per_class.items():
                result[expert_name][str(class_id)] = {
                    "anchor_points_per_class": 3,
                    **{key: metric_summary(torch.cat(chunks)) for key, chunks in values.items()},
                }
    return result


def compare_anchor_responses_to_base_test(anchor_response, base_test_by_class, base_ids):
    comparison = {}
    for expert_name, classes in anchor_response.items():
        comparison[expert_name] = {}
        actual_classes = base_test_by_class.get(expert_name, {})
        for class_id in torch.as_tensor(base_ids).reshape(-1).tolist():
            key = str(int(class_id))
            anchor, actual = classes.get(key), actual_classes.get(key)
            if anchor is None or actual is None:
                comparison[expert_name][key] = {"available": False}
                continue
            row = {"available": True, "anchor_points_per_class": 3}
            for field in ("gate", "residual_norm", "relative_residual"):
                anchor_mean, test_mean = anchor[field]["mean"], actual[field]["mean"]
                row[field] = {
                    "anchor_mean": anchor_mean,
                    "base_test_mean": test_mean,
                    "test_minus_anchor": (
                        None if anchor_mean is None or test_mean is None
                        else float(test_mean - anchor_mean)
                    ),
                }
            comparison[expert_name][key] = row
    return comparison


def _gradient_vector(gradients, parameters):
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(gradient.detach().float().reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def _gradient_relation(
    ce_gradient, pd_gradient, pd_weight, global_clip_coefficient,
    global_combined_norm, clip_max_norm,
):
    ce_norm = torch.linalg.vector_norm(ce_gradient)
    pd_norm = torch.linalg.vector_norm(pd_gradient)
    weighted_pd = float(pd_weight) * pd_gradient
    weighted_pd_norm = torch.linalg.vector_norm(weighted_pd)
    combined = ce_gradient + weighted_pd
    group_combined_norm = torch.linalg.vector_norm(combined)

    denominator = ce_norm * pd_norm
    weighted_denominator = ce_norm * weighted_pd_norm
    cosine = (
        float(torch.dot(ce_gradient, pd_gradient) / denominator)
        if bool((denominator > 0).item()) else None
    )
    weighted_cosine = (
        float(torch.dot(ce_gradient, weighted_pd) / weighted_denominator)
        if bool((weighted_denominator > 0).item()) else None
    )
    return {
        "ce_gradient_norm": float(ce_norm),
        "pd_gradient_norm": float(pd_norm),
        "weighted_pd_gradient_norm": float(weighted_pd_norm),
        "weighted_pd_to_ce_norm_ratio": (
            None if not bool((ce_norm > 0).item())
            else float(weighted_pd_norm / ce_norm)
        ),
        "ce_pd_cosine": cosine,
        "ce_weighted_pd_cosine": weighted_cosine,
        "ce_pd_dot": float(torch.dot(ce_gradient, pd_gradient)),
        "ce_weighted_pd_dot": float(torch.dot(ce_gradient, weighted_pd)),
        "combined_gradient_norm_pre_clip": float(group_combined_norm),
        "combined_gradient_norm_post_clip": float(
            group_combined_norm * float(global_clip_coefficient)
        ),
        "global_combined_gradient_norm_pre_clip": float(global_combined_norm),
        "global_combined_gradient_norm_post_clip": float(
            global_combined_norm * float(global_clip_coefficient)
        ),
        "global_clip_coefficient": float(global_clip_coefficient),
        "clip_max_norm": float(clip_max_norm),
    }


def gradient_objective_diagnostics(
    ce_gradient, pd_gradient, pd_weight, clip_max_norm=1.0,
):
    """Exact weighted CE+PD vector and PyTorch-style global L2 clip coefficient."""
    ce_gradient = torch.as_tensor(ce_gradient).float().reshape(-1)
    pd_gradient = torch.as_tensor(pd_gradient).float().reshape(-1)
    if ce_gradient.shape != pd_gradient.shape:
        raise ValueError("CE and PD gradient vectors must have identical shapes.")
    if not torch.isfinite(ce_gradient).all() or not torch.isfinite(pd_gradient).all():
        raise ValueError("CE and PD gradients must be finite.")
    if not np.isfinite(float(pd_weight)):
        raise ValueError("PD objective weight must be finite.")
    if not np.isfinite(float(clip_max_norm)) or float(clip_max_norm) <= 0:
        raise ValueError("Gradient clip max_norm must be finite and positive.")

    combined = ce_gradient + float(pd_weight) * pd_gradient
    combined_norm = torch.linalg.vector_norm(combined)
    coefficient = min(
        1.0, float(clip_max_norm) / (float(combined_norm.item()) + 1e-6)
    )
    return _gradient_relation(
        ce_gradient, pd_gradient, float(pd_weight), coefficient,
        float(combined_norm.item()), float(clip_max_norm),
    )


def _aggregate_gradient_rows(rows):
    fields = (
        "ce_gradient_norm", "pd_gradient_norm", "weighted_pd_gradient_norm",
        "weighted_pd_to_ce_norm_ratio", "ce_pd_cosine", "ce_weighted_pd_cosine",
        "ce_pd_dot", "ce_weighted_pd_dot", "combined_gradient_norm_pre_clip",
        "combined_gradient_norm_post_clip", "global_combined_gradient_norm_pre_clip",
        "global_combined_gradient_norm_post_clip", "global_clip_coefficient",
        "clip_max_norm",
    )
    output = {}
    for group in ("expert", "router", "combined"):
        output[group] = {}
        for field in fields:
            values = [
                row[group][field] for row in rows
                if row[group][field] is not None
            ]
            output[group][field] = metric_summary(values)
    return output


def collect_gradient_diagnostics(
    runner, support_loader, class_index, old_states, expanded,
    current_session=1, max_batches=8, dropout_repeats=5, clip_max_norm=1.0,
):
    """Measure the actual weighted CE+PD functional gradient without optimizer steps.

    CE and weighted PD gradients are combined over the newest expert/router
    parameters before applying the same global norm clip used in training.
    The RD objective belongs to the separate descriptor parameter group and is
    reported as out-of-scope for this functional-gradient comparison.
    """
    output = {}
    with preserve_diagnostic_state(runner):
        if hasattr(runner.data_manager, "get_support_view_dataloader"):
            gradient_loader = runner.data_manager.get_support_view_dataloader(
                int(current_session), view_id=0, mode="train"
            )
        else:
            gradient_loader = support_loader

        incremental_cfg = runner.cfg.TRAINER.BiMC.INCREMENTAL
        pd_weight = float(getattr(incremental_cfg, "OLD_SUPPRESSION_WEIGHT", 1.0))
        rd_weight = float(getattr(incremental_cfg, "RD_WEIGHT", 1.0))
        if not np.isfinite(pd_weight) or not np.isfinite(rd_weight):
            raise ValueError("Incremental objective weights must be finite.")
        if not np.isfinite(float(clip_max_norm)) or float(clip_max_norm) <= 0:
            raise ValueError("Gradient clip max_norm must be finite and positive.")

        for block in sorted(expanded):
            disabled_later = {
                later_block: ids for later_block, ids in expanded.items()
                if int(later_block) > int(block)
            }
            _set_masks(runner, disabled_later)
            runner.model.eval()
            # Rebuild the LOO classifier under the topology that existed when
            # this block was trained (later same-Session expansions are absent).
            bank = runner._build_incremental_bank(
                support_loader, class_index, old_states
            )
            adapter = runner._adapter_by_block(block)
            adapter.freeze_all()
            expert_parameters = list(adapter.experts[-1].parameters())
            router_parameters = list(adapter.router.columns[-1].parameters())
            functional_parameters = expert_parameters + router_parameters
            for parameter in functional_parameters:
                parameter.requires_grad_(True)
            adapter.eval()
            adapter.experts[-1].train()
            anchors = runner._all_old_anchor_sets(old_states, block)
            if any(anchors.get(key) is None for key in (
                "route_mean", "route_var", "expert_mean", "expert_var"
            )):
                raise RuntimeError(
                    f"Block {block} lacks historical anchors for PD gradient analysis."
                )

            batch_rows, dropout_losses = [], []
            for batch_index, batch in enumerate(gradient_loader):
                if batch_index >= int(max_batches):
                    break
                images, labels = runner.parse_batch(batch)
                source_ids = runner._batch_source_ids(
                    batch, labels.numel(), labels.device
                )
                logits, targets = runner._global_incremental_logits(
                    images, labels, source_ids, bank,
                    float(runner.cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE),
                )
                classification = F.cross_entropy(logits, targets)
                old_suppression = adapter.old_feature_preservation_loss(
                    anchors["route_mean"], anchors["route_var"],
                    anchors["expert_mean"], anchors["expert_var"],
                )
                ce_gradients = torch.autograd.grad(
                    classification, functional_parameters,
                    retain_graph=True, allow_unused=True,
                )
                pd_gradients = torch.autograd.grad(
                    old_suppression, functional_parameters,
                    retain_graph=False, allow_unused=True,
                )
                expert_count = len(expert_parameters)
                ce_expert = _gradient_vector(
                    ce_gradients[:expert_count], expert_parameters
                )
                pd_expert = _gradient_vector(
                    pd_gradients[:expert_count], expert_parameters
                )
                ce_router = _gradient_vector(
                    ce_gradients[expert_count:], router_parameters
                )
                pd_router = _gradient_vector(
                    pd_gradients[expert_count:], router_parameters
                )
                ce_combined = torch.cat([ce_expert, ce_router])
                pd_combined = torch.cat([pd_expert, pd_router])
                weighted_combined = ce_combined + pd_weight * pd_combined
                global_norm = float(torch.linalg.vector_norm(weighted_combined).item())
                clip_coefficient = min(
                    1.0, float(clip_max_norm) / (global_norm + 1e-6)
                )
                batch_rows.append({
                    "expert": _gradient_relation(
                        ce_expert, pd_expert, pd_weight, clip_coefficient,
                        global_norm, clip_max_norm,
                    ),
                    "router": _gradient_relation(
                        ce_router, pd_router, pd_weight, clip_coefficient,
                        global_norm, clip_max_norm,
                    ),
                    "combined": _gradient_relation(
                        ce_combined, pd_combined, pd_weight, clip_coefficient,
                        global_norm, clip_max_norm,
                    ),
                    "classification_loss": float(classification.detach()),
                    "pd_loss": float(old_suppression.detach()),
                    "weighted_pd_loss": float(pd_weight * old_suppression.detach()),
                    "functional_total_loss": float(
                        classification.detach() + pd_weight * old_suppression.detach()
                    ),
                })
                with torch.no_grad():
                    for _ in range(max(1, int(dropout_repeats))):
                        value = adapter.old_feature_preservation_loss(
                            anchors["route_mean"], anchors["route_var"],
                            anchors["expert_mean"], anchors["expert_var"],
                        )
                        dropout_losses.append(float(value))

            if not batch_rows:
                raise RuntimeError(
                    f"No batches available for block {block} gradient diagnostics."
                )
            pd_summary = metric_summary(dropout_losses)
            pd_mean = pd_summary["mean"]
            pd_summary["coefficient_of_variation"] = (
                None if pd_mean is None or abs(pd_mean) < 1e-12
                else float(pd_summary["std"] / abs(pd_mean))
            )
            output[str(block)] = {
                "expert_id": int(expanded[block][-1]),
                "disabled_later_session_blocks": sorted(
                    int(value) for value in disabled_later
                ),
                "batch_count": len(batch_rows),
                "objective_weights": {
                    "ce": 1.0,
                    "pd": pd_weight,
                    "rd": rd_weight,
                },
                "functional_gradient_clip": {
                    "parameter_group": "newest expert + newest router column",
                    "max_norm": float(clip_max_norm),
                    "applied_to": "gradient(CE + pd_weight * PD)",
                    "rd_parameter_group": "newest descriptor trainable parameters",
                    "rd_clip_max_norm": 1.0,
                    "rd_gradient_included_in_functional_comparison": False,
                },
                "gradient_alignment": _aggregate_gradient_rows(batch_rows),
                "classification_loss": metric_summary(
                    [row["classification_loss"] for row in batch_rows]
                ),
                "pd_loss": metric_summary([row["pd_loss"] for row in batch_rows]),
                "weighted_pd_loss": metric_summary(
                    [row["weighted_pd_loss"] for row in batch_rows]
                ),
                "functional_total_loss": metric_summary(
                    [row["functional_total_loss"] for row in batch_rows]
                ),
                "pd_dropout_repeatability": pd_summary,
            }
    return output
