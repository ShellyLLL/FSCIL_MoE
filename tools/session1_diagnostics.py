"""Post-hoc Session-1 diagnostics for the unified FSCIL MoE architecture.

These utilities are inference/analysis only: they do not update model parameters,
optimizer state, checkpoints, or the production training objective.
"""
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


def _gradient_relation(ce_gradient, pd_gradient):
    ce_norm = torch.linalg.vector_norm(ce_gradient)
    pd_norm = torch.linalg.vector_norm(pd_gradient)
    denominator = ce_norm * pd_norm
    cosine = (
        float(torch.dot(ce_gradient, pd_gradient) / denominator)
        if bool((denominator > 0).item()) else None
    )
    return {
        "ce_gradient_norm": float(ce_norm),
        "pd_gradient_norm": float(pd_norm),
        "ce_pd_cosine": cosine,
        "ce_pd_dot": float(torch.dot(ce_gradient, pd_gradient)),
    }


def _aggregate_gradient_rows(rows):
    output = {}
    for group in ("expert", "router", "combined"):
        output[group] = {}
        for field in ("ce_gradient_norm", "pd_gradient_norm", "ce_pd_cosine", "ce_pd_dot"):
            values = [row[group][field] for row in rows if row[group][field] is not None]
            output[group][field] = metric_summary(values)
    return output


def collect_gradient_diagnostics(
    runner, support_loader, class_index, old_states, expanded,
    current_session=1, max_batches=8, dropout_repeats=5,
):
    """Measure CE/PD gradients without optimizer steps, restoring RNG and model state."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    output = {}
    try:
        if hasattr(runner.data_manager, "get_support_view_dataloader"):
            gradient_loader = runner.data_manager.get_support_view_dataloader(
                int(current_session), view_id=0, mode="train"
            )
        else:
            gradient_loader = support_loader

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
                raise RuntimeError(f"Block {block} lacks historical anchors for PD gradient analysis.")

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
                ce_expert = _gradient_vector(
                    ce_gradients[:len(expert_parameters)], expert_parameters
                )
                pd_expert = _gradient_vector(
                    pd_gradients[:len(expert_parameters)], expert_parameters
                )
                ce_router = _gradient_vector(
                    ce_gradients[len(expert_parameters):], router_parameters
                )
                pd_router = _gradient_vector(
                    pd_gradients[len(expert_parameters):], router_parameters
                )
                ce_combined = torch.cat([ce_expert, ce_router])
                pd_combined = torch.cat([pd_expert, pd_router])
                batch_rows.append({
                    "expert": _gradient_relation(ce_expert, pd_expert),
                    "router": _gradient_relation(ce_router, pd_router),
                    "combined": _gradient_relation(ce_combined, pd_combined),
                    "classification_loss": float(classification.detach()),
                    "pd_loss": float(old_suppression.detach()),
                })
                with torch.no_grad():
                    for _ in range(max(1, int(dropout_repeats))):
                        value = adapter.old_feature_preservation_loss(
                            anchors["route_mean"], anchors["route_var"],
                            anchors["expert_mean"], anchors["expert_var"],
                        )
                        dropout_losses.append(float(value))

            if not batch_rows:
                raise RuntimeError(f"No batches available for block {block} gradient diagnostics.")
            pd_summary = metric_summary(dropout_losses)
            pd_mean = pd_summary["mean"]
            pd_summary["coefficient_of_variation"] = (
                None if pd_mean is None or abs(pd_mean) < 1e-12
                else float(pd_summary["std"] / abs(pd_mean))
            )
            output[str(block)] = {
                "expert_id": int(expanded[block][-1]),
                "disabled_later_session_blocks": sorted(int(value) for value in disabled_later),
                "batch_count": len(batch_rows),
                "gradient_alignment": _aggregate_gradient_rows(batch_rows),
                "classification_loss": metric_summary(
                    [row["classification_loss"] for row in batch_rows]
                ),
                "pd_loss": metric_summary([row["pd_loss"] for row in batch_rows]),
                "pd_dropout_repeatability": pd_summary,
            }
            adapter.freeze_all()
    finally:
        _set_masks(runner, {})
        for _, adapter in runner._moe_adapters():
            adapter.freeze_all()
        runner.model.eval()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
    return output
