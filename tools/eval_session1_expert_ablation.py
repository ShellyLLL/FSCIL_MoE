#!/usr/bin/env python
"""Counterfactual Session-1 expert ablation on an existing FSCIL checkpoint.

This script does not train or mutate checkpoint parameters. Each variant masks
only selected Session-1 expert residuals at inference, then rebuilds the novel
visual prototypes under that same encoder before global evaluation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

# Make the repository root importable when this file is executed directly as
# ``python tools/eval_session1_expert_ablation.py``. Python otherwise places
# only the ``tools`` directory at the front of sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from main import setup_cfg
from engine.engine import Runner
from tools.session1_diagnostics import (
    collect_features_and_expert_stats, collect_historical_anchor_responses,
    collect_gradient_diagnostics, compare_anchor_responses_to_base_test,
    metric_summary, response_posthoc, classification_diagnostics,
)


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def install_payload(runner, payload):
    model = runner.model_without_dp
    model.rebuild_moe_topology(payload["moe_topology"])
    model.load_state_dict(payload["model_state"], strict=True)
    for _, adapter in runner._moe_adapters():
        adapter.freeze_all()
        adapter.clear_inference_expert_mask()
    model.base_enhanced_visual_prototype = payload.get(
        "base_enhanced_visual_prototype"
    )
    if model.base_enhanced_visual_prototype is not None:
        model.base_enhanced_visual_prototype = model.base_enhanced_visual_prototype.to(
            runner.device
        )
    model.eval()


def set_masks(runner, disabled):
    disabled = {int(block): set(map(int, ids)) for block, ids in disabled.items()}
    for block, adapter in runner._moe_adapters():
        adapter.eval()
        adapter.set_inference_expert_mask(disabled.get(block, set()))


def accuracy_by_group(runner, logits, labels, class_ids, base_ids, novel_ids):
    result = runner.evaluator.calc_accuracy(
        logits, labels, 1, class_ids=class_ids
    )
    base_mask = torch.isin(labels, base_ids)
    novel_mask = torch.isin(labels, novel_ids)

    def accuracy(mask):
        if not bool(mask.any()):
            return float("nan")
        pred_columns = logits[mask].argmax(dim=1)
        pred_ids = class_ids[pred_columns]
        return float(pred_ids.eq(labels[mask]).float().mean().item() * 100.0)

    task_acc = [float(x) for x in result["task_acc"]]
    if len(task_acc) != 2:
        raise RuntimeError(f"Expected two task accuracies for Session 1, got {task_acc}.")
    return {
        "global_mean_acc": float(result["mean_acc"]),
        "task_acc": task_acc,
        "base_global_accuracy": task_acc[0],
        "novel_global_accuracy": task_acc[1],
        "base_sample_accuracy": accuracy(base_mask),
        "novel_sample_accuracy": accuracy(novel_mask),
        "error_flow": result.get("error_flow"),
    }


@torch.no_grad()
def evaluate_variant(runner, session0_states, support_loader, test_loader,
                     class_names, class_ids, disabled, base_ids, novel_ids):
    model = runner.model_without_dp
    set_masks(runner, disabled)

    # Rebuild the novel class visual prototype with the exact masked encoder.
    # Historical prototypes remain the immutable Session-0 checkpoint values.
    novel_state = runner._build_and_refine_task_state(
        class_names, support_loader, class_ids, session0_states
    )
    merged = runner.merge_dicts(session0_states + [novel_state])

    features, labels, expert_stats, raw_expert_stats = collect_features_and_expert_stats(
        runner, test_loader, current_session=1, base_ids=base_ids, novel_ids=novel_ids,
    )
    prototypes = F.normalize(merged["image_proto"].to(runner.device).float(), dim=-1)
    logits = 100.0 * features.to(runner.device) @ prototypes.t()
    class_tensor = torch.as_tensor(
        merged["class_ids"], device=runner.device, dtype=torch.long
    )
    metrics = accuracy_by_group(
        runner, logits, labels.to(runner.device), class_tensor,
        torch.as_tensor(base_ids, device=runner.device),
        torch.as_tensor(novel_ids, device=runner.device),
    )
    class_diagnostics, predictions = classification_diagnostics(
        logits, labels.to(runner.device), class_tensor,
        torch.as_tensor(base_ids, device=runner.device),
        torch.as_tensor(novel_ids, device=runner.device),
    )
    by_class, auc, correctness = response_posthoc(
        raw_expert_stats, labels, predictions, base_ids, novel_ids
    )
    metrics["expert_response"] = expert_stats
    metrics["expert_response_by_class"] = by_class
    metrics["expert_response_discrimination"] = auc
    metrics["expert_response_by_prediction_correctness"] = correctness
    metrics["classification_diagnostics"] = class_diagnostics
    metrics["prototype_calibration"] = {
        "alpha_mean": float(torch.as_tensor(novel_state["calibration_alpha"]).float().mean()),
        "semantic_confidence_mean": float(
            torch.as_tensor(novel_state["semantic_confidence"]).float().mean()
        ),
        "visual_reliability_mean": float(
            torch.as_tensor(novel_state["visual_prototype_reliability"]).float().mean()
        ),
    }
    return metrics, features, labels, merged, novel_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--session0", required=True)
    parser.add_argument("--session1", required=True)
    parser.add_argument("--output", default="diagnostics/session1_expert_ablation")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    overrides = [
        "TRAINER.BiMC.CHECKPOINT.RESUME", args.session1,
        "TRAINER.BiMC.END_SESSION", "1",
    ]
    if args.device:
        overrides += ["DEVICE.DEVICE_NAME", args.device]
    cfg = setup_cfg(args.data_cfg, args.train_cfg, overrides)
    runner = Runner(cfg)
    if runner._resume_completed_session != 1:
        raise RuntimeError("Expected Session-1 checkpoint to be loaded.")

    payload0, payload1 = load_payload(args.session0), load_payload(args.session1)
    if int(payload0["completed_session"]) != 0 or int(payload1["completed_session"]) != 1:
        raise ValueError("The supplied checkpoint files must be Session 00 and Session 01.")
    if payload0["support_manifest"] != payload1["support_manifest"]:
        raise ValueError("Session-0 and Session-1 checkpoints use different support manifests.")

    task_states0 = payload0["task_states"]
    task_states1 = payload1["task_states"]
    if len(task_states0) != 1 or len(task_states1) != 2:
        raise ValueError("Expected one Base state in Session 00 and two task states in Session 01.")

    base_ids = torch.as_tensor(task_states0[0]["class_ids"], dtype=torch.long)
    novel_ids = torch.as_tensor(task_states1[1]["class_ids"], dtype=torch.long)
    if set(base_ids.tolist()) & set(novel_ids.tolist()):
        raise ValueError("Base and Session-1 class IDs overlap.")

    support_loader = runner.data_manager.get_dataloader(
        1, source="train", mode="test", accumulate_past=False
    )
    test_loader = runner.data_manager.get_dataloader(
        1, source="test", mode="test"
    )
    class_names = np.array(runner.data_manager.class_names)[
        runner.data_manager.class_index_in_task[1]
    ].tolist()
    class_ids = runner.data_manager.class_index_in_task[1]

    install_payload(runner, payload1)
    expanded = {}
    for block, adapter in runner._moe_adapters():
        ids = [
            idx for idx, birth in enumerate(adapter.expert_birth_sessions)
            if int(birth) == 1 and idx >= adapter.base_expert_count
        ]
        if ids:
            expanded[block] = ids
    if not expanded:
        raise RuntimeError("Session-1 checkpoint contains no Session-1-born experts.")
    if set(expanded) != {8, 9}:
        raise RuntimeError(
            f"Expected Session-1 expansion in B8/B9 from the supplied diagnostics, got {expanded}."
        )
    if any(len(ids) != 1 for ids in expanded.values()):
        raise RuntimeError(f"Expected exactly one Session-1 expert per expanded block, got {expanded}.")

    variants = {
        "A_full": {},
        "B_disable_B8": {8: expanded[8]},
        "C_disable_B9": {9: expanded[9]},
        "D_disable_B8_B9": expanded,
    }

    # Establish the true pre-incremental feature reference on the exact same
    # deterministic Session-1 test loader, using the Session-00 model weights.
    install_payload(runner, payload0)
    base_feature_rows, base_label_rows = [], []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = runner.parse_batch(batch)
            base_feature_rows.append(F.normalize(
                runner.model_without_dp.extract_img_feature(images).float(), dim=-1
            ).cpu())
            base_label_rows.append(labels.cpu())
    session0_features = torch.cat(base_feature_rows)
    session0_labels = torch.cat(base_label_rows)
    install_payload(runner, payload1)

    rows, details, variant_features, variant_states = [], {}, {}, {}
    for name, masks in variants.items():
        metrics, features, labels, merged, novel_state = evaluate_variant(
            runner, task_states0, support_loader, test_loader,
            class_names, class_ids, masks, base_ids, novel_ids
        )
        variant_features[name] = features
        variant_states[name] = (merged, novel_state)
        if not torch.equal(labels, session0_labels):
            raise RuntimeError("Evaluation sample order changed between Session-0 and Session-1 models.")
        cosine_distance = 1.0 - F.cosine_similarity(
            session0_features, features, dim=-1
        )
        base_mask = torch.isin(labels, base_ids)
        novel_mask = torch.isin(labels, novel_ids)
        metrics["feature_drift_vs_session0"] = {
            "all": metric_summary(cosine_distance),
            "base": metric_summary(cosine_distance[base_mask]),
            "novel": metric_summary(cosine_distance[novel_mask]),
        }
        current_base = metrics["base_global_accuracy"]
        current_novel = metrics["novel_global_accuracy"]
        mean_acc = (current_base + current_novel) / 2.0
        baseline_base_acc = float(payload0["task_acc_list"][0][0])
        pd = max(0.0, baseline_base_acc - current_base) / 2.0
        metrics["derived"] = {
            "mean_acc": mean_acc,
            "average_accuracy": (baseline_base_acc + mean_acc) / 2.0,
            "performance_drop": pd,
        }
        details[name] = metrics
        rows.append({
            "variant": name,
            "global_mean_acc": metrics["global_mean_acc"],
            "base_global_accuracy": current_base,
            "novel_global_accuracy": current_novel,
            "AA_two_sessions": metrics["derived"]["average_accuracy"],
            "PD_two_sessions": pd,
        })
        print(
            f"[{name}] global={metrics['global_mean_acc']:.3f} "
            f"base={current_base:.3f} novel={current_novel:.3f} "
            f"AA={metrics['derived']['average_accuracy']:.3f} PD={pd:.3f}"
        )

    # Fixed-prototype counterfactual isolates query-encoder changes.
    full_merged, _ = variant_states["A_full"]
    fixed_prototypes = F.normalize(full_merged["image_proto"].to(runner.device).float(), dim=-1)
    fixed_class_ids = torch.as_tensor(full_merged["class_ids"], device=runner.device, dtype=torch.long)
    fixed_proto_comparison = {}
    for name, features in variant_features.items():
        logits = 100.0 * features.to(runner.device) @ fixed_prototypes.t()
        fixed_proto_comparison[name], _ = classification_diagnostics(
            logits, session0_labels.to(runner.device), fixed_class_ids,
            base_ids.to(runner.device), novel_ids.to(runner.device),
        )

    install_payload(runner, payload1)
    anchor_response = collect_historical_anchor_responses(runner, task_states0, expanded)
    anchor_vs_base_test = compare_anchor_responses_to_base_test(
        anchor_response, details["A_full"]["expert_response_by_class"], base_ids
    )
    gradient_diagnostics = collect_gradient_diagnostics(
        runner, support_loader, class_ids, task_states0, expanded, current_session=1,
    )
    details["diagnostic_analysis"] = {
        "fixed_full_prototype_query_encoder_counterfactual": fixed_proto_comparison,
        "historical_pd_anchor_response": anchor_response,
        "historical_anchor_vs_base_test": anchor_vs_base_test,
        "ce_pd_gradient_diagnostics": gradient_diagnostics,
        "interpretation_notes": {
            "auc_positive_class": "Novel; values near 0.5 indicate weak Base/Novel ranking separation.",
            "patch_residual": "RMS of actual expert residual over patch tokens before subsequent Transformer blocks.",
            "gradient_cosine": "Computed on newly trainable expert/router parameters; no optimizer step is executed.",
            "dropout_repeatability": "Repeated PD evaluations with newest expert in train mode.",
        },
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A compact table makes the post-hoc Base/Novel response comparison easy
    # to inspect without parsing the nested details.json structure.
    response_rows = []
    for expert_name, groups in details["A_full"]["expert_response"].items():
        for group_name, values in groups.items():
            row = {
                "expert": expert_name,
                "group": group_name,
                "sample_count": values["sample_count"],
            }
            for metric_name in (
                "gate_logit", "gate", "residual_norm", "patch_residual_norm",
                "token_residual_norm", "relative_residual",
                "relative_patch_residual", "relative_token_residual",
            ):
                for statistic in ("mean", "std", "median", "q05", "q25", "q75", "q90", "q95"):
                    row[f"{metric_name}_{statistic}"] = values[metric_name][statistic]
            response_rows.append(row)

    response_fields = [
        "expert", "group", "sample_count",
        *[
            f"{metric}_{statistic}"
            for metric in (
                "gate_logit", "gate", "residual_norm", "patch_residual_norm",
                "token_residual_norm", "relative_residual",
                "relative_patch_residual", "relative_token_residual",
            )
            for statistic in ("mean", "std", "median", "q05", "q25", "q75", "q90", "q95")
        ],
    ]
    with (output / "expert_response_by_group.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=response_fields)
        writer.writeheader()
        writer.writerows(response_rows)

    print("Expert response by label group (labels used only post hoc):")
    for row in response_rows:
        print(
            f"  {row['expert']} [{row['group']}] n={row['sample_count']} "
            f"gate_mean={row['gate_mean']:.6f} "
            f"cls_residual={row['residual_norm_mean']:.6f} "
            f"patch_residual={row['patch_residual_norm_mean']:.6f}"
        )
    def json_default(value):
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")

    with (output / "details.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2, default=json_default)
    print(f"Saved ablation report to {output.resolve()}")


if __name__ == "__main__":
    main()
