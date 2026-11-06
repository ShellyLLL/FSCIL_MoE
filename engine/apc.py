"""Layer-wise adaptation-aware prototype confusion control for FSCIL."""
import math

import numpy as np
import torch
import torch.nn.functional as F


class AdaptationAwarePrototypeController:
    """Topology-bound, class-balanced APC for multi-layer FSCIL expansion.

    Calibration constructs source-disjoint few-shot episodes from all classes
    available before the current session. At each candidate layer it records
    finite-sample upper bounds for (1) novel-to-old prototype confusion and
    (2) source-exclusive novel-query classification loss.

    APC only proposes layers. APCRunner must train a temporary expert and
    validate its marginal value with the deployed global BiMC classifier
    before committing a topology transition.
    """

    VERSION = 4

    def __init__(self, alpha=0.05, calibration_trials=256, seed=1,
                 logit_scale=20.0):
        self.alpha = float(alpha)
        self.calibration_trials = int(calibration_trials)
        self.seed = int(seed)
        self.logit_scale = float(logit_scale)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("APC alpha must be in (0, 1).")
        if self.calibration_trials < 2:
            raise ValueError("APC requires at least two calibration episodes.")
        if self.logit_scale <= 0.0:
            raise ValueError("APC logit scale must be positive.")

    @staticmethod
    def _aggregate_sources(item, feature_key):
        for key in (feature_key, "labels", "source_ids"):
            if key not in item:
                raise KeyError(f"APC features must contain {key!r}.")
        x = torch.as_tensor(item[feature_key]).detach().float().cpu()
        y = torch.as_tensor(item["labels"]).detach().long().cpu().reshape(-1)
        source = torch.as_tensor(item["source_ids"]).detach().long().cpu().reshape(-1)
        if x.ndim != 2 or x.size(0) != y.numel() or y.numel() != source.numel():
            raise ValueError("APC features, labels and source IDs must align.")
        if x.numel() == 0 or not torch.isfinite(x).all():
            raise ValueError("APC requires non-empty finite layer features.")
        # A source is one statistical observation and cannot belong to two
        # different classes. This also prevents cross-class source leakage.
        for source_id in torch.unique(source):
            source_labels = torch.unique(y[source.eq(source_id)])
            if source_labels.numel() != 1:
                raise ValueError(
                    f"APC source id {int(source_id)} is associated with multiple classes."
                )
        pairs = torch.stack([y, source], dim=1)
        unique_pairs, inverse = torch.unique(
            pairs, dim=0, sorted=True, return_inverse=True
        )
        sums = x.new_zeros((len(unique_pairs), x.size(1)))
        counts = x.new_zeros((len(unique_pairs), 1))
        sums.index_add_(0, inverse, x)
        counts.index_add_(0, inverse, torch.ones_like(
            inverse, dtype=x.dtype
        ).unsqueeze(1))
        x = F.normalize(sums / counts.clamp_min(1.0), dim=-1)
        return x, unique_pairs[:, 0].long(), unique_pairs[:, 1].long()

    @staticmethod
    def _upper_quantile(values, alpha):
        values = torch.as_tensor(values, dtype=torch.float64).flatten().sort().values
        if values.numel() < 2:
            raise ValueError("APC calibration distribution is too small.")
        # Finite-sample upper order statistic; no interpolation below the
        # requested tail probability.
        rank = math.ceil((1.0 - float(alpha)) * (values.numel() + 1)) - 1
        rank = min(values.numel() - 1, max(0, rank))
        return float(values[rank].item())

    def _episode_metrics(self, x, labels, source_ids, classes, new_classes,
                         shot, rng):
        del source_ids  # IDs have already been source-aggregated.
        groups = {
            int(cls): np.flatnonzero(labels.numpy() == int(cls))
            for cls in classes
        }
        support_rows, support_class_ids = [], []
        query_rows, query_class_ids = [], []
        for cls in classes:
            candidates = groups[int(cls)]
            if len(candidates) < 2:
                return None
            support_count = min(int(shot), len(candidates) - 1)
            chosen = rng.choice(candidates, size=support_count, replace=False)
            support_rows.extend(chosen.tolist())
            support_class_ids.extend([int(cls)] * len(chosen))
            if int(cls) in new_classes:
                remaining = np.setdiff1d(candidates, chosen, assume_unique=False)
                query_count = min(int(shot), len(remaining))
                if query_count < 1:
                    return None
                queries = rng.choice(remaining, size=query_count, replace=False)
                query_rows.extend(queries.tolist())
                query_class_ids.extend([int(cls)] * len(queries))

        if not query_rows:
            return None
        support_rows = torch.as_tensor(support_rows, dtype=torch.long)
        support_labels = torch.as_tensor(support_class_ids, dtype=torch.long)
        prototypes = []
        for cls in classes:
            selected = x[support_rows[support_labels.eq(int(cls))]]
            prototypes.append(F.normalize(selected.mean(dim=0), dim=0))
        prototypes = torch.stack(prototypes)
        class_to_column = {
            int(cls): index for index, cls in enumerate(classes.tolist())
        }
        new_columns = torch.as_tensor(
            [class_to_column[int(cls)] for cls in new_classes], dtype=torch.long
        )
        old_columns = torch.as_tensor(
            [class_to_column[int(cls)] for cls in classes
             if int(cls) not in new_classes],
            dtype=torch.long,
        )
        risk = float(
            (prototypes[new_columns] @ prototypes[old_columns].t()).max().item()
        )

        query_rows = torch.as_tensor(query_rows, dtype=torch.long)
        query_labels = torch.as_tensor(query_class_ids, dtype=torch.long)
        logits = self.logit_scale * (x[query_rows] @ prototypes.t())
        targets = torch.as_tensor(
            [class_to_column[int(cls)] for cls in query_labels.tolist()],
            dtype=torch.long,
        )
        per_query_loss = F.cross_entropy(logits, targets, reduction="none")
        per_query_correct = logits.argmax(dim=1).eq(targets).float()
        # Equal class weight is important when the support manifest contains
        # different numbers of source images for different classes.
        class_losses, class_accuracies = [], []
        for cls in new_classes:
            mask = query_labels.eq(int(cls))
            if not bool(mask.any().item()):
                return None
            class_losses.append(per_query_loss[mask].mean())
            class_accuracies.append(per_query_correct[mask].mean())
        loss = float(torch.stack(class_losses).mean().item())
        accuracy = float(torch.stack(class_accuracies).mean().item())
        return risk, loss, accuracy

    def calibrate(self, features_by_block, n_new_classes, k_shot,
                  topology_version=0, topology_signature="",
                  support_manifest_hash=""):
        if not features_by_block:
            raise ValueError("APC needs historical features from every configured MoE block.")
        if int(k_shot) < 1 or int(n_new_classes) < 1:
            raise ValueError("APC requires positive incremental way and shot.")
        if int(topology_version) < 0:
            raise ValueError("APC topology version must be non-negative.")

        block_states = {}
        block_ids = sorted(int(block) for block in features_by_block)
        alpha_per_metric = self.alpha / (2.0 * len(block_ids))
        reference_classes = None
        for block in block_ids:
            item = features_by_block[block]
            x, labels, sources = self._aggregate_sources(item, "block_out")
            classes = torch.unique(labels, sorted=True)
            if reference_classes is None:
                reference_classes = classes
            elif not torch.equal(classes, reference_classes):
                raise ValueError(
                    "Every APC block must be calibrated on the same historical classes."
                )
            if classes.numel() < 2:
                raise ValueError(f"Block {block} calibration needs at least two classes.")
            minimum_sources = min(
                int(labels.eq(int(cls)).sum().item()) for cls in classes
            )
            if minimum_sources < 2:
                raise ValueError(
                    f"Block {block} has fewer than two distinct sources in a base class."
                )
            # For very small synthetic/base partitions, use the largest valid
            # pseudo-novel way while always retaining at least one old class.
            way = min(int(n_new_classes), len(classes) - 1)
            rng = np.random.default_rng(self.seed + 1009 * block)
            risk_scores, capacity_losses, capacity_accuracies = [], [], []
            for _ in range(self.calibration_trials):
                selected = rng.choice(classes.numpy(), size=way, replace=False)
                metrics = self._episode_metrics(
                    x, labels, sources, classes,
                    {int(value) for value in selected}, int(k_shot), rng,
                )
                if metrics is None:
                    continue
                risk, loss, accuracy = metrics
                risk_scores.append(risk)
                capacity_losses.append(loss)
                capacity_accuracies.append(accuracy)

            if len(risk_scores) < 2:
                raise RuntimeError(
                    f"Block {block} produced too few valid APC episodes."
                )
            risk_threshold = self._upper_quantile(
                risk_scores, alpha_per_metric
            )
            capacity_threshold = self._upper_quantile(
                capacity_losses, alpha_per_metric
            )
            block_states[str(block)] = {
                "feature_dim": int(x.size(1)),
                "num_historical_classes": int(len(classes)),
                "num_sources_per_class_min": int(minimum_sources),
                "risk_scores": torch.as_tensor(
                    risk_scores, dtype=torch.float32
                ).sort().values,
                "capacity_losses": torch.as_tensor(
                    capacity_losses, dtype=torch.float32
                ).sort().values,
                "capacity_accuracies": torch.as_tensor(
                    capacity_accuracies, dtype=torch.float32
                ).sort().values,
                "risk_threshold": risk_threshold,
                "capacity_threshold": capacity_threshold,
                "calibration_episodes": len(risk_scores),
            }
            print(
                f"=> [APC-Calibrate][B{block}] episodes={len(risk_scores)} "
                f"risk_thr={risk_threshold:.4f} "
                f"capacity_loss_thr={capacity_threshold:.4f}"
            )

        return {
            "version": self.VERSION,
            "alpha": self.alpha,
            "calibration_trials": self.calibration_trials,
            "seed": self.seed,
            "logit_scale": self.logit_scale,
            "n_new_classes": int(n_new_classes),
            "k_shot": int(k_shot),
            "num_blocks": len(block_ids),
            "calibration_class_ids": reference_classes.tolist(),
            "topology_version": int(topology_version),
            "topology_signature": str(topology_signature),
            "support_manifest_hash": str(support_manifest_hash),
            "blocks": block_states,
        }

    def decide(self, features_by_block, old_ids, new_ids, state,
               topology_version=None, topology_signature=None,
               support_manifest_hash=None):
        if not isinstance(state, dict) or int(state.get("version", -1)) != self.VERSION:
            raise ValueError(
                "APC state is missing or uses an incompatible calibration version."
            )
        stored_signature = str(state.get("topology_signature", ""))
        if stored_signature:
            if topology_version is None or topology_signature is None:
                raise ValueError(
                    "Topology-bound APC decisions require the current topology metadata."
                )
            if int(topology_version) != int(state["topology_version"]):
                raise ValueError("APC calibration belongs to a different topology version.")
            if str(topology_signature) != stored_signature:
                raise ValueError("APC calibration belongs to a different MoE topology.")

        stored_manifest = str(state.get("support_manifest_hash", ""))
        if stored_manifest:
            if support_manifest_hash is None:
                raise ValueError("Topology-bound APC decisions require the support-manifest hash.")
            if str(support_manifest_hash) != stored_manifest:
                raise ValueError("APC calibration belongs to a different support manifest.")

        old = torch.as_tensor(old_ids, dtype=torch.long).cpu().reshape(-1)
        new = torch.as_tensor(new_ids, dtype=torch.long).cpu().reshape(-1)
        if (old.numel() == 0 or new.numel() == 0
                or torch.unique(old).numel() != old.numel()
                or torch.unique(new).numel() != new.numel()
                or torch.isin(new, old).any()):
            raise ValueError("APC old/new global class IDs must be unique and disjoint.")
        calibrated_old = torch.as_tensor(
            state.get("calibration_class_ids", []), dtype=torch.long
        ).sort().values
        if not torch.equal(old.sort().values, calibrated_old):
            raise ValueError(
                "APC must be recalibrated on exactly the historical classes before each session."
            )
        expected = torch.sort(torch.cat([old, new])).values
        blocks = state.get("blocks", {})
        if not blocks:
            raise ValueError("APC calibration has no per-block states.")
        expected_blocks = {int(key) for key in blocks}
        observed_blocks = {int(key) for key in features_by_block}
        if observed_blocks != expected_blocks:
            raise ValueError(
                f"APC feature blocks {sorted(observed_blocks)} do not match "
                f"calibrated blocks {sorted(expected_blocks)}."
            )

        decisions = []
        alpha_per_metric = float(state["alpha"]) / (2.0 * max(1, len(blocks)))
        for key, block_state in sorted(
            blocks.items(), key=lambda pair: int(pair[0])
        ):
            block = int(key)
            x, labels, sources = self._aggregate_sources(
                features_by_block[block], "block_out"
            )
            observed = torch.unique(labels, sorted=True)
            if not torch.equal(observed, expected):
                raise ValueError(
                    f"Block {block} support classes {observed.tolist()} do not match "
                    f"the expected seen classes {expected.tolist()}."
                )
            if x.size(1) != int(block_state["feature_dim"]):
                raise ValueError(f"Block {block} APC feature dimension changed.")

            def prototype(cls):
                selected = x[labels.eq(int(cls))]
                if selected.numel() == 0:
                    raise ValueError(f"Block {block} is missing class {int(cls)}.")
                return F.normalize(selected.mean(dim=0), dim=0)

            old_prototypes = torch.stack([prototype(cls) for cls in old.tolist()])
            new_prototypes = torch.stack([prototype(cls) for cls in new.tolist()])
            risk = float(
                (new_prototypes @ old_prototypes.t()).max().item()
            )

            class_ids = expected.tolist()
            class_to_column = {
                int(cls): index for index, cls in enumerate(class_ids)
            }
            new_query_rows = torch.nonzero(
                torch.isin(labels, new), as_tuple=False
            ).flatten()
            query_losses_by_class = {int(cls): [] for cls in new.tolist()}
            query_correct_by_class = {int(cls): [] for cls in new.tolist()}
            for row in new_query_rows.tolist():
                query_class = int(labels[row].item())
                query_source = int(sources[row].item())
                local_prototypes = []
                for cls in class_ids:
                    mask = labels.eq(int(cls))
                    if int(cls) == query_class:
                        mask = mask & sources.ne(query_source)
                    if not bool(mask.any().item()):
                        raise ValueError(
                            f"Block {block} cannot form a source-exclusive prototype "
                            f"for class {query_class}."
                        )
                    local_prototypes.append(
                        F.normalize(x[mask].mean(dim=0), dim=0)
                    )
                local_prototypes = torch.stack(local_prototypes)
                logits = self.logit_scale * (
                    x[row:row + 1] @ local_prototypes.t()
                )
                target = torch.tensor(
                    [class_to_column[query_class]], dtype=torch.long
                )
                query_losses_by_class[query_class].append(
                    F.cross_entropy(logits, target)
                )
                query_correct_by_class[query_class].append(
                    logits.argmax(dim=1).eq(target).float().mean()
                )

            if any(not values for values in query_losses_by_class.values()):
                raise ValueError(f"Block {block} is missing current novel queries.")
            capacity_loss = float(torch.stack([
                torch.stack(values).mean()
                for values in query_losses_by_class.values()
            ]).mean().item())
            capacity_accuracy = float(torch.stack([
                torch.stack(values).mean()
                for values in query_correct_by_class.values()
            ]).mean().item())
            risk_threshold = float(block_state["risk_threshold"])
            capacity_threshold = float(block_state["capacity_threshold"])
            risk_ratio = risk / max(abs(risk_threshold), 1e-8)
            capacity_ratio = capacity_loss / max(abs(capacity_threshold), 1e-8)
            risk_excess = risk - risk_threshold
            capacity_excess = capacity_loss - capacity_threshold
            expand = risk_excess > 0.0 or capacity_excess > 0.0
            decision = {
                "block_idx": block,
                "expand": bool(expand),
                "risk": risk,
                "risk_threshold": risk_threshold,
                "risk_excess": risk_excess,
                "risk_ratio": risk_ratio,
                "adaptation_loss": capacity_loss,
                "adaptation_loss_threshold": capacity_threshold,
                "adaptation_loss_excess": capacity_excess,
                "adaptation_accuracy": capacity_accuracy,
                "adaptation_ratio": capacity_ratio,
                "need_score": max(risk_ratio, capacity_ratio),
                "reason": (
                    "both" if risk_excess > 0.0 and capacity_excess > 0.0
                    else "prototype_confusion" if risk_excess > 0.0
                    else "insufficient_adaptation" if capacity_excess > 0.0
                    else "reuse"
                ),
                "comparison_count": int(old.numel() * new.numel()),
                "query_count": int(sum(
                    len(values) for values in query_losses_by_class.values()
                )),
                "calibration_alpha_per_metric": alpha_per_metric,
                "topology_version": int(state.get("topology_version", 0)),
                "topology_signature": stored_signature,
            }
            decisions.append(decision)
            print(
                f"=> [APC][B{block}] risk={risk:.4f}/{risk_threshold:.4f} "
                f"adapt_loss={capacity_loss:.4f}/{capacity_threshold:.4f} "
                f"adapt_acc={capacity_accuracy:.3f} "
                f"decision={'CANDIDATE' if expand else 'REUSE'}"
            )
        return decisions
