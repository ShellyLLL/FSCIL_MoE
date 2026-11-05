"""Multi-layer APC-controlled dynamic MoE FSCIL runner."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from engine.apc import AdaptationAwarePrototypeController
from engine.engine import Runner, _cfg_value


class APCRunner(Runner):
    """Greedy, cross-validated multi-layer expert expansion.

    Every configured MoE block receives its own calibrated confusion and
    adaptation-capacity thresholds. Blocks passing either trigger are only
    candidates. Candidate experts are trained and evaluated with source-
    disjoint folds; the best statistically supported marginal improvement is
    committed, then all remaining blocks are re-evaluated under the updated
    topology. This permits zero, one, or multiple layer expansions per session.
    """

    def __init__(self, cfg):
        apc_cfg = cfg.TRAINER.BiMC.APC
        self.apc_controller = AdaptationAwarePrototypeController(
            apc_cfg.ALPHA, apc_cfg.CALIBRATION_TRIALS,
            apc_cfg.SEED if apc_cfg.SEED >= 0 else cfg.SEED,
            logit_scale=float(cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE),
        )
        super().__init__(cfg)

    @staticmethod
    def _concat_layer_features(old, new, expected_classes):
        result = {}
        expected = torch.sort(torch.as_tensor(expected_classes).long().cpu()).values
        if set(old) != set(new):
            raise ValueError("Historical and current feature maps must contain identical MoE blocks.")
        for block in sorted(old):
            item = {}
            for key in ("cls_in", "cls_out", "labels", "source_ids"):
                left = torch.as_tensor(old[block][key]).detach().cpu()
                right = torch.as_tensor(new[block][key]).detach().cpu()
                if left.ndim != right.ndim or left.shape[1:] != right.shape[1:]:
                    raise ValueError(f"Block {block} {key} dimensions differ.")
                item[key] = torch.cat([left, right], dim=0)
            if item["labels"].numel() != item["cls_out"].size(0):
                raise ValueError(f"Block {block} labels do not align with features.")
            observed = torch.unique(item["labels"], sorted=True)
            if not torch.equal(observed, expected):
                raise ValueError(
                    f"Block {block} support classes {observed.tolist()} do not match "
                    f"the expected classes {expected.tolist()}."
                )
            result[int(block)] = item
        return result

    def _all_seen_feature_loader(self, task_id):
        return self.data_manager.get_dataloader(
            task_id, source="train", mode="test", accumulate_past=True
        )

    def _calibrate_apc(self, base_loader):
        self.model.eval()
        features = self.model_without_dp.collect_blockwise_cls_features(base_loader)
        if not features:
            raise RuntimeError("No configured MoE block produced base calibration features.")
        for block, item in features.items():
            if "cls_out" not in item:
                raise RuntimeError(f"MoE block {block} does not expose adapted CLS features.")
        self.apc_state = self.apc_controller.calibrate(
            features, self.cfg.DATASET.NUM_INC_CLS, self.cfg.DATASET.NUM_INC_SHOT
        )
        print("=> [APC] calibrated blocks:",
              sorted(int(key) for key in self.apc_state["blocks"]))

    def _ensure_apc_ready(self, completed_session):
        if not self._moe_adapters() or completed_session < 0:
            return
        if self.apc_state and int(self.apc_state.get("version", -1)) == self.apc_controller.VERSION:
            return
        if completed_session > 0:
            raise RuntimeError(
                "This checkpoint contains a legacy single-layer APC topology. "
                "Start a fresh Session-0 run for the multi-layer expansion protocol."
            )
        base_loader = self.data_manager.get_dataloader(
            0, source="train", mode="test", accumulate_past=False
        )
        self._calibrate_apc(base_loader)

    def _evaluate_null_losses(self, folds, class_ids, text, description,
                              version, calibrate):
        values = []
        for _, fit_extract, validation in folds:
            values.append(self._held_out_loss(
                validation, fit_extract, class_ids, text, description,
                version, calibrate, previous_version=None
            ))
        return np.asarray(values, dtype=np.float64)

    def _candidate_fold_losses(self, task_id, block_idx, folds, class_ids,
                               text, description, calibrate, initial_expert,
                               initial_router, previous_version):
        adapter = self._adapter_by_block(block_idx)
        values = []
        for fit_train, fit_extract, validation in folds:
            adapter.experts[-1].load_state_dict(copy.deepcopy(initial_expert))
            adapter.router.columns[-1].load_state_dict(copy.deepcopy(initial_router))
            self._train_candidate(
                task_id, block_idx, fit_train, fit_extract, class_ids,
                text, description, calibrate
            )
            version = self._current_encoder_version()
            values.append(self._held_out_loss(
                validation, fit_extract, class_ids, text, description,
                version, calibrate, previous_version=previous_version
            ))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _paired_gain(null_losses, candidate_losses):
        if null_losses.shape != candidate_losses.shape or null_losses.ndim != 1:
            raise ValueError("Candidate and null validation folds must be paired.")
        gains = null_losses - candidate_losses
        mean = float(gains.mean())
        standard_error = (
            float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else float("inf")
        )
        return mean, standard_error, gains.tolist()

    def train_incremental_task(self, task_id, train_loader, extract_loader,
                               class_ids, class_names, calibrate_visual):
        if self.apc_state is None:
            raise RuntimeError("APC must be calibrated on the base session.")
        folds = self._support_folds(train_loader, extract_loader)
        if not folds:
            raise RuntimeError(
                "Multi-layer candidate verification requires at least three distinct "
                "support sources per novel class; refusing unverified expansion."
            )

        text, description = self._text_prototypes(class_names, class_ids)
        all_seen_loader = self._all_seen_feature_loader(task_id)
        self.model.eval()
        features = self.model_without_dp.collect_blockwise_cls_features(all_seen_loader)
        old_ids = torch.cat([
            torch.as_tensor(classes, dtype=torch.long)
            for classes in self.data_manager.class_index_in_task[:task_id]
        ])
        decisions = self.apc_controller.decide(
            features, old_ids, class_ids, self.apc_state
        )
        candidate_blocks = [
            int(row["block_idx"]) for row in decisions if bool(row["expand"])
        ]
        print(f"=> [APC][Session {task_id}] candidates={candidate_blocks}")

        expanded, accepted_diagnostics = [], []
        for _, adapter in self._moe_adapters():
            adapter.freeze_all()

        # Greedy forward selection: every accepted layer is retrained and
        # evaluated jointly with each next candidate. This measures marginal
        # value under the already committed within-session topology.
        while candidate_blocks:
            previous_version = self._current_encoder_version()
            null_losses = self._evaluate_null_losses(
                folds, class_ids, text, description, previous_version,
                bool(calibrate_visual)
            )
            trials = []
            for block_idx in list(candidate_blocks):
                adapter = self._adapter_by_block(block_idx)
                self.model_without_dp.add_expert(block_idx, task_id)
                initial_expert = copy.deepcopy(adapter.experts[-1].state_dict())
                initial_router = copy.deepcopy(adapter.router.columns[-1].state_dict())
                try:
                    candidate_losses = self._candidate_fold_losses(
                        task_id, block_idx, folds, class_ids, text, description,
                        bool(calibrate_visual), initial_expert, initial_router,
                        previous_version
                    )
                    gain, gain_se, fold_gains = self._paired_gain(null_losses, candidate_losses)
                    trials.append({
                        "block_idx": block_idx,
                        "gain": gain,
                        "gain_se": gain_se,
                        "fold_gains": fold_gains,
                        "null_loss": float(null_losses.mean()),
                        "candidate_loss": float(candidate_losses.mean()),
                    })
                finally:
                    adapter.remove_newest_expert()

            best = max(trials, key=lambda row: row["gain"], default=None)
            accepted_diagnostics.append({"round": len(expanded) + 1, "trials": trials})
            if best is None or best["gain"] <= best["gain_se"]:
                print(
                    f"=> [ExpansionStop][Session {task_id}] "
                    "no candidate exceeds paired one-standard-error threshold."
                )
                break

            block_idx = int(best["block_idx"])
            adapter = self._adapter_by_block(block_idx)
            expert_id = self.model_without_dp.add_expert(block_idx, task_id)
            training_history = self._train_candidate(
                task_id, block_idx, train_loader, extract_loader, class_ids,
                text, description, bool(calibrate_visual)
            )
            expanded.append({
                "block_idx": block_idx,
                "expert_id": expert_id,
                "validation_gain": best["gain"],
                "validation_gain_se": best["gain_se"],
                "training": training_history,
            })
            candidate_blocks.remove(block_idx)
            print(
                f"=> [ExpansionAccepted][Session {task_id}] B{block_idx} "
                f"E{expert_id} gain={best['gain']:.5f} se={best['gain_se']:.5f}"
            )
            for _, current_adapter in self._moe_adapters():
                current_adapter.freeze_all()
            self.model.eval()

            # Earlier-layer expansion changes downstream inputs. Recompute
            # every layer's current and historical support features before
            # the next APC decision; never reuse stale downstream activations.
            all_seen_loader = self._all_seen_feature_loader(task_id)
            features = self.model_without_dp.collect_blockwise_cls_features(all_seen_loader)
            remaining = self.apc_controller.decide(
                features, old_ids, class_ids, self.apc_state
            )
            eligible = {int(row["block_idx"]) for row in remaining if row["expand"]}
            candidate_blocks = [block for block in candidate_blocks if block in eligible]

        for _, adapter in self._moe_adapters():
            adapter.freeze_all()
        return {
            "trained": bool(expanded),
            "decisions": decisions,
            "candidate_rounds": accepted_diagnostics,
            "expanded": expanded,
            "encoder_version": task_id if expanded else self._current_encoder_version(),
        }

    def run(self):
        configured_end = int(_cfg_value(self.cfg.TRAINER.BiMC, "END_SESSION", -1))
        end_task = (
            self.data_manager.num_tasks - 1 if configured_end < 0
            else min(configured_end, self.data_manager.num_tasks - 1)
        )
        states = list(self._resume_state_dict_list)
        start_task = self._resume_completed_session + 1
        if start_task > end_task:
            print("=> Resume checkpoint already contains all configured FSCIL sessions.")
            return
        self._ensure_apc_ready(self._resume_completed_session)

        for task_id in range(start_task, end_task + 1):
            self.model.eval()
            class_ids = self.data_manager.class_index_in_task[task_id]
            class_names = np.asarray(self.data_manager.class_names)[class_ids].tolist()
            extract_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="test", accumulate_past=False
            )
            train_loader = self.data_manager.get_dataloader(
                task_id, source="train", mode="train", accumulate_past=False
            )
            adaptation = None

            if task_id == 0:
                task_state = self.model_without_dp.build_task_statistics(
                    class_names, extract_loader, class_ids,
                    calibrate_novel_vision_proto=False, encoder_version=0
                )
                task_state["class_ids"] = torch.as_tensor(
                    class_ids, device=self.device, dtype=torch.long
                )
                if self._moe_adapters():
                    self.train_base_task(task_state, train_loader, extract_loader)
                    for _, adapter in self._moe_adapters():
                        adapter.freeze_all()
                    self.model.eval()
                    task_state = self.model_without_dp.build_task_statistics(
                        class_names, extract_loader, class_ids,
                        calibrate_novel_vision_proto=False, encoder_version=0
                    )
                    task_state["class_ids"] = torch.as_tensor(
                        class_ids, device=self.device, dtype=torch.long
                    )
                    self._calibrate_apc(extract_loader)
                task_state["creation_session"] = 0
                states.append(task_state)
            else:
                adaptation = self.train_incremental_task(
                    task_id, train_loader, extract_loader, class_ids, class_names,
                    bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION)
                )
                self.model.eval()
                encoder_version = int(adaptation["encoder_version"])
                task_state = self.model_without_dp.build_task_statistics(
                    class_names, extract_loader, class_ids,
                    calibrate_novel_vision_proto=bool(self.cfg.TRAINER.BiMC.VISION_CALIBRATION),
                    encoder_version=encoder_version
                )
                task_state["class_ids"] = torch.as_tensor(
                    class_ids, device=self.device, dtype=torch.long
                )
                task_state["creation_session"] = int(task_id)
                states.append(task_state)

            gate_diagnostics = self._calibrate_task_gate(task_id, states)
            merged_state = self.merge_dicts(states)
            start_time = time.perf_counter()
            result = super().inference_task_bilevel(task_id, merged_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            result["AA"] = float(result["macro_task_acc"])
            result["PD"] = self._average_forgetting(task_id, result["task_acc"])
            self.acc_list.append(float(result["mean_acc"]))
            self.task_acc_list.append(list(result["task_acc"]))
            routing = self._routing_diagnostics(
                extract_loader, f"task{task_id}-final", current_task=task_id
            )
            self.session_diagnostics.append({
                "session": int(task_id),
                "adaptation": self._cpu_copy(adaptation),
                "anchor_gate": self._cpu_copy(gate_diagnostics),
                "routing": self._cpu_copy(routing),
                "expert_counts": {
                    int(block): int(adapter.num_experts)
                    for block, adapter in self._moe_adapters()
                },
                "classifier": self._cpu_copy(result),
                "evaluation_seconds": float(elapsed),
            })
            self._resume_state_dict_list = list(states)
            self._resume_completed_session = task_id
            self.save_checkpoint(task_id, states)
            print(
                f"=> Session {task_id}: Acc={result['mean_acc']:.3f} "
                f"AA={result['AA']:.3f} PD={result['PD']:.3f} "
                f"expanded={[] if adaptation is None else adaptation['expanded']}"
            )

        print("Final accuracy:", self.acc_list)
        print("Task-wise accuracy:", self.task_acc_list)

    def _average_forgetting(self, task, current):
        if task <= 0:
            return 0.0
        drops = []
        for old in range(task):
            best = max(
                float(self.task_acc_list[session][old])
                for session in range(old, task)
                if old < len(self.task_acc_list[session])
            )
            drops.append(max(0.0, best - float(current[old])))
        return float(np.mean(drops)) if drops else 0.0
