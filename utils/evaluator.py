import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_continual_metrics(session_accuracies, task_accuracies=None):
    """Compute FSCIL summary metrics with an explicit, stable definition.

    Accuracy values are percentages (not fractions). AA is the arithmetic mean
    of per-session overall accuracies. PD is initial overall accuracy minus
    final overall accuracy, in percentage points. The former mean old-task drop
    is retained as a separate diagnostic and must not be reported as PD.
    """
    sessions = np.asarray(session_accuracies, dtype=np.float64).reshape(-1)
    if not np.isfinite(sessions).all():
        raise ValueError("session_accuracies must contain only finite values.")
    aa = float(sessions.mean()) if sessions.size else 0.0
    initial_acc = float(sessions[0]) if sessions.size else 0.0
    final_acc = float(sessions[-1]) if sessions.size else 0.0
    pd = initial_acc - final_acc if sessions.size > 1 else 0.0

    task_rows = []
    if task_accuracies is not None:
        task_rows = [
            np.asarray(row, dtype=np.float64).reshape(-1)
            for row in task_accuracies
        ]
        if len(task_rows) != sessions.size:
            raise ValueError(
                "task_accuracies must contain one row per session accuracy."
            )
        if any(not np.isfinite(row).all() for row in task_rows):
            raise ValueError("task_accuracies must contain only finite values.")
        for session_id, row in enumerate(task_rows):
            if row.size != session_id + 1:
                raise ValueError(
                    "Each task-accuracy row must contain one value for every "
                    "class-incremental task seen by that session."
                )

    old_task_drops = []
    final_novel_avg_acc = 0.0
    if task_rows:
        final_row = task_rows[-1]
        final_novel_avg_acc = (
            float(final_row[1:].mean()) if final_row.size > 1 else 0.0
        )
        for task_id in range(len(task_rows) - 1):
            old_task_drops.append(
                float(task_rows[task_id][task_id]) - float(final_row[task_id])
            )
    old_task_avg_drop = (
        float(np.mean(old_task_drops)) if old_task_drops else 0.0
    )
    return {
        "aa": aa,
        "pd": pd,
        "initial_acc": initial_acc,
        "final_acc": final_acc,
        "final_novel_avg_acc": final_novel_avg_acc,
        "old_task_avg_drop": old_task_avg_drop,
        "old_task_drops": old_task_drops,
    }


class AccuracyEvaluator:
    
    def __init__(self, class_index_per_task):
        self.class_index_per_task = class_index_per_task
        self.num_tasks = len(class_index_per_task)


    def confusion_matrix(self, logits, targets, task_id, normalize=False, class_ids=None):
        class_conf_matrix = self._class_wise_confusion_matrix(
            logits, targets, task_id, normalize, class_ids=class_ids
        )
        task_conf_matrix = self._task_wise_confusion_matrix(
            logits, targets, task_id, normalize, class_ids=class_ids
        )

        return {'class_conf_matrix': class_conf_matrix,
                'task_conf_matrix': task_conf_matrix}


    @staticmethod
    def _as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _prediction_labels(self, logits, class_ids=None):
        logits = self._as_numpy(logits)
        pred_cols = np.argmax(logits, axis=1)
        if class_ids is None:
            return pred_cols
        class_ids = self._as_numpy(class_ids).reshape(-1)
        if logits.shape[1] != len(class_ids):
            raise ValueError(
                "class_ids must align with the logit columns: "
                f"got {len(class_ids)} ids for {logits.shape[1]} columns."
            )
        return class_ids[pred_cols]

    def calc_accuracy(self, logits, targets, task_id, class_ids=None):
        logits = logits.cpu().numpy()
        targets = targets.cpu().numpy()

        overall_right_cnt = self._count_right_pred_num(logits, targets, class_ids=class_ids)
        overall_acc_mean = overall_right_cnt / len(targets)

        seen_task_classes = self.class_index_per_task[:task_id + 1]
        task_accuracies = []
        for classes in seen_task_classes:
            task_sample_indices = np.where(np.isin(targets, classes))[0]
            if len(task_sample_indices) == 0:
                task_accuracies.append(0.0)
                continue

            task_sample_logits = logits[task_sample_indices]
            task_sample_targets = targets[task_sample_indices]
            task_right_cnt = self._count_right_pred_num(
                task_sample_logits, task_sample_targets, class_ids=class_ids
            )

            task_acc_mean = task_right_cnt / len(task_sample_indices)
            task_accuracies.append(round(100 * task_acc_mean, 2))

        base_avg_acc = task_accuracies[0]
        inc_avg_acc = sum(task_accuracies[1:]) / (len(task_accuracies) - 1) if len(task_accuracies) > 1 else 0.0
        harmonic_acc = 2 * base_avg_acc * inc_avg_acc / (base_avg_acc + inc_avg_acc) if inc_avg_acc > 0 else 0.0
        macro_task_acc = float(np.mean(task_accuracies)) if task_accuracies else 0.0
        return {'mean_acc': round(100 * overall_acc_mean, 2), 
                'task_acc': task_accuracies,
                'harmonic_acc': round(harmonic_acc, 2),
                'base_avg_acc': round(base_avg_acc, 2),
                'inc_avg_acc': round(inc_avg_acc, 2),
                'macro_task_acc': round(macro_task_acc, 2)}


    def _count_right_pred_num(self, logits, targets, class_ids=None):
        pred = self._prediction_labels(logits, class_ids=class_ids)
        return np.sum(pred == targets)

    def error_flow(self, logits, targets, task_id, class_ids=None):
        """Return interpretable base/old-novel/current-novel error flows.

        ``class_ids`` maps logit columns back to global labels.  It is optional
        for backwards compatibility with the repository's contiguous-label
        experiments, but required by any reordered PrototypeBank.
        """
        logits_np = self._as_numpy(logits)
        targets_np = self._as_numpy(targets).reshape(-1)
        preds_np = self._prediction_labels(logits_np, class_ids=class_ids)
        if targets_np.shape[0] != preds_np.shape[0]:
            raise ValueError("targets and logits must contain the same number of samples.")

        seen_tasks = self.class_index_per_task[:task_id + 1]
        target_tasks = self._determine_tasks(targets_np, seen_tasks)
        pred_tasks = self._determine_tasks(preds_np, seen_tasks)

        def group(task_values):
            result = np.full_like(task_values, 1)  # old novel by default
            result[task_values == 0] = 0
            result[task_values == task_id] = 2
            return result

        actual_groups = group(target_tasks)
        predicted_groups = group(pred_tasks)
        names = ("base", "old_novel", "current_novel")
        counts = np.zeros((3, 3), dtype=np.int64)
        for source in range(3):
            for target in range(3):
                counts[source, target] = int(
                    np.sum((actual_groups == source) & (predicted_groups == target))
                )

        rates = np.zeros_like(counts, dtype=np.float64)
        row_sums = counts.sum(axis=1, keepdims=True)
        np.divide(counts, row_sums, out=rates, where=row_sums > 0)
        named_rates = {
            f"{names[source]}->{names[target]}": float(rates[source, target])
            for source in range(3) for target in range(3)
        }
        return {
            "counts": counts,
            "rates": rates,
            "named_rates": named_rates,
            "labels": names,
        }


    def _determine_tasks(self, samples, task_classes):
        tasks = np.zeros_like(samples)
        for task_id, classes in enumerate(task_classes):
            class_mask = np.isin(samples, classes)
            tasks[class_mask] = task_id
        return tasks
    


    def _task_wise_confusion_matrix(self, logits, targets, task_id, normalize=False, class_ids=None):
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        seen_task_classes = [cls for cls in self.class_index_per_task[:task_id + 1]]
        actual_tasks = self._determine_tasks(targets_np, seen_task_classes)
        predicted_tasks = self._determine_tasks(
            self._prediction_labels(logits_np, class_ids=class_ids), seen_task_classes
        )

        task_conf_matrix = confusion_matrix(actual_tasks, predicted_tasks, labels=range(len(seen_task_classes)))

        if normalize:
            task_conf_matrix = task_conf_matrix.astype('float')
            row_sums = task_conf_matrix.sum(axis=1, keepdims=True)
            task_conf_matrix /= row_sums

        return task_conf_matrix

    def _class_wise_confusion_matrix(self, logits, targets, task_id, normalize=False, class_ids=None):
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()

        seen_classes = np.concatenate([cls for cls in self.class_index_per_task[:task_id + 1]])
        unique_seen_classes = np.unique(seen_classes)
        valid_indices = np.isin(targets_np, unique_seen_classes)
        valid_logits = logits_np[valid_indices]
        valid_targets = targets_np[valid_indices]

        if class_ids is None:
            preds = np.argmax(valid_logits, axis=1)
        else:
            preds = self._prediction_labels(valid_logits, class_ids=class_ids)
        conf_matrix = confusion_matrix(valid_targets, preds, labels=unique_seen_classes)

        if normalize:
            conf_matrix = conf_matrix.astype('float')
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix /= row_sums

        return conf_matrix


    def task_class_confusion_matrix(self, class_labels, true_task_labels, logits):
        """
        Compute the task-class confusion matrix.

        Args:
        - class_labels (torch.Tensor): Tensor of ground truth class labels for each sample.
        - true_task_labels (torch.Tensor): Tensor of ground truth task labels for each sample.
        - logits (torch.Tensor): The logits output from the model for each sample.
        
        Returns:
        - np.array: A confusion matrix of shape (num_classes, num_tasks)
        """
        if isinstance(class_labels, torch.Tensor):
            class_labels = class_labels.cpu().numpy()
        if isinstance(true_task_labels, torch.Tensor):
            true_task_labels = true_task_labels.cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        predicted_task_labels = np.argmax(logits, axis=1)
        
        unique_classes = np.unique(class_labels)
        unique_tasks = np.arange(10)
        
        confusion_mat = np.zeros((len(unique_classes), len(unique_tasks)))
        
        for i, cls in enumerate(unique_classes):
            for j, task in enumerate(unique_tasks):
                idx = np.where((class_labels == cls) & (predicted_task_labels == task))[0]
                task_correct = np.sum(predicted_task_labels[idx] == task)
                confusion_mat[i, j] = task_correct
        
        return confusion_mat
