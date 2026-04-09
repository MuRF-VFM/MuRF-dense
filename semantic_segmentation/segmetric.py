"""
Code from mmsegmentation but placed into a torchmetrics Metric class
for easy metric computation
"""

import torch
import numpy as np
from torchmetrics import Metric
from collections import OrderedDict
from .metrics import mean_iou, mean_dice, mean_fscore, total_intersect_and_union, total_area_to_metrics


class SegmentationMetrics(Metric):
    def __init__(self, dist_sync_on_step=False, num_classes=21, ignore_index=255, metric='mIoU'):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metric = metric
        self.all_results = []
        self.max_mIoU = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_np = preds.detach().cpu().numpy().astype(np.uint8)
        targets_np = targets.detach().cpu().numpy().astype(np.uint8)

        if preds_np.ndim == 3:
            preds_np = [p for p in preds_np]
            targets_np = [t for t in targets_np]
        else:
            preds_np = [preds_np]
            targets_np = [targets_np]

        intersect, union, pred_label, label = total_intersect_and_union(
            preds_np, targets_np, self.num_classes, self.ignore_index)
        self.all_results.append((intersect, union, pred_label, label))

    def compute(self):
        if len(self.all_results) == 0:
            return {}

        # sum over all batches
        pre_eval_results = tuple(zip(*self.all_results))
        total_area_intersect = sum(pre_eval_results[0])
        total_area_union = sum(pre_eval_results[1])
        total_area_pred_label = sum(pre_eval_results[2])
        total_area_label = sum(pre_eval_results[3])

        # get per-class metrics
        per_class_metrics = total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            metrics=['mDice', 'mIoU', 'mFscore'],
            nan_to_num=0
        )

        # compute mean metrics explicitly
        metrics = {}
        if 'IoU' in per_class_metrics:
            metrics['mIoU'] = np.nanmean(per_class_metrics['IoU'])
            metrics['maxIoU'] = np.nanmax(per_class_metrics['IoU'])
        if 'Dice' in per_class_metrics:
            metrics['mDice'] = np.nanmean(per_class_metrics['Dice'])
            metrics['maxDice'] = np.nanmax(per_class_metrics['Dice'])
        if 'Fscore' in per_class_metrics:
            metrics['mFscore'] = np.nanmean(per_class_metrics['Fscore'])
            metrics['maxFscore'] = np.nanmax(per_class_metrics['Fscore'])
        metrics['aAcc'] = per_class_metrics.get('aAcc', 0)

        if np.nanmean(per_class_metrics['IoU']) > self.max_mIoU:
            self.max_mIoU = np.nanmean(per_class_metrics['IoU'])
            metrics['best_mIoU'] = self.max_mIoU

        return metrics

    def __str__(self):
        ret_metrics = self.compute()
        metric_str = ', '.join([f'{k}: {np.nanmean(v):.4f}' if isinstance(v, np.ndarray) else f'{k}: {v:.4f}' for k, v in ret_metrics.items()])
        return metric_str

