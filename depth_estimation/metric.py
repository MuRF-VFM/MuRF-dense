import torch
from torch.nn import Module
from torchmetrics import Metric
import numpy as np
from collections import OrderedDict


class RMSE(Metric):
    """
    Root Mean Squared Error Metric
    """

    def __init__(self, dist_sync_on_step=False, depth_range=(1e-3, 10),):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.min_depth = depth_range[0]
        self.max_depth = depth_range[1]

        self.add_state("sum_squared_error", default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("number_elements", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape. Got {preds.shape} and {target.shape}")

        if self.max_depth is not None:
            valid_mask = torch.logical_and(target > self.min_depth, target <= self.max_depth)

        preds = preds[valid_mask]
        target = target[valid_mask]

        se = (preds - target) ** 2
        self.sum_squared_error += se.sum()
        self.number_elements += target.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.number_elements)


class AllMetrics(Metric):
    def __init__(self, dist_sync_on_step=False, depth_range=(1e-3, 10)):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.min_depth = depth_range[0]
        self.max_depth = depth_range[1]

        self.all_results = []

    def calculate(self, gt, pred):
        if gt.shape[0] == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred) - np.log(gt)

        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
        if np.isnan(silog):
            silog = 0

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

    def metrics(self, gt, pred):
        mask_1 = gt > self.min_depth
        mask_2 = gt < self.max_depth
        mask = np.logical_and(mask_1, mask_2)

        gt = gt[mask]
        pred = pred[mask]

        a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = self.calculate(
            gt, pred)

        return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

    def pre_eval_to_metrics(self, pre_eval_results):

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        pre_eval_results = tuple(zip(*pre_eval_results))
        ret_metrics = OrderedDict({})

        ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
        ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
        ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
        ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
        ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
        ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
        ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[6])
        ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
        ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])

        ret_metrics = {
            metric: value
            for metric, value in ret_metrics.items()
        }

        return ret_metrics

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_np = preds.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        self.all_results.append(self.metrics(target_np, preds_np))

    def compute(self):
        return self.pre_eval_to_metrics(self.all_results)

    def __str__(self):
        ret_metrics = self.compute()
        metric_str = ', '.join([f'{k}: {v:.4f}' for k, v in ret_metrics.items()])
        return metric_str

