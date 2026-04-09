from depth_estimation.metric import AllMetrics

import torch
from tqdm import tqdm

from depth_estimation.losses.gradientloss import GradientLoss
from depth_estimation.losses.sigloss import SigLoss
from depth_estimation.losses.siloss import ScaleInvariantLoss


class Criterion(torch.nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.sig_loss = SigLoss()
        self.grad_loss = GradientLoss()

    def forward(self, pred, target):
        return self.sig_loss(pred, target) + 0.5 * self.grad_loss(pred, target)


def val(model, val_dataloader) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = Criterion()
    metric = AllMetrics()
    # load the model and freeze the backbone weights

    model.to(device)
    metric.to(device)
    criterion.to(device)

    # put model in validation mode
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader)):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            gt_depths = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device):
                outputs = model(pixel_values)
                predicted_depth = outputs["predicted_depth"]
                metric.update(predicted_depth.squeeze(1), gt_depths.squeeze(1))

                loss = criterion(predicted_depth.squeeze(1), gt_depths.squeeze(1))
                running_loss += loss.item()

    metrics = metric.compute()
    metric.reset()
    metrics['loss'] = running_loss / len(val_dataloader)
    return metrics

