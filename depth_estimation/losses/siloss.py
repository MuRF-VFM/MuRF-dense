import torch
from torch.nn import Module

class ScaleInvariantLoss(Module):
    """
    Scale Invariant Loss from eigen et al.
    """

    def __init__(self, lambda_l=0.5, epsilon=1e-6):
        super().__init__()
        self.lambda_l = lambda_l
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        See the formula in eigen et al
        """
        # avoid log(0)
        pred = pred.clamp(min=self.epsilon)
        target = target.clamp(min=self.epsilon)

        d = torch.log(pred) - torch.log(target)
        n = d.numel()

        lhs = torch.sum(d ** 2) / n
        rhs = (torch.sum(d) ** 2) / (n ** 2)

        return lhs - self.lambda_l * rhs