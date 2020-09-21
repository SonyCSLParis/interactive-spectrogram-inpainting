import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.0,
                 dim: int = -1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - self.smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))