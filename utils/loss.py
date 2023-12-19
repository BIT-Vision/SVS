import torch
import torch.nn as nn
from .pytorch_iou import IOU
from .pytorch_ssim import SSIM
import torch.nn.functional as F


class BCE_SSIM_LOSS(nn.Module):
    def __init__(self):
        super(BCE_SSIM_LOSS, self).__init__()
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.iou_loss = IOU(size_average=True)
        self.ratio = None
        self.n_ratio = None

    # targets N B 1 H W  preds N B step H W
    def forward(self, preds, targets):
        loss = 0
        N, _, step, _, _ = preds.shape
        if self.ratio is None:
            ratio = list(range(step, 0, -1))
            self.ratio = torch.Tensor(ratio).cuda()
            self.ratio = self.ratio / self.ratio.sum()
        
        if self.n_ratio is None:
            ratio = list(range(N, 0, -1))
            self.n_ratio = torch.Tensor(ratio).cuda()
            self.n_ratio = self.n_ratio / self.n_ratio.sum()
        
        for j in range(N):
            pred = preds[j]
            target = targets[j]
            for i in range(step):
                bce_out = self.bce_loss(pred[:, i:i + 1], target)
                ssim_out = 1 - self.ssim_loss(pred[:, i:i + 1], target)
                iou_out = self.iou_loss(pred[:, i:i + 1], target)
                loss += ((bce_out + ssim_out + iou_out) * self.ratio[i] * self.n_ratio[j])
        return loss