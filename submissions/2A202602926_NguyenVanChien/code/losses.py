"""losses.py - loss và trộn mẫu (Mixup/CutMix)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """CE với label smoothing: q'(k)=(1-eps)*1[k==y]+eps/K. eps=0 -> đúng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss: FL(p_t) = -alpha_t*(1-p_t)^gamma*log(p_t). gamma=0 -> đúng CE (sai số <1e-6)."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
        p_t = logp_t.exp()
        mod = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[target]
            loss = -at * mod * logp_t
        else:
            loss = -mod * logp_t
        return loss.mean()


def build_criterion(kind: str = "ce", **kw):
    """kind: ce | ls | focal | ce_weighted."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing=float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(gamma=float(kw.get("gamma", 2.0)), alpha=kw.get("alpha", None))
    if kind == "ce_weighted":
        w = kw.get("weight", None)
        if w is not None:
            w = torch.as_tensor(w, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w)
    raise ValueError(f"loss không hỗ trợ: {kind}")


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh TRAIN. beta=0: 1/n_c (chuẩn hoá mean=1); beta>0: class-balanced
    w=(1-beta)/(1-beta^n), chuẩn hoá tổng về số lớp. Chỉ dùng số liệu train."""
    import numpy as np
    n = np.asarray(list(counts), dtype=np.float64)
    assert (n > 0).all(), "mọi lớp phải có ảnh trong train"
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    else:
        w = (1.0 / n)
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch. Trả về (x_mix, (y_a, y_b, lam)). lam~Beta(alpha,alpha)."""
    assert mode in ("mixup", "cutmix"), mode
    lam = float(torch.distributions.Beta(alpha, alpha).sample(()))
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        return lam * x + (1.0 - lam) * x[perm], (y_a, y_b, lam)
    # CutMix: hộp ngẫu nhiên, lam hiệu chỉnh theo diện tích thực sau khi cắt biên.
    W, H = x.size(-1), x.size(-2)
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
    xm = x.clone()
    xm[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return xm, (y_a, y_b, float(lam))


def mixed_loss(criterion, logits, targets):
    """lam*crit(logits,y_a)+(1-lam)*crit(logits,y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
