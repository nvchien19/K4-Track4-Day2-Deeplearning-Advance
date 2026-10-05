"""inference.py - suy luận Bước 3: TTA, gộp view, ensemble, temperature scaling, gộp BN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # noqa: E402  (torch truoc numpy: tranh xung dot OpenMP/MKL tren Windows)


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    model.eval()
    names, ys, outs = [], [], []
    for x, y, fn in loader:
        if view is not None:
            x = view(x)
        x = x.to(device, non_blocking=True)
        outs.append(model(x).detach().cpu())
        ys.append(torch.as_tensor(y))
        names.extend(list(fn))
    return names, torch.cat(ys).numpy().astype(np.int64), torch.cat(outs).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N,C,H,W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước crop x crop từ ảnh vuông."""
    _, _, h, w = x.shape
    assert crop <= min(h, w), f"crop {crop} > ảnh {h}x{w}"
    ys = (0, h - crop, (h - crop) // 2)
    xs = (0, w - crop, (w - crop) // 2)
    views = [x[:, :, y:y + crop, x0:x0 + crop] for y in (ys[0], ys[1], ys[2])
             for x0 in (xs[0], xs[1])][:4]
    views.append(x[:, :, ys[2]:ys[2] + crop, xs[2]:xs[2] + crop])
    return views[:5]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong sizes (CNN có global pooling thì OK;
    ViT/Swin cần xử lý pos-embed/cửa sổ -> có thể lỗi, ghi rõ giới hạn)."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: prob = mean softmax; logit = softmax(mean logit). Trả về xác suất."""
    assert space in ("prob", "logit"), space
    arr = [np.asarray(l, dtype=np.float64) for l in logits_per_view]
    if space == "prob":
        return np.mean([_softmax(l) for l in arr], axis=0)
    return _softmax(np.mean(arr, axis=0))


def ensemble_probs(list_of_probs):
    """Trung bình xác suất nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    arr = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    out = np.mean(arr, axis=0)
    return out / out.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T>0 cực tiểu NLL trên VAL (LBFGS trên logT). Accuracy không đổi. KHÔNG khớp trên test."""
    z = torch.tensor(np.asarray(val_logits), dtype=torch.float32)
    y = torch.tensor(np.asarray(val_labels), dtype=torch.long)
    logT = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(logT.exp().item())
    return min(max(T, 0.05), 10.0)


def apply_temperature(logits, T: float):
    return _softmax(np.asarray(logits, dtype=np.float64) / float(T))


def fuse_conv_bn(model):
    """Gộp từng cặp Conv2d+BatchNorm2d liền kề (model.eval trước). Trả về model.
    ViT/Swin/ConvNeXt dùng LayerNorm -> không áp dụng, ghi rõ."""
    model.eval()
    fused = 0
    max_err = 0.0
    for parent in list(model.modules()):
        children = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(children, children[1:]):
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                with torch.no_grad():
                    w = m1.weight
                    gamma = m2.weight if m2.weight is not None else torch.ones(m2.num_features)
                    beta = m2.bias if m2.bias is not None else torch.zeros(m2.num_features)
                    mean, var, eps = m2.running_mean, m2.running_var, m2.eps
                    s = gamma / torch.sqrt(var + eps)
                    w_new = w * s.view(-1, 1, 1, 1)
                    b_new = beta + ( (m1.bias if m1.bias is not None else torch.zeros_like(mean)) - mean) * s
                    conv = nn.Conv2d(m1.in_channels, m1.out_channels, m1.kernel_size, m1.stride,
                                     m1.padding, m1.dilation, m1.groups, True, m1.padding_mode).to(w.device)
                    conv.weight.copy_(w_new)
                    conv.bias.copy_(b_new)
                    setattr(parent, n1, conv)
                    setattr(parent, n2, nn.Identity())
                    fused += 1
    if fused == 0:
        print("fuse_conv_bn: không tìm thấy cặp Conv+BN (có thể là ViT/Swin/LayerNorm) -> giữ nguyên")
    else:
        print(f"fuse_conv_bn: đã gộp {fused} cặp Conv+BN")
    return model
