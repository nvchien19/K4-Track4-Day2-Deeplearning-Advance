"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _head_param_ids(model) -> set[int]:
    """Tập id() tham số của head/classifier mới (để tách LR head gấp 10)."""
    head = None
    for attr in ("get_classifier",):
        if hasattr(model, attr):
            try:
                head = model.get_classifier()
                break
            except Exception:
                pass
    if head is None:
        for name in ("head", "fc", "classifier", "heads"):
            if hasattr(model, name):
                head = getattr(model, name)
                break
    ids: set[int] = set()
    if head is not None:
        for p in head.parameters():
            ids.add(id(p))
    return ids


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model 9 lớp. init: scratch | frozen | finetune. Ghi tag trọng số vào model.pretrained_tag."""
    if init == "scratch":
        pretrained = False
    try:
        import timm
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                                  drop_rate=drop_rate)
        tag = getattr(getattr(model, "pretrained_cfg", None), "tag", None) or getattr(
            getattr(model, "pretrained_cfg", None), "url", "") or name
        model.pretrained_tag = str(tag)
    except ImportError:
        import torchvision.models as M
        if not hasattr(M, name):
            raise ValueError(f"không có backbone {name} (cần timm hoặc torchvision)")
        w = "DEFAULT" if pretrained else None
        model = M.__dict__[name](weights=w)
        in_f = model.fc.in_features if hasattr(model, "fc") else model.classifier[-1].in_features
        model.fc = nn.Linear(in_f, num_classes) if hasattr(model, "fc") else model.classifier
        if hasattr(model, "classifier") and not hasattr(model, "fc"):
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        if drop_rate:
            model.dropout_rate = drop_rate
        model.pretrained_tag = f"torchvision:{name}:{w}"
    if init == "frozen":
        freeze_backbone(model)
    elif init not in ("finetune", "scratch", "frozen"):
        raise ValueError(f"init không hỗ trợ: {init}")
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head. Lưu ý: phần đóng băng phải chạy ở eval (xem train loop)."""
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model._frozen_bn = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm (slide tr.52): backbone decay / backbone norm+bias (wd=0) / head."""
    head_ids = _head_param_ids(model)
    g_bb, g_nobias, g_head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            g_head.append(p)
        elif p.ndim <= 1:
            g_nobias.append(p)
        else:
            g_bb.append(p)
    groups = []
    if g_bb:
        groups.append({"params": g_bb, "lr": lr_backbone, "weight_decay": weight_decay})
    if g_nobias:
        groups.append({"params": g_nobias, "lr": lr_backbone, "weight_decay": 0.0})
    if g_head:
        groups.append({"params": g_head, "lr": lr_head, "weight_decay": weight_decay})
    assert groups, "không còn tham số train được (kiểm tra freeze_backbone)"
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), kể cả tham số đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho 1 ảnh 3xHxW. Thử thop -> fvcore -> ptflops, cuối cùng hook Conv2d/Linear."""
    device = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for lib in ("thop", "fvcore", "ptflops"):
        try:
            if lib == "thop":
                from thop import profile
                macs, _ = profile(model, inputs=(x,), verbose=False)
                return macs / 1e9
            elif lib == "fvcore":
                from fvcore.nn import FlopCountAnalysis
                return FlopCountAnalysis(model, x).total() / 2 / 1e9
            else:
                from ptflops import get_model_complexity_info
                model.eval()
                macs, _ = get_model_complexity_info(model, (3, img_size, img_size),
                                                    as_strings=False, verbose=False)
                return macs / 1e9
        except Exception:
            continue
    # Fallback: hook đếm Conv2d + Linear (MAC, không phải FLOPs 2x).
    macs = [0]

    def conv_hook(m, inp, out):
        b, oc, oh, ow = out.shape
        ic = m.in_channels // m.groups
        kh, kw = m.kernel_size
        macs[0] += b * oc * oh * ow * ic * kh * kw

    def lin_hook(m, inp, out):
        macs[0] += out.numel() * m.in_features  # batch*out*in
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_train = model.training
    model.eval()
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()
    if was_train:
        model.train()
    return macs[0] / 1e9
