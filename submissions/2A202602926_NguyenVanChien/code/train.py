"""train.py - một hàm run(cfg) cho mọi thí nghiệm (B, T, F).

Chạy: python train.py --set exp_id=B01 backbone=resnet50 seed=0
Checkpoint chọn theo macro-F1 val (hòa lấy epoch sớm hơn). Test chỉ ghi khi save_test_predictions=True (Bước 4).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np  # noqa: E402  (torch phai import truoc numpy/pandas: tranh xung dot OpenMP/MKL tren Windows)
import pandas as pd  # noqa: E402


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
    resume: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg: Config):
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính (warmup_epochs) rồi cosine về ~0, cập nhật theo bước."""
    total = max(cfg.epochs * steps_per_epoch, 1)
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))

    def lr_fn(step):
        if step < max(warmup, 1):
            return (step + 1) / max(warmup, 1)
        t = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


class EMA:
    """W_ema <- d*W_ema + (1-d)*W sau mỗi bước. Đánh giá bằng bản sao ema_model."""

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        d = self.decay
        for e, m in zip(self.ema_model.parameters(), model.parameters()):
            e.mul_(d).add_(m.detach(), alpha=1.0 - d)
        for be, bm in zip(self.ema_model.buffers(), model.buffers()):
            be.copy_(bm)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.ema_model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    from losses import mix_batch, mixed_loss
    frozen = getattr(model, "_frozen_bn", False)
    model.train()
    if frozen:  # backbone đóng băng: giữ BN ở eval
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
    tot, n, t0 = 0.0, 0, time.time()
    use_amp = cfg.amp and device.type == "cuda"
    for batch_idx, (x, y, _) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            if cfg.mix:
                xm, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                ta, tb, lam = targets[0], targets[1], targets[2]
                loss = mixed_loss(criterion, model(xm),
                                  (ta, tb, lam if isinstance(lam, float) else float(lam)))
            else:
                loss = criterion(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        tot += loss.item() * x.size(0)
        n += x.size(0)
        if (batch_idx + 1) % 25 == 0:
            print(f"{cfg.exp_id} seed={cfg.seed} batch={batch_idx+1}/{len(loader)} "
                  f"loss={tot/max(n,1):.4f} elapsed_s={time.time()-t0:.0f}", flush=True)
    return {"train_loss": tot / max(n, 1), "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.time() - t0}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    names, ys, outs, tot, n = [], [], [], 0.0, 0
    for x, y, fn in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        outs.append(logits.detach().cpu())
        ys.append(torch.as_tensor(y))
        names.extend(list(fn))
        if criterion is not None:
            tot += criterion(logits, y.to(logits.device)).item() * x.size(0)
            n += x.size(0)
    logits = torch.cat(outs).numpy()
    y_true = torch.cat(ys).numpy().astype(np.int64)
    return names, y_true, logits, (tot / max(n, 1))


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train loss")
    if "val_loss" in h:
        ax[0].plot(h["epoch"], h["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(h["epoch"], h["val_macro_f1"], marker="o", label="val macro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro-F1"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _repo_root() -> Path:
    # code/ nằm trong submissions/<mssv>/code -> repo root cách 3 cấp.
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "eval.py").exists():
            return p
    return Path.cwd()


def run(cfg: Config) -> dict:
    import dataset as D
    import model as M
    import losses as L
    sys.path.insert(0, str(_repo_root()))
    from eval import compute_metrics, save_predictions

    set_seed(cfg.seed)
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    if cfg.resume and (rdir / "config.json").exists():
        previous = json.loads((rdir / "config.json").read_text())
        current = dataclasses.asdict(cfg)
        previous.pop("resume", None); current.pop("resume", None)
        if previous != current:
            raise ValueError(f"Cannot resume with a changed configuration: {rdir}")
        if (rdir / "summary.json").exists():
            print(f"Reuse completed: {cfg.exp_id} seed={cfg.seed}", flush=True)
            return json.loads((rdir / "summary.json").read_text())
    (rdir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))

    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_loader = D.make_loader(train_df, cfg.images_dir, D.build_transforms(True, cfg.img_size, cfg.aug),
                                 cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = D.make_loader(val_df, cfg.images_dir, D.build_transforms(False, cfg.img_size),
                               cfg.batch_size, False, None, cfg.num_workers)
    test_loader = (D.make_loader(test_df, cfg.images_dir, D.build_transforms(False, cfg.img_size),
                                 cfg.batch_size, False, None, cfg.num_workers)
                   if cfg.save_test_predictions else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    latest_path = rdir / "latest.pt"
    restoring = cfg.resume and latest_path.exists()
    model = M.build_model(cfg.backbone, not restoring, 9, cfg.drop_rate, cfg.init).to(device)
    if device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if cfg.loss == "ls":
        criterion = L.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    elif cfg.loss == "focal":
        criterion = L.build_criterion("focal", gamma=cfg.focal_gamma)
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().reindex(range(9), fill_value=1)
        w = L.class_weights(counts.tolist(), cfg.class_weight_beta or 0.0).to(device)
        criterion = L.build_criterion("ce_weighted", weight=w)
    else:
        criterion = L.build_criterion("ce")

    opt = build_optimizer(model, cfg)
    sch = build_scheduler(opt, cfg, max(len(train_loader), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.ema_model if ema else model

    history: list = []
    best_f1, best_ep, best_state = -1.0, -1, None
    t_epoch = []
    start_epoch = 0
    if restoring:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"]); sch.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        if ema: ema.ema_model.load_state_dict(saved["ema"])
        history, t_epoch = saved["history"], saved["t_epoch"]
        best_f1, best_ep = saved["best_f1"], saved["best_ep"]
        best_state = torch.load(rdir / "best.pt", map_location=device, weights_only=True)["state"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch = saved["epoch"] + 1
        del saved
        print(f"Resume {cfg.exp_id} at epoch {start_epoch+1}", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        tr = train_one_epoch(model, train_loader, criterion, opt, sch, scaler, cfg, device, ema)
        fn, yt, logits, vl = evaluate(eval_model, val_loader, criterion, device)
        probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
        m = compute_metrics(yt, probs.argmax(1), probs)
        t_epoch.append(tr["time_s"])
        history.append({"epoch": epoch, "train_loss": tr["train_loss"], "val_loss": vl,
                        "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "lr": tr["lr"]})
        print(f"{cfg.exp_id} seed={cfg.seed} epoch={epoch+1}/{cfg.epochs} "
              f"loss={tr['train_loss']:.4f} val_f1={m['macro_f1']:.4f} "
              f"train_s={tr['time_s']:.1f}", flush=True)
        pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
        if m["macro_f1"] > best_f1 + 1e-12:  # hòa -> giữ epoch sớm hơn
            best_f1, best_ep = m["macro_f1"], epoch
            best_state = copy.deepcopy(eval_model.state_dict())
            torch.save({"epoch": epoch, "state": best_state, "cfg": dataclasses.asdict(cfg)},
                       rdir / "best.pt")
        if cfg.resume:
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sch.state_dict(), "scaler": scaler.state_dict(),
                        "ema": ema.ema_model.state_dict() if ema else None,
                        "history": history, "t_epoch": t_epoch, "best_f1": best_f1, "best_ep": best_ep,
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                       rdir / "latest.tmp")
            (rdir / "latest.tmp").replace(latest_path)
    eval_model.load_state_dict(best_state)
    torch.save({"epoch": best_ep, "state": best_state, "cfg": dataclasses.asdict(cfg)}, rdir / "best.pt")

    fn, yt, logits, _ = evaluate(eval_model, val_loader, None, device)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    save_predictions(pred_path(cfg, "val"), fn, yt, probs)
    vm = compute_metrics(yt, probs.argmax(1), probs)
    summary = {"best_epoch": best_ep, "val_macro_f1": vm["macro_f1"], "val_top1": vm["top1"],
               "params_M": M.count_params(eval_model), "train_s_per_epoch": float(np.mean(t_epoch)),
               "tag": getattr(eval_model, "pretrained_tag", cfg.backbone)}
    try:
        summary["gmacs"] = M.count_gmacs(eval_model, cfg.img_size)
    except Exception as e:
        summary["gmacs"] = None
        summary["gmacs_note"] = str(e)
    if test_loader is not None:  # chỉ Bước 4, đúng một lần mỗi seed
        fn, yt, logits, _ = evaluate(eval_model, test_loader, None, device)
        probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
        save_predictions(pred_path(cfg, "test"), fn, yt, probs)
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}_s{cfg.seed}.png",
                f"{cfg.exp_id} {cfg.backbone} seed{cfg.seed}")
    print(json.dumps(summary, indent=2))
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override phải dạng KEY=VALUE, nhận {p!r}")
        k, v = p.split("=", 1)
        if k not in fields:
            raise ValueError(f"Config không có trường {k!r}")
        t = str(fields[k])
        if v in ("none", "None", "null") and "None" in t:
            out[k] = None
        elif v in ("true", "True"):
            out[k] = True
        elif v in ("false", "False"):
            out[k] = False
        elif "int" in t and "float" not in t:
            out[k] = int(v)
        elif "float" in t:
            out[k] = float(v)
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        if k in ("epochs", "batch_size", "img_size", "num_workers") and isinstance(out[k], float):
            out[k] = int(out[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="override KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(run(cfg))


if __name__ == "__main__":
    main()
