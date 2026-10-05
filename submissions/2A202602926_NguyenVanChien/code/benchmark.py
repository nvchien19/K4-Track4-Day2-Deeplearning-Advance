"""benchmark.py - đo độ trễ đúng cách: warmup, synchronize, p50/p95/p99."""
from __future__ import annotations

import time

import torch
import numpy as np  # noqa: E402  (torch truoc numpy: tranh xung dot OpenMP/MKL tren Windows)


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo mili-giây của fn(). sync()=torch.cuda.synchronize hoặc None trên CPU."""
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()
    ts = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "n": int(iters)}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo forward với đầu vào ngẫu nhiên. dtype: fp32 | amp | fp16."""
    assert dtype in ("fp32", "amp", "fp16")
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.eval().to(dev)
    use_half = dtype == "fp16" and dev.type == "cuda"
    if use_half:
        model = model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if use_half:
        x = x.half()
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    @torch.inference_mode()
    def fn():
        if dtype == "amp" and dev.type == "cuda":
            with torch.autocast("cuda"):
                model(x)
        else:
            model(x)
    r = bench(fn, warmup, iters, sync)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    return {"gpu": gpu, "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view: đo thật 1 view rồi so với K*p50 (chi phí ~tuyến tính theo K)."""
    one = latency_report(model, **kw)
    return {**one, "k_views": k_views, "p50_k": one["p50"] * k_views,
            "p95_k": one["p95"] * k_views, "note": "ước lượng K*p50; đo thật khi chạy TTA đầy đủ"}
