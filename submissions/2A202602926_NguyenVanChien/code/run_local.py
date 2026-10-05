"""Full local experiment queue. Resume with the same command after interruption."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("MPLBACKEND", "Agg")
import torch
import argparse
import copy
import dataclasses
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_outputs"
sys.path.insert(0, str(ROOT))
import dataset as D
import model as M
from train import Config, run
from inference import aggregate_views, apply_temperature, fit_temperature, views_multicrop
from benchmark import bench
from eval import compute_metrics, save_predictions

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)

def status(state, **details):
    write_json(OUT / "status.json", dict(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S"), **details))
    print(state, details, flush=True)

def collect():
    rows = []
    for p in sorted((OUT / "runs").glob("*/seed*/summary.json")):
        rows.append({**json.loads((p.parent / "config.json").read_text()),
                     **json.loads(p.read_text()), "source": str(p)})
    pd.DataFrame(rows).to_csv(OUT / "all_runs_summary.csv", index=False)
    return rows

def load_net(cfg):
    net = M.build_model(cfg.backbone, False, 9, cfg.drop_rate, "finetune")
    ck = torch.load(Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}" / "best.pt",
                    map_location="cpu", weights_only=True)
    net.load_state_dict(ck["state"])
    return net.eval()

@torch.inference_mode()
def forward(net, x, method):
    if method == "flip":
        return (net(x) + net(torch.flip(x, [-1]))) / 2
    if method == "fivecrop":
        probs = torch.stack([net(v).softmax(1) for v in views_multicrop(x, 224)]).mean(0)
        return probs.clamp_min(1e-12).log()
    return net(x)

def predict(net, df, base, method="single"):
    from torchvision import transforms as T
    tf = D.build_transforms(False, 224)
    if method == "fivecrop":
        tf = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize(D.IMAGENET_MEAN,D.IMAGENET_STD)])
    loader = D.make_loader(df, base["images_dir"], tf, base["batch_size"], False, None, 0)
    names, ys, logits = [], [], []
    for x,y,fn in loader:
        names.extend(fn); ys.extend(y.tolist()); logits.append(forward(net,x,method).numpy())
    return names, np.array(ys), np.concatenate(logits)

def latency(net, method):
    x = torch.randn(1,3,256 if method == "fivecrop" else 224,256 if method == "fivecrop" else 224)
    return {**bench(lambda: forward(net,x,method), 10, 50), "device":"CPU", "dtype":"fp32", "batch":1}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--threads",type=int,default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=True); os.chdir(SUB)
    base = dict(epochs=args.epochs,batch_size=args.batch_size,num_workers=0,amp=False,
                images_dir=str(ROOT / "data"), labels_dir=str(ROOT / "data/labels"),
                out_dir=str(OUT / "runs"),pred_dir=str(SUB / "predictions"),resume=True)
    write_json(OUT / "environment.json", dict(python=sys.version,torch=torch.__version__,
                cpu=platform.processor(),cuda=torch.cuda.is_available(),threads=args.threads,base=base))
    train,val,test = D.load_split(base["labels_dir"])
    write_json(OUT / "split_check.json",D.check_split(train,val,test,base["images_dir"]))
    def execute(eid, backbone, seed=0, **recipe):
        status("training",experiment=eid,backbone=backbone,seed=seed)
        cfg = Config(exp_id=eid,backbone=backbone,seed=seed,**dict(base,**recipe))
        result = run(cfg); collect(); gc.collect()
        return cfg,result
    # Small networks first; all use the same recipe and complete official split.
    backbones = [("B07","mobilenetv3_large_100"),("B06","efficientnet_b0"),
                 ("B04","deit_small_patch16_224"),("B01","resnet50"),
                 ("B03","convnext_tiny"),("B02","resnext50_32x4d"),("B05","swin_tiny_patch4_window7_224")]
    results = [(eid,bb,execute(eid,bb)[1]) for eid,bb in backbones]
    best = max(results,key=lambda r:r[2]["val_macro_f1"])[1]
    recipes = [("T00",{}),("T01",dict(init="scratch")),("T02",dict(init="frozen")),
               ("T03",dict(aug="color")),("T04",dict(aug="trivial")),("T05",dict(mix="cutmix")),
               ("T06",dict(loss="ls",label_smoothing=.1)),("T07",dict(loss="focal")),
               ("T08",dict(loss="ce_weighted")),("T09",dict(sampler="balanced")),
               ("T10",dict(lr_backbone=1e-3)),("T11",dict(ema_decay=.999)),
               ("T12",dict(aug="color",mix="cutmix",loss="ls",label_smoothing=.1))]
    trained = [(eid,kw,*execute(eid,best,**kw)) for eid,kw in recipes]
    eid,recipe,cfg,_ = max(trained,key=lambda r:r[3]["val_macro_f1"])
    status("inference_validation",experiment=eid)
    net = load_net(cfg); comparisons=[]
    for method in ("single","flip","fivecrop"):
        fn,y,z = predict(net,val,base,method)
        p = apply_temperature(z,1.)
        comparisons.append(dict(method=method,**compute_metrics(y,p.argmax(1),p),latency=latency(net,method)))
    fn,y,z = predict(net,val,base)
    temp=fit_temperature(z,y); p=apply_temperature(z,temp)
    comparisons.append(dict(method="temperature",T=temp,**compute_metrics(y,p.argmax(1),p),latency=latency(net,"single")))
    write_json(OUT / "inference_results.json", comparisons)
    selected = max(comparisons,key=lambda r:r["macro_f1"])
    method=selected["method"] if selected["method"] != "temperature" else "single"
    write_json(OUT / "final_selection.json",dict(backbone=best,experiment=eid,recipe=recipe,method=method))
    del net; gc.collect()
    for exp,kw,view in (("F01",recipe,method),("BASE12",{},"single")):
        for seed in (0,1,2):
            c,_=execute(exp,best,seed,**kw)
            raw=OUT / f"{exp}_seed{seed}_test_logits.npz"
            calibrated=SUB / "predictions" / f"{exp}_seed{seed}_test.csv"
            if calibrated.exists() and raw.exists(): continue
            net=load_net(c); fn,y,z=predict(net,val,base,view)
            t=fit_temperature(z,y) if exp=="F01" else 1.
            save_predictions(SUB / "predictions" / f"{exp}_seed{seed}_val.csv",fn,y,apply_temperature(z,t))
            # Persist logits immediately; reruns never forward the test set again.
            if raw.exists():
                saved=np.load(raw); fn,y,z=saved["names"].tolist(),saved["y"],saved["logits"]
            else:
                fn,y,z=predict(net,test,base,view)
                np.savez_compressed(raw,names=np.array(fn),y=y,logits=z)
            save_predictions(SUB / "predictions" / f"{exp}uncal_seed{seed}_test.csv",fn,y,apply_temperature(z,1.))
            save_predictions(calibrated,fn,y,apply_temperature(z,t))
            write_json(OUT / f"{exp}_seed{seed}_temperature.json",dict(T=t,method=view))
            del net; gc.collect()
    common=["--test-csv",str(ROOT / "data/labels/test_subset0.csv"),"--labels",str(ROOT / "data/labels/labels.csv"),"--out",str(OUT / "eval_out")]
    for exp in ("F01","BASE12"):
        subprocess.run([sys.executable,str(ROOT / "eval.py"),"score","--pred",str(SUB / "predictions" / f"{exp}_seed*_test.csv"),"--tag",exp,*common],check=True)
    subprocess.run([sys.executable,str(ROOT / "eval.py"),"grade","--final",str(SUB / "predictions/F01_seed*_test.csv"),
                    "--baseline",str(SUB / "predictions/BASE12_seed*_test.csv"),"--uncal",str(SUB / "predictions/F01uncal_seed*_test.csv"),
                    "--final-val",str(SUB / "predictions/F01_seed*_val.csv"),"--val-csv",str(ROOT / "data/labels/val_subset0.csv"),*common],check=True)
    collect(); status("experiments_complete",note="Measured outputs ready; workbook and narrative report still require synthesis.")

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        status("failed",error=f"{type(exc).__name__}: {exc}")
        raise
