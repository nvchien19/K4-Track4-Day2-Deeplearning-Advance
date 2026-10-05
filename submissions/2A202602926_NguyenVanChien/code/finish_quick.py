"""Evaluate the existing MobileNet checkpoint. No additional training or fabricated data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import torch
import copy
import dataclasses
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from run_local import ROOT, SUB, OUT, write_json, status, load_net, predict, latency
from train import Config, plot_curves
from inference import apply_temperature, fit_temperature, fuse_conv_bn
from eval import compute_metrics, save_predictions
import dataset as D

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    quick=OUT/'quick';quick.mkdir(exist_ok=True)
    ckdir=OUT/'runs/B07/seed0'
    cfg=Config(**json.loads((ckdir/'config.json').read_text()))
    base=dataclasses.asdict(cfg)
    net=load_net(cfg)
    train,val,test=D.load_split(cfg.labels_dir)
    history=pd.read_csv(ckdir/'history.csv')
    plot_curves(history.to_dict('records'),SUB/'curves/B07_mobilenetv3_large_100_s0.png','B07 MobileNet • interrupted local training')
    records=[];cache={}
    for method in ('single','flip'):
        status('quick_validation',method=method,completed_epochs=len(history))
        fn,y,z=predict(net,val,base,method)
        cache[method]=(fn,y,z)
        p=apply_temperature(z,1)
        records.append(dict(method=method,**compute_metrics(y,p.argmax(1),p),latency=latency(net,method)))
        write_json(quick/'inference.json',records)
    chosen=max(records,key=lambda x:x['macro_f1'])['method']
    fn,y,z=cache[chosen];temp=fit_temperature(z,y);p=apply_temperature(z,temp)
    records.append(dict(method=chosen+' + temperature',T=temp,**compute_metrics(y,p.argmax(1),p),latency=latency(net,chosen)))
    write_json(quick/'inference.json',records)
    status('quick_test',method=chosen,temperature=temp)
    save_predictions(SUB/'predictions/Q01_seed0_val.csv',fn,y,p)
    raw=quick/'Q01_seed0_test_logits.npz'
    if raw.exists():
        data=np.load(raw);fn,y,z=data['names'].tolist(),data['y'],data['logits']
    else:
        fn,y,z=predict(net,test,base,chosen)
        np.savez_compressed(raw,names=np.array(fn),y=y,logits=z)
    p=apply_temperature(z,temp)
    save_predictions(SUB/'predictions/Q01uncal_seed0_test.csv',fn,y,apply_temperature(z,1))
    save_predictions(SUB/'predictions/Q01_seed0_test.csv',fn,y,p)
    metrics=compute_metrics(y,p.argmax(1),p)
    write_json(quick/'test_metrics.json',metrics)
    summary=dict(exp_id='Q01',backbone=cfg.backbone,seed=0,completed_epochs=len(history),
                 planned_epochs=cfg.epochs,selected_method=chosen,temperature=temp,
                 val_macro_f1=records[-1]['macro_f1'],test_macro_f1=metrics['macro_f1'],
                 test_top1=metrics['top1'],test_ece=metrics['ece'],
                 note='Partial lab: one backbone, one seed, no training ablations. Training stopped after saved epochs; original 12-epoch scheduler.')
    write_json(quick/'summary.json',summary)
    pd.DataFrame([summary]).to_csv(quick/'results_actual.csv',index=False)
    report=f'''# Kết quả chạy nhanh local — số liệu thật

MobileNetV3, seed 0, đã lưu {len(history)} epoch trong kế hoạch 12 epoch.
Backbone ImageNet finetune, 224 px, batch 16, AdamW, CPU 4 thread. LR scheduler vẫn theo kế hoạch 12 epoch ban đầu.
Chọn checkpoint bằng macro-F1 val. Chọn single hoặc flip-logit trên toàn bộ val và khớp nhiệt độ trên val.

- Phương pháp chọn: {chosen}, nhiệt độ T = {temp:.4f}.
- Macro-F1 val: {summary['val_macro_f1']:.4f}.
- Macro-F1 test: {metrics['macro_f1']:.4f}.
- Top-1 test: {metrics['top1']:.4f}.
- ECE test: {metrics['ece']:.4f}.

Test dùng đủ 3.507 ảnh, dự đoán lưu ở predictions/Q01_seed0_test.csv. Các bản hiệu chuẩn và chưa hiệu chuẩn dùng cùng một lượt logit test.

## Phần chưa hoàn thành

Chỉ một backbone và một seed. Chưa có >=5 backbone, ablation trên >=3 trục, >=4 phương pháp suy luận và chung kết >=3 seed. Không báo mean ± std từ một seed. Đây là báo cáo tiến độ, chưa phải bài nộp đầy đủ.
'''
    (SUB/'report_quick_actual.md').write_text(report,encoding='utf-8')
    status('quick_complete',**summary)

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('quick_failed',error=str(e));raise
