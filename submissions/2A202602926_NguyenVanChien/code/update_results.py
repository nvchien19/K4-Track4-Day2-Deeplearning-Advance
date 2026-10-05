"""Collect real quick-run evidence and populate report/README without inventing missing experiments."""
import torch
import json
import re
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];SUB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval import read_pred,compute_metrics
import model as M

def main():
    quick=ROOT/'local_outputs/quick'
    summary=json.loads((quick/'summary.json').read_text())
    hist=pd.read_csv(ROOT/'local_outputs/runs/B07/seed0/history.csv')
    cfg=json.loads((ROOT/'local_outputs/runs/B07/seed0/config.json').read_text())
    env=json.loads((ROOT/'local_outputs/environment.json').read_text())
    inference=json.loads((quick/'inference.json').read_text())
    def metrics(name):
        p=read_pred(str(SUB/'predictions'/name));return compute_metrics(p.y_true,p.y_pred,p.probs)
    test=metrics('Q01_seed0_test.csv');uncal=metrics('Q01uncal_seed0_test.csv');val=metrics('Q01_seed0_val.csv')
    torch.set_num_threads(4)
    net=M.build_model(cfg['backbone'],False,9)
    params=M.count_params(net);gmac=M.count_gmacs(net,224)
    log=(ROOT/'logs/local_full.log').read_text()
    times=[float(x) for x in re.findall(r'epoch=\d+/12 .*?train_s=([0-9.]+)',log)]
    evidence=SUB/'evidence';evidence.mkdir(exist_ok=True)
    data=dict(summary=summary,history=hist.to_dict('records'),config=cfg,environment=env,
              inference=inference,test=test,val=val,uncal_test=uncal,params_M=params,gmacs=gmac,
              train_s_per_epoch=float(np.mean(times)) if times else None,epoch_train_times=times,
              split=json.loads((ROOT/'local_outputs/split_check.json').read_text()))
    (evidence/'results_actual.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')
    hist.to_csv(evidence/'B07_seed0_history.csv',index=False)
    subprocess.run([sys.executable,str(ROOT/'eval.py'),'score','--pred',str(SUB/'predictions/Q01_seed0_test.csv'),
                    '--test-csv',str(ROOT/'data/labels/test_subset0.csv'),'--labels',str(ROOT/'data/labels/labels.csv'),
                    '--tag','Q01','--out',str(evidence/'eval_Q01')],check=True)
    classes=['Chinee apple','Lantana','Parkinsonia','Parthenium','Prickly acacia','Rubber vine','Siam weed','Snake weed','Negative']
    rows='\n'.join(f"| {i+1} | {r.train_loss:.4f} | {r.val_loss:.4f} | {r.val_macro_f1:.4f} | {r.val_top1:.4f} |" for i,r in hist.iterrows())
    infrows='\n'.join(f"| {r['method']} | {r['macro_f1']:.4f} | {r['top1']:.4f} | {r['ece']:.4f} | {r['latency']['p50']:.2f} | {r['latency']['p95']:.2f} | {r['latency']['p99']:.2f} |" for r in inference)
    pcrows='\n'.join(f"| {n} | {int(test['support'][i])} | {test['precision'][i]:.4f} | {test['recall'][i]:.4f} | {test['f1'][i]:.4f} |" for i,n in enumerate(classes))
    cmrows='\n'.join('| '+classes[i]+' | '+' | '.join(str(int(x)) for x in test['confusion'][i])+' |' for i in range(9))
    report=f'''# DeepWeeds — kết quả thật của bản chạy rút gọn

MSSV: 2A202602926. Họ tên: Nguyễn Văn Chiến. Cập nhật ngày 05/10/2026.

## 1. Tóm tắt

Đã huấn luyện MobileNetV3 Large trên toàn bộ train fold 0, hoàn thành {len(hist)} epoch và dừng hàng đợi dài theo yêu cầu. Chọn checkpoint bằng macro-F1 val, so sánh single-view và hflip trên toàn bộ val, sau đó khớp temperature scaling trên val. Cấu hình Q01 dùng {summary['selected_method']}, seed 0, T={summary['temperature']:.4f}. Macro-F1 test={test['macro_f1']:.4f}, top-1={test['top1']:.4f}, balanced accuracy={test['balanced_acc']:.4f}, ECE={test['ece']:.4f}. Đây là bài lab một phần, chưa đủ số thí nghiệm và seed theo rubric.

## 2. Dữ liệu và thiết lập

DeepWeeds có 17.509 ảnh, 9 lớp. Dùng nguyên bản fold 0: train 10.501, val 3.501, test 3.507. Kiểm tra local xác nhận giao các tập rỗng và mọi file ảnh tồn tại. Không thay đổi split, không gộp val vào train. Test chỉ dùng sau khi đã chọn cách suy luận trên val.

| Lớp | Train | Val | Test |
|---|---:|---:|---:|
'''+ '\n'.join(f"| {n} | {data['split']['per_class']['train'][str(i)]} | {data['split']['per_class']['val'][str(i)]} | {data['split']['per_class']['test'][str(i)]} |" for i,n in enumerate(classes))+f'''

Negative có 9.106 ảnh, khoảng 52%. Nhãn CSV có Chinee apple 1.126 và Lantana 1.063, khác bảng tham khảo trong README mỗi lớp một ảnh. Giữ nguyên CSV của tác giả.

Máy chạy CPU {env['cpu']}, 4 thread, không CUDA. Python 3.11.9, torch 2.13.0+cpu, torchvision 0.29.1+cpu, timm 1.0.30, NumPy 1.26.4, pandas 1.5.3. MobileNetV3 dùng trọng số ImageNet của timm, finetune toàn bộ, {params:.3f} triệu tham số, khoảng {gmac:.3f} GMAC (đếm Conv2d/Linear, chưa gồm mọi phép toán).

Train: RandomResizedCrop224, hflip, chuẩn hóa ImageNet. Val/test: Resize256, CenterCrop224, chuẩn hóa ImageNet. CE, AdamW, LR backbone 1e-4, head 1e-3, weight decay 0.05, warmup 1 epoch, batch 16, AMP tắt. Hoàn thành 4 epoch trong lịch warmup+cosine 12 epoch. Không coi đây là cosine 4 epoch đã kết thúc. Chọn epoch có val macro-F1 cao nhất, hòa lấy sớm hơn.

38 test repo và forward 9 lớp đã đạt. Chưa có bằng chứng lưu cho kiểm tra overfit một batch và hình augmentation nên không ghi chúng đã hoàn thành.

## 3. Backbone và quá trình huấn luyện

Chỉ B07 MobileNet đã có kết quả thật. B01–B06 chưa chạy hoàn tất nên chưa thể xếp hạng các kiến trúc hoặc kết luận FLOPs dự báo độ trễ.

| Epoch (từ 1) | Train loss | Val loss | Val macro-F1 | Val top-1 |
|---:|---:|---:|---:|---:|
{rows}

![Đường cong MobileNet](curves/B07_mobilenetv3_large_100_s0.png)

Macro-F1 val tăng từ {hist.iloc[0].val_macro_f1:.4f} lên {hist.iloc[-1].val_macro_f1:.4f}. Epoch 4 có F1 tốt hơn epoch 3 nhưng val loss tăng. Chưa đủ bằng chứng để kết luận về hội tụ hoặc quá khớp dài hạn.

## 4. Công thức huấn luyện

Đã chạy công thức nền ở B07. T00–T12 chưa được chạy như các thí nghiệm độc lập. Chưa có ablation khởi tạo, augmentation, loss, sampler, LR hay EMA. Chưa có delta so với T00 hoặc bằng chứng kết hợp các yếu tố.

## 5. Suy luận và độ trễ

| Phương pháp | Macro-F1 val | Top-1 val | ECE val | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|
{infrows}

Hflip gộp logits, không gộp xác suất. T khớp riêng trên val. Độ trễ đo CPU fp32, batch 1, ảnh 224, 10 warmup và 50 lần đo, chỉ tính forward với input tổng hợp. Chưa gồm tải ảnh, tiền xử lý hoặc I/O. Hàng temperature đo forward của phương pháp đã chọn, chưa cộng chi phí softmax/calibration. Không coi đó là độ trễ toàn pipeline.

## 6. Kết quả test Q01 seed 0

Toàn bộ 3.507 ảnh. Các bản hiệu chuẩn và chưa hiệu chuẩn được tạo từ cùng một lượt logit test đã lưu.

| Chỉ số | Giá trị |
|---|---:|
| Macro-F1 val sau calibration | {val['macro_f1']:.4f} |
| Macro-F1 test | {test['macro_f1']:.4f} |
| Top-1 test | {test['top1']:.4f} |
| Balanced accuracy test | {test['balanced_acc']:.4f} |
| ECE test trước calibration | {uncal['ece']:.4f} |
| ECE test sau calibration | {test['ece']:.4f} |

Chỉ một seed nên chưa có mean ± std. `eval.py score` được chạy từ file dự đoán gốc. Chưa chạy `grade` vì không có nhóm chung kết/mốc >=3 seed tương ứng.

| Lớp | Số ảnh test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{pcrows}

### Ma trận nhầm lẫn

Hàng là nhãn thật, cột là dự đoán, thứ tự lớp theo Label 0–8.

| Nhãn thật | '''+' | '.join(classes)+''' |
|---|'''+ '|'.join(['---:']*9)+f'''|
{cmrows}

Chinee apple bị đoán thành Snake weed: {int(test['confusion'][0,7])} ảnh. Chiều ngược lại: {int(test['confusion'][7,0])} ảnh. Chưa có tập ảnh lỗi minh họa được xuất.

## 7. Kết luận

Q01 là cấu hình đã đánh giá trong phạm vi rút gọn. Chưa thể gọi là cấu hình tốt nhất của toàn bộ lab, chưa thể so sánh cải thiện với mốc nhiều seed hoặc đánh giá độ ổn định. Các số độ trễ chỉ dùng để mô tả CPU này, chưa đủ để khuyến nghị triển khai robot.

## 8. Hạn chế và phần chưa hoàn thành

- Một backbone, một seed, dừng sau 4 epoch.
- Chưa đủ >=5 backbone, >=3 trục ablation, >=4 phương pháp suy luận và >=3 seed cho chung kết/mốc.
- Chưa có kiểm tra overfit một batch, scatter F1–latency, ảnh lỗi và benchmark toàn pipeline.
- Fold 0 chia ngẫu nhiên theo ảnh, không theo địa điểm nên test có thể lạc quan. Chưa đo lệch miền mùa/ánh sáng/địa điểm.

## Bằng chứng tái lập

`evidence/results_actual.json`, `evidence/B07_seed0_history.csv`, `evidence/eval_Q01/`, `predictions/Q01_seed0_val.csv`, `predictions/Q01_seed0_test.csv`, `predictions/Q01uncal_seed0_test.csv`. Code gốc `eval.py` không sửa. `results.xlsx` chứa số đã đo và ghi rõ các dòng chưa chạy.
'''
    (SUB/'report.md').write_text(report,encoding='utf-8')
    readme=f'''# DeepWeeds — Nguyễn Văn Chiến (2A202602926)

## Trạng thái thực tế

Bản local rút gọn hoàn tất ngày 05/10/2026. Một backbone MobileNetV3, 4 epoch đã lưu, seed 0. Test Q01: macro-F1 {test['macro_f1']:.4f}, top-1 {test['top1']:.4f}, ECE {test['ece']:.4f}. Đây là bài lab một phần. Các thí nghiệm chưa chạy được ghi rõ trong report và Excel.

## Xem kết quả

- `results.xlsx`: bảng số liệu thật, chưa chạy được ghi rõ.
- `report.md`: thiết lập, đường cong, suy luận, test, từng lớp, ma trận nhầm lẫn và hạn chế.
- `evidence/`: cấu hình, số liệu, lịch sử và kết quả tính lại bằng eval.py.
- `predictions/Q01_seed0_test.csv`: đủ 3.507 ảnh, y_true/y_pred/p0…p8.

## Môi trường đã kiểm tra

Python 3.11.9, torch 2.13.0+cpu, torchvision 0.29.1+cpu, timm 1.0.30, NumPy 1.26.4, pandas 1.5.3. CPU 4 thread, không CUDA. 38 test repo đã đạt. `.venv-local` kế thừa các gói Python đã cài trên máy và ghim NumPy 1.26.4 riêng, không phải môi trường portable.

## Chạy lại trên máy hiện tại

Từ thư mục gốc repo, checkpoint `local_outputs/runs/B07/seed0/best.pt` phải tồn tại.

```powershell
& ./.venv-local/Scripts/python.exe -u submissions/2A202602926_NguyenVanChien/code/finish_quick.py
& ./.venv-local/Scripts/python.exe submissions/2A202602926_NguyenVanChien/code/update_results.py
& ./.venv-local/Scripts/python.exe eval.py score --pred "submissions/2A202602926_NguyenVanChien/predictions/Q01_seed0_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag Q01
```

`finish_quick.py` tái sử dụng logit test đã lưu nếu có. Không xóa logit đó để tránh đánh giá test lại. Chọn single/hflip và khớp T chỉ bằng val.

## Muốn chạy đầy đủ sau này

`code/run_local.py --epochs 12 --batch-size 16 --threads 4` tiếp tục hàng đợi đầy đủ. Hàng đợi này đã dừng theo yêu cầu vì CPU chậm. Chưa có kết quả F01/BASE12 với seed 0,1,2. Chỉ seed 0 đã thực sự được dùng trong bản Q01.

Notebook đã dùng trước đây: https://colab.research.google.com/drive/1eP28eoCj_IswTTaONWfsRa-fUdskUp8_
`code/colab_autorun.ipynb` chứa code nhúng tại thời điểm chuẩn bị Colab, không tự đồng bộ với chỉnh sửa local mới nhất. Kết quả báo cáo hiện tại chỉ lấy từ chạy local.

Không commit data, checkpoint, môi trường ảo hoặc logit NPZ. Có thể commit code, Excel, report, curves, predictions và evidence. Checkpoint chưa có link chia sẻ; người chạy lại ngoài máy này cần huấn luyện từ đầu.
'''
    (SUB/'README.md').write_text(readme,encoding='utf-8')
    print('Updated real evidence, report and README',flush=True)

if __name__=='__main__':main()
