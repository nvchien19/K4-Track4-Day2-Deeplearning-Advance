# Lab Day 2 — DeepWeeds | 2A202602926_NguyenVanChien

## Chạy trên Colab (khuyên dùng, GPU T4 miễn phí)
1. Mở https://colab.research.google.com → Upload → chọn file `code/colab_full.ipynb` trong bài nộp này.
2. Runtime → Change runtime type → **GPU (T4)** → Save.
3. Upload 2 thứ vào Colab (kéo-thả vào Files): thư mục `code/` và file `eval.py` (gốc repo, không sửa) đặt cạnh nhau.
4. Runtime → **Run all**. Notebook tự: cài `timm`, tải images.zip (check MD5) + 4 CSV fold 0,
   EDA + kiểm tra split, kiểm tra pipeline, train backbone → ablation → suy luận + đo trễ →
   chung kết 3 seed (test 1 lần/seed) → `eval.py score/grade` → đóng gói outputs.
5. Ô `FULL`: để `False` chạy khói (2 epoch, kiểm tra code), đổi `True` để chạy thật 12 epoch (~20–25 lần train).
6. Sau khi xong: đặt `BEST_BB` (Bước 1), `EXP` (Bước 3), `FINAL_CFG` (Bước 4) theo kết quả val của chính bạn;
   chạy lại Bước 4 rồi tải `colab_outputs.zip`, `all_runs_summary.csv`, `curves/*.png`, `eval_out/grade_I.json` về,
   điền số vào `results.xlsx` + `report.md` (khung có sẵn).
7. Link notebook Colab (dán vào đây trước khi nộp): TODO _paste-link_

## Phiên bản thư viện (đo trên máy smoke-test; cập nhật đúng số Colab trước khi nộp)
- python 3.11.9 | torch 2.13.0 | torchvision + timm: cài trên Colab bằng `!pip -q install timm openpyxl`
  (ghi `torch.__version__`, `timm.__version__` in từ notebook vào đây)
- numpy 1.26.4 | pandas 1.5.3 | scikit-learn 1.9.0 (chỉ cho test) | Pillow 10.4.0 | matplotlib 3.11.0

## Thứ tự chạy (trên Colab/Kaggle, GPU T4 trở lên, bật AMP)
1. Ô cài đặt + tải dữ liệu (kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`, tải 4 CSV fold 0).
2. Bước 0: `dataset.load_split` + `dataset.check_split` (in số ảnh, giao rỗng, hợp 17.509) → EDA → kiểm tra
   pipeline (loss ban đầu ≈ 2.197, overfit 1 batch, xem ảnh sau augmentation).
3. Bước 1 (seed 0 chung): `python code/train.py --set exp_id=B01 backbone=resnet50 seed=0 ...`
   cho B01–B07 (6–7 backbone: ResNet, ResNeXt, ConvNeXt, DeiT, Swin, EfficientNet-B0, MobileNetV3).
4. Bước 2: T00 (nền) rồi T01–T12, mỗi lần khác nền đúng 1 yếu tố (N1).
5. Bước 3: lưu logit val rồi chạy TTA/ensemble/temperature scaling/gộp BN/FP16 + đo trễ
   (`benchmark.latency_report`: warmup ≥ 10, ≥ 50 lần, p50/p95/p99, ghi GPU/dtype/batch).
6. Bước 4 (cuối cùng): chốt cấu hình trên val → chạy F01 + T00 với seed 0,1,2,
   `save_test_predictions=True` (test đúng 1 lần/seed) → `eval.py score` + `eval.py grade`.
7. Điền `results.xlsx`, xuất `curves/*.png` (tự động từ `train.run`), viết `report.md`.

## Seed đã dùng
- Sàng lọc (B, T): seed 0. Chung kết (F01, T00): seed 0, 1, 2. Seed chỉ đổi khởi tạo head,
  thứ tự batch, augmentation — không đổi cách chia (S5).

## Kiểm tra trước nộp
- `python -m unittest discover -s tests` (repo gốc, không cần GPU)
- `python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01`
- `python eval.py grade --final ... --baseline ... --uncal ... --final-val ... --latency-p95-ms ...`
- Không commit `data/`, checkpoint (`*.pt/*.pth`), `runs/`, `eval_out/` (đã có trong `.gitignore`).
