# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds
**MSSV:** 2A202602926_NguyenVanChien · **Trạng thái:** khung + EDA thật, kết quả huấn luyện điền sau khi chạy Colab.
Mọi ô `TODO-Colab` được thay bằng số chạy thật + `exp_id` trước khi nộp.

## 1. Tóm tắt (≤ 10 dòng)
- Bài toán: phân loại 9 lớp cỏ dại DeepWeeds (17.509 ảnh 256×256, mất cân bằng, Negative ≈ 52%).
- TODO-Colab: liệt kê thí nghiệm (≥5 backbone, ≥3 trục huấn luyện, ≥4 PP suy luận), cấu hình tốt nhất
  + test cuối (mean ± std, macro-F1/test, top-1, ECE), kết luận chính 1–2 câu.

## 2. Dữ liệu và thiết lập
- Fold 0 nguyên bản (S1): `train_subset0.csv` / `val_subset0.csv` / `test_subset0.csv`, không sửa/không chia lại.
- Kiểm tra `check_split` (chạy thật ngày 03/10/2026 trên labels gốc):
  - train = **10.501** (59,98%), val = **3.501** (20,00%), test = **3.507** (20,03%); tổng **17.509** ✔
  - giao train∩val = train∩test = val∩test = **0** (rỗng) ✔; hợp = 17.509 ✔
  - tỉ lệ 60/20/20 trong ±1pp ✔. (Kiểm tra file ảnh tồn tại chạy trên Colab sau giải nén.)
- Per-class (train/val/test): Chinee apple 675/225/226; Lantana 637/213/213; Parkinsonia 618/206/207;
  Parthenium 613/204/205; Prickly acacia 637/212/213; Rubber vine 605/202/202; Siam weed 644/215/215;
  Snake weed 609/203/204; Negative 5463/1821/1822.
- Đối chiếu Table 1 bài báo: Negative 9.106 ✔; 6/8 loài khớp; **Chinee apple 1.126 (+1), Lantana 1.063 (−1)**
  — lệch nhỏ, ghi nhận để giảng viên đối chiếu; tỉ lệ max/min ≈ 9106/1009 ≈ 9,0×.
- Chỉ số: chính macro-F1 (val để chọn, test 1 lần/seed ở cuối); phụ top-1, balanced acc, P/R/F1 từng lớp
  (bắt buộc Chinee apple, Snake weed), ECE 15 bin, trễ p50/p95/p99, mean ± std (ddof=1, ≥3 seed).
- Công thức nền T00: ImageNet-finetune toàn bộ; train RandomResizedCrop(224)+hflip, val/test Resize256+CenterCrop224,
  chuẩn hoá ImageNet; AdamW (lr_bb=1e-4, lr_head=1e-3, wd=0,05, trừ norm/bias); warmup 1 epoch + cosine; CE; bs=64;
  12 epoch; AMP; checkpoint = macro-F1 val cao nhất (hòa lấy sớm hơn).
- Phần cứng + lib: TODO-Colab (tên GPU, torch/timm version, `IMAGES_DIR` thực tế).
- Kiểm tra pipeline (§1.3 GUIDE): TODO-Colab (loss ≈ 2,197 + overfit 1 batch + ảnh sau aug) — code hỗ trợ sẵn.

## 3. So sánh backbone (TODO-Colab: điền từ sheet Backbones)
- Bảng + scatter F1–trễ/params; nhận xét hội tụ/quá khớp; thứ hạng DeepWeeds vs ImageNet; FLOPs có dự báo trễ không;
  lý do chọn 1–2 backbone đi tiếp (số liệu, gồm đánh đổi trễ).

## 4. Công thức huấn luyện (TODO-Colab: điền từ sheet Training)
- Bảng ablation từng trục (A–G) với Δ vs T00 + so với std; yếu tố nào giúp/hại và vì sao (liên hệ slide);
  ít nhất 1 kết hợp (cộng dồn hay triệt tiêu); Mixup/CutMix đánh giá bằng val.

## 5. Suy luận (TODO-Colab: điền từ sheet Inference/Latency)
- Bảng + scatter đánh đổi F1–trễ; ECE trước/sau temperature scaling (T khớp trên val);
  kết luận ngoại tuyến vs thời gian thực; TTA có đáng chi phí K× không.

## 6. Cấu hình tốt nhất (chung kết, test 1 lần/seed)
- TODO-Colab: mô tả đầy đủ để tái lập; bảng mean ± std (macro-F1/top-1 test, F1 Chinee apple/Snake weed, ECE);
  `eval.py score/grade` khớp xlsx; ma trận nhầm lẫn + phân tích lỗi (cặp Chinee apple ↔ Snake weed) kèm ảnh sai;
  2 đề xuất: tốt nhất ngoại tuyến + tốt nhất realtime (p95 ≤ 100 ms batch-1).

## 7. Kết luận và khuyến nghị (TODO-Colab)
- Cấu hình nào tốt nhất? Hơn mốc T00+I00 bao nhiêu, có vượt nhiễu (Δ vs s) không?
- Đóng góp lớn nhất: backbone / huấn luyện / suy luận?
- Robot 30–100 ms/khung: chọn gì, vì sao?

## 8. Hạn chế và việc tiếp theo
- Mới 1 fold (fold 0), sàng lọc 1 seed; chia ngẫu nhiên không theo địa điểm → test có thể lạc quan;
  ngân sách GPU Colab (ghi rõ chỗ đã cắt giảm); rủi ro lệch miền (mùa/ánh sáng/địa điểm mới);
  hướng tiếp theo: nhiều fold/seed, chưng cất, TTA-adaptation trên miền lệch tự tạo (không dùng test fold 0).

## Phụ lục
- Danh sách `exp_id` đầy đủ trong `results.xlsx` (Summary); log/config/history/checkpoint tốt nhất ở `runs/<exp_id>/seed<k>/`
  trên Colab (chỉ commit xlsx/png/code/predictions, không commit checkpoint/dataset).
