# Kết quả chạy nhanh local — số liệu thật

MobileNetV3, seed 0, đã lưu 4 epoch trong kế hoạch 12 epoch.
Backbone ImageNet finetune, 224 px, batch 16, AdamW, CPU 4 thread. LR scheduler vẫn theo kế hoạch 12 epoch ban đầu.
Chọn checkpoint bằng macro-F1 val. Chọn single hoặc flip-logit trên toàn bộ val và khớp nhiệt độ trên val.

- Phương pháp chọn: flip, nhiệt độ T = 1.2052.
- Macro-F1 val: 0.7938.
- Macro-F1 test: 0.7770.
- Top-1 test: 0.8386.
- ECE test: 0.0126.

Test dùng đủ 3.507 ảnh, dự đoán lưu ở predictions/Q01_seed0_test.csv. Các bản hiệu chuẩn và chưa hiệu chuẩn dùng cùng một lượt logit test.

## Phần chưa hoàn thành

Chỉ một backbone và một seed. Chưa có >=5 backbone, ablation trên >=3 trục, >=4 phương pháp suy luận và chung kết >=3 seed. Không báo mean ± std từ một seed. Đây là báo cáo tiến độ, chưa phải bài nộp đầy đủ.
