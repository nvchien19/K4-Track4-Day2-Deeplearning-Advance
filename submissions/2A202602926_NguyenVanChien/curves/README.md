# curves/

Mỗi thí nghiệm huấn luyện (B, T, F) một ảnh `<exp_id>_<mota>.png`: loss train/val +
macro-F1 val theo epoch (+LR), có tiêu đề/nhãn/chú thích (GUIDE 6.2).
`B07_mobilenetv3_large_100_s0.png` được tạo từ 4 epoch chạy thật trên CPU local. Lịch sử gốc ở
`../evidence/B07_seed0_history.csv`. Các thí nghiệm chưa chạy chưa có đường cong.
`code/train.run` tự xuất ảnh khi một lần chạy đầy đủ hoàn tất; `finish_quick.py` xuất ảnh từ lần chạy bị dừng.
