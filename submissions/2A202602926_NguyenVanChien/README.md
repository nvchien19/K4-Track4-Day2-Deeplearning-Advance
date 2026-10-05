# DeepWeeds — Nguyễn Văn Chiến (2A202602926)

## Trạng thái thực tế

Bản local rút gọn hoàn tất ngày 05/10/2026. Một backbone MobileNetV3, 4 epoch đã lưu, seed 0. Test Q01: macro-F1 0.7770, top-1 0.8386, ECE 0.0126. Đây là bài lab một phần. Các thí nghiệm chưa chạy được ghi rõ trong report và Excel.

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
