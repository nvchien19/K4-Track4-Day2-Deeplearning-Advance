# predictions/

Chứa file dự đoán TEST của chung kết (F01) và mốc (T00), mỗi seed một file, đúng định dạng
`eval.py` (README mục 2.2): cột `Filename, y_true, y_pred, p0..p8` (softmax, tổng = 1,
`y_pred` = argmax). Tạo bằng `eval.save_predictions` (tự động trong `code/train.run`
khi `save_test_predictions=True`, hoặc từ xác suất cuối của TTA/ensemble/temperature scaling).

Tên chuẩn: `<exp_id>_seed<k>_test.csv` (vd `F01_seed0_test.csv`), kèm bản val
(`F01_seed<k>_val.csv`) và bản chưa temperature scaling (`F01uncal_seed<k>_test.csv`)
để `eval.py grade` chấm I4.

Hiện thư mục trống — chạy Bước 4 trên Colab rồi copy file vào đây. Không tự viết số liệu.
