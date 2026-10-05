# predictions/

Chứa file dự đoán TEST của chung kết (F01) và mốc (T00), mỗi seed một file, đúng định dạng
`eval.py` (README mục 2.2): cột `Filename, y_true, y_pred, p0..p8` (softmax, tổng = 1,
`y_pred` = argmax). Tạo bằng `eval.save_predictions` (tự động trong `code/train.run`
khi `save_test_predictions=True`, hoặc từ xác suất cuối của TTA/ensemble/temperature scaling).

Tên chuẩn: `<exp_id>_seed<k>_test.csv` (vd `F01_seed0_test.csv`), kèm bản val
(`F01_seed<k>_val.csv`) và bản chưa temperature scaling (`F01uncal_seed<k>_test.csv`)
để `eval.py grade` chấm I4.

Hiện có kết quả thật của bản local rút gọn, seed 0:

- `Q01_seed0_val.csv`: 3.501 ảnh validation, hflip gộp logits và temperature scaling.
- `Q01_seed0_test.csv`: đủ 3.507 ảnh test, đã hiệu chuẩn.
- `Q01uncal_seed0_test.csv`: cùng logit test, chưa hiệu chuẩn.

`eval.py score` đã đối chiếu tên ảnh/nhãn với fold 0 và tính lại chỉ số; kết quả ở `../evidence/eval_Q01/`.
Chưa có nhóm F01/T00 nhiều seed. Logit test đã lưu ở `local_outputs/quick/` trong repo để tránh chạy test lại.
