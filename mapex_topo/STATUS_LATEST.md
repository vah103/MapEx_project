# TOPO-002 R2 — trạng thái cập nhật 03/11/2026

Tệp `README.md` và 43 file trong `SHA256SUMS` là snapshot niêm phong **trước khi chạy bảy cấu hình bổ sung**. Các dòng ghi “7 cấu hình đề xuất chạy” mô tả trạng thái ở thời điểm đóng băng, không phải trạng thái hiện tại. Giữ nguyên byte để đối chiếu checksum.

## Mã thực thi

- Phiên bản: TOPO-002 Planner R2, RoomNode L1 v2, p(c) frozen, native MapEx 2D.
- Nguồn GitHub niêm phong: `6ad6c67190726f5a60a8fc5bae835e65c3822e9c`.
- Planner R2 `77f24477…`, policy `4269077b…`, native `445608a1…`, L1 v2 `7aba149a…`, p(c) `fc2c40f8…`, cấu hình `fe1f75a0…`.
- Giữ nguyên 43/43 SHA-256 của snapshot.

## Kết quả

- [Ba run đánh giá ban đầu](eval_final/REPORT.md): KTH49 P00, KTH50 P00, KTH51 P01.
- [Bảy run KTH bổ sung](eval_holdout7/REPORT.md): KTH49 P01/P03, KTH50 P01/P03, KTH51 P00/P02/P03.
- [Bảng tổng hợp MapEx vs TOPO R2 cho cả 10 run](RESULTS_10_RUNS.csv): coverage AUC, coverage tick 999 và quãng đường.
- [Ba bộ ảnh cuộn HTML](galleries/README.md): KTH49 P00 (42 decisions), KTH50 P00 (27 decisions), KTH51 P01 (37 decisions); nội dung chuyển nguyên byte từ `results/topo_r2/`.
- Cả 7 run bổ sung có `frozen_check=PASS`, `launch_exit=0`, 1.000/1.000 tick; batch `COMPLETED`. Không thực hiện retry, không chỉnh thuật toán.
- Trong `eval_final/per_run/` và `eval_holdout7/per_run/`: dữ liệu compact để kiểm chứng từng run, cùng đường dẫn và SHA kiểm kê của dữ liệu raw trên Dell.
- Chạy `python3 tools/verify_topo_results.py` từ repo root để kiểm tra gói công bố.

Bản đồ/LaMa checkpoint và chuỗi ảnh native 1.000 tick mỗi run vẫn được bảo toàn trên Dell; Git không chứa bản sao dữ liệu raw cồng kềnh. Ba run ban đầu được dùng chẩn đoán R1, nên không phải hold-out độc lập. Kết quả cần được diễn giải theo giới hạn thực nghiệm, không mặc định suy ra tổng quát hóa.
