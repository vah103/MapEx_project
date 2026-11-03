# Kết quả nghiên cứu MapEx — snapshot ban đầu 25/10, bổ sung TOPO R2 03/11/2026

- [G1/G2 hiện hành](g1_g2): New Room 32 run, KTH audits, bảng số liệu, receipt, quỹ đạo và gallery P00.
- [MapEx-GTP](gtp): 36 run R7 hoàn tất, metrics/quỹ đạo/mask cuối; riêng R7AL là trạng thái pending và kiểm offline mới nhất.
- [MapEx + TOPO R2 — ba bộ ảnh cuộn KTH49 P00 (42 decisions), KTH50 P00 (27 decisions), KTH51 P01 (37 decisions)](../mapex_topo/galleries/README.md): các gallery HTML nay được lưu trực tiếp trong `mapex_topo/galleries/`, giải thích cách robot chọn goal qua từng decision. Đây là các run phát triển, không phải kết luận đa run.
- [TOPO R2 — 10 run KTH và bằng chứng](../mapex_topo): 3 run phát triển có trong `eval_final/`, 7 run bổ sung có [báo cáo và dữ liệu từng tick](../mapex_topo/eval_holdout7/REPORT.md), decision logs, quỹ đạo, receipts, ảnh debug, SHA kiểm kê dữ liệu raw trên Dell.

Chỉ lưu kết quả tương ứng các nhánh hiện hành. Baseline/ablation cùng điều kiện được giữ để so sánh. Không lấy G2-R8 hoặc gộp kết quả G1 cũ.

RESULTS_MANIFEST.json ghi source path/hash của bộ lưu trữ G1/GTP và phép biến đổi CSV/RLE. Bộ TOPO R2 sử dụng `mapex_topo/SHA256SUMS`, các `RUN_MANIFEST.json` và `raw_inventory.json` độc lập. Audit gốc được giữ nguyên kể cả lỗi và giới hạn; không tạo số liệu mô phỏng mới. Các mask GTP RLE giữ nguyên giá trị logic, không giả vờ khớp byte NPZ.

Đây là gói số liệu và bằng chứng gọn, không sao lưu toàn bộ ảnh LiDAR/prediction mỗi tick hoặc checkpoint model.
