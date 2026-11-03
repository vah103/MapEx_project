# Kết quả nghiên cứu MapEx — snapshot ban đầu 25/10, bổ sung TOPO R2 03/11/2026

- [G1/G2 hiện hành](g1_g2): New Room 32 run, KTH audits, bảng số liệu, receipt, quỹ đạo và gallery P00.
- [MapEx-GTP](gtp): 36 run R7 hoàn tất, metrics/quỹ đạo/mask cuối; riêng R7AL là trạng thái pending và kiểm offline mới nhất.
- [MapEx + TOPO R2 — KTH49 v3 P00, bộ ảnh cuộn 42 decisions](https://raw.githack.com/vah103/MapEx_project/main/results/topo_r2/kth49_p00/MAPEX_TOPO_R2_KTH49_P00_42_DECISIONS_REASONING.html): 168 ảnh nhúng và giải thích lựa chọn goal của một run 1.000 tick; đây là minh họa theo từng decision, không phải kết luận đa run.
- [MapEx + TOPO R2 — KTH50 v3 P00, 27 decisions](https://raw.githack.com/vah103/MapEx_project/main/results/topo_r2/kth50_p00/MAPEX_TOPO_R2_KTH50_P00_27_DECISIONS_REASONING.html): 108 ảnh nhúng, tour dự kiến, lựa chọn goal và quỹ đạo hậu kiểm trong ngân sách 1.000 tick.
- [MapEx + TOPO R2 — KTH51 v3 P01, 37 decisions](https://raw.githack.com/vah103/MapEx_project/main/results/topo_r2/kth51_p01/MAPEX_TOPO_R2_KTH51_P01_37_DECISIONS_REASONING.html): 148 ảnh nhúng với cùng bố cục; đây là run đánh giá ban đầu P01. Run KTH51 P00 bổ sung có trong gói holdout7.
- [TOPO R2 — 10 run KTH và bằng chứng](../mapex_topo): 3 run phát triển có trong `eval_final/`, 7 run bổ sung có [báo cáo và dữ liệu từng tick](../mapex_topo/eval_holdout7/REPORT.md), decision logs, quỹ đạo, receipts, ảnh debug, SHA kiểm kê dữ liệu raw trên Dell.

Chỉ lưu kết quả tương ứng các nhánh hiện hành. Baseline/ablation cùng điều kiện được giữ để so sánh. Không lấy G2-R8 hoặc gộp kết quả G1 cũ.

RESULTS_MANIFEST.json ghi source path/hash của bộ lưu trữ G1/GTP và phép biến đổi CSV/RLE. Bộ TOPO R2 sử dụng `mapex_topo/SHA256SUMS`, các `RUN_MANIFEST.json` và `raw_inventory.json` độc lập. Audit gốc được giữ nguyên kể cả lỗi và giới hạn; không tạo số liệu mô phỏng mới. Các mask GTP RLE giữ nguyên giá trị logic, không giả vờ khớp byte NPZ.

Đây là gói số liệu và bằng chứng gọn, không sao lưu toàn bộ ảnh LiDAR/prediction mỗi tick hoặc checkpoint model.
