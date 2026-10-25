# Kết quả đi kèm hai hướng — snapshot 25/10/2026

- [G1/G2 hiện hành](g1_g2): New Room 32 run, KTH audits, bảng số liệu, receipt, quỹ đạo và gallery P00.
- [MapEx-GTP](gtp): 36 run R7 hoàn tất, metrics/quỹ đạo/mask cuối; riêng R7AL là trạng thái pending và kiểm offline mới nhất.

Chỉ lưu kết quả tương ứng các nhánh hiện hành. Baseline/ablation cùng điều kiện được giữ để so sánh. Không lấy G2-R8 hoặc gộp kết quả G1 cũ.

RESULTS_MANIFEST.json ghi source path/hash của file gốc và phép biến đổi của các biểu diễn CSV/RLE. Audit gốc được giữ nguyên kể cả lỗi và giới hạn; không tạo số liệu mô phỏng mới. Các mask GTP RLE giữ nguyên giá trị logic, không giả vờ khớp byte NPZ.

Đây là gói số liệu và bằng chứng gọn, không sao lưu toàn bộ ảnh LiDAR/prediction mỗi tick hoặc checkpoint model.
