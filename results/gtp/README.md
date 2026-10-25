# Kết quả MapEx-GTP mới nhất đang có trên Dell

## R7 đã chạy — kết quả nền cho code R7/AL

r7/ có 36 run B hoàn tất và sync-verified: New Room 8 start x J0/JR/JC = 24; KTH51-v3 4 start x J0/JR/JC = 12.
Mỗi run có metrics gốc, manifest, execution/sync receipts, tham chiếu R5, quỹ đạo và final observed mask mã hóa RLE có kiểm khôi phục.

summary.csv dùng canonical native odometry gồm đoạn cuối postscan cho cả R7 và đối chứng R5. Cache evaluator cũ của R5 chỉ có saved odom nên được giữ riêng, không so trực tiếp với tổng đã gồm đoạn cuối. Các tổng R5/R7 đều được đối chiếu với summary native tới 1e-6 m. Quỹ đạo CSV giữ toàn bộ odom đã lưu và thêm hàng final_postscan được gắn nhãn. Các số tổng được đối chiếu lại tới 1e-6 m.

Đây là arm R7, solver nền r5 bên trong R7. Không gọi chúng là kết quả R7AL. Bộ này không phải báo cáo adoption đầy đủ A/B/C của cohort R7; không suy ra kết luận J5 từ dữ liệu B. R5 metrics đi kèm chỉ phục vụ đối chứng. DESCRIPTIVE_SUMMARY.json chỉ thống kê mô tả.

## R7AL J0 mới nhất — chưa có kết quả native

r7al_j0/ lưu PENDING_EVALUATION và snapshot trạng thái trên Dell: 12 case J0 chưa có receipt locally; saved state WAITING_COM1_ONLINE. Không kiểm lại COM1 trong thao tác lưu này. Kết quả tương lai thuộc unreviewed (RD-041), không tự thay đổi status thành accepted.

r7al_offline/ là kết quả kiểm source/parity/RAM trên 5 trạng thái cố định. Reserve 3072 MiB là RAM cấp phát giả lập, không phải LaMa thật; không chứng minh quãng đường hoặc peak RAM native.

Mọi kết quả là snapshot ngày 25/10/2026, không tự đồng bộ với lần chạy tương lai.
