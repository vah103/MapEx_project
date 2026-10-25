# MapEx project

Mã MapEx gốc và hai hướng nghiên cứu trên nền MapEx, lưu từ Dell ngày 25/10/2026.

| Thư mục | Bản được lưu | Vai trò |
|---|---|---|
| [mapex_original](mapex_original) | MapEx upstream, commit 53636bd | Mã gốc, cấu hình và source range_libc; LaMa submodule được ghim |
| [mapex_g1_g2](mapex_g1_g2) | G1-001 R2 + G2 SWEEP-R1 + EARLYDOOR-R1 | Nhận diện phòng và tổ chức khám phá theo phòng |
| [mapex_gtp](mapex_gtp) | MapEx-GTP R7 với solver AL, bản sửa RAM mới nhất | Chọn viewpoint và lập tour toàn cục |

Mỗi hướng chỉ lưu một snapshot source hiện tại. Không chứa G2-R8 hoặc lịch sử các bản thử nghiệm.

## Trạng thái

- G1 hiện hành là G1-001 R2. SWEEP/EARLYDOOR được tích hợp trên R2, nhưng chưa chứng minh giảm tổng quãng đường so với G1 R2 trên bộ New Room 8 vị trí.
- GTP có solver="r5" mặc định và solver="al" thử nghiệm; early_release=False. Bản AL đã sửa lưu trữ để giảm RAM, chưa có kết luận thắng R7 trên live LaMa.
- Đây là source native 2D, chưa phải bản ROS2/Gazebo được tích hợp và kiểm chứng.

## Bắt đầu

Clone repo này, rồi chạy: python3 tools/verify_snapshot.py

Đọc [khôi phục môi trường](docs/RESTORE.md), [lựa chọn nguồn](docs/SOURCE_SELECTION.md), rồi README từng hướng.

Đây là snapshot bảo toàn code, không phải gói cài độc lập. Worker và import guard vẫn dùng đường dẫn Dell; cần đúng manifest, map, ba checkpoint LaMa và thư viện native. Không chạy vào thư mục kết quả cũ.

## Nội dung

- mapex_g1_g2/src: Room Node, LOCAL/GLOBAL, SWEEP, EARLYDOOR, vòng lặp native.
- mapex_gtp/src: planner R7, AL, feedback, visibility, logger và worker mẫu.
- dependencies: source hỗ trợ MapEx, cấu hình môi trường/model và giấy phép.
- SOURCE_MANIFEST.json: đường dẫn nguồn Dell và SHA256 từng file sao chép.
- ARCHIVE_VERIFICATION.json: kết quả kiểm snapshot, bao gồm đối chiếu source GTP với stage J0 mới nhất.
- tools/verify_snapshot.py: kiểm hash và cú pháp, không nạp model hoặc mô phỏng.

Checkpoint, map môi trường nhị phân và cấu hình tài khoản không nằm trong repo. Kết quả và bằng chứng chọn lọc nằm trong [results/](results/README.md).

## Upstream

Dựa trên [castacks/MapEx](https://github.com/castacks/MapEx), commit Dell 53636bd1c79153acc3c74a532837d78c926bae5e. Giữ giấy phép Clear BSD tại dependencies/mapex/LICENSE và Apache-2.0 cho cấu hình LaMa tại dependencies/lama/LICENSE. Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Kết quả đi kèm

Xem [results/](results/README.md): G1/G2 New Room/KTH, 36 run GTP R7 và trạng thái/kiểm offline R7AL mới nhất. R7AL chưa có kết quả native; không dùng số R7 thay cho AL.
