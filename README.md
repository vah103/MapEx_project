# MapEx project

Mã MapEx gốc và các hướng nghiên cứu trên nền MapEx. Các snapshot ban đầu lưu ngày 25/10/2026; TOPO-002 R2 bổ sung ngày 03/11/2026.

| Thư mục | Bản được lưu | Vai trò |
|---|---|---|
| [mapex_original](mapex_original) | MapEx upstream, commit 53636bd | Mã gốc, cấu hình và source range_libc; LaMa submodule được ghim |
| [mapex_g1_g2](mapex_g1_g2) | G1-001 R2 + G2 SWEEP-R1 + EARLYDOOR-R1 | Nhận diện phòng và tổ chức khám phá theo phòng |
| [mapex_gtp](mapex_gtp) | MapEx-GTP R7 với solver AL, bản sửa RAM mới nhất | Chọn viewpoint và lập tour toàn cục |
| [mapex_topo](mapex_topo) | TOPO-002 Planner R2 đã đóng băng (03/11/2026) | RoomNode L1 v2, p(c), lập kế hoạch expected gain và tích hợp native |

Mỗi hướng chỉ lưu một snapshot source hiện tại. Không chứa G2-R8 hoặc lịch sử các bản thử nghiệm. TOPO R2 có manifest/checksum riêng trong `mapex_topo/`; xem [trạng thái TOPO R2 mới nhất](mapex_topo/STATUS_LATEST.md) (giữ nguyên README niêm phong trước đợt holdout).

## Trạng thái

- G1 hiện hành là G1-001 R2. SWEEP/EARLYDOOR được tích hợp trên R2, nhưng chưa chứng minh giảm tổng quãng đường so với G1 R2 trên bộ New Room 8 vị trí.
- GTP có solver="r5" mặc định và solver="al" thử nghiệm; early_release=False. Bản AL đã sửa lưu trữ để giảm RAM, chưa có kết luận thắng R7 trên live LaMa.
- TOPO R2 đã có 10 run KTH × 1.000 tick: [3 cấu hình phát triển](mapex_topo/eval_final/REPORT.md) và [7 cấu hình bổ sung](mapex_topo/eval_holdout7/REPORT.md) theo R2 đóng băng. Kết quả cần được diễn giải với giới hạn đánh giá: ba cấu hình ban đầu không phải hold-out và p(c)/L1 v2 trượt kiểm tra DEV; không mặc định khẳng định tổng quát hóa.
- Đây là source native 2D, chưa phải bản ROS2/Gazebo được tích hợp và kiểm chứng.

## Bắt đầu

Clone repo này, rồi kiểm tra từ thư mục gốc:

```bash
python3 tools/verify_snapshot.py
(cd mapex_topo && sha256sum -c SHA256SUMS)
python3 tools/verify_topo_results.py
```

Đọc [khôi phục môi trường](docs/RESTORE.md), [lựa chọn nguồn](docs/SOURCE_SELECTION.md), rồi README từng hướng.

Đây là snapshot bảo toàn code, không phải gói cài độc lập. Worker và import guard vẫn dùng đường dẫn Dell; cần đúng manifest, map, ba checkpoint LaMa và thư viện native. Không chạy vào thư mục kết quả cũ.

## Nội dung

- mapex_g1_g2/src: Room Node, LOCAL/GLOBAL, SWEEP, EARLYDOOR, vòng lặp native.
- mapex_gtp/src: planner R7, AL, feedback, visibility, logger và worker mẫu.
- mapex_topo: p(c), L1 v2, planner R2, native hook, spec, test, 3 manifest đánh giá ban đầu, 7 gói holdout (quỹ đạo, quyết định, coverage theo tick, receipt, checksum và ảnh debug), báo cáo/đồ thị và [ba bộ ảnh cuộn HTML](mapex_topo/galleries/README.md).
- dependencies: source hỗ trợ MapEx, cấu hình môi trường/model và giấy phép.
- SOURCE_MANIFEST.json: đường dẫn nguồn Dell và SHA256 từng file sao chép.
- ARCHIVE_VERIFICATION.json: kết quả kiểm snapshot, bao gồm đối chiếu source GTP với stage J0 mới nhất.
- tools/verify_snapshot.py: kiểm hash và cú pháp, không nạp model hoặc mô phỏng.

Checkpoint, map môi trường nhị phân và cấu hình tài khoản không nằm trong repo. Kết quả và bằng chứng chọn lọc nằm trong [results/](results/README.md).

## Upstream

Dựa trên [castacks/MapEx](https://github.com/castacks/MapEx), commit Dell 53636bd1c79153acc3c74a532837d78c926bae5e. Giữ giấy phép Clear BSD tại dependencies/mapex/LICENSE và Apache-2.0 cho cấu hình LaMa tại dependencies/lama/LICENSE. Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Kết quả đi kèm

Xem [results/](results/README.md): G1/G2 New Room/KTH và 36 run GTP R7. [Ba bộ ảnh cuộn TOPO R2](mapex_topo/galleries/README.md) cùng dữ liệu khoa học của 10 run nằm trực tiếp trong [mapex_topo](mapex_topo) (3 development + 7 bổ sung). R7AL chưa có kết quả native; không dùng số R7 thay cho AL.
