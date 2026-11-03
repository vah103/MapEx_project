# MapEx + TOPO — TOPO-002 Planner R2 (frozen snapshot, 03/11/2026)

Bản lưu **mã đã chạy thực nghiệm**, giữ nguyên byte so với nguồn trên Dell, không phải một package tự cài độc lập. Các nhánh `mapex_original/`, `mapex_g1_g2/` và `mapex_gtp/` của repository không bị thay đổi.

## Các thành phần

| Đường dẫn | Vai trò |
|---|---|
| `pc_model/pc_model.py`, `pc_frozen.json` | p(c) ước lượng xác suất ô UNKNOWN là free, dùng làm trọng số expected gain |
| `topo_l1/topo_l1.py` | RoomNode / TOPO L1 v2: vùng, chuỗi hành lang/phòng, cửa, viewpoint |
| `planner/topo_planner/planner_r2.py` và các module cùng thư mục | Planner R2: chọn điểm nhìn theo lợi ích khám phá kỳ vọng có xét thời điểm thu được |
| `native/topo_policy.py` | Hook chọn goal, kiểm tra goal/đường native và fallback MapEx |
| `native/explore_mapex_topo_r1.py` | Vòng lặp MapEx native đã tích hợp TOPO (tên file cũ được giữ vì checksum) |
| `native/{prepare_run_topo,launch_topo,integrate_topo,run_final}.py` | Chuẩn bị run, tích hợp, chạy có receipt; nhiều đường dẫn vẫn phụ thuộc Dell |
| `pc_model/test_pc_model.py`, `topo_l1/test_topo_l1.py`, `planner/topo_planner/tests/`, `native/tests/` | Kiểm thử thuật toán và tích hợp |
| `docs/TOPO-002.md`, `docs/DESIGN_NOTES_R2.md` | Spec và giải thích thiết kế/giới hạn |
| `eval_final/` | Báo cáo, số liệu JSON, đồ thị 3 cấu hình và mã phân tích |
| `run_manifests/` | Manifest của 3 run R2 được lưu, gồm hash, cấu hình, baseline |
| `SHA256SUMS` | Hash toàn bộ file trong snapshot này; kiểm bằng `sha256sum -c` |

Lưu ý: `planner/topo_planner/planner.py` là phiên bản R1 tham chiếu; **R2 thực thi `UnifiedPlanner` trong `planner_r2.py`**.

## Pipeline

1. MapEx/LaMa cập nhật occupancy, mean và variance từ quan sát.
2. p(c) tính trọng số free cho các ô chưa biết; L1 v2 suy luận vùng/cửa/chuỗi và viewpoint **không nhận p(c)** ở cấu hình đã chạy (`l1_pfree=false`).
3. Planner R2 xây ứng viên gồm chain, door, viewpoint, **viền vùng UNKNOWN lớn**, frontier; tối ưu gần đúng `J = Σ gᵢ·(T − tᵢ)` với gain kỳ vọng, ngân sách `T=1000` tick.
4. Native policy kiểm reachability và A* trước khi chọn goal. Nếu không có goal hợp lệ/lỗi thì fallback MapEx. Mỗi decision cập nhật lại kế hoạch (receding horizon).

Những xấp xỉ được ghi rõ trong `docs/DESIGN_NOTES_R2.md`: BFS để ước lượng thời gian, lazy-greedy không bảo đảm tính đơn điệu, Held–Karp chỉ thử hoán vị ≤10 điểm đầu. Cấu hình đóng băng: L1 v2, `pc_frozen.json`, `TOPO_PC=pc_model`, `TOPO_L1_PFREE=0`, `TOPO_PLANNER_T=1000`. Phần online **không đọc ground truth**; GT chỉ dùng offline để tính coverage.

## Bằng chứng hiện có, không phải kết luận tổng quát

Ba cặp đánh giá MapEx vs TOPO R2 trên KTH49 P00, KTH50 P00, KTH51 P01 (mỗi loại n=1) đều hoàn tất 1000 tick. Theo `eval_final/REPORT.md`, R2 có AUC lần lượt **0,439 / 0,470 / 0,762**, MapEx **0,296 / 0,328 / 0,743**. Cùng 3 cấu hình này từng được dùng để phân tích lỗi R1, do đó **không phải hold-out**. L1 v2 và p(c) còn không đạt gate B trên DEV; chưa có chứng cứ đủ để khẳng định tổng quát hóa. Báo cáo và số liệu đầy đủ trong `eval_final/`.

Bảy cấu hình KTH khác mới được **đề xuất chạy** sau khi đóng băng R2, chưa có kết quả trong snapshot này. Không được gộp dữ liệu từ các phiên bản khác làm kết quả R2.

## Kiểm tra và tái lập

- Tại thư mục `mapex_topo/`, chạy `sha256sum -c SHA256SUMS` để xác nhận file lưu trữ.
- Đối chiếu hash 6 file quan trọng trong `run_manifests/KTH49_P00.json` (planner R2, policy, native script, L1 v2, p(c) code và trọng số), đồng thời đối chiếu `run_manifests/KTH50_P00.json`, `KTH51_P01.json`.
- Môi trường đầy đủ cần MapEx native 2D, `pyastar2d`, `numpy`, `numba`, `scipy`, `shapely`, `opencv`, ba checkpoint LaMa, bản đồ v3, baseline/worker khớp manifest, cùng các thư viện tại `dependencies/` của repo. Các script native có tham chiếu đường dẫn Dell; cần khôi phục môi trường trước khi chạy, **không chạy thẳng vào thư mục kết quả cũ**.
- Không lưu checkpoint, bản đồ nhị phân, dữ liệu hàng nghìn tick hay thông tin tài khoản vào GitHub. Chưa thực hiện chạy thí nghiệm mới trong thao tác đóng gói.

Nguồn gốc: `/home/dell/DELL_NATIVE2D/build/topo002/`, spec `/home/dell/mapex_workspace/tasks/TOPO-002.md`. Dữ liệu theo biên nhận RD-047, ngày 03/11/2026. Xem giấy phép và nguồn upstream trong [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
