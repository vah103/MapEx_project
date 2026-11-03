# TOPO-002 — Đánh giá cuối (RD-047): MapEx | TOPO R1 | TOPO R2, 3 cấu hình × 1000 tick

Ngày 2026-11-03, phiên topo-planner. Receipt: `native/receipts/RD047_EVAL_FINAL.json`.

- Chạy tuần tự qua `native/run_final.py`, một worker, không retry. Hàng đợi topo2-data được tạm dừng bằng file PAUSE và dùng chung lock.
- Code đóng băng. Trước **mỗi** run, hash được so với manifest của smoke `TOPO2-50015847-P00-SMOKE-01`: 3/3 PASS. Các thành phần:
  - planner_r2 `77f24477…`
  - topo_policy `4269077b…`
  - TOPO script `445608a1…`
  - L1 v2 `7aba149a…` (preset v2, không truyền pfree)
  - pc_model `fc2c40f8…` + pc_frozen `fe1f75a0…` (`fallback_used` = False ở cả 3 run)
- Cả 3 run exit 0, đủ tick 0..999, 0 traceback, 0 lần đâm tường.

**Ghi chú bắt buộc:**
1. Ba cấu hình này (KTH49 P00, KTH50 P00, KTH51 P01) **đã được dùng để chẩn đoán R1**. Thiết kế R2 (đặc biệt viewpoint viền vùng unknown lớn) được gợi ý từ lỗi R1 trên KTH50, nên đây **không phải** tập test sạch.
2. **L1 v2 và p(c) trượt gate B trên DEV** (p(c): ECE CV 0,16 so với ngưỡng ≤ 0,05; L1 v2 không đạt door recall/precision mục tiêu — xem `topo002/eval_dev/REPORT.md`). Điều phối cho dùng tạm; p(c) chỉ dùng để xếp hạng gain.
3. n = 1 mỗi cấu hình. Số liệu mang tính mô tả, không kết luận thống kê.

Chỉ số dùng `v0_metrics.coverage_distance` (đóng băng, GT chỉ dùng offline); AUC = trung bình coverage tick 0..999, giống wave 1.
Script: `scripts/analyze_final.py` (dùng lại hàm của `topo001/eval_wave1/scripts/analyze.py`). Số thô: `figs/metrics.json`.

## 1. Kết quả (MapEx | R1 | R2)

| Cấu hình | AUC | t50 | t80 | t90 | cov@250 | cov@500 | cov@999 | quãng đường (m) |
|---|---|---|---|---|---|---|---|---|
| KTH49 P00 | 0,296 \| **0,452** \| 0,439 | – \| 629 \| **590** | – \| 949 \| **793** | – \| – \| – | 0,168 \| **0,256** \| 0,163 | 0,349 \| **0,448** \| 0,396 | 0,491 \| 0,807 \| **0,869** | 288 \| 290 \| 289 |
| KTH50 P00 | 0,328 \| 0,266 \| **0,470** | 843 \| – \| **451** | – \| – \| – | – \| – \| – | **0,242** \| 0,198 \| 0,240 | 0,318 \| 0,295 \| **0,603** | 0,690 \| 0,462 \| **0,757** | 284 \| 290 \| 286 |
| KTH51 P01 | 0,743 \| 0,722 \| **0,762** | 190 \| 229 \| **181** | **302** \| 364 \| 305 | 758 \| **611** \| 702 | 0,655 \| 0,515 \| **0,711** | 0,877 \| **0,891** \| 0,887 | 0,962 \| 0,964 \| **0,976** | 289 \| 290 \| 292 |

ΔAUC của R2 so với MapEx: **+0,143 / +0,142 / +0,019**. So với R1: −0,013 / +0,203 / +0,040.
R2 là nhánh duy nhất vượt MapEx ở cả 3 cấu hình. R1 thua MapEx ở 2/3.

Đường cong 3 nhánh trên cùng một hình: `figs/curve_KTH49_P00.png`, `curve_KTH50_P00.png`, `curve_KTH51_P01.png`.
Vạch dưới trục là các decision của R2: đậm = stop đầu nằm trên chuỗi hành lang, nhạt = loại khác, đỏ = fallback.

## 2. Hành vi R2

| | KTH49 | KTH50 | KTH51 |
|---|---|---|---|
| decision | 42 | 27 | 37 |
| stop đầu trên chuỗi hành lang (tổng) | 20/41 (49 %) | 2/26 (8 %) | 10/36 (28 %) |
| … theo phần ba thời gian [0,333) / [333,666) / [666,1000) | 6/14, 13/18, 1/9 | 0/6, 1/9, 1/11 | 6/19, 2/7, 2/10 |
| loại stop đầu | chain_corridor 20, rim 14, chain_hall 3, vp_chain 2, chain_room 1, door 1, fallback 1 | rim 11, chain_hall 7, frontier 3, chain_corridor 2, chain_room 1, vp_chain 1, door 1, fallback 1 | chain_room 12, chain_corridor 10, rim 7, frontier 4, door 1, vp_chain 1, vp_pocket 1, fallback 1 |
| replan trigger | targets_observed 15 | targets_observed 6, path_blowup 1 | targets_observed 19 |
| fallback MapEx | 1 (no_valid_goal, t990) | 1 (t960) | 1 (t973) |
| reselect | 0 | 0 | 0 |
| min-dist relax (≥ 3 m → 1,2 m) | 0 | 0 | 0 |
| HK prefix cải thiện được thứ tự | **0/40** | **0/25** | **0/34** |
| stale goal ticks | 296 | 272 | 66 |
| hook s/decision, trung vị (p90, max) | 2,09 (2,46, 4,24) | 1,98 (2,30, 4,47) | 1,10 (1,38, 4,20) |
| L1 v2 / planner, trung vị | 1,08 / 0,70 | 1,29 / 0,60 | 0,68 / 0,41 |
| RSS đỉnh (GiB) | 2,40 | 2,27 | 1,98 |
| thời gian chạy | 35,8 phút | 24,3 phút | 15,0 phút |

Max hook ≈ 4,2–4,5 s xảy ra ở decision đầu (compile numba). Gate 3 (≤ 6 s) đạt ở mọi decision.

**"Sườn trước" có tự nảy sinh không?** Một phần, và tùy bản đồ.
- **KTH49:** có, ở khúc giữa run (13/18 decision của phần ba thứ hai có stop đầu trên hành lang; t316–t630 gần như toàn chain_corridor). Phần cuối chuyển sang rim (đi về vùng unknown lớn).
- **KTH51:** có ở đầu run (6/19), phần sau chủ yếu là chain_room (túi).
- **KTH50:** không. Stop đầu chủ yếu là rim (11) và chain_hall (7). L1 v2 xếp phần lớn "trục" của KTH50 vào lớp hall, không phải corridor, nên chỉ số này có thể đánh giá thấp mức "đi sườn" thực.

Ảnh debug tiêu biểu (4 ảnh mỗi run) nằm trong `figs/`: `TOPO2-*_decision_*.png`. Toàn bộ ảnh ở `runs/TOPO2-*-EVAL-01/.../topo_debug/`.

## 3. Chỗ R2 KHÔNG làm đúng ý

1. **Held–Karp sắp lại prefix không có tác dụng: 0/99 decision cải thiện.** 2-opt/Or-opt đã cho thứ tự tối ưu trên ≤ 10 stop đầu ở mọi decision. Bước này chỉ tốn thời gian (nhỏ) mà không đổi kết quả.

2. **KTH49 đầu run chậm hơn R1** (cov@250 0,163 so với 0,256; R1 dẫn đến ~t600). Cụ thể:
   - t160: một chặng chain_room 102 tick, dự đoán g = 904, thực tế 12 083 ô.
   - t411→t457: 46 tick chỉ thu 125 ô.
   - R2 chỉ vượt nhờ đoạn quét dài t680–t820 (rim, mỗi chặng 25–29 k ô).
   - AUC thấp hơn R1 0,013 dù cov@999 cao hơn (0,869 so với 0,807). Sai thứ tự thời gian: phần lớn gain đến muộn.

3. **Gain dự đoán lệch lớn so với thực tế, theo cả hai hướng.**
   - Đánh giá thấp: stop đầu chỉ được tính gain cận biên tại viewpoint, trong khi robot còn thấy thêm trên đường đi. Ví dụ KTH49 t0 dự đoán 154, thực tế 15 369; KTH50 t64 dự đoán 384, thực tế 17 837.
   - Thổi phồng, đúng như lưu ý của điều phối về vùng unknown lớn với mean ≈ 0: KTH50 t852 dự đoán 14 082, thực tế 2 785; KTH49 t362 dự đoán 49 331, thực tế 23 182; KTH50 t554 dự đoán 4 970, thực tế 173.
   - p(c) hiệu chỉnh kém giữa các tòa nhà (trượt gate B), nên đây là lỗi đã biết trước.

4. **Cao nguyên ở KTH50 t510–t650** (coverage đứng ở ~0,62): các decision t529–t574 (chain_hall, frontier) có gain thực rất thấp (173–1 311 ô). Riêng t574 là một chặng 100 tick.

5. **Decision dày, goal gần:**
   - KTH49 t316–t362 và t479–t497: decision mỗi 8–13 tick.
   - KTH51 t96–t182: mỗi 6–15 tick.
   - Stale goal ticks cao (296 / 272): goal thường nằm giữa vùng đã biết để "nhìn xa".
   - Không tốn tick, nhưng mỗi decision tốn LaMa + L1 (wall time).

6. **Đuôi KTH51 (t394–t973):** các chặng chain_room dài thu ít (t576: 133 tick, 2 461 ô; t818: 73 tick, 390 ô). Coverage đã 0,89 nên ảnh hưởng tới AUC nhỏ, nhưng R2 không tự dừng đuổi các túi nhỏ ở xa.

7. **Fallback MapEx đúng 1 lần mỗi run, ở cuối** (t960–990, no_valid_goal): không còn ứng viên có gain ≥ 20 ô kỳ vọng. Đúng thiết kế.

## 4. Kết luận mô tả
R2 bỏ luật pha và tính gain trên toàn bộ vùng unknown (có viewpoint viền). Nhờ đó R2 sửa được lỗi chính của R1 trên KTH50 (AUC 0,266 → 0,470, vượt MapEx +0,142) và giữ được phần lớn lợi thế ở KTH49 (+0,143 so với MapEx).

Bằng chứng còn yếu: n = 1, chỉ 3 cấu hình, và cả 3 đã được dùng để chẩn đoán. Cần chạy trên TEST-IN/TEST-OUT (16 run theo §4 gate 4) trước khi có thể nói về khả năng tổng quát.
