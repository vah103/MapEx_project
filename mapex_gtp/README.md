# MapEx-GTP R7 với tùy chọn AL — snapshot mới nhất

Nguồn: /home/dell/DELL_NATIVE2D/build/gtp_r7_al_mem_20261024/staged

Source R7 có solver AL sửa RAM; không tích hợp COMP-001 riêng.

- gt_planner.py: solver="r5" mặc định hoặc "al"; early_release=False.
- gt_al_solver.py: gain/cost insertion rồi local search, ngữ nghĩa AL giữ nguyên, lưu trữ giới hạn RAM.
- gt_feedback.py: event refresh, failure regions, feedback thực thi.
- gt_visibility.py: visibility tăng dần chính xác.
- tourlib.py: graph, raycast, greedy cover và open TSP.
- gt_planner_r5.py: module tham chiếu thuộc snapshot hiện tại, không phải snapshot lịch sử riêng.
- explore_mapex_gt_r1.py: vòng lặp native.
- dell_worker_NR.py / dell_worker_KTH50052751.py: template worker đã pin cho Dell.

solver="r5" chọn bộ giải cover/TSP gốc bên trong R7; không biến toàn bộ thuật toán thành R5. R7 vẫn có các cờ ER/FR'/J5/T3/VIS.

AL là candidate thử nghiệm, chưa được chấp nhận như cải thiện trên live LaMa. SPEC_R7.md mô tả nền R7, chưa mô tả phần AL.

Đọc ../docs/RESTORE.md. Guard gắn với source/model/runtime/manifest cụ thể; không tắt guard để chạy clone chưa stage lại.
