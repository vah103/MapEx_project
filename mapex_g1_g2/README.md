# MapEx + G1/G2 — snapshot hiện tại

Nguồn: /home/dell/DELL_NATIVE2D/runs/NR-RAYEDR2-03

G1 = G1-001 R2, Room Node V4-R4.1 và sửa frontier scope theo UNKNOWN liên quan trong phòng.
G2 = SWEEP-R1 + EARLYDOOR-R1 tích hợp trên G1 R2.

- room_native_integration.py: LOCAL/GLOBAL và tích hợp G2.
- g2_room_sweep_r1.py: chọn frontier theo trục PCA, đổi chiều dọn phần còn lại có giới hạn.
- g2_early_global_door_r1.py: ứng viên GLOBAL theo cửa thoát.
- v41src: source Room Node dùng bởi sidecar.
- dell_worker.py: New Room P00, 3 checkpoint, paper-cumulative raycast, đường dẫn Dell.

Đây là snapshot tích hợp hiện hành; không khẳng định G2 vượt G1 trên nhiều vị trí/bản đồ. Không chứa G2-R8.

Lỗi còn mở G1-005: LOCAL A* grid cũ trong quá trình di chuyển từng gây hit_wall ở KTH. Snapshot giữ nguyên source R2 đã lưu; không âm thầm sửa thuật toán.

Đọc ../docs/RESTORE.md trước khi chạy. Không chạy lại vào thư mục run cũ.
