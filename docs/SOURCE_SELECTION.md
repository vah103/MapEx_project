# Lựa chọn nguồn — 25/10/2026

Yêu cầu: lưu bản mới nhất của hai hướng, không lấy G2-R8.

1. G1/G2: G1-001 R2 + SWEEP-R1 + EARLYDOOR-R1 tại NR-RAYEDR2-03. Khác các thử nghiệm trên đối chứng cũ NR-RN-04.
2. GTP: gtp_r7_al_mem_20261024/staged, R7 có AL sửa RAM. Đối chiếu byte với source trong stage thực thi J0 ngày 25/10; kết quả trong ARCHIVE_VERIFICATION.json.
3. Tên file gốc giữ nguyên để bảo toàn import/provenance.
4. Source sao chép giữ nguyên byte. README và verifier là phần mới của gói lưu trữ.
5. Không đưa snapshot cũ, candidate G2-R5/R6/R7/R8/R9, checkpoint, dataset, gallery, run data hoặc điều phối liên tác nhân lên repo.

Không chạy native/model hoặc thay đổi thuật toán trong thao tác lưu này.
