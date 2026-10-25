# Khôi phục môi trường và chạy

## Kiểm source

Chạy python3 tools/verify_snapshot.py từ repo root. Chỉ kiểm hash/cú pháp, không nạp model.

## Phụ thuộc ngoài repo

- MapEx upstream commit 53636bd1c79153acc3c74a532837d78c926bae5e, LaMa submodule và thư viện native.
- Native environment cũ: Python 3.6.13, NumPy 1.19.5, SciPy 1.5.3, OpenCV 4.5.3, Torch 1.10.2.
- pyastar2d, Numba, range_libc và các thư viện import trong source.
- dependencies/lama/conda_env.yml là cấu hình upstream tham khảo, không phải lockfile đầy đủ của runtime Dell hiện tại.
- Ba checkpoint lama_ensemble/train_1, train_2, train_3 cùng config khớp. Link tải upstream trong dependencies/mapex/UPSTREAM_README.md. Không lưu weights trong Git.
- Sidecar G1/G2 dùng Python >=3.10, NumPy/OpenCV/SciPy/scikit-image/Shapely theo import source. Sidecar và native worker là hai môi trường riêng.
- Đúng map, resolution, padding, vị trí xuất phát và manifest.

## G1/G2

dell_worker.py đọc DELL_NATIVE2D_ROOT, dùng /home/dell/MapEx, fixture New Room gốc và checkpoint gốc. MX_R41_PYTHON chọn Python sidecar.

Để tái hiện trên Dell: tạo thư mục run mới; sao chép mapex_g1_g2/src; stage đúng map/config/model cùng runner/guard theo manifest. Dùng Python native gọi dell_worker.py với DELL_NATIVE2D_ROOT trỏ thư mục mới sau khi kiểm thông số/tài nguyên. Không dùng NR-RAYEDR2-03 làm đích chạy.

## GTP

dell_worker_NR.py và dell_worker_KTH50052751.py là template pin theo Dell. Cần RUN_MANIFEST, đúng model/map, import pins và runtime_versions.

Trên host khác phải stage experiment mới, cập nhật đường dẫn và pin theo host, giữ guard, kiểm parity rồi mới thực thi. Manifest và authorization cũ không áp dụng tự động cho clone. Không dùng GLOBAL001_IMPORT_PROBE=1 để bỏ guard khi chạy khoa học.

Repo chưa có launcher portable hoàn chỉnh và chưa được kiểm native từ clean clone. Đường dẫn tuyệt đối được giữ nguyên để bảo toàn source; cần cấu hình/stage lại trước khi chạy.
