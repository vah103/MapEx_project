# MapEx gốc

Mã nguồn nguyên bản lấy từ Git trên Dell, commit 53636bd1c79153acc3c74a532837d78c926bae5e của [castacks/MapEx](https://github.com/castacks/MapEx).

## Nội dung

- src/scripts: planner/executor/mapper/raycast và các baseline upstream.
- src/configs/base.yaml: cấu hình gốc.
- src/range_libc: mã C++/CUDA/Cython, build scripts và vendor source.
- src/clusters: script chạy upstream.
- src/LICENSE và src/README.md: giấy phép Clear BSD và hướng dẫn upstream giữ nguyên.
- src/lama: Git submodule ở đúng commit b61dcb33e063fe9586b50e2dc7b70f97d5046c1d.

Source được lấy trực tiếp từ Git object của commit, không lấy thay đổi cục bộ chưa commit. Các file và executable modes giữ nguyên. Manifest ghi SHA256 và Git blob SHA.

Các file nhị phân upstream (bản đồ KTH, ảnh minh họa và PDF range_libc) không chép vào snapshot code này; danh sách và hash nằm trong SOURCE_MANIFEST.json. Checkpoint không lưu trong Git.

## Khôi phục phụ thuộc

Từ repo root:

    git submodule update --init --recursive

LaMa được khai báo tại .gitmodules ở repo root cho đường dẫn mapex_original/src/lama. File .gitmodules gốc trong src được giữ làm provenance.

Cài môi trường và checkpoint theo src/README.md. Nếu cần đúng các map/media bị lược khỏi snapshot code, lấy từ upstream ở cùng commit, ví dụ:

    git clone --no-checkout https://github.com/castacks/MapEx.git /tmp/mapex-upstream
    git -C /tmp/mapex-upstream checkout 53636bd1c79153acc3c74a532837d78c926bae5e -- kth_test_maps range_libc/maps range_libc/media range_libc/docs

Sau đó đặt các thư mục tương ứng vào src trước khi chạy. Không lấy map ở phiên bản khác để thay cho fixture đã ghim trong các thí nghiệm của dự án.

## Phân biệt với baseline thực nghiệm

Thư mục này giữ nguyên code upstream, bao gồm raycast chưa áp dụng bản sửa paper-cumulative và cơ chế dự đoán gốc. Baseline MapEx trong các bộ kết quả G1/G2 đã dùng native stage với ba LaMa và bản sửa raycast. Vì vậy không mặc định coi chạy source gốc này sẽ tái hiện đúng số liệu baseline đó.

Xem dữ liệu đối chứng đã lưu ở ../results/g1_g2; provenance từng run xác định chính xác source thực thi. Các baseline R5 trong ../results/gtp là MapEx-GTP R5, không phải MapEx gốc.

Không sửa code nguồn trên Dell và không chạy thí nghiệm trong thao tác lưu thư mục này.
