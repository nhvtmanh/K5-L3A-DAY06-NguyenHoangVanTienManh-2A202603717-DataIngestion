# Hướng dẫn setup, chạy và dữ liệu

Bài làm dùng PostgreSQL + pgvector trong Docker và EmbeddingGemma 2 chạy local bằng CPU. Các lệnh dưới đây dùng **Windows PowerShell, chạy từ thư mục gốc repo**, không phải từ `report/individual/`.

## 1. Setup môi trường

Cài Python và Docker Desktop, mở Docker Desktop và chờ Engine hoạt động ở chế độ Linux containers. Môi trường đã chạy bài dùng Python **3.14.8**; không cần cài PostgreSQL riêng trên Windows hoặc API key cho model local.

Tạo môi trường Python nếu chưa có `.venv`:

```powershell
py -3.14 -m venv starter/.venv
```

Cài thư viện cơ bản và thư viện chạy model:

```powershell
.\starter\.venv\Scripts\python.exe -m pip install -r starter/requirements.txt
.\starter\.venv\Scripts\python.exe -m pip install -r starter/requirements-local.txt
```

Tạo `.env` từ cấu hình mẫu nếu file chưa tồn tại:

```powershell
if (-not (Test-Path starter/.env)) {
    Copy-Item starter/.env.example starter/.env
}
```

Giữ `DATABASE_URL` khớp cấu hình Docker local trong `.env.example`. Không commit `.env`. Các lệnh dùng trực tiếp Python trong `.venv` nên không cần kích hoạt môi trường bằng `Activate.ps1`.

Khởi động database và kiểm tra kết nối:

```powershell
docker compose -p day6_student_starter -f starter/docker-compose.yml up -d --wait
.\starter\.venv\Scripts\python.exe starter/scripts/check_setup.py
```

Kết quả mong đợi có `Setup OK`. PostgreSQL được truy cập qua `127.0.0.1:55433`; dữ liệu được giữ trong Docker volume `pg_data`.

## 2. Chạy pipeline

Trước khi chạy, kiểm tra PDF đã nằm đúng đường dẫn ở phần 3, rồi xác minh benchmark:

```powershell
.\starter\.venv\Scripts\python.exe benchmark/validate.py
```

Chạy dev lần đầu, cho phép tải model nếu cache chưa đầy đủ:

```powershell
.\starter\.venv\Scripts\python.exe -m starter.src.run --split dev --allow-download
```

Sau khi model đã được tải, có thể chạy bằng cache local:

```powershell
.\starter\.venv\Scripts\python.exe -m starter.src.run --split dev
```

Pipeline đọc 30 trang theo manifest, chia chunk A/B, tạo vector 768 chiều, lưu pgvector và tìm top-3 cho 15 câu dev. C dùng lại chunk/vector của B và thêm filter khi câu hỏi có yêu cầu. Model local không gọi API trả phí; chi phí điện/phần cứng chưa được tính.

Mỗi lần chạy tạo thư mục `starter/results/dev_<thoi_gian_UTC>_<id>/`. Xem kết quả dev và chốt cấu hình trước khi chạy test. Ví dụ dùng cấu hình baseline đã lưu trong repo:

```powershell
.\starter\.venv\Scripts\python.exe -m starter.src.run --split test --config starter/results/dev_20261113T141158Z_373575/run_config.json
```

Nếu muốn dùng một lần dev mới, thay đường dẫn sau `--config` bằng `run_config.json` của lần dev đã chọn. Nếu cache thiếu file model, thêm `--allow-download`. Test chạy đủ 9 câu với cấu hình đã chốt; chương trình kiểm tra hash manifest/benchmark/code, nên nếu đã sửa các file đó cần chạy dev lại trước. Không chỉnh cấu hình dựa trên test rồi coi lần chạy lại là đánh giá độc lập.

Kết quả mỗi lần chạy gồm:

| File                                  | Nội dung                                                    |
| ------------------------------------- | ----------------------------------------------------------- |
| `traces.jsonl`                        | Top-3, text, score, metadata, gold và nhãn hit tự động      |
| `summary.json`, `metrics.csv`         | Hit@1, Evidence Hit@3 và các số đếm                         |
| `resources.json`                      | Số chunk, ký tự/token, thời gian, cache và kiểm tra nạp lại |
| `run_config.json`, `environment.json` | Cấu hình, commit model và phiên bản thư viện                |
| `manual_coherence.json`               | Mẫu chấm coherence tại cùng 10 vị trí nguồn                 |
| `manual_retrieval_review.json`        | Mẫu chấm grounding và kiểm tra metadata                     |
| `run.log`                             | Tiến độ chạy; nếu lỗi có thêm `error.json`                  |

Vector được cache ở `starter/.embedding_cache/` để dùng lại khi nội dung và cấu hình model không đổi. Cache không được đưa lên Git. Nhận xét/điểm AI trong các artifact đánh giá hiện có là tạm thời, chưa phải nhãn do người học hoặc người chấm độc lập xác nhận.

Dừng database nhưng giữ dữ liệu:

```powershell
docker compose -p day6_student_starter -f starter/docker-compose.yml stop
```

## 3. Đường dẫn dữ liệu và bài làm

Các đường dẫn trong cột đầu được tính từ gốc repo:

| Đường dẫn                                                                                            | Nội dung                                                           |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [starter/data/IntroductionToBusiness.pdf](../../starter/data/IntroductionToBusiness.pdf)             | PDF gốc _Introduction to Business_, OpenStax/Rice University, 2018 |
| [starter/data/manifest.json](../../starter/data/manifest.json)                                       | Nguồn, giấy phép, SHA-256, phạm vi trang và số ký tự               |
| [benchmark/questions.json](../../benchmark/questions.json)                                           | 24 câu hỏi: 15 dev / 9 test, gold và filter                        |
| [benchmark/evidence_validation.json](../../benchmark/evidence_validation.json)                       | Kiểm chứng evidence tồn tại tại đúng trang nguồn                   |
| [starter/src/](../../starter/src/)                                                                   | Mã ingestion, chunking, embedding, store, evaluation và CLI        |
| [starter/results/dev_20261113T141158Z_373575/](../../starter/results/dev_20261113T141158Z_373575/)   | Kết quả dev baseline đã chạy                                       |
| [starter/results/test_20261113T144121Z_55e021/](../../starter/results/test_20261113T144121Z_55e021/) | Kết quả test đã chạy với cấu hình cố định                          |
| [artifacts/experiment_log.csv](../../artifacts/experiment_log.csv)                                   | Nhật ký cấu hình và chỉ số A/B/C                                   |
| [report/individual/2A202603717_NguyenHoangVanTienManh.md](2A202603717_NguyenHoangVanTienManh.md)     | Báo cáo cá nhân                                                    |
| [STUDENT.md](../../STUDENT.md)                                                                       | Thông tin học viên và repo bài nộp                                 |

Corpus chỉ dùng **trang PDF 28–57**, tương ứng **trang in 16–45**, không embedding toàn bộ 744 trang. Giữ nguyên PDF gốc để số trang và SHA-256 khớp manifest; không thay bằng PDF đã tách 30 trang mà chưa cập nhật dữ liệu liên quan.

Nếu bản clone chưa có PDF, tải từ [nguồn OpenStax ghi trong manifest](https://assets.openstax.org/oscms-prodcms/media/documents/IntroductionToBusiness-OP_8D04gAa.pdf) và lưu đúng tên `starter/data/IntroductionToBusiness.pdf`, sau đó chạy lại `benchmark/validate.py`.
