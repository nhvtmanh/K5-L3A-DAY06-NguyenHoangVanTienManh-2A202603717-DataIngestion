# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline

- **Chủ đề:** Hệ thống kinh tế và môi trường kinh doanh.
- **Nguồn/quyền:** Sách _Introduction to Business_, OpenStax/Rice University, 2018.
- **Corpus:** **30 trang PDF 28–57**, tương ứng trang in **16–45**.
- **Quy tắc tính:** Chỉ tính các trang có nội dung hữu ích như giảng giải, ví dụ, bảng/hình kèm chữ; không tính bìa, mục lục, trang trống hoặc trang chỉ có tài liệu tham khảo. Nguồn và danh sách trang được ghi trong `starter/data/manifest.json`.
- **Chuẩn hóa:** Trích xuất text từng trang bằng `pypdf`, giữ số trang gốc; chuẩn hóa Unicode NFC, xuống dòng và khoảng trắng, giữ ranh giới đoạn. Tổng **79.640 ký tự** sau chuẩn hóa.
- **Xử lý lỗi:** Cảnh báo trang không có text; báo lỗi khi trích xuất thất bại. Không OCR hoặc tự sửa nội dung.
- **Giới hạn:** Bảng bị trải phẳng, thứ tự text có thể khác bố cục PDF, nội dung bị ngắt qua trang; heuristic tách câu có thể nhầm dấu chấm trong viết tắt như `U.S.`.
- **Embedding:** `google/embeddinggemma-2` của Google, chạy local bằng CPU `float32`, batch size 8, vector **768 chiều**.
- **Vector store:** PostgreSQL **16.16** + pgvector **0.8.7**, chạy bằng Docker và lưu dữ liệu bền vững qua volume.
- **Metric:** tìm kiếm **exact cosine**, điểm tương đồng `score = 1 - cosine_distance`; chưa dùng HNSW/IVFFlat.
- **Lệnh chạy** từ thư mục `starter`, trong môi trường `.venv`:

    ```powershell
    docker compose -p day6_student_starter up -d --wait
    ```

- **Chi phí:** **0 USD**.
- **Metadata bắt buộc:** `document_id` từ manifest, `source` là URL nguồn và `page` là số trang PDF gốc.
- **Trường lọc bổ sung:** `section` dạng chuỗi, lấy từ heading đánh số trong tài liệu, ví dụ `"1.5"`.
- **Lý do chọn:** Nguồn/trang giúp truy vết bằng chứng; `section` giới hạn tìm kiếm theo phạm vi câu hỏi yêu cầu.

## Cấu hình và benchmark

- A / B / C (tham số cụ thể, khác biệt duy nhất B–C): **A** chia cửa sổ cố định 500 ký tự, overlap 50; **B** nhóm theo đoạn/ranh giới câu, tối đa 500 ký tự, không overlap; **C** dùng cùng chunk, vector và dữ liệu lưu của B, chỉ thêm filter metadata khi câu hỏi cung cấp. Câu không có filter thì C chạy như B. Các chunk không vượt trang hoặc ranh giới mục đã nhận diện. A–B thay cách chia, bao gồm chính sách overlap; B–C chỉ thay lọc. Cả ba giữ nguyên corpus, model/commit, cosine metric và benchmark.
- ≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence: Bộ [questions.json](../../benchmark/questions.json) gồm **24 câu, chia trước 15 dev / 9 test**, có câu trực tiếp, diễn đạt lại và điều kiện/ngoại lệ. Có **6 câu filter**: dev q02, q06, q11, q14; test q16, q20. Filter section lấy từ phạm vi mục được nêu ngay trong câu hỏi, không lấy từ gold answer hoặc gold page. Mỗi câu lưu đáp án, nguồn, trang PDF gốc và evidence; q07 có hai nguồn bằng chứng hợp lệ tại PDF 38 và 34. Script [validate.py](../../benchmark/validate.py) kiểm chứng đủ 25 đoạn evidence chính/thay thế tồn tại ở đúng trang sau chuẩn hóa. Các tập được chia theo chủ đề con; do cùng sách/chương, một số câu dev/test vẫn dùng chung trang hoặc thuật ngữ, chưa độc lập theo tài liệu. Không đưa gold vào embedding query/filter. Các nhãn và câu hỏi được giữ nguyên giữa A/B/C.
- Tiêu chí chọn chiến lược đã chốt trước khi xem test: Ưu tiên **dev Evidence Hit@3**, sau đó **dev Hit@1**, rồi số chunk nếu vẫn hòa.

## Kết quả

| Strategy | Split | Hit@1 (n/N)    | Evidence Hit@3 (n/N) | Chunk count | Thời gian                           | Chi phí   |
| -------- | ----- | -------------- | -------------------- | ----------- | ----------------------------------- | --------- |
| A        | dev   | 5/15 (33,33%)  | 12/15 (80%)          | 192         | Ingest 45,393 giây, tạo vector mới  | 0 USD API |
| B        | dev   | 10/15 (66,67%) | 12/15 (80%)          | 220         | Ingest 45,527 giây, tạo vector mới  | 0 USD API |
| C        | dev   | 11/15 (73,33%) | 12/15 (80%)          | 220, dùng B | Tái sử dụng ingest của B            | 0 USD API |
| A        | test  | 4/9 (44,44%)   | 5/9 (55,56%)         | 192         | Ingest 0,855 giây, đọc cache vector | 0 USD API |
| B        | test  | 7/9 (77,78%)   | 7/9 (77,78%)         | 220         | Ingest 0,868 giây, đọc cache vector | 0 USD API |
| C        | test  | 7/9 (77,78%)   | 7/9 (77,78%)         | 220, dùng B | Tái sử dụng ingest của B            | 0 USD API |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
| -------- | --------------------------- | ------------------------- | ------------------------------ | ----------------------------------------- |
| A        | 0,90, n=10                  | 24/27                     | Dev 1/4; test 2/2              | 2/0/7 câu; 1,56/2                         |
| B        | 1,30, n=10                  | 27/27                     | Dev 1/4; test 1/2              | 1/1/7 câu; 1,67/2                         |
| C        | 1,30, n=10                  | 27/27                     | Dev 4/4; test 2/2              | 1/1/7 câu; 1,67/2                         |

Đường dẫn trace, nhãn thủ công và cách chọn mẫu: [Dev traces](../../starter/results/dev_20261113T141158Z_373575/traces.jsonl) có 45 bản ghi (15 câu × 3 cấu hình); [test traces](../../starter/results/test_20261113T144121Z_55e021/traces.jsonl) có 27 bản ghi (9 câu × 3 cấu hình), tổng 81 hit top-3. Mỗi bản ghi giữ text, ID, score, nguồn/trang, metadata, gold và filter.

## Phân tích và kết luận

- So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn: Trên test, B cải thiện Hit@1 từ 4/9 lên 7/9 và Hit@3 từ 5/9 lên 7/9 so với A. B/C cùng đạt 7/9, nhưng C tuân thủ filter tốt hơn. Ví dụ: dev A/q02, PDF 29 của Introduction to Business, chunk bị cắt nên thiếu danh sách bảy yếu tố; test C/q20, PDF 57, kết quả đúng mục nhưng thiếu đủ hai đặc điểm độc quyền nhóm.
- Chiến lược chọn, đánh đổi, điều kiện áp dụng: Giữ C theo quyết định dựa trên dev trước khi chạy test. C dùng lại vector của B và phù hợp khi câu hỏi cung cấp phạm vi section rõ ràng.
- Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí: Corpus chỉ có 30 trang từ một sách, test 9 câu và chấm thủ công chưa hoàn tất. Coherence, grounding và tỷ lệ metadata trong bảng thứ hai là đánh giá AI tạm thời, chưa có xác nhận người học hoặc chấm độc lập; metadata mới được đối chiếu text, chưa kiểm tra bố cục PDF trực quan. Khớp nguyên quote có thể bỏ sót bằng chứng tương đương. Thời gian đo một lần, test dùng cache; chi phí API 0 USD chưa tính điện/phần cứng.
- Nếu làm lại hoặc mở rộng, cải tiến gì? Cải thiện tách câu tại viết tắt, xử lý bảng và nhận diện ranh giới mục; tinh chỉnh trên dev rồi đánh giá bằng tập test mới.
