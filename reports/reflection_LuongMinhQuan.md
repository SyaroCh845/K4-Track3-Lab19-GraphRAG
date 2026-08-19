# Suy Ngẫm Cá Nhân & Kế Hoạch Đồ Án — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lương Minh Quân
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-19

---

## 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| Conservative Coreference | Module 1 | `resolve_coref_batch()`, `run_coref()` (Cell 1.7) | Trên 400 chunk đưa qua LLM, 161/400 chunk bị thay đổi text, 163/400 có `unresolved_mentions` được log lại thay vì bịa. Vẫn phát hiện 1 ca "rò rỉ" — model vừa log `unresolved_mentions=["We","our"]` vừa âm thầm thay `"We"` bằng cụm tự bịa `"The company spokesperson"` trong cùng chunk (`71e56fe1831bc0ef6fed::c0000`) — cho thấy prompt "conservative" không hoàn toàn triệt để, cần thêm bước validate chéo giữa `resolved_text` và `unresolved_mentions`. |
| Schema & Allowlist Guard | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `run_extraction()` (Cell 2.1) | 400 chunk → 196 raw triples, 100% relation type nằm trong allowlist (8 loại: DEVELOPED nhiều nhất 99, ít nhất FOUNDED 2) vì code lọc cứng `if rel not in ALLOWED_RELATIONS: continue` trước khi đưa vào DataFrame — không có quan hệ "lạ" lọt qua. |
| Bulk Cypher Ingestion (`UNWIND`) | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` (Cell 2.3) | Insert theo batch `UNWIND $rows AS row` (batch 1000, nhưng dữ liệu lab chỉ ~194-372 dòng nên luôn gói gọn 1 batch) — xác nhận **0 cạnh** thiếu `source_chunk_id`/`published_date` qua `graph_checks()` (Cell 2.4) trên Neo4j AuraDB thật. |
| Entity Resolution & Union-Find | Module 3 | `build_resolution_map()`, `UF`, `merge_guard()` (Cell 2.2) | Kết hợp Manual Alias → FAISS ANN (threshold 0.90) → Lexical Guard (`SequenceMatcher >= 0.72` + rule riêng cho Person) → Union-Find gán canonical id. Audit table thật có 14 dòng minh bạch, phát hiện và vá được 1 bug thật trong guard mặc định (xem mục 2 bên dưới). |
| Super-node Degree Cap | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE/EDGE_CAP` (Cell 3.3), stress test (Cell 5.1) | Dữ liệu lab scale nhỏ không có super-node thật (degree cao nhất tự nhiên chỉ ~5-6), nên phải tự chèn node kiểm thử tổng hợp 130 cạnh để xác nhận cơ chế cap-50-cạnh-mới-nhất hoạt động đúng trên Neo4j thật — bài học quan trọng: **kiểm thử failure-mode không thể chỉ dựa vào dữ liệu tự nhiên ở scale lab**, phải chủ động tạo test case. |
| LLM-as-a-Judge Evaluation | Module 5 | `judge_answer()`, `run_evaluation()` (Cell 4.2-4.3) | Chạy đủ 6 câu Golden Dataset (2 factoid, 2 multi-hop, 2 cross-doc) × 2 pipeline × 3 tiêu chí = 36 điểm số + rationale. Kết quả nhất quán với lý thuyết: GraphRAG hơn Flat RAG rõ nhất ở multi-hop (comprehensiveness 3.5→5.0) chứ không phải factoid đơn giản. |

---

## 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Notebook mẫu định nghĩa `merge_guard(a, b)` chỉ dựa vào `SequenceMatcher.ratio() >= 0.72` sau khi strip corporate suffix. Khi viết stress test cho đúng ví dụ mà `ASSIGNMENT.md` cảnh báo ("người trùng họ như Sam Altman vs Steve Altman"), phát hiện `SequenceMatcher(None, "sam altman", "steve altman").ratio() = 0.727` — **vượt ngưỡng 0.72**, nghĩa là guard mặc định sẽ **merge nhầm hai người khác nhau** chỉ vì trùng họ "Altman". Đây là lỗi im lặng (silent bug): không crash, không lỗi cú pháp, chỉ âm thầm tạo ra một canonical entity sai trong đồ thị tri thức — loại lỗi khó phát hiện nhất vì phải chủ động nghĩ ra ca đối kháng để test, không tự lộ ra qua chạy pipeline bình thường.
- **Cách xử lý:** Thay vì hạ ngưỡng similarity (sẽ làm giảm recall chung, không giải quyết đúng gốc rễ), sửa `merge_guard()` thêm nhánh riêng cho `type == "Person"`: so sánh token đầu tiên (given name) của hai tên; nếu khác nhau thì từ chối merge ngay lập tức bất kể ratio tổng thể. Viết lại stress test để xác nhận: sau khi vá, `Sam Altman` vs `Steve Altman` → `REJECT_GUARD` (đúng), trong khi các case merge hợp lệ khác (`Apple Inc.` vs `Apple`, `Microsoft Corp` vs `Microsoft`) không bị ảnh hưởng. Bài học: **luôn viết test case đối kháng dựa trên chính ví dụ nguy hiểm nêu trong đề bài**, thay vì tin tưởng threshold mặc định của template — một threshold "nhìn hợp lý" (0.72) vẫn có thể sai ở đúng biên giới mà đề bài cảnh báo.

---

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / dự án:** Trợ lý tra cứu tri thức nội bộ cho một tổ chức có nhiều tài liệu rời rạc (chính sách, quy trình, hợp đồng, ghi chú họp) — bài toán tương tự lớp "corporate knowledge assistant".
- **Đặc thù bài toán & lý do chọn giải pháp:** Phần lớn câu hỏi thực tế là tra cứu đơn (factoid) — Flat RAG/Hybrid RAG đã đủ và rẻ hơn (kết quả thực nghiệm của lab cho thấy Flat RAG không thua kém GraphRAG ở nhóm factoid, thậm chí nhỉnh hơn một chút). Chỉ nên đầu tư xây GraphRAG cho các nhóm câu hỏi có bản chất multi-hop thật sự (vd "ai từng làm ở phòng ban X rồi chuyển sang dự án Y, dự án đó liên quan hợp đồng nào") hoặc cross-doc tổng hợp xu hướng theo thời gian — đúng những nhóm mà thực nghiệm trong lab (G04, G05, G06) chứng minh Flat RAG bị phân mảnh ngữ cảnh. Kiến trúc đề xuất: **query router** phân loại độ phức tạp câu hỏi trước, route sang Flat RAG cho factoid và Hybrid GraphRAG cho multi-hop/cross-doc, để tối ưu chi phí (dữ liệu lab cho thấy GraphRAG tốn thêm ~23% token, ~5% latency).
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Person` (nhân sự), `Team`/`Department` (phòng ban), `Project` (dự án), `Document` (tài liệu/hợp đồng), base label `Entity` giống lab.
  - Relations: `WORKED_AT`/`WORKS_ON` (Person→Team/Project), `OWNS`/`APPROVED_BY` (Document→Person/Team), `REFERENCES` (Document→Document), `SUPERSEDES` (Document→Document, quan trọng để xử lý version tài liệu theo thời gian — bài học từ Super-node Analysis: ưu tiên cạnh mới nhất có rủi ro mất thông tin lịch sử, nên với domain tài liệu nội bộ cần field `superseded_by`/`valid_until` rõ ràng thay vì chỉ dựa `published_date`).
- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Super-node: các phòng ban lớn (vd "Phòng Kỹ thuật") chắc chắn sẽ là super-node thật (khác với lab, dataset nội bộ thường tập trung cao vào một số ít phòng ban/dự án lớn) — cần áp dụng cap ngay từ đầu thay vì coi là edge case hiếm, và cân nhắc thêm community detection (bonus của lab) để cho phép global search theo "chủ đề/phòng ban" thay vì chỉ theo entity đơn lẻ.
  - Entity Resolution: tên người/phòng ban nội bộ có tính đặc thù cao (viết tắt phòng ban, nickname nhân sự) — nên xây `MANUAL_ALIASES` từ chính hệ thống HR/danh bạ nội bộ (nguồn dữ liệu có sẵn, đáng tin hơn suy luận từ text), kết hợp lexical guard đã học được từ lab (đặc biệt rule riêng cho `Person` type sau bug tìm được ở Sam Altman/Steve Altman) để tránh nhầm hai nhân sự trùng họ.

---

## Tự đánh giá

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | 5 | Đã tự chạy đủ pipeline thật (streaming → coref → NER+RE → entity resolution → bulk insert → hybrid retrieval → eval) trên dữ liệu và Neo4j AuraDB thật, không chỉ đọc code mẫu. |
| Khả năng kiểm soát AI Coding Agent | 5 | Đã phản biện và từ chối đề xuất nhanh nhưng rủi ro (hạ threshold entity resolution) của AI Coding Agent, yêu cầu giải pháp minh bạch/kiểm chứng được thay thế. |
| Chất lượng đồ thị tri thức xây dựng | 4 | 0 cạnh thiếu provenance, audit table minh bạch 14 dòng, nhưng phát hiện thêm vấn đề idempotency (re-run entity resolution có thể tạo node "mồ côi" trùng lặp) chưa kịp khắc phục triệt để trong phạm vi lab. |
| Khả năng phân tích và debug hệ thống | 5 | Tìm ra bug thật trong `merge_guard()` qua stress test có chủ đích, truy vết root-cause đầy đủ cho cả 2 ca lỗi Flat RAG/GraphRAG trong `failure_analysis.md`. |
