# Phân Tích Ca Lỗi (Failure Analysis) — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lương Minh Quân
**Ngày thực hiện:** 2026-08-19

> Dữ liệu lấy từ lần chạy thực tế `outputs/graphrag_eval_results.csv` (6 câu Golden Dataset, subset 10.000 dòng đầu `HackerNoon/tech-company-news-data-dump`, LLM: orimise `gemini-2.5-flash`).

---

## Ca lỗi 1: Flat RAG — mất thông tin do phân mảnh ngữ cảnh (context fragmentation)

- **Question ID & câu hỏi:** `G04` (multi-hop) — *"L&T Technology Services Ltd both developed something and made an acquisition, reported in two separate articles. What are these two events?"*
- **Hệ thống lỗi:** Flat RAG.
- **Triệu chứng (Symptom):** Judge chấm Comprehensiveness = 2/5, Multi-hop reasoning = 2/5. Câu trả lời: *"The provided documents do not contain evidence of L&T Technology Services Ltd developing a specific product or technology."* — model tự tin phủ định một sự thật có thật trong dữ liệu.
- **Truy vết nguyên nhân gốc rễ (Root cause):**
  1. `answer_flat_rag()` gọi `retrieve_flat_context(question, k=6)` — chỉ lấy 6 chunk có embedding gần câu hỏi nhất.
  2. Hai sự kiện liên quan tới cùng thực thể `L&T Technology Services Ltd` nằm ở **hai chunk khác nhau, hai bài báo khác nhau** (`493e89a07cb7b2a5b47c::c0000` — DEVELOPED Engineering R&D, ngày 2023-09-06; `e72f68c2e141879b6b58::c0000` — ACQUIRED Smart World & Communication, ngày 2023-01-13).
  3. Về mặt ngữ nghĩa, hai chunk này khá khác nhau (một nói về dịch vụ ER&D, một nói về thương vụ M&A bằng tiếng liên quan tới Ấn Độ) — nên khi encode câu hỏi ghép cả hai vế "developed" và "acquisition", vector similarity không đủ để **cả hai** chunk cùng lọt vào top-6, chỉ 1/2 chunk được kéo về.
  4. Vì thiếu một nửa ngữ cảnh, generator kết luận sai là "không có bằng chứng" thay vì chỉ đơn giản là "chưa được cấp bằng chứng" — một dạng lỗi do retrieval, không phải do generation bịa đặt (faithfulness với context đã cấp vẫn cao).
- **Bằng chứng:** context thực tế đưa vào Flat RAG chỉ chứa 6 chunk, trong đó có chunk `e72f68c2e141879b6b58::c0000` (acquisition) nhưng KHÔNG có `493e89a07cb7b2a5b47c::c0000` (development) — kiểm chứng lại bằng cách chạy trực tiếp `retrieve_flat_context()` với câu hỏi G04 và so khớp `chunk_id` trong kết quả.
- **Đề xuất khắc phục:** (a) tăng k cho câu hỏi multi-hop được nhận diện qua heuristic/LLM phân loại độ phức tạp; (b) hoặc — giải pháp triệt để hơn — chuyển sang GraphRAG cho lớp câu hỏi multi-hop, vì bản chất vấn đề là top-k similarity search không đảm bảo lấy đủ *tất cả* thông tin quanh một thực thể, trong khi graph traversal theo cạnh thì đảm bảo được điều đó (xem Ca lỗi 2 bên dưới — GraphRAG giải quyết đúng câu hỏi này với điểm 5/5).

---

## Ca lỗi 2: GraphRAG — giảm chi tiết do bất đối xứng k giữa hai nhánh hybrid

- **Question ID & câu hỏi:** `G02` (factoid) — *"What company did Fidelity National Information Services (FIS) acquire?"*
- **Hệ thống lỗi:** GraphRAG (Hybrid).
- **Triệu chứng (Symptom):** Judge chấm Comprehensiveness = 4/5 (thấp hơn Flat RAG's 5/5) dù relation cốt lõi (`FIS ACQUIRED Worldpay`) được trả lời đúng và có trích dẫn `chunk_id` đầy đủ. Rationale của judge: *"slightly less comprehensive... omits the additional context provided about Worldpay being a Cincinnati-based payments technology company and the strategic reason for the acquisition."*
- **Truy vết nguyên nhân gốc rễ (Root cause):**
  1. `answer_graph_rag()` build context = `GRAPH` (1 dòng cạnh cô đọng, chỉ gồm tên thực thể + relation + evidence ngắn) + `VECTOR` (`retrieve_flat_context(question, k=4)` — **k=4**, thấp hơn k=6 mà `answer_flat_rag()` dùng).
  2. Cạnh graph tự nó chỉ mang "FIS ACQUIRED Worldpay" + một câu evidence ngắn, KHÔNG có thông tin "Cincinnati-based" hay lý do chiến lược "widen its reach" — những chi tiết này nằm rải rác trong văn xuôi phần còn lại của chunk, không được `evidence` field của triple bắt lại (vì evidence chỉ trích 1 câu ngắn lúc extraction).
  3. Nhánh vector trong hybrid context, do k=4 nhỏ hơn Flat RAG's k=6, có xác suất bỏ sót đúng những chunk mang chi tiết phụ đó cao hơn.
  4. Kết quả: câu trả lời đúng về mặt sự kiện cốt lõi (faithfulness vẫn 5/5) nhưng thiếu độ đầy đủ so với Flat RAG trên chính câu hỏi factoid đơn giản — một trade-off ngược lại với kỳ vọng thông thường (GraphRAG thường thắng, nhưng ở factoid đơn-hop, structure không giúp ích và k nhỏ hơn lại là bất lợi thuần túy).
- **Bằng chứng:** so sánh trực tiếp `flat_answer` (đầy đủ "Cincinnati-based... widen its reach") vs `graph_answer` (chỉ nêu tên Worldpay + chunk_id) trong `outputs/graphrag_eval_results.csv`, dòng `id=G02`.
- **Đề xuất khắc phục:** (a) đồng bộ k của nhánh vector trong hybrid context lên bằng k của Flat RAG (hoặc điều chỉnh động: nếu graph trả về ít cạnh, tăng bù k phía vector); (b) mở rộng `evidence` field lúc extraction để chứa câu văn đầy đủ hơn (2–3 câu quanh relation) thay vì chỉ 1 câu ngắn, giúp graph context tự nó giàu thông tin hơn mà không cần dựa hoàn toàn vào nhánh vector bù đắp.
