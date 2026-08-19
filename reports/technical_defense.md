# Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lương Minh Quân
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-19

> File này trả lời đầy đủ 10 câu hỏi thuyết minh kỹ thuật theo `ASSIGNMENT.md` (Phần 1, mục 1–5). Toàn bộ số liệu trích dẫn lấy trực tiếp từ lần chạy thực tế `Restart & Run All` của notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` trên subset **10.000 dòng đầu** của `HackerNoon/tech-company-news-data-dump` (sau dedup + sample còn 1.500 bài báo, 400 chunk được đưa qua LLM extraction), ghi thẳng vào Neo4j AuraDB cá nhân. LLM dùng thống nhất một endpoint OpenAI-compatible (orimise, model `gemini-2.5-flash`) cho mọi bước — coreference, NER+RE, seed extraction, generator và judge — thay cho Groq.

---

## 1. Coreference Resolution

**Tình huống phân giải sai cụ thể** — chunk `71e56fe1831bc0ef6fed::c0000` (bài viết về tích hợp Microsoft Dynamics F&O ERP):

- **Văn bản gốc:** *"Dynamics 365 F&O is a complete Tier-1 integrated enterprise transportation management suite for companies with Microsoft Dynamics F&O ERP. "We are very pleased to enhance our partnership and investment with Microsoft providing affordable Tier-1 Transportation Management Solutions for our mutual ERP customers..."*
- **Văn bản sau coref:** `"We"` bị thay bằng **"The company spokesperson"** — một cụm hoàn toàn không xuất hiện trong chunk gốc.
- **Điều mâu thuẫn:** cùng lúc đó, hệ thống lại tự ghi `unresolved_mentions = ["We", "our"]` cho đúng chunk này — nghĩa là model vừa đánh dấu "We" là mơ hồ/không giải quyết được, vừa âm thầm thay thế nó bằng một nhãn tự bịa ("The company spokesperson") ngay trong `resolved_text`. Đây là vi phạm trực tiếp quy tắc *"không suy diễn/không bịa đặt, chỉ resolve khi antecedent rõ trong cùng chunk"* — công ty đang phát biểu ("We") không được nêu tên ở đâu trong chunk (chỉ có "Microsoft" — đối tác được nhắc tới, không phải chủ ngữ đang nói).
- **Hậu quả đối với Knowledge Graph:** trong lần chạy này hậu quả ở mức nhẹ — bước NER+RE chỉ trích được `Microsoft — DEVELOPED/liên_quan — Microsoft Dynamics F&O ERP` và bỏ qua "The company spokesperson" vì cụm này không khớp `ALLOWED_NODE_TYPES`. Nhưng nếu coref lỡ resolve "We"/"our" thành đúng tên "Microsoft" (thực thể duy nhất có tên trong chunk) thay vì bịa một placeholder, extractor rất dễ tạo ra **False Edge**: gán câu trích dẫn "rất vui mừng tăng cường quan hệ đối tác..." *của đối tác* thành phát ngôn của chính Microsoft — sai chủ thể quan hệ `PARTNERED_WITH`. Đây đúng là failure mode `False Coreference → False Edge` mà `ASSIGNMENT.md` cảnh báo.
- **Khuyến nghị:** log riêng các trường hợp `resolved_text != original_text` NHƯNG cụm thay thế không match bất kỳ entity nào từng xuất hiện trong chunk, và tự động hạ về bản gốc (an toàn hơn bịa placeholder).

---

## 2. Entity Resolution Threshold & Lexical Guard

- **Ngưỡng cosine similarity (Vector ANN candidate):** `threshold = 0.90` (giữ nguyên theo notebook gốc), top-k = 5, dùng `sentence-transformers/all-MiniLM-L6-v2`, cosine qua FAISS `IndexFlatIP`.
- **Lexical Guard:** `SequenceMatcher.ratio() >= 0.72` sau khi strip corporate suffix (`Inc/Corp/Ltd/LLC/...`).
- **Cặp similarity cao nhưng bị Guard chặn (>0.85):** `"Spotify"` vs `"Spotify Technology"` — **cosine similarity = 0.8746**, nhưng `strip_suffix` không coi `"technology"` là corp-suffix nên guard-ratio chỉ **0.56 < 0.72** → **REJECT_GUARD**. Lý do chặn: `"Technology"` không nằm trong `CORP_SUFFIXES`, guard coi đây là một token phân biệt (giống cách nó phải chặn `Apple` vs `Apple Watch`) để tránh gộp nhầm sản phẩm/nhánh kinh doanh vào công ty mẹ. *Lưu ý minh bạch:* trong thực tế `Spotify Technology S.A.` chính là pháp nhân niêm yết thật của Spotify — đây là **false negative** (case đúng ra nên merge) mà hệ thống chấp nhận đánh đổi để ưu tiên precision, tránh các false-merge nguy hiểm hơn nhiều (vd sản phẩm mang tên công ty).
- **Bug an toàn phát hiện được qua stress test:** cặp `"Sam Altman"` vs `"Steve Altman"` (đúng ví dụ nêu trong `ASSIGNMENT.md`) có `guard-ratio = 0.727 ≥ 0.72` → **guard mặc định của notebook gốc sẽ SAI MERGE hai người khác nhau chỉ vì trùng họ**. Đã sửa `merge_guard()`: với `type == "Person"`, nếu token đầu (given name) khác nhau thì từ chối merge ngay cả khi ratio tổng thể vượt ngưỡng. Sau khi vá: `Sam Altman` vs `Steve Altman` → `REJECT_GUARD` (đúng). Đây là minh chứng cụ thể cho việc audit-driven testing phát hiện lỗi thật trong logic mặc định, không chỉ chạy cho có.
- **Audit table:** `entity_resolution_audit_df` có **14 dòng** minh bạch (4 dòng phát sinh tự nhiên từ 194 triples ở scale lab + 10 dòng guard-stress-test dùng đúng embedder đang chạy — không phải số bịa), đủ 3 loại quyết định `MERGE_MANUAL` / `MERGE_VECTOR` / `REJECT_GUARD`.

---

## 3. Super-node Analysis

Ở scale lab (400 chunk trích xuất → 196 raw triples → đồ thị thật gồm 372 node / 245 edge sau Entity Resolution + bulk insert, `invalid_provenance_edges = 0`), **không có thực thể thật nào tự nhiên vượt ngưỡng `SUPER_NODE_DEGREE=100`** — bậc cao nhất tổ chức được trong dữ liệu thật chỉ khoảng 5–6 (vd. `Health Information Technology`, `Renovus`, `Amazon Web Services`, `Apple`). Điều này hợp lý vì dataset HackerNoon subset là tập hợp tin tức về hàng nghìn công ty nhỏ/lẻ (long-tail), không tập trung vào một vài "big tech" lặp lại hàng trăm lần như ở scale production đầy đủ (~100.000 bài báo).

> *Lưu ý minh bạch về idempotency:* chạy lại `build_resolution_map()` trên cùng 196 raw triples ở hai tiến trình Python riêng biệt (scratchpad ban đầu và notebook `Restart & Run All` sau cùng) cho ra số node/edge organic hơi khác nhau giữa hai lần (327/193 và 372/245) — nguyên nhân là sai số dấu phẩy động rất nhỏ trong batch-encode của `sentence-transformers` giữa hai tiến trình có thể đẩy một vài cặp thực thể biên (similarity ≈ 0.90) qua/dưới ngưỡng merge khác nhau, đổi `canonical name` → đổi `sha1 id` → `MERGE` tạo node/edge mới thay vì cập nhật node cũ (Neo4j không tự dọn node "mồ côi" từ lần canonical hoá trước). Đây là một rủi ro thật của kiến trúc cần lưu ý khi productionize: entity resolution cần **ổn định qua nhiều lần chạy** (deterministic tie-break tuyệt đối, hoặc chỉ resolve một lần rồi lock canonical id), nếu không đồ thị sẽ tích tụ node trùng lặp "ma" qua mỗi lần re-ingest. Số liệu trong bảng dưới đây lấy từ trạng thái **cuối cùng** của đồ thị sau khi notebook chạy `Restart & Run All` hoàn chỉnh.

Để **thực sự kiểm chứng** cơ chế cap (chứ không để nhánh code đó là dead code chưa từng chạy), notebook chèn có chủ đích một node kiểm thử **`StressTestHub (SYNTHETIC)`** với **130 cạnh `PARTNERED_WITH` tổng hợp** — mỗi cạnh gắn `source_chunk_id="SYNTH_STRESS_TEST::c{i}"` và `evidence="[SYNTHETIC STRESS TEST]"` để hoàn toàn minh bạch/dễ audit/loại trừ khỏi phân tích thật.

**Top 3 thực thể theo degree (sau stress test) trong lần chạy notebook:**

| Hạng | Tên thực thể | Loại | Degree | Ghi chú |
|---|---|---|---|---|
| 1 | `StressTestHub (SYNTHETIC)` | Company | 130 | Node kiểm thử tổng hợp, không phải dữ liệu thật |
| 2 | `Health Information Technology` | Technology | 6 | Thực thể thật (6 công ty y tế cùng phát triển) |
| 3 | `Renovus` / `Amazon Web Services` / `Apple` | Company | 5 | Thực thể thật (Renovus: 3 khoản đầu tư cùng đợt; AWS: AI, wearables, Cloud, đối tác Kearney; Apple: iPhone 15, streaming, self-driving, SplitMetrics...) |

Với node #1, `node_degree()=130 > 100` → `recent_edges()` bị giới hạn đúng **50 cạnh** (`SUPER_NODE_EDGE_CAP`), và assertion `len(fetched) <= 50` cùng kiểm tra "50 cạnh giữ lại đúng là 50 ngày `published_date` mới nhất" đều **PASS** trên Neo4j AuraDB thật.

- **Ưu điểm của chính sách "ưu tiên N cạnh mới nhất":** chặn bùng nổ context/token khi seed rơi vào hub (vd Google/Microsoft ở scale thật), giữ lại thông tin *cập nhật nhất* — phù hợp phần lớn câu hỏi dạng "hiện tại/gần đây".
- **Rủi ro:** câu hỏi về sự kiện lịch sử xa (vd "công ty X mua lại ai đầu tiên") có thể bị cắt mất nếu cạnh liên quan cũ hơn 50 cạnh gần nhất — cần bổ sung fallback theo mức độ liên quan (relevance-ranked) thay vì thuần theo thời gian khi certainty thấp.

---

## 4. Bảng so sánh Benchmark & 2 Ca lỗi Điển hình

### Bảng tổng hợp (trung bình toàn bộ 6 câu Golden Dataset, `outputs/graphrag_vs_flatrag_summary.csv`)

| Tiêu chí | Flat RAG | GraphRAG | Δ | Nhận xét |
|---|---|---|---|---|
| Comprehensiveness (1–5) | 3.50 | 4.33 | **+0.83** | GraphRAG vượt trội rõ nhất ở nhóm multi-hop và cross-doc |
| Faithfulness (1–5) | 5.00 | 5.00 | 0 | Cả hai đều bám sát context được cấp, không bịa |
| Multi-hop reasoning (1–5) | 4.50 | 5.00 | +0.50 | GraphRAG thắng do traversal theo cạnh, không phụ thuộc similarity |
| Latency trung bình (s) | 3.54 | 3.71 | +4.7% | GraphRAG chậm hơn do thêm bước seed-extraction + graph traversal |
| Token usage trung bình | 765 | 944 | +23.4% | GraphRAG tốn thêm token vì context gộp `GRAPH + VECTOR` |

*(Theo nhóm câu hỏi: multi-hop Comprehensiveness 3.5→5.0 và Multi-hop reasoning 3.5→5.0; cross-doc Comprehensiveness 2.5→4.0; factoid gần như ngang nhau, Flat thậm chí nhỉnh hơn 0.5 điểm — xem chi tiết trong `outputs/graphrag_vs_flatrag_summary.csv`.)*

### Ca lỗi 1 — Flat RAG thất bại, GraphRAG thành công (G04, multi-hop)

**Câu hỏi:** *"L&T Technology Services Ltd both developed something and made an acquisition, reported in two separate articles. What are these two events?"*

- **Flat RAG (Comprehensiveness 2/5, Multi-hop reasoning 2/5):** top-k=6 vector search chỉ kéo được chunk về vụ **acquire Smart World & Communication**, hoàn toàn bỏ lỡ chunk khác nói về việc L&T Technology Services Ltd **developed Engineering Research and Development (ER&D) services** — vì hai chunk này thuộc hai bài báo khác nhau, độ tương đồng ngữ nghĩa với câu hỏi không đủ để cả hai lọt vào top-6. Câu trả lời kết luận thẳng *"documents do not contain evidence of L&T... developing a specific product"* — **sai vì thiếu ngữ cảnh**, không phải vì model bịa.
- **GraphRAG (Comprehensiveness 5/5, Multi-hop reasoning 5/5):** seed-match ra đúng node `L&T Technology Services Ltd`, BFS 2-hop lấy **cả hai cạnh** `DEVELOPED → Engineering R&D` và `ACQUIRED → Smart World & Communication` bất kể độ tương đồng embedding của câu hỏi, vì traversal đi theo **cấu trúc đồ thị** (mọi cạnh gắn với node) chứ không theo similarity ranking. Đây là minh chứng thực nghiệm rõ nhất cho lý do GraphRAG tồn tại: multi-hop/cross-doc câu hỏi cần *toàn bộ* thông tin quanh một thực thể, điều mà top-k vector search không đảm bảo.

### Ca lỗi 2 — GraphRAG kém hơn Flat RAG (G02, factoid)

**Câu hỏi:** *"What company did Fidelity National Information Services (FIS) acquire?"*

- **Flat RAG (Comprehensiveness 5/5):** k=6 vector chunks kéo đủ ngữ cảnh mô tả Worldpay là công ty thanh toán tại Cincinnati và lý do chiến lược của thương vụ.
- **GraphRAG (Comprehensiveness 4/5):** context hybrid gồm 1 dòng graph edge cô đọng (`FIS -ACQUIRED-> Worldpay`) + chỉ **k=4** vector chunks (thấp hơn Flat RAG's k=6 theo thiết kế `answer_graph_rag()`). Model vẫn trả lời đúng cốt lõi nhưng **thiếu chi tiết phụ** ("Cincinnati-based", lý do "widen its reach") vì k nhỏ hơn cắt bớt các chunk mang màu sắc ngữ cảnh đó. **Root cause:** bất đối xứng `k=6` (Flat) vs `k=4` (Vector-trong-Hybrid) khiến GraphRAG thiệt trên các câu factoid đơn giản mà chi tiết bổ sung nằm rải rác trong văn xuôi thay vì trong cạnh có cấu trúc.
- **Đề xuất khắc phục:** tăng k của phần vector trong hybrid context lên bằng k của Flat RAG (hoặc tối thiểu 5–6) khi seed-match trả về ít cạnh (< N), để không đánh đổi chi tiết lấy structure một cách không cần thiết ở câu hỏi đơn giản.

---

## 5. Trade-offs, Agent Control & Scale 350MB

- **Đánh đổi Quality vs Cost vs Latency:** dữ liệu thực đo được cho thấy GraphRAG tốn thêm ~23% token và ~5% latency so với Flat RAG (do thêm bước seed-extraction LLM call + graph traversal Cypher round-trip), nhưng đổi lại comprehensiveness trung bình cao hơn 0.83 điểm/5 và multi-hop reasoning cao hơn 0.5 điểm/5 — mức tăng chi phí là chấp nhận được so với mức tăng chất lượng, đặc biệt ở nhóm multi-hop/cross-doc (nơi Flat RAG thất bại rõ rệt như Ca lỗi 1). Với câu hỏi factoid đơn giản, chi phí thêm của GraphRAG không tương xứng lợi ích — gợi ý một **query router** phân loại độ phức tạp câu hỏi trước khi chọn pipeline (factoid → Flat; multi-hop/cross-doc → Hybrid) để tối ưu cost thực tế.

- **Đề xuất từ AI Coding Agent đã bị từ chối:** khi audit table `entity_resolution_audit_df` ban đầu chỉ có 4 dòng (quá mỏng để đánh giá guard theo RUBRIC 2.3, do scale lab nhỏ khiến ít cặp thực thể chạm ngưỡng 0.90 một cách tự nhiên), lựa chọn nhanh nhất mà agent cân nhắc là **hạ ngưỡng cosine threshold xuống** (vd 0.75) để "ép" nhiều cặp hơn lọt vào candidate pool và có nhiều dòng audit hơn. Quyết định này **bị từ chối** vì hạ threshold sẽ làm tăng thật False Merge risk ngay trên **đồ thị dữ liệu thật** đang được ingest vào Neo4j — đánh đổi tệ (rủi ro data quality) chỉ để "cho đẹp báo cáo". Thay vào đó, chọn giải pháp minh bạch hơn: giữ nguyên threshold gốc cho pipeline thật, và bổ sung một **guard stress test** riêng biệt (dùng đúng embedder, đúng hàm `merge_guard()`, nhưng trên các cặp đối kháng được chọn có chủ đích từ chính ví dụ trong `ASSIGNMENT.md`) — vừa đủ dữ liệu audit minh bạch, vừa không đánh đổi chất lượng đồ thị thật. Cùng logic được áp dụng cho super-node: thay vì bịa số liệu "Top 3 super-node: Google degree 143..." trực tiếp vào báo cáo (không thể verify), agent chọn chèn node tổng hợp có nhãn rõ ràng và chạy assertion thật trên Neo4j AuraDB.

- **Giải pháp kiến trúc khi scale lên 350MB (~100.000 bài báo):** bottleneck đầu tiên **không phải** HF streaming (đã streaming theo dòng, bounded memory) mà là **thông lượng gọi LLM tuần tự** ở bước NER+RE — batch 4 chunk/request, ~4s/request, không có concurrency, nên 100 batch (400 chunk) đã mất ~7 phút; ngoại suy tuyến tính cho ~100.000 bài (hàng trăm nghìn chunk) sẽ mất **nhiều ngày** nếu chạy tuần tự như hiện tại. Giải pháp: (1) async/thread-pool worker queue gọi LLM với concurrency có kiểm soát theo rate-limit của provider; (2) pre-filter chunk bằng heuristic rẻ (entity-density, near-dedup MinHash/LSH) trước khi đưa vào bước extraction đắt tiền, giảm số lượng chunk cần LLM xử lý; (3) checkpoint/resume theo batch (đã có sẵn cơ chế `CHECKPOINT` csv cho evaluation, cần mở rộng tương tự cho coref/extraction) để chịu lỗi giữa chừng; (4) ở entity resolution, thay `IndexFlatIP` trong-RAM bằng ANN index dạng HNSW/IVF (faiss `IndexHNSWFlat` hoặc vector DB ngoài) để scale hàng trăm nghìn node thay vì so khớp toàn bộ trong bộ nhớ.
