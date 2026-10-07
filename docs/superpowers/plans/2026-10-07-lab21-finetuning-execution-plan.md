# Kế hoạch Thực hiện Lab 21 — Fine-tuning LLMs (LoRA/QLoRA & Đánh giá Đa nhóm)

> **Mục tiêu:** Fine-tune mô hình ngôn ngữ mở bằng LoRA trên tác vụ CSKH Triage (tiếng Việt), đối chứng công bằng với các biến thể sai lầm phổ biến (`attn_only`, `wrong_lr`, `qlora`), đánh giá khách quan qua 4 nhóm chỉ số (target, regression, format, latency) và phán quyết xem bản fine-tune có thực sự thắng base model đã prompt tối ưu hay không; hoàn thiện báo cáo và bài học kinh nghiệm.

**Architecture / Workflow:**
- NB1: Data inspection, Chat Template check (<think>), Loss Mask Proof (assistant-only), p95 Token stats, Stratified split.
- NB2: Base Model Zero-shot / Few-shot evaluation (Baseline A: Naive prompt, Baseline B: Optimized prompt), Freeze Baselines.
- NB3: LoRA SFT đúng chuẩn (vùng không hối tiếc: text-linear all projections, lr=10x full-FT, batch effective < 32), lưu adapter `correct`.
- NB4: Misconfiguration Autopsy (3 run đối chứng cùng max_steps: `attn_only` matched rank, `wrong_lr`, `qlora`).
- NB5: 4-group Evaluation, Regression Gate Verdict, chấm 3 contrast runs trên target task (`autopsy.json`), qualitative failure analysis.
- NB6 (Bonus B1): Merge & Unload verification (delta >= -0.01) + Multi-adapter Hot-swapping.
- Verification & Reporting: `scripts/verify.py` pass 100%, hoàn thiện `submission/REPORT.md` (khớp chính xác số liệu trong `results/`), `submission/REFLECTION.md`.

**Tech Stack:** PyTorch, Transformers, PEFT, TRL, Accelerate, Datasets.

---

## Danh sách các phần việc (Task Breakdown)

### Phần 1: Kiểm thử Môi trường & Khởi động Pipeline (NB1)
- **Mục tiêu:** Xác minh môi trường phần cứng/phần mềm, kiểm tra chat template, chạy chứng minh mask loss, tính toán p95 length, tạo tập split chuẩn seed 42.
- **Tác vụ cụ thể:**
  1. Kiểm tra cấu hình Tier (`.env` / `config.py`), kiểm tra tính tương thích `device.banner()`.
  2. Thực thi `notebooks/01_data_and_mask.py` (hoặc script tương ứng).
  3. Kiểm tra các file kết quả: `results/template_check.json`, `results/mask_proof.json`, `results/token_stats.json`, `data/split/{train,val}.jsonl`.
  4. Đảm bảo 2 assertions trong `mask_proof.json` đạt chuẩn: `answer_is_supervised = True`, `question_is_masked = True`, `supervised_fraction < 0.95`.

### Phần 2: Đóng băng Mốc Đánh giá & Đo 2 Baseline A/B (NB2)
- **Mục tiêu:** Đo đạc năng lực của Base Model trước khi train để thiết lập mốc so sánh công bằng.
- **Tác vụ cụ thể:**
  1. Nạp Base Model.
  2. Đánh giá Baseline (a): Base model + Naive Prompt trên 4 nhóm (target, regression, format, latency).
  3. Đánh giá Baseline (b): Base model + Optimized Prompt trên 4 nhóm.
  4. Xác nhận điều kiện tiên quyết: `(b) > (a)` và tính SHA256 của `OPTIMIZED_PROMPT`.
  5. Đóng băng kết quả vào `results/baselines_frozen.json`.

### Phần 3: Huấn luyện Mô hình Chuẩn Vùng-Không-Hối-Tiếc (NB3)
- **Mục tiêu:** Fine-tune adapter `correct` với cấu hình LoRA tối ưu cho text decoder.
- **Tác vụ cụ thể:**
  1. Xác định target modules (`text-linear`: bao gồm cả linear attention và full attention, loại bỏ vision tower).
  2. Nạp dataset pre-tokenized với loss mask đã kiểm chứng ở NB1.
  3. Huấn luyện SFT với `learning_rate` thang LoRA (2e-4), batch hiệu dụng 16, số epoch/step đồng bộ.
  4. Lưu adapter tại `adapters/correct/`.
  5. Ghi nhận log huấn luyện (loss cuối, peak VRAM, thời gian) vào `results/runs.csv`.

### Phần 4: Giải phẫu Lỗi Cấu hình — 3 Run Đối chứng (NB4)
- **Mục tiêu:** Huấn luyện 3 mô hình đối chứng cùng ngân sách bước (step budget) để trả lời các câu hỏi cốt lõi về bản chất LoRA.
- **Tác vụ cụ thể:**
  1. Run 1 (`attn_only`): Chỉ gắn adapter vào `q_proj, v_proj` nhưng nâng rank `r` bằng `matched_rank()` để bằng số tham số huấn luyện với `correct`.
  2. Run 2 (`wrong_lr`): Dùng LR thang full fine-tune (2e-5, thấp hơn 10x) để thấy hiện tượng underfitting / loss phẳng.
  3. Run 3 (`qlora`): Huấn luyện lượng tử hóa 4-bit (QLoRA) để đo lường trade-off VRAM vs chất lượng/tốc độ.
  4. Lưu các adapter và cập nhật đầy đủ 3 dòng vào `results/runs.csv`.

### Phần 5: Đánh giá Toàn diện, Phán quyết Cổng Hồi quy & Phân tích Định tính (NB5)
- **Mục tiêu:** Chấm điểm adapter `correct` và 3 adapter đối chứng trên cùng thang đo tác vụ thực tế.
- **Tác vụ cụ thể:**
  1. Chấm adapter `correct` trên target, regression, format, latency; so sánh với Baseline (a) và (b).
  2. Chạy cổng hồi quy `regression_gate`, xuất phán quyết vào `results/verdict.json`.
  3. Chấm 3 adapter đối chứng (`attn_only`, `wrong_lr`, `qlora`) trên tập target/format, xuất `results/autopsy.json`.
  4. Trích xuất ít nhất 5 ca kiểm thử định tính (bắt buộc gồm ≥2 ca FT thắng và ≥2 ca FT thua), xuất `results/qualitative.json`.

### Phần 6: Bonus B1 — Merge Trọng số & Phục vụ Đa Adapter (NB6) *(Tùy chọn/Khuyến nghị)*
- **Mục tiêu:** Hợp nhất LoRA vào base model không làm tụt điểm và thử nghiệm hot-swap adapter.
- **Tác vụ cụ thể:**
  1. Thực hiện `merge_and_unload()` và assert độ tụt điểm `delta >= -0.01`, xuất `results/merge_check.json`.
  2. Thực hiện load và chuyển đổi linh hoạt nhiều adapter trên cùng một base model đang nạp trong VRAM.

### Phần 7: Kiểm định Gatekeeper & Hoàn thiện Báo cáo (Verification & Reporting)
- **Mục tiêu:** Vượt qua toàn bộ bộ kiểm tra tự động của `scripts/verify.py` và hoàn thiện văn bản báo cáo khoa học, phản tư cá nhân.
- **Tác vụ cụ thể:**
  1. Chạy `python scripts/verify.py` kiểm tra 100% PASS (mask proof, checksums, prompt SHA, step budget match, parameter matched rank, verdict).
  2. Điền đầy đủ và chính xác tất cả số liệu từ `results/*.json` và `results/runs.csv` vào [REPORT.md](file:///d:/Document/Ai_Thuc_Chien/7-10-2026/Day21-Track3-DaoDuyHieu-2A202602651/submission/REPORT.md).
  3. Viết phân tích giải phẫu lỗi NB4 (trả lời sâu sắc 3 câu hỏi 4.1, 4.2, 4.3).
  4. Viết phân tích phán quyết (≥100 từ) và kết luận khoa học (≥150 từ).
  5. Hoàn thiện [REFLECTION.md](file:///d:/Document/Ai_Thuc_Chien/7-10-2026/Day21-Track3-DaoDuyHieu-2A202602651/submission/REFLECTION.md) trả lời 5 câu hỏi phản tư.

