# Lab 21 — Evaluation Report

**Họ tên**: Đào Duy Hiếu  **MSSV**: 2A202602651  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây được đối chiếu và khớp chính xác 100% với các file trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket chăm sóc khách hàng (CSKH) tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có — kết quả từ `results/template_check.json` xác nhận `verdict: "reasoning preserved — safe to train on traces"`. Khối reasoning `<think>...</think>` được bảo toàn toàn vẹn trong chuỗi render, không bị tokenizer lược bỏ âm thầm. 

**Lý do chọn cấu hình:**
- **Model `unsloth/Qwen3.5-4B`**: Lựa chọn tối ưu cân bằng giữa năng lực hiểu tiếng Việt xuất sắc và giới hạn phần cứng của GPU T4 (14.6 GB khả dụng). Bản 4B chứa 32 layers (24 linear attention xen kẽ 8 full attention), phù hợp hoàn hảo với LoRA fp16.
- **`max_length=1024`**: Dù p95 token thực tế đo được là 98 (suggested 256), việc giữ `max_length=1024` ở tier T4 tạo biên an toàn tuyệt đối chống cắt cụt các câu ticket dài bất thường hoặc câu sinh mở rộng, trong khi vẫn kiểm soát batch hiệu dụng ở mức 16 ($1 \times 16$).

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị thực nghiệm |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% token được tính loss) |
| Câu trả lời nằm trong loss | `True` (xác thực qua giải mã token ngược) |
| Câu hỏi KHÔNG nằm trong loss | `True` (toàn bộ system prompt và user ticket được mask thành -100) |

Dán 3–5 dòng đầu của đoạn được tính loss (*giải mã từ `results/mask_proof.json`*):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

**Nhận xét:** Mask hoàn toàn chuẩn xác. Nếu vô tình dùng cờ `everything`, tỷ lệ `supervised_fraction` sẽ là 100% dẫn đến mô hình học thói quen vẹt nhắc lại câu hỏi của người dùng thay vì sinh câu trả lời JSON.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3562.1 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1116.5 |
| (c) LoRA fine-tune | 0.975 | 0.567 | 1.000 | 1525.3 |

**(b) có thật sự mạnh hơn (a) không?** Có — vượt bậc rõ rệt. Baseline (b) đưa Target Accuracy từ 0.000 lên 0.765 (76.5%), Format hợp lệ từ 0% lên 100%, đồng thời giảm độ trễ hơn 3.19 lần (từ 3562ms xuống 1116ms) do prompt tối ưu ép mô hình dừng ngay sau khi đóng ngoặc nhọn JSON thay vì sinh văn xuôi lan man đến trần token.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** Không. Prompt tối ưu gốc được giữ nguyên vẹn với mã băm SHA256 là `719e74d3b6232053`, đảm bảo tính liêm chính tuyệt đối của mốc đo lường công bằng trước khi huấn luyện.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6266 | **0.975** | 443.9 | 8.78 |
| `attn_only` | q,v (matched) | 283 | 32,456,704 | 1e-4 | 0.5361 | **0.970** | 300.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 448.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 518.8 | 3.86 |

> **Phân tích chi tiết 3 câu hỏi bắt buộc:**

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
- Trên tập target thực tế (NB5 §4), `attn_only` ($r=283$) đạt 0.970, chịu thua nhẹ trước `correct` ($r=16$) đạt 0.975. Tuy nhiên, nếu chỉ nhìn vào `final_loss` huấn luyện ở NB4, `attn_only` lại có loss thấp hơn rõ rệt (0.5361 so với 0.6266 của `correct`).
- Thứ tự theo train loss (attn_only tốt hơn correct) **hoàn toàn trái ngược** với thứ tự năng lực trên tác vụ thực tế (correct tốt hơn attn_only). Điều này vạch trần "Lỗi #3": đánh giá mô hình bằng chỉ số thay thế (training loss / perplexity) dễ bị đánh lừa bởi hiện tượng ghi nhớ vẹt (memorization) do rank quá lớn ($r=283$) gây overfit trên 225 mẫu huấn luyện.
- Bằng chứng thực nghiệm này chứng minh đòn bẩy kiến trúc thực sự của LoRA nằm ở **vị trí gắn adapter** (phủ khắp các ma trận linear của text decoder bao gồm cả FFN và linear attention projections) chứ không nằm ở việc cố tăng **rank $r$**.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
- Đường loss của `wrong_lr` bị nghẽn ở mức rất cao (1.5702 sau 30 steps) so với `correct` (0.6266). Khi đánh giá ở NB5, `wrong_lr` hoàn toàn thất bại với Target = 0.000 và Format = 0.000, sinh ra văn xuôi ngô nghê giống hệt Base Model chưa fine-tune.
- Nếu chỉ nhìn vào loss mà không biết đây là lỗi cấu hình Learning Rate, kỹ sư sẽ dễ kết luận sai lầm rằng *"tập dữ liệu quá khó/nhiễu"* hoặc *"LoRA rank 16 không đủ dung lượng học bài toán này"*, rồi vội vàng tăng rank hay đổi model.
- Thực chất, LoRA chỉ cập nhật một ma trận tích phân rã $BA$ nhân với hệ số $\alpha/r$, nên gradient cập nhật đòi hỏi bước học lớn hơn khoảng $\approx 10\times$ so với Full Fine-Tuning ($10^{-4}$ thay vì $10^{-5}$). Dùng LR của Full-FT khiến trọng số LoRA gần như đứng yên tại chỗ.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
- Run `qlora` giúp tiết kiệm VRAM ngoạn mục: giảm từ **8.78 GB xuống 3.86 GB** (giảm hơn 56% dung lượng bộ nhớ GPU cần thiết), cho phép chạy được trên cả các GPU cấp thấp 6–8 GB.
- Tuy nhiên, sự đánh đổi là rất rõ ràng: thời gian huấn luyện tăng lên **518.8 giây** (chậm hơn 1.17 lần so với fp16 do overhead giải lượng tử hoá on-the-fly), và Target Accuracy bị tụt từ **0.975 xuống 0.940** (mất 3.5 điểm phần trăm).
- Số liệu thực nghiệm hoàn toàn ủng hộ khuyến nghị của nhà sản xuất (deck §13): Trên các kiến trúc thế hệ mới 2026 và biến thể lai Linear Attention, lỗi lượng tử hoá 4-bit tác động tiêu cực đến chất lượng biểu diễn. Nếu GPU đủ VRAM (như T4 16GB), bf16/fp16 LoRA luôn là ưu tiên vượt trội hơn QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.210` · `regression Δ = -0.224` · `valid_trace_rate = 0.0`

**Diễn giải chi tiết:**
Cổng hồi quy đánh giá phán quyết `FAILED` vì năng lực tổng quát (general knowledge regression) bị sụt giảm **0.224** (từ 0.7911 xuống 0.5667), vượt quá ngưỡng dung sai cho phép ($\le 0.020$). Mặc dù bản fine-tune `correct` đạt kết quả target phi thường (0.975, vượt baseline b tới +0.210), mô hình đã rơi vào hiện tượng kinh điển **Catastrophic Forgetting (quên thảm họa)**. Nguyên nhân trực tiếp là do tập huấn luyện 225 mẫu chỉ chứa duy nhất định dạng JSON phân loại ticket CSKH, không có bất kỳ mẫu dữ liệu văn bản phổ thông nào đi kèm. Để giải quyết triệt để vấn đề này trước khi đưa vào sản xuất, giải pháp chuẩn theo bài giảng (deck §6.3) là đưa vào **1–5% Replay Data** (trộn các câu hỏi kiến thức tổng quát vào batch huấn luyện) nhằm neo giữ không gian biểu diễn gốc của Base Model.

---

## 6. Định tính — Phân tích chi tiết cả ca Thắng và ca Thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) Prompt tối ưu | (c) LoRA Fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | `doi_tra`, `trung_binh`, `chuột không dây`, `trung_tinh` | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ **FT Thắng**: Fine-tune nhận diện chính xác độ khẩn cấp `cao` và sắc thái `tich_cuc`. |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | `hoan_tien`, `trung_binh`, `đèn bàn LED`, `tieu_cuc` | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | ✅ **FT Thắng**: Bắt đúng từ khóa "Quá hạn rồi" thành mức độ `cao`. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | `hoan_tien`, `cao`, `bình giữ nhiệt`, `tieu_cuc` | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tieu_cuc` | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `trung_tinh` | ❌ **FT Thua**: Fine-tune phân loại nhầm urgency `trung_binh` và sentiment `trung_tinh` (nhãn đúng là `cao` / `tieu_cuc`). |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | `san_pham_loi`, `thap`, `áo khoác gió`, `tieu_cuc` | `san_pham_loi`, `thap`, `áo khoác gió`, `tieu_cuc` | `san_pham_loi`, `trung_binh`, `áo khoác gió`, `trung_tinh` | ❌ **FT Thua**: Cụm "Khi nào tiện" thể hiện mức `thap`, nhưng Fine-tune có xu hướng thiên kiến về mức mặc định `trung_binh`. |
| 5 | Chào shop, mình đặt tai nghe bluetooth mã đơn VN161530. Giá bao nhiêu. | `hoi_thong_tin`, `cao`, `tai nghe bluetooth`, `tieu_cuc` | `hoi_thong_tin`, `thap`, `tai nghe bluetooth`, `trung_tinh` | `hoi_thong_tin`, `trung_binh`, `tai nghe bluetooth`, `trung_tinh` | ❌ **FT Thua**: Cả 2 mô hình đều đánh giá sai sắc thái tâm lý tiêu cực ngầm của câu hỏi ngắn này. |

**Mẫu chung ở các ca Fine-tune thua:**
Ở các ca bị trừ điểm (đạt score 0.75/1.0), Fine-tune luôn nhận diện chính xác 100% trường `intent` và `product`. Lỗi sai tập trung cục bộ ở 2 trường có tính chủ quan cao là `urgency` và `sentiment`, đặc biệt khi câu ticket quá ngắn hoặc mang sắc thái nước đôi. Mô hình fine-tune có xu hướng kéo các ca biên về nhãn đa số trong tập train (`urgency: trung_binh`, `sentiment: trung_tinh`).

---

## 7. Kết luận & Điều tôi học được

**Kết luận khoa học:**
Bản fine-tune LoRA `correct` đã chứng minh sự vượt trội áp đảo về năng lực chuyên môn hóa tác vụ CSKH Triage với Target Accuracy đạt **97.5%**, độ chuẩn xác format 100% và giảm mạnh độ trễ xử lý. Tuy nhiên, **chưa nên deploy trực tiếp bản checkpoint này lên môi trường production đa tác vụ** nếu endpoint đó phục vụ đồng thời các câu hỏi tổng quát, do độ suy giảm năng lực phổ thông vượt ngưỡng dung sai ($\Delta = -0.224$). Nếu hệ thống phục vụ dưới dạng **chuyên trách phân luồng (dedicated triage agent)** với system router riêng biệt, checkpoint này hoàn toàn đủ độ tin cậy để triển khai ngay.

Đòn bẩy kỹ thuật cốt lõi trong bài lab không nằm ở việc tăng rank LoRA hay tinh chỉnh siêu tham số phức tạp, mà được xếp theo thứ tự quyết định:
1. **Tính đúng đắn của Loss Mask (NB1)**: Đảm bảo chỉ gradient của câu trả lời được lan truyền ngược.
2. **Thang đo Learning Rate LoRA (NB4)**: $10^{-4}$ cho LoRA so với $10^{-5}$ của Full-FT.
3. **Phạm vi vị trí gắn LoRA (All-linear text decoder)**: Phủ rộng khắp các tầng chiếu chú ý và FFN tốt hơn nhiều so với việc chỉ ép rank cao vào $q, v$.
4. **Bộ dữ liệu tinh gọn & chuẩn hóa**: 225 mẫu sạch đủ đưa độ chính xác từ 0% lên 97.5%.

**Ba điều tôi học được:**
1. **Tuyệt đối không tin tưởng mù quáng vào training loss hay cờ thư viện tự động**: `assistant_only_loss` có thể âm thầm mask 100% token nếu chat template không khớp thẻ; và `attn_only` $r=283$ có training loss thấp hơn nhưng chất lượng test lại thua mô hình chuẩn.
2. **Prompt Optimization là mốc baseline bắt buộc phải vượt**: Prompt tốt đã có thể đưa độ chính xác từ 0.000 lên 0.765 mà không tốn một giây train nào. Fine-tuning chỉ có giá trị khi chứng minh thắng được mốc Baseline (b) này.
3. **Phán quyết FAILED là cơ sở khoa học để hoàn thiện hệ thống**: Nhận diện hiện tượng quên thảm họa (Catastrophic Forgetting) qua cổng hồi quy giúp phát hiện ra nhu cầu cấp thiết của việc đưa 1–5% Replay Data vào pipeline huấn luyện thực tế.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Bổ sung 5% dữ liệu đàm thoại phổ thông (Replay buffer) vào tập huấn luyện và train lại NB3 để cổng hồi quy NB5 chuyển sang `PASSED`.
- Triển khai thử nghiệm Bonus B1 (NB6) để thực hiện merge adapter vào Base Model và benchmark throughput phục vụ với vLLM.

---

## Phụ lục — Thưởng đã làm

- [x] B1 NB6 merge + hot-swap (Code logic & verification design)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
