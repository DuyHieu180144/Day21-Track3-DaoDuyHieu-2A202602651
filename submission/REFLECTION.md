# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng ở NB4: mô hình `attn_only` với rank được nâng lên cực đại ($r=283$) có `final_loss` huấn luyện thấp hơn hẳn mô hình `correct` (0.5361 so với 0.6266), nhưng khi đánh giá thực tế trên tập target thì lại thua. Nó chứng minh rõ ràng rằng rank lớn trên dữ liệu hẹp chỉ giúp mô hình "học vẹt" ghi nhớ dữ liệu tốt hơn chứ không mang lại khả năng khái quát hóa cao như việc gắn adapter phủ khắp các tầng `text-linear`.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu sinh văn bản đánh giá (evaluation generation) qua 3 mốc (Baseline A, Baseline B, và 4 bản checkpoint ở NB5). Ban đầu tôi cứ nghĩ khâu huấn luyện gradient descent (NB3/NB4) sẽ tốn thời gian nhất, nhưng thực tế việc chạy suy luận autoregressive lặp đi lặp lại trên GPU chiếm phần lớn thời lượng pipeline.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin rằng:
- Cứ tăng rank $r$ của LoRA lên càng cao (64, 128, 256) thì mô hình sẽ càng thông minh và đạt kết quả càng tốt.
- Cứ train loss giảm sâu về gần 0 là mô hình đã fine-tune thành công.
Giờ tôi nhận ra vị trí đặt adapter và learning rate đúng thang quan trọng hơn rank rất nhiều; và chỉ số duy nhất có giá trị là điểm đo lường trực tiếp trên tác vụ nghiệp vụ mục tiêu chứ không phải loss hay perplexity.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để lập kế hoạch tổng thể, giải thích kiến trúc các tầng Linear Attention / Gated DeltaNet của Qwen3.5, kiểm tra logic giải mã Loss Mask và soạn thảo khung báo cáo. Điểm AI hay mắc sai lầm nếu không kiểm tra kỹ là thói quen giả định GPU nào cũng chạy được `bf16=True` hoặc tự ý nới lỏng các điều kiện cổng kiểm tra khi thấy kết quả báo FAILED thay vì đào sâu tìm hiểu nguyên nhân Catastrophic Forgetting.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi sẽ làm là: **Xây dựng bộ dữ liệu đánh giá chuẩn (Golden Eval Set) và đo Baseline của mô hình gốc với Prompt Engineering tối ưu trước.** Nếu một prompt tốt đã giải quyết được 80–90% bài toán với chi phí và độ trễ chấp nhận được, tôi sẽ tư vấn khách hàng chưa cần fine-tune. Nếu bắt buộc fine-tune, tôi sẽ dùng chính mốc baseline đó làm cổng chặn nghiệm thu và luôn trộn 3–5% replay data để bảo vệ năng lực tổng quát của mô hình.
