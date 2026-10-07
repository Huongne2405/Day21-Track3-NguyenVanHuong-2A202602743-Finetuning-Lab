# Reflection — Lab 21

**Họ tên:** Nguyễn Văn Hưởng

**MSSV:** 2A202602743

**Ngày:** 07/10/2026

**1. Điều gì làm bạn ngạc nhiên nhất?**

Kết quả đáng chú ý nhất với tôi là fine-tune đạt target 0,970 so với 0,765 của base model được prompt tối ưu, nhưng vẫn nhận verdict FAILED. Nguyên nhân là regression giảm từ 0,7911 xuống 0,5222. Khi xem đầu ra cụ thể, câu hỏi “Một năm có bao nhiêu tháng?” bị trả thành JSON phân loại ticket thay vì câu trả lời về số tháng. Tôi hiểu rằng học tốt một tác vụ hẹp có thể làm thay đổi hành vi trên đầu vào khác; điểm target cao chưa đủ để quyết định triển khai.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Trong lượt tiếp tục chạy NB3–NB5 được lưu lại, NB5 là bước lâu nhất: 687 giây, so với 474 giây của NB3 và 52 giây của NB4. Con số 52 giây không phải thời gian train ba đối chứng, vì NB4 đã dùng lại các adapter có sẵn. Khi hoàn thiện bài, tôi còn chạy thêm phép tái lập đầu ra để kiểm tra các ví dụ thay vì chỉ dùng bảng điểm tổng hợp. Bài học về kế hoạch thời gian là phải dành riêng thời gian cho đánh giá và đối chiếu bằng chứng, không chỉ dự trù huấn luyện; thời gian từng notebook cần lấy từ log của phiên thực tế.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Quan niệm cần điều chỉnh là “loss thấp hơn hoặc target cao hơn thì model đã tốt hơn”. `attn_only` có train loss trung bình 0,5373, thấp hơn 0,6260 của `correct`, nhưng cả hai cùng đạt target 0,970. Bản `correct` cũng thắng baseline trên target mà thua rõ ở regression. Sau lab, tôi đánh giá lợi ích của fine-tuning theo yêu cầu sử dụng và nhiều nhóm chỉ số, thay vì lấy một con số làm kết luận chung về model.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc README và rubric, hướng dẫn chạy Colab T4, giải thích mask proof và baseline, kiểm tra các artefact, viết script lấy dự đoán đầy đủ và hỗ trợ cấu trúc report. Các con số cuối cùng được đối chiếu với `results/`, không lấy từ suy đoán của AI.

Điểm hướng dẫn ban đầu cần điều chỉnh là cách xử lý yêu cầu hai ca FT thua: phải kiểm tra chúng có tồn tại trong tập target hay không trước khi chọn ví dụ. Kết quả đối chiếu đầy đủ cho thấy FT thắng 33 ticket, hòa 17 và không thua ticket nào. Vì vậy, tôi không dùng các ca FT sai urgency nhưng hòa baseline để gọi là ca thua; hai ca thua được phân tích thuộc nhóm regression và được ghi rõ. Ước lượng thời gian dựa trên README cũng cần kiểm tra lại bằng log thực tế của Colab, nhất là khi notebook tiếp tục chạy và bỏ qua adapter đã lưu.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ xác định nhiệm vụ, định dạng đầu ra và những hành vi model phải giữ được, rồi xây dựng tập đánh giá độc lập với train. Sau đó đo base model với một prompt đủ tốt và đóng băng mốc trước khi train. Nếu prompt đã đáp ứng yêu cầu, tôi cần cân nhắc lợi ích thực tế của fine-tuning trước khi đầu tư thêm. Nếu vẫn cần train, tôi sẽ kiểm chứng template và mask, rồi đánh giá cả tác vụ chính, regression, format và latency. Với kết quả lab này, một hướng thử tiếp là trộn dữ liệu phổ thông tách biệt khỏi eval vào một run mới để kiểm tra khả năng giảm lỗi trả lời JSON triage cho câu hỏi phổ thông; tôi chưa coi đó là giải pháp đã được chứng minh.
