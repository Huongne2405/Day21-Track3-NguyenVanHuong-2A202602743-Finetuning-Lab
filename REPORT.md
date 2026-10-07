# Lab 21 — LoRA cải thiện phân loại ticket nhưng không vượt cổng hồi quy

**Họ tên:** Nguyễn Văn Hưởng

**MSSV:** 2A202602743

**Ngày thực hiện:** 07/10/2026

**Tier:** T4

**Base model:** `unsloth/Qwen3.5-4B`

**GPU thực tế:** Tesla T4, CUDA sm_75, 14,6 GB bộ nhớ được báo cáo

**Precision thực tế:** fp16

## 1. Câu hỏi và thiết kế thí nghiệm

Thí nghiệm kiểm tra liệu LoRA có cải thiện tác vụ ticket chăm sóc khách hàng tiếng Việt → JSON so với chính base model được hướng dẫn bằng prompt tối ưu hay không, đồng thời kiểm tra tác động lên năng lực phổ thông. Câu hỏi phụ là vị trí adapter, learning rate và lượng tử hóa ảnh hưởng thế nào khi giữ ngân sách huấn luyện tương đương.

Thí nghiệm dùng model và corpus mặc định để phù hợp với Colab T4 và giữ nguyên khung đánh giá của lab. Model gốc và bản fine-tune cùng sử dụng `unsloth/Qwen3.5-4B`. Corpus gồm 250 mẫu với đầu ra có bốn trường `intent`, `urgency`, `product`, `sentiment`. Bài toán có nhãn cụ thể và có thể chấm theo từng trường, nên không cần dùng LLM judge.

| Thành phần | Cấu hình thực tế |
|---|---|
| Train / validation | 225 / 25 mẫu, split seed 42 |
| Target / regression eval | 50 ticket / 15 câu hỏi phổ thông |
| Loss mask | `assistant-only` |
| Epochs / ngân sách step | 2 / 30 step cho cả bốn run |
| Batch mỗi thiết bị / gradient accumulation | 1 / 16; batch hiệu dụng 16 |
| Rank / alpha của `correct` | 16 / 32 |
| Learning rate của `correct` | `1e-4` |
| Packing / padding-free | Đều tắt |
| Sinh văn bản | Greedy, batch 4; target tối đa 160 token, regression tối đa 96 token |
| Chạy rút gọn | Không; `EVAL_LIMIT` không đặt, `smoke_mode=false` |

NB1 và NB2 chạy trước NB3. Baseline (b) được lưu trước khi train; SHA của prompt tối ưu là `719e74d3b6232053`. Verify xác nhận prompt (b) và tập eval không bị sửa. NB4 trong lần chạy được gửi dùng lại ba adapter có sẵn; verify xác nhận cả bốn run đều ghi nhận 30 step. Thời gian NB4 là 52 giây trong lần tiếp tục chạy, không phải thời gian train ba đối chứng. CSV lưu hai dòng `correct`; bảng kết quả sử dụng dòng cuối cùng (418,2 giây, loss 0,6260), không trộn với lần trước (429,2 giây, loss 0,6269). Giữ nguyên các dòng CSV để bảo toàn lịch sử.

## 2. Bằng chứng template, mask và độ dài

Phép thử template cho kết quả `reasoning preserved — safe to train on traces`: nội dung suy luận thử nghiệm trong `<think>` được giữ sau khi render. Đây là phép thử khả năng giữ trace; corpus mặc định vẫn có câu trả lời JSON, không phải corpus suy luận có trace.

| Mask proof trên mẫu NB1 | Kết quả |
|---|---:|
| Tổng token / token tính loss | 94 / 39 |
| `supervised_fraction` | 0,4149 |
| `answer_is_supervised` | `true` |
| `question_is_masked` | `true` |

Đoạn được giải mã từ các token tính loss trong log:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Chuỗi được giám sát có câu trả lời JSON và token theo template, còn nội dung ticket không nằm trong phần này. Phép minh họa `everything` tính loss trên toàn bộ 94/94 token và chứa cả ticket; đó không phải cấu hình train. Trên 225 mẫu train ở NB3, 9.014/20.951 token được giám sát, tương đương khoảng 43,0%; thống kê toàn tập này khác với tỷ lệ trên một mẫu NB1.

| Độ dài trên 250 mẫu | Token |
|---|---:|
| Mean | 93,1 |
| p50 / p95 / p99 | 93 / 98 / 100 |
| Max | 101 |
| Giới hạn NB1 gợi ý | 256 |
| Giới hạn thực tế của tier | 1024 |

Thí nghiệm giữ `max_length=1024` mặc định của tier T4 xuyên suốt các run. Lựa chọn này khác với gợi ý 256 từ số đo và chưa được xem là tối ưu. Mẫu dài nhất chỉ có 101 token nên cả hai giới hạn đều đủ chứa corpus hiện tại. Trần 1024 không tự chứng minh mỗi mẫu được padding đến 1024. Một thí nghiệm tiếp theo có thể kiểm tra cấu hình 256, nhưng không thay đổi riêng một run của phép so sánh hiện tại.

## 3. Baseline đóng băng và kết quả bốn nhóm

| Run | Target | Regression | Format | Latency báo cáo, ms/mẫu |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3273,7 |
| (b) Base + optimized prompt | 0,7650 | 0,7911 | 1,0000 | 1067,1 |
| (c) LoRA `correct` + naive prompt | 0,9700 | 0,5222 | 1,0000 | 1393,0 |

Baseline (b) mạnh hơn (a) trên target và format, đáp ứng yêu cầu so với base model đã được prompt tử tế. Không sửa prompt tối ưu; kiểm tra SHA của verify đã qua. So với (b), fine-tune tăng target 0,205, tức 20,5 điểm phần trăm, nhưng giảm regression khoảng 0,2689, tức 26,89 điểm phần trăm. Format giữ nguyên và latency báo cáo tăng khoảng 30,5%.

Target là trung bình độ chính xác theo bốn trường, không phải tỷ lệ ticket đúng toàn bộ bốn trường. Regression là keyword recall của phép chấm trong lab, không bao quát mọi năng lực phổ thông. Format dùng parser có thể trích object JSON từ đầu ra; điểm 1,0 xác nhận đủ bốn khóa theo scorer, chưa chứng minh đầu ra chỉ chứa JSON hay không có khóa thừa. Latency là thời gian sinh chia cho số mẫu trong batch, không phải độ trễ một yêu cầu riêng lẻ khi triển khai.

Target và format bằng 0 ở (a) chưa đủ để kết luận model không hiểu ticket; cần xem đầu ra để phân biệt lỗi định dạng, bị cắt và phân loại sai. Baseline (b) cho thấy prompt đã cải thiện đáng kể trước khi thay đổi trọng số.

## 4. Đối chứng: vị trí, learning rate và QLoRA

| Run | Vị trí | Rank | Trainable | LR | Train loss trung bình | Target | Train, giây | Peak VRAM, GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32464896 | 0,0001 | 0,6260 | 0,9700 | 418,2 | 8,78 |
| `attn_only` | q,v | 283 | 32456704 | 0,0001 | 0,5373 | 0,9700 | 271,8 | 8,79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 0,00001 | 1,5702 | 0,0000 | 405,3 | 8,78 |
| `qlora` | text-linear | 16 | 32464896 | 0,0001 | 0,7058 | 0,9400 | 472,9 | 3,86 |

| Run | Regression | Format | Latency báo cáo, ms/mẫu |
|---|---|---:|---:|
| `correct` | 0,5222 | 1,0000 | 1393,0 |
| `attn_only` | Không đo trong NB5 | 1,0000 | 895,3 |
| `wrong_lr` | Không đo trong NB5 | 0,0000 | 5313,6 |
| `qlora` | Không đo trong NB5 | 1,0000 | 1820,9 |

Cột `final_loss` trong CSV được code lấy từ `result.training_loss`, tức loss trung bình toàn quá trình train. Không diễn giải 0,6260 là loss riêng bước cuối. Xếp hạng target là `correct = attn_only > qlora > wrong_lr`; xếp hạng theo train loss thấp hơn là `attn_only > correct > qlora > wrong_lr`. Loss phân biệt hai run đầu, còn target không phân biệt chúng.

### 4.1. Vị trí so với rank

`attn_only` và `correct` đều đạt target 0,9700 trên cùng 50 ticket. Để giữ ngân sách tham số tương đương, `attn_only` tăng rank lên 283; chênh lệch trainable chỉ 8.192 tham số, khoảng 0,0252%. Đây là đối chứng cho cấu hình vị trí với ngân sách đã khớp, không phải quét rank độc lập tại một vị trí cố định. Loss thấp hơn của `attn_only` không tạo ra target cao hơn, nên không thể kết luận nó phân loại tốt hơn. Gắn adapter vào text-linear không thể hiện ưu thế target trong phép đo này, nhưng chưa có cơ sở khái quát sang dataset khác hoặc mọi rank. NB5 không đo regression cho `attn_only`, nên chưa thể khuyến nghị triển khai chỉ dựa vào target và latency.

### 4.2. Learning rate

`wrong_lr` giảm LR từ `1e-4` xuống `1e-5`, giữ vị trí, rank, precision và ngân sách 30 step. Train loss trung bình của nó cao hơn, 1,5702 so với 0,6260, còn target và format đều bằng 0. Kết quả cho thấy LR thấp hơn hoạt động kém trong ngân sách này, không chứng minh nó không bao giờ học được nếu tăng ngân sách. Log được gửi không có đường loss đầy đủ của `wrong_lr`, nên không mô tả đường đó là phẳng hoặc so từng step. Riêng `correct` có loss logging giảm từ 2,163 xuống 0,0262 ở mốc cuối. Nếu chỉ nhìn loss cao mà không biết LR, có thể quy nguyên nhân sai cho dữ liệu hoặc thiếu rank; ở đây biến thay đổi được kiểm soát là LR. Cần đầu ra cụ thể của `wrong_lr` để phân biệt lỗi định dạng và lỗi phân loại.

### 4.3. QLoRA

Peak VRAM giảm từ 8,78 xuống 3,86 GB, tiết kiệm 4,92 GB, khoảng 56,0%. Đổi lại, target giảm 3 điểm phần trăm; thời gian train tăng từ 418,2 lên 472,9 giây, khoảng 13,1%; latency báo cáo tăng khoảng 30,7%. Format vẫn đạt 1,0. NB5 chấm adapter QLoRA với base 4-bit theo đúng cấu hình train. Số đo ủng hộ ưu tiên LoRA 16-bit cho thí nghiệm này khi T4 đủ bộ nhớ, nhưng không bác bỏ QLoRA nói chung: giảm hơn một nửa bộ nhớ vẫn có giá trị khi tài nguyên hạn chế. Chưa đo regression của QLoRA nên chưa biết nó giữ năng lực phổ thông tốt hơn hay kém hơn `correct`.

## 5. Phán quyết và giới hạn của bằng chứng

**Phán quyết: FAILED.**

**Target Δ:** +0,2050.

**Regression Δ:** khoảng −0,2689.

**Ngưỡng giảm regression cho phép:** 0,0200.

**`valid_trace_rate`:** 0,0.

Bản fine-tune vượt baseline tối ưu trên target, nhưng không giữ được điểm regression trong ngưỡng của lab. Mức giảm quan sát lớn hơn nhiều so với giới hạn 0,02, vì vậy cổng đưa ra FAILED dù điểm phân loại đạt 0,97. Kết quả cho thấy đánh giá chỉ dựa vào train loss hoặc target sẽ bỏ sót một đánh đổi quan trọng. Một giả thuyết phù hợp là adapter học mạnh hành vi phân loại từ corpus hẹp, làm thay đổi hành vi trả lời các chỉ dẫn phổ thông; tuy nhiên keyword recall không đủ để xác định cơ chế gây giảm điểm. Đầu ra bổ sung ở mục 6 cho thấy một số câu hỏi phổ thông bị chuyển thành JSON triage, không trả lời thông tin được yêu cầu; các ca khác vẫn cần phân biệt sai kiến thức, thiếu từ khóa và đầu ra bị cắt. Chưa có đủ bằng chứng khuyến nghị dùng adapter cho một trợ lý đa nhiệm. Cũng không thể tuyên bố nó chắc chắn không phù hợp cho tuyến phân loại chuyên biệt, vì quyết định đó cần yêu cầu triển khai và đánh giá thêm.

`valid_trace_rate=0` không tự chứng minh reasoning-trace collapse: corpus có câu trả lời JSON và giao thức sinh đặt `enable_thinking=False`. Chưa train hai chế độ mask trên corpus có trace nên chưa thực hiện đối chứng reasoning-collapse.

Log NB3 có `grad_norm=nan` ở hai mốc logging. Loss vẫn hữu hạn và adapter được lưu, đánh giá thành công, nhưng log chưa đủ để xác định nguyên nhân hoặc số cập nhật có thể bị ảnh hưởng. Đây là hạn chế cần ghi nhận, không mặc định mọi cập nhật số học đều ổn định.

## 6. Ví dụ định tính và đối chiếu baseline

<!-- qualitative:start -->
Để lấy đầu ra từng mẫu, model được sinh lại sau train theo đúng giao thức đã đóng băng: cùng model, adapter, precision fp16, prompt, eval, batch và greedy decode. Đây là phép tái lập bổ sung, không thay thế baseline đo trước train. SHA-256 của các artefact gốc và tập eval đều được đối chiếu; không file gốc nào bị ghi đè.

Cả sáu điểm target, format, regression của baseline (b) và FT tái lập đều khớp artefact gốc trong ngưỡng làm tròn 0,0001. Dự đoán đầy đủ của 50 ticket và 15 câu hỏi regression được lưu trong `results/qualitative_comparison.json`.

### 6.1. Năm ví dụ target, có cả ca FT trả lời sai

Trên toàn bộ 50 ticket, FT thắng 33 ca, hòa 17 ca và **không có ca thua** baseline (b) theo điểm từng trường. Sáu ca FT chưa đúng hoàn toàn đều sai urgency; bốn ca hòa baseline và hai ca vẫn thắng baseline. Không thể chọn hai ca FT thua ở target từ tập này. Bảng sau gồm hai ca thắng rõ, hai ca cùng sai urgency và một ca FT thắng dù vẫn sai urgency, thay vì chỉ chọn ca đúng hoàn toàn.

| Index target, từ 0 | Ticket | Nhãn đúng | Baseline (b) | Fine-tune | Điểm b / FT | Phân tích |
|---|---|---|---|---|---|---|
| 6 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây. | {"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"} | 0.50 / 1.00 | FT thắng; sửa intent và urgency. |
| 7 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình. | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | {"intent": "van_chuyen", "urgency": "cao", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | 0.50 / 1.00 | FT thắng; sửa intent và urgency. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | {"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | 0.75 / 0.75 | Hòa; cả hai dự đoán urgency=trung_binh thay vì thap. |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | {"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"} | {"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"} | 0.50 / 0.75 | FT thắng nhưng vẫn sai urgency; baseline còn sai intent. |
| 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | {"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"} | 0.75 / 0.75 | Hòa; cả hai cùng sai urgency. |

Tổng hợp theo trường từ đầu ra bổ sung:

| Trường | Baseline (b) | FT |
|---|---:|---:|
| intent | 0,54 | 1,00 |
| urgency | 0,64 | 0,88 |
| product | 1,00 | 1,00 |
| sentiment | 0,88 | 1,00 |

Các ca FT sai urgency có index 3, 5, 12, 39, 41, 46. Cả sáu ticket chứa “Khi nào tiện”, nhãn urgency là `thap`, nhưng FT dự đoán `trung_binh`. Mẫu lỗi này gợi ý kiểm tra độ nhạy với tín hiệu thời gian hoặc phân bố urgency trong corpus; chưa có thí nghiệm can thiệp để xác định nguyên nhân. Baseline cũng sai urgency ở sáu ca này, nên không gọi chúng là ca FT thua.

### 6.2. Hai ca FT thua rõ ở regression

Để thể hiện cả mặt suy giảm, dưới đây là hai ca FT thua baseline trong nhóm regression. Đây không phải hai ticket target; chúng bổ sung bằng chứng cho verdict FAILED, không được trình bày như đã đáp ứng yêu cầu hai ca thua target của bảng mẫu rubric.

**Regression index 2: 1 km bằng bao nhiêu mét?**

Từ khóa của scorer: `1000`. Baseline có keyword recall 1.00; FT có 0.00.

Đầu ra baseline (b):

```text
Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.

Kết quả: **1 km = 1000 m**.
```

Đầu ra FT:

```json
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "intent_confidence": 0.98, "urgency_confidence": 0.95, "product_confidence": 0.0, "sentiment": "trung_tinh", "sentiment_confidence": 0.92}
```

**Regression index 9: Một năm có bao nhiêu tháng?**

Từ khóa của scorer: `12`. Baseline có keyword recall 1.00; FT có 0.00.

Đầu ra baseline (b):

```text
Một năm bình thường có **12 tháng**.

Cụ thể, các tháng trong năm là:
1. Tháng 1
2. Tháng 2
3. Tháng 3
4. Tháng 4
5. Tháng 5
6. Tháng 6
7. Tháng 7
8. Tháng 8
9. Tháng 9
10. Tháng 10
11. Tháng 11
12. Tháng
```

Đầu ra FT:

```json
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```

Trong cả hai ca, baseline trả lời đúng thông tin được yêu cầu, còn FT phát ra object phân loại ticket và không trả lời số đo/số tháng. Các câu regression được sinh không có system prompt triage, nên đây là bằng chứng hành vi phân loại lan sang đầu vào phổ thông trong giao thức đang đo. Nó hỗ trợ giả thuyết chuyên biệt hóa hành vi; chưa chứng minh kiến thức tương ứng đã bị xóa khỏi trọng số.

Trên 15 câu regression tái lập, FT thua 7 ca, thắng 1 ca và hòa 7 ca theo keyword recall. Scorer có giới hạn: chẳng hạn câu giải thích tục ngữ có thể đúng ý nhưng thiếu từ khóa; điểm keyword recall không đồng nghĩa chấm ngữ nghĩa toàn diện. Không sửa scorer hoặc eval sau khi quan sát kết quả.
<!-- qualitative:end -->

## 7. Kết luận và bài học

Thí nghiệm cung cấp bằng chứng rằng LoRA đã học tác vụ phân loại ticket trong giao thức của lab: target tăng từ 0,765 của base model được prompt tối ưu lên 0,970, còn format giữ ở 1,0. Tuy nhiên, nếu chỉ dùng hai chỉ số đó để chọn model, kết luận triển khai sẽ bỏ sót mức giảm regression từ 0,7911 xuống 0,5222. Vì cổng yêu cầu vừa cải thiện target vừa giữ năng lực phổ thông, kết quả FAILED phù hợp với số đo. Tôi chưa chọn adapter `correct` làm trợ lý đa nhiệm thay cho base model. Baseline (b) vẫn có bằng chứng tốt hơn về sự cân bằng trong thí nghiệm này. Về các đòn bẩy, đối chứng LR cho thấy cùng ngân sách 30 step nhưng LR thấp hơn không đạt mục tiêu đầu ra; đối chứng vị trí cho thấy `attn_only` ngân sách tương đương hòa target với `correct`, dù train loss thấp hơn. QLoRA tiết kiệm bộ nhớ rõ rệt nhưng đánh đổi target và thời gian. Các kết quả không cho phép kết luận rank hoặc vị trí luôn thắng trong mọi bài toán. Mask đúng là điều kiện để số đo có ý nghĩa, còn corpus hẹp là giả thuyết cần kiểm tra khi tìm cách giảm regression. Bằng chứng từng mẫu đã được bổ sung; bước tiếp theo nên thực hiện thí nghiệm có kiểm soát, giữ nguyên mốc đánh giá và lưu riêng kết quả mới.

Ba bài học rút ra trực tiếp từ số đo:

1. Prompt tối ưu đạt target 0,765 trước khi train, trong khi prompt đơn giản đạt 0; mốc so sánh mạnh quyết định cách diễn giải lợi ích của fine-tuning.
2. Train loss thấp hơn không đủ để xếp hạng chất lượng: `attn_only` có loss 0,5373 so với 0,6260 của `correct`, nhưng target đều là 0,970.
3. Cải thiện tác vụ không đồng nghĩa giữ năng lực khác: target tăng 20,5 điểm phần trăm nhưng regression giảm 26,89 điểm phần trăm.

<!-- reflection:start -->
Từ thí nghiệm này, tôi rút ra rằng một bảng target tốt phải được đọc cùng bảng regression. Mức tăng target 20,5 điểm phần trăm không đủ để chọn model khi năng lực khác giảm mạnh. Tôi cũng cần đọc đúng ý nghĩa chỉ số: loss trung bình không phải loss bước cuối, và điểm theo trường không phải tỷ lệ ticket đúng hoàn toàn. Trong lần thực hành này, AI assistant hỗ trợ đọc hướng dẫn, giải thích log, kiểm tra hồ sơ và soạn phân tích; số liệu vẫn phải lấy từ artefact thật, còn các ca thua phải được đối chiếu từng mẫu.

Nếu có thêm hai giờ, phương án thử tiếp là một run riêng với 1–5% dữ liệu phổ thông trộn vào train theo gợi ý NB5, giữ nguyên tập eval và prompt (b), rồi đo lại đủ bốn nhóm. Đây là phương án đề xuất, chưa được thực hiện và chưa có kết quả để khẳng định sẽ cải thiện regression. Các mẫu phổ thông dùng để train phải tách khỏi regression eval để tránh nhiễm đánh giá. Kết quả hiện tại được giữ nguyên để so sánh với thí nghiệm tiếp theo.
<!-- reflection:end -->

## 8. Kiểm tra và nguồn đối chiếu

Log verify được gửi ghi nhận 119 unit test passed và tổng gatekeeper 25 passed, 1 warning, 1 failure. Failure tại thời điểm đó là report còn placeholder; warning giải thích verdict FAILED vẫn chấm được. Verify xác nhận các artefact chính tồn tại, mask đúng, eval đầy đủ, prompt và eval không đổi, đủ bốn run, cùng 30 step và ngân sách `attn_only` khớp. Sau khi hoàn thiện report, `make verify` trên bản hồ sơ tại máy cá nhân đã trả về **26 passed, 1 warning, 0 failures**, exit code 0. Unit test tại máy cá nhân: 116 passed, 3 skipped vì chưa cài PyTorch; log phiên Colab đã có 119 passed. Warning còn lại giải thích verdict FAILED vẫn chấm được, không phải lỗi hồ sơ. Gatekeeper xác nhận artefact và tính liêm chính; không tự bảo đảm đạt đủ mọi mục chấm nội dung trong rubric.

Số liệu đã được đối chiếu trực tiếp với các file gốc lấy từ `lab21_backup.zip`: `results/mask_proof.json`, `template_check.json`, `token_stats.json`, `baselines_frozen.json`, `runs.csv`, `verdict.json`, `autopsy.json`, `qualitative.json`; đầu ra tái lập ở `qualitative_comparison.json`. Log NB1–NB5 và verify bổ sung ngữ cảnh về thứ tự chạy và cấu hình. Không tái tạo các file kết quả gốc bằng cách điền số từ log. Chưa có bằng chứng đã làm NB6, dataset riêng, quét rank, reasoning-collapse hoặc upload adapter lên Hugging Face Hub nên không yêu cầu điểm thưởng cho các mục này.
