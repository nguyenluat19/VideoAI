# Video Template SaaS

Nền tảng SaaS tạo video từ template động: render nhanh bằng FFmpeg, render nâng cao bằng MimicMotion (GPU),
ví credit tự trừ/hoàn, nạp tiền tự động 100% qua VietQR + webhook SePay/Casso, AI Inkstone (14 model) để viết
kịch bản / phân tích ảnh / chọn luồng render. Chạy local, dữ liệu trong SQLite.

## Chạy nhanh

```bash
git clone -b claude/wizardly-lamport-qfjxhi https://github.com/nguyenluat1905/dashboard
cd dashboard
pip install -r requirements.txt
python run.py
```

`run.py` tự tạo `.env` (sinh `SECRET_KEY` và mật khẩu admin, in ra màn hình), kiểm tra FFmpeg rồi mở
**http://127.0.0.1:8000**. Việc duy nhất bạn phải làm: mở `.env`, điền

```
INKSTONE_BASE_URL=<địa chỉ API Inkstone, dạng https://.../v1>
INKSTONE_API_KEY=<key chính của bạn>
```

rồi chạy lại `python run.py`. Một key chính dùng cho cả 14 model; tên model đã có sẵn trong code.
Chưa điền thì render và nạp tiền vẫn chạy, chỉ phần AI tắt (tự chọn luồng sẽ dùng quy tắc dự phòng).

Cần cài sẵn **FFmpeg** (`brew install ffmpeg` / `apt install ffmpeg` / ffmpeg.org cho Windows).

## Model AI (Intern Inkstone)

**Đa phương thức (ảnh + tool calling) — dùng cho phân tích ảnh**

| Model ID | Tên | Vai trò |
|---|---|---|
| `deepseek-v4-flash-vision` | DeepSeek-V4-Flash-Vision | Ngữ cảnh 1M, tool calling, tư duy sâu; phân tích ảnh gameplay/nội dung thị giác |
| `kimi-k2.6` | Kimi-K2.6 | Ngữ cảnh 256K, text+ảnh, tool calling, tư duy sâu; tổng hợp kịch bản |
| `qwen3.8-27b` | Qwen3.8-27B | Đa phương thức text+ảnh, tool calling, tư duy sâu |
| `Agents-A1` | Agents-A1 | Đa modal, tool calling, suy luận khoa học |
| `Sage-A1.5` | Sage-A1.5 | Đa modal, tool calling, suy luận khoa học |
| `intern-s2` | 书生-S2 | Đa phương thức chuyên xử lý không gian hình ảnh và văn bản |
| `deepseek-v4.1-flash-1017` | DeepSeek-V4.1-Flash-1017 | Đa modal tốc độ cao, ngữ cảnh lớn |

**LLM / lập luận — dùng cho viết kịch bản**

| Model ID | Tên | Vai trò |
|---|---|---|
| `deepseek-v4-pro-0813` | DeepSeek-V4-Pro-0813 | Ngữ cảnh 1M; sinh prompt, lời bình, kịch bản video |
| `deepseek-v4-flash-0731` | DeepSeek-V4-Flash-0731 | Ngữ cảnh 1M; sinh prompt, lời bình, kịch bản video (nhanh) |
| `glm-5.3` | GLM-5.3 | Tư duy sâu, tool calling, tối ưu workflow |
| `minimax-m3` | MiniMax-M3 | Tư duy sâu, tool calling, tối ưu workflow |
| `Atria-Dawn` | Atria-Dawn | Suy luận logic, tính toán kịch bản phức tạp |
| `Atria-Dawn-Preview` | Atria-Dawn-Preview | Suy luận logic, tính toán kịch bản phức tạp (preview) |

**Ra quyết định cấu trúc — tự chọn luồng render**

| Model ID | Tên | Vai trò |
|---|---|---|
| `intern-decision-4b` | 书生-Decision-4B | Quyết định cấu trúc (SystemOne): chọn luồng render nhanh hay nâng cao |

- Gọi `INKSTONE_BASE_URL/chat/completions` (chuẩn OpenAI) với `Authorization: Bearer <key>`. Nếu Inkstone dùng
  đường dẫn khác, sửa hàm `chat()` trong `app/ai.py`.
- Model được chọn lỗi → tự thử tối đa 2 model khác cùng nhóm; kết quả cho biết model nào đã trả lời.
- Mặc định: kịch bản `deepseek-v4-pro-0813`, ảnh `deepseek-v4-flash-vision`, điều hướng `intern-decision-4b`
  (sửa ở cuối `app/config.py`). Danh sách + trạng thái cấu hình: `GET /api/ai/models`.

## Video AI: MiniMax-H3 (+ MimicMotion cho video nhảy)

Điền `MINIMAX_API_KEY` trong `.env` (key riêng của MiniMax, khác key Inkstone). Hai template dựng sẵn xuất hiện
trong Thư viện:

| Template | Việc xảy ra | Credit |
|---|---|---|
| **Video AI từ mô tả** | MiniMax-H3 tạo toàn bộ video từ mô tả (+ ảnh khung hình đầu tuỳ chọn), 4–15 giây, tỉ lệ 9:16/16:9/…, 768P/2K | giây × 4 (2K × 1,625) |
| **Nhảy theo ảnh nhân vật** | (1) H3 tạo video nhảy mẫu — prompt tự được tối ưu cho toàn thân, nền trơn, camera tĩnh, **để MimicMotion bắt chuyển động tốt**; (2) MimicMotion chép chuyển động sang ảnh nhân vật; (3) nhạc H3 sinh ra được ghép lại vào kết quả | giây × 4 + 10, tối đa 10 giây |
| ↳ nếu bạn **upload video nhảy của riêng mình** | bỏ qua H3, chỉ chạy MimicMotion | 10 (rẻ nhất) |

Giá gốc MiniMax-H3 (trang [Pricing](https://platform.minimax.io/docs/pricing/overview)): 0,08 USD/giây ở 768P, 0,13 USD/giây ở 2K; mặc định tool tính 4 credit/giây (≈ gấp đôi giá gốc nếu 1 credit = 1.000đ), chỉnh bằng `AI_VIDEO_CREDITS_PER_SEC`.

Hai giai đoạn của "nhảy" chạy trên hai hàng đợi (`ai`: nhiều job song song vì chỉ chờ API; `gpu`: một job một lúc)
nên GPU không bị giữ trong lúc chờ API. Giai đoạn nào lỗi thì **hoàn toàn bộ** credit (cả phần API lẫn GPU). Không có
Redis: video AI thường vẫn chạy ngay trong ứng dụng, còn "nhảy" (cần hàng đợi GPU) báo lỗi 503 và không trừ tiền.
Mọi video AI cũng vào hàng chờ duyệt của admin như video khác, và người dùng sau đó sửa được prompt/thời lượng/ảnh.

> **Lưu ý trung thực:** địa chỉ `POST /v2/video_generation`, tham số `content/resolution/duration/ratio` và việc đọc kết
> quả từ `content.url` tôi lấy từ tài liệu công khai của MiniMax, nhưng đường dẫn *tra cứu task* giữa các bản tài liệu
> không thống nhất nên client tự thử vài đường dẫn phổ biến. Chưa gọi được API thật (không có key). Hãy chạy
> `python scripts/check_minimax.py --dry-run` rồi `python scripts/check_minimax.py` (tạo 1 clip 4 giây, tính phí) để xác
> nhận; nếu lệch, chỉ cần sửa `MINIMAX_QUERY_PATH` / `MINIMAX_EXTRA_PARAMS` trong `.env`.

## Bảng giá (admin)

Tab **Bảng giá** (chỉ admin) là nguồn giá duy nhất của cả hệ thống, lưu trong cơ sở dữ liệu. Mỗi dòng: giá (credit),
bật/tắt, giá mặc định, giá vốn, biên lợi nhuận, nút Lưu / Mặc định. Có 4 nhóm:

| Nhóm | Mục | Đơn vị |
|---|---|---|
| Render | Render nhanh (FFmpeg); MimicMotion | credit / video |
| Model tạo video | MiniMax-H3 768P; MiniMax-H3 2K | credit / **giây** |
| Model Inkstone (14 model) | từng model: hình ảnh, ngôn ngữ, điều hướng | credit / lần gọi (mặc định 0 = miễn phí) |

- Đồng bộ ngay: form tạo video, danh sách model ở tab Trợ lý AI, mức trừ credit đều đọc từ bảng này. Video nhảy =
  giá H3 × số giây + giá MimicMotion. Chọn "tự động" mà AI điều hướng được gọi thì cộng thêm giá của
  `intern-decision-4b`.
- Giá mới chỉ áp cho lượt **sau đó**; job đã tạo giữ giá đã trừ, lỗi thì hoàn đúng số đó (đổi giá không làm sai tiền).
- Model Inkstone: trả giá của model bạn chọn; không có câu trả lời (mọi model lỗi) thì hoàn lại. Tắt "bật" thì model đó
  không dùng được và không bao giờ được chọn làm phương án dự phòng.
- Biên lợi nhuận tính từ **tỉ giá** và **giá trị 1 credit** (đặt ngay trên tab, cùng chỗ với mức nạp tối thiểu), chỉ hiện khi biết giá vốn nhà cung cấp
  (hiện có cho MiniMax-H3: $0,08/s ở 768P, $0,13/s ở 2K; MimicMotion chạy GPU của bạn nên không có giá vốn).
- Lịch sử thay đổi (ai, lúc nào, giá cũ → mới) hiện ở cuối tab. Các biến `FAST_RENDER_COST`, `HEAVY_RENDER_COST`,
  `AI_VIDEO_CREDITS_PER_SEC`, `AI_VIDEO_2K_MULTIPLIER` trong `.env` chỉ là giá khởi tạo lần đầu; sau đó bảng giá quyết định.

## Luồng nghiệp vụ

**Credit.** Trước khi render, kiểm tra số dư (thiếu → 402 "vui lòng nạp thêm"); đủ thì trừ bằng một lệnh UPDATE
nguyên tử (không bao giờ âm dù gửi song song). Render lỗi ở bất kỳ bước nào → hoàn đúng số credit, tối đa một lần.
Job bị kẹt (server sập, worker mất, Redis mất hàng đợi) được tiến trình quét tự đánh lỗi + hoàn credit sau
`FAST_TIMEOUT_SECONDS` / `HEAVY_TIMEOUT_SECONDS` / `HEAVY_QUEUE_TIMEOUT_SECONDS`.

**Nạp tiền (VietQR).** Người dùng **tự nhập số tiền bất kỳ, tối thiểu 20.000đ** (admin đổi mức tối thiểu ở tab Bảng giá;
tối đa 50.000.000đ mỗi lần để chặn gõ nhầm), có sẵn các nút gợi ý 20k/50k/100k/200k/500k. Credit = số tiền ÷ giá trị 1
credit (mặc định 1.000đ, admin chỉnh ở tab Bảng giá; không còn thưởng theo gói). Tỉ lệ được chốt vào giao dịch lúc tạo mã,
nên đổi tỉ lệ sau đó không ảnh hưởng mã QR đang chờ. Tạo giao dịch `pending` + mã `NAPxxxxxxxx` + link QR. **QR có hiệu
lực 5 phút** (`TX_TTL_SECONDS`): hết hạn thì lệnh bị huỷ → `failed`, giao diện đếm ngược và báo hết hạn. Webhook
`POST /api/webhooks/payment` (SePay: `Authorization: Apikey <SEPAY_API_KEY>`, Casso: `secure-token`) cộng credit khi
mã tồn tại, số tiền khớp chính xác và giao dịch còn `pending`; gửi lại không cộng thêm. Tiền về **sau khi hết hạn**
không tự cộng (tránh cộng cho lệnh đã huỷ): ghi vào nhật ký webhook, admin bấm "Duyệt tay"; đặt
`LATE_PAYMENT_AUTOCREDIT=1` nếu muốn tự cộng. Sai số tiền cũng chờ admin. Mọi webhook đều được ghi log (tab Admin).

**Nạp tiền chỉ dành cho người dùng; admin cộng/trừ credit thủ công.** Tài khoản admin không có tab Nạp tiền và
`POST /api/billing/topup` trả 403 với admin. Thay vào đó tab **Tài khoản** (admin) cho tìm tài khoản theo tên/email và
**cộng hoặc trừ số credit tuỳ ý** (0,01 đến 10.000.000, kèm lý do, kể cả tài khoản admin/chính mình). Trừ không bao giờ làm
số dư âm (kể cả khi bấm song song), mỗi thay đổi được ghi lại (ai, cho ai, bao nhiêu, lý do, số dư sau) ở cuối trang và
`GET /api/admin/credit-log`; người dùng thấy các điều chỉnh của mình trong trang Nạp tiền.

**Khuyến mãi nạp tiền.** Tab **Khuyến mãi** (admin): tạo chương trình thưởng % credit theo mức nạp, ví dụ *Lễ 30/4:
nạp từ 100.000đ thưởng 10%* (có thể thêm nhiều mức: 500.000đ thưởng 15%, ...). Mỗi chương trình có tên, thời gian bắt đầu /
kết thúc (bỏ trống = áp dụng ngay / không hạn), bật-tắt, sửa, xoá. Quy tắc: đạt mức nào thì lấy mức cao nhất mình đạt;
thưởng = credit cơ bản × %; nhiều chương trình cùng chạy thì chỉ áp mức thưởng **cao nhất** (không cộng dồn). Người
dùng thấy banner khuyến mãi, nút gợi ý có nhãn `+10%`, dòng quy đổi trực tiếp ("100.000đ = 100 credit + 10 credit khuyến
mãi = 110 credit") và gợi ý "nạp thêm X để được +Y%". Mức thưởng được **chốt khi tạo mã QR** (hết khuyến mãi giữa chừng
vẫn được thưởng; chương trình tạo sau không đổi QR cũ), và webhook cộng đúng tổng credit đã hứa; lịch sử giao dịch ghi
rõ phần thưởng và tên chương trình.

**Thư viện chung.** Mỗi video render xong: (1) trả về cho người dùng, (2) tự thêm vào hàng chờ của admin ở trạng thái
`private` kèm ảnh thu nhỏ và video xem trước. Admin xem, bấm **Phê duyệt (publish)** để đưa vào thư viện chung (hoặc
Xoá). Người khác dùng lại mục đó bắt đầu từ nội dung gốc (chữ, logo, ảnh, nền... là mặc định) và **sửa được mọi
layer**. Người dùng có thể bỏ chọn "Gửi video này cho admin" khi tạo.

**Render nhanh** (FFmpeg): thay chữ / ảnh / logo / nền ngay, vài giây. **Render nâng cao** (MimicMotion): hàng đợi
Celery + Redis, chạy nền trên GPU local, không chặn giao diện.

## Render nâng cao (MimicMotion) — tuỳ chọn
```bash
./scripts/setup_mimicmotion.sh        # clone Tencent/MimicMotion vào vendor/, in các bước cài còn lại
```
Cần card NVIDIA ~16GB VRAM, `redis-server`, và đặt model vào `vendor/MimicMotion/models/` (xem README của repo đó).
Có sẵn `vendor/MimicMotion` (và/hoặc `MINIMAX_API_KEY`) + `redis-server` thì `python run.py` tự bật Redis và các worker
Celery (`ai` và `gpu`).

## Docker (tuỳ chọn)
```bash
python run.py --init                  # tạo .env
docker compose up --build             # web + Redis;  thêm --profile worker để có worker video AI
```
MimicMotion cần GPU: chạy worker trên máy chủ (`celery -A app.tasks.celery_app worker -n gpu@%h -Q gpu -c 1`) trỏ vào Redis
`127.0.0.1:6379` do compose mở ra. *Lưu ý: file Docker chưa được build thử (môi trường phát triển không có Docker daemon).*

## Nạp tiền thật
Mở server ra internet (ví dụ `ngrok http 8000`), đặt webhook của SePay/Casso tới `/api/webhooks/payment`, điền
`BANK_BIN`, `BANK_ACCOUNT_NO`, `BANK_ACCOUNT_NAME`, `SEPAY_API_KEY`/`CASSO_SECURE_TOKEN`. Thử không cần ngân hàng:
```bash
curl -X POST localhost:8000/api/webhooks/payment -H "Authorization: Apikey <SEPAY_API_KEY>" \
  -H "Content-Type: application/json" -d '{"transferType":"in","transferAmount":<số tiền đã tạo mã>,"content":"NAPxxxxxxxx"}'
```

## Schema template
```json
{"width":720,"height":1280,"duration":5,"motion_transfer":false,
 "layers":[{"id":"title","type":"text","default":"Hi","fontsize":64,"color":"white","x":"(w-text_w)/2","y":"h*0.4"},
           {"id":"logo","type":"logo","x":30,"y":30,"w":160},
           {"id":"bg","type":"background"}]}
```
`input_data` ánh xạ id layer → chữ, hoặc URL ảnh đã upload (`/media/uploads/...`). `motion_transfer: true` + ảnh
`character_image` → AI điều hướng sang luồng nâng cao.

## Cơ sở dữ liệu
SQLite (`storage/app.db`), tự nâng cấp cấu trúc bằng Alembic khi khởi động (`migrations/`); DB tạo bởi bản cũ được
nhận diện tự động. Muốn làm lại từ đầu: tắt server, xoá `storage/app.db`.

## Bảo mật & vận hành
`APP_ENV=production` từ chối khởi động nếu `SECRET_KEY` yếu/mặc định hoặc thiếu `ADMIN_PASSWORD`. Đăng nhập sai
quá `LOGIN_MAX_FAILURES` lần/10 phút bị khoá tạm; giới hạn đăng ký `REGISTER_PER_HOUR`/IP. Chỉ `uploads/` và `outputs/`
được phục vụ qua `/media`.

## Test
`pip install -r requirements-dev.txt && pytest` (155 test; test thiếu công cụ sẽ tự bỏ qua).

| Nhóm | Phạm vi | Cần |
|---|---|---|
| `test_flow`, `test_more`, `test_features` | ví, webhook, hết hạn QR, hoàn tiền, job kẹt, thư viện, AI (server giả), giới hạn, migration, render FFmpeg | ffmpeg |
| `test_admin_credits` | admin cộng/trừ credit tuỳ ý: validate, nhật ký, không âm khi song song, quyền, admin không nạp tiền |
| `test_promotions` | khuyến mãi: ngưỡng, nhiều mức, lịch, không cộng dồn, chốt lúc tạo QR, làm tròn, CRUD, validate |
| `test_topup_amount` | nạp số tiền tự do: mức tối thiểu, quy đổi, tối đa, đổi mức/tỉ lệ, hết hạn |
| `test_pricing` | bảng giá admin: phân quyền, validate, nhật ký, đồng bộ với mọi khoản trừ, hoàn đúng giá cũ, tắt model/độ phân giải |
| `test_aivideo` | MiniMax-H3 qua server giả: tạo/thăm dò/tải, API kiểu cũ, lỗi + hoàn tiền, dance 2 giai đoạn + ghép nhạc, giá |
| `test_e2e` | Redis + 2 worker Celery (`ai`/`gpu`) + uvicorn thật, MimicMotion giả cùng giao diện dòng lệnh, Redis sập | ffmpeg, redis-server, celery |
| `test_ui` | Chromium thật: đăng ký → QR đếm ngược/hết hạn → duyệt thư viện → người khác tuỳ chỉnh → tab AI | playwright + chromium |

Chưa kiểm chứng với dịch vụ thật: MimicMotion (GPU), MiniMax-H3, Inkstone, SePay/Casso, ảnh QR của img.vietqr.io, Docker build.
