"""Pipeline: yêu cầu văn bản -> storyboard -> ảnh từng cảnh -> video từng cảnh -> dựng thành phẩm.

Mỗi bước dùng MỘT model do người dùng chọn (theo model registry), không tự chuyển model khi lỗi.
"""
import json
import math
import re

from . import ai
from .modelregistry import get_model


def plan_storyboard(topic: str, model_id: str, duration_seconds: int,
                    aspect_ratio: str, character_image_url: str | None = None,
                    max_retries: int = 2) -> dict:
    """Tạo storyboard JSON từ yêu cầu văn bản bằng model LLM đã chọn.

    Không tự đổi model khi lỗi: thử lại chính model đó tối đa `max_retries` lần.
    """
    model = get_model(model_id)
    if not model or model.step != "llm":
        raise ValueError(f"Model {model_id} không phải model kịch bản (LLM)")
    if not model.enabled:
        raise ValueError(f"Model {model_id} đang bị tắt")

    min_per_scene = max(1, duration_seconds // 5)
    clip_count = max(1, math.ceil(duration_seconds / min_per_scene))

    sys_prompt = (
        "storyboard chi tiết dưới dạng JSON thuần túy (KHÔNG bọc trong markdown code block, "
        "KHÔNG giải thích thêm ngoài JSON).\n"
        "Schema JSON bắt buộc:\n"
        "{\n"
        '  "character_bible": {"mô_tả": "...", "trang_phúc": "...", "tuổi": "...", "tính_diện": "...", "ghi_chú_không_thay_đổi": "..."},\n'
        '  "style_bible": {"ánh_sáng": "...", "tông_màu": "...", "ống_kính": "...", "tỉ_lệ_khung_hình": "...", "phong_cách": "..."},\n'
        '  "location_bible": {"bối_cảnh": "...", "tái_sử_dụng": true},\n'
        '  "các_cảnh": [\n'
        "    {\n"
        '      "stt": 1,\n'
        '      "thời_lượng_giây": 10,\n'
        '      "loại_shot": "...",\n'
        '      "góc_máy": "...",\n'
        '      "chuyển_động_camera": "...",\n'
        '      "hành_động": "...",\n'
        '      "biểu_cảm": "...",\n'
        '      "thoại": "...",\n'
        '      "ghi_chú_âm_thanh": "...",\n'
        '      "ghi_chú_liên_tục": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Quy tắc:\n"
        "- Mỗi cảnh là một cú quay ngắn, cắt theo hành động (không quay dài liền bị cắt khúc).\n"
        f"- Tổng độ dài các cảnh bằng {duration_seconds} giây.\n"
        f"- Mỗi cảnh <= {min_per_scene} giây (giới hạn clip của model video đã chọn).\n"
        "- Lời thoại phải vừa độ dài cảnh (tiếng Việt).\n"
        "- Nhân vật giữ nhất quán qua mọi cảnh (mặt, tóc, dáng, trang phục khoá).\n"
        "- Nếu có ảnh nhân vật tham chiếu, dùng đúng nhân vật đó trong mọi cảnh.\n"
    )

    user_msg = (
        f"Yêu cầu video: {topic}\n"
        f"Tổng độ dài: {duration_seconds} giây\n"
        f"Tỉ lệ khung hình: {aspect_ratio}\n"
        f"Số clip ước tính: {clip_count}\n"
        f"{'Ảnh nhân vật tham chiếu: ' + character_image_url if character_image_url else ''}"
    )

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            msg = ai.chat(model_id, [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ], json_mode=True, timeout=30)
            text = (msg.get("content") or "").strip()
            m = re.search(r"\{.*\}", text or "", re.DOTALL)
            if not m:
                raise ValueError("LLM không trả về JSON hợp lệ")
            data = json.loads(m.group(0))

            scenes = data.get("các_cảnh", [])
            if not scenes:
                raise ValueError("Storyboard không có cảnh nào")

            normalized = []
            for i, scene in enumerate(scenes):
                dur = int(scene.get("thời_lượng_giây", min_per_scene))
                dur = max(1, min(dur, min_per_scene if min_per_scene > 0 else duration_seconds))
                normalized.append({
                    "stt": scene.get("stt", i + 1),
                    "thời_lượng_giây": dur,
                    "loại_shot": str(scene.get("loại_shot", "MS")),
                    "góc_máy": str(scene.get("góc_máy", "mặt đã")),
                    "chuyển_động_camera": str(scene.get("chuyển_động_camera", "tĩnh")),
                    "hành_động": str(scene.get("hành_động", "")),
                    "biểu_cảm": str(scene.get("biểu_cảm", "bình thường")),
                    "thoại": str(scene.get("thoại", "")),
                    "ghi_chú_âm_thanh": str(scene.get("ghi_chú_âm_thanh", "")),
                    "ghi_chú_liên_tục": str(scene.get("ghi_chú_liên_tục", "")),
                    "tỉ_lệ_khung_hình": aspect_ratio,
                })

            # Đảm bảo tổng ≈ duration_seconds
            total = sum(s["thời_lượng_giây"] for s in normalized)
            if total != duration_seconds and normalized:
                diff = duration_seconds - total
                normalized[0]["thời_lượng_giây"] = max(1, normalized[0]["thời_lượng_giây"] + diff)

            return {
                "model_used": model_id,
                "tổng_số_cảnh": len(normalized),
                "tổng_thời_lượng": sum(s["thời_lượng_giây"] for s in normalized),
                "character_bible": data.get("character_bible", {}),
                "style_bible": data.get("style_bible", {}),
                "location_bible": data.get("location_bible", {}),
                "các_cảnh": normalized,
            }
        except Exception as e:
            last_error = str(e)
    raise RuntimeError(f"Không tạo được storyboard sau {max_retries + 1} lần thử ({model_id}): {last_error}")


def gen_keyframe_prompt(scene: dict, character_bible: dict, style_bible: dict,
                        character_image_url: str | None = None) -> str:
    """Sinh prompt ảnh cho một cảnh từ storyboard + bible."""
    parts = [
        f"Hành động: {scene['hành_động']}",
        f"Biểu cảm: {scene['biểu_cảm']}",
        f"Góc máy: {scene['góc_máy']}",
        f"Chuyển động camera: {scene['chuyển_động_camera']}",
        f"Ánh sáng: {style_bible.get('ánh_sáng', 'ánh sáng tự nhiên')}",
        f"Tông màu: {style_bible.get('tông_màu', 'tự nhiên')}",
        f"Phong cách: {style_bible.get('phong_cách', 'cinematic')}",
        f"Tỉ lệ: {scene['tỉ_lệ_khung_hình']}",
    ]
    if character_image_url:
        parts.append(f"Ảnh nhân vật tham chiếu: {character_image_url}")
    cb = character_bible
    parts.append(f"Nhân vật: {cb.get('mô_tả', '')} | Trang phục: {cb.get('trang_phúc', '')} | Tuổi: {cb.get('tuổi', '')}")
    return ". ".join(parts)


def scene_clip_count(scene: dict, video_model_id: str) -> int:
    """Số clip cần gen cho một cảnh dựa trên giới hạn của model video đã chọn."""
    model = get_model(video_model_id)
    if not model or model.max_clip_seconds <= 0:
        return 1
    dur = scene.get("thời_lượng_giây", 10)
    return math.ceil(dur / model.max_clip_seconds)


def estimated_cost(selections: dict, duration_seconds: int,
                   num_scenes: int, images_per_scene: int = 1,
                   db=None) -> dict:
    """Ước tính chi phí trước khi bấm Gen (dựa trên model đã chọn)."""
    from . import pricing
    details = {}
    total = 0.0

    # LLM: 1 lần gọi
    llm_id = selections.get("llm")
    if llm_id:
        llm_price_key = f"ai.{llm_id}" if f"ai.{llm_id}" in pricing.catalog() else None
        llm_price = pricing.price(db, llm_price_key) if llm_price_key else 0.0
        details["llm"] = {"model": llm_id, "unit": "lần gọi", "số_lượng": 1, "giá": llm_price}
        total += llm_price

    # Images: 1 ảnh × số cảnh
    img_id = selections.get("image_generate")
    if img_id:
        img_price_key = f"m.{img_id}" if f"m.{img_id}" in pricing.catalog() else None
        img_price = pricing.price(db, img_price_key) if img_price_key else 0.0
        img_cost = round(img_price * num_scenes * images_per_scene, 2)
        details["image"] = {"model": img_id, "unit": "ảnh",
                            "số_lượng": num_scenes * images_per_scene, "giá": img_price,
                            "tổng": img_cost}
        total += img_cost

    # Video: tổng giây × giá/giây
    vid_id = selections.get("video_i2v")
    if vid_id:
        vid_price_key = f"m.{vid_id}" if f"m.{vid_id}" in pricing.catalog() else None
        vid_price = pricing.price(db, vid_price_key) if vid_price_key else 0.0
        total_sec = duration_seconds
        video_cost = round(total_sec * vid_price, 2)
        details["video"] = {"model": vid_id, "unit": "giây",
                            "số_lượng": total_sec, "giá_đơn_vị": vid_price,
                            "tổng_giây": total_sec, "tổng": video_cost}
        total += video_cost

    details["tổng_credit"] = round(total, 2)
    return details
