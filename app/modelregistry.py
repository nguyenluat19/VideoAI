"""Model registry: one record per AI model the user can pick for a pipeline step.

Each model declares capabilities (max clip seconds, resolution, aspect ratios, supports
first/last frame, reference image, audio, TTS, etc.) so the system adapts to whichever
model the user chose without hard-coding provider logic.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class ModelSpec:
    id: str
    name: str
    step: str            # llm | image_generate | image_edit | video_i2v | video_t2v | tts | lipsync | music | vlm
    provider: str        # inkstone | minimax | local | external
    kind: str            # api | local_gpu
    desc: str = ""
    strengths: str = ""
    weaknesses: str = ""
    enabled: bool = True
    # capabilities
    max_clip_seconds: int = 10
    min_clip_seconds: int = 1
    resolutions: list[str] = field(default_factory=lambda: ["768P"])
    aspect_ratios: list[str] = field(default_factory=lambda: ["9:16", "16:9", "1:1"])
    supports_first_frame: bool = True
    supports_last_frame: bool = False
    supports_reference_image: bool = False
    supports_audio: bool = False
    supports_voice_clone: bool = False
    supports_vietnamese: bool = True
    max_reference_images: int = 0
    # pricing unit hint
    price_unit: str = "call"  # call | image | second | 1k_chars
    default_price: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry: LLM models (from Inkstone catalogue)
# ---------------------------------------------------------------------------
LLM_MODELS: list[ModelSpec] = [
    ModelSpec("deepseek-v4-pro-0813", "DeepSeek-V4-Pro-0813", "llm", "inkstone", "api",
              desc="LLM mạnh", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("deepseek-v4-flash-0731", "DeepSeek-V4-Flash-0731", "llm", "inkstone", "api",
              desc="LLM nhanh", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("glm-5.3", "GLM-5.3", "llm", "inkstone", "api",
              desc="LLM", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("minimax-m3", "MiniMax-M3", "llm", "inkstone", "api",
              desc="LLM", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("Atria-Dawn", "Atria-Dawn", "llm", "inkstone", "api",
              desc="LLM logic", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("Atria-Dawn-Preview", "Atria-Dawn-Preview", "llm", "inkstone", "api",
              desc="LLM logic (preview)", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
]

VISION_MODELS: list[ModelSpec] = [
    ModelSpec("deepseek-v4-flash-vision", "DeepSeek-V4-Flash-Vision", "image_generate", "inkstone", "api",
              desc="Ngữ cảnh 1M, tool calling, tư duy sâu; phân tích ảnh gameplay", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("kimi-k2.6", "Kimi-K2.6", "image_generate", "inkstone", "api",
              desc="Ngữ cảnh 256K, text+ảnh, tool calling, tư duy sâu", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("qwen3.8-27b", "Qwen3.8-27B", "image_generate", "inkstone", "api",
              desc="Vision", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("Agents-A1", "Agents-A1", "image_generate", "inkstone", "api",
              desc="Vision", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("Sage-A1.5", "Sage-A1.5", "image_generate", "inkstone", "api",
              desc="Vision", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("intern-s2", "书生-S2", "image_generate", "inkstone", "api",
              desc="Vision", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
    ModelSpec("deepseek-v4.1-flash-1017", "DeepSeek-V4.1-Flash-1017", "image_generate", "inkstone", "api",
              desc="Vision", max_clip_seconds=0,
              price_unit="call", default_price=0.0),
]

VIDEO_MODELS: list[ModelSpec] = [
    ModelSpec("wan2.1", "Wan2.1", "video_i2v", "local", "local_gpu",
              desc="Video local (Wan, cần GPU 8GB+)", max_clip_seconds=5, min_clip_seconds=5,
              resolutions=["720P", "480P"], aspect_ratios=["16:9", "9:16", "1:1"],
              supports_first_frame=False, supports_audio=False,
              price_unit="second", default_price=0.0, enabled=True),
]

# Muse AI models (external, run through FlowKit gateway + Muse Kit Chrome extension)
# These are two separate models that operate independently.
MUSE_MODELS: list[ModelSpec] = []

ALL_MODELS: list[ModelSpec] = LLM_MODELS + VISION_MODELS + VIDEO_MODELS + MUSE_MODELS

STEPS = {
    "llm": "Tạo prompt/kịch bản/storyboard",
    "image_generate": "Tạo ảnh nhân vật / ảnh từng cảnh",
    "image_edit": "Sửa ảnh theo tham chiếu",
    "video_i2v": "Gen video (image-to-video)",
    "video_t2v": "Gen video (text-to-video)",
    "tts": "Giọng đọc (TTS)",
    "lipsync": "Ghép khẩu hình",
    "music": "Nhạc nền",
    "vlm": "Chấm chất lượng (VLM)",
}


def models_for_step(step: str) -> list[ModelSpec]:
    return [m for m in ALL_MODELS if m.step == step and m.enabled]


def get_model(model_id: str) -> ModelSpec | None:
    for m in ALL_MODELS:
        if m.id == model_id:
            return m
    return None


def clip_count_for_duration(model: ModelSpec, total_seconds: int) -> int:
    """How many clips needed to cover total_seconds with this model's max clip length."""
    if model.max_clip_seconds <= 0:
        return 1
    import math
    return math.ceil(total_seconds / model.max_clip_seconds)


def validate_selections(selections: dict) -> list[str]:
    """Validate that user picked exactly one model per required step. Returns list of errors."""
    errors = []
    required = ["llm", "image_generate", "video_i2v"]
    for step in required:
        mid = selections.get(step)
        if not mid:
            errors.append(f"Chưa chọn model cho bước: {STEPS.get(step, step)}")
            continue
        m = get_model(mid)
        if not m:
            errors.append(f"Model không tồn tại: {mid}")
            continue
        if m.step != step:
            errors.append(f"Model {mid} không thuộc bước {step}")
            continue
        if not m.enabled:
            errors.append(f"Model {mid} đang bị tắt")
    return errors
