"""Local pipeline configuration with environment overrides.
Hardware-aware defaults: auto-detects VRAM and selects appropriate model sizes.
"""
from dataclasses import dataclass, field
import os
import torch


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "y")


def detect_vram_gb() -> float:
    """Detect available VRAM in GB. Returns 0 if no CUDA GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


@dataclass
class PipelineConfig:
    """Configuration for the local video pipeline.

    Groups all tunable parameters. Values can be overridden via environment
    variables with the same name (uppercase).
    """

    # --- Pipeline mode ---
    # "recreate" = use input image as inspiration for a NEW image (T2V after I2I)
    # "animate"  = animate the user's ORIGINAL image directly (skip T2I)
    mode: str = _env_str("PIPELINE_MODE", "recreate")

    # --- Step 1: Vision-Language Model (prompt generation) ---
    vlm_model_id: str = _env_str("VLM_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
    vlm_revision: str = _env_str("VLM_REVISION", "main")
    vlm_device_map: str = _env_str("VLM_DEVICE_MAP", "auto")
    vlm_dtype: str = _env_str("VLM_DTYPE", "bfloat16")  # bfloat16 or float16
    vlm_load_in_4bit: bool = _env_bool("VLM_LOAD_IN_4BIT", True)
    vlm_max_pixels: int = _env_int("VLM_MAX_PIXELS", 128 * 128 * 2048)  # cap visual tokens
    vlm_min_pixels: int = _env_int("VLM_MIN_PIXELS", 256 * 28 * 28)
    vlm_temperature: float = _env_float("VLM_TEMPERATURE", 0.1)

    # --- Step 2: Text-to-Image (FLUX.1-schnell) ---
    t2i_model_id: str = _env_str("T2I_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    t2i_revision: str = _env_str("T2I_REVISION", "main")
    t2i_device_map: str = _env_str("T2I_DEVICE_MAP", "balanced")
    t2i_dtype: str = _env_str("T2I_DTYPE", "bfloat16")
    t2i_guidance_scale: float = _env_float("T2I_GUIDANCE_SCALE", 0.0)
    t2i_num_inference_steps: int = _env_int("T2I_STEPS", 4)
    t2i_max_sequence_length: int = _env_int("T2I_MAX_SEQ", 256)
    t2i_image_size: int = _env_int("T2I_IMAGE_SIZE", 1024)  # multiple of 16

    # --- Step 3: Image-to-Video (Wan2.1-T2V-14B) ---
    i2v_model_id: str = _env_str("I2V_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B")
    i2v_revision: str = _env_str("I2V_REVISION", "main")
    i2v_device_map: str = _env_str("I2V_DEVICE_MAP", "balanced")
    i2v_dtype: str = _env_str("I2V_DTYPE", "bfloat16")
    i2v_size: str = _env_str("I2V_SIZE", "480P")  # 480P = 832x480, 720P = 1280x720
    i2v_num_frames: int = _env_int("I2V_NUM_FRAMES", 81)  # 4k+1, 81 ~ 5s @16fps
    i2v_guidance_scale: float = _env_float("I2V_GUIDANCE_SCALE", 5.0)
    i2v_num_inference_steps: int = _env_int("I2V_STEPS", 20)
    i2v_sample_stride: int = _env_int("I2V_SAMPLE_STRIDE", 1)
    i2v_noise_aug_strength: float = _env_float("I2V_NOISE_AUG", 0.0)
    i2v_guidance_scale_img: float = _env_float("I2V_GUIDANCE_SCALE_IMG", 2.0)
    i2v_sample_stride_img: int = _env_int("I2V_SAMPLE_STRIDE_IMG", 2)

    # --- Output ---
    output_dir: str = _env_str("PIPELINE_OUTPUT_DIR", "./pipeline_output")
    output_fps: int = _env_int("OUTPUT_FPS", 16)
    output_codec: str = _env_str("OUTPUT_CODEC", "libx264")

    # --- Hugging Face ---
    hf_home: str = _env_str("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_token: str = _env_str("HF_TOKEN", "")

    def __post_init__(self):
        # Apply env overrides for numeric fields that need casting
        self.vlm_max_pixels = _env_int("VLM_MAX_PIXELS", self.vlm_max_pixels)
        self.vlm_min_pixels = _env_int("VLM_MIN_PIXELS", self.vlm_min_pixels)
        self.vlm_temperature = _env_float("VLM_TEMPERATURE", self.vlm_temperature)
        self.t2i_guidance_scale = _env_float("T2I_GUIDANCE_SCALE", self.t2i_guidance_scale)
        self.t2i_num_inference_steps = _env_int("T2I_STEPS", self.t2i_num_inference_steps)
        self.t2i_max_sequence_length = _env_int("T2I_MAX_SEQ", self.t2i_max_sequence_length)
        self.t2i_image_size = _env_int("T2I_IMAGE_SIZE", self.t2i_image_size)
        self.i2v_num_frames = _env_int("I2V_NUM_FRAMES", self.i2v_num_frames)
        self.i2v_guidance_scale = _env_float("I2V_GUIDANCE_SCALE", self.i2v_guidance_scale)
        self.i2v_num_inference_steps = _env_int("I2V_STEPS", self.i2v_num_inference_steps)
        self.i2v_sample_stride = _env_int("I2V_SAMPLE_STRIDE", self.i2v_sample_stride)
        self.i2v_noise_aug_strength = _env_float("I2V_NOISE_AUG", self.i2v_noise_aug_strength)
        self.i2v_guidance_scale_img = _env_float("I2V_GUIDANCE_SCALE_IMG", self.i2v_guidance_scale_img)
        self.i2v_sample_stride_img = _env_int("I2V_SAMPLE_STRIDE_IMG", self.i2v_sample_stride_img)
        self.output_fps = _env_int("OUTPUT_FPS", self.output_fps)

    @property
    def vram_gb(self) -> float:
        return detect_vram_gb()

    @property
    def has_gpu(self) -> bool:
        return self.vram_gb > 0

    def log_config(self) -> None:
        """Log current configuration for debugging."""
        print(f"Pipeline config:")
        print(f"  mode: {self.mode}")
        print(f"  vlm_model_id: {self.vlm_model_id}")
        print(f"  vlm_load_in_4bit: {self.vlm_load_in_4bit}")
        print(f"  t2i_model_id: {self.t2i_model_id}")
        print(f"  i2v_model_id: {self.i2v_model_id}")
        print(f"  i2v_size: {self.i2v_size}")
        print(f"  i2v_num_frames: {self.i2v_num_frames}")
        print(f"  output_dir: {self.output_dir}")
        print(f"  HF_HOME: {self.hf_home}")
        print(f"  VRAM: {self.vram_gb:.1f} GB (GPU: {self.has_gpu})")