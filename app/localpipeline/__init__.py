"""Local open-source image->prompt->image->video pipeline.

A fully local pipeline with ZERO paid or hosted APIs.
Uses Qwen2-VL-7B-Instruct (VLM) + FLUX.1-schnell (T2I) + Wan2.1 I2V 14B (I2V).

Public API:
    run_pipeline(image_path, user_instruction=None, mode="recreate", options=None,
                 job_dir=None, on_progress=None) -> PipelineResult
"""
from .config import PipelineConfig, detect_vram_gb
from .pipeline import PipelineError, PipelineResult, run_pipeline
from .vlm import VLMError, run_vlm
from .image_gen import ImageGenError, run_image_gen
from .video_gen import VideoGenError, run_video_gen

__all__ = [
    "PipelineConfig",
    "detect_vram_gb",
    "PipelineResult",
    "PipelineError",
    "InvalidImageError",
    "ModelLoadError",
    "OutOfMemoryAfterRetries",
    "run_pipeline",
    "run_vlm",
    "run_image_gen",
    "run_video_gen",
    "VLMError",
    "ImageGenError",
    "VideoGenError",
]