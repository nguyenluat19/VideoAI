"""Pipeline orchestrator: text/image -> prompt -> image -> video."""
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from .config import PipelineConfig
from .vlm import VLMError, VLMPipeline
from .image_gen import ImageGenError, ImageGenPipeline
from .video_gen import VideoGenError, VideoGenPipeline


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    image_prompt: str = ""
    motion_prompt: str = ""
    image_path: str = ""
    video_path: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    vram_peaks: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    job_id: str = ""
    mode: str = "recreate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_prompt": self.image_prompt,
            "motion_prompt": self.motion_prompt,
            "image_path": self.image_path,
            "video_path": self.video_path,
            "timings": self.timings,
            "vram_peaks": self.vram_peaks,
            "warnings": self.warnings,
            "job_id": self.job_id,
            "mode": self.mode,
        }


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class InvalidImageError(PipelineError):
    """Raised when input image is invalid."""
    pass


class ModelLoadError(PipelineError):
    """Raised when a model fails to load."""
    pass


class OutOfMemoryAfterRetries(PipelineError):
    """Raised when OOM occurs after all retries."""
    pass


def _log_vram(step: str, peak: float) -> None:
    """Log VRAM usage for a step."""
    print(f"[{step}] VRAM peak: {peak:.2f} GB")


def _cleanup() -> None:
    """Aggressive cleanup of GPU memory."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_pipeline(
    image_path: str,
    user_instruction: str | None = None,
    mode: str = "recreate",
    options: dict | None = None,
    job_dir: str | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    """Run the full image->prompt->image->video pipeline.

    Args:
        image_path: Path to input JPEG/PNG image.
        user_instruction: Optional text instruction for the VLM.
        mode: "recreate" (generate new image) or "animate" (use original image).
        options: Optional dict with config overrides.
        job_dir: Optional output directory (auto-generated if None).
        on_progress: Optional callback (step_name, progress) for progress updates.

    Returns:
        PipelineResult with all outputs and metadata.
    """
    import os
    import uuid

    result = PipelineResult(mode=mode)
    result.job_id = uuid.uuid4().hex[:12]

    # Create unique job directory
    if job_dir is None:
        job_dir = f"./pipeline_output/{result.job_id}"
    os.makedirs(job_dir, exist_ok=True)
    result.timings["job_dir"] = job_dir

    config = PipelineConfig()
    if options:
        for k, v in options.items():
            if hasattr(config, k):
                setattr(config, k, v)

    config.output_dir = job_dir

    # Validate image (skip if no image provided)
    if image_path:
        if not os.path.exists(image_path):
            raise InvalidImageError(f"Image file not found: {image_path}")
        try:
            from PIL import Image
            image = Image.open(image_path)
            image.verify()
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise InvalidImageError(f"Failed to load image: {e}")
    else:
        # No input image provided - use idea as prompt directly
        image = None
        image_path = None
        result.warnings.append("No input image provided. Using idea as prompt.")

    def progress(step: str, pct: float):
        if on_progress:
            on_progress(step, pct)

    # Step 1: VLM - Generate prompts
    progress("planning", 0.1)
    vlm = VLMPipeline(config)
    peak_vram = 0.0
    try:
        vlm.load()
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result.vram_peaks["vlm"] = peak_vram
        _log_vram("vlm", peak_vram)

        t0 = time.time()
        prompts = vlm.generate_prompts(image_path, user_instruction)
        result.timings["vlm"] = time.time() - t0
        result.image_prompt = prompts["image_prompt"]
        result.motion_prompt = prompts["motion_prompt"]
        progress("planning", 0.25)
    except Exception as e:
        raise ModelLoadError(f"VLM failed: {e}")
    finally:
        vlm.unload()

    # Step 2: Text-to-Image (skip in animate mode)
    if mode == "recreate":
        progress("image_gen", 0.3)
        image_gen = ImageGenPipeline(config)
        peak_vram = 0.0
        try:
            image_gen.load()
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            result.vram_peaks["image_gen"] = peak_vram
            _log_vram("image_gen", peak_vram)

            t0 = time.time()
            image_path = image_gen.generate(result.image_prompt, seed=42)
            result.timings["image_gen"] = time.time() - t0
            result.image_path = image_path
            progress("image_gen", 0.5)
        except torch.cuda.OutOfMemoryError:
            _cleanup()
            raise OutOfMemoryAfterRetries("Image generation failed due to OOM")
        except Exception as e:
            raise ModelLoadError(f"Image generation failed: {e}")
        finally:
            image_gen.unload()
    else:
        # Animate mode: use original image
        result.image_path = image_path
        result.warnings.append("Animate mode: using original image, skipping T2I")

    # Step 3: Image-to-Video
    progress("video_gen", 0.55)
    video_gen = VideoGenPipeline(config)
    peak_vram = 0.0
    try:
        video_gen.load()
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result.vram_peaks["video_gen"] = peak_vram
        _log_vram("video_gen", peak_vram)

        t0 = time.time()
        video_path = video_gen.generate(result.motion_prompt, result.image_path, seed=42)
        result.timings["video_gen"] = time.time() - t0
        result.video_path = video_path
        progress("video_gen", 0.9)
    except torch.cuda.OutOfMemoryError:
        _cleanup()
        raise OutOfMemoryAfterRetries("Video generation failed due to OOM")
    except Exception as e:
        raise ModelLoadError(f"Video generation failed: {e}")
    finally:
        video_gen.unload()

    # Verify video with ffprobe
    progress("verify", 0.95)
    try:
        import subprocess
        result_ffprobe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", result.video_path],
            capture_output=True, text=True, timeout=30,
        )
        if result_ffprobe.returncode != 0:
            result.warnings.append(f"ffprobe warning: {result_ffprobe.stderr[:200]}")
    except Exception as e:
        result.warnings.append(f"ffprobe check failed: {e}")

    progress("done", 1.0)
    return result