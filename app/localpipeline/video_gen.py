"""Step 3: Image-to-video using Wan2.1 via Diffusers."""
import gc
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch


class VideoGenError(RuntimeError):
    pass


def _resolve_wan_size(size_str: str) -> tuple[int, int]:
    """Convert size string to (width, height) tuple for Wan2.1."""
    mapping = {
        "480P": (832, 480),
        "720P": (1280, 720),
        "1080P": (1920, 1080),
    }
    return mapping.get(size_str, (832, 480))


class VideoGenPipeline:
    """Wan2.1 image-to-video generation via Diffusers."""

    def __init__(self, config):
        self.config = config
        self._pipe = None

    def load(self) -> None:
        """Load Wan2.1-T2V-14B model."""
        import torch
        from diffusers import WanImageToVideoPipeline

        model_id = self.config.i2v_model_id
        dtype = torch.bfloat16 if self.config.i2v_dtype == "bfloat16" else torch.float16

        # Check bf16 support
        if self.config.i2v_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 not supported, falling back to float16")
            dtype = torch.float16

        print(f"Loading Wan2.1: {model_id} (dtype={dtype})")

        # First try with the model ID directly; fallback to local directory
        try:
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                revision=self.config.i2v_revision,
            )
        except Exception as e:
            print(f"Failed to load from HF hub: {e}")
            print(f"Trying local path: {self.config.i2v_model_id}")
            pipe = WanImageToVideoPipeline.from_pretrained(
                self.config.i2v_model_id,
                torch_dtype=dtype,
                revision=self.config.i2v_revision,
            )

        pipe.enable_model_cpu_offload()

        # Enable VAE slicing/tiling for memory efficiency
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

        self._pipe = pipe

    def unload(self) -> None:
        """Unload the model to free VRAM."""
        import gc
        import torch
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"VideoGen unloaded. VRAM allocated: {vram:.2f} GB")

    def generate(self, prompt: str, image_path: str,
                 seed: int | None = None) -> str:
        """Generate video from input image and prompt."""
        import time
        from pathlib import Path
        from PIL import Image

        if self._pipe is None:
            raise VideoGenError("Model not loaded. Call load() first.")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "video.mp4"

        # Load and validate image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Resize/crop to target size keeping aspect ratio
        target_w, target_h = _resolve_wan_size(self.config.i2v_size)

        # Simple resize - maintains aspect ratio approximately
        image = image.resize((target_w, target_h))

        print(f"Generating video: {target_w}x{target_h}, "
              f"{self.config.i2v_num_frames} frames, {self.config.i2v_num_inference_steps} steps")

        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                image=image,
                num_frames=self.config.i2v_num_frames,
                height=target_h,
                width=target_w,
                guidance_scale=self.config.i2v_guidance_scale,
                num_inference_steps=self.config.i2v_num_inference_steps,
                generator=torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None,
            )

        video = result.frames[0]

        # Write to MP4 using imageio
        import imageio

        # imageio expects (T, H, W, C) uint8
        video.write_gif(str(output_dir / "video_temp.gif"), fps=self.config.output_fps)
        # imageio doesn't do mp4 directly in some versions, use ffmpeg
        import subprocess
        tmp_gif = output_dir / "video_temp.gif"
        out_path = output_dir / "video.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_gif),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise VideoGenError(f"FFmpeg failed: {result.stderr[-500:]}")

        tmp_gif.unlink(missing_ok=True)

        print(f"Video saved: {out_path}")
        return str(out_path)


def run_video_gen(prompt: str, image_path: str, seed: int | None = None,
                  config=None) -> str:
    """Convenience function to run video generation."""
    if config is None:
        from .config import PipelineConfig
        config = PipelineConfig()
    pipe = VideoGenPipeline(config)
    try:
        pipe.load()
        return pipe.generate(prompt, image_path, seed)
    finally:
        pipe.unload()