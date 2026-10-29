"""Step 2: Text-to-Image using FLUX.1-schnell via Hugging Face diffusers."""
import gc
import math
from pathlib import Path
from typing import Optional

import torch
from diffusers import FluxPipeline


class ImageGenError(RuntimeError):
    pass


class ImageGenPipeline:
    """FLUX.1-schnell text-to-image generation."""

    def __init__(self, config):
        self.config = config
        self._pipe = None

    def load(self) -> None:
        """Load the FLUX.1-schnell model."""
        import torch

        model_id = self.config.t2i_model_id
        dtype = torch.bfloat16 if self.config.t2i_dtype == "bfloat16" else torch.float16

        # Check bf16 support
        if self.config.t2i_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 not supported by GPU, falling back to float16")
            dtype = torch.float16

        print(f"Loading Flux: {model_id} (dtype={dtype})")

        try:
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                revision=self.config.t2i_revision,
            )
        except Exception as e:
            if "gated" in str(e).lower() or "403" in str(e) or "restricted" in str(e).lower():
                raise ImageGenError(
                    f"Model {model_id} is a gated repository. "
                    f"Visit https://huggingface.co/{model_id} to accept the license, "
                    f"then run: huggingface-cli login (or set HF_TOKEN env var)"
                ) from e
            raise ImageGenError(f"Failed to load FLUX model: {e}") from e

        # Enable CPU offload to save VRAM
        pipe.enable_model_cpu_offload()

        # Optionally enable memory efficient attention
        try:
            pipe.enable_model_cpu_offload()
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
            print(f"ImageGen unloaded. VRAM allocated: {vram:.2f} GB")

    def generate(self, prompt: str, seed: int | None = None) -> str:
        """Generate an image from prompt."""
        import time
        from pathlib import Path

        if self._pipe is None:
            raise ImageGenError("Model not loaded. Call load() first.")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / "image.png"

        # Build kwargs - snel has no guidance_scale
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=self.config.t2i_num_inference_steps,
            max_sequence_length=self.config.t2i_max_sequence_length,
            guidance_scale=self.config.t2i_guidance_scale,
            width=self.config.t2i_image_size,
            height=self.config.t2i_image_size,
        )

        if seed is not None:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        print(f"Generating image: {self.config.t2i_image_size}x{self.config.t2i_image_size}, "
              f"{self.config.t2i_num_inference_steps} steps")

        with torch.inference_mode():
            result = self._pipe(**kwargs)

        image = result.images[0]
        image.save(out_path)
        print(f"Image saved: {out_path}")
        return str(out_path)


def run_image_gen(prompt: str, seed: int | None = None,
                  config=None) -> str:
    """Convenience function to run image generation."""
    if config is None:
        from .config import PipelineConfig
        config = PipelineConfig()
    pipe = ImageGenPipeline(config)
    try:
        pipe.load()
        return pipe.generate(prompt, seed)
    finally:
        pipe.unload()