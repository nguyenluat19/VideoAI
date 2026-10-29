"""Step 1: Vision-Language model for prompt generation.

Uses Qwen2-VL-7B-Instruct via Hugging Face transformers.
Produces strict JSON: {"image_prompt": str, "motion_prompt": str}.
"""
import gc
import json
import re
from typing import Any, Optional

from .config import PipelineConfig


class VLMError(RuntimeError):
    pass


def _clean_json(text: str) -> str:
    """Strip markdown, preambles, and quotes from model output."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # Remove common preambles
    for prefix in ["Here is", "Here's", "Sure,", "Certainly,", "Of course,"]:
        text = text.replace(prefix, "")
    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_prompts(text: str) -> dict[str, str]:
    """Parse model output into strict JSON with image_prompt and motion_prompt."""
    cleaned = _clean_json(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Retry: try to fix common issues
        # Remove trailing commas
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise VLMError(f"Failed to parse JSON after cleanup: {e}\nRaw: {text[:500]}")

    image_prompt = data.get("image_prompt", "")
    motion_prompt = data.get("motion_prompt", "")

    # Validate and clean
    if not image_prompt or not isinstance(image_prompt, str):
        raise VLMError("Missing or invalid image_prompt in JSON response")
    if not motion_prompt or not isinstance(motion_prompt, str):
        raise VLMError("Missing or invalid motion_prompt in JSON response")

    # Truncate image_prompt to ~150 words for FLUX.1-schnell
    words = image_prompt.split()
    if len(words) > 150:
        image_prompt = " ".join(words[:150])
    # Truncate motion_prompt to 1-3 sentences
    sentences = motion_prompt.split(". ")
    if len(sentences) > 3:
        motion_prompt = ". ".join(sentences[:3])
    if not motion_prompt.endswith("."):
        motion_prompt += "."

    return {"image_prompt": image_prompt.strip(), "motion_prompt": motion_prompt.strip()}


class VLMPipeline:
    """Vision-Language model for prompt generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load the VLM model."""
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_id = self.config.vlm_model_id
        dtype = torch.bfloat16 if self.config.vlm_dtype == "bfloat16" else torch.float16

        # Check if bf16 is supported
        if self.config.vlm_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 not supported, falling back to float16")
            dtype = torch.float16

        print(f"Loading VLM: {model_id} (dtype={dtype}, 4bit={self.config.vlm_load_in_4bit})")

        # Load with 4-bit quantization if configured AND bitsandbytes is available
        if self.config.vlm_load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    torch_dtype=dtype,
                    device_map=self.config.vlm_device_map,
                    revision=self.config.vlm_revision,
                )
            except ImportError as e:
                print(f"WARNING: bitsandbytes not available, falling back to full precision: {e}")
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=self.config.vlm_device_map,
                    revision=self.config.vlm_revision,
                )
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=self.config.vlm_device_map,
                revision=self.config.vlm_revision,
            )

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=self.config.vlm_min_pixels,
            max_pixels=self.config.vlm_max_pixels,
            revision=self.config.vlm_revision,
        )

    def unload(self) -> None:
        """Completely unload the model to free VRAM."""
        import gc
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"VLM unloaded. VRAM allocated: {vram:.2f} GB")

    def generate_prompts(self, image_path: str | None, user_instruction: str | None = None,
                         config: PipelineConfig | None = None) -> dict[str, str]:
        """Generate image_prompt and motion_prompt from input image or text instruction."""
        import torch
        from PIL import Image

        # If no image provided, use user instruction directly as prompt
        if not image_path:
            if not user_instruction:
                raise VLMError("No image or instruction provided")
            # Generate prompts from text instruction only
            return self._generate_from_text(user_instruction)

        # Validate image
        if not os.path.exists(image_path):
            raise VLMError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
        except Exception as e:
            raise VLMError(f"Failed to load image: {e}")

        # Apply EXIF rotation
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if self._model is None or self._processor is None:
            raise VLMError("Model not loaded. Call load() first.")

        # Build messages with image
        system_prompt = (
            "You are a professional video production director. "
            "Analyze the input image and generate TWO outputs as strict JSON:\n"
            '{"image_prompt": "natural language description of the image for AI image generation", '
            '"motion_prompt": "description of motion/camera movement for video generation"}\n'
            "Rules:\n"
            "- image_prompt: English, natural sentences, NO keyword soup, NO weights, NO negative prompts. "
            "Cover: subject, composition, lighting, colour palette, style, lens. MAX 150 words. "
            "Put most important details first.\n"
            "- motion_prompt: English, ONLY motion: camera movement, subject action, speed, atmosphere. "
            "1-3 sentences. Do NOT re-describe the whole scene.\n"
            "Return ONLY valid JSON, no markdown, no preamble."
        )

        user_text = "Analyze this image and generate the prompts."
        if user_instruction:
            user_text += f" Additional instruction: {user_instruction}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]

        # Prepare inputs
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with inference mode
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.vlm_temperature,
                do_sample=True,
            )

        # Decode output
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        output_text = self._processor.decode(generated_ids, skip_special_tokens=True)

        # Parse JSON
        for attempt in range(2):
            try:
                return _parse_prompts(output_text)
            except VLMError:
                if attempt == 0:
                    # Retry with a simpler prompt
                    messages[-1]["content"][-1]["text"] = (
                        "Return ONLY valid JSON: "
                        '{"image_prompt": "...", "motion_prompt": "..."}'
                    )
                    text = self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        output_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.0,
                            do_sample=False,
                        )
                    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                    output_text = self._processor.decode(generated_ids, skip_special_tokens=True)
                else:
                    raise
        raise VLMError("Failed to generate valid JSON after retries")

    def _generate_from_text(self, text_instruction: str) -> dict[str, str]:
        """Generate prompts from text instruction only (no image)."""
        if self._model is None or self._processor is None:
            raise VLMError("Model not loaded. Call load() first.")
        import torch

        system_prompt = (
            "You are a professional video production director. "
            "Generate TWO outputs as strict JSON based on the user's text instruction:\n"
            '{"image_prompt": "natural language description for AI image generation", '
            '"motion_prompt": "description of motion/camera movement for video generation"}\n'
            "Rules:\n"
            "- image_prompt: English, natural sentences, NO keyword soup, NO weights, NO negative prompts. "
            "Cover: subject, composition, lighting, colour palette, style, lens. MAX 150 words. "
            "Put most important details first.\n"
            "- motion_prompt: English, ONLY motion: camera movement, subject action, speed, atmosphere. "
            "1-3 sentences. Do NOT re-describe the whole scene.\n"
            "Return ONLY valid JSON, no markdown, no preamble."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_instruction},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.vlm_temperature,
                do_sample=True,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        output_text = self._processor.decode(generated_ids, skip_special_tokens=True)

        for attempt in range(2):
            try:
                return _parse_prompts(output_text)
            except VLMError:
                if attempt == 0:
                    messages[-1]["content"] = (
                        "Return ONLY valid JSON: "
                        '{"image_prompt": "...", "motion_prompt": "..."}'
                    )
                    text = self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        text=[text],
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        output_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.0,
                            do_sample=False,
                        )
                    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                    output_text = self._processor.decode(generated_ids, skip_special_tokens=True)
                else:
                    raise
        raise VLMError("Failed to generate valid JSON after retries")


def run_vlm(image_path: str, user_instruction: str | None = None,
            config: PipelineConfig | None = None) -> dict[str, str]:
    """Convenience function to run the VLM step."""
    if config is None:
        config = PipelineConfig()
    vlm = VLMPipeline(config)
    try:
        vlm.load()
        return vlm.generate_prompts(image_path, user_instruction)
    finally:
        vlm.unload()