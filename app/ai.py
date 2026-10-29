"""Client for the Intern Inkstone API (OpenAI-compatible chat completions)."""
import base64
import json
import mimetypes
import re
from pathlib import Path

import httpx

from .config import (DECISION_MODEL, DEFAULT_SCRIPT_MODEL, DEFAULT_VISION_MODEL,
                     LLM_MODELS, VISION_MODELS, settings)

FAST, HEAVY = "fast_remotion", "heavy_mimicmotion"
AI_VIDEO, AI_DANCE = "ai_video", "ai_dance"


class AIError(RuntimeError):
    pass


def configured() -> bool:
    return bool(settings.inkstone_base_url and settings.inkstone_api_key)


def chat(model: str, messages: list[dict], *, json_mode: bool = False,
         tools: list[dict] | None = None, timeout: float | None = None) -> dict:
    if not configured():
        raise AIError("INKSTONE_BASE_URL / INKSTONE_API_KEY chưa được cấu hình")
    if timeout is None:
        timeout = settings.inkstone_timeout
    body: dict = {"model": model, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if tools:
        body["tools"] = tools
    try:
        r = httpx.post(f"{settings.inkstone_base_url}/chat/completions", json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {settings.inkstone_api_key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]
    except httpx.HTTPError as e:
        raise AIError(f"Inkstone API lỗi: {e}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise AIError(f"Inkstone API lỗi: {e}") from e


def _text(model: str, system: str, user: str) -> str:
    return chat(model, [{"role": "system", "content": system},
                        {"role": "user", "content": user}]).get("content") or ""


MAX_FALLBACKS = 2   # extra models tried (same group) when the chosen one errors


def _with_fallback(group: list[str], model: str, call, disabled: frozenset = frozenset()) -> tuple[str, str]:
    """Run call(model); on AIError retry with up to MAX_FALLBACKS other models of the same group
    (admin-disabled models are never used). Returns (text, model_that_answered)."""
    if model not in group:
        raise AIError(f"Model không hợp lệ: {model}")
    if model in disabled:
        raise AIError(f"Model {model} đang được admin tạm tắt")
    order = [model] + [m for m in group if m != model and m not in disabled][:MAX_FALLBACKS]
    last: AIError | None = None
    for m in order:
        try:
            return call(m), m
        except AIError as e:
            last = e
            if not configured():
                break
    raise AIError(f"Tất cả model thử đều lỗi ({', '.join(order)}): {last}")


def generate_script_ex(topic: str, *, tone: str = "hấp dẫn", seconds: int = 15,
                       model: str = DEFAULT_SCRIPT_MODEL, disabled: frozenset = frozenset()) -> tuple[str, str]:
    return _with_fallback(LLM_MODELS, model, lambda m: _text(
        m, "Bạn là biên kịch video ngắn. Viết lời bình tiếng Việt, ngắn gọn, có hook mở đầu và CTA cuối.",
        f"Chủ đề: {topic}\nGiọng điệu: {tone}\nĐộ dài: khoảng {seconds} giây."), disabled)


def generate_script(topic: str, **kw) -> str:
    return generate_script_ex(topic, **kw)[0]


def analyze_image_ex(path: Path, question: str = "Mô tả nội dung hình ảnh này để làm kịch bản video.",
                     model: str = DEFAULT_VISION_MODEL, disabled: frozenset = frozenset()) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    content = [{"type": "text", "text": question},
               {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]
    return _with_fallback(VISION_MODELS, model,
                          lambda m: chat(m, [{"role": "user", "content": content}]).get("content") or "", disabled)


def analyze_image(path: Path, question: str = "Mô tả nội dung hình ảnh này để làm kịch bản video.",
                  model: str = DEFAULT_VISION_MODEL) -> str:
    return analyze_image_ex(path, question, model)[0]


DANCE_RULES = ("A single person dancing, full body visible from head to toe, facing the camera, plain simple "
               "background, static camera, no cuts, clear and distinct arm and leg movements, good lighting, "
               "no other people, nothing covering the body.")


def optimize_video_prompt(idea: str, *, dance: bool = False, allow_llm: bool = True) -> str:
    """Turn a short idea into a prompt a video model handles well. A dance reference must be shot so that
    MimicMotion can read the pose (one person, full body, plain background, static camera); those
    constraints are always appended, whether or not the LLM step succeeded."""
    idea = idea.strip()
    text = idea
    if configured() and allow_llm:
        try:
            text = _text(DEFAULT_SCRIPT_MODEL,
                         "You write prompts for a text-to-video model. Rewrite the user's idea (any language) as ONE "
                         "concise English paragraph: subject, action, setting, lighting, camera. No lists, no preamble."
                         + (" The clip is a dance reference: one performer, the dance style described precisely."
                            if dance else ""), idea).strip() or idea
        except AIError:
            text = idea
    return (f"{text} {DANCE_RULES}" if dance else text)[:2000]


def heuristic_route(template_schema: dict, input_data: dict) -> str:
    """Offline fallback: heavy only when the template asks for motion transfer
    and the user supplied a character image."""
    wants_motion = bool((template_schema or {}).get("motion_transfer"))
    return HEAVY if wants_motion and input_data.get("character_image") else FAST


def decide_render_type(category: str, template_schema: dict, input_data: dict) -> str:
    return decide_render_type_ex(category, template_schema, input_data)[0]


def decide_render_type_ex(category: str, template_schema: dict, input_data: dict,
                          use_model: bool = True) -> tuple[str, bool]:
    """SystemOne: ask intern-decision-4b which pipeline fits, fall back to heuristic.
    Returns (render_type, model_was_called) — the call can be billed."""
    fallback = heuristic_route(template_schema, input_data)
    if not configured() or not use_model:
        return fallback, False
    try:
        msg = chat(DECISION_MODEL, [
            {"role": "system", "content":
                'Chọn luồng kết xuất video. Trả về JSON {"render_type": "fast_remotion"|"heavy_mimicmotion"}. '
                "fast_remotion: chỉ thay chữ/logo/nền. heavy_mimicmotion: cần sao chép chuyển động vào ảnh nhân vật."},
            {"role": "user", "content": json.dumps({
                "category": category,
                "motion_transfer": bool((template_schema or {}).get("motion_transfer")),
                "layers": [l.get("type") for l in (template_schema or {}).get("layers", [])],
                "provided_inputs": sorted(input_data.keys()),
            }, ensure_ascii=False)},
        ], json_mode=True, timeout=30)
        m = re.search(r"\{.*\}", msg.get("content") or "", re.S)
        choice = json.loads(m.group(0)).get("render_type") if m else None
        # Never route to heavy without the data it needs.
        if choice == HEAVY and not input_data.get("character_image"):
            return FAST, True
        return (choice if choice in (FAST, HEAVY) else fallback), True
    except (AIError, ValueError):
        return fallback, False
