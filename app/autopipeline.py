"""Pipeline automation: idea -> storyboard -> images -> video per scene -> final edit.

Mỗi bước dùng MỘT model do người dùng chọn, không tự chuyển model khi lỗi.
Chạy tự động end-to-end, khách hàng chỉ cần nhập yêu cầu và chọn model mỗi giai đoạn.
"""
import json
import math
import re
import secrets
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from . import ai, minimax, pricing
from .config import settings
from .db import SessionLocal
from .media import resolve, url_for
from .modelregistry import clip_count_for_duration, get_model, models_for_step
from .tasks import ai_render_task, broker_up, run_generation
from . import videopipeline as vp


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(s: Any, max_len: int = 500) -> str:
    """Safely convert to string, preserving Unicode characters."""
    try:
        text = str(s)
    except Exception:
        text = repr(s)
    # Encode to UTF-8 then decode with errors='replace' to handle any problematic chars
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        pass
    return text[:max_len]


# ---------------------------------------------------------------------------
# In-memory pipeline job store (dev/local). Prod có thể chuyển sang DB/Redis.
# ---------------------------------------------------------------------------
class PipelineJob:
    def __init__(self, job_id: str, user_id: int, selections: dict,
                 idea: str, duration: int, aspect_ratio: str,
                 character_image_url: str | None):
        self.job_id = job_id
        self.user_id = user_id
        self.selections = selections
        self.idea = idea
        self.duration = duration
        self.aspect_ratio = aspect_ratio
        self.character_image_url = character_image_url
        self.status = "queued"  # queued | planning | keyframing | rendering | editing | done | failed
        self.progress = 0
        self.step = ""
        self.error = ""
        self.result: dict | None = None
        self.created_at = _now()
        self.updated_at = _now()
        self._lock = threading.Lock()

    def update(self, status: str = None, progress: int = None, step: str = None,
               error: str = None, result: dict = None):
        with self._lock:
            if status is not None:
                self.status = status
            if progress is not None:
                self.progress = progress
            if step is not None:
                self.step = step
            if error is not None:
                self.error = error
            if result is not None:
                self.result = result
            self.updated_at = _now()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "step": self.step,
                "error": self.error,
                "result": self.result,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "selections": self.selections,
            }


JOBS: dict[str, PipelineJob] = {}
JOBS_LOCK = threading.Lock()


def create_job(user_id: int, selections: dict, idea: str, duration: int,
               aspect_ratio: str, character_image_url: str | None = None) -> str:
    job_id = "pipe_" + secrets.token_hex(8)
    job = PipelineJob(job_id, user_id, selections, idea, duration, aspect_ratio,
                      character_image_url)
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job_id


def get_job(job_id: str) -> PipelineJob | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run_planning(job: PipelineJob) -> dict:
    """Bước 1: Tạo storyboard bằng VLM local (Qwen2-VL-7B-Instruct)."""
    job.update(status="planning", progress=10, step="Đang tạo kịch bản...")
    from .localpipeline.vlm import run_vlm, VLMError
    from .localpipeline.config import PipelineConfig
    import os
    config = PipelineConfig()
    config.vlm_model_id = job.selections.get("vlm_model_id", "Qwen/Qwen2-VL-7B-Instruct")
    try:
        result = run_vlm(str(job.character_image_url) if job.character_image_url else None,
                        user_instruction=job.idea, config=config)
        return result
    except VLMError as e:
        raise ValueError(f"VLM lỗi: {e}")
    except Exception as e:
        raise ValueError(f"Lỗi chạy VLM: {e}")


def _run_keyframing(job: PipelineJob, plan: dict) -> list[dict]:
    """Bước 2: Tạo prompt ảnh cho từng cảnh (keyframe prompts).

    Lưu ý: Model ảnh (image_generate) hiện chỉ dùng để sinh prompt chi tiết.
    Việc gọi API sinh ảnh thực sự phụ thuộc nhà cung cấp bên ngoài (Qwen-Image,
    FLUX, v.v.) chưa kết nối vào hệ thống này. Nếu cần, admin có thể kết nối
    adapter tương ứng.
    """
    job.update(status="keyframing", progress=35, step="Đang chuẩn bị ảnh từng cảnh...")
    scenes = plan["các_cảnh"]
    character_bible = plan.get("character_bible", {})
    style_bible = plan.get("style_bible", {})

    keyframes = []
    for scene in scenes:
        prompt = vp.gen_keyframe_prompt(
            scene, character_bible, style_bible,
            job.character_image_url,
        )
        keyframes.append({
            "stt": scene["stt"],
            "thời_lượng_giây": scene["thời_lượng_giây"],
            "loại_shot": scene["loại_shot"],
            "góc_máy": scene["góc_máy"],
            "hành_động": scene["hành_động"],
            "biểu_cảm": scene["biểu_cảm"],
            "thoại": scene["thoại"],
            "image_prompt": prompt,
            "tỉ_lệ_khung_hình": scene["tỉ_lệ_khung_hình"],
            "video_url": "",
            "status": "pending",
        })
    return keyframes


def _run_rendering(job: PipelineJob, keyframes: list[dict], db) -> list[dict]:
    """Bước 3: Gen video từng cảnh bằng model video đã chọn (MiniMax-H3)."""
    job.update(status="rendering", progress=60, step="Đang tạo video từng cảnh...")
    video_model_id = job.selections["video_i2v"]
    model = get_model(video_model_id)
    if not model or model.step not in ("video_i2v", "video_t2v"):
        raise ValueError(f"Model {video_model_id} không phải model video")

    # Check pricing for this model
    model_price_key = f"m.{video_model_id}"
    model_price = pricing.price(db, model_price_key) if model_price_key in pricing.catalog() else 0.0

    total_seconds = job.duration
    per_scene_sec = max(1, total_seconds // max(len(keyframes), 1))

    # Charge for video rendering
    video_cost = round(total_seconds * model_price, 2)
    from app import credits
    if video_cost > 0:
        fresh = db.get(__import__("app.models", fromlist=["User"]).User, job.user_id)
        if not credits.charge(db, job.user_id, video_cost):
            raise ValueError(f"Không đủ credit để render video (cần {video_cost} credit)")
        db.commit()

    updated = []
    for i, kf in enumerate(keyframes):
        step_progress = 60 + int((i / max(len(keyframes), 1)) * 25)
        job.update(progress=step_progress, step=f"Đang render cảnh {i + 1}/{len(keyframes)}...")

        prompt = kf["image_prompt"]
        dur = min(kf["thời_lượng_giây"], model.max_clip_seconds or 10)

        # First frame: dùng ảnh nhân vật nếu có (giữ nhất quán nhân vật)
        first_frame = job.character_image_url

        try:
            if video_model_id.startswith("minimax-h3"):
                # Gọi MiniMax-H3 trực tiếp (blocking, chạy trong luồng nền)
                dest = settings.outputs_dir / f"pipe_{job.job_id}_scene_{kf['stt']}.mp4"
                minimax.generate(
                    prompt=prompt,
                    dest=dest,
                    first_frame=resolve(first_frame) if first_frame else None,
                    duration=dur,
                    ratio=job.aspect_ratio,
                    resolution="768P",
                )
                kf["video_url"] = url_for(dest)
            elif video_model_id == "hunyuan-video":
                # HunyuanVideo (local CLI)
                from .local_video_adapter import generate_hunyuan_video, LocalVideoError
                result = generate_hunyuan_video(
                    prompt=prompt,
                    duration=dur,
                    resolution="720P",
                )
                kf["video_url"] = result["output_video_url"]
            elif video_model_id == "hunyuan-video-i2v":
                # HunyuanVideo-I2V (image-to-video, local CLI)
                from .local_video_adapter import generate_hunyuan_i2v, LocalVideoError
                result = generate_hunyuan_i2v(
                    prompt=prompt,
                    first_frame_image_url=first_frame,
                    duration=dur,
                    resolution="720P",
                )
                kf["video_url"] = result["output_video_url"]
            elif video_model_id == "wan2.1":
                # Wan2.1 (local CLI)
                from .local_video_adapter import generate_wan21_video, LocalVideoError
                result = generate_wan21_video(
                    prompt=prompt,
                    duration=dur,
                    resolution="720P",
                )
                kf["video_url"] = result["output_video_url"]
            else:
                # Fallback: chưa hỗ trợ pipeline tự động
                kf["video_url"] = ""
                kf["error"] = f"Model {video_model_id} chưa được kết nối vào pipeline tự động"
        except Exception as e:
            kf["video_url"] = ""
            kf["error"] = _safe_str(e)

        kf["status"] = "done" if kf.get("video_url") else "failed"
        updated.append(kf)

    return updated


def _run_editing(job: PipelineJob, keyframes: list[dict]) -> str:
    """Bước 4: Dựng thành phẩm từ các clip (FFmpeg concat)."""
    job.update(status="editing", progress=90, step="Đang dựng thành phẩm cuối...")
    video_urls = [kf["video_url"] for kf in keyframes if kf.get("video_url")]
    if not video_urls:
        # Collect error messages from failed clips
        errors = [_safe_str(kf.get("error", "Loi khong ro")) for kf in keyframes if not kf.get("video_url")]
        error_detail = "; ".join(errors[:3]) if errors else "Khong ro nguyen nhan"
        raise ValueError("Khong co clip nao de duoc thanh pham. Chi tiet: " + error_detail)

    from app.media import resolve
    clips = []
    for url in video_urls:
        p = resolve(url)
        if p and p.exists():
            clips.append(p)

    if not clips:
        raise ValueError("Không tìm thấy file clip cục bộ để dựng")

    import subprocess
    import tempfile
    from pathlib import Path

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / f"pipeline_{job.user_id}_{secrets.token_hex(6)}.mp4"

    concat_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = Path(f.name)
            for clip in clips:
                f.write(f"file '{clip.resolve().as_posix()}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
               "-c:a", "aac", "-shortest", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg lỗi: {res.stderr[-500:]}")
    finally:
        if concat_file:
            concat_file.unlink(missing_ok=True)

    return url_for(out)


# ---------------------------------------------------------------------------
# Full pipeline runner (runs in background thread)
# ---------------------------------------------------------------------------

def _run_full_pipeline(job: PipelineJob):
    try:
        # Step 1: Planning
        plan = _run_planning(job)
        job.update(result={"plan": plan, "keyframes": [], "video_urls": [], "final_url": ""})

        # Step 2: Keyframing
        keyframes = _run_keyframing(job, plan)
        job.result["keyframes"] = keyframes
        job.update(result=job.result, progress=50)

        # Step 3: Rendering
        with SessionLocal() as db:
            keyframes = _run_rendering(job, keyframes, db)
        job.result["keyframes"] = keyframes
        job.update(result=job.result, progress=85)

        # Step 4: Editing
        final_url = _run_editing(job, keyframes)
        job.result["final_url"] = final_url
        job.update(status="done", progress=100, step="Hoàn tất!", result=job.result)

    except Exception as e:  # noqa: BLE001
        safe_err = _safe_str(e)
        job.update(status="failed", error=safe_err, step="Loi: " + safe_err[:200])
def start_pipeline(job_id: str):
    """Chạy pipeline trong luồng nền."""
    job = get_job(job_id)
    if not job:
        return
    thread = threading.Thread(target=_run_full_pipeline, args=(job,), daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Convenience: list available models per step for UI
# ---------------------------------------------------------------------------

def available_steps() -> dict:
    from .modelregistry import STEPS
    out = {}
    for s in ("llm", "image_generate", "video_i2v"):
        ms = models_for_step(s)
        if ms:
            out[s] = {
                "label": STEPS.get(s, s),
                "models": [m.to_dict() for m in ms],
            }
    return out
