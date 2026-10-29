"""AI Assistant: Pipeline from idea -> script prompt & suggestions -> character image -> video generation & summary table."""
import json
import re
import secrets
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import ai, credits, pricing
from ..activity import log_activity
from ..config import MODEL_CATALOG, settings
from ..db import get_db
from ..deps import current_user
from ..media import resolve, url_for
from ..models import GenerationHistory, SharedVideoLibrary, User
from ..tasks import ai_render_task, broker_up, run_generation
from ..modelregistry import models_for_step, get_model, clip_count_for_duration, validate_selections
from ..modelregistry import ModelSpec
from .. import videopipeline
from ..autopipeline import _safe_str

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class IdeaIn(BaseModel):
    idea: str = Field(min_length=3, max_length=3000)
    model_llm: str = Field(default="deepseek-v4-pro-0813")
    model_image: str = Field(default="deepseek-v4-flash-vision")
    model_video: str = Field(default="wan2.1")
    duration: int = Field(default=60, ge=5, le=300)
    aspect_ratio: str = "9:16"
    character_image_url: str = ""


class PlanIn(BaseModel):
    idea: str = Field(min_length=3, max_length=3000)
    model_llm: str = Field(default="deepseek-v4-pro-0813")
    model_image: str = Field(default="deepseek-v4-flash-vision")
    model_video: str = Field(default="wan2.1")
    duration: int = Field(default=60, ge=5, le=300)
    aspect_ratio: str = "9:16"
    character_image_url: str = ""


@router.post("/plan")
def plan_storyboard(body: PlanIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Phân tích yêu cầu + tạo storyboard (kịch bản + các cảnh)."""
    from ..modelregistry import validate_selections
    errs = validate_selections({"llm": body.model_llm, "video_i2v": body.model_video})
    if errs:
        raise HTTPException(422, "; ".join(errs))

    if pricing.ai_key(body.model_llm) in pricing.catalog() and not pricing.enabled(db, pricing.ai_key(body.model_llm)):
        raise HTTPException(403, f"Model {body.model_llm} đang bị admin tắt")

    try:
        result = videopipeline.plan_storyboard(
            topic=body.idea,
            model_id=body.model_llm,
            duration_seconds=body.duration,
            aspect_ratio=body.aspect_ratio,
            character_image_url=body.character_image_url.strip() or None,
        )
    except Exception as e:
        raise HTTPException(502, f"Lỗi tạo storyboard: {e}")

    log_activity(db, user.id, "assistant_plan", details={
        "idea": body.idea[:100], "model_llm": body.model_llm,
        "model_video": body.model_video, "duration": body.duration,
        "num_scenes": result["tong_so_canh"],
    })
    return {
        "plan": result,
        "models": {"llm": body.model_llm, "video": body.model_video},
        "clip_count": videopipeline.clip_count_for_duration(
            videopipeline.get_model(body.model_video), body.duration) if videopipeline.get_model(body.model_video) else 1,
    }


class KeyframesIn(BaseModel):
    plan: dict
    model_image: str = Field(default="deepseek-v4-flash-vision")
    character_image_url: str = ""


@router.post("/keyframes")
def generate_keyframes(body: KeyframesIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Tạo ảnh từng cảnh (keyframe) từ storyboard."""
    from ..modelregistry import validate_selections
    errs = validate_selections({"image_generate": body.model_image})
    if errs:
        raise HTTPException(422, "; ".join(errs))

    plan = body.plan
    scenes = plan.get("cac_canh", [])
    character_bible = plan.get("character_bible", {})
    style_bible = plan.get("style_bible", {})

    keyframes = []
    for scene in scenes:
        prompt = videopipeline.gen_keyframe_prompt(
            scene, character_bible, style_bible,
            body.character_image_url.strip() or None,
        )
        keyframes.append({
            "stt": scene["stt"],
            "prompt": prompt,
            "scene": scene,
        })

    log_activity(db, user.id, "assistant_keyframes", details={
        "num_scenes": len(keyframes), "model_image": body.model_image,
    })
    return {
        "keyframes": keyframes,
        "model_image": body.model_image,
        "num_scenes": len(keyframes),
    }


class GenScenesIn(BaseModel):
    plan: dict
    keyframes: list[dict] = Field(default_factory=list)
    model_video: str = "wan2.1"
    resolution: str = "720P"
    ratio: str = "9:16"
    character_image_url: str = ""


@router.post("/gen-scenes")
def gen_scenes(body: GenScenesIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Chuẩn bị gen video từng cảnh."""
    from ..modelregistry import get_model
    vm = get_model(body.model_video)
    if not vm or vm.step not in ("video_i2v", "video_t2v"):
        raise HTTPException(422, f"Model {body.model_video} không phải model video")

    plan = body.plan
    scenes = plan.get("cac_canh", [])
    total_duration = plan.get("tong_thoi_luong", 60)

    estimated = 0.0

    fresh = db.get(User, user.id)
    if fresh.credits_balance < estimated and estimated > 0:
        raise HTTPException(402, detail={
            "message": "Không đủ credit, vui lòng nạp thêm",
            "required": estimated, "balance": fresh.credits_balance})

    log_activity(db, user.id, "assistant_gen_scenes", details={
        "model_video": body.model_video, "num_scenes": len(scenes),
        "estimated_cost": estimated,
    })
    return {
        "status": "ready",
        "num_scenes": len(scenes),
        "estimated_cost": estimated,
        "model_video": body.model_video,
        "clip_count_per_scene": [videopipeline.scene_clip_count(s, body.model_video) for s in scenes],
        "message": "Sẵn sàng gen video.",
    }


class EditIn(BaseModel):
    scene_video_urls: list[str] = Field(default_factory=list)
    transition: str = "fade"
    music_url: str = ""
    subtitle_srt: str = ""
    output_ratio: str = "9:16"


@router.post("/edit")
def edit_final(body: EditIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Dựng thành phẩm từ các clip rời (FFmpeg)."""
    if not body.scene_video_urls:
        raise HTTPException(422, "Cần ít nhất một clip để dựng")

    from ..media import resolve
    clips = []
    for url in body.scene_video_urls:
        p = resolve(url)
        if p and p.exists():
            clips.append(p)

    if not clips:
        raise HTTPException(422, "Không tìm thấy clip cục bộ để dựng")

    import subprocess
    import tempfile
    from pathlib import Path

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / f"final_{user.id}_{secrets.token_hex(6)}.mp4"

    concat_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = Path(f.name)
            for clip in clips:
                f.write(f"file '{clip.resolve().as_posix()}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-c:a", "aac", "-shortest", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            raise HTTPException(500, f"FFmpeg lỗi: {res.stderr[-500:]}")
    finally:
        if concat_file:
            concat_file.unlink(missing_ok=True)

    video_url = url_for(out)
    log_activity(db, user.id, "assistant_edit", details={
        "num_clips": len(clips), "output_url": video_url,
    })
    return {
        "output_video_url": video_url,
        "success": True,
        "num_clips": len(clips),
    }


class EstimateIn(BaseModel):
    selections: dict = Field(default_factory=dict)
    duration: int = Field(default=60)
    num_scenes: int = Field(default=6)


@router.post("/estimate")
def estimate_cost(body: EstimateIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Ước tính chi phí trước khi bấm Gen."""
    est = videopipeline.estimated_cost(
        body.selections, body.duration, body.num_scenes, db=db,
    )
    return est


# ==================== LOCAL PIPELINE ENDPOINTS ====================

@router.get("/local-pipeline/check")
def local_pipeline_check():
    """Kiểm tra trạng thái pipeline local."""
    from ..localpipeline.config import detect_vram_gb
    vram = detect_vram_gb()
    requirements = {
        "Qwen2-VL-7B-Instruct": {"vram_gb": 8, "disk_gb": 14},
        "FLUX.1-schnell": {"vram_gb": 8, "disk_gb": 36},
        "Wan2.1-T2V-14B": {"vram_gb": 24, "disk_gb": 59},
        "Wan2.1-T2V-1.3B": {"vram_gb": 8, "disk_gb": 10},
    }
    return {
        "vram_gb": vram,
        "has_gpu": vram > 0,
        "requirements": requirements,
        "feasible": {
            "Qwen2-VL-7B-Instruct": vram >= 4,
            "FLUX.1-schnell": vram >= 8,
            "Wan2.1-T2V-14B": vram >= 24,
            "Wan2.1-T2V-1.3B": vram >= 8,
        },
        "warnings": [] if vram >= 8 else ["VRAM < 8GB: chỉ chay duoc model nho voi quantization"],
    }


class LocalVLMIn(BaseModel):
    image_url: str = Field(min_length=1)
    user_instruction: str = ""
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"


@router.post("/local-vlm")
def local_vlm(body: LocalVLMIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Buoc 1: Chay VLM local de tao prompt tu anh."""
    from ..localpipeline.vlm import run_vlm, VLMError
    from ..localpipeline.config import PipelineConfig

    p = resolve(body.image_url)
    if not p or not p.exists():
        raise HTTPException(422, "Khong tim thay anh dau vao")

    config = PipelineConfig()
    config.vlm_model_id = body.model_id

    try:
        result = run_vlm(str(p), user_instruction=body.user_instruction or None, config=config)
    except VLMError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"Loi chay VLM: {e}")

    log_activity(db, user.id, "local_vlm", details={
        "model": body.model_id, "image_url": body.image_url,
    })
    return result


class LocalImageGenIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    seed: int = 42
    model_id: str = "black-forest-labs/FLUX.1-schnell"


@router.post("/local-image-gen")
def local_image_gen(body: LocalImageGenIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Buoc 2: Chay FLUX.1-schnell de tao anh tu prompt."""
    from ..localpipeline.image_gen import run_image_gen, ImageGenError
    from ..localpipeline.config import PipelineConfig

    config = PipelineConfig()
    config.t2i_model_id = body.model_id

    try:
        image_path = run_image_gen(body.prompt, seed=body.seed, config=config)
    except ImageGenError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"Loi tao anh: {e}")

    log_activity(db, user.id, "local_image_gen", details={
        "model": body.model_id, "prompt": body.prompt[:100],
    })
    return {"image_url": url_for(Path(image_path)), "image_path": image_path}


class LocalPipelineIn(BaseModel):
    image_url: str = ""
    idea: str = ""
    user_instruction: str = ""
    mode: str = "recreate"
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    t2i_model_id: str = "black-forest-labs/FLUX.1-schnell"
    i2v_model_id: str = "Wan-AI/Wan2.1-T2V-14B"
    duration: int = Field(default=5, ge=1, le=30)


@router.post("/local-pipeline")
def local_pipeline(body: LocalPipelineIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Chay pipeline toan bo: idea -> storyboard -> anh -> video -> thanh pham."""
    from ..localpipeline.pipeline import run_pipeline, PipelineError

    p = resolve(body.image_url) if body.image_url else None
    if body.image_url and (not p or not p.exists()):
        raise HTTPException(422, "Khong tim thay anh dau vao")

    options = {
        "vlm_model_id": body.vlm_model_id,
        "t2i_model_id": body.t2i_model_id,
        "i2v_model_id": body.i2v_model_id,
        "mode": body.mode,
    }

    try:
        result = run_pipeline(
            image_path=str(p) if p else "",
            user_instruction=(body.user_instruction or body.idea) or None,
            mode=body.mode,
            options=options,
        )
        log_activity(db, user.id, "local_pipeline_done", details={
            "job_id": result.job_id, "idea": (body.user_instruction or body.idea or "")[:100],
            "result": result.to_dict(),
        })
        out = result.to_dict()
        if result.image_path and Path(result.image_path).exists():
            out["image_url"] = url_for(Path(result.image_path))
        if result.video_path and Path(result.video_path).exists():
            out["video_url"] = url_for(Path(result.video_path))
        return {
            "status": "done",
            "message": "Pipeline hoan tat.",
            "result": out,
        }
    except Exception as e:
        log_activity(db, user.id, "local_pipeline_failed", details={
            "image_url": body.image_url,
            "mode": body.mode,
            "error": str(e),
        })
        raise HTTPException(502, str(e))


# ==================== REMOTE GPU QUEUE (Kaggle / rented GPU) ====================

class RemotePipelineIn(BaseModel):
    idea: str = Field(min_length=3, max_length=3000)
    duration: int = Field(default=5, ge=1, le=30)
    mode: str = "recreate"
    user_instruction: str = ""


class RemoteJobResult(BaseModel):
    status: str  # done | failed
    result: dict | None = None
    error: str | None = None


# In-memory job store (dev/demo). Worker (Kaggle) polls /next, UI polls /status.
REMOTE_JOBS: dict[str, dict] = {}
REMOTE_JOBS_LOCK = __import__("threading").Lock()


def _new_job_id() -> str:
    return "rpipe_" + secrets.token_hex(8)


@router.post("/remote-pipeline/submit")
def remote_pipeline_submit(body: RemotePipelineIn, user: User = Depends(current_user)):
    """Nguoi dung gui job sang GPU xa (Kaggle). Tra job_id de the doi tien do."""
    job_id = _new_job_id()
    job = {
        "job_id": job_id,
        "user_id": user.id,
        "idea": body.idea,
        "duration": body.duration,
        "mode": body.mode,
        "user_instruction": body.user_instruction,
        "status": "queued",   # queued | processing | done | failed
        "progress": 0,
        "step": "",
        "error": "",
        "result": None,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    with REMOTE_JOBS_LOCK:
        REMOTE_JOBS[job_id] = job
    return {"job_id": job_id, "status": "queued"}


@router.get("/remote-pipeline/jobs/next")
def remote_pipeline_next():
    """Worker (Kaggle) goi lay lam viec tiep theo trong hang doi."""
    with REMOTE_JOBS_LOCK:
        for job in REMOTE_JOBS.values():
            if job["status"] == "queued":
                job["status"] = "processing"
                job["step"] = "Dang lay nhiem vu"
                return job
    return {"job_id": None}


@router.post("/remote-pipeline/jobs/{job_id}/result")
def remote_pipeline_result(job_id: str, body: RemoteJobResult):
    """Worker (Kaggle) bao ket qua: video duoc upload tro ve tool."""
    with REMOTE_JOBS_LOCK:
        job = REMOTE_JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Khong tim thay job")
        job["status"] = body.status
        job["result"] = body.result or {}
        job["error"] = body.error or ""
        if body.status == "done":
            job["progress"] = 100
            job["step"] = "Hoan tat"
        elif body.status == "failed":
            job["step"] = "That bai"
    return {"ok": True}


@router.get("/remote-pipeline/jobs/{job_id}/status")
def remote_pipeline_status(job_id: str, user: User = Depends(current_user)):
    """UI kiem tra tien do job (poll moi 2s)."""
    with REMOTE_JOBS_LOCK:
        job = REMOTE_JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Khong tim thay job")
        if job["user_id"] != user.id and getattr(user, "role", "") != "admin":
            raise HTTPException(403, "Khong co quyen xem job nay")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "step": job["step"],
            "error": job["error"],
            "result": job["result"],
        }


@router.post("/remote-pipeline/upload")
def remote_pipeline_upload(job_id: str, file: UploadFile = File(...)):
    """Worker (Kaggle) up file video/anh tro ve tool; luu vao outputs, tra URL."""
    job = None
    with REMOTE_JOBS_LOCK:
        job = REMOTE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Khong tim thay job")
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / f"remote_{job_id}_{secrets.token_hex(4)}_{Path(file.filename or 'file').name}"
    with out.open("wb") as f:
        f.write(file.file.read())
    return {"url": url_for(out)}


@router.get("/models")
def list_models(step: str = "", db: Session = Depends(get_db)):
    """Models pickable for a pipeline step (llm | image_generate | video_i2v ...), with admin price."""
    from ..modelregistry import ALL_MODELS, STEPS
    snap = pricing.snapshot(db)

    def _price(mid: str) -> tuple[float, str]:
        for key in (f"m.{mid}", pricing.ai_key(mid)):
            row = snap.get(key)
            cat = pricing.catalog().get(key)
            if row is not None or cat is not None:
                unit = (cat or {}).get("unit", "")
                return (row.price if row else (cat or {}).get("default", 0.0)), unit
        return 0.0, ""

    def _out(m) -> dict:
        p, unit = _price(m.id)
        enabled = True
        for key in (f"m.{m.id}", pricing.ai_key(m.id)):
            row = snap.get(key)
            if row is not None:
                enabled = row.enabled
        label = f"{p:g} credit" if p > 0 else "Free"
        return {"id": m.id, "name": m.name, "step": m.step, "group": m.step,
                "price": p, "unit": unit, "enabled": enabled and m.enabled,
                "price_label": f"{label} - {m.name}"}

    steps = [step] if step else ["llm", "image_generate", "video_i2v"]
    out = {"models": [_out(m) for m in ALL_MODELS if m.step in steps],
           "steps": {s: STEPS.get(s, s) for s in steps}}
    defaults = {"llm": "deepseek-v4-pro-0813", "image_generate": "deepseek-v4-flash-vision",
                "video_i2v": "wan2.1"}
    out["default"] = defaults.get(step, "wan2.1")
    return out


@router.get("/check-image-models")
def check_image_models(db: Session = Depends(get_db)):
    """Enabled image-generation models (image_generate step) with price."""
    from ..modelregistry import ALL_MODELS
    snap = pricing.snapshot(db)
    models = []
    for m in ALL_MODELS:
        if m.step != "image_generate":
            continue
        row = snap.get(f"m.{m.id}") or snap.get(pricing.ai_key(m.id))
        models.append({"id": m.id, "name": m.name,
                       "price": row.price if row else 0.0,
                       "enabled": (row.enabled if row else True) and m.enabled})
    return {"models": models}