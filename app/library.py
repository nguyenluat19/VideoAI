"""Turn finished renders into reviewable library entries; thumbnails."""
import copy
import logging
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from .config import settings
from .media import resolve, url_for
from .models import GenerationHistory, SharedVideoLibrary

log = logging.getLogger("library")


def make_thumbnail(video: Path, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    for ts in ("0.5", "0"):          # very short clips have no frame at 0.5s
        proc = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-ss", ts, "-i", str(video),
                               "-frames:v", "1", "-vf", "scale=360:-2", str(out)],
                              capture_output=True, timeout=60)
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return True
    return False


def thumbnail_url_for(video_url: str, name: str) -> str:
    """Best-effort: '' when the source is missing/unreadable (e.g. an image, a broken file)."""
    src = resolve(video_url)
    if not src or src.suffix.lower() not in (".mp4", ".mov", ".webm"):
        return ""
    out = settings.outputs_dir / "thumbs" / f"{name}.jpg"
    try:
        return url_for(out) if make_thumbnail(src, out) else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def schema_with_defaults(schema: dict, inputs: dict) -> dict:
    """Copy of `schema` where every value the user supplied becomes the new default, so anyone
    reusing the entry starts from this video but can still change every layer."""
    out = copy.deepcopy(schema or {})
    inputs = {k: v for k, v in inputs.items() if not k.startswith("_")}   # internal pipeline state
    layer_ids = set()
    for layer in out.get("layers", []):
        layer_ids.add(layer["id"])
        if layer["id"] in inputs:
            layer["default"] = inputs[layer["id"]]
    extra = {k: v for k, v in inputs.items() if k not in layer_ids}   # e.g. character_image
    if extra:
        out["defaults"] = {**out.get("defaults", {}), **extra}
    return out


def create_entry_from_generation(db: Session, gen_id: int) -> int | None:
    gen = db.get(GenerationHistory, gen_id)
    if gen is None or gen.status != "success" or not gen.share_to_library:
        return None
    if db.query(SharedVideoLibrary.id).filter_by(source_generation_id=gen_id).first():
        return None
    tpl = db.get(SharedVideoLibrary, gen.template_id)
    entry = SharedVideoLibrary(
        title=tpl.title, category=tpl.category, base_video_url=tpl.base_video_url,
        thumbnail_url=thumbnail_url_for(gen.output_video_url, f"gen_{gen_id}"),
        preview_video_url=gen.output_video_url,
        template_schema=schema_with_defaults(tpl.template_schema, gen.input_data or {}),
        status="private",                # never public until an admin approves
        creator_id=gen.user_id, source_generation_id=gen_id)
    db.add(entry)
    db.commit()
    return entry.id
