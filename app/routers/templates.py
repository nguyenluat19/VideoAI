import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .. import ai
from .. import credits, library, pricing
from ..config import MODEL_CATALOG, settings
from ..db import get_db
from ..deps import current_user
from ..media import resolve, url_for
from ..models import SharedVideoLibrary, User

router = APIRouter(prefix="/api", tags=["templates"])

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm"}
MAX_UPLOAD = 200 * 1024 * 1024
LAYER_TYPES = {"text", "image", "logo", "background"}


class TemplateIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    category: str = Field(default="full", pattern=r"^(hook|cta|full)$")
    base_video_url: str = ""
    thumbnail_url: str = ""
    template_schema: dict = Field(default_factory=dict)


def validate_schema(schema: dict) -> None:
    if not isinstance(schema.get("defaults", {}), dict):
        raise HTTPException(422, "template_schema.defaults phải là object")
    layers = schema.get("layers", [])
    if not isinstance(layers, list):
        raise HTTPException(422, "template_schema.layers phải là danh sách")
    ids = set()
    for l in layers:
        if not isinstance(l, dict) or l.get("type") not in LAYER_TYPES or not l.get("id"):
            raise HTTPException(422, f"Layer không hợp lệ (type thuộc {sorted(LAYER_TYPES)}, cần id): {l}")
        if l["id"] in ids:
            raise HTTPException(422, f"Trùng layer id: {l['id']}")
        ids.add(l["id"])


def tpl_out(t: SharedVideoLibrary) -> dict:
    return {"id": t.id, "title": t.title, "category": t.category, "base_video_url": t.base_video_url,
            "thumbnail_url": t.thumbnail_url, "preview_video_url": t.preview_video_url,
            "source_generation_id": t.source_generation_id, "template_schema": t.template_schema, "status": t.status,
            "creator_id": t.creator_id, "usage_count": t.usage_count, "created_at": t.created_at}


def visible_to(t: SharedVideoLibrary, user: User) -> bool:
    return t.status == "publish" or t.creator_id == user.id or user.role == "admin"


@router.post("/uploads")
async def upload(file: UploadFile = File(...), user: User = Depends(current_user)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Định dạng không hỗ trợ: {ext or '?'}")
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.uploads_dir / f"{user.id}_{secrets.token_hex(8)}{ext}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File quá lớn (tối đa 200MB)")
            f.write(chunk)
    return {"url": url_for(dest)}


@router.get("/templates")
def list_templates(category: str | None = None, mine: bool = False,
                   db: Session = Depends(get_db), user: User = Depends(current_user)):
    q = select(SharedVideoLibrary)
    if mine:
        q = q.where(SharedVideoLibrary.creator_id == user.id)
    else:
        q = q.where(SharedVideoLibrary.status == "publish")
    if category:
        q = q.where(SharedVideoLibrary.category == category)
    q = q.order_by(SharedVideoLibrary.usage_count.desc(), SharedVideoLibrary.id.desc())
    return [tpl_out(t) for t in db.scalars(q)]


@router.post("/templates")
def create_template(body: TemplateIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    validate_schema(body.template_schema)
    for u in (body.base_video_url, body.thumbnail_url):
        if u and not resolve(u):
            raise HTTPException(422, f"File chưa được upload: {u}")
    t = SharedVideoLibrary(**body.model_dump(), creator_id=user.id, status="private")  # always private
    db.add(t)
    db.commit()
    if t.base_video_url and not t.thumbnail_url:
        t.thumbnail_url = library.thumbnail_url_for(t.base_video_url, f"tpl_{t.id}")
        db.commit()
    return tpl_out(t)


@router.get("/templates/{tid}")
def get_template(tid: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = db.get(SharedVideoLibrary, tid)
    if not t or not visible_to(t, user):
        raise HTTPException(404, "Không tìm thấy template")
    return tpl_out(t)


# ---- AI helpers (free, no credit charge) ----
class ScriptIn(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    tone: str = "hấp dẫn"
    seconds: int = Field(default=15, ge=5, le=120)
    model: str = ai.DEFAULT_SCRIPT_MODEL


class AnalyzeIn(BaseModel):
    image_url: str
    question: str = "Mô tả nội dung hình ảnh này để làm kịch bản video."
    model: str = ai.DEFAULT_VISION_MODEL


def _bill_model(db: Session, user: User, model: str) -> float:
    """Charge the price of the model the user picked (0 = free). Refused if the admin switched it off."""
    if pricing.ai_key(model) not in pricing.catalog():
        return 0.0                                  # unknown id: the AI layer will reject it, nothing to charge
    if not pricing.enabled(db, pricing.ai_key(model)):
        raise HTTPException(403, f"Model {model} đang được admin tạm tắt")
    p = pricing.price(db, pricing.ai_key(model))
    if p > 0 and not credits.charge(db, user.id, p):
        db.rollback()
        raise HTTPException(402, detail={"message": "Không đủ credit, vui lòng nạp thêm", "required": p})
    db.commit()
    return p


def _refund_model(db: Session, user: User, amount: float) -> None:
    if amount > 0:
        credits.add_credits(db, user.id, amount)
        db.commit()


@router.get("/ai/models")
def ai_models(db: Session = Depends(get_db), user: User = Depends(current_user)):
    snap = pricing.snapshot(db)
    models = [{**m, "price": snap[pricing.ai_key(m["id"])].price if pricing.ai_key(m["id"]) in snap else 0,
               "enabled": snap[pricing.ai_key(m["id"])].enabled if pricing.ai_key(m["id"]) in snap else True}
              for m in MODEL_CATALOG]
    return {"models": models, "default_vision": ai.DEFAULT_VISION_MODEL,
            "default_script": ai.DEFAULT_SCRIPT_MODEL, "decision": ai.DECISION_MODEL,
            "configured": ai.configured()}


@router.post("/ai/script")
def ai_script(body: ScriptIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    paid = _bill_model(db, user, body.model)
    try:
        text, used = ai.generate_script_ex(body.topic, tone=body.tone, seconds=body.seconds, model=body.model,
                                           disabled=frozenset(pricing.disabled_models(db)))
        return {"script": text, "model_used": used, "credits_charged": paid}
    except ai.AIError as e:
        _refund_model(db, user, paid)               # no answer, no charge
        raise HTTPException(502, str(e))


@router.post("/ai/analyze")
def ai_analyze(body: AnalyzeIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = resolve(body.image_url)
    if not p:
        raise HTTPException(404, "Ảnh chưa được upload")
    paid = _bill_model(db, user, body.model)
    try:
        text, used = ai.analyze_image_ex(p, body.question, body.model,
                                         disabled=frozenset(pricing.disabled_models(db)))
        return {"analysis": text, "model_used": used, "credits_charged": paid}
    except ai.AIError as e:
        _refund_model(db, user, paid)
        raise HTTPException(502, str(e))
