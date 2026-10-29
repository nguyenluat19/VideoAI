from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import ai, credits, minimax, pricing
from ..activity import log_activity
from ..config import settings
from ..db import get_db
from ..deps import current_user
from ..media import resolve
from ..models import GenerationHistory, SharedVideoLibrary, User
from ..tasks import ai_render_task, broker_up, heavy_render_task, run_generation
from .templates import visible_to

router = APIRouter(prefix="/api", tags=["generate"])


class GenerateIn(BaseModel):
    template_id: int
    render_type: str = Field(default="auto", pattern=r"^(auto|fast_remotion|heavy_mimicmotion|ai_video|ai_dance)$")
    input_data: dict = Field(default_factory=dict)
    # Finished videos are offered to the admin library (private until approved). Opt out per render.
    share_to_library: bool = True
    user_category: str = "Mặc định"
    title: str = ""


def cost_of(db: Session, render_type: str, inputs: dict | None = None) -> float:
    """Price of a job, read from the admin-editable pricing table."""
    inputs = inputs or {}
    if render_type == ai.HEAVY:
        return pricing.price(db, pricing.K_MIMIC)
    if render_type in (ai.AI_VIDEO, ai.AI_DANCE):
        api = pricing.video_cost(db, int(inputs.get("duration", 5)), inputs.get("resolution", "768P"))
        return round(api + (pricing.price(db, pricing.K_MIMIC) if render_type == ai.AI_DANCE else 0), 2)
    return pricing.price(db, pricing.K_FAST)


def ensure_enabled(db: Session, render_type: str, inputs: dict) -> None:
    """An admin can switch a flow or a video resolution off; refuse before charging anything."""
    needs = {ai.FAST: [(pricing.K_FAST, "Render nhanh")],
             ai.HEAVY: [(pricing.K_MIMIC, "MimicMotion")],
             ai.AI_VIDEO: [(pricing.K_H3_2K if inputs.get("resolution") == "2K" else pricing.K_H3_768,
                            f"MiniMax-H3 {inputs.get('resolution', '768P')}")],
             ai.AI_DANCE: [(pricing.K_H3_2K if inputs.get("resolution") == "2K" else pricing.K_H3_768,
                            f"MiniMax-H3 {inputs.get('resolution', '768P')}"), (pricing.K_MIMIC, "MimicMotion")]}
    for key, name in needs.get(render_type, []):
        if not pricing.enabled(db, key):
            raise HTTPException(422, f"{name} đang được admin tạm tắt")


def normalize_ai_inputs(inputs: dict, *, need_prompt: bool, dance: bool) -> dict:
    """Validate/complete the inputs of an AI-video template or raise 422."""
    out = dict(inputs)
    prompt = str(out.get("prompt", "")).strip()
    if need_prompt and not prompt:
        raise HTTPException(422, "Cần nhập mô tả (prompt) cho video")
    if len(prompt) > 2000:
        raise HTTPException(422, "Prompt tối đa 2000 ký tự")
    try:
        duration = int(out.get("duration", 5))
    except (TypeError, ValueError):
        raise HTTPException(422, "duration phải là số giây")
    top = 10 if dance else minimax.MAX_SECONDS
    if not minimax.MIN_SECONDS <= duration <= top:
        raise HTTPException(422, f"duration phải từ {minimax.MIN_SECONDS} đến {top} giây")
    ratio, res = out.get("ratio", "9:16"), out.get("resolution", "768P")
    if ratio not in minimax.RATIOS or res not in minimax.RESOLUTIONS:
        raise HTTPException(422, f"ratio thuộc {minimax.RATIOS}, resolution thuộc {minimax.RESOLUTIONS}")
    for key in ("first_frame_image", "reference_video", "character_image"):
        if out.get(key) and not resolve(out[key]):
            raise HTTPException(422, f"{key} chưa được upload")
    out.update(prompt=prompt, duration=duration, ratio=ratio, resolution=res)
    return out


def plan_ai(sc: dict, inputs: dict, requested: str) -> tuple[str, dict]:
    """AI-video templates: pick the pipeline from what the user provided (no guessing, no model call)."""
    dance = bool(sc.get("dance"))
    own_ref = bool(inputs.get("reference_video"))
    if dance and own_ref:       # the user already has a dance clip -> skip MiniMax, MimicMotion only
        rt = ai.HEAVY
    else:
        rt = ai.AI_DANCE if dance else ai.AI_VIDEO
    if requested not in ("auto", rt):
        raise HTTPException(422, f"Template này chạy bằng luồng {rt}")
    out = normalize_ai_inputs(inputs, need_prompt=rt != ai.HEAVY, dance=dance)
    if rt in (ai.AI_DANCE, ai.HEAVY) and not out.get("character_image"):
        raise HTTPException(422, "Cần upload ảnh nhân vật (character_image)")
    if rt != ai.HEAVY and not minimax.configured():
        raise HTTPException(503, "Chưa cấu hình MINIMAX_API_KEY nên chưa tạo được video AI")
    return rt, out


class Prepared:
    def __init__(self, user, template, render_type, cost, input_data):
        self.user, self.template, self.render_type, self.cost = user, template, render_type, cost
        self.input_data = input_data


def credit_guard(body: GenerateIn, user: User = Depends(current_user),
                 db: Session = Depends(get_db)) -> Prepared:
    """Mandatory pre-flight: template access, routing, then balance check (402 if short)."""
    tpl = db.get(SharedVideoLibrary, body.template_id)
    if not tpl or not visible_to(tpl, user):
        raise HTTPException(404, "Không tìm thấy template")
    # Library entries carry the original inputs as defaults; the caller's values override them.
    sc = tpl.template_schema or {}
    routing_fee = 0.0
    inputs = {**sc.get("defaults", {}), **body.input_data}
    if sc.get("ai_video"):
        rt, inputs = plan_ai(sc, inputs, body.render_type)
    else:
        rt = body.render_type
        if rt in (ai.AI_VIDEO, ai.AI_DANCE):
            raise HTTPException(422, "Template này không dùng luồng video AI")
        if rt == "auto":
            use_model = pricing.DECISION_KEY[3:] not in pricing.disabled_models(db)
            rt, used = ai.decide_render_type_ex(tpl.category, tpl.template_schema, inputs, use_model)
            if used:                                   # the routing model call is billed too
                routing_fee = pricing.price(db, pricing.DECISION_KEY)
        if rt == ai.HEAVY and not tpl.base_video_url and not inputs.get("reference_video"):
            raise HTTPException(422, "Luồng nâng cao cần template có video mẫu")
        if rt == ai.HEAVY and not resolve(inputs.get("character_image", "")):
            raise HTTPException(422, "Luồng nâng cao cần input_data.character_image (ảnh đã upload)")
    ensure_enabled(db, rt, inputs)
    cost = round(cost_of(db, rt, inputs) + routing_fee, 2)
    fresh = db.get(User, user.id)
    db.refresh(fresh)
    if fresh.credits_balance < cost:
        raise HTTPException(402, detail={
            "message": "Không đủ credit, vui lòng nạp thêm",
            "required": cost, "balance": fresh.credits_balance})
    return Prepared(fresh, tpl, rt, cost, inputs)


def gen_out(g: GenerationHistory) -> dict:
    return {"id": g.id, "template_id": g.template_id, "render_type": g.render_type,
            "input_data": {k: v for k, v in (g.input_data or {}).items() if not k.startswith("_")}, "output_video_url": g.output_video_url,
            "credits_cost": g.credits_cost, "status": g.status, "share_to_library": g.share_to_library,
            "title": g.title or "", "user_category": g.user_category or "Mặc định",
            "error_message": g.error_message, "created_at": g.created_at}


@router.post("/generate")
def generate(body: GenerateIn, bg: BackgroundTasks, p: Prepared = Depends(credit_guard),
             db: Session = Depends(get_db)):
    # The guard's check is advisory; the guarded UPDATE is what prevents overdraft under concurrency.
    if not credits.charge(db, p.user.id, p.cost):
        db.rollback()
        raise HTTPException(402, detail={"message": "Không đủ credit, vui lòng nạp thêm", "required": p.cost})
    title = body.title.strip() or f"Tác phẩm #{p.template.id}"
    cat = body.user_category.strip() or "Mặc định"
    gen = GenerationHistory(user_id=p.user.id, template_id=p.template.id, render_type=p.render_type,
                            input_data=p.input_data, credits_cost=p.cost, status="pending",
                            share_to_library=body.share_to_library, title=title, user_category=cat)
    db.add(gen)
    db.execute(update(SharedVideoLibrary).where(SharedVideoLibrary.id == p.template.id)
               .values(usage_count=SharedVideoLibrary.usage_count + 1))
    db.commit()

    log_activity(db, p.user.id, "generate_video",
                 details={"gen_id": gen.id, "template_id": p.template.id, "cost": p.cost,
                          "title": title, "category": cat, "render_type": p.render_type})

    if p.render_type in (ai.HEAVY, ai.AI_DANCE, ai.AI_VIDEO):
        task = heavy_render_task if p.render_type == ai.HEAVY else ai_render_task
        try:
            if not broker_up():
                raise ConnectionError("Redis không phản hồi")
            task.delay(gen.id)
        except Exception as e:  # broker down
            if p.render_type == ai.AI_VIDEO:
                bg.add_task(run_generation, gen.id)   # pure API wait: fine to run in-process without Redis
            else:                                      # needs the GPU queue -> refund immediately
                credits.fail_and_refund(db, gen.id, f"Không gửi được vào hàng đợi: {e}")
                raise HTTPException(503, "Hàng đợi render (Redis/Celery) không khả dụng; đã hoàn credit")
    else:
        bg.add_task(run_generation, gen.id)
    db.refresh(gen)
    return gen_out(gen)


@router.get("/pricing")
def get_pricing(db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Credit prices so the UI can show a cost before the user presses Generate."""
    return {"fast": pricing.price(db, pricing.K_FAST), "heavy": pricing.price(db, pricing.K_MIMIC),
            "h3_768p": pricing.price(db, pricing.K_H3_768), "h3_2k": pricing.price(db, pricing.K_H3_2K),
            "fast_enabled": pricing.enabled(db, pricing.K_FAST), "heavy_enabled": pricing.enabled(db, pricing.K_MIMIC),
            "ai_video_enabled": minimax.configured() and (pricing.enabled(db, pricing.K_H3_768)
                                                          or pricing.enabled(db, pricing.K_H3_2K)),
            "ratios": minimax.RATIOS,
            "resolutions": [r for r, k in (("768P", pricing.K_H3_768), ("2K", pricing.K_H3_2K))
                            if pricing.enabled(db, k)],
            "min_seconds": minimax.MIN_SECONDS, "max_seconds": minimax.MAX_SECONDS, "dance_max_seconds": 10}


@router.get("/generations")
def history(category: str | None = None, db: Session = Depends(get_db), user: User = Depends(current_user)):
    q = select(GenerationHistory).where(GenerationHistory.user_id == user.id)
    if category and category.strip():
        q = q.where(GenerationHistory.user_category == category.strip())
    q = q.order_by(GenerationHistory.id.desc()).limit(100)
    return [gen_out(g) for g in db.scalars(q)]


@router.get("/generations/categories")
def user_categories(db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Return distinct categories of user's generated templates."""
    q = select(GenerationHistory.user_category).where(GenerationHistory.user_id == user.id).distinct()
    cats = [c for c in db.scalars(q) if c]
    if "Mặc định" not in cats:
        cats.insert(0, "Mặc định")
    return cats


class UpdateGenerationIn(BaseModel):
    title: str | None = None
    user_category: str | None = None


@router.patch("/generations/{gid}")
def update_generation(gid: int, body: UpdateGenerationIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    g = db.get(GenerationHistory, gid)
    if not g or g.user_id != user.id:
        raise HTTPException(404, "Không tìm thấy tác phẩm")
    if body.title is not None:
        g.title = body.title.strip()
    if body.user_category is not None:
        g.user_category = body.user_category.strip() or "Mặc định"
    db.commit()
    db.refresh(g)
    return gen_out(g)


@router.get("/generations/{gid}")
def get_generation(gid: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    g = db.get(GenerationHistory, gid)
    if not g or (g.user_id != user.id and user.role != "admin"):
        raise HTTPException(404, "Không tìm thấy")
    return gen_out(g)
