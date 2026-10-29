"""Single source of truth for every price in the system (credits), editable by admins.

The env values (FAST_RENDER_COST, ...) only seed the table the first time; after that the database wins.
Prices apply to jobs created afterwards: a job keeps the price it was charged at creation.
"""
import math
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import DEFAULT_MIN_TOPUP_VND, MODEL_CATALOG, settings
from .modelregistry import ALL_MODELS, get_model
from .models import AppSetting, PricingItem, PricingLog

MAX_PRICE = 100_000

K_FAST = "render.fast"
K_MIMIC = "render.mimicmotion"
K_H3_768 = "video.minimax-h3.768p"
K_H3_2K = "video.minimax-h3.2k"
DECISION_KEY = "ai.intern-decision-4b"

GROUP_LABELS = {"render": "Render", "video": "Model tạo video", "vision": "Model hình ảnh (Inkstone)",
                "llm": "Model ngôn ngữ (Inkstone)", "decision": "Model điều hướng (Inkstone)"}

MODEL_STEP_GROUPS = {"llm": "llm", "image_generate": "vision", "video_i2v": "video", "video_t2v": "video"}


def ai_key(model_id: str) -> str:
    return f"ai.{model_id}"


def _defaults() -> list[dict]:
    """key, label, group, unit, default price, provider cost in USD per unit (None = unknown/local)."""
    items = [
        {"key": K_FAST, "label": "Render nhanh (FFmpeg)", "group": "render", "unit": "video",
         "default": settings.fast_cost, "cost_usd": None},
        {"key": K_MIMIC, "label": "MimicMotion · chép chuyển động (GPU máy bạn)", "group": "render",
         "unit": "video", "default": settings.heavy_cost, "cost_usd": None},
        {"key": K_H3_768, "label": "MiniMax-H3 · 768P", "group": "video", "unit": "giây",
         "default": settings.ai_video_credits_per_sec, "cost_usd": 0.08},
        {"key": K_H3_2K, "label": "MiniMax-H3 · 2K", "group": "video", "unit": "giây",
         "default": round(settings.ai_video_credits_per_sec * settings.ai_video_2k_multiplier, 2),
         "cost_usd": 0.13},
    ]
    for m in MODEL_CATALOG:
        items.append({"key": ai_key(m["id"]), "label": f"{m['name']}  ({m['id']})", "group": m["group"],
                      "unit": "lần điều hướng" if m["group"] == "decision" else "lần gọi",
                      "default": 0.0, "cost_usd": None, "desc": m["desc"]})
    # Model registry entries (user-pickable per pipeline step)
    # Only add models NOT already in MODEL_CATALOG (which has ai.* keys)
    inkstone_ids = {m["id"] for m in MODEL_CATALOG}
    for m in ALL_MODELS:
        if m.id in inkstone_ids:
            continue  # Already has ai.* pricing key
        if m.step in ("video_i2v", "video_t2v", "image_generate", "llm"):
            group = MODEL_STEP_GROUPS.get(m.step, "llm")
            unit = {"video_i2v": "giây", "video_t2v": "giây", "image_generate": "ảnh", "llm": "lần gọi"}[m.step]
            items.append({"key": f"m.{m.id}", "label": f"{m.name}  ({m.id})", "group": group,
                          "unit": unit, "default": m.default_price,
                          "cost_usd": None, "desc": m.desc})
    return items


def catalog() -> dict[str, dict]:
    return {i["key"]: i for i in _defaults()}


def seed(db: Session) -> None:
    """Insert any missing item (idempotent; never overwrites an admin's edits)."""
    have = set(db.scalars(select(PricingItem.key)))
    for i in _defaults():
        if i["key"] not in have:
            db.add(PricingItem(key=i["key"], price=i["default"], enabled=True))
    for k, v in (("usd_vnd", "25000"), ("vnd_per_credit", "1000"), ("min_topup_vnd", str(DEFAULT_MIN_TOPUP_VND))):
        if db.get(AppSetting, k) is None:
            db.add(AppSetting(key=k, value=v))
    db.commit()


def snapshot(db: Session) -> dict[str, PricingItem]:
    return {p.key: p for p in db.scalars(select(PricingItem))}


def price(db: Session, key: str) -> float:
    row = db.get(PricingItem, key)
    return row.price if row else catalog()[key]["default"]


def enabled(db: Session, key: str) -> bool:
    row = db.get(PricingItem, key)
    return row.enabled if row else True


def disabled_models(db: Session) -> set[str]:
    """Inkstone model ids an admin switched off."""
    return {k[3:] for k, p in snapshot(db).items() if k.startswith("ai.") and not p.enabled}


def video_price_per_second(db: Session, resolution: str) -> float:
    return price(db, K_H3_2K if resolution == "2K" else K_H3_768)


def video_cost(db: Session, duration: int, resolution: str) -> float:
    return round(duration * video_price_per_second(db, resolution), 2)


def validate_price(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("Giá phải là số")
    if not math.isfinite(v) or v < 0 or v > MAX_PRICE:
        raise ValueError(f"Giá phải từ 0 đến {MAX_PRICE}")
    return round(v, 2)


def update_item(db: Session, key: str, *, price_value=None, is_enabled: bool | None = None,
                admin_id: int | None = None) -> PricingItem:
    if key not in catalog():
        raise KeyError(key)
    row = db.get(PricingItem, key)
    if row is None:
        seed(db)
        row = db.get(PricingItem, key)
    new_price = row.price if price_value is None else validate_price(price_value)
    new_enabled = row.enabled if is_enabled is None else bool(is_enabled)
    if new_price == row.price and new_enabled == row.enabled:
        return row
    db.add(PricingLog(key=key, old_price=row.price, new_price=new_price, old_enabled=row.enabled,
                      new_enabled=new_enabled, admin_id=admin_id))
    row.price, row.enabled = new_price, new_enabled
    row.updated_at, row.updated_by = datetime.now(timezone.utc), admin_id
    db.commit()
    return row


def get_setting(db: Session, key: str, default: float) -> float:
    row = db.get(AppSetting, key)
    try:
        return float(row.value) if row else default
    except ValueError:
        return default


def credits_for(db: Session, amount_vnd: float) -> float:
    """Credits a payment of `amount_vnd` buys at today's rate (locked into the transaction when created)."""
    return round(amount_vnd / get_setting(db, "vnd_per_credit", 1000), 2)


def set_setting(db: Session, key: str, value) -> float:
    try:
        v = round(float(value), 2)
    except (TypeError, ValueError):
        raise ValueError("Giá trị phải là số")
    if not math.isfinite(v) or v <= 0 or v > 10_000_000:
        raise ValueError("Giá trị phải lớn hơn 0 và không quá 10.000.000")
    if key == "min_topup_vnd" and not 1000 <= v <= 5_000_000:
        raise ValueError("Mức nạp tối thiểu phải từ 1.000đ đến 5.000.000đ")
    row = db.get(AppSetting, key)
    if row:
        row.value = str(v)
    else:
        db.add(AppSetting(key=key, value=str(v)))
    db.commit()
    return v


def margin_percent(db: Session, item: dict, current_price: float) -> float | None:
    """Profit over provider cost, in %, when the provider cost is known."""
    if not item.get("cost_usd") or current_price <= 0:
        return None
    revenue_vnd = current_price * get_setting(db, "vnd_per_credit", 1000)
    cost_vnd = item["cost_usd"] * get_setting(db, "usd_vnd", 25000)
    return round((revenue_vnd - cost_vnd) / cost_vnd * 100, 1)
