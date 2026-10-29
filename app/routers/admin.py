from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import math

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from datetime import datetime

from .. import credits, pricing, promotions
from ..db import get_db
from ..deps import admin_user
from ..maintenance import as_utc
from ..models import (CreditAdjustment, GenerationHistory, PricingLog, Promotion, SharedVideoLibrary, Transaction, User,
                      WebhookLog)
from .billing import tx_out
from .templates import tpl_out

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(admin_user)])


@router.get("/templates")
def list_all(status: str | None = None, source: str | None = None, db: Session = Depends(get_db)):
    """source=generated -> entries auto-created from renders; source=created -> hand-made templates."""
    q = select(SharedVideoLibrary).order_by(SharedVideoLibrary.id.desc())
    if status:
        q = q.where(SharedVideoLibrary.status == status)
    if source == "generated":
        q = q.where(SharedVideoLibrary.source_generation_id.is_not(None))
    elif source == "created":
        q = q.where(SharedVideoLibrary.source_generation_id.is_(None))
    return [tpl_out(t) for t in db.scalars(q)]


def _set_status(db: Session, tid: int, status: str) -> dict:
    t = db.get(SharedVideoLibrary, tid)
    if not t:
        raise HTTPException(404, "Không tìm thấy template")
    t.status = status
    db.commit()
    return tpl_out(t)


class TemplateStatusIn(BaseModel):
    status: str = "publish"


@router.patch("/templates/{tid}")
def update_status(tid: int, body: TemplateStatusIn, db: Session = Depends(get_db)):
    st = "publish" if body.status == "publish" else "private"
    return _set_status(db, tid, st)


@router.post("/templates/{tid}/publish")
def publish(tid: int, db: Session = Depends(get_db)):
    return _set_status(db, tid, "publish")


@router.post("/templates/{tid}/unpublish")
def unpublish(tid: int, db: Session = Depends(get_db)):
    return _set_status(db, tid, "private")


@router.delete("/templates/{tid}")
def reject(tid: int, db: Session = Depends(get_db)):
    """Discard an entry that is not fit for the library. Only private entries can be removed."""
    t = db.get(SharedVideoLibrary, tid)
    if not t:
        raise HTTPException(404, "Không tìm thấy template")
    if t.status == "publish":
        raise HTTPException(409, "Hãy gỡ công khai (unpublish) trước khi xoá")
    if db.query(GenerationHistory.id).filter_by(template_id=tid).first():
        raise HTTPException(409, "Template đã có lịch sử render, chỉ có thể để ở trạng thái private")
    db.delete(t)
    db.commit()
    return {"deleted": tid}


@router.get("/users")
def users(q: str | None = None, limit: int = 200, db: Session = Depends(get_db)):
    """All accounts (admins included), optionally filtered by username/email substring."""
    stmt = select(User).order_by(User.id)
    if q and q.strip():
        like = f"%{q.strip().lower()}%"
        stmt = stmt.where(or_(func.lower(User.username).like(like), func.lower(User.email).like(like)))
    return [{"id": u.id, "username": u.username, "email": u.email, "role": u.role,
             "credits_balance": round(u.credits_balance, 2), "created_at": u.created_at}
            for u in db.scalars(stmt.limit(max(1, min(limit, 500))))]


MAX_ADJUST = 10_000_000


class GrantIn(BaseModel):
    amount: float
    reason: str = ""


@router.post("/users/{uid}/credits")
def grant(uid: int, body: GrantIn, db: Session = Depends(get_db), admin: User = Depends(admin_user)):
    """Add (positive) or remove (negative) any amount of credits on any account, with an audit entry."""
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "Không tìm thấy người dùng")
    amount = round(body.amount, 2)
    if not math.isfinite(amount) or amount == 0 or abs(amount) > MAX_ADJUST:
        raise HTTPException(422, f"Số credit phải khác 0 và không quá {MAX_ADJUST:,}".replace(",", "."))
    bal = credits.adjust(db, uid, amount, admin.id, body.reason)
    if bal is None:
        raise HTTPException(422, "Số dư không thể âm")
    return {"id": u.id, "credits_balance": bal, "amount": amount}


@router.get("/credit-log")
def credit_log(user_id: int | None = None, limit: int = 200, db: Session = Depends(get_db)):
    stmt = select(CreditAdjustment).order_by(CreditAdjustment.id.desc()).limit(max(1, min(limit, 500)))
    if user_id is not None:
        stmt = stmt.where(CreditAdjustment.user_id == user_id)
    names = {u.id: u.username for u in db.scalars(select(User))}
    return [{"id": a.id, "user_id": a.user_id, "username": names.get(a.user_id, "?"), "amount": a.amount,
             "balance_after": a.balance_after, "reason": a.reason, "admin": names.get(a.admin_id, "?"),
             "created_at": as_utc(a.created_at).isoformat()} for a in db.scalars(stmt)]


@router.get("/transactions")
def transactions(status: str | None = None, db: Session = Depends(get_db)):
    q = select(Transaction).order_by(Transaction.id.desc()).limit(200)
    if status:
        q = q.where(Transaction.status == status)
    return [{**tx_out(t), "user_id": t.user_id} for t in db.scalars(q)]


@router.post("/transactions/{tid}/approve")
def approve(tid: int, db: Session = Depends(get_db)):
    """Manual settlement for cases the webhook refused to auto-credit (amount mismatch, late payment)."""
    if not credits.settle_topup(db, tid, allow_from=("pending", "failed")):
        raise HTTPException(409, "Giao dịch đã thành công hoặc không tồn tại")
    return {"id": tid, "status": "success"}


# ---------------- pricing dashboard ----------------
class PriceIn(BaseModel):
    price: float | None = None
    enabled: bool | None = None


class PricingSettingsIn(BaseModel):
    usd_vnd: float | None = None
    vnd_per_credit: float | None = None
    min_topup_vnd: float | None = None


def _pricing_item_out(db: Session, cat: dict, row) -> dict:
    return {"key": cat["key"], "label": cat["label"], "group": cat["group"],
            "group_label": pricing.GROUP_LABELS[cat["group"]], "unit": cat["unit"], "desc": cat.get("desc", ""),
            "price": row.price, "default": cat["default"], "enabled": row.enabled, "cost_usd": cat["cost_usd"],
            "margin_percent": pricing.margin_percent(db, cat, row.price),
            "updated_at": row.updated_at, "updated_by": row.updated_by}


def _pricing_settings(db: Session) -> dict:
    return {"usd_vnd": pricing.get_setting(db, "usd_vnd", 25000),
            "vnd_per_credit": pricing.get_setting(db, "vnd_per_credit", 1000),
            "min_topup_vnd": pricing.get_setting(db, "min_topup_vnd", 20000)}


@router.get("/pricing")
def pricing_dashboard(db: Session = Depends(get_db)):
    pricing.seed(db)
    snap, cat = pricing.snapshot(db), pricing.catalog()
    names = {u.id: u.username for u in db.scalars(select(User))}
    log = db.scalars(select(PricingLog).order_by(PricingLog.id.desc()).limit(60)).all()
    return {
        "items": [_pricing_item_out(db, c, snap[k]) for k, c in cat.items()],
        "settings": _pricing_settings(db),
        "log": [{"id": l.id, "key": l.key, "label": cat.get(l.key, {}).get("label", l.key),
                 "old_price": l.old_price, "new_price": l.new_price,
                 "old_enabled": l.old_enabled, "new_enabled": l.new_enabled,
                 "admin": names.get(l.admin_id, "?"), "created_at": l.created_at} for l in log]}


@router.put("/pricing/{key:path}")
def pricing_update(key: str, body: PriceIn, db: Session = Depends(get_db), admin: User = Depends(admin_user)):
    if key not in pricing.catalog():
        raise HTTPException(404, "Không có mục giá này")
    try:
        row = pricing.update_item(db, key, price_value=body.price, is_enabled=body.enabled, admin_id=admin.id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _pricing_item_out(db, pricing.catalog()[key], row)


@router.post("/pricing-reset/{key:path}")
def pricing_reset(key: str, db: Session = Depends(get_db), admin: User = Depends(admin_user)):
    cat = pricing.catalog().get(key)
    if not cat:
        raise HTTPException(404, "Không có mục giá này")
    row = pricing.update_item(db, key, price_value=cat["default"], is_enabled=True, admin_id=admin.id)
    return _pricing_item_out(db, cat, row)


@router.put("/pricing-settings")
def pricing_settings(body: PricingSettingsIn, db: Session = Depends(get_db)):
    try:
        for k in ("usd_vnd", "vnd_per_credit", "min_topup_vnd"):
            if getattr(body, k) is not None:
                pricing.set_setting(db, k, getattr(body, k))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _pricing_settings(db)


# ---------------- top-up promotions ----------------
class TierIn(BaseModel):
    min_vnd: int
    bonus_percent: float


class PromoIn(BaseModel):
    name: str
    tiers: list[TierIn]
    start_at: datetime | None = None
    end_at: datetime | None = None
    enabled: bool = True


def _promo_out(p: Promotion) -> dict:
    # always send explicit UTC: SQLite hands back naive datetimes and browsers would read them as local time
    return {"id": p.id, "name": p.name, "tiers": p.tiers,
            "start_at": as_utc(p.start_at).isoformat() if p.start_at else None,
            "end_at": as_utc(p.end_at).isoformat() if p.end_at else None,
            "enabled": p.enabled, "state": promotions.state(p), "created_at": p.created_at}


def _apply_promo(p: Promotion, body: PromoIn) -> None:
    name = body.name.strip()
    if not name or len(name) > 120:
        raise HTTPException(422, "Tên chương trình từ 1 đến 120 ký tự")
    try:
        tiers = promotions.validate_tiers([t.model_dump() for t in body.tiers])
        promotions.validate_window(body.start_at, body.end_at)
    except ValueError as e:
        raise HTTPException(422, str(e))
    p.name, p.tiers, p.start_at, p.end_at, p.enabled = name, tiers, body.start_at, body.end_at, body.enabled


@router.get("/promotions")
def promo_list(db: Session = Depends(get_db)):
    return [_promo_out(p) for p in db.scalars(select(Promotion).order_by(Promotion.id.desc()))]


@router.post("/promotions")
def promo_create(body: PromoIn, db: Session = Depends(get_db), admin: User = Depends(admin_user)):
    p = Promotion(created_by=admin.id, tiers=[])
    _apply_promo(p, body)
    db.add(p)
    db.commit()
    return _promo_out(p)


@router.put("/promotions/{pid}")
def promo_update(pid: int, body: PromoIn, db: Session = Depends(get_db)):
    p = db.get(Promotion, pid)
    if not p:
        raise HTTPException(404, "Không tìm thấy chương trình")
    _apply_promo(p, body)
    db.commit()
    return _promo_out(p)


@router.delete("/promotions/{pid}")
def promo_delete(pid: int, db: Session = Depends(get_db)):
    p = db.get(Promotion, pid)
    if not p:
        raise HTTPException(404, "Không tìm thấy chương trình")
    db.delete(p)      # past transactions keep their locked bonus and promo_name
    db.commit()
    return {"deleted": pid}


@router.get("/webhook-log")
def webhook_log(result: str | None = None, db: Session = Depends(get_db)):
    q = select(WebhookLog).order_by(WebhookLog.id.desc()).limit(200)
    if result:
        q = q.where(WebhookLog.result == result)
    return [{"id": w.id, "provider": w.provider, "transaction_code": w.transaction_code, "amount": w.amount,
             "content": w.content, "result": w.result, "created_at": w.created_at} for w in db.scalars(q)]


# Muse AI endpoints removed - only Wan2.1 video model remains


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    return {
        "users": db.scalar(select(func.count(User.id))),
        "revenue_vnd": db.scalar(select(func.coalesce(func.sum(Transaction.amount_vnd), 0))
                                 .where(Transaction.status == "success")),
        "generations": {s: n for s, n in db.execute(
            select(GenerationHistory.status, func.count()).group_by(GenerationHistory.status))},
        "pending_templates": db.scalar(select(func.count(SharedVideoLibrary.id))
                                       .where(SharedVideoLibrary.status == "private")),
    }
