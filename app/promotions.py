"""Top-up bonus campaigns ("nạp từ 100k thưởng 10%").

Rules: a campaign is live when enabled and now is inside [start_at, end_at] (either end may be open).
The bonus is the best tier reached (highest min_vnd <= amount). If several campaigns are live the single
largest percentage applies — campaigns never stack. Bonus = base credits x percent, rounded to 2 decimals.
"""
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import MAX_TOPUP_VND
from .maintenance import as_utc, utcnow
from .models import Promotion

MAX_BONUS_PERCENT = 300.0


def validate_tiers(tiers: list[dict]) -> list[dict]:
    if not tiers:
        raise ValueError("Cần ít nhất một mức khuyến mãi")
    if len(tiers) > 20:
        raise ValueError("Tối đa 20 mức")
    out, seen = [], set()
    for t in tiers:
        try:
            lo, pct = int(t["min_vnd"]), round(float(t["bonus_percent"]), 2)
        except (KeyError, TypeError, ValueError):
            raise ValueError("Mỗi mức cần min_vnd (số nguyên) và bonus_percent (số)")
        if lo < 1000 or lo > MAX_TOPUP_VND:
            raise ValueError(f"Mức nạp tối thiểu của khuyến mãi phải từ 1.000đ đến {MAX_TOPUP_VND:,}đ".replace(",", "."))
        if not 0 < pct <= MAX_BONUS_PERCENT:
            raise ValueError(f"% khuyến mãi phải lớn hơn 0 và không quá {MAX_BONUS_PERCENT:g}")
        if lo in seen:
            raise ValueError(f"Trùng mức nạp {lo:,}đ".replace(",", "."))
        seen.add(lo)
        out.append({"min_vnd": lo, "bonus_percent": pct})
    return sorted(out, key=lambda t: t["min_vnd"])


def validate_window(start: datetime | None, end: datetime | None) -> None:
    if start and end and as_utc(end) <= as_utc(start):
        raise ValueError("Thời điểm kết thúc phải sau thời điểm bắt đầu")


def state(p: Promotion, now: datetime | None = None) -> str:
    now = now or utcnow()
    if not p.enabled:
        return "disabled"
    if p.start_at and as_utc(p.start_at) > now:
        return "scheduled"
    if p.end_at and as_utc(p.end_at) <= now:
        return "expired"
    return "active"


def active(db: Session, now: datetime | None = None) -> list[Promotion]:
    now = now or utcnow()
    return [p for p in db.scalars(select(Promotion).order_by(Promotion.id)) if state(p, now) == "active"]


def tier_for(p: Promotion, amount_vnd: float) -> dict | None:
    hit = [t for t in p.tiers if amount_vnd >= t["min_vnd"]]
    return max(hit, key=lambda t: t["min_vnd"]) if hit else None


def quote(db: Session, amount_vnd: int, base_credits: float, now: datetime | None = None) -> dict:
    """What a payment of `amount_vnd` is worth right now."""
    best = None
    for p in active(db, now):
        t = tier_for(p, amount_vnd)
        if t and (best is None or t["bonus_percent"] > best[1]["bonus_percent"]):
            best = (p, t)
    pct = best[1]["bonus_percent"] if best else 0.0
    bonus = round(base_credits * pct / 100, 2)
    return {"amount_vnd": amount_vnd, "base_credits": base_credits, "bonus_percent": pct, "bonus_credits": bonus,
            "total_credits": round(base_credits + bonus, 2), "promo_name": best[0].name if best else "",
            "promo_id": best[0].id if best else None}
