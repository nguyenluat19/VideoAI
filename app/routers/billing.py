import hmac
import re
import secrets
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import credits
from .. import pricing, promotions
from ..activity import log_activity
from ..config import MAX_TOPUP_VND, QUICK_TOPUP_AMOUNTS, settings
from ..db import get_db
from ..deps import current_user
from ..maintenance import as_utc, expire_transactions, tx_expires_at, utcnow
from ..models import CreditAdjustment, Transaction, User, WebhookLog

router = APIRouter(tags=["billing"])
CODE_RE = re.compile(r"NAP[A-Z0-9]{8}")


class TopupIn(BaseModel):
    amount_vnd: int


def qr_url(amount: float, code: str) -> str:
    """Standard VietQR image (img.vietqr.io) with account, amount and transfer content."""
    return (f"https://img.vietqr.io/image/{settings.bank_bin}-{settings.bank_account_no}-compact2.png"
            f"?amount={int(amount)}&addInfo={quote(code)}&accountName={quote(settings.bank_account_name)}")


def tx_out(t: Transaction) -> dict:
    exp = tx_expires_at(t)
    return {"id": t.id, "amount_vnd": t.amount_vnd, "credits_added": t.credits_added,
            "bonus_credits": t.bonus_credits, "promo_name": t.promo_name,
            "base_credits": round(t.credits_added - t.bonus_credits, 2),
            "payment_method": t.payment_method, "transaction_code": t.transaction_code,
            "status": t.status, "created_at": t.created_at,
            "expires_at": exp.isoformat(),
            "seconds_left": max(0, int((exp - utcnow()).total_seconds())) if t.status == "pending" else 0,
            "qr_url": qr_url(t.amount_vnd, t.transaction_code) if t.status == "pending" else None}


def _min_topup(db: Session) -> int:
    return int(pricing.get_setting(db, "min_topup_vnd", 20000))


@router.get("/api/billing/config")
def billing_config(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rate = pricing.get_setting(db, "vnd_per_credit", 1000)
    lo = _min_topup(db)
    quick = []
    for a in QUICK_TOPUP_AMOUNTS:
        if a >= lo:
            q = promotions.quote(db, a, pricing.credits_for(db, a))
            quick.append({"amount_vnd": a, "credits": q["base_credits"], "bonus_percent": q["bonus_percent"],
                          "bonus_credits": q["bonus_credits"], "total_credits": q["total_credits"]})
    live = [{"id": p.id, "name": p.name, "ends_at": as_utc(p.end_at).isoformat() if p.end_at else None, "tiers": p.tiers}
            for p in promotions.active(db)]
    return {"min_vnd": lo, "max_vnd": MAX_TOPUP_VND, "vnd_per_credit": rate, "ttl_seconds": settings.tx_ttl_seconds,
            "quick_amounts": quick, "promotions": live}


@router.get("/api/billing/quote")
def billing_quote(amount_vnd: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Live preview for the amount box: base credits + the promotion that would apply right now."""
    if amount_vnd < 0 or amount_vnd > MAX_TOPUP_VND:
        raise HTTPException(422, "Số tiền không hợp lệ")
    return promotions.quote(db, amount_vnd, pricing.credits_for(db, amount_vnd))


@router.post("/api/billing/topup")
def create_topup(body: TopupIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if user.role == "admin":
        raise HTTPException(403, "Tài khoản admin không nạp tiền. Dùng tab Tài khoản để cộng credit thủ công.")
    lo = _min_topup(db)
    if body.amount_vnd < lo:
        raise HTTPException(422, f"Số tiền nạp tối thiểu là {lo:,}đ".replace(",", "."))
    if body.amount_vnd > MAX_TOPUP_VND:
        raise HTTPException(422, f"Số tiền nạp tối đa mỗi lần là {MAX_TOPUP_VND:,}đ".replace(",", "."))
    # Alphanumeric only: banks often strip separators from the transfer content.
    code = "NAP" + secrets.token_hex(4).upper()
    q = promotions.quote(db, body.amount_vnd, pricing.credits_for(db, body.amount_vnd))   # locked in now
    tx = Transaction(user_id=user.id, amount_vnd=body.amount_vnd,
                     credits_added=q["total_credits"], bonus_credits=q["bonus_credits"], promo_name=q["promo_name"],
                     payment_method="vietqr", transaction_code=code, status="pending")
    db.add(tx)
    db.commit()
    log_activity(db, user.id, "topup_created", details={"amount_vnd": body.amount_vnd, "code": code, "credits": q["total_credits"]})
    return {**tx_out(tx), "bank": {"bin": settings.bank_bin, "account_no": settings.bank_account_no,
                                   "account_name": settings.bank_account_name}}


@router.get("/api/billing/transactions")
def my_transactions(db: Session = Depends(get_db), user: User = Depends(current_user)):
    expire_transactions(db)
    q = select(Transaction).where(Transaction.user_id == user.id).order_by(Transaction.id.desc()).limit(50)
    return [tx_out(t) for t in db.scalars(q)]


@router.get("/api/billing/adjustments")
def my_adjustments(db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Credits an admin added or removed on my account (so the balance is never a mystery)."""
    q = select(CreditAdjustment).where(CreditAdjustment.user_id == user.id).order_by(CreditAdjustment.id.desc()).limit(50)
    return [{"id": a.id, "amount": a.amount, "balance_after": a.balance_after, "reason": a.reason,
             "created_at": as_utc(a.created_at).isoformat()} for a in db.scalars(q)]


@router.get("/api/billing/transactions/{tid}")
def transaction_status(tid: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    expire_transactions(db)   # lazy: the answer is right even between sweeper runs
    t = db.get(Transaction, tid)
    if not t or t.user_id != user.id:
        raise HTTPException(404, "Không tìm thấy giao dịch")
    return tx_out(t)


# ---------------- Webhook (SePay / Casso) ----------------
def _eq(a: str, b: str) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode(), b.encode())


def _authenticate(request: Request) -> str:
    """Return 'sepay' | 'casso'. SePay sends `Authorization: Apikey <key>`; Casso sends `secure-token`."""
    auth = request.headers.get("authorization", "")
    if _eq(auth.removeprefix("Apikey ").strip() if auth.startswith("Apikey ") else "", settings.sepay_api_key):
        return "sepay"
    if _eq(request.headers.get("secure-token", ""), settings.casso_secure_token):
        return "casso"
    raise HTTPException(401, "Webhook không được xác thực")


def _normalize(provider: str, payload: dict) -> list[tuple[str, float]]:
    """-> [(transfer content, incoming amount)] — outgoing transfers are dropped."""
    out: list[tuple[str, float]] = []
    if provider == "sepay":
        if payload.get("transferType", "in") == "in":
            out.append((f"{payload.get('content', '')} {payload.get('code') or ''}",
                        float(payload.get("transferAmount", 0))))
    else:
        for item in payload.get("data", []) or []:
            amt = float(item.get("amount", 0))
            if amt > 0:
                out.append((str(item.get("description", "")), amt))
    return out


@router.post("/api/webhooks/payment")
async def payment_webhook(request: Request, db: Session = Depends(get_db)):
    provider = _authenticate(request)
    try:
        payload = await request.json()
        items = _normalize(provider, payload)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "Payload không hợp lệ")

    expire_transactions(db)   # an unpaid QR past its lifetime is already `failed` here
    results = []

    def record(result: str, code: str, amount: float, content: str, extra: dict | None = None) -> None:
        db.add(WebhookLog(provider=provider, transaction_code=code, amount=amount,
                          content=content[:500], result=result))
        db.commit()
        results.append({"code": code or None, "status": result, **(extra or {})})

    for content, amount in items:
        m = CODE_RE.search(re.sub(r"[^A-Za-z0-9]", "", content).upper())
        if not m:
            record("ignored_no_code", "", amount, content)
            continue
        code = m.group(0)
        tx = db.scalar(select(Transaction).where(Transaction.transaction_code == code))
        if tx is None:
            record("ignored_unknown_code", code, amount, content)
        elif tx.status == "success":
            record("duplicate", code, amount, content)          # idempotent replay
        elif int(amount) != int(tx.amount_vnd):
            # Amount mismatch is never auto-credited; stays as-is for admin review.
            record("amount_mismatch", code, amount, content, {"expected": tx.amount_vnd, "received": amount})
        elif tx.status == "failed":
            # QR expired before the money arrived.
            if settings.late_payment_autocredit and credits.settle_topup(db, tx.id, allow_from=("failed",)):
                record("late_credited", code, amount, content)
            else:
                record("late_payment", code, amount, content)    # admin can approve it
        else:
            ok = credits.settle_topup(db, tx.id)
            record("success" if ok else "duplicate", code, amount, content)
    # 200 + success flag so SePay/Casso do not retry-storm on benign cases.
    return {"success": True, "results": results}
