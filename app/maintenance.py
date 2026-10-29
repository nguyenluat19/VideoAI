"""Background housekeeping: expire unpaid QR codes, fail+refund stuck renders."""
import logging
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import credits
from .config import settings
from .db import SessionLocal
from .models import GenerationHistory, Transaction

log = logging.getLogger("maintenance")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)   # SQLite returns naive UTC


def tx_expires_at(tx: Transaction) -> datetime:
    return as_utc(tx.created_at) + timedelta(seconds=settings.tx_ttl_seconds)


def expire_transactions(db: Session, now: datetime | None = None) -> int:
    """Cancel every pending top-up whose QR lifetime has elapsed (pending -> failed)."""
    cutoff = (now or utcnow()) - timedelta(seconds=settings.tx_ttl_seconds)
    res = db.execute(update(Transaction)
                     .where(Transaction.status == "pending", Transaction.created_at <= cutoff)
                     .values(status="failed"))
    db.commit()
    return res.rowcount


def reap_generations(db: Session, now: datetime | None = None) -> int:
    """Fail + refund renders that can no longer finish (server died, worker lost, queue wiped)."""
    now = now or utcnow()
    rules = [  # (render_type, status, max age seconds)
        ("fast_remotion", "pending", settings.fast_timeout),
        ("fast_remotion", "processing", settings.fast_timeout),
        ("heavy_mimicmotion", "pending", settings.heavy_queue_timeout),
        ("heavy_mimicmotion", "processing", settings.heavy_timeout),
        ("ai_video", "pending", settings.ai_queue_timeout),
        ("ai_video", "processing", settings.ai_timeout),
        ("ai_dance", "pending", settings.ai_queue_timeout),
        # processing spans the API stage, the wait for the GPU queue and the GPU stage
        ("ai_dance", "processing", max(settings.ai_timeout + settings.heavy_timeout, settings.heavy_queue_timeout)),
    ]
    reaped = 0
    for rtype, status, age in rules:
        ids = db.scalars(select(GenerationHistory.id).where(
            GenerationHistory.render_type == rtype, GenerationHistory.status == status,
            GenerationHistory.updated_at <= now - timedelta(seconds=age))).all()
        for gid in ids:
            if credits.fail_and_refund(db, gid, f"Quá thời hạn xử lý ({status} > {age}s); đã hoàn credit"):
                reaped += 1
    return reaped


def run_sweeps() -> dict:
    with SessionLocal() as db:
        return {"expired_transactions": expire_transactions(db), "reaped_generations": reap_generations(db)}


class Sweeper(threading.Thread):
    """Daemon thread started with the web app; one sweep immediately, then every interval."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="sweeper")
        self._halt = threading.Event()

    def run(self) -> None:
        while True:
            try:
                r = run_sweeps()
                if any(r.values()):
                    log.info("sweep: %s", r)
            except Exception:  # noqa: BLE001 - never let housekeeping die
                log.exception("sweep failed")
            if self._halt.wait(settings.sweep_interval):
                return

    def stop(self) -> None:
        self._halt.set()
