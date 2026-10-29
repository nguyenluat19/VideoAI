"""Activity logging helper for user actions."""
from typing import Any
from sqlalchemy.orm import Session
from .models import ActivityLog


def log_activity(db: Session, user_id: int, action: str, ip: str = "", details: dict[str, Any] | None = None) -> ActivityLog:
    """Record a user action into the activity_logs table."""
    try:
        log = ActivityLog(
            user_id=user_id,
            action=action,
            ip=ip or "",
            details=details or {},
        )
        db.add(log)
        db.commit()
        return log
    except Exception as e:
        db.rollback()
        # Non-blocking, return dummy
        return None
