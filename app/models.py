from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    credits_balance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="user", nullable=False)  # user | admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SharedVideoLibrary(Base):
    __tablename__ = "shared_video_library"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(32), default="full")  # hook | cta | full
    base_video_url: Mapped[str] = mapped_column(Text, default="")
    thumbnail_url: Mapped[str] = mapped_column(Text, default="")
    template_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="private", nullable=False, index=True)  # private | publish
    creator_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # Entries auto-created from a finished render: the finished video for preview and the
    # generation it came from (unique -> at most one library entry per render).
    preview_video_url: Mapped[str] = mapped_column(Text, default="")
    source_generation_id: Mapped[int | None] = mapped_column(Integer, nullable=True, unique=True)


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    amount_vnd: Mapped[float] = mapped_column(Float, nullable=False)
    credits_added: Mapped[float] = mapped_column(Float, nullable=False)
    payment_method: Mapped[str] = mapped_column(String(32), default="vietqr")
    transaction_code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False, index=True)  # pending|success|failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # credits_added = base + bonus. Both the rate and the promotion are locked in when the QR is created.
    bonus_credits: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    promo_name: Mapped[str] = mapped_column(String(120), default="", nullable=False)


class GenerationHistory(Base):
    __tablename__ = "generation_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    template_id: Mapped[int] = mapped_column(ForeignKey("shared_video_library.id"), nullable=False, index=True)
    render_type: Mapped[str] = mapped_column(String(32), nullable=False)  # fast_remotion | heavy_mimicmotion
    input_data: Mapped[dict] = mapped_column(JSON, default=dict)
    output_video_url: Mapped[str] = mapped_column(Text, default="")
    credits_cost: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False, index=True)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)  # last state change
    share_to_library: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    user_category: Mapped[str] = mapped_column(String(64), default="Mặc định", nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="", nullable=False)


class ActivityLog(Base):
    """User account activity log: logins, top-ups, video renders, template actions, downloads, AI assistant usage."""
    __tablename__ = "activity_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    ip: Mapped[str] = mapped_column(String(64), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class WebhookLog(Base):
    __tablename__ = "webhook_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    transaction_code: Mapped[str] = mapped_column(String(32), default="")
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    content: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PricingItem(Base):
    """One row per priced thing (render flow, video model resolution, Inkstone model). Price in credits."""
    __tablename__ = "pricing_items"
    key: Mapped[str] = mapped_column(String(96), primary_key=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PricingLog(Base):
    __tablename__ = "pricing_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(96), nullable=False)
    old_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    old_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    new_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    admin_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)


class Promotion(Base):
    """Top-up bonus campaign: tiers = [{"min_vnd": 100000, "bonus_percent": 10}, ...]."""
    __tablename__ = "promotions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tiers: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CreditAdjustment(Base):
    """Every manual credit change by an admin (+ or -): who, whom, how much, why, resulting balance."""
    __tablename__ = "credit_adjustments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    admin_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    balance_after: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
