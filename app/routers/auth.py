from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import ratelimit
from ..config import settings
from ..db import get_db
from ..deps import current_user
from ..activity import log_activity
from ..models import ActivityLog, User
from ..security import hash_password, make_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    username: str
    password: str


def user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "email": u.email,
            "credits_balance": u.credits_balance, "role": u.role}


def _ip(request: Request) -> str:
    return request.client.host if request.client else "?"


@router.post("/register")
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)):
    if ratelimit.hit(f"reg:{_ip(request)}", 3600) > settings.register_per_hour:
        raise HTTPException(429, "Đăng ký quá nhiều lần, vui lòng thử lại sau")
    user = User(username=body.username, email=body.email.lower(),
                password_hash=hash_password(body.password), role="user")
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Tên đăng nhập hoặc email đã tồn tại")
    log_activity(db, user.id, "register", ip=_ip(request), details={"username": user.username, "email": user.email})
    return {"token": make_token(user.id), "user": user_out(user)}


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    key = f"login:{_ip(request)}:{body.username.lower()}"
    if ratelimit.count(key, 600) >= settings.login_max_failures:
        raise HTTPException(429, "Đăng nhập sai quá nhiều lần, vui lòng thử lại sau 10 phút")
    user = db.scalar(select(User).where(or_(User.username == body.username,
                                            User.email == body.username.lower())))
    if not user or not verify_password(body.password, user.password_hash):
        ratelimit.hit(key, 600)
        raise HTTPException(401, "Sai tên đăng nhập hoặc mật khẩu")
    ratelimit.clear(key)
    log_activity(db, user.id, "login", ip=_ip(request), details={"username": user.username})
    return {"token": make_token(user.id), "user": user_out(user)}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


@router.get("/me/activity")
def my_activity(user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = select(ActivityLog).where(ActivityLog.user_id == user.id).order_by(ActivityLog.id.desc()).limit(100)
    logs = db.scalars(q).all()
    return [{
        "id": a.id,
        "action": a.action,
        "ip": a.ip,
        "details": a.details or {},
        "created_at": a.created_at
    } for a in logs]
