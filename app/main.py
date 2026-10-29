from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .config import settings
from .db import SessionLocal
from .maintenance import Sweeper
from . import pricing
from .migrate import upgrade_db
from .models import SharedVideoLibrary, User
from .routers import admin, assistant, auth, billing, generate, templates, videotools
from .security import hash_password

STATIC = Path(__file__).resolve().parent.parent / "static"


def check_production_settings() -> None:
    """In APP_ENV=production refuse to start with the built-in secret or no admin password."""
    if settings.app_env != "production":
        return
    problems = []
    if settings.secret_key in ("dev-secret-change-me", "change-me-to-a-long-random-string") or len(settings.secret_key) < 32:
        problems.append("SECRET_KEY must be a random string of at least 32 characters")
    if not settings.admin_password or settings.admin_password.startswith("change-me"):
        problems.append("ADMIN_PASSWORD must be set to a real password")
    if problems:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))


AI_TEMPLATES = [
    {"title": "Video AI từ mô tả (MiniMax-H3)", "category": "full",
     "schema": {"ai_video": True, "layers": []}},
    {"title": "Nhảy theo ảnh nhân vật (MiniMax-H3 + MimicMotion)", "category": "full",
     "schema": {"ai_video": True, "dance": True, "motion_transfer": True, "layers": []}},
]


def seed_ai_templates(db) -> None:
    """Built-in templates for the AI flows, published under the first admin. Idempotent."""
    admin = db.scalar(select(User).where(User.role == "admin").order_by(User.id))
    if not admin:
        return
    for t in AI_TEMPLATES:
        if not db.scalar(select(SharedVideoLibrary.id).where(SharedVideoLibrary.title == t["title"])):
            db.add(SharedVideoLibrary(title=t["title"], category=t["category"], template_schema=t["schema"],
                                      status="publish", creator_id=admin.id))
    db.commit()


def init() -> None:
    check_production_settings()
    for d in (settings.storage_dir, settings.uploads_dir, settings.outputs_dir):
        d.mkdir(parents=True, exist_ok=True)
    upgrade_db()
    with SessionLocal() as db:
        if settings.admin_password and not db.scalar(select(User).where(User.role == "admin")):
            db.add(User(username=settings.admin_username, email=settings.admin_email.lower(),
                        password_hash=hash_password(settings.admin_password), role="admin"))
            db.commit()
        pricing.seed(db)
        seed_ai_templates(db)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init()
    sweeper = Sweeper()
    sweeper.start()
    yield
    sweeper.stop()


app = FastAPI(title="Video Template SaaS", lifespan=lifespan)
for r in (auth, templates, generate, billing, admin, assistant, videotools):
    app.include_router(r.router)

# Mount only uploads/outputs — never the storage root (it holds the SQLite DB).
for _d in (settings.uploads_dir, settings.outputs_dir):
    _d.mkdir(parents=True, exist_ok=True)
    app.mount(f"/media/{_d.name}", StaticFiles(directory=_d), name=f"media-{_d.name}")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}
