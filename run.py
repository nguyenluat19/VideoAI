#!/usr/bin/env python3
"""One-command launcher:  python run.py

First run: creates .env (random SECRET_KEY + admin password), checks FFmpeg, starts the app.
If Redis and MimicMotion are present it also starts the Celery worker for heavy renders.
"""
import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"


def ensure_env() -> None:
    if ENV.exists():
        return
    text = (ROOT / ".env.example").read_text()
    admin_pw = secrets.token_urlsafe(12)
    text = text.replace("SECRET_KEY=\n", f"SECRET_KEY={secrets.token_urlsafe(48)}\n")
    text = text.replace("ADMIN_PASSWORD=\n", f"ADMIN_PASSWORD={admin_pw}\n")
    ENV.write_text(text)
    print(f"\n  Created .env\n  Admin login:  admin / {admin_pw}   (saved in .env)\n")


def env_value(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    for line in ENV.read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-worker", action="store_true", help="never start Redis/Celery")
    ap.add_argument("--init", action="store_true", help="only create .env (e.g. before docker compose up)")
    args = ap.parse_args()
    if args.init:
        ensure_env()
        return 0

    try:
        import fastapi, uvicorn, sqlalchemy, celery, alembic  # noqa: F401,E401
    except ImportError as e:
        print(f"Missing dependency ({e.name}). Run:  pip install -r requirements.txt")
        return 1
    if not shutil.which("ffmpeg"):
        print("FFmpeg not found. Install it (mac: brew install ffmpeg | ubuntu: apt install ffmpeg | "
              "windows: ffmpeg.org) and run again.")
        return 1

    ensure_env()
    os.chdir(ROOT)
    if not (env_value("INKSTONE_BASE_URL") and env_value("INKSTONE_API_KEY")):
        print("  ! INKSTONE_BASE_URL / INKSTONE_API_KEY are empty in .env — AI features are off "
              "(rendering and payments still work).")

    children: list[subprocess.Popen] = []
    mm = Path(env_value("MIMICMOTION_DIR") or "vendor/MimicMotion")
    mm = mm if mm.is_absolute() else ROOT / mm
    gpu_ready = (mm / "inference.py").exists()
    ai_ready = bool(env_value("MINIMAX_API_KEY"))
    celery = [sys.executable, "-m", "celery", "-A", "app.tasks.celery_app", "worker", "--loglevel=warning"]
    if not args.no_worker and shutil.which("redis-server") and (gpu_ready or ai_ready):
        redis_url = env_value("REDIS_URL") or "redis://localhost:6379/0"
        port = redis_url.rsplit(":", 1)[-1].split("/")[0]
        children.append(subprocess.Popen(["redis-server", "--port", port, "--save", "", "--appendonly", "no"],
                                         stdout=subprocess.DEVNULL))
        time.sleep(1)
        if ai_ready:   # MiniMax jobs just wait on an API: run several at once
            children.append(subprocess.Popen(celery + ["-n", "ai@%h", "-Q", "ai", "-c", "4"]))
        if gpu_ready:  # MimicMotion: one job at a time on the GPU
            children.append(subprocess.Popen(celery + ["-n", "gpu@%h", "-Q", "gpu", "-c", "1"]))
        print(f"  Workers started: {'AI-video ' if ai_ready else ''}{'GPU(MimicMotion)' if gpu_ready else ''}")
    if not args.no_worker:
        if not ai_ready:
            print("  AI video (MiniMax-H3) is off: set MINIMAX_API_KEY in .env.")
        if not gpu_ready:
            print("  MimicMotion (dance / motion copy) is off: run scripts/setup_mimicmotion.sh to enable it.")
        if (ai_ready or gpu_ready) and not shutil.which("redis-server"):
            print("  redis-server not found: GPU jobs need it; AI video falls back to running inside the web app.")

    print(f"\n  Open http://{args.host}:{args.port}\n")
    try:
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port)
    finally:
        for c in children:
            c.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
