"""Render execution.

Queues: `ai` (I/O-bound, many in parallel: MiniMax polling) and `gpu` (MimicMotion, one at a time).
Fast renders run in-process. ai_dance is two stages so the GPU is not held while waiting for the API:
  ai queue: MiniMax-H3 makes the reference dance clip  ->  gpu queue: MimicMotion transfers it + audio mux.
"""
import logging

from celery import Celery

from . import credits, library
from .config import settings
from .db import SessionLocal
from .models import GenerationHistory, SharedVideoLibrary
from .rendering import aivideo, fast, heavy

log = logging.getLogger("tasks")
celery_app = Celery("videosaas", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_always_eager=settings.celery_eager,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # Redis down must fail fast (a few seconds), not hang the HTTP request for a minute.
    broker_transport_options={"socket_connect_timeout": 2, "socket_timeout": 5},
    broker_connection_timeout=2,
    task_publish_retry=True,
    task_publish_retry_policy={"max_retries": 1, "interval_start": 0, "interval_step": 0.2, "interval_max": 0.5},
    task_routes={"render.ai": {"queue": "ai"}, "render.heavy": {"queue": "gpu"}},
)


def broker_up() -> bool:
    """Fast reachability check (Celery's own publish retries can block a request for ~20s)."""
    if settings.celery_eager:
        return True
    try:
        import redis
        return bool(redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1, socket_timeout=1).ping())
    except Exception:  # noqa: BLE001 - any failure means "not usable right now"
        return False


def _execute(db, gen: GenerationHistory, stage: str) -> tuple[str | None, bool]:
    """-> (output_url, handoff). handoff=True means stage 1 of ai_dance is done and stage 2 was queued."""
    tpl = db.get(SharedVideoLibrary, gen.template_id)
    values = gen.input_data or {}
    rt = gen.render_type
    if rt == "fast_remotion":
        return fast.render(gen.id, tpl.template_schema or {}, tpl.base_video_url, values), False
    if rt == "ai_video":
        return aivideo.generate(gen.id, values, dance=False, on_progress=lambda: credits.touch(db, gen.id)), False
    if rt == "ai_dance" and stage == "start":
        ref = aivideo.generate(gen.id, values, dance=True, on_progress=lambda: credits.touch(db, gen.id))
        gen.input_data = {**values, "_reference_video": ref}
        db.commit()
        credits.touch(db, gen.id)
        heavy_render_task.delay(gen.id, "gpu")       # a failure here is caught by the caller -> refund
        return None, True
    # heavy_mimicmotion, or stage 2 of ai_dance
    reference = values.get("_reference_video") or values.get("reference_video") or tpl.base_video_url
    url = heavy.render(gen.id, reference, values)
    if rt == "ai_dance":
        aivideo.mux_audio(url, values.get("_reference_video", ""))
    return url, False


def run_generation(gen_id: int, stage: str = "start") -> None:
    """Shared by every pipeline; any failure refunds the exact charged credits."""
    with SessionLocal() as db:
        gen = db.get(GenerationHistory, gen_id)
        if gen is None:
            return
        if stage == "start":
            if not credits.mark_processing(db, gen_id):
                return
        elif gen.status != "processing":     # stage 2 only continues a job that is still ours
            return
        else:
            credits.touch(db, gen_id)
        try:
            url, handoff = _execute(db, gen, stage)
            if handoff:
                return
            credits.mark_success(db, gen_id, url)
        except Exception as e:  # noqa: BLE001 - every failure must end in refund
            db.rollback()
            credits.fail_and_refund(db, gen_id, f"{type(e).__name__}: {e}")
            return
        try:   # after success nothing may undo the render: a library failure is only logged
            library.create_entry_from_generation(db, gen_id)
        except Exception:  # noqa: BLE001
            db.rollback()
            log.exception("could not add generation %s to the library", gen_id)


@celery_app.task(name="render.heavy", bind=True)
def heavy_render_task(self, gen_id: int, stage: str = "start") -> None:
    run_generation(gen_id, stage)


@celery_app.task(name="render.ai", bind=True)
def ai_render_task(self, gen_id: int) -> None:
    run_generation(gen_id)
