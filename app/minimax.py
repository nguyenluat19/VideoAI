"""MiniMax-H3 video generation client (async task API: create -> poll -> download).

Request shape follows the xah.io v2 public gateway: POST {base}/v2/video_generation with
{model, content:[{type:text|image_url,...}], duration, ratio}. The query endpoint is
/v2/query/video_generation/{task_id}. Run scripts/check_minimax.py once with your key
to confirm against the live API.
"""
import base64
import io
import json
import mimetypes
import subprocess
import time
from pathlib import Path

import httpx

try:
    from PIL import Image as _PIL_Image
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False

from .config import settings

RATIOS = ("9:16", "16:9", "1:1", "4:3", "3:4", "21:9")
RESOLUTIONS = ("768P", "2K")
MIN_SECONDS, MAX_SECONDS = 4, 15
MIN_IMAGE_SIDE = 256   # MiniMax rejects images smaller than this on either side
OK_STATES = {"succeeded", "success", "completed", "done", "finished"}
FAIL_STATES = {"failed", "fail", "cancelled", "canceled", "error", "expired"}
QUERY_FALLBACKS = (
    "/v2/query/video_generation/{task_id}",   # xah.io public gateway (primary)
    "/v2/video_generation/{task_id}",          # minimax.io direct
    "/v2/query/video_generation?task_id={task_id}",
    "/v1/query/video_generation?task_id={task_id}",
)


class VideoError(RuntimeError):
    pass


def configured() -> bool:
    return bool(settings.minimax_api_key)


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.minimax_api_key}", "Content-Type": "application/json"}


def _dig(d, *paths):
    """First non-empty value found along any of the dotted paths."""
    for path in paths:
        cur = d
        for key in path.split("."):
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur not in (None, "", []):
            return cur
    return None


def _ensure_min_size(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """Resize image so both sides are >= MIN_IMAGE_SIDE px. Returns (bytes, mime).
    Falls back to the original if Pillow is unavailable or the image is already large enough.
    """
    if not _PILLOW_AVAILABLE:
        return image_bytes, mime
    try:
        img = _PIL_Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if w >= MIN_IMAGE_SIDE and h >= MIN_IMAGE_SIDE:
            return image_bytes, mime
        # Scale up so the smaller side reaches MIN_IMAGE_SIDE, preserving aspect ratio.
        scale = MIN_IMAGE_SIDE / min(w, h)
        new_w, new_h = max(MIN_IMAGE_SIDE, int(w * scale)), max(MIN_IMAGE_SIDE, int(h * scale))
        img = img.resize((new_w, new_h), _PIL_Image.LANCZOS)
        buf = io.BytesIO()
        fmt = img.format or "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue(), f"image/{fmt.lower()}"
    except Exception:
        return image_bytes, mime


def build_payload(prompt: str, duration: int, ratio: str, resolution: str,
                  first_frame: Path | None = None) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if first_frame is not None:
        raw = first_frame.read_bytes()
        mime = mimetypes.guess_type(first_frame.name)[0] or "image/png"
        raw, mime = _ensure_min_size(raw, mime)
        b64 = base64.b64encode(raw).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"},
                        "role": "first_frame"})
    body = {"model": settings.minimax_model, "content": content, "resolution": resolution,
            "duration": duration, "ratio": ratio}
    if settings.minimax_extra:
        try:
            body.update(json.loads(settings.minimax_extra))
        except ValueError as e:
            raise VideoError(f"MINIMAX_EXTRA_PARAMS không phải JSON hợp lệ: {e}") from e
    return body


def _check_base_resp(data: dict) -> None:
    br = data.get("base_resp") or {}
    if br.get("status_code") not in (None, 0):
        raise VideoError(f"MiniMax từ chối: {br.get('status_msg') or br}")
    err = data.get("error")
    if err:
        raise VideoError(f"MiniMax lỗi: {err}")


def create_task(client: httpx.Client, payload: dict) -> str:
    r = client.post(settings.minimax_base_url + settings.minimax_create_path, json=payload, headers=_headers())
    if r.status_code >= 400:
        raise VideoError(f"MiniMax tạo task lỗi HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    _check_base_resp(data)
    tid = _dig(data, "task_id", "id", "data.task_id", "task.id", "task.task_id")
    if not tid:
        raise VideoError(f"MiniMax không trả task_id: {str(data)[:300]}")
    return str(tid)


_query_template: list[str] = []   # remembers the first query path that worked


def query_task(client: httpx.Client, task_id: str) -> tuple[str, str | None, dict]:
    """-> (state: 'ok'|'failed'|'pending', video_url|None, raw)"""
    candidates = list(_query_template) or [settings.minimax_query_path, *QUERY_FALLBACKS]
    last = None
    for tmpl in candidates:
        r = client.get(settings.minimax_base_url + tmpl.format(task_id=task_id), headers=_headers())
        if r.status_code in (404, 405):
            last = r
            continue
        if r.status_code >= 400:
            raise VideoError(f"MiniMax tra cứu lỗi HTTP {r.status_code}: {r.text[:300]}")
        _query_template[:] = [tmpl]
        data = r.json()
        _check_base_resp(data)
        status = str(_dig(data, "status", "task.status", "data.status") or "").lower()
        url = _dig(data, "content.url", "task.content.url", "data.content.url", "video_url", "url")
        if not url and status in OK_STATES:
            fid = _dig(data, "file_id", "task.file_id", "data.file_id")
            if fid:
                url = _file_url(client, fid)
        if status in FAIL_STATES:
            return "failed", None, data
        return ("ok" if (status in OK_STATES and url) else "pending"), url, data
    raise VideoError(f"MiniMax: không tìm thấy endpoint tra cứu task (HTTP {last.status_code if last else '?'}); "
                     "chạy scripts/check_minimax.py và đặt MINIMAX_QUERY_PATH")


def _file_url(client: httpx.Client, file_id: str) -> str | None:
    r = client.get(f"{settings.minimax_base_url}/v1/files/retrieve", params={"file_id": file_id},
                   headers=_headers())
    return _dig(r.json(), "file.download_url", "download_url") if r.status_code < 400 else None


def has_video_stream(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_type", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return "video" in out.stdout


def generate(prompt: str, dest: Path, *, duration: int = 5, ratio: str = "9:16", resolution: str = "768P",
             first_frame: Path | None = None, on_progress=lambda: None) -> Path:
    """Blocks (polling) until the video is saved to `dest`; raises VideoError on any failure."""
    if not configured():
        raise VideoError("MINIMAX_API_KEY chưa được cấu hình")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=60) as client:
            tid = create_task(client, build_payload(prompt, duration, ratio, resolution, first_frame))
            deadline = time.monotonic() + settings.minimax_max_wait
            while True:
                state, url, raw = query_task(client, tid)
                on_progress()   # heartbeat so the reaper knows we are alive
                if state == "ok":
                    break
                if state == "failed":
                    raise VideoError(f"MiniMax tạo video thất bại: {str(raw)[:300]}")
                if time.monotonic() > deadline:
                    raise VideoError(f"MiniMax quá thời gian chờ ({int(settings.minimax_max_wait)}s) cho task {tid}")
                time.sleep(settings.minimax_poll)
            with client.stream("GET", url, timeout=300, follow_redirects=True) as resp:
                if resp.status_code >= 400:
                    raise VideoError(f"Tải video lỗi HTTP {resp.status_code}")
                with dest.open("wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        f.write(chunk)
    except httpx.HTTPError as e:
        raise VideoError(f"Lỗi mạng khi gọi MiniMax: {e}") from e
    if not dest.exists() or dest.stat().st_size < 1000 or not has_video_stream(dest):
        dest.unlink(missing_ok=True)
        raise VideoError("File tải về từ MiniMax không phải video hợp lệ")
    return dest
