"""Adapter cho Muse AI qua FlowKit provider queue (127.0.0.1:8100).

Muse AI là model video qua muse.ai (tối đa 10 giây/clip), chạy trên Chrome extension
Muse Kit. Dùng FlowKit gateway để điều phối job. Cần cookie muse.ai đã nhập vào muse2api
gateway trước khi dùng.

Hai model Muse hoạt động độc lập nhau:
- muse-ai-video: gen video (video_i2v), tối đa 10 giây/clip, cần ảnh đầu + prompt
- muse-ai-image: gen ảnh (image_generate), qua muse.ai, cần cookie

Khoá API/cookie chỉ lưu phía máy chủ, không gửi xuống trình duyệt, không ghi vào log.
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from .config import settings
from .media import url_for


class MuseError(RuntimeError):
    pass


from .muse_settings import muse_cookies_configured, auto_import_cookies_to_muse2api


def configured() -> bool:
    """Muse Kit đang sẵn sàng (gateway FlowKit có thể truy cập + cookies đã cấu hình)."""
    if not muse_cookies_configured():
        return False
    try:
        r = httpx.get(f"{settings.muse_base_url}/api/provider-jobs/wait-next",
                      params={"provider": "muse_ext", "worker_id": "dashboard-check",
                              "lease_ttl_s": "5", "timeout_s": "1"},
                      timeout=3)
        # 200 = có job, 204/404 = không có job, bất kỳ gì cũng cho biết gateway sống
        return r.status_code in (200, 204, 404)
    except Exception:
        return False


def submit_video_job(prompt: str, first_frame_image_url: str | None,
                     duration: int = 10, ratio: str = "9:16") -> dict:
    """Gửi job gen video Muse qua FlowKit gateway.

    Tự động nhập cookies từ DB vào muse2api nếu chưa có session.
    """
    # Auto-import cookies from DB to muse2api gateway
    auto_import_cookies_to_muse2api()

    if not configured():
        raise MuseError("Muse Kit gateway (FlowKit) không khả dụng hoặc chưa cấu hình cookies. "
                        "Hãy bật muse2api + Chrome extension Muse Kit, nhập cookie qua tab Admin.")

    # Lấy đường dẫn file ảnh nếu có
    first_frame_path = None
    if first_frame_image_url:
        from .media import resolve
        p = resolve(first_frame_image_url)
        if p:
            first_frame_path = str(p.resolve())

    payload = {
        "provider": "muse_ext",
        "prompt": prompt,
        "duration": duration,
        "ratio": ratio,
        "extra": {
            "start_data_url": f"file://{first_frame_path}" if first_frame_path else "",
            "aspect": ratio,
            "seconds": duration,
        },
    }

    try:
        r = httpx.post(f"{settings.muse_base_url}/api/provider-jobs",
                       json=payload, timeout=30)
        r.raise_for_status()
        job = r.json()
    except httpx.HTTPError as e:
        raise MuseError(f"Không gửi được job Muse: {e}") from e

    job_id = job.get("id") or job.get("job_id")
    if not job_id:
        raise MuseError(f"FlowKit không trả job_id: {str(job)[:300]}")

    # Poll cho kết quả (Muse Kit extension sẽ xử lý trong trình duyệt)
    deadline = time.monotonic() + settings.muse_timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{settings.muse_base_url}/api/provider-jobs/{job_id}", timeout=10)
            if r.status_code == 200:
                result = r.json()
                status = result.get("status", "pending")
                if status == "completed":
                    clip = result.get("result", {})
                    b64 = clip.get("b64", "")
                    mime = clip.get("mime", "video/mp4")
                    if not b64:
                        raise MuseError("FlowKit trả completed nhưng không có base64 clip")
                    return _save_clip(b64, mime, job_id)
                elif status == "failed":
                    err = result.get("error", "lỗi không rõ")
                    raise MuseError(f"Muse AI thất bại: {err}")
                # pending -> tiếp tục poll
            elif r.status_code == 404:
                raise MuseError(f"Job {job_id} không tồn tại trên FlowKit")
        except httpx.HTTPError:
            pass
        time.sleep(3)

    raise MuseError(f"Muse AI quá thời gian chờ ({settings.muse_timeout_s}s)")


def _save_clip(b64: str, mime: str, job_id: str) -> dict:
    """Giải mã base64, lưu file video vào storage/outputs."""
    import base64

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise MuseError(f"Base64 từ Muse không hợp lệ: {e}") from e

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / f"muse_{job_id}.mp4"

    try:
        out.write_bytes(raw)
    except OSError as e:
        raise MuseError(f"Không lưu được file Muse: {e}") from e

    # Kiểm tra file video hợp lệ
    if not _has_video_stream(out):
        out.unlink(missing_ok=True)
        raise MuseError("File từ Muse AI không phải video hợp lệ")

    return {"output_video_url": url_for(out), "mime": mime, "job_id": job_id}


def _has_video_stream(path: Path) -> bool:
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                               str(path)], capture_output=True, text=True, timeout=15)
        return "video" in proc.stdout
    except Exception:
        return False


def generate_image(prompt: str, aspect_ratio: str = "9:16") -> dict:
    """Gen ảnh qua Muse AI (image_generate).

    Hiện tại Muse Kit chủ yếu làm video, adapter này dành cho tương lai khi FlowKit hỗ trợ
    gen ảnh qua muse.ai. Hiện tại báo 503 nếu chưa hỗ trợ.
    """
    raise MuseError("Muse AI hình ảnh chưa được FlowKit hỗ trợ. Hãy dùng model ảnh khác.")
