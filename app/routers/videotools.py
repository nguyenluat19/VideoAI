"""Video tools: download from social media, transcribe & translate, add subtitles & blur."""
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import ai
from ..activity import log_activity
from ..config import settings
from ..db import get_db
from ..deps import current_user
from ..media import resolve, url_for
from ..models import User

router = APIRouter(prefix="/api/video", tags=["videotools"])

# Theo dõi tiến độ tải theo job_id (dùng cho progress bar trên UI). Key: job_id, value: {percent, status}
DOWNLOAD_PROGRESS: dict[str, dict] = {}


def _dl_progress(job_id: str):
    """Trả về progress_hook của yt-dlp ghi tiến độ vào DOWNLOAD_PROGRESS."""
    state = {"completed": 0, "current": 0, "total": 1}

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            cur = int(done / total * 100) if total > 0 else state["current"]
            state["current"] = max(state["current"], cur)
            pct = int((state["completed"] + state["current"] / 100) / state["total"] * 100)
            DOWNLOAD_PROGRESS[job_id] = {"percent": min(99, pct), "status": "downloading"}
        elif d.get("status") == "finished":
            state["current"] = 100
            pct = int((state["completed"] + state["current"] / 100) / state["total"] * 100)
            DOWNLOAD_PROGRESS[job_id] = {"percent": min(99, pct), "status": "finished"}

    return hook


class DownloadIn(BaseModel):
    url: str = Field(min_length=5)


class TranscribeIn(BaseModel):
    video_url: str
    translate_to_vi: bool = True
    hf_api_key: str | None = None


class ProcessIn(BaseModel):
    video_url: str
    srt_content: str | None = None
    sub_position: str = "bottom"  # bottom | top | middle
    sub_font_size: int = 18
    sub_color: str = "&H00FFFFFF"  # White with black outline in ASS hex
    blur_region: dict[str, Any] | None = None  # {x, y, width, height} in pixels or percent


def _clean_path(p: Path) -> str:
    # Convert windows path to forward slashes and escape for ffmpeg filter
    s = str(p.resolve()).replace("\\", "/")
    # Escape colon for ffmpeg filter: C:/... -> C\:/...
    if len(s) > 1 and s[1] == ":":
        s = s[0] + "\\:" + s[2:]
    return s


def _transcribe_hf(audio_path: Path, hf_key: str) -> dict | None:
    """Use Hugging Face Cloud Inference API (Whisper-Large-v3)."""
    try:
        import httpx
        url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {"Authorization": f"Bearer {hf_key.strip()}"}
        audio_bytes = audio_path.read_bytes()
        with httpx.Client(timeout=120) as client:
            resp = client.post(url, headers=headers, data=audio_bytes, params={"return_timestamps": "true"})
            if resp.status_code == 200:
                data = resp.json()
                chunks = data.get("chunks", [])
                segments = []
                for c in chunks:
                    ts = c.get("timestamp") or [0.0, 0.0]
                    txt = (c.get("text") or "").strip()
                    if txt:
                        start = round(float(ts[0] or 0.0), 2)
                        end = round(float(ts[1] if ts[1] is not None else start + 3.0), 2)
                        segments.append({"start": start, "end": end, "text": txt})
                if segments:
                    return {"segments": segments, "language": "auto"}
                if data.get("text"):
                    return {"segments": [{"start": 0.0, "end": 5.0, "text": data["text"].strip()}], "language": "auto"}
    except Exception:
        pass
    return None


def _translate_hf(texts: list[str], hf_key: str) -> list[str] | None:
    """Translate sentences using Hugging Face translation model (Helsinki-NLP/opus-mt-en-vi)."""
    try:
        import httpx
        url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-vi"
        headers = {"Authorization": f"Bearer {hf_key.strip()}"}
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json={"inputs": texts})
            if resp.status_code == 200:
                res = resp.json()
                if isinstance(res, list) and len(res) == len(texts):
                    return [item.get("translation_text", "").strip() for item in res]
    except Exception:
        pass
    return None


def _format_srt_time(seconds: float) -> str:
    millis = int((seconds % 1) * 1000)
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


@router.get("/hf-status")
def get_hf_status():
    """Return whether HUGGINGFACE_API_KEY is configured in system environment."""
    has_key = bool(settings.huggingface_api_key and settings.huggingface_api_key.strip())
    masked = (settings.huggingface_api_key[:6] + "..." + settings.huggingface_api_key[-4:]) if has_key and len(settings.huggingface_api_key) > 10 else ("Đã cấu hình" if has_key else "")
    return {
        "configured": has_key,
        "masked_key": masked,
        "models": ["openai/whisper-large-v3", "Helsinki-NLP/opus-mt-en-vi"]
    }


@router.post("/transcribe")
def transcribe_video(body: TranscribeIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Transcribe video audio using Hugging Face Whisper API (or local fallback), and translate to Vietnamese."""
    local_path = resolve(body.video_url)
    if not local_path or not local_path.exists():
        raise HTTPException(404, "Không tìm thấy file video")

    # 1. Extract audio to 16kHz mono wav
    wav_path = settings.outputs_dir / f"audio_{secrets.token_hex(8)}.wav"
    try:
        cmd = ["ffmpeg", "-y", "-i", str(local_path), "-vn", "-ar", "16000", "-ac", "1", str(wav_path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
    except Exception as e:
        if wav_path.exists():
            wav_path.unlink()
        raise HTTPException(500, f"Lỗi trích xuất âm thanh: {e}")

    hf_token = (body.hf_api_key or settings.huggingface_api_key or "").strip()
    segments = []
    detected_lang = "unknown"
    engine_used = "faster-whisper (Local)"

    # 2. Try Hugging Face Cloud Whisper API first if key provided
    if hf_token:
        hf_res = _transcribe_hf(wav_path, hf_token)
        if hf_res and hf_res.get("segments"):
            segments = hf_res["segments"]
            detected_lang = hf_res.get("language", "auto")
            engine_used = "Hugging Face Cloud (openai/whisper-large-v3)"

    # 3. Fallback to local faster-whisper if HF was not used or failed
    if not segments:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments_raw, info = model.transcribe(str(wav_path), beam_size=5)
            detected_lang = getattr(info, "language", "unknown")
            for s in segments_raw:
                text = s.text.strip()
                if text:
                    segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            engine_used = f"faster-whisper (Local CPU)"
        except Exception as e:
            if wav_path.exists():
                wav_path.unlink(missing_ok=True)
            raise HTTPException(500, f"Lỗi nhận dạng giọng nói: {e}")

    if wav_path.exists():
        wav_path.unlink(missing_ok=True)

    if not segments:
        return {"language": detected_lang, "engine_used": engine_used, "segments": [], "srt": ""}

    # 4. Translate to Vietnamese if requested and language is not vi
    if body.translate_to_vi and detected_lang != "vi":
        translated_via_hf = False
        # 4a. Try Hugging Face Translation API if key is available
        if hf_token:
            texts_to_trans = [s["text"] for s in segments]
            hf_trans = _translate_hf(texts_to_trans, hf_token)
            if hf_trans and len(hf_trans) == len(segments):
                for idx, trans_txt in enumerate(hf_trans):
                    if trans_txt:
                        segments[idx]["translated"] = trans_txt
                translated_via_hf = True
                engine_used += " + Hugging Face Translation (Opus-MT-en-vi)"

        # 4b. Fallback to Inkstone LLM if HF translation was skipped
        if not translated_via_hf and ai.configured():
            try:
                prompts = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(segments))
                sys_msg = (
                    "Bạn là chuyên gia dịch phụ đề video. Hãy dịch từng dòng sau sang tiếng Việt tự nhiên, "
                    "giữ đúng số thứ tự và ngữ cảnh ngắn gọn của video ngắn. "
                    "Định dạng trả về: mỗi dòng bắt đầu bằng 'X. [bản dịch]' tương ứng."
                )
                ans = ai._text(ai.DEFAULT_SCRIPT_MODEL, sys_msg, prompts)
                lines = [line.strip() for line in ans.splitlines() if line.strip()]
                for line in lines:
                    m = re.match(r"^(\d+)\.\s*(.*)$", line)
                    if m:
                        idx = int(m.group(1)) - 1
                        if 0 <= idx < len(segments):
                            segments[idx]["translated"] = m.group(2).strip()
                engine_used += " + AI Translator (Inkstone)"
            except Exception:
                pass

    # Build SRT string
    srt_lines = []
    for i, s in enumerate(segments):
        txt = s.get("translated") or s["text"]
        srt_lines.append(f"{i+1}")
        srt_lines.append(f"{_format_srt_time(s['start'])} --> {_format_srt_time(s['end'])}")
        srt_lines.append(txt)
        srt_lines.append("")

    srt_content = "\n".join(srt_lines)
    log_activity(db, user.id, "video_transcribe", details={"video_url": body.video_url, "segments_count": len(segments), "engine": engine_used})

    return {
        "language": detected_lang,
        "engine_used": engine_used,
        "segments": segments,
        "srt": srt_content,
    }


def _download_tiktok_tikwm(url: str, dest_path: Path) -> dict | None:
    """Download TikTok/Douyin video without watermark using TikWM API."""
    try:
        import httpx
        with httpx.Client(timeout=45, follow_redirects=True) as client:
            resp = client.post("https://www.tikwm.com/api/", data={"url": url})
            if resp.status_code == 200:
                body = resp.json()
                if body.get("code") == 0 and body.get("data"):
                    data = body["data"]
                    # 'play' is no-watermark HD stream
                    stream_url = data.get("play") or data.get("wmplay")
                    if stream_url:
                        # TikWM relative or absolute URL
                        if stream_url.startswith("/"):
                            stream_url = "https://www.tikwm.com" + stream_url
                        v_res = client.get(stream_url)
                        v_res.raise_for_status()
                        dest_path.write_bytes(v_res.content)
                        return {
                            "title": data.get("title") or "TikTok Video",
                            "duration": data.get("duration") or 0,
                            "thumbnail": data.get("cover") or "",
                        }
    except Exception:
        pass
    return None


@router.post("/download")
def download_social_video(body: DownloadIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Download video from YouTube, TikTok, Facebook, Reels, etc. with multi-engine fallback."""
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_id = f"dl_{user.id}_{secrets.token_hex(6)}"
    dest_path = settings.outputs_dir / f"{out_id}.mp4"

    url_str = body.url.strip()

    # 1. Specialized TikTok & Douyin direct bypass
    if "tiktok.com" in url_str.lower() or "douyin.com" in url_str.lower():
        tik_info = _download_tiktok_tikwm(url_str, dest_path)
        if tik_info and dest_path.exists() and dest_path.stat().st_size > 1000:
            video_url = url_for(dest_path)
            log_activity(db, user.id, "video_download", details={"url": url_str, "title": tik_info["title"], "video_url": video_url, "source": "tikwm"})
            return {
                "video_url": video_url,
                "title": tik_info["title"],
                "duration": tik_info["duration"],
                "thumbnail": tik_info["thumbnail"],
            }

    # 2. Universal yt-dlp with updated engine & browser headers
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(500, "yt-dlp chưa được cài đặt")

    out_tmpl = str(settings.outputs_dir / f"{out_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "Sec-Fetch-Mode": "navigate",
        },
        "max_filesize": 200 * 1024 * 1024,  # 200MB
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_str, download=True)
            filename = ydl.prepare_filename(info)
            final_path = Path(filename).with_suffix(".mp4")
            if not final_path.exists():
                matches = list(settings.outputs_dir.glob(f"{out_id}.*"))
                if matches:
                    final_path = matches[0]
                else:
                    raise HTTPException(502, "Không tìm thấy file video sau khi tải")

            final_path = _ensure_web_mp4(final_path)
            video_url = url_for(final_path)
            title = info.get("title") or "Video tải về"
            duration = info.get("duration") or 0
            thumb = info.get("thumbnail") or ""

            log_activity(db, user.id, "video_download", details={"url": url_str, "title": title, "video_url": video_url, "source": "yt-dlp"})

            return {
                "video_url": video_url,
                "title": title,
                "duration": duration,
                "thumbnail": thumb,
            }
    except Exception as e:
        raise HTTPException(502, f"Không thể tải video từ link này: {e}")


# ==================== PLAYLIST / CHANNEL RESOLVE ====================

class ResolveIn(BaseModel):
    url: str = Field(min_length=5)


class ResolvedItem(BaseModel):
    id: str
    title: str
    url: str
    thumbnail: str = ""
    duration: int = 0
    like_count: int = 0
    comment_count: int = 0
    view_count: int = 0
    available_qualities: list[str] = []


class ResolveOut(BaseModel):
    items: list[ResolvedItem]
    total: int
    is_playlist: bool


class BatchDownloadItem(BaseModel):
    url: str
    title: str = ""
    quality: str = "720"   # best | 1080 | 720 | 480 | audio


class BatchDownloadIn(BaseModel):
    items: list[BatchDownloadItem] = Field(default_factory=list)
    job_id: str = ""


def _format_selector(quality: str) -> str:
    """Chuyển chất lượng người dùng chọn thành format string của yt-dlp."""
    q = (quality or "720").strip().lower()
    if q == "best":
        return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    if q == "audio":
        return "bestaudio[ext=m4a]/bestaudio/bestaudio"
    # 1080 / 720 / 480 -> height <= n
    num = "".join(ch for ch in q if ch.isdigit())
    if not num:
        num = "720"
    return (
        f"bestvideo[height<={num}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={num}][ext=mp4]/best[height<={num}]/best"
    )


def _probe_codecs(path: Path) -> tuple[str, str]:
    """Trả về (video_codec, audio_codec) của file; '' nếu không có stream."""
    try:
        v = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        vc = (v.stdout.strip().splitlines() or [""])[0].strip()
        a = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        ac = (a.stdout.strip().splitlines() or [""])[0].strip()
        return vc, ac
    except Exception:
        return "", ""


def _ensure_web_mp4(path: Path) -> Path:
    """Đảm bảo file mp4 có H.264 video + AAC audio để trình duyệt phát được (xem trước).
    Nếu codec không tương thích, transcode bằng ffmpeg. Trả về path gốc (đã ghi đè)."""
    vc, ac = _probe_codecs(path)
    if vc in ("h264", "avc1") and ac in ("aac", ""):
        return path
    tmp = path.with_name(path.stem + ".web.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
             str(tmp)],
            capture_output=True, text=True, timeout=900, check=True,
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return path
def _extract_flat_entries(info: dict) -> list[dict]:
    """Lấy danh sách entries từ info của yt-dlp (playlist/channel hoặc video đơn)."""
    entries: list[dict] = []
    if info.get("_type") == "playlist" and info.get("entries"):
        for e in info["entries"]:
            if not e:
                continue
            entries.append({
                "id": e.get("id") or "",
                "title": e.get("title") or "",
                "url": e.get("url") or e.get("webpage_url") or "",
                "thumbnail": e.get("thumbnail") or "",
                "duration": int(e.get("duration") or 0),
                "like_count": int(e.get("like_count") or 0),
                "comment_count": int(e.get("comment_count") or 0),
                "view_count": int(e.get("view_count") or 0),
                "available_qualities": ["best", "1080", "720", "480", "audio"],
            })
    else:
        entries.append({
            "id": info.get("id") or "",
            "title": info.get("title") or "",
            "url": info.get("webpage_url") or info.get("url") or "",
            "thumbnail": info.get("thumbnail") or "",
            "duration": int(info.get("duration") or 0),
            "like_count": int(info.get("like_count") or 0),
            "comment_count": int(info.get("comment_count") or 0),
            "view_count": int(info.get("view_count") or 0),
            "available_qualities": ["best", "1080", "720", "480", "audio"],
        })
    return entries


@router.post("/resolve", response_model=ResolveOut)
def resolve_social_url(body: ResolveIn, user: User = Depends(current_user)):
    """Phân tích link mạng xã hội: video đơn trả 1 mục, kênh/trang/playlist trả toàn bộ danh sách.
    Chỉ lấy metadata (không tải file), để người dùng chọn muốn tải cái nào."""
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(500, "yt-dlp chưa được cài đặt")

    url_str = body.url.strip()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # không tải, chỉ liệt kê metadata nhanh
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
        },
    }

    last_err = None
    for attempt in range(5):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_str, download=False)
            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(2)
    if last_err is not None:
        raise HTTPException(502, f"Không phân tích được link này: {last_err}")

    entries = _extract_flat_entries(info)
    items = [ResolvedItem(**e) for e in entries]
    return ResolveOut(items=items, total=len(items), is_playlist=info.get("_type") == "playlist")


@router.post("/batch-download")
def batch_download_social(body: BatchDownloadIn, user: User = Depends(current_user),
                          db: Session = Depends(get_db)):
    """Tải nhiều video/ảnh cùng lúc theo chất lượng người dùng chọn. Trả về danh sách file_url."""
    if not body.items:
        raise HTTPException(422, "Danh sách tải trống")

    results: list[dict] = []
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    job_id = (body.job_id or "").strip()
    hook = None
    state = None
    if job_id:
        DOWNLOAD_PROGRESS[job_id] = {"percent": 0, "status": "starting"}
        hook = _dl_progress(job_id)
        state = hook.__closure__ and None
    # state access via closure not trivial; instead recompute completed by tracking
    completed_count = {"n": 0}
    total_items = len(body.items)

    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(500, "yt-dlp chưa được cài đặt")

    for item in body.items:
        url_str = (item.url or "").strip()
        if not url_str:
            continue
        out_id = f"batch_{user.id}_{secrets.token_hex(6)}"
        out_tmpl = str(settings.outputs_dir / f"{out_id}.%(ext)s")

        ydl_opts = {
            "outtmpl": out_tmpl,
            "format": _format_selector(item.quality),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            },
            "max_filesize": 500 * 1024 * 1024,  # 500MB cho batch
        }
        if job_id and hook is not None:
            ydl_opts["progress_hooks"] = [hook]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_str, download=True)
            filename = ydl.prepare_filename(info)
            final_path = Path(filename)
            if not final_path.exists():
                # tìm file khớp out_id
                matches = list(settings.outputs_dir.glob(f"{out_id}.*"))
                final_path = matches[0] if matches else None
            if not final_path or not final_path.exists():
                results.append({"url": url_str, "title": item.title, "status": "failed",
                                "error": "Không tìm thấy file sau khi tải"})
                continue

            final_path = _ensure_web_mp4(final_path)
            video_url = url_for(final_path)
            log_activity(db, user.id, "video_batch_download", details={
                "url": url_str, "title": item.title, "quality": item.quality,
                "video_url": video_url, "source": "yt-dlp",
            })
            results.append({
                "url": url_str,
                "title": info.get("title") or item.title or "Video tải về",
                "status": "done",
                "video_url": video_url,
                "duration": info.get("duration") or 0,
                "thumbnail": info.get("thumbnail") or "",
                "quality": item.quality,
            })
            completed_count["n"] += 1
            if job_id:
                pct = int(completed_count["n"] / total_items * 100)
                DOWNLOAD_PROGRESS[job_id] = {"percent": min(99, pct), "status": "downloading"}
        except Exception as e:
            results.append({"url": url_str, "title": item.title, "status": "failed",
                            "error": str(e)[:300]})

    if job_id:
        DOWNLOAD_PROGRESS[job_id] = {"percent": 100, "status": "done"}
    return {"results": results, "total": len(results),
            "success": sum(1 for r in results if r["status"] == "done")}


@router.get("/preview")
def preview_video(url: str, user: User = Depends(current_user)):
    """Tải video về server tạm thời, chuyển sang H.264/AAC để xem trước trên tool (local /media URL)."""
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(500, "yt-dlp chưa được cài đặt")
    url_str = (url or "").strip()
    if not url_str:
        raise HTTPException(400, "Thiếu url")
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_id = f"preview_{user.id}_{secrets.token_hex(6)}"
    out_tmpl = str(settings.outputs_dir / f"{out_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
        },
        "max_filesize": 300 * 1024 * 1024,  # 300MB cho preview
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_str, download=True)
        filename = ydl.prepare_filename(info)
        final_path = Path(filename)
        if not final_path.exists():
            matches = list(settings.outputs_dir.glob(f"{out_id}.*"))
            final_path = matches[0] if matches else None
        if not final_path or not final_path.exists():
            raise HTTPException(502, "Không tải được video để xem trước")
        final_path = _ensure_web_mp4(final_path)
        video_url = url_for(final_path)
        return {
            "video_url": video_url,
            "title": info.get("title") or "Video xem trước",
            "duration": info.get("duration") or 0,
            "thumbnail": info.get("thumbnail") or "",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Không lấy được link xem trước: {e}")


@router.get("/download-progress/{job_id}")
def download_progress(job_id: str, user: User = Depends(current_user)):
    """Trả về tiến độ tải theo job_id (dùng cho progress bar)."""
    data = DOWNLOAD_PROGRESS.get(job_id, {"percent": 0, "status": "unknown"})
    return {"job_id": job_id, "percent": data.get("percent", 0), "status": data.get("status", "unknown")}




@router.post("/process")
def process_sub_and_blur(body: ProcessIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Burn subtitles and/or apply blur area onto video with FFmpeg."""
    src = resolve(body.video_url)
    if not src or not src.exists():
        raise HTTPException(404, "Không tìm thấy video gốc")

    out_file = settings.outputs_dir / f"proc_{user.id}_{secrets.token_hex(8)}.mp4"
    srt_file = settings.outputs_dir / f"sub_{secrets.token_hex(8)}.srt"

    try:
        filters = []

        # 1. Blur filter if specified
        # blur_region: {x: int, y: int, width: int, height: int}
        if body.blur_region:
            bx = int(body.blur_region.get("x", 0))
            by = int(body.blur_region.get("y", 0))
            bw = int(body.blur_region.get("width", 100))
            bh = int(body.blur_region.get("height", 60))
            # Boxblur filter overlay
            filters.append(f"split[vmain][vcrop];[vcrop]crop={bw}:{bh}:{bx}:{by},boxblur=15:5[vblur];[vmain][vblur]overlay={bx}:{by}")

        # 2. Subtitle filter
        if body.srt_content and body.srt_content.strip():
            srt_file.write_text(body.srt_content, encoding="utf-8")
            clean_srt = _clean_path(srt_file)

            # Alignment in ASS: 2 = bottom-center, 6 = top-center, 10 = middle-center
            align_map = {"bottom": 2, "top": 6, "middle": 10}
            align = align_map.get(body.sub_position, 2)
            fsize = body.sub_font_size or 18

            # subtitles filter
            sub_filter = f"subtitles={clean_srt}:force_style='Alignment={align},FontSize={fsize},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=30'"
            filters.append(sub_filter)

        cmd = ["ffmpeg", "-y", "-i", str(src)]
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", "-b:a", "128k", str(out_file)])

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="ignore")
            raise HTTPException(500, f"FFmpeg lỗi: {err[-500:]}")

        output_url = url_for(out_file)
        log_activity(db, user.id, "video_process", details={"output_url": output_url, "has_sub": bool(body.srt_content), "has_blur": bool(body.blur_region)})

        return {
            "output_video_url": output_url,
            "success": True,
        }
    finally:
        if srt_file.exists():
            srt_file.unlink(missing_ok=True)
