"""AI video flows: MiniMax-H3 text/image -> video, and the dance combo (H3 reference -> MimicMotion)."""
import subprocess
from pathlib import Path

from .. import ai, minimax, pricing
from ..config import DEFAULT_SCRIPT_MODEL, settings
from ..db import SessionLocal
from ..media import resolve, url_for
from .fast import RenderError

DANCE_MAX_SECONDS = 10


def _opts(values: dict) -> dict:
    return {"duration": int(values.get("duration", 5)), "ratio": values.get("ratio", "9:16"),
            "resolution": values.get("resolution", "768P")}


def generate(gen_id: int, values: dict, *, dance: bool, on_progress=lambda: None) -> str:
    """Create the video with MiniMax-H3. dance=True makes a reference clip (outputs/ref_<id>.mp4)."""
    dest = settings.outputs_dir / (f"ref_{gen_id}.mp4" if dance else f"gen_{gen_id}.mp4")
    first = resolve(values.get("first_frame_image", "")) if not dance else None
    with SessionLocal() as db:
        allow_llm = DEFAULT_SCRIPT_MODEL not in pricing.disabled_models(db)
    prompt = ai.optimize_video_prompt(values.get("prompt", ""), dance=dance, allow_llm=allow_llm)
    try:
        minimax.generate(prompt, dest, first_frame=first, on_progress=on_progress, **_opts(values))
    except minimax.VideoError as e:
        raise RenderError(str(e)) from e
    return url_for(dest)


def mux_audio(video_url: str, audio_source_url: str) -> None:
    """MimicMotion output is silent; put the soundtrack H3 generated for the reference back on it."""
    video, src = resolve(video_url), resolve(audio_source_url)
    if not video or not src:
        return
    tmp = video.with_suffix(".mux.mp4")
    proc = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(video), "-i", str(src),
                           "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(video)
    else:
        tmp.unlink(missing_ok=True)   # keep the silent video rather than failing a finished render
