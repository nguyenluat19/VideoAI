"""Fast render: swap text / images / logo / background on a template video with FFmpeg."""
import subprocess
import tempfile
from pathlib import Path

from ..config import settings
from ..media import resolve, url_for

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


class RenderError(RuntimeError):
    pass


def _font() -> str:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return f
    raise RenderError("Không tìm thấy font hệ thống để vẽ chữ")


def _ffmpeg_path(p) -> str:
    """Escape colons for ffmpeg filtergraph on Windows, e.g. C:/ -> C\\:/."""
    s = str(Path(p).resolve()).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = s[0] + "\\:" + s[2:]
    return s


def _expr(v) -> str:
    """Coordinates may be numbers or ffmpeg expressions like '(w-text_w)/2'."""
    return str(v)


def build_command(schema: dict, values: dict, out: Path, tmp: Path) -> list[str]:
    width = int(schema.get("width", 720))
    height = int(schema.get("height", 1280))
    duration = float(schema.get("duration", 5))
    fps = int(schema.get("fps", 30))
    layers = schema.get("layers", [])

    inputs: list[str] = []
    n = 0

    def add_input(*args: str) -> int:
        nonlocal n
        inputs.extend(args)
        n += 1
        return n - 1

    def layer_value(layer: dict):
        return values.get(layer["id"], layer.get("default"))

    # Base: background layer > template video > solid colour.
    bg_layer = next((l for l in layers if l.get("type") == "background"), None)
    bg_file = resolve(layer_value(bg_layer)) if bg_layer and layer_value(bg_layer) else None
    base_video = resolve(schema.get("_base_video", ""))
    if bg_file:
        base = add_input("-loop", "1", "-t", str(duration), "-i", str(bg_file))
    elif base_video:
        base = add_input("-stream_loop", "-1", "-t", str(duration), "-i", str(base_video))
    else:
        base = add_input("-f", "lavfi", "-t", str(duration),
                         "-i", f"color=c={schema.get('bg_color', 'black')}:s={width}x{height}:r={fps}")

    filters = [f"[{base}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
               f"crop={width}:{height},setsar=1,fps={fps}[v0]"]
    cur = "v0"
    step = 0

    for layer in layers:
        kind = layer.get("type")
        if kind in ("image", "logo"):
            src = resolve(layer_value(layer) or "")
            if not src:
                continue
            idx = add_input("-loop", "1", "-t", str(duration), "-i", str(src))
            step += 1
            w, h = int(layer.get("w", 200)), int(layer.get("h", -1))
            filters.append(f"[{idx}:v]scale={w}:{h}[s{step}]")
            filters.append(f"[{cur}][s{step}]overlay=x={_expr(layer.get('x', 0))}:y={_expr(layer.get('y', 0))}"
                           f":shortest=1[v{step}]")
            cur = f"v{step}"
        elif kind == "text":
            text = str(layer_value(layer) or "")
            if not text:
                continue
            step += 1
            tf = tmp / f"text{step}.txt"
            tf.write_text(text, encoding="utf-8")  # textfile= avoids all quoting/escaping issues
            color = str(layer.get("color", "white")).replace(":", "")
            extra = ""
            if layer.get("box"):
                extra = ":box=1:boxcolor=black@0.5:boxborderw=12"
            font_p = _ffmpeg_path(_font())
            tf_p = _ffmpeg_path(tf)
            filters.append(
                f"[{cur}]drawtext=fontfile='{font_p}':textfile='{tf_p}':fontsize={int(layer.get('fontsize', 56))}"
                f":fontcolor={color}:x={_expr(layer.get('x', '(w-text_w)/2'))}"
                f":y={_expr(layer.get('y', 'h*0.8'))}{extra}[v{step}]")
            cur = f"v{step}"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(filters), "-map", f"[{cur}]"]
    # Keep template audio if the base is a real video that has some.
    if base_video and not bg_file:
        cmd += ["-map", f"{base}:a?"]
    cmd += ["-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)]
    return cmd


def render(gen_id: int, schema: dict, base_video_url: str, values: dict) -> str:
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / f"gen_{gen_id}.mp4"
    schema = {**schema, "_base_video": base_video_url}
    with tempfile.TemporaryDirectory() as td:
        cmd = build_command(schema, values, out, Path(td))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not out.exists():
        raise RenderError(f"FFmpeg lỗi: {proc.stderr[-800:]}")
    return url_for(out)
