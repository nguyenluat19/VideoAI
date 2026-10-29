"""Adapter for local CLI-based video models (HunyuanVideo, HunyuanVideo-I2V, Wan2.1).

These models run via subprocess commands, NOT HTTP API.
Each model needs its own repo cloned into vendor/ and model weights downloaded.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import settings

# Use sys.executable to ensure we use the same Python that's running the server
PYTHON_EXECUTABLE = sys.executable


class LocalVideoError(RuntimeError):
    pass


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> tuple[int, str, str]:
    """Run a command safely (no shell=True). Returns (returncode, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _find_repo(repo_name: str) -> Path | None:
    """Look for a cloned repo in vendor/ directory."""
    vendor = Path(settings.storage_dir).parent / "vendor"
    if not vendor.exists():
        return None
    candidate = vendor / repo_name
    if candidate.exists() and (candidate / "requirements.txt").exists():
        return candidate
    # Also try storage_dir/vendor
    candidate = settings.storage_dir / "vendor" / repo_name
    if candidate.exists() and (candidate / "requirements.txt").exists():
        return candidate
    return None


def _find_model_weights(repo: Path, weight_dir_name: str) -> Path | None:
    """Look for model weights in various locations."""
    candidates = [
        repo / weight_dir_name,
        repo / "models" / weight_dir_name,
        repo / "ckpts" / weight_dir_name,
        Path(settings.storage_dir) / "models" / weight_dir_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def generate_hunyuan_video(
    prompt: str,
    duration: int = 5,
    resolution: str = "720P",
    output_path: Path | None = None,
) -> dict:
    """
    Generate video using HunyuanVideo (Tencent-Hunyuan/HunyuanVideo).
    CLI: python3 sample_video.py --prompt "..." --video-length N --save-path ./output/
    """
    repo = _find_repo("HunyuanVideo")
    if not repo:
        raise LocalVideoError(
            "HunyuanVideo chưa được clone. Chạy: git clone https://github.com/Tencent-Hunyuan/HunyuanVideo vendor/HunyuanVideo"
        )

    if output_path is None:
        output_path = settings.outputs_dir / f"hunyuan_{os.urandom(4).hex()}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # HunyuanVideo uses --video-length for frames
    frames = max(1, min(duration * 24, 129))

    # Find model weights
    model_base = _find_model_weights(repo, "HunyuanVideo")
    if not model_base:
        # Check if there's a models root with ckpts
        model_base = repo / "ckpts"
        if not model_base.exists():
            model_base = None

    cmd = [
        PYTHON_EXECUTABLE,
        "sample_video.py",
        "--prompt", prompt,
        "--video-length", str(frames),
        "--save-path", str(output_path.parent),
    ]

    if model_base:
        cmd.extend(["--model-base", str(model_base)])
    else:
        cmd.extend(["--model-base", str(repo / "ckpts")])

    returncode, stdout, stderr = _run_cmd(cmd, cwd=repo, timeout=settings.hunyuan_video_timeout_s)

    if returncode != 0:
        raise LocalVideoError(f"HunyuanVideo lỗi (code {returncode}): {stderr[-1000:]}")

    # HunyuanVideo saves to save-path with auto-generated filename
    # Look for the most recent mp4 in the output dir
    found = None
    for mp4 in sorted(output_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if mp4.stat().st_size > 1000:
            found = mp4
            break

    if not found:
        raise LocalVideoError("HunyuanVideo không tạo ra file video hợp lệ")

    # Copy to predictable name if needed
    if found != output_path:
        import shutil
        shutil.copy2(found, output_path)

    from .media import url_for
    return {"output_video_url": url_for(output_path), "model": "hunyuan-video"}


def generate_hunyuan_i2v(
    prompt: str,
    first_frame_image_url: str | None,
    duration: int = 5,
    resolution: str = "720P",
    output_path: Path | None = None,
) -> dict:
    """
    Generate video using HunyuanVideo-I2V (image-to-video).
    CLI: python3 sample_image2video.py --prompt "..." --i2v-image-path <img> --i2v-mode i2v --video-length N --save-path ./output/
    """
    repo = _find_repo("HunyuanVideo-I2V")
    if not repo:
        raise LocalVideoError(
            "HunyuanVideo-I2V chưa được clone. Chạy: git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V vendor/HunyuanVideo-I2V"
        )

    if not first_frame_image_url:
        raise LocalVideoError("HunyuanVideo-I2V cần ảnh đầu vào (character_image)")

    from .media import resolve
    image_path = resolve(first_frame_image_url)
    if not image_path or not image_path.exists():
        raise LocalVideoError(f"Không tìm thấy ảnh đầu vào: {first_frame_image_url}")

    if output_path is None:
        output_path = settings.outputs_dir / f"hunyuan_i2v_{os.urandom(4).hex()}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = max(1, min(duration * 24, 129))

    model_base = _find_model_weights(repo, "HunyuanVideo-I2V")
    if not model_base:
        model_base = repo / "ckpts"
        if not model_base.exists():
            model_base = None

    cmd = [
        PYTHON_EXECUTABLE,
        "sample_image2video.py",
        "--prompt", prompt,
        "--i2v-image-path", str(image_path.resolve()),
        "--i2v-mode", "i2v",
        "--video-length", str(frames),
        "--save-path", str(output_path.parent),
    ]

    if model_base:
        cmd.extend(["--model-base", str(model_base)])

    returncode, stdout, stderr = _run_cmd(cmd, cwd=repo, timeout=settings.hunyuan_i2v_timeout_s)

    if returncode != 0:
        raise LocalVideoError(f"HunyuanVideo-I2V lỗi (code {returncode}): {stderr[-1000:]}")

    # Look for the most recent mp4 in the output dir
    found = None
    for mp4 in sorted(output_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if mp4.stat().st_size > 1000:
            found = mp4
            break

    if not found:
        raise LocalVideoError("HunyuanVideo-I2V không tạo ra file video hợp lệ")

    if found != output_path:
        import shutil
        shutil.copy2(found, output_path)

    from .media import url_for
    return {"output_video_url": url_for(output_path), "model": "hunyuan-video-i2v"}


def generate_wan21_video(
    prompt: str,
    duration: int = 5,
    resolution: str = "720P",
    output_path: Path | None = None,
) -> dict:
    """
    Generate video using Wan2.1 (Wan-Video/Wan2.1).
    CLI: python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "..." --save_file output.mp4
    """
    repo = _find_repo("Wan2.1")
    if not repo:
        raise LocalVideoError(
            "Wan2.1 chưa được clone. Chạy: git clone https://github.com/Wan-Video/Wan2.1 vendor/Wan2.1"
        )

    if output_path is None:
        output_path = settings.outputs_dir / f"wan21_{os.urandom(4).hex()}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map resolution to Wan size
    size_map = {
        "720P": "1280*720",
        "480P": "832*480",
        "1080P": "1920*1080",
    }
    size = size_map.get(resolution, "1280*720")

    # Find model weights
    ckpt_dir = repo / "Wan2.1-T2V-14B"
    if not ckpt_dir.exists():
        # Try t2v-1.3B
        ckpt_dir = repo / "Wan2.1-T2V-1.3B"
        if not ckpt_dir.exists():
            ckpt_dir = None

    cmd = [
        PYTHON_EXECUTABLE,
        "generate.py",
        "--task", "t2v-1.3B" if ckpt_dir is None or "1.3B" in str(ckpt_dir) else "t2v-14B",
        "--size", size,
        "--prompt", prompt,
        "--save_file", str(output_path),
    ]

    if ckpt_dir:
        cmd.extend(["--ckpt_dir", str(ckpt_dir)])

    returncode, stdout, stderr = _run_cmd(cmd, cwd=repo, timeout=settings.wan21_timeout_s)

    if returncode != 0:
        raise LocalVideoError(f"Wan2.1 lỗi (code {returncode}): {stderr[-1000:]}")

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise LocalVideoError("Wan2.1 không tạo ra file video hợp lệ")

    from .media import url_for
    return {"output_video_url": url_for(output_path), "model": "wan2.1"}
