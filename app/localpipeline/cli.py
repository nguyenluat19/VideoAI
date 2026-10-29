"""CLI entry point for the local video pipeline."""
import argparse
import json
import os
import sys

from .config import PipelineConfig
from .pipeline import run_pipeline, PipelineError


def main():
    parser = argparse.ArgumentParser(
        description="Local open-source image->prompt->image->video pipeline",
    )
    parser.add_argument("image", nargs="?", help="Input image path (JPEG/PNG)")
    parser.add_argument("--instruction", "-i", default=None, help="User instruction for the VLM")
    parser.add_argument("--mode", choices=["recreate", "animate"], default="recreate",
                        help="recreate = new image from prompt; animate = animate original image")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--check", action="store_true", help="Run smoke test (small resolution, few frames)")
    parser.add_argument("--config", default=None, help="JSON config file path")
    args = parser.parse_args()

    # Load config
    config = PipelineConfig()
    if args.config:
        import json as json_mod
        with open(args.config) as f:
            overrides = json_mod.load(f)
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    if args.check:
        # Smoke test: use a tiny image
        import tempfile
        from PIL import Image
        img = Image.new("RGB", (64, 64), color="red")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        args.image = tmp.name
        config.t2i_image_size = 64
        config.i2v_num_frames = 5
        print("Running smoke test with 64x64 image, 5 frames...")

    if not args.image:
        parser.error("image is required (or use --check)")

    # Progress callback
    def on_progress(step, pct):
        print(f"  [{step}] {pct*100:.0f}%")

    print(f"Running pipeline: mode={args.mode}, image={args.image}")
    config.log_config()

    try:
        result = run_pipeline(
            image_path=args.image,
            user_instruction=args.instruction,
            mode=args.mode,
            options=None,
            job_dir=args.output_dir,
            on_progress=on_progress,
        )
        print("\n=== Pipeline Result ===")
        print(f"Image prompt: {result.image_prompt[:200]}")
        print(f"Motion prompt: {result.motion_prompt[:200]}")
        print(f"Image path: {result.image_path}")
        print(f"Video path: {result.video_path}")
        print(f"Timings: {result.timings}")
        print(f"VRAM peaks: {result.vram_peaks}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")

        # Verify video with ffprobe
        if result.video_path:
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", result.video_path],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                print(f"Video duration: {r.stdout.strip()}s (verified by ffprobe)")
            else:
                print(f"ffprobe warning: {r.stderr[:200]}")

        print("\nDone!")
    except PipelineError as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()