import json
import logging
import math
import os
import subprocess
from typing import Dict

log = logging.getLogger(__name__)

TARGETS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}

SCALE_FLAGS = "lanczos+accurate_rnd+full_chroma_int"
SCENE_CRF = "14"          # unchanged
FINAL_CRF = "17"          # unchanged; used only when subtitles are burned in
COVER_MAX_ENLARGE = 1.06  # up to ~6% enlargement is invisible; more is letterboxed instead


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-600:]}")


def _even_floor(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _even_ceil(value: float) -> int:
    n = math.ceil(value)
    return max(2, n + n % 2)


def probe_display_size(path: str) -> Dict[str, int]:
    """Width and height as displayed, after rotation metadata is applied."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = int(float((stream.get("tags") or {}).get("rotate") or 0))
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return {"width": width, "height": height}


def _cover_chain(sw: int, sh: int, tw: int, th: int, crop_x: float, scale: float) -> str:
    """Fill the frame by cropping. Used when that shrinks the picture or barely enlarges it."""
    cw = max(tw, _even_ceil(sw * scale))
    ch = max(th, _even_ceil(sh * scale))
    chain = [
        f"scale={cw}:{ch}:flags={SCALE_FLAGS}",
        f"crop={tw}:{th}:(iw-{tw})*{crop_x:.3f}:(ih-{th})/2",
    ]
    if scale < 0.95:
        chain.append("unsharp=5:5:0.6:5:5:0.0")  # sharpen only after a real downscale
    chain += ["fps=30", "eq=brightness=0.02:contrast=1.05", "format=yuv420p", "setsar=1"]
    return ",".join(chain)


def _fit_blur_graph(sw: int, sh: int, tw: int, th: int, scale: float) -> str:
    """Whole clip, never cropped: sharp foreground over a blurred fill of the same clip."""
    fw = min(tw, _even_floor(sw * scale))
    fh = min(th, _even_floor(sh * scale))
    fill = max(tw / sw, th / sh)
    bw = max(tw, _even_ceil(sw * fill))
    bh = max(th, _even_ceil(sh * fill))
    return (
        "[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={bw}:{bh}:flags=bicubic,crop={tw}:{th},"
        "boxblur=luma_radius=30:luma_power=2,eq=brightness=-0.05[bg];"
        f"[fgsrc]scale={fw}:{fh}:flags={SCALE_FLAGS}[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,fps=30,"
        "eq=brightness=0.02:contrast=1.05,format=yuv420p,setsar=1[v]"
    )


def normalize_and_render_scene(input_mp4: str, output_mp4: str, duration: float,
                               aspect_ratio: str, crop_x_center: float = 0.5) -> str:
    tw, th = TARGETS.get(aspect_ratio, TARGETS["16:9"])
    crop_x = max(0.0, min(1.0, float(crop_x_center)))
    src = probe_display_size(input_mp4)
    sw, sh = src["width"], src["height"]
    name = os.path.basename(input_mp4)

    cover = max(tw / sw, th / sh)  # factor needed to fill the frame
    if cover <= COVER_MAX_ENLARGE:
        mode, scale = "cover", cover
    else:
        mode, scale = "fit_blur", min(tw / sw, th / sh)

    log.info("%s %dx%d -> %dx%d as %s (x%.2f)", name, sw, sh, tw, th, mode, scale)
    if scale > COVER_MAX_ENLARGE:
        log.warning("%s is %dx%d and is enlarged x%.2f; replace it with a higher-resolution clip",
                    name, sw, sh, scale)

    cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", input_mp4,
           "-t", f"{float(duration):.3f}", "-an"]
    if mode == "cover":
        cmd += ["-vf", _cover_chain(sw, sh, tw, th, crop_x, scale)]
    else:
        cmd += ["-filter_complex", _fit_blur_graph(sw, sh, tw, th, scale), "-map", "[v]"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", SCENE_CRF,
            "-profile:v", "high", "-pix_fmt", "yuv420p", output_mp4]
    _run(cmd)
    return output_mp4


def assemble_final_video(scene_files, audio_path, srt_path, burn_in, out_mp4):
    concat_txt = out_mp4 + ".txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in scene_files:
            safe = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-i", audio_path]
    if burn_in and srt_path and os.path.exists(srt_path):
        srt = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
        vf = (f"subtitles='{srt}':force_style='FontName=Noto Sans,FontSize=16,"
              "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,"
              "BorderStyle=4,Alignment=2,MarginV=30'")
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", FINAL_CRF,
                "-profile:v", "high", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out_mp4]
    try:
        _run(cmd)
    finally:
        if os.path.exists(concat_txt):
            os.remove(concat_txt)
    return out_mp4