import os
import re
import json
import subprocess
import httpx
from typing import List, Dict, Any, Tuple
from app.config import settings


def detect_script_language(text: str) -> str:
    """Detects Tamil ('ta'), Hindi ('hi'), or defaults to English ('en') via Unicode ranges."""
    if not text:
        return "en"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def get_audio_info(audio_path: str) -> Tuple[float, bool]:
    """Returns (duration_seconds, is_completely_silent)."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        raise ValueError("CORRUPTED_OR_INVALID_AUDIO")
    duration = float(res.stdout.strip())

    # Check if audio is completely silent using volumedetect
    vol_cmd = ["ffmpeg", "-i", audio_path, "-af", "volumedetect", "-f", "null", "-"]
    vol_res = subprocess.run(vol_cmd, capture_output=True, text=True)
    is_silent = (
        "mean_volume: -91.0 dB" in vol_res.stderr
        or "max_volume: -91.0 dB" in vol_res.stderr
        or "max_volume: -inf dB" in vol_res.stderr
    )
    return duration, is_silent


def _align_known_script_to_duration(script_text: str, duration: float) -> List[Dict[str, Any]]:
    """
    Splits a known script (English, Tamil, or Hindi) on sentence boundaries and aligns
    proportionally to the exact audio duration, preserving 100% of native Unicode characters.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", script_text.strip()) if s.strip()]
    if not sentences:
        sentences = [script_text.strip()]

    total_chars = max(sum(len(s) for s in sentences), 1)
    segments = []
    curr = 0.0

    for idx, sent in enumerate(sentences):
        share = (len(sent) / total_chars) * duration
        end = duration if idx == len(sentences) - 1 else min(duration, curr + max(share, 1.5))
        if end > curr:
            segments.append({
                "id": idx,
                "start": round(curr, 3),
                "end": round(end, 3),
                "text": sent,
            })
            curr = end

    if segments:
        segments[-1]["end"] = round(duration, 3)

    return segments


async def transcribe_and_align(
    audio_path: str,
    fallback_script: str = ""
) -> Tuple[List[Dict[str, Any]], float, str]:
    duration, is_silent = get_audio_info(audio_path)

    if duration <= 0.1:
        raise ValueError("EMPTY_AUDIO")
    if duration > getattr(settings, "MAX_AUDIO_DURATION_SEC", 600.0):
        raise ValueError("AUDIO_TOO_LONG")
    if is_silent and settings.PROVIDER_MODE != "mock":
        raise ValueError("SILENT_AUDIO_DETECTED")

    clean_script = (fallback_script or "").strip()
    lang_code = detect_script_language(clean_script)

    # 1. If the user provided a Tamil or Hindi script, preserve exact native Unicode sentences
    # so Whisper never transliterates them into Latin/Tanglish ("Kalaiil Chekiramaka...").
    if clean_script and lang_code in ("ta", "hi"):
        segments = _align_known_script_to_duration(clean_script, duration)
        return segments, duration, f"unicode-script-forced-aligner-{lang_code}"

    # 2. Use Groq Whisper API if key is configured and in live mode
    groq_key = getattr(settings, "GROQ_API_KEY", None)
    if settings.PROVIDER_MODE == "live" and groq_key and "your_" not in groq_key:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
                    data: Dict[str, Any] = {
                        "model": "whisper-large-v3-turbo",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": ["segment", "word"],
                    }
                    if clean_script:
                        data["language"] = lang_code
                        data["prompt"] = clean_script[:220]

                    headers = {"Authorization": f"Bearer {groq_key}"}
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        data=data,
                        files=files,
                    )
                    if resp.status_code == 200:
                        payload = resp.json()
                        text = (payload.get("text") or "").strip()
                        raw_segments = payload.get("segments") or []
                        if not text or len(raw_segments) == 0:
                            raise ValueError("MUSIC_ONLY_OR_NO_SPEECH")

                        segments = [
                            {
                                "id": idx,
                                "start": round(float(s.get("start", 0.0)), 3),
                                "end": round(float(s.get("end", duration)), 3),
                                "text": (s.get("text") or "").strip(),
                            }
                            for idx, s in enumerate(raw_segments)
                            if (s.get("text") or "").strip()
                        ]
                        if segments:
                            segments[-1]["end"] = round(duration, 3)
                            return segments, duration, "groq/whisper-large-v3-turbo"
        except ValueError:
            raise
        except Exception:
            pass

    # 3. Fallback deterministic proportional alignment
    text_to_align = clean_script or "Saving money consistently builds long term financial freedom. Start tracking expenses today."
    segments = _align_known_script_to_duration(text_to_align, duration)
    return segments, duration, "proportional-forced-aligner"


def write_srt(segments: List[Dict[str, Any]], srt_path: str):
    def fmt_ts(sec: float) -> str:
        hrs = int(sec // 3600)
        mins = int((sec % 3600) // 60)
        secs = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}")
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))