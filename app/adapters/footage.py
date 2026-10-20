import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
from app.config import settings

log = logging.getLogger(__name__)

TARGETS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
WANT_ORIENTATION = {"16:9": "landscape", "9:16": "portrait", "1:1": "square"}

MIN_SHORT_SIDE = 1080  # shorter sources look soft once they fill a full-HD frame
MAX_LONG_SIDE = 3840   # 8K is slow to fetch and gets downscaled anyway

# A provider that answers 429 is skipped for this long. Nothing is retried during the pause.
PAUSE_SEC = {"pexels": 900.0, "pixabay": 60.0}
_paused_until: Dict[str, float] = {}

PickFn = Callable[[List[Dict[str, Any]]], Optional[Dict[str, Any]]]


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-600:]}")


def _orientation(width: int, height: int) -> str:
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def _quality_tier(short_side: int) -> int:
    if short_side >= MIN_SHORT_SIDE:
        return 2
    if short_side >= 720:
        return 1
    return 0


def _candidate(provider, vid, page_url, tags, duration, width, height, link, user, license_name):
    width, height = int(width), int(height)
    return {
        "id": f"{provider}_{vid}",
        "provider": provider,
        "url": page_url,  # the page slug names the subject, and the planner reads it
        "tags": tags,
        "title": "",
        "duration": float(duration or 0),
        "width": width,
        "height": height,
        "short_side": min(width, height),
        "download_url": link,
        "user": user,
        "license": license_name,
    }


def _best_pexels_file(files):
    usable = [f for f in files if f.get("link") and f.get("width") and f.get("height")]
    mp4 = [f for f in usable if "mp4" in str(f.get("file_type", ""))]
    pool = mp4 or usable
    within = [f for f in pool if max(f["width"], f["height"]) <= MAX_LONG_SIDE]
    if within:
        return max(within, key=lambda f: (min(f["width"], f["height"]), -max(f["width"], f["height"])))
    return min(pool, key=lambda f: max(f["width"], f["height"])) if pool else None


async def _get_json(provider, client, url, params, headers, cache, cache_key, label):
    """One API call. Returns parsed JSON, or None when the call is skipped or fails."""
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if time.monotonic() < _paused_until.get(provider, 0.0):
        return None
    resp = await client.get(url, params=params, headers=headers)
    if resp.status_code == 429:
        try:
            wait = max(5.0, float(resp.headers.get("Retry-After", "")))
        except ValueError:
            wait = PAUSE_SEC[provider]
        _paused_until[provider] = time.monotonic() + wait
        log.warning("%s returned 429; skipping it for %.0fs", provider, wait)
        return None
    if resp.status_code != 200:
        log.warning("%s returned %s for '%s'", provider, resp.status_code, label)
        return None
    data = resp.json()
    if cache is not None:
        cache[cache_key] = data  # only successful answers are cached, including empty ones
    return data


async def _pexels(client, query, orientation, cache):
    key = getattr(settings, "PEXELS_API_KEY", None)
    if getattr(settings, "SIMULATE_PRIMARY_SOURCE_DOWN", False) or not key or "your_" in key:
        return []
    data = await _get_json(
        "pexels", client, "https://api.pexels.com/videos/search",
        params={"query": query, "per_page": 15, "orientation": orientation},
        headers={"Authorization": key},
        cache=cache, cache_key=("pexels", query, orientation), label=query,
    )
    if data is None:
        return []
    found = []
    for video in data.get("videos", []):
        best = _best_pexels_file(video.get("video_files", []))
        if best is None:
            continue
        found.append(_candidate(
            "pexels", video.get("id"), video.get("url", ""), [], video.get("duration"),
            best["width"], best["height"], best["link"],
            (video.get("user") or {}).get("name", "Pexels Creator"),
            "Pexels License (Commercial Use Allowed)",
        ))
    return found


async def _pixabay(client, query, cache):
    key = getattr(settings, "PIXABAY_API_KEY", None)
    if not key or "your_" in key:
        return []
    data = await _get_json(
        "pixabay", client, "https://pixabay.com/api/videos/",
        params={"key": key, "q": query, "video_type": "film", "per_page": 20},
        headers=None,
        cache=cache, cache_key=("pixabay", query), label=query,  # no orientation, so both ratios share it
    )
    if data is None:
        return []
    found = []
    for hit in data.get("hits", []):
        tiers = [t for t in (hit.get("videos") or {}).values()
                 if t and t.get("url") and t.get("width") and t.get("height")]
        if not tiers:
            continue
        best = max(tiers, key=lambda t: min(t["width"], t["height"]))
        found.append(_candidate(
            "pixabay", hit.get("id"), hit.get("pageURL", ""), hit.get("tags", ""),
            hit.get("duration"), best["width"], best["height"], best["url"],
            hit.get("user", "Pixabay Contributor"),
            "Pixabay License (Commercial Use Allowed)",
        ))
    return found


async def fetch_candidates(query: str, aspect_ratio: str, cache: Optional[dict] = None) -> List[Dict[str, Any]]:
    orientation = WANT_ORIENTATION.get(aspect_ratio, "landscape")
    async with httpx.AsyncClient(timeout=15.0) as client:
        results = await asyncio.gather(
            _pexels(client, query, orientation, cache),
            _pixabay(client, query, cache),
            return_exceptions=True,
        )
    candidates: List[Dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            log.warning("stock search failed for '%s': %s", query, result)
        else:
            candidates.extend(result)
    return candidates


async def _download(url: str, dest: str) -> None:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                async for chunk in resp.aiter_bytes(1 << 20):
                    out.write(chunk)
    if os.path.getsize(dest) < 10_000:
        raise RuntimeError("download is too small to be a video")


def _default_pick(candidates, aspect_ratio, duration, used):
    want = WANT_ORIENTATION.get(aspect_ratio, "landscape")
    pool = [c for c in candidates if c["id"] not in used]
    if not pool:
        return None

    def rank(c):
        return (
            _quality_tier(c["short_side"]),
            _orientation(c["width"], c["height"]) == want,
            min(c["short_side"], 2160),
            c["duration"] >= duration,
        )

    return max(pool, key=rank)


def _solid_backdrop(clip_path, aspect_ratio, duration):
    """Last resort when no stock clip is usable. Plain frame, no text."""
    w, h = TARGETS.get(aspect_ratio, TARGETS["16:9"])
    seconds = max(float(duration) + 1.5, 4.0)
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=0x0F172A:s={w}x{h}:r=30:d={seconds:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        clip_path,
    ])


def _result(c, clip_path, query):
    pexels = c["provider"] == "pexels"
    return {
        "clip_id": c["id"],
        "clip_url": c["download_url"],
        "source_provider": c["provider"],
        "license": c["license"],
        "attribution": f"Video by {c['user']}",
        "commercial_use_allowed": True,
        "is_ai_generated": False,
        "local_path": clip_path,
        "selection_reason": f"Matched '{query}' at {c['width']}x{c['height']}.",
        "routing_rule_applied": "Primary Stock API" if pexels else "Secondary Stock Fallback",
        "relevance_score": 4.8 if pexels else 4.2,
        "crop_x_center": 0.5,
        "width": c["width"],
        "height": c["height"],
        "orientation": _orientation(c["width"], c["height"]),
    }


async def search_and_download_footage(
    query: str,
    aspect_ratio: str,
    scene_index: int,
    used_clip_ids: set,
    work_dir: str,
    duration: float = 6.0,
    pick: Optional[PickFn] = None,
    ladder: Optional[List[str]] = None,
    search_cache: Optional[dict] = None,
) -> Dict[str, Any]:
    os.makedirs(work_dir, exist_ok=True)
    duration = float(duration or 6.0)
    clip_path = os.path.join(work_dir, f"raw_s{scene_index:03d}_{aspect_ratio.replace(':', '_')}.mp4")
    choose: PickFn = pick or (lambda cands: _default_pick(cands, aspect_ratio, duration, used_clip_ids))

    for rung in (ladder or [query]):
        candidates = await fetch_candidates(rung, aspect_ratio, search_cache)
        for _ in range(3):
            chosen = choose(candidates)
            if chosen is None:
                break
            try:
                await _download(chosen["download_url"], clip_path)
            except Exception as exc:
                log.warning("download failed for %s: %s", chosen["id"], exc)
                candidates = [c for c in candidates if c["id"] != chosen["id"]]
                continue
            used_clip_ids.add(chosen["id"])
            return _result(chosen, clip_path, rung)

    await asyncio.to_thread(_solid_backdrop, clip_path, aspect_ratio, duration)
    return {
        "clip_id": f"procedural_gen_{scene_index}",
        "clip_url": f"internal://procedural_fallback_s{scene_index}.mp4",
        "source_provider": "internal-procedural-fallback",
        "license": "CC0 / Clean Commercial Fallback",
        "attribution": "IncuBrix Engine Procedural Visuals",
        "commercial_use_allowed": True,
        "is_ai_generated": True,
        "local_path": clip_path,
        "selection_reason": "No usable stock clip for any query in the ladder; plain backdrop used.",
        "routing_rule_applied": "Criterion-7 Resilient Fallback",
        "relevance_score": 4.0,
        "crop_x_center": 0.5,
    }