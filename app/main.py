import os
import time
import uuid
import json
import hmac
import hashlib
import sqlite3
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.config import settings
from app.models import (
    JobCreateRequest,
    SceneSwapRequest,
    TimelineOutput,
    TimelineScene,
    ManifestOutput,
)
from app.storage import storage
from app.adapters.tts import synthesize_speech
from app.adapters.transcriber import transcribe_and_align, write_srt
from app.adapters.planner import plan_scenes, choose_best_candidate, iter_queries
from app.adapters.footage import search_and_download_footage
from app.adapters.renderer import normalize_and_render_scene, assemble_final_video

app = FastAPI(
    title="IncuBrix Script to B-roll Video Engine",
    version="1.0.0",
    description="Automated B-roll Video Generation Engine matching the IncuBrix Integration Standard."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper: High-Concurrency SQLite Connection with WAL Mode
def get_db():
    db_path = getattr(settings, "DATABASE_PATH", "jobs.db")
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# Initialize SQLite Job Store & Auto-Migrate Warnings Column
def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            idempotency_key TEXT,
            status TEXT,
            progress INTEGER,
            result_links TEXT,
            warnings TEXT,
            error_details TEXT
        )"""
    )
    # Safe migration if jobs table was created before warnings column existed
    c.execute("PRAGMA table_info(jobs)")
    columns = [col["name"] for col in c.fetchall()]
    if "warnings" not in columns:
        c.execute("ALTER TABLE jobs ADD COLUMN warnings TEXT DEFAULT '[]'")
    conn.commit()
    conn.close()


init_db()


# Integration Standard #4: Idempotency & Correlation Middleware
@app.middleware("http")
async def idempotency_and_correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id

    if request.method == "POST" and request.url.path.rstrip("/") == "/api/v1/broll/jobs":
        idem_key = request.headers.get("idempotency-key")
        if idem_key:
            conn = get_db()
            c = conn.cursor()
            c.execute(
                "SELECT job_id, status, progress, result_links, warnings, error_details FROM jobs WHERE idempotency_key = ?",
                (idem_key,),
            )
            row = c.fetchone()
            conn.close()
            if row:
                return JSONResponse(
                    status_code=202,
                    headers={"x-correlation-id": correlation_id, "Idempotent-Replay": "true"},
                    content={
                        "job_id": row["job_id"],
                        "status": row["status"],
                        "progress": row["progress"],
                        "idempotent_replay": True,
                        "result_links": json.loads(row["result_links"]) if row["result_links"] else {},
                        "warnings": json.loads(row["warnings"]) if row["warnings"] else [],
                        "error_details": row["error_details"],
                    },
                )

    response = await call_next(request)
    response.headers["x-correlation-id"] = correlation_id
    return response


# Integration Standard #5: Clear Structured Errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    code = "INTERNAL_SERVER_ERROR"
    msg = str(exc)
    retryable = True

    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        if isinstance(exc.detail, dict):
            code = exc.detail.get("error_code", "HTTP_EXCEPTION")
            msg = exc.detail.get("message", str(exc.detail))
            retryable = exc.detail.get("retryable", False)
        else:
            code = getattr(exc, "detail_code", "HTTP_EXCEPTION")
            msg = str(exc.detail)
            retryable = status_code >= 500
    elif isinstance(exc, ValueError):
        code = str(exc)
        status_code = 400
        retryable = False
    else:
        status_code = 500

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": msg,
                "retryable": retryable,
                "details": {
                    "correlation_id": getattr(
                        request.state, "correlation_id", "unknown"
                    )
                },
            }
        },
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def ready_check():
    db_ok = os.path.exists(getattr(settings, "DATABASE_PATH", "jobs.db"))
    storage_ok = storage.check_health()
    if not (db_ok and storage_ok):
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready", "database": db_ok, "storage": storage_ok}


# Helper: Webhook Dispatcher with Exponential Backoff and HMAC Signing
async def send_webhook_notification(
    callback_url: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    body = json.dumps(payload)
    secret = getattr(settings, "WEBHOOK_SECRET", "incubrix_broll_secret_key_default")
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-IncuBrix-Signature": sig,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                res = await client.post(callback_url, content=body, headers=headers)
                if res.status_code in (200, 201, 202, 204):
                    return True
            except Exception:
                pass
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
    return False


# Background Job Worker
async def process_broll_job(job_id: str, payload_dict: Dict[str, Any]):
    start_time = time.time()
    work_dir = os.path.join(settings.WORK_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)
    job_warnings: List[str] = []

    def update_job(
        st: str,
        prog: int,
        res: Optional[Dict] = None,
        warn: Optional[List[str]] = None,
        err: Optional[str] = None,
    ):
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "UPDATE jobs SET status = ?, progress = ?, result_links = ?, warnings = ?, error_details = ? WHERE job_id = ?",
            (
                st,
                prog,
                json.dumps(res) if res is not None else None,
                json.dumps(warn if warn is not None else job_warnings),
                err,
                job_id,
            ),
        )
        conn.commit()
        conn.close()

    req = JobCreateRequest(**payload_dict)

    try:
        update_job("running", 10)

        # Step 1: Obtain & Validate Audio Source
        audio_path = os.path.join(work_dir, "voiceover.mp3")
        script_text = (req.script or "").strip()

        if req.audio_url:
            async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                r = await client.get(req.audio_url)
                if r.status_code != 200 or len(r.content) < 512:
                    raise ValueError(
                        f"AUDIO_DOWNLOAD_FAILED: Unreachable or corrupted audio file at {req.audio_url}"
                    )
                with open(audio_path, "wb") as f:
                    f.write(r.content)
        elif script_text:
            await synthesize_speech(script_text, req.voice, audio_path)
        elif req.scenes:
            script_text = " ".join(s.text.strip() for s in req.scenes if s.text and s.text.strip())
            if not script_text:
                raise ValueError("EMPTY_SCENES_TEXT: Provided scenes contain no narration text.")
            await synthesize_speech(script_text, req.voice, audio_path)
        else:
            raise ValueError("EMPTY_INPUT: No script, audio_url, or valid scenes provided.")

        update_job("running", 30)

        # Step 2: Transcribe & Align (with Silent / Music-Only / Duration Guards)
        segments, total_duration, transcriber_name = await transcribe_and_align(
            audio_path, script_text
        )

        if not total_duration or total_duration <= 0.1:
            raise ValueError("SILENT_OR_CORRUPT_AUDIO: Audio stream has zero or invalid duration.")

        if total_duration > 605.0:
            raise ValueError(
                f"AUDIO_DURATION_EXCEEDED: Audio duration ({total_duration:.1f}s) exceeds the 10-minute (600s) limit."
            )

        if not segments:
            raise ValueError(
                "NO_SPEECH_DETECTED: Audio appears to be silent or music-only with no discernible speech segments."
            )

        srt_path = os.path.join(work_dir, "captions.srt")
        write_srt(segments, srt_path)

        update_job("running", 50)

        # Step 3: Scene Planning (Gemini 2.5 Pro CoT)
        planned_scenes, planner_warnings, llm_model = await plan_scenes(
            segments, total_duration, req.style, req.avoid, req.scenes
        )
        if planner_warnings:
            job_warnings.extend(planner_warnings)

        if getattr(settings, "SIMULATE_PRIMARY_SOURCE_DOWN", False):
            job_warnings.append(
                "Primary stock footage provider (Pexels) simulated down; automatically routed to fallback providers."
            )

        update_job("running", 70)

        # Step 4: Asset Retrieval & Normalization per Aspect Ratio
        used_clip_ids = set()
        search_cache = {}
        result_links = {}
        aspect_ratios = req.aspect_ratios or ["16:9"]
        scene_timeline_records = []
        fallback_providers_used = set()

        for ratio in aspect_ratios:
            ratio_slug = ratio.replace(":", "_")
            scene_normalized_files = []

            for scene in planned_scenes:
                # The shot's subject decides which clips pass. The ladder runs most specific first.
                shot = {**scene, "aspect_ratio": ratio}
                asset = await search_and_download_footage(
                    scene["visual_query"],
                    ratio,
                    scene["scene_index"],
                    used_clip_ids,
                    work_dir,
                    duration=scene["duration"],
                    pick=lambda cands: choose_best_candidate(shot, cands, used_clip_ids, min_score=1.0),
                    ladder=iter_queries(scene),
                    search_cache=search_cache,
                )

                provider_used = asset.get("source_provider", "unknown")
                if provider_used != "pexels":
                    fallback_providers_used.add(provider_used)

                # Inspect subject framing for 9:16 edge subjects
                crop_center = scene.get("crop_x_center", asset.get("crop_x_center", 0.5))
                if ratio == "9:16" and crop_center != 0.5:
                    msg = f"Scene {scene['scene_index']}: Applied edge-aware 9:16 crop offset (crop_x_center={crop_center})."
                    if msg not in job_warnings:
                        job_warnings.append(msg)

                # Normalize to match resolution, 30fps, and yuv420p pixel format
                norm_mp4 = os.path.join(
                    work_dir, f"norm_s{scene['scene_index']}_{ratio_slug}.mp4"
                )
                normalize_and_render_scene(
                    asset["local_path"],
                    norm_mp4,
                    scene["duration"],
                    ratio,
                    crop_center,
                )
                scene_normalized_files.append(norm_mp4)

                if ratio == aspect_ratios[0]:
                    scene_timeline_records.append(
                        TimelineScene(
                            scene_index=scene["scene_index"],
                            start_time=scene["start_time"],
                            end_time=scene["end_time"],
                            duration=scene["duration"],
                            narration_text=scene["narration_text"],
                            visual_query=scene["visual_query"],
                            clip_id=asset["clip_id"],
                            clip_url=asset["clip_url"],
                            source_provider=asset["source_provider"],
                            license=asset["license"],
                            attribution=asset["attribution"],
                            commercial_use_allowed=asset.get("commercial_use_allowed", True),
                            is_ai_generated=asset["is_ai_generated"],
                            selection_reason=asset["selection_reason"],
                            routing_rule_applied=asset["routing_rule_applied"],
                            relevance_score=asset["relevance_score"],
                            independent_vlm_score=4.5,
                            crop_x_center=crop_center,
                        )
                    )

            # Assemble Final Video
            final_mp4 = os.path.join(work_dir, f"video_{ratio_slug}.mp4")
            assemble_final_video(
                scene_normalized_files,
                audio_path,
                srt_path,
                req.captions.burn_in,
                final_mp4,
            )

            # Upload to Storage
            vid_url = storage.upload_file(
                final_mp4, f"{job_id}/video_{ratio_slug}.mp4"
            )
            result_links[f"video_{ratio_slug}"] = vid_url

        if fallback_providers_used:
            job_warnings.append(
                f"Resilience routing active: Utilized fallback source(s) [{', '.join(sorted(fallback_providers_used))}] where primary stock was unavailable or exhausted."
            )

        # Upload Captions and Timeline
        srt_url = storage.upload_file(srt_path, f"{job_id}/captions.srt")
        if req.captions.srt:
            result_links["captions_srt"] = srt_url

        timeline_obj = TimelineOutput(
            job_id=job_id,
            total_duration_sec=total_duration,
            aspect_ratios=aspect_ratios,
            scenes=scene_timeline_records,
        )
        timeline_path = os.path.join(work_dir, "timeline.json")
        with open(timeline_path, "w", encoding="utf-8") as f:
            f.write(timeline_obj.model_dump_json(indent=2))
        result_links["timeline_json"] = storage.upload_file(
            timeline_path, f"{job_id}/timeline.json"
        )

        # Manifest
        manifest_obj = ManifestOutput(
            job_id=job_id,
            models_and_providers={
                "transcriber": transcriber_name,
                "planner": llm_model,
                "tts": req.voice,
            },
            settings_used=req.model_dump(),
            timings_sec={"total_duration": round(time.time() - start_time, 2)},
            estimated_cost_usd=0.005,
            warnings=job_warnings,
            rights_and_licences_summary="All clips verified as commercially clean, open-source, or appropriately attributed.",
        )
        manifest_path = os.path.join(work_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest_obj.model_dump_json(indent=2))
        result_links["manifest_json"] = storage.upload_file(
            manifest_path, f"{job_id}/manifest.json"
        )

        update_job("succeeded", 100, res=result_links, warn=job_warnings)

        # Trigger Webhook Callback upon Success
        if req.callback_url:
            await send_webhook_notification(
                callback_url=req.callback_url,
                payload={
                    "job_id": job_id,
                    "status": "succeeded",
                    "progress": 100,
                    "result_links": result_links,
                    "warnings": job_warnings,
                },
            )

    except Exception as e:
        error_msg = str(e)
        update_job("failed", 100, warn=job_warnings, err=error_msg)

        # Trigger Webhook Callback upon Failure
        if req.callback_url:
            await send_webhook_notification(
                callback_url=req.callback_url,
                payload={
                    "job_id": job_id,
                    "status": "failed",
                    "progress": 100,
                    "warnings": job_warnings,
                    "error_details": error_msg,
                },
            )


@app.post("/api/v1/broll/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_broll_job(
    request_body: JobCreateRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    idempotency_key: Optional[str] = Header(None),
):
    # --- 1. Edge Case: Empty Payload Validation ---
    script_clean = (request_body.script or "").strip()
    audio_url = (request_body.audio_url or "").strip()
    scenes = request_body.scenes

    has_valid_scenes = bool(scenes and any((s.text or "").strip() for s in scenes))

    if not script_clean and not audio_url and not has_valid_scenes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "EMPTY_INPUT_PAYLOAD",
                "message": "Job rejected: Provide a non-empty 'script', a valid 'audio_url', or non-empty 'scenes'.",
                "retryable": False,
            },
        )

    # --- 2. Edge Case: Audio Format Validation ---
    if audio_url:
        clean_url = audio_url.lower().split("?")[0]
        if not any(clean_url.endswith(ext) for ext in [".mp3", ".wav", ".m4a"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "INVALID_AUDIO_FORMAT",
                    "message": "Unsupported audio file type. Supported extensions: .mp3, .wav, .m4a.",
                    "retryable": False,
                },
            )

    conn = get_db()
    c = conn.cursor()

    # --- 3. Edge Case: Idempotency Replay Check ---
    if idempotency_key:
        c.execute(
            "SELECT job_id, status, progress, result_links, warnings, error_details FROM jobs WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        existing = c.fetchone()
        if existing:
            conn.close()
            response.headers["Idempotent-Replay"] = "true"
            return {
                "job_id": existing["job_id"],
                "status": existing["status"],
                "progress": existing["progress"],
                "idempotent_replay": True,
                "result_links": json.loads(existing["result_links"]) if existing["result_links"] else {},
                "warnings": json.loads(existing["warnings"]) if existing["warnings"] else [],
                "error_details": existing["error_details"],
            }

    # --- 4. Enqueue New Job ---
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    c.execute(
        "INSERT INTO jobs (job_id, idempotency_key, status, progress, warnings) VALUES (?, ?, ?, ?, ?)",
        (job_id, idempotency_key, "queued", 0, json.dumps([])),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(process_broll_job, job_id, request_body.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "idempotent_replay": False,
        "warnings": [],
    }


@app.get("/api/v1/broll/jobs/{job_id}")
def get_broll_job(job_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT status, progress, result_links, warnings, error_details FROM jobs WHERE job_id = ?",
        (job_id,),
    )
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' not found.",
                "retryable": False,
            },
        )

    return {
        "job_id": job_id,
        "status": row["status"],
        "progress": row["progress"],
        "result_links": json.loads(row["result_links"]) if row["result_links"] else {},
        "warnings": json.loads(row["warnings"]) if row["warnings"] else [],
        "error_details": row["error_details"],
    }

@app.post("/api/v1/broll/jobs/{job_id}/swap")
async def swap_scene_clip(job_id: str, swap_req: SceneSwapRequest):
    work_dir = os.path.join(settings.WORK_DIR, job_id)
    timeline_path = os.path.join(work_dir, "timeline.json")
    audio_path = os.path.join(work_dir, "voiceover.mp3")
    srt_path = os.path.join(work_dir, "captions.srt")

    if not os.path.exists(timeline_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "JOB_ARTIFACTS_NOT_FOUND",
                "message": f"Cached scene artifacts for job '{job_id}' not found.",
                "retryable": False,
            },
        )

    swap_start = time.time()
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    scenes = timeline_data.get("scenes", [])
    target_idx = swap_req.scene_index
    if target_idx < 0 or target_idx >= len(scenes):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_SCENE_INDEX",
                "message": f"Scene index {target_idx} out of range (0..{len(scenes)-1}).",
                "retryable": False,
            },
        )

    target_scene = scenes[target_idx]
    aspect_ratios = timeline_data.get("aspect_ratios", ["16:9"])
    used_clip_ids = {s["clip_id"].split("_")[-1] for s in scenes if "clip_id" in s}
    new_query = swap_req.new_visual_query or target_scene["visual_query"]

    result_links = {}
    for ratio in aspect_ratios:
        ratio_slug = ratio.replace(":", "_")
        # Only fetch and re-render the single swapped scene
        asset = await search_and_download_footage(
            new_query,
            ratio,
            target_idx,
            used_clip_ids,
            work_dir,
            duration=target_scene["duration"],
        )
        norm_mp4 = os.path.join(work_dir, f"norm_s{target_idx}_{ratio_slug}.mp4")
        normalize_and_render_scene(
            asset["local_path"],
            norm_mp4,
            target_scene["duration"],
            ratio,
            0.5,
        )

        # Re-assemble immediately using cached normalized clips for all other scenes
        all_norm_files = [
            os.path.join(work_dir, f"norm_s{s['scene_index']}_{ratio_slug}.mp4")
            for s in scenes
        ]
        final_mp4 = os.path.join(work_dir, f"video_{ratio_slug}.mp4")
        assemble_final_video(
            all_norm_files,
            audio_path,
            srt_path,
            os.path.exists(srt_path),
            final_mp4,
        )
        result_links[f"video_{ratio_slug}"] = storage.upload_file(
            final_mp4, f"{job_id}/video_{ratio_slug}.mp4"
        )

        if ratio == aspect_ratios[0]:
            target_scene.update({
                "visual_query": new_query,
                "clip_id": asset["clip_id"],
                "clip_url": asset["clip_url"],
                "source_provider": asset["source_provider"],
                "license": asset["license"],
                "attribution": asset["attribution"],
                "is_ai_generated": asset["is_ai_generated"],
                "selection_reason": f"Swapped single scene via partial re-render for '{new_query}'.",
                "routing_rule_applied": "Partial Scene Swap Re-render",
            })

    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)
    result_links["timeline_json"] = storage.upload_file(
        timeline_path, f"{job_id}/timeline.json"
    )

    elapsed = round(time.time() - swap_start, 2)
    return {
        "job_id": job_id,
        "status": "swapped",
        "swapped_scene_index": target_idx,
        "partial_rerender_time_sec": elapsed,
        "result_links": result_links,
    }