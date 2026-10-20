from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SceneInput(BaseModel):
    text: str
    visual_hint: Optional[str] = None


class CaptionsConfig(BaseModel):
    burn_in: bool = True
    srt: bool = True


class JobCreateRequest(BaseModel):
    audio_url: Optional[str] = None
    script: Optional[str] = None
    scenes: Optional[List[SceneInput]] = None
    voice: str = "en-IN-female"
    aspect_ratios: List[str] = Field(default_factory=lambda: ["16:9", "9:16"])
    style: str = "bright, modern, real-world footage"
    avoid: List[str] = Field(default_factory=list)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    callback_url: Optional[str] = Field(
        default=None,
        description="Optional client webhook/callback URL to notify when processing finishes."
    )


class SceneSwapRequest(BaseModel):
    visual_query: Optional[str] = None
    replacement_clip_url: Optional[str] = None


class TimelineScene(BaseModel):
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    narration_text: str
    visual_query: str
    clip_id: str
    clip_url: str
    source_provider: str
    license: str
    attribution: str
    commercial_use_allowed: bool = True
    is_ai_generated: bool = False
    selection_reason: str
    routing_rule_applied: str
    relevance_score: float
    independent_vlm_score: Optional[float] = None
    crop_x_center: float = 0.5


class TimelineOutput(BaseModel):
    job_id: str
    total_duration_sec: float
    aspect_ratios: List[str]
    scenes: List[TimelineScene]


class ManifestOutput(BaseModel):
    job_id: str
    models_and_providers: Dict[str, str]
    settings_used: Dict[str, Any]
    seed: int = 42
    timings_sec: Dict[str, float]
    estimated_cost_usd: float
    warnings: List[str]
    rights_and_licences_summary: str
    
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    warnings: List[str] = []
    video_urls: Dict[str, str] = {}
    manifest_url: Optional[str] = None
    timeline_url: Optional[str] = None
    error: Optional[Dict[str, Any]] = None