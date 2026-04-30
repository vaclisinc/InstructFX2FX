from typing import Any

from pydantic import BaseModel, Field


DEFAULT_AVAILABLE_FX = ["eq", "comp", "rev", "dist", "delay", "pitchshift", "bitcrush"]


class RunSettings(BaseModel):
    n_iterations: int = Field(default=10, ge=1, le=500)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    snapshot_interval: int | None = Field(default=5, ge=1, le=500)
    llm_model: str | None = None


class SessionCreateRequest(BaseModel):
    available_fx: list[str] = Field(default_factory=lambda: list(DEFAULT_AVAILABLE_FX))
    name: str | None = None


class SessionUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SessionResponse(BaseModel):
    session_id: str
    name: str
    available_fx: list[str]
    audio_uploaded: bool
    audio_filename: str | None
    audio_artifact: str | None = None
    history_length: int
    runs: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[dict[str, Any]] = Field(default_factory=list)


class AudioUploadResponse(BaseModel):
    session_id: str
    filename: str
    content_type: str | None
    audio_artifact: str


class RunCreateRequest(BaseModel):
    instruction: str = Field(min_length=1)
    settings: RunSettings = Field(default_factory=RunSettings)


class RunEnvelope(BaseModel):
    run_id: str
    session_id: str
    status: str
    instruction: str
    settings: dict[str, Any]
    progress: float = 0.0
    current_iteration: int = 0
    total_iterations: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class FxMetadataResponse(BaseModel):
    available_fx: list[dict[str, Any]]
