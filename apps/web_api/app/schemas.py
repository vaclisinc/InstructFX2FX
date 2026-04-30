from typing import Any

from pydantic import BaseModel, Field


DEFAULT_AVAILABLE_FX = ["eq", "comp", "rev", "dist", "delay", "pitchshift", "bitcrush"]


class RunSettings(BaseModel):
    n_iterations: int = Field(default=10, ge=1, le=500)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    snapshot_interval: int | None = Field(default=5, ge=1, le=500)


class SessionCreateRequest(BaseModel):
    available_fx: list[str] = Field(default_factory=lambda: list(DEFAULT_AVAILABLE_FX))


class SessionResponse(BaseModel):
    session_id: str
    available_fx: list[str]
    audio_uploaded: bool
    audio_filename: str | None
    history_length: int


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
    result: dict[str, Any] | None = None
    error: str | None = None


class FxMetadataResponse(BaseModel):
    available_fx: list[dict[str, Any]]
