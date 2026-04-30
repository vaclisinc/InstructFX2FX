from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import (
    AudioUploadResponse,
    FxMetadataResponse,
    RunCreateRequest,
    RunEnvelope,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    SessionUpdateRequest,
)
from .service import ARTIFACTS_ROOT, available_fx_metadata, artifact_url, service

app = FastAPI(title="text2preset Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_ROOT)), name="artifacts")


def _session_to_response(record) -> SessionResponse:
    return SessionResponse(
        session_id=record.session_id,
        name=record.name,
        available_fx=list(record.session.available_fx),
        audio_uploaded=record.audio_path is not None,
        audio_filename=record.audio_path.name if record.audio_path else None,
        audio_artifact=artifact_url(record.audio_path) if record.audio_path else None,
        history_length=len(record.session.history),
        runs=service.session_runs_payload(record.session_id),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _run_to_response(record) -> RunEnvelope:
    return RunEnvelope(
        run_id=record.run_id,
        session_id=record.session_id,
        status=record.status,
        instruction=record.instruction,
        settings=record.settings,
        progress=record.progress,
        current_iteration=record.current_iteration,
        total_iterations=record.total_iterations,
        result=record.result,
        error=record.error,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/fx-metadata", response_model=FxMetadataResponse)
def fx_metadata() -> FxMetadataResponse:
    return FxMetadataResponse(available_fx=available_fx_metadata())


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    return SessionListResponse(sessions=service.sessions_payload())


@app.post("/sessions", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest) -> SessionResponse:
    record = service.create_session(payload.available_fx, payload.name)
    return _session_to_response(record)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    record = service.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_response(record)


@app.patch("/sessions/{session_id}", response_model=SessionResponse)
def update_session(session_id: str, payload: SessionUpdateRequest) -> SessionResponse:
    record = service.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        updated = service.rename_session(session_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _session_to_response(updated)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    record = service.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    service.delete_session(session_id)
    return {"status": "deleted"}


@app.post("/sessions/{session_id}/audio", response_model=AudioUploadResponse)
async def upload_audio(session_id: str, file: UploadFile = File(...)) -> AudioUploadResponse:
    record = service.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")

    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        updated = service.attach_audio(session_id, tmp_path, file.filename or tmp_path.name)
    finally:
        tmp_path.unlink(missing_ok=True)

    return AudioUploadResponse(
        session_id=updated.session_id,
        filename=file.filename or updated.audio_path.name,
        content_type=file.content_type,
        audio_artifact=artifact_url(updated.audio_path),
    )


@app.post("/sessions/{session_id}/runs", response_model=RunEnvelope)
def create_run(session_id: str, payload: RunCreateRequest) -> RunEnvelope:
    if service.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    record = service.submit_run(session_id, payload.instruction, payload.settings.model_dump())
    return _run_to_response(record)


@app.get("/runs/{run_id}", response_model=RunEnvelope)
def get_run(run_id: str) -> RunEnvelope:
    record = service.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_response(record)
