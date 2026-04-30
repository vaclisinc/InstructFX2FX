from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from effects.fx import FXChainFactory, FXFamily
from embeddings.clap import CLAPWrapper
from llms.llmclient import LLMClient
from pipeline.orchestrator import Orchestrator
from session.session import PromptRecord, Session

from .schemas import DEFAULT_AVAILABLE_FX

ARTIFACTS_ROOT = PROJECT_ROOT / "apps" / "web_api" / "artifacts"
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
SESSIONS_ROOT = ARTIFACTS_ROOT / "sessions"
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

PB_CANONICAL_TO_EFFECT_NAME = {
    "comp": "compressor",
    "dist": "distortion",
    "delay": "delay",
    "pitchshift": "pitchshift",
    "bitcrush": "bitcrush",
}

PB_CANONICAL_TO_DICT_KEY = {
    "comp": "Compressor",
    "dist": "Distortion",
    "delay": "Delay",
    "pitchshift": "PitchShift",
    "bitcrush": "Bitcrush",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_url(path: Path) -> str:
    relative = path.resolve().relative_to(ARTIFACTS_ROOT.resolve())
    return f"/artifacts/{relative.as_posix()}"


def load_audio_tensor(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
    return waveform.unsqueeze(0)


def save_audio_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_np = tensor.detach().cpu().squeeze(0).numpy()
    sf.write(path, audio_np.T if audio_np.ndim == 2 else audio_np, 44100)


def session_history_payload(session: Session) -> list[dict[str, Any]]:
    return [
        {
            "prompt": record.prompt,
            "fx_chain": list(record.fx_chain),
            "params": dict(record.params),
        }
        for record in session.history
    ]


def available_fx_metadata() -> list[dict[str, Any]]:
    return [
        {"name": "eq", "backend": "DASP", "optimizer": "gradient_descent"},
        {"name": "rev", "backend": "DASP", "optimizer": "gradient_descent"},
        {"name": "comp", "backend": "Pedalboard", "optimizer": "bayesian_optimization"},
        {"name": "dist", "backend": "Pedalboard", "optimizer": "bayesian_optimization"},
        {"name": "delay", "backend": "Pedalboard", "optimizer": "bayesian_optimization"},
        {"name": "pitchshift", "backend": "Pedalboard", "optimizer": "bayesian_optimization"},
        {"name": "bitcrush", "backend": "Pedalboard", "optimizer": "bayesian_optimization"},
    ]


class EngineRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm: LLMClient | None = None
        self._clap: CLAPWrapper | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_components(self) -> tuple[LLMClient, CLAPWrapper, str]:
        with self._lock:
            load_dotenv(PROJECT_ROOT / ".env")
            if not os.getenv("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY is not configured in .env")

            if self._llm is None:
                self._llm = LLMClient()
            if self._clap is None:
                self._clap = CLAPWrapper(device=self._device)
            return self._llm, self._clap, self._device


@dataclass
class SessionRecord:
    session_id: str
    name: str
    session: Session
    audio_path: Path | None
    created_at: str
    updated_at: str


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    instruction: str
    settings: dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    progress: float = 0.0
    current_iteration: int = 0
    total_iterations: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class WebDemoService:
    def __init__(self) -> None:
        self.runtime = EngineRuntime()
        self.sessions: dict[str, SessionRecord] = {}
        self.runs: dict[str, RunRecord] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_persisted_state()

    @staticmethod
    def _session_dir(session_id: str) -> Path:
        return SESSIONS_ROOT / session_id

    @staticmethod
    def _session_summary_path(session_id: str) -> Path:
        return WebDemoService._session_dir(session_id) / "session.json"

    @staticmethod
    def _run_dir(session_id: str, run_id: str) -> Path:
        return WebDemoService._session_dir(session_id) / "runs" / run_id

    @staticmethod
    def _run_summary_path(session_id: str, run_id: str) -> Path:
        return WebDemoService._run_dir(session_id, run_id) / "run.json"

    def _persist_session(self, session_record: SessionRecord) -> None:
        session_dir = self._session_dir(session_record.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_record.session_id,
            "name": session_record.name,
            "available_fx": list(session_record.session.available_fx),
            "current_params": dict(session_record.session.current_params),
            "history": session_history_payload(session_record.session),
            "audio_path": str(session_record.audio_path.relative_to(ARTIFACTS_ROOT)) if session_record.audio_path else None,
            "created_at": session_record.created_at,
            "updated_at": session_record.updated_at,
        }
        with self._session_summary_path(session_record.session_id).open("w") as handle:
            json.dump(payload, handle, indent=2)

    def _persist_run(self, run: RunRecord) -> None:
        run_dir = self._run_dir(run.session_id, run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run.run_id,
            "session_id": run.session_id,
            "instruction": run.instruction,
            "settings": run.settings,
            "status": run.status,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "progress": run.progress,
            "current_iteration": run.current_iteration,
            "total_iterations": run.total_iterations,
            "error": run.error,
        }
        with self._run_summary_path(run.session_id, run.run_id).open("w") as handle:
            json.dump(payload, handle, indent=2)

    def _load_persisted_state(self) -> None:
        if not SESSIONS_ROOT.exists():
            return

        for session_dir in sorted(SESSIONS_ROOT.iterdir()):
            if not session_dir.is_dir():
                continue

            session_path = session_dir / "session.json"
            if not session_path.exists():
                continue

            with session_path.open() as handle:
                payload = json.load(handle)

            session = Session(available_fx=payload.get("available_fx", list(DEFAULT_AVAILABLE_FX)))
            session.current_params = payload.get("current_params", {})
            session.history = [
                PromptRecord(
                    prompt=item.get("prompt", ""),
                    fx_chain=list(item.get("fx_chain", [])),
                    params=dict(item.get("params", {})),
                )
                for item in payload.get("history", [])
            ]
            audio_path = payload.get("audio_path")
            record = SessionRecord(
                session_id=payload["session_id"],
                name=payload.get("name") or (
                    payload.get("history", [{}])[-1].get("prompt")
                    if payload.get("history")
                    else "Untitled session"
                ),
                session=session,
                audio_path=(ARTIFACTS_ROOT / audio_path) if audio_path else None,
                created_at=payload.get("created_at", utc_now_iso()),
                updated_at=payload.get("updated_at", utc_now_iso()),
            )
            self.sessions[record.session_id] = record

            runs_dir = session_dir / "runs"
            if not runs_dir.exists():
                continue

            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_summary_path = run_dir / "run.json"
                if not run_summary_path.exists():
                    continue
                with run_summary_path.open() as handle:
                    run_payload = json.load(handle)

                result_path = run_dir / "result.json"
                result_payload = None
                if result_path.exists():
                    with result_path.open() as handle:
                        result_payload = json.load(handle)

                run_record = RunRecord(
                    run_id=run_payload["run_id"],
                    session_id=run_payload["session_id"],
                    instruction=run_payload["instruction"],
                    settings=run_payload.get("settings", {}),
                    status=run_payload.get("status", "completed"),
                    created_at=run_payload.get("created_at", utc_now_iso()),
                    updated_at=run_payload.get("updated_at", utc_now_iso()),
                    progress=run_payload.get("progress", 0.0),
                    current_iteration=run_payload.get("current_iteration", 0),
                    total_iterations=run_payload.get("total_iterations"),
                    result=result_payload,
                    error=run_payload.get("error"),
                )
                self.runs[run_record.run_id] = run_record

    @staticmethod
    def _ui_state_for(route_case: str | None, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        has_intermediate_checkpoints = any(
            item.get("iteration") not in (None, 0)
            for item in trajectory
        )
        initialization_only = route_case == "initialize_all"
        optimization_performed = route_case in {"reuse_all", "reuse_and_initialize"}
        can_browse_trajectory = optimization_performed and has_intermediate_checkpoints

        if initialization_only:
            trajectory_reason = (
                "This run only used LLM initialization for newly added FX, so no optimization trajectory is available."
            )
        elif optimization_performed and not can_browse_trajectory:
            trajectory_reason = (
                "Optimization is running or this backend path does not currently expose intermediate checkpoints for slider browsing."
            )
        elif can_browse_trajectory:
            trajectory_reason = "Trajectory slider is available for this optimized run."
        else:
            trajectory_reason = "No trajectory is available for this run."

        return {
            "initialization_only": initialization_only,
            "optimization_performed": optimization_performed,
            "can_browse_trajectory": can_browse_trajectory,
            "trajectory_reason": trajectory_reason,
        }

    def _build_session_snapshot(self, session_record: SessionRecord) -> dict[str, Any]:
        return {
            "available_fx": list(session_record.session.available_fx),
            "current_params": dict(session_record.session.current_params),
            "history": session_history_payload(session_record.session),
        }

    def _previous_run(self, session_id: str, run_id: str) -> RunRecord | None:
        with self.lock:
            runs = [item for item in self.runs.values() if item.session_id == session_id]
        runs.sort(key=lambda item: item.created_at)
        for index, item in enumerate(runs):
            if item.run_id == run_id:
                return runs[index - 1] if index > 0 else None
        return None

    def _comparison_input_audio(self, session_record: SessionRecord, run: RunRecord) -> str | None:
        previous_run = self._previous_run(session_record.session_id, run.run_id)
        if previous_run and previous_run.result:
            previous_audio = previous_run.result.get("artifacts", {}).get("final_audio")
            if previous_audio:
                return previous_audio
        return artifact_url(session_record.audio_path) if session_record.audio_path else None

    def session_runs_payload(self, session_id: str) -> list[dict[str, Any]]:
        with self.lock:
            runs = [run for run in self.runs.values() if run.session_id == session_id]

        runs.sort(key=lambda item: item.created_at)
        payload = []
        for run in runs:
            result = run.result or {}
            metadata = result.get("metadata", {})
            ui = metadata.get("ui", {})
            artifacts = result.get("artifacts", {})
            payload.append(
                {
                    "run_id": run.run_id,
                    "instruction": run.instruction,
                    "status": run.status,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "progress": run.progress,
                    "current_iteration": run.current_iteration,
                    "total_iterations": run.total_iterations,
                    "route_case": metadata.get("route_case"),
                    "llm_model": run.settings.get("llm_model"),
                    "fx_chain": result.get("fx_chain", []),
                    "final_audio": artifacts.get("final_audio"),
                    "can_browse_trajectory": ui.get("can_browse_trajectory", False),
                }
            )
        return payload

    def sessions_payload(self) -> list[dict[str, Any]]:
        with self.lock:
            session_records = list(self.sessions.values())

        session_records.sort(key=lambda item: item.updated_at, reverse=True)
        return [
            {
                "session_id": record.session_id,
                "name": record.name,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "audio_uploaded": record.audio_path is not None,
                "history_length": len(record.session.history),
                "latest_prompt": record.session.history[-1].prompt if record.session.history else None,
            }
            for record in session_records
        ]

    def _initial_run_payload(self, session_record: SessionRecord, run: RunRecord) -> dict[str, Any]:
        trajectory: list[dict[str, Any]] = []
        return {
            "fx_chain": [],
            "params": {},
            "artifacts": {
                "dry_audio": artifact_url(session_record.audio_path) if session_record.audio_path else None,
                "input_audio": self._comparison_input_audio(session_record, run),
                "final_audio": None,
            },
            "trajectory": trajectory,
            "settings_used": dict(run.settings),
            "metadata": {
                "route_case": None,
                "stages": {},
                "ui": self._ui_state_for(None, trajectory),
                "progress": {
                    "current_iteration": 0,
                    "total_iterations": run.total_iterations,
                    "percent": 0.0,
                    "status": run.status,
                },
            },
            "session_snapshot": self._build_session_snapshot(session_record),
        }

    def _update_progress_payload(self, run: RunRecord) -> None:
        if run.result is None:
            return
        progress_meta = run.result.setdefault("metadata", {}).setdefault("progress", {})
        progress_meta.update(
            {
                "current_iteration": run.current_iteration,
                "total_iterations": run.total_iterations,
                "percent": run.progress,
                "status": run.status,
            }
        )

    def _upsert_checkpoint(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: dict[str, Any],
    ) -> list[dict[str, Any]]:
        filtered = [item for item in trajectory if item.get("label") != checkpoint.get("label")]
        filtered.append(checkpoint)
        filtered.sort(
            key=lambda item: (
                10**9 if item.get("iteration") is None else item.get("iteration", 0),
                item.get("label", ""),
            )
        )
        return filtered

    def create_session(self, available_fx: list[str] | None = None, name: str | None = None) -> SessionRecord:
        available_fx = available_fx or list(DEFAULT_AVAILABLE_FX)
        session_id = uuid.uuid4().hex
        now = utc_now_iso()
        record = SessionRecord(
            session_id=session_id,
            name=(name or "Untitled session").strip() or "Untitled session",
            session=Session(available_fx=available_fx),
            audio_path=None,
            created_at=now,
            updated_at=now,
        )
        with self.lock:
            self.sessions[session_id] = record
            self._persist_session(record)
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.lock:
            return self.sessions.get(session_id)

    def rename_session(self, session_id: str, name: str) -> SessionRecord:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Session name cannot be empty")
        with self.lock:
            record = self.sessions[session_id]
            record.name = cleaned_name
            record.updated_at = utc_now_iso()
            self._persist_session(record)
            return record

    def delete_session(self, session_id: str) -> None:
        with self.lock:
            record = self.sessions.pop(session_id, None)
            if record is None:
                raise KeyError(session_id)

            run_ids = [run_id for run_id, run in self.runs.items() if run.session_id == session_id]
            for run_id in run_ids:
                self.runs.pop(run_id, None)

        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)

    def attach_audio(self, session_id: str, source_path: Path, filename: str) -> SessionRecord:
        with self.lock:
            record = self.sessions[session_id]
        ext = source_path.suffix or ".wav"
        session_audio_dir = ARTIFACTS_ROOT / "sessions" / session_id / "audio"
        session_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = session_audio_dir / f"input{ext}"
        shutil.copy2(source_path, audio_path)
        record.audio_path = audio_path
        record.updated_at = utc_now_iso()
        with self.lock:
            self._persist_session(record)
        return record

    def submit_run(self, session_id: str, instruction: str, settings: dict[str, Any]) -> RunRecord:
        run_id = uuid.uuid4().hex
        now = utc_now_iso()
        record = RunRecord(
            run_id=run_id,
            session_id=session_id,
            instruction=instruction,
            settings=settings,
            status="queued",
            created_at=now,
            updated_at=now,
            total_iterations=int(settings.get("n_iterations", 0)) or None,
        )
        with self.lock:
            self.runs[run_id] = record
            self._persist_run(record)
        self.executor.submit(self._execute_run, run_id)
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.lock:
            return self.runs.get(run_id)

    def _execute_run(self, run_id: str) -> None:
        with self.lock:
            run = self.runs[run_id]
            session_record = self.sessions[run.session_id]
            run.status = "running"
            run.updated_at = utc_now_iso()
            run.result = self._initial_run_payload(session_record, run)
            self._update_progress_payload(run)
            self._persist_run(run)

        try:
            if session_record.audio_path is None:
                raise RuntimeError("No audio has been uploaded for this session")

            llm, clap, device = self.runtime.get_components()
            orchestrator = Orchestrator(
                llm,
                clap,
                device=device,
                n_iterations=int(run.settings.get("n_iterations", 10)),
                lr=float(run.settings.get("learning_rate", 0.01)),
            )
            audio = load_audio_tensor(session_record.audio_path)
            run_dir = ARTIFACTS_ROOT / "sessions" / session_record.session_id / "runs" / run.run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            def progress_callback(event: dict[str, Any]) -> None:
                self._handle_progress_event(run_id, run_dir, event)

            raw_result = orchestrator.run_with_metadata(
                run.instruction,
                session_record.session,
                audio,
                run_settings={**run.settings, "progress_callback": progress_callback},
            )

            payload = self._materialize_run_payload(session_record, run, raw_result, run_dir)
            with self.lock:
                run.result = payload
                run.status = "completed"
                run.progress = 100.0
                run.current_iteration = run.total_iterations or run.current_iteration
                run.updated_at = utc_now_iso()
                session_record.updated_at = utc_now_iso()
                self._update_progress_payload(run)
                self._persist_run(run)
                self._persist_session(session_record)
        except Exception as exc:
            with self.lock:
                run.error = str(exc)
                run.status = "failed"
                run.updated_at = utc_now_iso()
                self._update_progress_payload(run)
                self._persist_run(run)

    def _handle_progress_event(self, run_id: str, run_dir: Path, event: dict[str, Any]) -> None:
        with self.lock:
            run = self.runs[run_id]
            session_record = self.sessions[run.session_id]
            if run.result is None:
                run.result = self._initial_run_payload(session_record, run)

            payload = run.result
            metadata = payload.setdefault("metadata", {})

            if event.get("type") == "route":
                route_case = event.get("route_case")
                payload["fx_chain"] = list(event.get("fx_chain", []))
                metadata["route_case"] = route_case
                metadata["ui"] = self._ui_state_for(route_case, payload.get("trajectory", []))

            elif event.get("type") == "progress":
                iteration = int(event.get("iteration", 0))
                total_iterations = int(event.get("total_iterations", run.total_iterations or 0)) or None
                run.current_iteration = iteration
                run.total_iterations = total_iterations
                run.progress = (
                    min(100.0, (iteration / total_iterations) * 100.0)
                    if total_iterations
                    else run.progress
                )

            elif event.get("type") == "checkpoint":
                label = str(event.get("label"))
                checkpoint_audio = event.get("audio")
                checkpoint_path = run_dir / "trajectory" / f"{label}.wav"
                save_audio_tensor(checkpoint_audio, checkpoint_path)
                checkpoint = {
                    "label": label,
                    "iteration": event.get("iteration"),
                    "audio_artifact": artifact_url(checkpoint_path),
                    "params": event.get("params", {}),
                    "loss": event.get("loss"),
                }
                payload["trajectory"] = self._upsert_checkpoint(payload.get("trajectory", []), checkpoint)
                route_case = metadata.get("route_case")
                metadata["ui"] = self._ui_state_for(route_case, payload["trajectory"])

                iteration = event.get("iteration")
                total_iterations = event.get("total_iterations")
                if iteration is not None and total_iterations:
                    run.current_iteration = int(iteration)
                    run.total_iterations = int(total_iterations)
                    run.progress = min(100.0, (run.current_iteration / run.total_iterations) * 100.0)

            payload["session_snapshot"] = self._build_session_snapshot(session_record)
            run.updated_at = utc_now_iso()
            self._update_progress_payload(run)
            self._persist_run(run)

    def _apply_pedalboard_chain(
        self,
        audio_tensor: torch.Tensor,
        final_params: dict[str, Any],
        pb_fxs: list[str],
    ) -> torch.Tensor:
        if not pb_fxs:
            return audio_tensor

        pb_effects = [PB_CANONICAL_TO_EFFECT_NAME[fx] for fx in pb_fxs if fx in PB_CANONICAL_TO_EFFECT_NAME]
        pb_chain = FXChainFactory.create_fx_chain_from_effects(
            pb_effects,
            backend=FXFamily.PEDALBOARD,
        )
        pb_config = {
            PB_CANONICAL_TO_DICT_KEY[fx]: final_params[fx]
            for fx in pb_fxs
            if fx in final_params and fx in PB_CANONICAL_TO_DICT_KEY
        }
        rendered = pb_chain(audio_tensor, pb_config)
        if isinstance(rendered, torch.Tensor):
            return rendered.detach().cpu()
        return torch.from_numpy(rendered).float()

    def _build_trajectory(
        self,
        session_record: SessionRecord,
        raw_result: dict[str, Any],
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        trajectory_dir = run_dir / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        final_params = raw_result["params"]
        final_audio = raw_result["audio"]
        metadata = raw_result.get("metadata", {})
        dasp_meta = metadata.get("stages", {}).get("dasp", {})
        dasp_trajectory = dasp_meta.get("trajectory", [])
        pb_fxs = [fx for fx in raw_result["fx_chain"] if fx in PB_CANONICAL_TO_EFFECT_NAME]

        checkpoints: list[dict[str, Any]] = []
        if dasp_trajectory:
            for item in dasp_trajectory:
                label = item["label"]
                iteration = item["iteration"]
                audio_tensor = item["audio"]
                merged_params = dict(final_params)
                merged_params.update(item.get("params", {}))

                if label == "end":
                    current_audio = final_audio
                    merged_params = final_params
                else:
                    current_audio = self._apply_pedalboard_chain(audio_tensor, final_params, pb_fxs)

                checkpoint_path = trajectory_dir / f"{label}.wav"
                save_audio_tensor(current_audio, checkpoint_path)
                checkpoints.append(
                    {
                        "label": label,
                        "iteration": iteration,
                        "audio_artifact": artifact_url(checkpoint_path),
                        "params": merged_params,
                    }
                )
        else:
            checkpoints.append(
                {
                    "label": "start",
                    "iteration": 0,
                    "audio_artifact": artifact_url(session_record.audio_path),
                    "params": {},
                }
            )
            end_path = trajectory_dir / "end.wav"
            save_audio_tensor(final_audio, end_path)
            checkpoints.append(
                {
                    "label": "end",
                    "iteration": None,
                    "audio_artifact": artifact_url(end_path),
                    "params": final_params,
                }
            )

        return checkpoints

    def _materialize_run_payload(
        self,
        session_record: SessionRecord,
        run: RunRecord,
        raw_result: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        final_audio_path = run_dir / "final.wav"
        save_audio_tensor(raw_result["audio"], final_audio_path)

        trajectory = self._build_trajectory(session_record, raw_result, run_dir)
        metadata = raw_result.get("metadata", {})
        route_case = metadata.get("route_case")
        stage_summaries = {}
        for stage_name, stage_data in metadata.get("stages", {}).items():
            stage_summaries[stage_name] = {
                key: value
                for key, value in stage_data.items()
                if key not in {"trajectory"}
            }
            if "trajectory" in stage_data:
                stage_summaries[stage_name]["trajectory_count"] = len(stage_data["trajectory"])
        payload = {
            "fx_chain": raw_result["fx_chain"],
            "params": raw_result["params"],
            "artifacts": {
                "dry_audio": artifact_url(session_record.audio_path),
                "input_audio": self._comparison_input_audio(session_record, run),
                "final_audio": artifact_url(final_audio_path),
            },
            "trajectory": trajectory,
            "settings_used": metadata.get("settings_used", run.settings),
            "metadata": {
                "route_case": route_case,
                "stages": stage_summaries,
                "ui": self._ui_state_for(route_case, trajectory),
                "progress": {
                    "current_iteration": run.current_iteration,
                    "total_iterations": run.total_iterations,
                    "percent": run.progress,
                    "status": run.status,
                },
            },
            "session_snapshot": self._build_session_snapshot(session_record),
        }
        result_json_path = run_dir / "result.json"
        with result_json_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        self._persist_run(run)
        return payload


service = WebDemoService()
