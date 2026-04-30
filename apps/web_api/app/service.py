from __future__ import annotations

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
from session.session import Session

from .schemas import DEFAULT_AVAILABLE_FX

ARTIFACTS_ROOT = PROJECT_ROOT / "apps" / "web_api" / "artifacts"
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

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
    result: dict[str, Any] | None = None
    error: str | None = None


class WebDemoService:
    def __init__(self) -> None:
        self.runtime = EngineRuntime()
        self.sessions: dict[str, SessionRecord] = {}
        self.runs: dict[str, RunRecord] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_session(self, available_fx: list[str] | None = None) -> SessionRecord:
        available_fx = available_fx or list(DEFAULT_AVAILABLE_FX)
        session_id = uuid.uuid4().hex
        now = utc_now_iso()
        record = SessionRecord(
            session_id=session_id,
            session=Session(available_fx=available_fx),
            audio_path=None,
            created_at=now,
            updated_at=now,
        )
        with self.lock:
            self.sessions[session_id] = record
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.lock:
            return self.sessions.get(session_id)

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
        )
        with self.lock:
            self.runs[run_id] = record
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
            raw_result = orchestrator.run_with_metadata(
                run.instruction,
                session_record.session,
                audio,
                run_settings=run.settings,
            )

            run_dir = ARTIFACTS_ROOT / "sessions" / session_record.session_id / "runs" / run.run_id
            payload = self._materialize_run_payload(session_record, run, raw_result, run_dir)
            with self.lock:
                run.result = payload
                run.status = "completed"
                run.updated_at = utc_now_iso()
                session_record.updated_at = utc_now_iso()
        except Exception as exc:
            with self.lock:
                run.error = str(exc)
                run.status = "failed"
                run.updated_at = utc_now_iso()

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
                "Optimization ran, but this backend path does not currently expose intermediate checkpoints for slider browsing."
            )
        elif can_browse_trajectory:
            trajectory_reason = "Trajectory slider is available for this optimized run."
        else:
            trajectory_reason = "No trajectory is available for this run."

        return {
            "fx_chain": raw_result["fx_chain"],
            "params": raw_result["params"],
            "artifacts": {
                "input_audio": artifact_url(session_record.audio_path),
                "final_audio": artifact_url(final_audio_path),
            },
            "trajectory": trajectory,
            "settings_used": metadata.get("settings_used", run.settings),
            "metadata": {
                "route_case": route_case,
                "stages": stage_summaries,
                "ui": {
                    "initialization_only": initialization_only,
                    "optimization_performed": optimization_performed,
                    "can_browse_trajectory": can_browse_trajectory,
                    "trajectory_reason": trajectory_reason,
                },
            },
            "session_snapshot": {
                "available_fx": list(session_record.session.available_fx),
                "current_params": dict(session_record.session.current_params),
                "history": session_history_payload(session_record.session),
            },
        }


service = WebDemoService()
