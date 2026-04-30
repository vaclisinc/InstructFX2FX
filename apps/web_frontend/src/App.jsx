import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const DEFAULT_SETTINGS = {
  n_iterations: 10,
  learning_rate: 0.01,
  snapshot_interval: 5,
};

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed with ${response.status}`);
  }
  return response.json();
}

export default function App() {
  const [session, setSession] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [instruction, setInstruction] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [run, setRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedCheckpointIndex, setSelectedCheckpointIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const created = await fetchJson("/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        if (!cancelled) {
          setSession(created);
        }
      } catch (err) {
        if (!cancelled) {
          setError(String(err));
        }
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!run || run.status === "completed" || run.status === "failed") {
      return undefined;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextRun = await fetchJson(`/runs/${run.run_id}`);
        setRun(nextRun);
        if (nextRun.status === "completed") {
          setBusy(false);
          setSelectedCheckpointIndex(0);
          if (session) {
            const refreshedSession = await fetchJson(`/sessions/${session.session_id}`);
            setSession(refreshedSession);
          }
        }
        if (nextRun.status === "failed") {
          setBusy(false);
          setError(nextRun.error || "Run failed");
        }
      } catch (err) {
        setBusy(false);
        setError(String(err));
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [run, session]);

  const checkpoints = run?.result?.trajectory || [];
  const selectedCheckpoint = checkpoints[selectedCheckpointIndex] || null;
  const sessionHistory = run?.result?.session_snapshot?.history || [];
  const uiMetadata = run?.result?.metadata?.ui || {};
  const routeCase = run?.result?.metadata?.route_case || null;
  const hasCompletedHistory =
    (run?.result?.session_snapshot?.history?.length || 0) > 0 || (session?.history_length || 0) > 0;
  const promptModeLabel = hasCompletedHistory ? "Refinement prompt" : "Initial prompt";
  const canBrowseTrajectory = Boolean(uiMetadata.can_browse_trajectory);
  const initializationOnly = Boolean(uiMetadata.initialization_only);
  const currentAudioSrc = canBrowseTrajectory && selectedCheckpoint?.audio_artifact
    ? `${API_BASE}${selectedCheckpoint.audio_artifact}`
    : run?.result?.artifacts?.final_audio
      ? `${API_BASE}${run.result.artifacts.final_audio}`
      : "";
  const inputAudioSrc = run?.result?.artifacts?.input_audio
    ? `${API_BASE}${run.result.artifacts.input_audio}`
    : "";

  const selectedParams =
    canBrowseTrajectory &&
    selectedCheckpoint?.params &&
    Object.keys(selectedCheckpoint.params).length > 0
      ? selectedCheckpoint.params
      : run?.result?.params || {};

  async function uploadSelectedAudio() {
    if (!session || !audioFile) {
      throw new Error("Choose an audio file before running.");
    }

    const formData = new FormData();
    formData.append("file", audioFile);
    await fetchJson(`/sessions/${session.session_id}/audio`, {
      method: "POST",
      body: formData,
    });
    const refreshedSession = await fetchJson(`/sessions/${session.session_id}`);
    setSession(refreshedSession);
  }

  async function handleRun(event) {
    event.preventDefault();
    setError("");

    if (!instruction.trim()) {
      setError("Enter a prompt before running.");
      return;
    }

    try {
      setBusy(true);
      if (audioFile) {
        await uploadSelectedAudio();
      } else if (!session?.audio_uploaded) {
        throw new Error("Upload a WAV file before running.");
      }

      const createdRun = await fetchJson(`/sessions/${session.session_id}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction,
          settings: {
            n_iterations: Number(settings.n_iterations),
            learning_rate: Number(settings.learning_rate),
            snapshot_interval: Number(settings.snapshot_interval),
          },
        }),
      });
      setRun(createdRun);
    } catch (err) {
      setBusy(false);
      setError(String(err));
    }
  }

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function renderRunModeMessage() {
    if (!run?.result) {
      return (
        <p className="muted">
          Initial prompts create the first FX chain. Later prompts refine the existing session.
        </p>
      );
    }

    if (initializationOnly) {
      return (
        <div className="info-box">
          This run was an initial LLM initialization for newly selected FX. No optimization trajectory exists, so slider browsing is disabled.
        </div>
      );
    }

    if (routeCase === "reuse_all") {
      return (
        <div className="info-box">
          This run refined an existing chain. Optimization checkpoints are available only when the backend path exposes them.
        </div>
      );
    }

    if (routeCase === "reuse_and_initialize") {
      return (
        <div className="info-box">
          This run mixed refinement with new FX initialization. Existing FX were optimized; newly added FX were initialized and merged into the result.
        </div>
      );
    }

    return <p className="muted">{uiMetadata.trajectory_reason || "Run metadata unavailable."}</p>;
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">Phase 2 Demo</p>
        <h1>text2preset Web Client</h1>
        <p className="lede">
          Thin frontend over the current orchestration pipeline. Sessions, uploads, prompt runs, and checkpoint browsing stay linked to the same engine repo.
        </p>
      </header>

      <main className="layout">
        <section className="panel">
          <h2>Run</h2>
          <div className="mode-pill">{promptModeLabel}</div>
          <form onSubmit={handleRun} className="stack">
            <label className="field">
              <span>Dry audio</span>
              <input
                type="file"
                accept=".wav,audio/wav"
                onChange={(event) => setAudioFile(event.target.files?.[0] || null)}
              />
            </label>

            <label className="field">
              <span>Prompt</span>
              <textarea
                value={instruction}
                onChange={(event) => setInstruction(event.target.value)}
                rows={4}
                placeholder="bright but soft, like a warm room"
              />
            </label>

            <div className="settings-grid">
              <label className="field">
                <span>Iterations</span>
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={settings.n_iterations}
                  onChange={(event) => updateSetting("n_iterations", event.target.value)}
                />
              </label>

              <label className="field">
                <span>Learning rate</span>
                <input
                  type="number"
                  min="0.0001"
                  max="1"
                  step="0.0001"
                  value={settings.learning_rate}
                  onChange={(event) => updateSetting("learning_rate", event.target.value)}
                />
              </label>

              <label className="field">
                <span>Snapshot interval</span>
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={settings.snapshot_interval}
                  onChange={(event) => updateSetting("snapshot_interval", event.target.value)}
                />
              </label>
            </div>

            <button type="submit" disabled={busy || !session}>
              {busy ? "Running..." : "Run prompt"}
            </button>
          </form>

            <div className="status-card">
              <strong>Session</strong>
              <div>{session?.session_id || "Bootstrapping..."}</div>
              <div>Audio uploaded: {session?.audio_uploaded ? "yes" : "no"}</div>
              <div>Run status: {run?.status || "idle"}</div>
            </div>

            {renderRunModeMessage()}

          {error ? <div className="error-box">{error}</div> : null}
        </section>

        <section className="panel">
          <h2>Preview</h2>
          <div className="audio-grid">
            <div>
              <h3>Input</h3>
              <audio controls src={inputAudioSrc} />
            </div>
            <div>
              <h3>{canBrowseTrajectory && selectedCheckpoint ? `Checkpoint: ${selectedCheckpoint.label}` : "Final"}</h3>
              <audio controls src={currentAudioSrc} />
            </div>
          </div>

          {canBrowseTrajectory && checkpoints.length > 0 ? (
            <div className="slider-block">
              <label htmlFor="trajectory-slider">Trajectory checkpoint</label>
              <input
                id="trajectory-slider"
                type="range"
                min="0"
                max={Math.max(checkpoints.length - 1, 0)}
                step="1"
                value={selectedCheckpointIndex}
                onChange={(event) => setSelectedCheckpointIndex(Number(event.target.value))}
              />
              <div className="checkpoint-meta">
                <span>{selectedCheckpoint?.label}</span>
                <span>
                  {selectedCheckpoint?.iteration === null || selectedCheckpoint?.iteration === undefined
                    ? "final"
                    : `iter ${selectedCheckpoint.iteration}`}
                </span>
              </div>
              <p className="muted">
                Each slider stop is one saved checkpoint. The label shows the actual saved iteration, not every integer step.
              </p>
            </div>
          ) : run?.result ? (
            <div className="info-box subtle">
              {uiMetadata.trajectory_reason || "No trajectory slider is available for this run."}
            </div>
          ) : (
            <p className="muted">No checkpoints yet.</p>
          )}

          <div className="stack compact">
            <h3>FX chain</h3>
            <div className="chips">
              {(run?.result?.fx_chain || []).map((fx) => (
                <span key={fx} className="chip">{fx}</span>
              ))}
            </div>
          </div>
        </section>

        <section className="panel">
          <h2>Parameters</h2>
          <pre>{JSON.stringify(selectedParams, null, 2)}</pre>
        </section>

        <section className="panel wide">
          <h2>Session history</h2>
          {sessionHistory.length === 0 ? (
            <p className="muted">No runs completed yet.</p>
          ) : (
            <div className="history-list">
              {sessionHistory.map((item, index) => (
                <article key={`${item.prompt}-${index}`} className="history-item">
                  <strong>{item.prompt}</strong>
                  <div>{item.fx_chain.join(" -> ")}</div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
