import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const DEFAULT_SETTINGS = {
  n_iterations: 10,
  learning_rate: 0.01,
  snapshot_interval: 5,
  llm_model: "",
};

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed with ${response.status}`);
  }
  return response.json();
}

function formatPercent(value) {
  return `${Math.round(value || 0)}%`;
}

export default function App() {
  const [sessionCatalog, setSessionCatalog] = useState([]);
  const [session, setSession] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [instruction, setInstruction] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [activeRun, setActiveRun] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedCheckpointIndex, setSelectedCheckpointIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const listed = await fetchJson("/sessions");
        const sessions = listed.sessions || [];
        let targetSession;
        let nextCatalog = sessions;
        if (sessions.length > 0) {
          targetSession = await fetchJson(`/sessions/${sessions[0].session_id}`);
        } else {
          targetSession = await fetchJson("/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          const refreshedList = await fetchJson("/sessions");
          nextCatalog = refreshedList.sessions || [];
        }
        if (!cancelled) {
          setSessionCatalog(nextCatalog);
          setSession(targetSession);
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

  const runs = session?.runs || [];
  const liveRunSummary = runs.length > 0 ? runs[runs.length - 1] : null;
  const liveRunId = liveRunSummary?.run_id || null;
  const viewingHistory = Boolean(selectedRunId && liveRunId && selectedRunId !== liveRunId);
  const isInitialPrompt = !viewingHistory && (session?.history_length || 0) === 0;
  const promptModeLabel = (session?.history_length || 0) > 0 ? "Refinement prompt" : "Initial prompt";

  useEffect(() => {
    if (!selectedRunId && liveRunId) {
      setSelectedRunId(liveRunId);
    }
  }, [selectedRunId, liveRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      return undefined;
    }

    let cancelled = false;
    async function loadRun() {
      try {
        const fetched = await fetchJson(`/runs/${selectedRunId}`);
        if (!cancelled) {
          setActiveRun(fetched);
        }
      } catch (err) {
        if (!cancelled) {
          setError(String(err));
        }
      }
    }

    loadRun();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!activeRun || activeRun.status === "completed" || activeRun.status === "failed") {
      return undefined;
    }

    const timer = window.setInterval(async () => {
      try {
        const [nextRun, refreshedSession, listed] = await Promise.all([
          fetchJson(`/runs/${activeRun.run_id}`),
          fetchJson(`/sessions/${activeRun.session_id}`),
          fetchJson("/sessions"),
        ]);
        setActiveRun(nextRun);
        setSession(refreshedSession);
        setSessionCatalog(listed.sessions || []);
        if (nextRun.status === "completed") {
          setBusy(false);
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
  }, [activeRun]);

  useEffect(() => {
    if (activeRun?.status === "running") {
      const checkpointCount = activeRun.result?.trajectory?.length || 0;
      if (checkpointCount > 0) {
        setSelectedCheckpointIndex(checkpointCount - 1);
      }
    }
  }, [activeRun?.status, activeRun?.result?.trajectory?.length]);

  const checkpoints = activeRun?.result?.trajectory || [];
  const selectedCheckpoint = checkpoints[selectedCheckpointIndex] || null;
  const uiMetadata = activeRun?.result?.metadata?.ui || {};
  const routeCase = activeRun?.result?.metadata?.route_case || null;
  const canBrowseTrajectory = Boolean(uiMetadata.can_browse_trajectory);
  const currentAudioSrc = canBrowseTrajectory && selectedCheckpoint?.audio_artifact
    ? `${API_BASE}${selectedCheckpoint.audio_artifact}`
    : activeRun?.result?.artifacts?.final_audio
      ? `${API_BASE}${activeRun.result.artifacts.final_audio}`
      : "";
  const inputAudioSrc = activeRun?.result?.artifacts?.input_audio
    ? `${API_BASE}${activeRun.result.artifacts.input_audio}`
    : "";

  const selectedParams =
    canBrowseTrajectory &&
    selectedCheckpoint?.params &&
    Object.keys(selectedCheckpoint.params).length > 0
      ? selectedCheckpoint.params
      : activeRun?.result?.params || {};

  let statusText = "No run selected";
  if (activeRun) {
    if (activeRun.status === "running") {
      statusText = `Running ${activeRun.current_iteration || 0}/${activeRun.total_iterations || "?"}`;
    } else if (activeRun.status === "completed") {
      statusText = "Completed";
    } else if (activeRun.status === "failed") {
      statusText = "Failed";
    } else {
      statusText = activeRun.status;
    }
  }

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
    const [refreshedSession, listed] = await Promise.all([
      fetchJson(`/sessions/${session.session_id}`),
      fetchJson("/sessions"),
    ]);
    setSession(refreshedSession);
    setSessionCatalog(listed.sessions || []);
  }

  async function handleRun(event) {
    event.preventDefault();
    setError("");

    if (viewingHistory) {
      setError("History entries are read-only. Return to the live session to submit a new refinement prompt.");
      return;
    }

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
            llm_model: settings.llm_model.trim() || null,
          },
        }),
      });
      setActiveRun(createdRun);
      setSelectedRunId(createdRun.run_id);
      const [refreshedSession, listed] = await Promise.all([
        fetchJson(`/sessions/${session.session_id}`),
        fetchJson("/sessions"),
      ]);
      setSession(refreshedSession);
      setSessionCatalog(listed.sessions || []);
      setSelectedCheckpointIndex(0);
    } catch (err) {
      setBusy(false);
      setError(String(err));
    }
  }

  async function handleSelectRun(runId) {
    setSelectedRunId(runId);
    setSelectedCheckpointIndex(0);
    setError("");
  }

  async function handleSelectSession(sessionId) {
    try {
      setError("");
      setBusy(false);
      const loaded = await fetchJson(`/sessions/${sessionId}`);
      setSession(loaded);
      const nextRunId = loaded.runs?.length ? loaded.runs[loaded.runs.length - 1].run_id : null;
      setSelectedRunId(nextRunId);
      setActiveRun(null);
      setSelectedCheckpointIndex(0);
      setAudioFile(null);
    } catch (err) {
      setError(String(err));
    }
  }

  async function handleCreateSession() {
    try {
      setError("");
      const created = await fetchJson("/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const listed = await fetchJson("/sessions");
      setSessionCatalog(listed.sessions || []);
      setSession(created);
      setSelectedRunId(null);
      setActiveRun(null);
      setInstruction("");
      setAudioFile(null);
      setSelectedCheckpointIndex(0);
    } catch (err) {
      setError(String(err));
    }
  }

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function renderRunModeMessage() {
    if (!activeRun?.result) {
      return (
        <p className="muted">
          Initial prompts create the first FX chain. Later prompts refine the live session head.
        </p>
      );
    }

    if (uiMetadata.initialization_only) {
      return (
        <div className="info-box">
          This run only performed LLM initialization for new FX. No optimization trajectory exists, so slider browsing is disabled.
        </div>
      );
    }

    if (routeCase === "reuse_all") {
      return (
        <div className="info-box">
          This run refined the existing chain. Checkpoints appear only at saved snapshot intervals.
        </div>
      );
    }

    if (routeCase === "reuse_and_initialize") {
      return (
        <div className="info-box">
          This run mixed refinement with new FX initialization. Existing FX were optimized while newly added FX were initialized and merged.
        </div>
      );
    }

    return <p className="muted">{uiMetadata.trajectory_reason || "Run metadata unavailable."}</p>;
  }

  return (
    <div className="workspace">
      <aside className="sidebar">
        <div className="sidebar-card session-card">
          <p className="eyebrow">Session</p>
          <h1>InstructFX2FX</h1>
          <p className="sidebar-note">
            Prompt-guided tone design for musicians, with session memory, checkpoint playback, and refinement history.
          </p>
          <div className="session-actions">
            <button type="button" className="secondary-action" onClick={handleCreateSession}>
              New session
            </button>
          </div>
        </div>

        <div className="sidebar-card history-card">
          <div className="section-head">
            <h2>Sessions</h2>
            <span>{sessionCatalog.length}</span>
          </div>
          <div className="history-stack compact-stack">
            {sessionCatalog.length === 0 ? (
              <p className="muted">No saved sessions yet.</p>
            ) : (
              sessionCatalog.map((item) => {
                const isActive = item.session_id === session?.session_id;
                return (
                  <button
                    key={item.session_id}
                    type="button"
                    className={`session-button${isActive ? " active" : ""}`}
                    onClick={() => handleSelectSession(item.session_id)}
                  >
                    <div className="history-topline">
                      <strong>{item.latest_prompt || "Untitled session"}</strong>
                    </div>
                    <div className="history-subline">
                      <span>{item.history_length} prompt{item.history_length === 1 ? "" : "s"}</span>
                      <span>{item.audio_uploaded ? "audio ready" : "no audio"}</span>
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        <div className="sidebar-card history-card">
          <div className="section-head">
            <h2>History</h2>
            <span>{runs.length}</span>
          </div>
          <div className="history-stack">
            {runs.length === 0 ? (
              <p className="muted">No prompts submitted yet.</p>
            ) : (
              [...runs].reverse().map((item) => {
                const isActive = item.run_id === selectedRunId;
                return (
                  <button
                    key={item.run_id}
                    type="button"
                    className={`history-button${isActive ? " active" : ""}`}
                    onClick={() => handleSelectRun(item.run_id)}
                  >
                    <div className="history-topline">
                      <strong>{item.instruction}</strong>
                      <span className="history-status">{item.status}</span>
                    </div>
                    <div className="history-subline">
                      <span>{item.fx_chain?.join(" -> ") || "Awaiting output"}</span>
                      <span>{formatPercent(item.progress)}</span>
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>
      </aside>

      <main className="main-column">
        <section className="composer-card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Live Session</p>
              <h2>{viewingHistory ? "History View" : promptModeLabel}</h2>
            </div>
            {viewingHistory ? (
              <button type="button" onClick={() => handleSelectRun(liveRunId)} disabled={!liveRunId}>
                Return to live
              </button>
            ) : null}
          </div>

          {viewingHistory ? (
            <div className="info-box">
              You are viewing a historical run. This view is read-only; refining from history is disabled for now.
            </div>
          ) : null}

          <form onSubmit={handleRun} className="stack">
            <label className="field">
              <span>Dry audio</span>
              <div className="file-picker">
                <label className={`file-trigger${viewingHistory ? " disabled" : ""}`} htmlFor="dry-audio-input">
                  Select audio
                </label>
                <span className="file-name">
                  {audioFile?.name || (session?.audio_uploaded ? "Current session audio loaded" : "No file selected")}
                </span>
                <input
                  id="dry-audio-input"
                  className="native-file-input"
                  type="file"
                  accept=".wav,audio/wav"
                  disabled={viewingHistory}
                  onChange={(event) => setAudioFile(event.target.files?.[0] || null)}
                />
              </div>
            </label>

            <label className="field">
              <span>Prompt</span>
              <textarea
                value={instruction}
                onChange={(event) => setInstruction(event.target.value)}
                rows={3}
                disabled={viewingHistory}
                placeholder="make it sound like in a bathroom"
              />
            </label>

            <label className="field">
              <span>OpenRouter model</span>
              <input
                type="text"
                value={settings.llm_model}
                disabled={viewingHistory}
                onChange={(event) => updateSetting("llm_model", event.target.value)}
                placeholder="Optional override, e.g. google/gemini-2.5-pro or openai/gpt-4o"
              />
            </label>

            {isInitialPrompt ? (
              <div className="info-box subtle">
                This is an initial prompt. Optimization controls become available after the first run, once the session has something to refine.
              </div>
            ) : (
              <div className="settings-grid">
                <label className="field">
                  <span>Iterations</span>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    disabled={viewingHistory}
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
                    disabled={viewingHistory}
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
                    disabled={viewingHistory}
                    value={settings.snapshot_interval}
                    onChange={(event) => updateSetting("snapshot_interval", event.target.value)}
                  />
                </label>
              </div>
            )}

            <button type="submit" className="primary-action" disabled={busy || !session || viewingHistory}>
              {busy ? "Running..." : "Submit prompt"}
            </button>
          </form>

          {error ? <div className="error-box">{error}</div> : null}
        </section>

        <section className="run-card">
          <div className="section-head">
            <div>
              <p className="eyebrow">Selected Run</p>
              <h2>{activeRun?.instruction || "No run selected"}</h2>
            </div>
            <div className={`status-pill status-${activeRun?.status || "idle"}`}>{statusText}</div>
          </div>

          {activeRun ? (
            <>
              <div className="progress-block">
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${activeRun.progress || 0}%` }} />
                </div>
                <div className="progress-meta">
                  <span>{formatPercent(activeRun.progress)}</span>
                  <span>
                    {activeRun.current_iteration || 0}
                    {activeRun.total_iterations ? ` / ${activeRun.total_iterations}` : ""}
                  </span>
                </div>
              </div>

              {renderRunModeMessage()}

              <div className="audio-grid">
                <div className="audio-card">
                  <h3>Input</h3>
                  <audio controls src={inputAudioSrc} />
                </div>
                <div className="audio-card">
                  <h3>{canBrowseTrajectory && selectedCheckpoint ? `Checkpoint: ${selectedCheckpoint.label}` : "Current output"}</h3>
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
                    The slider only stops on real saved checkpoints. If your interval is `5`, you should see `iter_5`, `iter_10`, `iter_15`, and so on.
                  </p>
                </div>
              ) : activeRun.result ? (
                <div className="info-box subtle">
                  {uiMetadata.trajectory_reason || "No trajectory slider is available for this run."}
                </div>
              ) : null}

              <div className="stack compact">
                <h3>FX chain</h3>
                <div className="chips">
                  {(activeRun.result?.fx_chain || []).map((fx) => (
                    <span key={fx} className="chip">{fx}</span>
                  ))}
                </div>
              </div>

              <div className="param-card">
                <h3>Parameters</h3>
                <pre>{JSON.stringify(selectedParams, null, 2)}</pre>
              </div>
            </>
          ) : (
            <p className="muted">Select a run from the history or submit a new prompt.</p>
          )}
        </section>
      </main>
    </div>
  );
}
