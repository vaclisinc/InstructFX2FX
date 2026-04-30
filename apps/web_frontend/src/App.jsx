import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const FX_LABELS = {
  eq: "EQ",
  comp: "compressor",
  rev: "reverb",
  dist: "distortion",
  delay: "delay",
  pitchshift: "pitch shift",
  bitcrush: "bitcrush",
};

const ROUTE_LABELS = {
  reuse_all: "Case 1 · reuse and optimize existing FX",
  initialize_all: "Case 2 · initialize new FX only",
  reuse_and_initialize: "Case 3 · mix existing FX with new initialization",
  empty_chain: "Empty chain",
};
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

function formatFxLabel(fx) {
  return FX_LABELS[fx] || fx;
}

function formatFxChain(fxChain) {
  return (fxChain || []).map(formatFxLabel).join(" -> ");
}

function formatRouteCase(routeCase) {
  return ROUTE_LABELS[routeCase] || "Waiting for parser route";
}

export default function App() {
  const [sessionCatalog, setSessionCatalog] = useState([]);
  const [session, setSession] = useState(null);
  const [sessionNameDraft, setSessionNameDraft] = useState("");
  const [openSessionMenuId, setOpenSessionMenuId] = useState(null);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [instruction, setInstruction] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [activeRun, setActiveRun] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedCheckpointIndex, setSelectedCheckpointIndex] = useState(0);
  const [pendingAudioPreviewUrl, setPendingAudioPreviewUrl] = useState("");

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
    setSessionNameDraft(session?.name || "");
  }, [session?.session_id, session?.name]);

  useEffect(() => {
    if (!audioFile) {
      setPendingAudioPreviewUrl("");
      return undefined;
    }

    const nextUrl = URL.createObjectURL(audioFile);
    setPendingAudioPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [audioFile]);

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
    if (!openSessionMenuId) {
      return undefined;
    }

    function handlePointerDown(event) {
      if (event.target.closest(".session-menu-wrap")) {
        return;
      }
      setOpenSessionMenuId(null);
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [openSessionMenuId]);

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
  const runMetadata = activeRun?.result?.metadata || {};
  const uiMetadata = runMetadata.ui || {};
  const routeCase = runMetadata.route_case || null;
  const stageMetadata = runMetadata.stages || {};
  const canBrowseTrajectory = Boolean(uiMetadata.can_browse_trajectory);
  const selectedCheckpointIsSessionHead =
    Boolean(
      session?.active_anchor_run_id &&
      session?.active_anchor_label &&
      activeRun?.run_id &&
      selectedCheckpoint?.label &&
      session.active_anchor_run_id === activeRun.run_id &&
      session.active_anchor_label === selectedCheckpoint.label
    );
  const currentAudioSrc = canBrowseTrajectory && selectedCheckpoint?.audio_artifact
    ? `${API_BASE}${selectedCheckpoint.audio_artifact}`
    : activeRun?.result?.artifacts?.final_audio
      ? `${API_BASE}${activeRun.result.artifacts.final_audio}`
      : "";
  const inputAudioSrc = activeRun?.result?.artifacts?.input_audio
    ? `${API_BASE}${activeRun.result.artifacts.input_audio}`
    : "";
  const dryAudioSrc = pendingAudioPreviewUrl || (session?.audio_artifact ? `${API_BASE}${session.audio_artifact}` : "");
  const audioLocked = Boolean(session?.audio_uploaded);
  const comparisonLabel =
    (activeRun?.result?.artifacts?.input_audio && activeRun?.result?.artifacts?.dry_audio !== activeRun?.result?.artifacts?.input_audio)
      ? "Previous output"
      : "Dry audio";

  const selectedParams =
    canBrowseTrajectory &&
    selectedCheckpoint?.params &&
    Object.keys(selectedCheckpoint.params).length > 0
      ? selectedCheckpoint.params
      : activeRun?.result?.params || {};
  const activeFxChain = activeRun?.result?.fx_chain || [];
  const hasDaspStage = activeFxChain.some((fx) => ["eq", "rev"].includes(fx));
  const hasPedalboardStage = activeFxChain.some((fx) => !["eq", "rev"].includes(fx));
  const daspStageStatus = stageMetadata.dasp
    ? "Completed"
    : activeRun?.status === "running" && hasDaspStage
      ? "Running"
      : hasDaspStage
        ? "Planned"
        : "Not used";
  const pedalboardStageStatus = stageMetadata.pedalboard
    ? "Completed"
    : activeRun?.status === "running" && hasPedalboardStage
      ? hasDaspStage ? "Queued after DASP" : "Running"
      : hasPedalboardStage
        ? "Planned"
        : "Not used";

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

  async function handleUseCheckpointAsSessionHead() {
    if (!session || !activeRun || !selectedCheckpoint) {
      return;
    }

    try {
      setError("");
      const updated = await fetchJson(`/sessions/${session.session_id}/active-checkpoint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: activeRun.run_id,
          checkpoint_label: selectedCheckpoint.label,
        }),
      });
      const listed = await fetchJson("/sessions");
      setSession(updated);
      setSessionCatalog(listed.sessions || []);
    } catch (err) {
      setError(String(err));
    }
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

  async function handleDeleteSession(item) {
    if (!window.confirm(`Delete session "${item.name || "Untitled session"}"? This will remove its saved runs and audio.`)) {
      return;
    }

    try {
      setError("");
      await fetchJson(`/sessions/${item.session_id}`, { method: "DELETE" });
      const listed = await fetchJson("/sessions");
      const nextCatalog = listed.sessions || [];
      setSessionCatalog(nextCatalog);
      setOpenSessionMenuId(null);
      setEditingSessionId(null);

      if (session?.session_id === item.session_id) {
        if (nextCatalog.length > 0) {
          const nextSession = await fetchJson(`/sessions/${nextCatalog[0].session_id}`);
          setSession(nextSession);
          const nextRunId = nextSession.runs?.length ? nextSession.runs[nextSession.runs.length - 1].run_id : null;
          setSelectedRunId(nextRunId);
        } else {
          const created = await fetchJson("/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          const refreshed = await fetchJson("/sessions");
          setSessionCatalog(refreshed.sessions || []);
          setSession(created);
          setSelectedRunId(null);
        }
        setActiveRun(null);
        setSelectedCheckpointIndex(0);
        setAudioFile(null);
        setInstruction("");
      }
    } catch (err) {
      setError(String(err));
    }
  }

  async function handleRenameSession(event) {
    event.preventDefault();
    if (!editingSessionId) {
      return;
    }

    const nextName = sessionNameDraft.trim();
    if (!nextName) {
      setError("Session name cannot be empty.");
      return;
    }

    try {
      setError("");
      const updated = await fetchJson(`/sessions/${editingSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: nextName }),
      });
      const listed = await fetchJson("/sessions");
      if (session?.session_id === editingSessionId) {
        setSession(updated);
      }
      setSessionCatalog(listed.sessions || []);
      setEditingSessionId(null);
      setOpenSessionMenuId(null);
    } catch (err) {
      setError(String(err));
    }
  }

  function handleStartRenameSession(item) {
    setEditingSessionId(item.session_id);
    setSessionNameDraft(item.name || "");
    setOpenSessionMenuId(null);
    setError("");
  }

  function handleCancelRenameSession() {
    setEditingSessionId(null);
    setSessionNameDraft(session?.name || "");
    setOpenSessionMenuId(null);
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
        </div>

        <div className="sidebar-card history-card">
          <div className="section-head">
            <div className="section-head-group">
              <h2>Sessions</h2>
              <span>{sessionCatalog.length}</span>
            </div>
            <button type="button" className="plus-action" onClick={handleCreateSession} aria-label="Create new session">
              +
            </button>
          </div>
          <div className="history-stack compact-stack">
            {sessionCatalog.length === 0 ? (
              <p className="muted">No saved sessions yet.</p>
            ) : (
              sessionCatalog.map((item) => {
                const isActive = item.session_id === session?.session_id;
                const isEditing = item.session_id === editingSessionId;
                const isMenuOpen = item.session_id === openSessionMenuId;
                return (
                  <div
                    key={item.session_id}
                    className={`session-row${isActive ? " active" : ""}${isEditing ? " editing" : ""}${isMenuOpen ? " menu-open" : ""}`}
                  >
                    {isEditing ? (
                      <form className="session-rename-inline" onSubmit={handleRenameSession}>
                        <input
                          type="text"
                          value={sessionNameDraft}
                          onChange={(event) => setSessionNameDraft(event.target.value)}
                          placeholder="Untitled session"
                          maxLength={120}
                          autoFocus
                        />
                        <div className="session-rename-actions">
                          <button
                            type="submit"
                            className="mini-action"
                            disabled={sessionNameDraft.trim() === (item.name || "")}
                          >
                            Save
                          </button>
                          <button type="button" className="mini-action ghost" onClick={handleCancelRenameSession}>
                            Cancel
                          </button>
                        </div>
                      </form>
                    ) : (
                      <>
                        <button
                          type="button"
                          className={`session-button${isActive ? " active" : ""}`}
                          onClick={() => handleSelectSession(item.session_id)}
                        >
                          <div className="history-topline">
                            <strong>{item.name || "Untitled session"}</strong>
                          </div>
                          <div className="history-subline">
                            <span>{item.latest_prompt || "No prompts yet"}</span>
                          </div>
                          <div className="history-subline">
                            <span>{item.history_length} prompt{item.history_length === 1 ? "" : "s"}</span>
                            <span>{item.audio_uploaded ? "audio ready" : "no audio"}</span>
                          </div>
                        </button>
                        <div className="session-menu-wrap">
                          <button
                            type="button"
                            className="session-menu-trigger"
                            aria-label={`Session actions for ${item.name || "Untitled session"}`}
                            onClick={() => setOpenSessionMenuId(isMenuOpen ? null : item.session_id)}
                          >
                            <span />
                            <span />
                            <span />
                          </button>
                          {isMenuOpen ? (
                            <div className="session-menu">
                              <button type="button" onClick={() => handleStartRenameSession(item)}>
                                Rename
                              </button>
                              <button type="button" className="danger-item" onClick={() => handleDeleteSession(item)}>
                                Delete
                              </button>
                            </div>
                          ) : null}
                        </div>
                      </>
                    )}
                  </div>
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
                      <span>{formatFxChain(item.fx_chain) || "Awaiting output"}</span>
                      <span>{formatPercent(item.progress)}</span>
                    </div>
                    <div className="history-subline">
                      <span>{item.llm_model || "default model"}</span>
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
                {dryAudioSrc ? (
                  <div className="uploaded-audio-block">
                    <div className="uploaded-audio-meta">
                      <span className="file-name">
                        {audioFile?.name || session?.audio_filename || "Current session audio"}
                      </span>
                    </div>
                    <audio controls src={dryAudioSrc} />
                  </div>
                ) : (
                  <>
                    <label className={`file-trigger${viewingHistory ? " disabled" : ""}`} htmlFor="dry-audio-input">
                      Select audio
                    </label>
                    <span className="file-name">No file selected</span>
                  </>
                )}
                <input
                  id="dry-audio-input"
                  className="native-file-input"
                  type="file"
                  accept=".wav,audio/wav"
                  disabled={viewingHistory || audioLocked}
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

              <details className="trace-panel" open={activeRun?.status === "running"}>
                <summary>Pipeline trace</summary>
                <div className="trace-stack">
                  <div className="trace-item">
                    <div className="trace-head">
                      <h3>Layer 1 · FX selector</h3>
                      <span className="trace-status">{activeFxChain.length > 0 ? "Resolved" : "Pending"}</span>
                    </div>
                    <p className="trace-copy">
                      {activeFxChain.length > 0 ? `Selected chain: ${formatFxChain(activeFxChain)}` : "Waiting for selected FX chain."}
                    </p>
                    <div className="chips">
                      <span className="chip">{activeRun.settings?.llm_model || "default model"}</span>
                    </div>
                  </div>

                  <div className="trace-item">
                    <div className="trace-head">
                      <h3>Layer 2 · Session router</h3>
                      <span className="trace-status">{routeCase ? "Resolved" : "Pending"}</span>
                    </div>
                    <p className="trace-copy">{formatRouteCase(routeCase)}</p>
                  </div>

                  <div className="trace-item">
                    <div className="trace-head">
                      <h3>Layer 3A · DASP optimization</h3>
                      <span className="trace-status">{daspStageStatus}</span>
                    </div>
                    <p className="trace-copy">
                      {hasDaspStage
                        ? `Gradient descent on ${formatFxChain(activeFxChain.filter((fx) => ["eq", "rev"].includes(fx)))}`
                        : "No DASP FX in this chain."}
                    </p>
                  </div>

                  <div className="trace-item">
                    <div className="trace-head">
                      <h3>Layer 3B · Pedalboard optimization</h3>
                      <span className="trace-status">{pedalboardStageStatus}</span>
                    </div>
                    <p className="trace-copy">
                      {hasPedalboardStage
                        ? `Bayesian optimization on ${formatFxChain(activeFxChain.filter((fx) => !["eq", "rev"].includes(fx)))}`
                        : "No Pedalboard FX in this chain."}
                    </p>
                  </div>
                </div>
              </details>

              <div className="audio-grid">
                <div className="audio-card">
                  <h3>{comparisonLabel}</h3>
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
                  <div className="checkpoint-actions">
                    <button
                      type="button"
                      className="secondary-inline-action"
                      onClick={handleUseCheckpointAsSessionHead}
                      disabled={selectedCheckpointIsSessionHead}
                    >
                      {selectedCheckpointIsSessionHead ? "Current session head" : "Use this checkpoint for next prompt"}
                    </button>
                  </div>
                  <p className="muted">
                    The slider only stops on real saved checkpoints. If your interval is `5`, you should see `iter_5`, `iter_10`, `iter_15`, and so on.
                  </p>
                </div>
              ) : activeRun.result && !uiMetadata.initialization_only ? (
                <div className="info-box subtle">
                  {uiMetadata.trajectory_reason || "No trajectory slider is available for this run."}
                </div>
              ) : null}

              <div className="run-summary-grid">
                <div className="detail-card">
                  <h3>FX chain</h3>
                  <div className="chips">
                    {(activeRun.result?.fx_chain || []).map((fx) => (
                      <span key={fx} className="chip">{formatFxLabel(fx)}</span>
                    ))}
                  </div>
                </div>

                <div className="detail-card">
                  <h3>Run settings</h3>
                  <div className="chips">
                    <span className="chip">{activeRun.settings?.llm_model || "default model"}</span>
                    <span className="chip">{`${activeRun.settings?.n_iterations ?? "?"} iter`}</span>
                    <span className="chip">{`lr ${activeRun.settings?.learning_rate ?? "?"}`}</span>
                    <span className="chip">{`snap ${activeRun.settings?.snapshot_interval ?? "off"}`}</span>
                  </div>
                </div>
              </div>

              <div className="param-card">
                <h3>Parameters</h3>
                <pre className="param-json">{JSON.stringify(selectedParams, null, 2)}</pre>
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
