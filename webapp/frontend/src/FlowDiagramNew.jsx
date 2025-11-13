import { useState } from 'react'
import './FlowDiagramNew.css'

const GRID_COLS = 5
const GRID_ROWS = 3

const layout = {
  input: { col: 1, row: 2 },
  llm: { col: 2, row: 2 },
  parameters: { col: 3, row: 2 },
  system: { col: 4, row: 2 },
  result: { col: 5, row: 2 },
  audio: { col: 5, row: 1 },
  judge: { col: 3, row: 3 },
  refine: { col: 4, row: 3 }
}

const connectors = [
  { id: 'input-llm', from: 'input', to: 'llm', stage: 2 },
  { id: 'llm-parameters', from: 'llm', to: 'parameters', stage: 3 },
  { id: 'parameters-system', from: 'parameters', to: 'system', stage: 4 },
  { id: 'system-result', from: 'system', to: 'result', stage: 5 },
  { id: 'audio-system', from: 'audio', to: 'system', stage: 4, dashed: true },
  { id: 'result-judge', from: 'result', to: 'judge', dashed: true, disabled: true },
  { id: 'judge-refine', from: 'judge', to: 'refine', dashed: true, disabled: true },
  { id: 'refine-llm', from: 'refine', to: 'llm', dashed: true, disabled: true }
]

const getStageClass = (stage, target) => {
  if (stage > target) return 'complete'
  if (stage === target) return 'active'
  return ''
}

const getCoords = (key) => {
  const { col, row } = layout[key]
  const xStep = 1000 / GRID_COLS
  const yStep = 500 / GRID_ROWS
  return {
    x: (col - 0.5) * xStep,
    y: (row - 0.5) * yStep
  }
}

const PROMPT_TEMPLATE = `Generate parameters for the following audio effects:

1. Reverb (creates spatial ambience):
   - delay_time: float (0.001 to 0.1 seconds)
   - decay: float (0.1 to 1.0, higher = longer reverb tail)
   - stereo_spread: float (-1.0 to 1.0, stereo width)
   - cutoff_freq: float (1000 to 20000 Hz, high frequency rolloff)
   - wet_dry: float (0.0 to 1.0, 0=dry, 1=wet)

2. EQ (equalizer, list of bands):
   Each band has:
   - freq: float (20 to 20000 Hz, center frequency)
   - gain: float (-20 to 20 dB, boost or cut)
   - Q: float (0.1 to 10, bandwidth, higher = narrower)

3. Compressor (dynamic range compression):
   - threshold: float (-60 to 0 dB, level above which compression starts)
   - ratio: float (1.0 to 20.0, compression ratio)
   - attack: float (0.001 to 0.1 seconds, how fast compressor responds)
   - release: float (0.01 to 1.0 seconds, how fast compressor releases)
   - makeup_gain: float (0 to 20 dB, output gain compensation)

Return ONLY valid JSON in this exact format:
{
  "reverb": {
    "delay_time": <float>,
    "decay": <float>,
    "stereo_spread": <float>,
    "cutoff_freq": <float>,
    "wet_dry": <float>
  },
  "eq": [
    {"freq": <float>, "gain": <float>, "Q": <float>},
    ...
  ],
  "compressor": {
    "threshold": <float>,
    "ratio": <float>,
    "attack": <float>,
    "release": <float>,
    "makeup_gain": <float>
  }
}

Do not include any explanatory text, only the JSON.`

function FlowDiagramNew({
  stage,
  userInput,
  hasAudio,
  parameters,
  processedAudio,
  onTextChange,
  onAudioChange,
  onGenerate,
  onProcess,
  isGenerating,
  isProcessing,
  audioFileName,
  systemPrompt,
  audioPreviewUrl,
  models,
  selectedModel,
  onModelChange
}) {
  const [paramTab, setParamTab] = useState('reverb')

  const renderParameters = () => {
    if (!parameters) {
      return <p className="placeholder">Parameters will land here.</p>
    }

    if (paramTab === 'reverb') {
      return (
        <ul className="parameter-preview">
          <li>Delay: <strong>{parameters.reverb.delay_time.toFixed(3)}s</strong></li>
          <li>Decay: <strong>{parameters.reverb.decay.toFixed(2)}</strong></li>
          <li>Stereo: <strong>{parameters.reverb.stereo_spread.toFixed(2)}</strong></li>
          <li>Cutoff: <strong>{Math.round(parameters.reverb.cutoff_freq)} Hz</strong></li>
          <li>Wet/Dry: <strong>{parameters.reverb.wet_dry.toFixed(2)}</strong></li>
        </ul>
      )
    }

    if (paramTab === 'eq') {
      return (
        <div className="eq-scroll">
          {parameters.eq.map((band, idx) => (
            <div key={`${band.freq}-${idx}`} className="eq-row">
              <span>Band {idx + 1}</span>
              <span>{Math.round(band.freq)} Hz</span>
              <span>{band.gain.toFixed(2)} dB</span>
              <span>Q {band.Q.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )
    }

    return (
      <ul className="parameter-preview">
        <li>Threshold: <strong>{parameters.compressor.threshold} dB</strong></li>
        <li>Ratio: <strong>{parameters.compressor.ratio}:1</strong></li>
        <li>Attack: <strong>{parameters.compressor.attack.toFixed(3)}s</strong></li>
        <li>Release: <strong>{parameters.compressor.release.toFixed(3)}s</strong></li>
        <li>Makeup: <strong>{parameters.compressor.makeup_gain} dB</strong></li>
      </ul>
    )
  }

  return (
    <section className="flow-diagram-new">
      <div className="flow-board">
        <div className="board-header">
          <div>
            <p className="micro-label">CNMAT · Research Group 2</p>
            <h2>Experimental Architecture</h2>
          </div>
          <span className="micro-label">2025/10/16</span>
        </div>

        <div className="board-body">
          <div className="board-inner">
            <svg className="flow-connectors" viewBox="0 0 1000 500" preserveAspectRatio="none">
            <defs>
              <marker id="arrow-active" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="#5b21b6" />
              </marker>
              <marker id="arrow-idle" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="#c7cedd" />
              </marker>
              <marker id="arrow-disabled" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="#d4d4d8" />
              </marker>
            </defs>
            {connectors.map(({ id, from, to, stage: arrowStage, dashed, disabled }) => {
              const start = getCoords(from)
              const end = getCoords(to)
              const isActive = arrowStage ? stage >= arrowStage : false
              const marker = disabled ? 'url(#arrow-disabled)' : isActive ? 'url(#arrow-active)' : 'url(#arrow-idle)'
              return (
                <line
                  key={id}
                  x1={start.x}
                  y1={start.y}
                  x2={end.x}
                  y2={end.y}
                  markerEnd={marker}
                  className={[
                    'connector-line',
                    isActive ? 'is-active' : '',
                    dashed ? 'is-dashed' : '',
                    disabled ? 'is-disabled' : ''
                  ].join(' ')}
                />
              )
            })}
            </svg>

            <div className="node-grid">
            <div className={`flow-node node-input ${getStageClass(stage, 1)}`} style={{ gridColumn: '1 / span 1', gridRow: '2' }}>
              <header>
                <p className="micro-label">Input</p>
                <h3>User Prompt + System Context</h3>
              </header>
              <p className="system-context">
                <strong>You are an audio processing expert.</strong> Given a high-level textual description of a desired audio effect or tone,
                generate audio effect parameters in JSON format.
              </p>
              <p className="micro-label">User description:</p>
              <textarea
                value={userInput}
                onChange={(e) => onTextChange(e.target.value)}
                placeholder="e.g. warm cathedral reverb with shimmering highs"
                rows={3}
              />
              <details className="prompt-toggle">
                <summary>Show system prompt template</summary>
                <pre>{systemPrompt || PROMPT_TEMPLATE}</pre>
              </details>
              <button
                onClick={onGenerate}
                disabled={!userInput.trim() || isGenerating}
                className="primary-btn"
              >
                {isGenerating ? 'Generating...' : 'Generate Parameters'}
              </button>
            </div>

            <div className={`flow-node node-llm ${getStageClass(stage, 2)}`} style={{ gridColumn: '2', gridRow: '2' }}>
              <header>
                <p className="micro-label">LLM</p>
                <h3>Inference Engine</h3>
              </header>
              {models ? (
                <select
                  value={selectedModel ? `${selectedModel.provider}:${selectedModel.model}` : ''}
                  onChange={(e) => {
                    if (!onModelChange) return
                    const [provider, ...modelParts] = e.target.value.split(':')
                    onModelChange({
                      provider,
                      model: modelParts.join(':')
                    })
                  }}
                >
                  {Object.entries(models).map(([provider, list]) => (
                    Object.entries(list).map(([key, info]) => (
                      <option key={`${provider}-${key}`} value={`${provider}:${info.model}`}>
                        {info.name} · {info.speed} · {info.cost}
                      </option>
                    ))
                  ))}
                </select>
              ) : (
                <p className="placeholder">Waiting for available models…</p>
              )}
              <p className="status-chip">
                {stage >= 3 ? 'Completed' : stage === 2 ? 'Running' : 'Idle'}
              </p>
            </div>

            <div className={`flow-node node-parameters ${getStageClass(stage, 3)}`} style={{ gridColumn: '3', gridRow: '2' }}>
              <header>
                <p className="micro-label">Output</p>
                <h3>Parameters (JSON)</h3>
              </header>
              <div className="param-tabs">
                {['reverb', 'eq', 'compressor'].map((tab) => (
                  <button
                    key={tab}
                    className={`param-tab ${paramTab === tab ? 'active' : ''}`}
                    onClick={() => setParamTab(tab)}
                    type="button"
                  >
                    {tab === 'reverb' && 'Reverb'}
                    {tab === 'eq' && 'EQ'}
                    {tab === 'compressor' && 'Compressor'}
                  </button>
                ))}
              </div>
              {renderParameters()}
            </div>

            <div className={`flow-node node-system ${getStageClass(stage, 4)}`} style={{ gridColumn: '4', gridRow: '2' }}>
              <header>
                <p className="micro-label">System</p>
                <h3>Apply Effects</h3>
              </header>
              <p className="placeholder">
                Upload an audio sample to apply the generated chain directly in the browser.
              </p>
              <button
                onClick={onProcess}
                disabled={!hasAudio || !parameters || isProcessing}
                className="secondary-btn"
              >
                {isProcessing ? 'Processing…' : 'Apply Effects'}
              </button>
            </div>

            <div className={`flow-node node-result ${getStageClass(stage, 5)}`} style={{ gridColumn: '5', gridRow: '2' }}>
              <header>
                <p className="micro-label">Result</p>
                <h3>Processed Audio</h3>
              </header>
              {processedAudio ? (
                <div className="result-summary">
                  <p>Ready for listening.</p>
                  <span>{processedAudio.processed_file}</span>
                  <div className="audio-compare">
                    {audioPreviewUrl && (
                      <div>
                        <label>Original Audio</label>
                        <audio controls src={audioPreviewUrl}>
                          Original preview unavailable.
                        </audio>
                      </div>
                    )}
                    <div>
                      <label>Processed Audio</label>
                      <audio controls src={processedAudio.download_url}>
                        Processed preview unavailable.
                      </audio>
                    </div>
                  </div>
                  <a
                    className="download-link"
                    href={processedAudio.download_url}
                    download={processedAudio.processed_file}
                  >
                    ⬇️ Download processed audio
                  </a>
                </div>
              ) : (
                <div className="placeholder">
                  Provide dry audio and generate parameters to compare original vs processed output here.
                </div>
              )}
            </div>

            <div className="flow-node node-audio" style={{ gridColumn: '5', gridRow: '1' }}>
              <header>
                <p className="micro-label">Input Audio Sample</p>
                <h3>Optional Reference</h3>
              </header>
              <input
                id="audio-upload-input"
                type="file"
                accept="audio/*"
                onChange={onAudioChange}
              />
              <label htmlFor="audio-upload-input" className="upload-tile">
                {hasAudio ? (
                  <>
                    <span className="emoji">🎵</span>
                    <span>{audioFileName}</span>
                  </>
                ) : (
                  <>
                    <span className="emoji">📁</span>
                    <span>Upload audio</span>
                  </>
                )}
              </label>
            </div>

            <div className="flow-node node-judge disabled" style={{ gridColumn: '3', gridRow: '3' }}>
              <header>
                <p className="micro-label">Judge System</p>
                <h3>Coming Soon</h3>
              </header>
              <p className="placeholder">Automated scoring &amp; critique lives here.</p>
            </div>

            <div className="flow-node node-refine disabled" style={{ gridColumn: '4', gridRow: '3' }}>
              <header>
                <p className="micro-label">Refine Loop</p>
                <h3>Coming Soon</h3>
              </header>
              <p className="placeholder">Iterative reprompting after evaluation.</p>
            </div>
          </div>
        </div>
        </div>
      </div>

    </section>
  )
}

export default FlowDiagramNew
