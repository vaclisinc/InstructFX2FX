import { useState } from 'react'
import './FlowDiagram.css'

function FlowDiagram({
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
  systemPrompt
}) {
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)

  return (
    <div className="flow-diagram">

      {/* Main Flow */}
      <div className="flow-main">
        {/* Audio Upload - Top Right */}
        <div className="audio-upload-top">
          <input
            type="file"
            id="diagram-audio-upload"
            accept="audio/*"
            onChange={onAudioChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="diagram-audio-upload" className="audio-upload-box-top">
            <div className="audio-label">Input audio sample</div>
            <div className="audio-icon">{hasAudio ? `🎵 ${audioFileName}` : '📁 Click to upload'}</div>
          </label>
        </div>

        {/* Main horizontal flow */}
        <div className="flow-horizontal">
          {/* Input Box - Text only */}
          <div className={`flow-box input-box ${stage >= 1 ? 'active' : ''}`}>
            <div className="box-title">Input</div>
            <div className="box-subtitle">user input + system prompt</div>

            {/* Text Input */}
            <textarea
              className="integrated-input"
              placeholder="e.g., after rain campus in October"
              value={userInput}
              onChange={(e) => onTextChange(e.target.value)}
              rows={3}
            />

            {/* System Prompt Toggle */}
            {systemPrompt && (
              <div className="system-prompt-toggle-section">
                <button
                  className="system-prompt-btn"
                  onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                >
                  {showSystemPrompt ? '▼ Hide' : '▶ Show'} System Prompt
                </button>
                {showSystemPrompt && (
                  <div className="system-prompt-preview">
                    {systemPrompt}
                  </div>
                )}
              </div>
            )}

            <button
              className="generate-btn-diagram"
              onClick={onGenerate}
              disabled={!userInput || isGenerating}
            >
              {isGenerating ? '⚙️ Generating...' : '→ Generate'}
            </button>
          </div>

          {/* Arrow */}
          <div className={`flow-arrow-h ${stage >= 2 ? 'flowing' : ''}`}>
            <svg viewBox="0 0 40 20" preserveAspectRatio="none">
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                  <polygon points="0 0, 10 3, 0 6" fill="#aaa" />
                </marker>
              </defs>
              <line x1="0" y1="10" x2="40" y2="10" stroke="#aaa" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </svg>
          </div>

          {/* LLM */}
          <div className={`flow-box llm-box ${stage >= 2 ? 'active' : ''}`}>
            <div className="box-title">LLM</div>
            {stage >= 2 && <div className="llm-spinner">⚙️</div>}
          </div>

          {/* Arrow */}
          <div className={`flow-arrow-h ${stage >= 3 ? 'flowing' : ''}`}>
            <svg viewBox="0 0 40 20" preserveAspectRatio="none">
              <line x1="0" y1="10" x2="40" y2="10" stroke="#aaa" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </svg>
          </div>

          {/* Output Parameters */}
          <div className={`flow-box output-box ${stage >= 3 ? 'active' : ''}`}>
            <div className="box-title">Parameters</div>
            <div className="box-subtitle">JSON format</div>
            <div className="params-preview">
              {parameters && (
                <div className="param-json">
                  reverb: [{parameters.reverb.delay_time.toFixed(2)}, ...]<br/>
                  EQ: [...]<br/>
                  compressor: [...]
                </div>
              )}
            </div>
          </div>

          {/* Arrow */}
          <div className={`flow-arrow-h ${stage >= 4 ? 'flowing' : ''}`}>
            <svg viewBox="0 0 40 20" preserveAspectRatio="none">
              <line x1="0" y1="10" x2="40" y2="10" stroke="#aaa" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </svg>
          </div>

          {/* System */}
          <div className={`flow-box system-box ${stage >= 4 ? 'active' : ''}`}>
            <div className="box-title">Apply</div>
            {hasAudio && parameters ? (
              <button
                className="process-btn-diagram"
                onClick={onProcess}
                disabled={isProcessing}
              >
                {isProcessing ? '⚙️ Processing...' : '🎵 Process'}
              </button>
            ) : (
              <div className="system-waiting">
                {!hasAudio && 'Need audio'}
                {hasAudio && !parameters && 'Need params'}
              </div>
            )}
          </div>

          {/* Arrow */}
          <div className={`flow-arrow-h ${stage >= 5 ? 'flowing' : ''}`}>
            <svg viewBox="0 0 40 20" preserveAspectRatio="none">
              <line x1="0" y1="10" x2="40" y2="10" stroke="#aaa" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </svg>
          </div>

          {/* Processed Audio */}
          <div className={`flow-box processed-box ${stage >= 5 ? 'active' : ''}`}>
            <div className="box-title">Result</div>
            {processedAudio && (
              <div className="waveform-blue">🎧</div>
            )}
          </div>
        </div>

        {/* Bottom: Judge System Loop (Disabled) */}
        <div className="flow-bottom-section">
          {/* Arrow down from Processed Audio */}
          <div className="flow-arrow-down disabled">
            <svg viewBox="0 0 20 60" preserveAspectRatio="none">
              <line x1="10" y1="0" x2="10" y2="60" stroke="#666" strokeWidth="2" strokeDasharray="5,5" />
            </svg>
          </div>

          {/* Judge System */}
          <div className="flow-box judge-box disabled">
            <div className="box-title">Judge System</div>
            <div className="coming-soon">Coming Soon</div>
          </div>

          {/* Arrow back with score */}
          <div className="flow-arrow-back disabled">
            <div className="score-label">Score</div>
            <svg viewBox="0 0 200 60" preserveAspectRatio="none">
              <path d="M 200 30 L 20 30 L 20 10 L 0 30 L 20 50 L 20 30" stroke="#666" strokeWidth="2" fill="none" strokeDasharray="5,5" />
            </svg>
          </div>

          {/* Refine Box */}
          <div className="flow-box refine-box disabled">
            <div className="box-title">Refine</div>
            <div className="refine-text">
              Reprompt by providing score<br/>
              e.g. The score is {'{'}score{'}'}, the user<br/>
              input is {'{'}original input{'}'}, your<br/>
              generated parameters: {'{'}json{'}'},<br/>
              Please redesign the sound.
            </div>
          </div>

          {/* Arrow back to LLM */}
          <div className="flow-arrow-up disabled">
            <svg viewBox="0 0 20 100" preserveAspectRatio="none">
              <line x1="10" y1="0" x2="10" y2="100" stroke="#666" strokeWidth="2" strokeDasharray="5,5" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FlowDiagram
