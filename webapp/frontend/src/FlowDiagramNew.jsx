import { useCallback } from 'react'
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './FlowDiagramNew.css'

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
  systemPrompt
}) {
  const initialNodes = [
    // Audio Upload (top right)
    {
      id: 'audio-upload',
      type: 'default',
      position: { x: 550, y: 20 },
      data: { label: hasAudio ? `🎵 ${audioFileName}` : '📁 Click to Upload Audio' },
      style: {
        background: 'white',
        border: '2px solid #d1d5db',
        borderRadius: '10px',
        padding: '12px 20px',
        cursor: 'pointer',
        fontSize: '13px',
        fontWeight: '500',
        minWidth: '200px',
        textAlign: 'center',
      },
    },
    // Main flow
    {
      id: 'input',
      type: 'default',
      position: { x: 30, y: 120 },
      data: { label: '📝 Input\nuser input + system prompt' },
      style: {
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        border: 'none',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '160px',
        minHeight: '80px',
        fontSize: '13px',
        fontWeight: '600',
        textAlign: 'center',
        opacity: stage >= 1 ? 1 : 0.5,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        whiteSpace: 'pre-line',
      },
    },
    {
      id: 'llm',
      type: 'default',
      position: { x: 230, y: 120 },
      data: { label: stage >= 2 ? '⚙️ LLM\nProcessing' : '🤖 LLM' },
      style: {
        background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        color: 'white',
        border: 'none',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '140px',
        minHeight: '80px',
        fontSize: '13px',
        fontWeight: '600',
        textAlign: 'center',
        opacity: stage >= 2 ? 1 : 0.5,
        whiteSpace: 'pre-line',
      },
    },
    {
      id: 'parameters',
      type: 'default',
      position: { x: 410, y: 120 },
      data: { label: '📊 Parameters\nJSON format' },
      style: {
        background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        color: 'white',
        border: 'none',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '160px',
        minHeight: '80px',
        fontSize: '13px',
        fontWeight: '600',
        textAlign: 'center',
        opacity: stage >= 3 ? 1 : 0.5,
        whiteSpace: 'pre-line',
      },
    },
    {
      id: 'system',
      type: 'default',
      position: { x: 610, y: 120 },
      data: { label: '⚡ System\nApply Effects' },
      style: {
        background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
        color: 'white',
        border: 'none',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '150px',
        minHeight: '80px',
        fontSize: '13px',
        fontWeight: '600',
        textAlign: 'center',
        opacity: stage >= 4 ? 1 : 0.5,
        whiteSpace: 'pre-line',
      },
    },
    {
      id: 'result',
      type: 'default',
      position: { x: 800, y: 120 },
      data: { label: processedAudio ? '🎧 Result\nProcessed Audio' : '🎵 Result\nAudio Output' },
      style: {
        background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        color: 'white',
        border: 'none',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '160px',
        minHeight: '80px',
        fontSize: '13px',
        fontWeight: '600',
        textAlign: 'center',
        opacity: stage >= 5 ? 1 : 0.5,
        whiteSpace: 'pre-line',
      },
    },
    // Judge system (bottom, disabled)
    {
      id: 'judge',
      type: 'default',
      position: { x: 300, y: 300 },
      data: { label: '👨‍⚖️ Judge System\n(Coming Soon)' },
      style: {
        background: '#fef3c7',
        color: '#92400e',
        border: '2px dashed #d97706',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '140px',
        fontSize: '12px',
        textAlign: 'center',
        opacity: 0.4,
        whiteSpace: 'pre-line',
      },
    },
    {
      id: 'refine',
      type: 'default',
      position: { x: 480, y: 300 },
      data: { label: '🔄 Refine\n(Coming Soon)' },
      style: {
        background: '#f3f4f6',
        color: '#6b7280',
        border: '2px dashed #9ca3af',
        borderRadius: '12px',
        padding: '16px',
        minWidth: '140px',
        fontSize: '12px',
        textAlign: 'center',
        opacity: 0.4,
        whiteSpace: 'pre-line',
      },
    },
  ]

  const initialEdges = [
    { id: 'e1', source: 'input', target: 'llm', animated: stage >= 2 },
    { id: 'e2', source: 'llm', target: 'parameters', animated: stage >= 3 },
    { id: 'e3', source: 'parameters', target: 'system', animated: stage >= 4 },
    { id: 'e4', source: 'system', target: 'result', animated: stage >= 5 },
    { id: 'e5', source: 'audio-upload', target: 'system', type: 'step', style: { strokeDasharray: '5,5' } },
    // Judge loop (disabled)
    { id: 'e6', source: 'result', target: 'judge', style: { strokeDasharray: '5,5', opacity: 0.3 } },
    { id: 'e7', source: 'judge', target: 'refine', style: { strokeDasharray: '5,5', opacity: 0.3 } },
    { id: 'e8', source: 'refine', target: 'llm', style: { strokeDasharray: '5,5', opacity: 0.3 } },
  ]

  const [nodes] = useNodesState(initialNodes)
  const [edges] = useEdgesState(initialEdges)

  return (
    <div className="flow-diagram-new">
      <div className="flow-header">
        <h2>Experimental Architecture</h2>
        <span className="flow-date">2025/10/16</span>
      </div>

      <div className="flow-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>

      {/* Input controls panel */}
      <div className="controls-panel">
        <div className="control-section">
          <label>Text Description</label>
          <textarea
            value={userInput}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder="e.g., after rain campus in October"
            rows={3}
          />
        </div>

        <div className="control-section">
          <label>Audio Sample</label>
          <input
            type="file"
            id="audio-upload-input"
            accept="audio/*"
            onChange={onAudioChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="audio-upload-input" className="file-upload-btn">
            {hasAudio ? `🎵 ${audioFileName}` : '📁 Upload Audio'}
          </label>
        </div>

        {systemPrompt && (
          <details className="control-section">
            <summary>System Prompt</summary>
            <pre className="system-prompt-display">{systemPrompt}</pre>
          </details>
        )}

        <div className="control-buttons">
          <button
            onClick={onGenerate}
            disabled={!userInput || isGenerating}
            className="btn-generate"
          >
            {isGenerating ? '⚙️ Generating...' : '→ Generate Parameters'}
          </button>

          <button
            onClick={onProcess}
            disabled={!hasAudio || !parameters || isProcessing}
            className="btn-process"
          >
            {isProcessing ? '⚙️ Processing...' : '🎵 Apply Effects'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default FlowDiagramNew
