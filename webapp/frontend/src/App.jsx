import { useState, useEffect } from 'react'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [models, setModels] = useState(null)
  const [activeTab, setActiveTab] = useState('reverb')

  // Load available models on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/models`)
      .then(res => res.json())
      .then(data => {
        if (data.models && Object.keys(data.models).length > 0) {
          setModels(data.models)
          // Set default model (first available)
          const firstProvider = Object.keys(data.models)[0]
          const firstModel = Object.keys(data.models[firstProvider])[0]
          setSelectedModel({
            provider: firstProvider,
            model: data.models[firstProvider][firstModel].model
          })
        } else {
          // No API keys configured
          setError('No API keys configured. Please add at least one API key to baseline-system/.env')
        }
      })
      .catch(err => {
        console.error('Failed to load models:', err)
        setError('Failed to connect to backend. Make sure the server is running on port 8000.')
      })
  }, [])

  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter a text description')
      return
    }

    setIsGenerating(true)
    setError(null)
    setResult(null)

    try {
      // Upload audio file if provided
      let audioFilename = null
      if (audioFile) {
        const formData = new FormData()
        formData.append('file', audioFile)

        const uploadRes = await fetch(`${API_BASE_URL}/api/upload`, {
          method: 'POST',
          body: formData
        })

        if (!uploadRes.ok) {
          throw new Error('Failed to upload audio file')
        }

        const uploadData = await uploadRes.json()
        audioFilename = uploadData.filename
      }

      // Generate parameters
      const generateFormData = new FormData()
      generateFormData.append('text', text)
      generateFormData.append('model_provider', selectedModel.provider)
      generateFormData.append('model_name', selectedModel.model)
      if (audioFilename) {
        generateFormData.append('audio_filename', audioFilename)
      }

      const generateRes = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        body: generateFormData
      })

      if (!generateRes.ok) {
        const errorData = await generateRes.json()
        throw new Error(errorData.detail || 'Failed to generate parameters')
      }

      const data = await generateRes.json()
      setResult(data)
      setActiveTab('reverb') // Reset to first tab
    } catch (err) {
      setError(err.message)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      // Check file type
      const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/mp4']
      if (!allowedTypes.some(type => file.type.startsWith('audio/'))) {
        setError('Please select a valid audio file (WAV, MP3, FLAC, OGG)')
        return
      }
      setAudioFile(file)
      setError(null)
    }
  }

  const renderParameters = () => {
    if (!result) return null

    const { reverb, eq, compressor } = result.parameters

    return (
      <div className="result-container">
        <h2>Generated Parameters</h2>

        {/* Tab Navigation */}
        <div className="tabs">
          <button
            className={activeTab === 'reverb' ? 'active' : ''}
            onClick={() => setActiveTab('reverb')}
          >
            Reverb
          </button>
          <button
            className={activeTab === 'eq' ? 'active' : ''}
            onClick={() => setActiveTab('eq')}
          >
            EQ
          </button>
          <button
            className={activeTab === 'compressor' ? 'active' : ''}
            onClick={() => setActiveTab('compressor')}
          >
            Compressor
          </button>
        </div>

        {/* Parameter Display */}
        <div className="parameter-panel">
          {activeTab === 'reverb' && (
            <div className="param-section">
              <h3>Reverb Parameters</h3>
              <div className="param-grid">
                <ParamItem label="Delay Time" value={reverb.delay_time} unit="s" />
                <ParamItem label="Decay" value={reverb.decay} unit="" />
                <ParamItem label="Stereo Spread" value={reverb.stereo_spread} unit="" />
                <ParamItem label="Cutoff Frequency" value={reverb.cutoff_freq} unit="Hz" />
                <ParamItem label="Wet/Dry Mix" value={reverb.wet_dry} unit="" />
              </div>
            </div>
          )}

          {activeTab === 'eq' && (
            <div className="param-section">
              <h3>EQ Bands ({eq.length} bands)</h3>
              <div className="eq-bands">
                {eq.map((band, idx) => (
                  <div key={idx} className="eq-band">
                    <div className="band-header">Band {idx + 1}</div>
                    <ParamItem label="Frequency" value={band.freq} unit="Hz" />
                    <ParamItem label="Gain" value={band.gain} unit="dB" />
                    <ParamItem label="Q Factor" value={band.Q} unit="" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'compressor' && (
            <div className="param-section">
              <h3>Compressor Parameters</h3>
              <div className="param-grid">
                <ParamItem label="Threshold" value={compressor.threshold} unit="dB" />
                <ParamItem label="Ratio" value={compressor.ratio} unit=":1" />
                <ParamItem label="Attack" value={compressor.attack} unit="s" />
                <ParamItem label="Release" value={compressor.release} unit="s" />
                <ParamItem label="Makeup Gain" value={compressor.makeup_gain} unit="dB" />
              </div>
            </div>
          )}
        </div>

        {/* JSON Export */}
        <details className="json-export">
          <summary>View Full JSON</summary>
          <pre>{JSON.stringify(result.parameters, null, 2)}</pre>
        </details>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>Text2Preset</h1>
        <p>LLM-powered audio effect parameter generation</p>
      </header>

      <main>
        <div className="input-section">
          {/* Text Input */}
          <div className="input-group">
            <label htmlFor="text-input">
              Describe the audio effect you want:
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g., warm cathedral reverb with bright EQ..."
              rows={4}
            />
          </div>

          {/* Model Selection */}
          {models && (
            <div className="input-group">
              <label htmlFor="model-select">LLM Model:</label>
              <select
                id="model-select"
                value={selectedModel ? `${selectedModel.provider}:${selectedModel.model}` : ''}
                onChange={(e) => {
                  const [provider, ...modelParts] = e.target.value.split(':')
                  const model = modelParts.join(':')
                  setSelectedModel({ provider, model })
                }}
              >
                {Object.entries(models).map(([provider, providerModels]) => (
                  <optgroup key={provider} label={provider.toUpperCase()}>
                    {Object.entries(providerModels).map(([key, modelInfo]) => (
                      <option key={key} value={`${provider}:${modelInfo.model}`}>
                        {modelInfo.name} - {modelInfo.speed} / {modelInfo.cost}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          )}

          {/* Audio File Upload */}
          <div className="input-group">
            <label htmlFor="audio-upload">
              Upload dry audio (optional):
            </label>
            <input
              id="audio-upload"
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
            />
            {audioFile && (
              <div className="file-info">
                Selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
              </div>
            )}
          </div>

          {/* Generate Button */}
          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
          >
            {isGenerating ? 'Generating...' : 'Generate Parameters'}
          </button>

          {/* Error Display */}
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        {renderParameters()}
      </main>
    </div>
  )
}

// Helper component for parameter display
function ParamItem({ label, value, unit }) {
  const displayValue = typeof value === 'number'
    ? value.toFixed(value % 1 === 0 ? 0 : 4)
    : value

  return (
    <div className="param-item">
      <span className="param-label">{label}</span>
      <span className="param-value">
        {displayValue} <span className="param-unit">{unit}</span>
      </span>
    </div>
  )
}

export default App
