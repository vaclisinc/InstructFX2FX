import { useState, useEffect } from 'react'
import './App.css'
import FlowDiagramNew from './FlowDiagramNew'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [processedAudio, setProcessedAudio] = useState(null)
  const [error, setError] = useState(null)
  const [models, setModels] = useState(null)
  const [activeTab, setActiveTab] = useState('reverb')
  const [showSystemPrompt, setShowSystemPrompt] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState(null)

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
      // Generate parameters (no need to upload audio file)
      const generateFormData = new FormData()
      generateFormData.append('text', text)
      generateFormData.append('model_provider', selectedModel.provider)
      generateFormData.append('model_name', selectedModel.model)

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
      setSystemPrompt(data.system_prompt) // Store system prompt if returned
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

  const handleProcess = async () => {
    if (!audioFile) {
      setError('Please upload an audio file first')
      return
    }

    if (!result || !result.parameters) {
      setError('Please generate parameters first')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      // Process audio directly in browser using Sony's Web Audio API
      const processedBlob = await processAudioInBrowser(audioFile, result.parameters)

      // Create download URL
      const url = URL.createObjectURL(processedBlob)

      setProcessedAudio({
        success: true,
        processed_file: `processed_${audioFile.name}`,
        download_url: url,
        blob: processedBlob
      })
      setError(null)
    } catch (err) {
      setError('Processing failed: ' + err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const processAudioInBrowser = async (file, parameters) => {
    return new Promise(async (resolve, reject) => {
      try {
        // Create audio context
        const audioContext = new (window.AudioContext || window.webkitAudioContext)()

        // Read audio file
        const arrayBuffer = await file.arrayBuffer()
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

        // Create offline context for processing
        const offlineContext = new OfflineAudioContext(
          audioBuffer.numberOfChannels,
          audioBuffer.length,
          audioBuffer.sampleRate
        )

        // Create source
        const source = offlineContext.createBufferSource()
        source.buffer = audioBuffer

        // === 1. EQ (Sony Equalizer) ===
        // Convert webapp EQ params to Sony 40-band format
        const eqCurve = convertEQToSonyFormat(parameters.eq)
        const equalizer = new window.Equalizer(offlineContext, {
          curve: eqCurve,
          range: 1
        })

        // === 2. Compressor (Web Audio API native) ===
        const compressor = offlineContext.createDynamicsCompressor()
        compressor.threshold.value = parameters.compressor.threshold
        compressor.ratio.value = parameters.compressor.ratio
        compressor.attack.value = parameters.compressor.attack
        compressor.release.value = parameters.compressor.release
        // Note: makeup_gain not directly supported, would need additional gain node

        // === 3. Reverb (Sony Reverb) ===
        const reverb = new window.Reverb(offlineContext, {
          d: parameters.reverb.delay_time,
          g: parameters.reverb.decay,
          m: parameters.reverb.stereo_spread,
          f: parameters.reverb.cutoff_freq,
          E: 0.5,  // wet_gain
          wetdry: parameters.reverb.wet_dry
        })

        // Connect chain: source → EQ → Compressor → Reverb → destination
        source.connect(equalizer.input)
        equalizer.connect(compressor)
        compressor.connect(reverb.input)
        reverb.connect(offlineContext.destination)

        // Process
        source.start(0)
        const renderedBuffer = await offlineContext.startRendering()

        // Convert to WAV blob
        const wavBlob = audioBufferToWav(renderedBuffer)
        resolve(wavBlob)

      } catch (error) {
        reject(error)
      }
    })
  }

  const convertEQToSonyFormat = (eqParams) => {
    // Sony's 40 fixed frequencies
    const sonyFreqs = [
      20, 50, 83, 120, 161, 208, 259, 318, 383, 455, 537, 628, 729, 843,
      971, 1114, 1273, 1452, 1652, 1875, 2126, 2406, 2719, 3070, 3462,
      3901, 4392, 4941, 5556, 6244, 7014, 7875, 8839, 9917, 11124, 12474,
      13984, 15675, 17566, 19682
    ]

    // Initialize with 0 dB (flat)
    const sonyGains = new Array(40).fill(0.0)

    // For each Sony band, calculate gain from webapp parametric bands
    sonyFreqs.forEach((freq, i) => {
      let totalGain = 0.0

      // Sum contributions from all parametric bands
      eqParams.forEach(band => {
        const fc = band.freq
        const gain = band.gain
        const Q = band.Q

        // Calculate bell filter response
        const octaveDistance = Math.abs(Math.log2(freq / fc))
        const bandwidth = 1 / Q

        // Apply gain if within bandwidth
        if (octaveDistance < bandwidth) {
          const attenuation = Math.exp(-Math.pow(octaveDistance / bandwidth, 2))
          totalGain += gain * attenuation
        }
      })

      sonyGains[i] = totalGain
    })

    return sonyGains
  }

  const audioBufferToWav = (audioBuffer) => {
    const numOfChannels = audioBuffer.numberOfChannels
    const length = audioBuffer.length * numOfChannels * 2
    const buffer = new ArrayBuffer(44 + length)
    const view = new DataView(buffer)

    // WAV header
    const channels = numOfChannels
    const sampleRate = audioBuffer.sampleRate
    const bitsPerSample = 16

    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    writeString(view, 0, 'RIFF')
    view.setUint32(4, 36 + length, true)
    writeString(view, 8, 'WAVE')
    writeString(view, 12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, channels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * channels * bitsPerSample / 8, true)
    view.setUint16(32, channels * bitsPerSample / 8, true)
    view.setUint16(34, bitsPerSample, true)
    writeString(view, 36, 'data')
    view.setUint32(40, length, true)

    // Write audio data
    let offset = 44
    for (let i = 0; i < audioBuffer.length; i++) {
      for (let channel = 0; channel < numOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(channel)[i]))
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
        offset += 2
      }
    }

    return new Blob([buffer], { type: 'audio/wav' })
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

        {/* Process Audio Button */}
        {audioFile && (
          <div className="process-section">
            <button
              className="process-btn"
              onClick={handleProcess}
              disabled={isProcessing}
            >
              {isProcessing ? 'Processing Audio...' : '🎵 Apply Effects to Audio'}
            </button>
          </div>
        )}

        {/* Audio Playback */}
        {processedAudio && (
          <div className="audio-playback">
            <h3>🎧 Audio Comparison</h3>
            <div className="audio-players">
              <div className="audio-player">
                <label>Original Audio</label>
                <audio controls src={URL.createObjectURL(audioFile)}>
                  Your browser does not support audio playback.
                </audio>
              </div>
              <div className="audio-player">
                <label>Processed Audio</label>
                <audio controls src={processedAudio.download_url}>
                  Your browser does not support audio playback.
                </audio>
              </div>
            </div>
            <a
              href={processedAudio.download_url}
              download={processedAudio.processed_file}
              className="download-btn"
            >
              ⬇️ Download Processed Audio
            </a>
          </div>
        )}

        {/* System Prompt Display */}
        {systemPrompt && (
          <div className="system-prompt-section">
            <div
              className="system-prompt-header"
              onClick={() => setShowSystemPrompt(!showSystemPrompt)}
            >
              <h3>System Prompt Used</h3>
              <span className="system-prompt-toggle">
                {showSystemPrompt ? '▼ Hide' : '▶ Show'}
              </span>
            </div>
            {showSystemPrompt && (
              <div className="system-prompt-content">
                {systemPrompt}
              </div>
            )}
          </div>
        )}

        {/* JSON Export */}
        <details className="json-export">
          <summary>View Full JSON</summary>
          <pre>{JSON.stringify(result.parameters, null, 2)}</pre>
        </details>
      </div>
    )
  }

  // Calculate flow diagram stage
  const getFlowStage = () => {
    if (processedAudio) return 5
    if (audioFile && result) return 4
    if (result) return 3
    if (isGenerating) return 2
    if (text) return 1
    return 0
  }

  return (
    <div className="app">
      <header>
        <h1>Text2Preset</h1>
        <p>LLM-powered audio effect parameter generation</p>
      </header>

      <FlowDiagramNew
        stage={getFlowStage()}
        userInput={text}
        hasAudio={!!audioFile}
        parameters={result?.parameters}
        processedAudio={processedAudio}
        onTextChange={setText}
        onAudioChange={handleFileChange}
        onGenerate={handleGenerate}
        onProcess={handleProcess}
        isGenerating={isGenerating}
        isProcessing={isProcessing}
        audioFileName={audioFile?.name}
        systemPrompt={systemPrompt}
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <main>
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
