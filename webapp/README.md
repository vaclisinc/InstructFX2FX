# Text2Preset Web Application

LLM-powered audio effect parameter generation web interface.

## Features

- **Text-to-Parameters**: Describe audio effects in natural language
- **Multi-LLM Support**: Choose between OpenRouter, OpenAI, or Anthropic Claude
- **Real-time Generation**: See parameters generated instantly
- **Audio Upload**: Optional audio file upload for future processing
- **Interactive UI**: Tab-based parameter display for Reverb, EQ, and Compressor
- **JSON Export**: Download generated parameters for use in audio processing

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- At least one API key configured:
  - `OPENROUTER_API_KEY`
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`

### Installation

1. **Setup environment variables**:
   ```bash
   cd ../baseline-system
   cp .env.example .env
   # Edit .env and add your API keys
   ```

2. **Install backend dependencies**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

**Option 1: Separate terminals (recommended for development)**

Terminal 1 - Backend:
```bash
cd webapp/backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

Terminal 2 - Frontend:
```bash
cd webapp/frontend
npm run dev
```

**Option 2: Use the startup script**

```bash
cd webapp
./start.sh  # On Windows: start.bat
```

### Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Usage

1. Enter a text description of the audio effect you want (e.g., "warm cathedral reverb with bright EQ")
2. Select your preferred LLM model from the dropdown
3. (Optional) Upload a dry audio file for future processing
4. Click "Generate Parameters"
5. View the generated Reverb, EQ, and Compressor parameters
6. Export parameters as JSON if needed

## API Endpoints

### `GET /api/models`
Get list of available LLM models based on configured API keys.

### `POST /api/upload`
Upload audio file for processing.
- **Body**: multipart/form-data with `file` field
- **Returns**: File metadata

### `POST /api/generate`
Generate audio effect parameters from text description.
- **Body**:
  - `text`: Description string
  - `model_provider`: Provider name (openrouter/openai/claude)
  - `model_name`: Model identifier
  - `audio_filename`: (Optional) Uploaded audio filename
- **Returns**: Generated parameters JSON

### `GET /api/health`
Health check endpoint with system status.

## Project Structure

```
webapp/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── api/                 # API endpoints (future)
│   ├── uploads/             # Uploaded audio files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   └── App.css          # Styles
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Configuration

The backend automatically loads configuration from:
- `baseline-system/.env` - API keys
- `baseline-system/configs/default.yaml` - LLM settings and prompts

## Troubleshooting

### Backend won't start
- Check if baseline-system is accessible
- Verify at least one API key is configured
- Check Python path includes baseline-system

### Frontend can't connect to backend
- Verify backend is running on port 8000
- Check CORS settings in main.py
- Try accessing http://localhost:8000/api/health

### No models available
- Add API keys to `baseline-system/.env`
- Restart backend server
- Check `/api/models` endpoint response

### LLM generation fails
- Verify API key is valid
- Check API rate limits
- Review prompt templates in `baseline-system/prompts/`

## Development

### Hot Reload
- Frontend: Vite provides instant hot reload
- Backend: uvicorn runs with `reload=True`

### Testing API
Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

### Adding New Models
Edit `AVAILABLE_MODELS` dictionary in `backend/main.py`.

## Next Steps (Phase 2)

- [ ] Integrate fx-processor for actual audio processing
- [ ] Add audio playback with A/B comparison
- [ ] Implement parameter refinement loop with judge system
- [ ] Add real-time parameter editing
- [ ] WebSocket support for long-running generations
- [ ] Session management and history

## Demo Tips

For tomorrow's presentation:
1. Test with these example prompts:
   - "warm cathedral reverb with bright EQ"
   - "tight punchy compression with boosted bass"
   - "spacious hall reverb with balanced frequency response"
2. Have sample audio files ready (WAV format works best)
3. Pre-configure API keys before the demo
4. Test both OpenRouter (fast, cheap) and Claude Sonnet (high quality)

## License

Part of the text2preset research project.
