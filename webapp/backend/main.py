"""
FastAPI backend for text2preset web application.
Provides REST API endpoints for audio effect parameter generation.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
import os
import shutil
from typing import Optional

# Add baseline-system to Python path
baseline_path = Path(__file__).parent.parent.parent / "baseline-system"
sys.path.insert(0, str(baseline_path))

from src.config.loader import load_config, load_env
# Use simplified wrapper to avoid CLAP/torch dependencies for demo
from generation_wrapper import generate_parameters

# Initialize FastAPI app
app = FastAPI(
    title="Text2Preset API",
    description="LLM-powered audio effect parameter generation",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Load environment and config
# Load .env from baseline-system directory
env_path = baseline_path / ".env"
load_env(str(env_path))
DEFAULT_CONFIG_PATH = baseline_path / "configs" / "default.yaml"

# Available LLM models
AVAILABLE_MODELS = {
    "openrouter": {
        "claude-haiku": {
            "name": "Claude 3.5 Haiku",
            "provider": "openrouter",
            "model": "anthropic/claude-3.5-haiku",
            "cost": "Low",
            "speed": "Fast"
        },
        "claude-sonnet": {
            "name": "Claude 3.5 Sonnet",
            "provider": "openrouter",
            "model": "anthropic/claude-3.5-sonnet",
            "cost": "Medium",
            "speed": "Medium"
        },
    },
    "openai": {
        "gpt-4": {
            "name": "GPT-4",
            "provider": "openai",
            "model": "gpt-4",
            "cost": "High",
            "speed": "Medium"
        },
        "gpt-4-turbo": {
            "name": "GPT-4 Turbo",
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "cost": "High",
            "speed": "Fast"
        },
    },
    "claude": {
        "claude-sonnet": {
            "name": "Claude 3.5 Sonnet (Direct)",
            "provider": "claude",
            "model": "claude-3-5-sonnet-20241022",
            "cost": "Medium",
            "speed": "Fast"
        },
    }
}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Text2Preset API is running",
        "version": "1.0.0"
    }


@app.get("/api/models")
async def get_models():
    """Get list of available LLM models."""
    # Check which API keys are available
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

    available = {}
    if has_openrouter:
        available["openrouter"] = AVAILABLE_MODELS["openrouter"]
    if has_openai:
        available["openai"] = AVAILABLE_MODELS["openai"]
    if has_claude:
        available["claude"] = AVAILABLE_MODELS["claude"]

    if not available:
        return {
            "error": "No API keys configured",
            "message": "Please set at least one of: OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        }

    return {
        "models": available,
        "available_keys": {
            "openrouter": has_openrouter,
            "openai": has_openai,
            "claude": has_claude
        }
    }


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload audio file for processing.
    Accepts: WAV, MP3, FLAC, OGG
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save file
    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    return {
        "filename": file.filename,
        "path": str(file_path),
        "size": file_path.stat().st_size,
        "format": file_ext[1:]
    }


@app.post("/api/generate")
async def generate_audio_parameters(
    text: str = Form(...),
    model_provider: str = Form("openrouter"),
    model_name: str = Form("anthropic/claude-3.5-haiku"),
    audio_filename: Optional[str] = Form(None)
):
    """
    Generate audio effect parameters from text description.

    Args:
        text: User's text description (e.g., "warm and spacious")
        model_provider: LLM provider (openrouter, openai, claude)
        model_name: Specific model identifier
        audio_filename: Optional uploaded audio filename (not used in Phase 1)

    Returns:
        JSON with reverb, EQ, and compressor parameters
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text description is required"
        )

    try:
        # Load config and override with selected model
        config = load_config(str(DEFAULT_CONFIG_PATH))
        config["llm"]["provider"] = model_provider
        config["llm"]["model"] = model_name

        # Generate parameters
        params = generate_parameters(text.strip(), config)

        return {
            "success": True,
            "text": text,
            "model": {
                "provider": model_provider,
                "name": model_name
            },
            "parameters": params,
            "audio_filename": audio_filename
        }

    except Exception as e:
        # Log error details
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating parameters: {error_details}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate parameters: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Detailed health check with system status."""
    # Check if config file exists
    config_exists = DEFAULT_CONFIG_PATH.exists()

    # Check API keys
    api_keys = {
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY"))
    }

    # Check baseline system
    baseline_exists = baseline_path.exists()

    return {
        "status": "healthy" if any(api_keys.values()) else "degraded",
        "config_file": str(DEFAULT_CONFIG_PATH) if config_exists else "missing",
        "baseline_system": "found" if baseline_exists else "missing",
        "api_keys": api_keys,
        "upload_dir": str(UPLOAD_DIR),
        "python_path": sys.path[0]
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Text2Preset API server...")
    print(f"📁 Baseline system: {baseline_path}")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"🔑 API keys configured: {sum([bool(os.getenv(k)) for k in ['OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']])}/3")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
