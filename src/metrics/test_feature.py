import sys
from pathlib import Path

# Allow running from src/metrics/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.metrics import extract_dsp_feature

INPUT_FILE = "/Users/yuxuancai/cnmat2026/text2preset-1/data/audio/piano.wav"

extract_dsp_feature(INPUT_FILE)
