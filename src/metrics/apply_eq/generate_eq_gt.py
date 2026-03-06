"""Apply a list of SocialFX EQ ids to piano.wav and save outputs.

Run from repo root:

    cd src/metrics/apply_eq
    python run_apply_eq_ids.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
from src.metrics.apply_eq.socialfx_gt import get_gt

AUDIO_PATH = Path("/Users/yuxuancai/cnmat2026/text2preset-1/data/audio/piano.wav")
OUT_DIR = Path(__file__).resolve().parent / "result_wav"
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw_audio, sr = sf.read(AUDIO_PATH)
if raw_audio.ndim > 1:
    raw_audio = raw_audio.mean(axis=1)

IDS = ['eq_0', 'eq_6', 'eq_10', 'eq_15', 'eq_21', 'eq_57', 'eq_66', 'eq_67', 'eq_73', 'eq_76', 'eq_78', 'eq_84', 'eq_86', 'eq_89', 'eq_99', 'eq_101', 'eq_123', 'eq_128', 'eq_131', 'eq_136', 'eq_138', 'eq_147', 'eq_170', 'eq_176', 'eq_178', 'eq_194', 'eq_207', 'eq_265', 'eq_267', 'eq_282', 'eq_295', 'eq_299', 'eq_344', 'eq_346', 'eq_351', 'eq_352', 'eq_353', 'eq_380', 'eq_395', 'eq_420', 'eq_421', 'eq_425', 'eq_430', 'eq_453', 'eq_455', 'eq_472', 'eq_479', 'eq_497', 'eq_503', 'eq_508', 'eq_514', 'eq_517', 'eq_521', 'eq_531', 'eq_547', 'eq_566', 'eq_576', 'eq_590', 'eq_610', 'eq_634', 'eq_645', 'eq_647', 'eq_651', 'eq_698', 'eq_711', 'eq_760', 'eq_793', 'eq_988', 'eq_1199', 'eq_1334', 'eq_1360', 'eq_1434', 'eq_1444', 'eq_1579']

for fx_id in IDS:
    print(f"Processing {fx_id}...")
    processed = get_gt("eq", fx_id, raw_audio, sr)
    out_path = OUT_DIR / f"piano_{fx_id}.wav"
    sf.write(out_path, processed, sr)
    print(f"  → saved {out_path}")