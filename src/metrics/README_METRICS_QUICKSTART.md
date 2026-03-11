## Metrics quickstart (for the team)

Very short guide to what we actually use.

---

### 1. EQ ground truth via SocialFX (`apply_eq/`)

- **Goal**: Reproduce SocialFX / Audealize EQ processing for specific parameter sets (ids like `eq_7`, `eq_170`) and use those as **ground‑truth audio**.
- **Files**:
  - `apply_eq/apply_eq.js`: Node.js implementation of the 40‑band Audealize EQ (40 peaking filters, Q=4.31, ±5 dB range).
  - `apply_eq/generate_eq_gt.py`: Python script to load the SocialFX EQ parquet from HuggingFace and apply EQ curves to an input wav.
  - `apply_eq/warm_gt/…` or `apply_eq/result_wav/…`: output WAVs with EQ applied.

**How it works (conceptually)**:

1. `generate_eq_gt.py` uses `pandas` + `hf://datasets/seungheondoh/socialfx-original/data/eq-00000-of-00001.parquet` to get rows with:
   - `id` (e.g. `"eq_7"`)
   - `text` (word like `"warm"`)
   - `param_values` (40‑dim EQ curve)
2. For each selected row, it writes:
   - A temp JSON file with the 40 numbers.
   - A temp WAV for the input audio (e.g. `piano.wav`).
3. It calls:

```bash
node apply_eq.js <input.wav> <output.wav> <params.json>
```

4. `apply_eq.js` runs the exact Audealize 40‑band graphic EQ over the audio and writes `<output.wav>`.

So for a list of ids for `"warm"`, you get many files like:

- `apply_eq/warm_gt/piano_eq_0.wav`
- `apply_eq/warm_gt/piano_eq_6.wav`
- …

These are your **EQ ground‑truth examples** per parameter set.

---

### 1b. Reverb ground truth via SocialFX (`apply_eq/`)

- **Goal**: Same idea as EQ, but for the SocialFX **reverb** split: 40‑param reverb curves applied to an input wav as ground truth.
- **Files**:
  - `apply_eq/apply_reverb.js`: Builds a noise‑shaped reverb IR from the 40‑band curve (same FREQS/Q as EQ), convolves input with it, wet/dry mix.
  - `apply_eq/generate_reverb_gt.py`: Loads the SocialFX reverb parquet and, for each selected word’s parameter set, calls the JS script to produce WAVs in e.g. `reverb_gt/`.

**How to run**:

```bash
cd src/metrics/apply_eq
python generate_reverb_gt.py
```

Edit the config at the top of `generate_reverb_gt.py` to set `INPUT_WAV`, `OUTPUT_DIR`, and `WORD`. Outputs are one WAV per row (e.g. `reverb_0.wav`, `reverb_42.wav`).

---

### 2. DSP features (`extract_dsp_feature`)

- **File**: `dsp_features.py`
- **Function**:

```python
from src.metrics import extract_dsp_feature

features = extract_dsp_feature("data/audio/piano.wav")
# → list of 35 floats (spectral, MFCC, RMS, ZCR, crest factor, brightness, loudness)
```

- You can also pass a numpy array + `sr` if you already have audio in memory.
- Features are the same 35‑D definition as in the LLM2Fx paper, so they’re compatible with MMD and other DSP‑based metrics.

**Test script**:

```bash
cd src/metrics
python test_feature.py
```

This prints the 35‑D feature list for `data/audio/piano.wav`.

---

### 3. MMD over DSP features (`cal_mmd_score`)

- **File**: `mmd_metric.py`
- **Function**:

```python
from src.metrics import cal_mmd_score
import numpy as np

matrix_gt = np.array([extract_dsp_feature(f) for f in gt_files])     # (n, 35)
matrix_ours = np.array([extract_dsp_feature(f) for f in pred_files]) # (m, 35)
score = cal_mmd_score(matrix_gt, matrix_ours)  # float, lower = closer
```

- Internally this uses a Gaussian‑kernel Maximum Mean Discrepancy (MMD) between the two 35‑D feature distributions.
- **Interpretation**: lower MMD → our outputs’ DSP features are closer to ground truth.

**Test script**:

```bash
cd src/metrics
python test_mmd.py
```

This uses two hard‑coded 3×35 matrices and prints a non‑zero MMD just to verify wiring.

---

### 4. What to run in practice

- **To generate EQ ground truth** for a word like `"warm"`:

```bash
cd src/metrics/apply_eq
python generate_eq_gt.py
```

This reads `piano.wav`, finds all SocialFX EQ rows for `"warm"`, and writes ground‑truth WAVs into the configured output folder (`result_wav` or `warm_gt`).

- **To compute features for any audio**:

```python
from src.metrics import extract_dsp_feature
features = extract_dsp_feature("path/to/audio.wav")
```

- **To compare GT vs ours with MMD**:

```python
from src.metrics import extract_dsp_feature, cal_mmd_score
import numpy as np

gt_files = [...]      # list of ground-truth wavs (e.g. from apply_eq outputs)
pred_files = [...]    # list of our model’s outputs

X = np.array([extract_dsp_feature(f) for f in gt_files])
Y = np.array([extract_dsp_feature(f) for f in pred_files])
score = cal_mmd_score(X, Y)
print("MMD:", score)
```

That’s all most people on the team need to know: **EQ ground truth via `apply_eq`**, **DSP features via `extract_dsp_feature`**, and **distribution comparison via `cal_mmd_score`**.

