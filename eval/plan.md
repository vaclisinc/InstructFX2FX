# InstructFX2FX Evaluation Pipeline Plan

## What Already Exists (DO NOT rewrite these)

| Component | Location | What it does |
|---|---|---|
| `run_InstructFX2FX()` | `src/experimentation/experiment.py` | OURS system: LLM init → DASP grad descent, saves trajectory to `optimization_steps/` |
| `run_LLM_LLM()` | `src/experimentation/experiment.py` | Baseline: two LLM calls, no optimization |
| `run_experiments()` | `src/experimentation/experiment.py` | Batch runner over audio files for multiple methods |
| `extract_dsp_feature()` | `src/metrics/__init__.py` | 35-dim DSP features from audio path or array |
| `cal_mmd_score()` | `src/metrics/__init__.py` | MMD between two feature matrices |
| `apply_eq.js` | `src/metrics/apply_eq/apply_eq.js` | Audealize 40-band EQ via Node.js subprocess — the GT renderer |
| `warm_gt/` | `src/metrics/apply_eq/warm_gt/` | 74 "warm" piano GT audio files (already rendered) |

**Key insight from `experiment.py`:**
- `run_InstructFX2FX()` saves intermediate trajectory audio to `{run_dir}/intermediate/optimization_steps/` as `start.wav`, `iter_N.wav`, ..., `end.wav`
- `run_LLM_LLM()` saves `01_initialized.wav` and `02_refined.wav` to `{run_dir}/intermediate/`
- Both save final output to `{run_dir}/{stem}_final_output.wav`

So the system pipeline **already writes audio to disk**. The eval runner just needs to drive it, read the files, and compute metrics.

---

## What Needs to Be Built

```
eval/
  plan.md                    ← this file
  config.py                  ← word pairs, instruments, paths, hyperparams
  data/
    socialfx_loader.py       ← load GT param lists from HuggingFace by word
    get_gt.py                ← Python wrapper calling apply_eq.js via subprocess
  gt_bank.py                 ← pre-render & cache all GT audio to disk
  run_system.py              ← drive run_experiments() for all pairs × instruments
  exp1_sequential.py         ← load system outputs + sequential GT → MMD per instrument
  exp2_trajectory.py         ← load trajectory steps + word GT dists → MMD curves
  report.py                  ← save JSON + matplotlib plots
  run.py                     ← CLI entry point
```

---

## File-by-File Function Plan

---

### `eval/config.py`

Constants only, no functions.

```python
WORD_PAIRS: List[Tuple[str, str]]
# 20 directed pairs from our 7 EQ words (warm, bright, soft, loud, harsh, calm, heavy)
# e.g. [("warm", "bright"), ("harsh", "soft"), ("calm", "loud"), ...]

INSTRUMENTS: List[str] = ["guitar", "drums", "piano"]

DRY_AUDIO_DIR: str
# root dir with subfolders per instrument, e.g. dry_audio/guitar/*.wav

GT_BANK_DIR: str
# cache dir for pre-rendered GT audio, e.g. eval/gt_cache/

SYSTEM_RESULTS_DIR: str
# where run_experiments() saves system outputs, e.g. eval/system_outputs/

RESULTS_DIR: str
# final eval results (JSON + plots), e.g. eval/results/

APPLY_EQ_JS_PATH: str
# absolute path to src/metrics/apply_eq/apply_eq.js

FX_TYPE: str = "eq"

SAMPLE_RATE: int = 44100

# System hyperparams (passed to run_experiments)
N_GRAD_ITER: int = 1000
SAVE_INTERVAL: int = 50     # snapshot_interval for trajectory (Exp 2)
NR_RUNS_PER_FILE: int = 1   # nr_of_experiments_per_file in run_experiments
```

---

### `eval/data/socialfx_loader.py`

Loads GT parameter lists from HuggingFace. Wraps HF datasets API.

```python
def load_word_ids(word: str, fx_type: str = "eq") -> List[str]:
    """
    Get all sample IDs for a word from seungheondoh/socialfx-gen-eval.

    Input:
        word: str       e.g. "warm"
        fx_type: str    "eq" or "reverb"

    Output:
        ids: List[str]  e.g. ["eq_0", "eq_6", "eq_10", ...]
    """

def load_params_by_id(sample_id: str) -> List[float]:
    """
    Fetch the 40-value EQ parameter array for one sample from
    seungheondoh/socialfx-original.

    Input:
        sample_id: str   e.g. "eq_0"

    Output:
        params: List[float]   length 40  (raw Audealize EQ curve values)
    """

def load_all_params_for_word(word: str, fx_type: str = "eq") -> List[List[float]]:
    """
    Load all available EQ parameter arrays for a word.
    Calls load_word_ids → load_params_by_id for each ID.

    Input:
        word: str
        fx_type: str

    Output:
        params_list: List[List[float]]   e.g. 74 arrays for "warm", each length 40
    """
```

---

### `eval/data/get_gt.py`

Python wrapper that calls `apply_eq.js` via subprocess.
This IS the `get_gt(fx_type, fx_parameters, raw_audio)` function the team described.

```python
def get_gt(
    fx_type: str,
    params: List[float],
    dry_audio: np.ndarray,
    sr: int = 44100,
    apply_eq_js_path: str = APPLY_EQ_JS_PATH,
) -> np.ndarray:
    """
    Apply Audealize FX to dry audio by calling apply_eq.js via subprocess.
    Writes temp WAV files, calls Node.js, reads back output.

    Input:
        fx_type: str            "eq" (reverb not yet supported in apply_eq.js)
        params: List[float]     length 40, raw Audealize EQ curve
        dry_audio: np.ndarray   shape (T,) or (1, T), float32, mono
        sr: int                 sample rate (44100)
        apply_eq_js_path: str   path to apply_eq.js

    Output:
        processed: np.ndarray   shape (T,), float32  — wet audio

    Notes:
        Uses tempfile for input/output WAVs and params JSON.
        Cleans up temp files after subprocess call.
    """
```

---

### `eval/gt_bank.py`

Pre-renders and caches all GT audio needed for both experiments.
Must be run once before exp1/exp2 (or when `--rebuild_gt` flag is passed).

```python
def build_word_gt_bank(
    word: str,
    instrument: str,
    dry_paths: List[str],
    params_list: List[List[float]],
    cache_dir: str,
    sr: int = 44100,
) -> List[str]:
    """
    Render GT audio for a single word: every param set × every dry clip.
    Used for Exp 2 reference distributions (word_A_dist and word_B_dist).

    Input:
        word: str
        instrument: str
        dry_paths: List[str]            dry audio file paths for this instrument
        params_list: List[List[float]]  all param arrays for this word (N arrays)
        cache_dir: str
        sr: int

    Output:
        audio_paths: List[str]
            length = N * len(dry_paths)
            cached at: cache_dir/{word}/{instrument}/{dry_idx}_{param_idx}.wav

    Skips files that already exist (idempotent).
    """

def build_sequential_gt_bank(
    word_A: str,
    word_B: str,
    instrument: str,
    dry_paths: List[str],
    params_A: List[List[float]],
    params_B: List[List[float]],
    cache_dir: str,
    sr: int = 44100,
) -> List[str]:
    """
    Render sequential GT for Exp 1: all combinations (params_A[i], params_B[j])
    applied sequentially to each dry clip.

    For each dry_path, for each (i, j):
        audio_A = get_gt("eq", params_A[i], dry)
        audio_AB = get_gt("eq", params_B[j], audio_A)   ← sequential!

    Input:
        word_A, word_B: str
        instrument: str
        dry_paths: List[str]
        params_A: List[List[float]]     N_A arrays
        params_B: List[List[float]]     N_B arrays
        cache_dir: str
        sr: int

    Output:
        audio_paths: List[str]
            length = len(dry_paths) * N_A * N_B
            cached at: cache_dir/{word_A}_to_{word_B}/{instrument}/{dry_idx}_{i}_{j}.wav

    Skips files that already exist (idempotent).
    """
```

---

### `eval/run_system.py`

Drives `run_experiments()` for every (word_A, word_B) pair × every instrument.
Organizes outputs on disk so exp1/exp2 runners can find them.

```python
def run_system_for_pair(
    word_A: str,
    word_B: str,
    instrument: str,
    dry_paths: List[str],
    method: Method,
    llm_client: LLMClient,
    clap: CLAPWrapper,
    fx_chain,
    output_dir: str,
    n_iter: int = N_GRAD_ITER,
    save_interval: int = SAVE_INTERVAL,
    nr_runs: int = NR_RUNS_PER_FILE,
    device: str = "cpu",
) -> str:
    """
    Run one (word_A → word_B) pair on all dry clips for one instrument.
    Calls run_experiments() internally, which saves all audio to disk.

    Input:
        word_A: str             initial prompt word (anchor)
        word_B: str             refinement target word
        instrument: str
        dry_paths: List[str]    dry audio paths for this instrument
        method: Method          Method.InstructFX2FX or Method.LLM_LLM
        llm_client, clap, fx_chain: model objects
        output_dir: str         base directory for this pair's outputs
        n_iter, save_interval, nr_runs: hyperparams
        device: str

    Output:
        pair_run_dir: str       path where run_experiments() saved outputs
                                structure: output_dir/{word_A}_to_{word_B}/{instrument}/

    Notes:
        InstructionSet is built from (word_A, word_B):
            instructionset_initialization: anchor=word_A, target=word_A (LLM generates initial params)
            instructionset_refinement:     anchor=word_A, target=word_B (gradient descent direction)
        snapshot_interval in Config is set to save_interval for Exp 2 trajectory.
    """

def collect_final_audio_paths(pair_run_dir: str, method: Method) -> List[str]:
    """
    Find all final output WAV files from a run_experiments() output directory.

    Input:
        pair_run_dir: str   directory returned by run_system_for_pair
        method: Method

    Output:
        paths: List[str]    one path per dry clip per run
                            e.g. [.../piano_final_output.wav, ...]
    """

def collect_trajectory_audio_paths(pair_run_dir: str) -> Dict[str, List[str]]:
    """
    Find all trajectory snapshot WAVs from optimization_steps/ directories.
    Only relevant for InstructFX2FX (gradient descent saves snapshots).

    Input:
        pair_run_dir: str

    Output:
        trajectory: Dict[str, List[str]]
            key = snapshot label (e.g. "start", "iter_50", "iter_100", "end")
            value = list of WAV paths (one per dry clip)
            ordered by iteration step
    """
```

---

### `eval/exp1_sequential.py`

Compares final system output distribution against sequential GT distribution.

```python
def run_exp1(
    method: Method,
    method_name: str,
    pairs: List[Tuple[str, str]],
    instruments: List[str],
    dry_paths_by_instrument: Dict[str, List[str]],
    gt_bank_dir: str,
    system_results_dir: str,
    results_dir: str,
    sr: int = 44100,
) -> Dict:
    """
    For each (word_A, word_B) pair, each instrument:
        1. Load sequential GT audio paths from gt_bank_dir (pre-rendered)
        2. Load system final output audio paths from system_results_dir
        3. Extract DSP features for both sets
        4. Compute MMD per instrument → average

    Input:
        method: Method
        method_name: str                    "ours" or "baseline"
        pairs: List[Tuple[str, str]]
        instruments: List[str]
        dry_paths_by_instrument: Dict[str, List[str]]
        gt_bank_dir: str
        system_results_dir: str
        results_dir: str
        sr: int

    Output:
        results: Dict
        {
            "method": method_name,
            "pairs": {
                "warm_to_bright": {
                    "guitar": float,
                    "drums": float,
                    "piano": float,
                    "avg": float
                },
                ...
            },
            "overall_avg": float
        }
    """

def _compute_mmd_for_pair_instrument(
    system_audio_paths: List[str],
    gt_audio_paths: List[str],
    sr: int,
) -> float:
    """
    Extract DSP features for both lists and compute MMD.
    Wraps extract_dsp_feature() and cal_mmd_score() from src/metrics.

    Input:
        system_audio_paths: List[str]
        gt_audio_paths: List[str]
        sr: int

    Output:
        mmd: float
    """
```

---

### `eval/exp2_trajectory.py`

Tracks MMD vs iteration step to show refinement moves toward word_B and away from word_A.

```python
def run_exp2(
    pairs: List[Tuple[str, str]],
    instruments: List[str],
    dry_paths_by_instrument: Dict[str, List[str]],
    gt_bank_dir: str,
    system_results_dir: str,
    results_dir: str,
    sr: int = 44100,
) -> Dict:
    """
    For each (word_A, word_B) pair:
        1. Load word_A_dist: GT audio paths for word_A (from gt_bank, NOT sequential)
        2. Load word_B_dist: GT audio paths for word_B (from gt_bank, NOT sequential)
        3. Load trajectory snapshots from system_results_dir (optimization_steps/)
           using collect_trajectory_audio_paths()
        4. At each step k:
               pool all dry clips' audio_at_k → pred_dist_at_k
               mmd_A_at_k = _compute_mmd_for_pair_instrument(pred_dist_at_k, word_A_dist)
               mmd_B_at_k = _compute_mmd_for_pair_instrument(pred_dist_at_k, word_B_dist)
        (MMD computed across all instruments pooled, or per-instrument then averaged)

    Input:
        pairs: List[Tuple[str, str]]
        instruments: List[str]
        dry_paths_by_instrument: Dict[str, List[str]]
        gt_bank_dir: str
        system_results_dir: str
        results_dir: str
        sr: int

    Output:
        results: Dict
        {
            "pairs": {
                "warm_to_bright": {
                    "steps": ["start", "iter_50", "iter_100", ..., "end"],
                    "mmd_toward_A": [float, ...],    # expect increasing
                    "mmd_toward_B": [float, ...]     # expect decreasing
                },
                ...
            }
        }

    Notes:
        Only InstructFX2FX produces a full trajectory.
        Baseline (LLM+LLM) only has two points: step 0 and final.
    """
```

---

### `eval/report.py`

```python
def save_exp1_table(results: Dict, output_path: str) -> None:
    """
    Save Exp 1 results as JSON and print a table
    (rows = pairs, cols = instruments + avg, grouped by method).

    Input:  results dict from run_exp1 (one or both methods)
    Output: writes {output_path}.json and {output_path}.txt
    """

def plot_exp2_trajectory(results: Dict, output_path: str) -> None:
    """
    For each word pair: one plot with two lines:
        - mmd_toward_A over steps (should increase)
        - mmd_toward_B over steps (should decrease)

    Input:  results dict from run_exp2
    Output: one PNG per pair at {output_path}/{pair_name}.png
    """
```

---

### `eval/run.py`

```python
def main(
    exp: str = "all",          # "1", "2", or "all"
    method: str = "all",       # "ours", "baseline", or "all"
    rebuild_gt: bool = False,  # force re-render GT bank
    rebuild_system: bool = False,  # force re-run system outputs
) -> None:
    """
    CLI entry point.

    Steps:
        1. Load config, dry audio paths per instrument
        2. Load all GT params from HuggingFace (socialfx_loader)
        3. Build GT bank if missing or rebuild_gt (gt_bank.py)
        4. Run system for all pairs × instruments if missing or rebuild_system (run_system.py)
        5. Run selected experiments (exp1, exp2, or both)
        6. Save results and plots (report.py)

    Usage:
        python eval/run.py
        python eval/run.py --exp 1 --method ours
        python eval/run.py --rebuild_gt --rebuild_system
    """
```

---

## Data Flow Summary

```
GT Bank (build once, cache to disk):

  HuggingFace socialfx-gen-eval
      ↓ load_all_params_for_word(word)
  params_list [N × 40 floats]
      ↓ get_gt() via apply_eq.js subprocess  ← Audealize renderer
  GT audio files cached to disk

  Exp 2 ref: word_X_dist = {instrument: [paths...]}
             built by build_word_gt_bank()  (N params × n_dry clips)

  Exp 1 GT:  sequential_dist = {instrument: [paths...]}
             built by build_sequential_gt_bank()  (N_A × N_B × n_dry clips)


System Side (run once per pair × instrument, cache to disk):

  dry audio paths
      ↓ run_system_for_pair() → run_experiments() (existing)
  Saves to disk:
    {pair_dir}/intermediate/optimization_steps/start.wav, iter_50.wav, ..., end.wav
    {pair_dir}/*_final_output.wav


Exp 1:

  collect_final_audio_paths() → system_paths
  sequential GT paths from gt_bank_dir
      ↓ _compute_mmd_for_pair_instrument()
        extract_dsp_feature() + cal_mmd_score()  ← existing src/metrics
  MMD per instrument → avg → results dict → save_exp1_table()


Exp 2:

  collect_trajectory_audio_paths() → {step: [paths]}
  word_A_dist paths, word_B_dist paths from gt_bank_dir
  At each step k:
      ↓ _compute_mmd_for_pair_instrument(pred@k, word_A_dist)  → mmd_A[k]
      ↓ _compute_mmd_for_pair_instrument(pred@k, word_B_dist)  → mmd_B[k]
  → plot_exp2_trajectory()
```

---

## Open Items Before Coding

| Item | Status | Notes |
|---|---|---|
| `socialfx_gt.py` | Not found — may not exist yet | `generate_eq_gt.py` imports `get_gt` from it; need to verify or create |
| Final 20 word pairs | Not decided | Need to choose from our 7 EQ words |
| Dry audio dataset | Not decided | MUSDB18-HQ or record our own |
| HF field names for EQ params in socialfx-original | Not verified | Need to check exact schema |
| `run_sample` (milan's function) | Not found as standalone — likely `run_experiments()` | Confirm with milan |
