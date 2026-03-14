# Eval Pipeline — Phased Implementation Plan

Derived from `plan.md`. Each phase is independently testable before moving on.

---

## Phase 0 — Smoke-test Existing Infrastructure

**Goal:** Confirm all upstream dependencies are callable without modification.

| Check | Command / Call | Pass Criterion |
|---|---|---|
| `apply_eq.js` | `node src/metrics/apply_eq/apply_eq.js --help` or manually run with a sample WAV | Exits 0, produces output WAV |
| `extract_dsp_feature` | `from src.metrics import extract_dsp_feature; extract_dsp_feature("some.wav")` | Returns list of 35 floats |
| `cal_mmd_score` | Pass two random (N×35) numpy arrays | Returns single float |
| `run_experiments()` | Call with one dry clip, one method, 5 iterations | Creates `*_final_output.wav` on disk |
| HuggingFace access | `from datasets import load_dataset; load_dataset("seungheondoh/socialfx-gen-eval", split="train")` | Downloads without error |

**Exit gate:** All 5 checks pass. No code written yet.

---

## Phase 1 — Config + Data Layer

**Files:** `eval/config.py`, `eval/data/__init__.py`, `eval/data/socialfx_loader.py`, `eval/data/get_gt.py`

### `eval/config.py`
- All constants defined: `WORD_PAIRS`, `INSTRUMENTS`, path roots, hyperparams
- `WORD_PAIRS`: 20 directed pairs from the 7 final EQ words
  _(warm, bright, soft, loud, harsh, calm, heavy — see `data/socialfx/final_eq_words.json`)_

### `eval/data/socialfx_loader.py`
Three functions: `load_word_ids`, `load_params_by_id`, `load_all_params_for_word`

### `eval/data/get_gt.py`
One function: `get_gt(fx_type, params, dry_audio, sr, apply_eq_js_path) → np.ndarray`
Uses `tempfile` for WAV + JSON I/O, calls `node apply_eq.js` via `subprocess`, cleans up.

**Pass criteria:**

```python
# P1-A: loader returns correct types
ids = load_word_ids("warm", "eq")
assert isinstance(ids, list) and len(ids) > 0
params = load_params_by_id(ids[0])
assert len(params) == 40

# P1-B: load_all_params_for_word returns N×40
all_params = load_all_params_for_word("warm")
assert all(len(p) == 40 for p in all_params)
# "warm" should have 74 entries (consistent with data/socialfx/ stats)

# P1-C: get_gt returns audio array
import numpy as np
dry = np.zeros(44100, dtype=np.float32)
wet = get_gt("eq", all_params[0], dry)
assert isinstance(wet, np.ndarray) and wet.shape[0] > 0
```

**Exit gate:** All 3 assertions pass on real HuggingFace data.

---

## Phase 2 — GT Bank

**File:** `eval/gt_bank.py`

Two functions: `build_word_gt_bank`, `build_sequential_gt_bank`

Both are idempotent (skip existing files) and return list of cached WAV paths.

**Pass criteria:**

```python
# P2-A: word GT bank
paths = build_word_gt_bank(
    word="warm", instrument="piano",
    dry_paths=["<one piano wav>"],
    params_list=warm_params[:3],   # 3 param sets
    cache_dir=GT_BANK_DIR,
)
assert len(paths) == 3            # 3 params × 1 dry clip
assert all(os.path.exists(p) for p in paths)

# P2-B: idempotent — re-running produces same paths, no re-renders
paths2 = build_word_gt_bank(...)
assert paths == paths2

# P2-C: sequential GT bank
seq_paths = build_sequential_gt_bank(
    word_A="warm", word_B="bright", instrument="piano",
    dry_paths=["<one piano wav>"],
    params_A=warm_params[:2], params_B=bright_params[:2],
    cache_dir=GT_BANK_DIR,
)
assert len(seq_paths) == 4        # 1 dry × 2 × 2 = 4
assert all(os.path.exists(p) for p in seq_paths)
```

**Exit gate:** All 3 pass. Cached WAVs are non-silent (rms > 0).

---

## Phase 3 — System Runner

**File:** `eval/run_system.py`

Three functions: `run_system_for_pair`, `collect_final_audio_paths`, `collect_trajectory_audio_paths`

**InstructionSet pattern** (mirrors `src/analysis.ipynb`):
```python
# init:   no anchor — "This is violin music. Make this sound more bright."
InstructionSet1(target=word_B, context=f"{instrument} music")
# refine: anchor=word_A — "This is violin music, but the sound is warm. Make this sound more bright."
InstructionSet1(anchor=word_A, target=word_B, context=f"{instrument} music")
```

**Pass criteria:**

```python
# P3-A: run produces output directory with final WAV
pair_dir = run_system_for_pair(
    word_A="warm", word_B="bright", instrument="piano",
    dry_paths=["<one piano wav>"],
    method=Method.InstructFX2FX,
    llm_client=..., clap=..., fx_chain=...,
    output_dir=SYSTEM_RESULTS_DIR,
    n_iter=50, save_interval=25, nr_runs=1,
)
assert os.path.isdir(pair_dir)

# P3-B: collect final audio
finals = collect_final_audio_paths(pair_dir, Method.InstructFX2FX)
assert len(finals) >= 1
assert all(p.endswith("_final_output.wav") for p in finals)

# P3-C: collect trajectory
traj = collect_trajectory_audio_paths(pair_dir)
assert "start" in traj and "end" in traj
# With save_interval=25 and n_iter=50, expect: start, iter_25, iter_50, end
assert len(traj) >= 3
assert all(len(v) >= 1 for v in traj.values())

# P3-D: LLM+LLM has no trajectory (or minimal)
pair_dir_baseline = run_system_for_pair(..., method=Method.LLM_LLM, ...)
finals_b = collect_final_audio_paths(pair_dir_baseline, Method.LLM_LLM)
assert len(finals_b) >= 1
```

**Exit gate:** P3-A through P3-D pass with real model objects (not mocks).

---

## Phase 4 — Experiment 1: Sequential MMD

**File:** `eval/exp1_sequential.py`

Two functions: `_compute_mmd_for_pair_instrument`, `run_exp1`

**Note:** DSP feature extraction and MMD computation are already fully implemented by Yuxuan:
- `from src.metrics import extract_dsp_feature` — 35-D feature vector per WAV file
- `from src.metrics import cal_mmd_score` — Gaussian-kernel MMD between two (N×35) matrices
- See `src/metrics/README_METRICS_QUICKSTART.md` for usage

`_compute_mmd_for_pair_instrument` is a **thin wrapper** — no new metric logic needed:
```python
X = np.array([extract_dsp_feature(f) for f in gt_audio_paths])
Y = np.array([extract_dsp_feature(f) for f in system_audio_paths])
return cal_mmd_score(X, Y)
```

**Pass criteria:**

```python
# P4-A: MMD helper returns a float
mmd = _compute_mmd_for_pair_instrument(
    system_audio_paths=finals,      # from Phase 3
    gt_audio_paths=seq_paths,       # from Phase 2
)
assert isinstance(mmd, float) and mmd >= 0.0

# P4-B: run_exp1 returns correctly-structured dict
results = run_exp1(
    method=Method.InstructFX2FX,
    method_name="ours",
    pairs=[("warm", "bright")],
    instruments=["piano"],
    dry_paths_by_instrument={"piano": ["<one wav>"]},
    gt_bank_dir=GT_BANK_DIR,
    system_results_dir=SYSTEM_RESULTS_DIR,
    results_dir=RESULTS_DIR,
)
assert results["method"] == "ours"
assert "warm_to_bright" in results["pairs"]
assert "piano" in results["pairs"]["warm_to_bright"]
assert "avg" in results["pairs"]["warm_to_bright"]
assert isinstance(results["overall_avg"], float)

# P4-C: ours MMD <= baseline MMD (sanity, not hard requirement)
# Log both values; flag if ours > baseline but don't fail the phase
```

**Exit gate:** P4-A and P4-B pass. P4-C logged as info.

---

## Phase 5 — Experiment 2: Trajectory MMD

**File:** `eval/exp2_trajectory.py`

One function: `run_exp2`

**Note:** No new metric logic — reuse existing code:
- `_compute_mmd_for_pair_instrument` imported from `eval/exp1_sequential.py` (Phase 4)
- `extract_dsp_feature` + `cal_mmd_score` from `src/metrics` (Yuxuan's implementation)
- `word_A_dist` and `word_B_dist` paths come from `build_word_gt_bank()` in `gt_bank.py` (Phase 2)
- `trajectory` dict comes from `collect_trajectory_audio_paths()` in `run_system.py` (Phase 3)

Core loop (no new abstractions needed):
```python
for step, step_paths in trajectory.items():
    mmd_toward_A = _compute_mmd_for_pair_instrument(step_paths, word_A_dist_paths)
    mmd_toward_B = _compute_mmd_for_pair_instrument(step_paths, word_B_dist_paths)
```

Phase 5 is a pure **orchestration** file — load GT paths, load trajectory paths, loop, aggregate. All heavy lifting is already done.

**Pass criteria:**

```python
# P5-A: output has correct keys and types
results = run_exp2(
    pairs=[("warm", "bright")],
    instruments=["piano"],
    dry_paths_by_instrument={"piano": ["<one wav>"]},
    gt_bank_dir=GT_BANK_DIR,
    system_results_dir=SYSTEM_RESULTS_DIR,
    results_dir=RESULTS_DIR,
)
pair_data = results["pairs"]["warm_to_bright"]
assert "steps" in pair_data
assert "mmd_toward_A" in pair_data
assert "mmd_toward_B" in pair_data

# P5-B: lengths match
n = len(pair_data["steps"])
assert len(pair_data["mmd_toward_A"]) == n
assert len(pair_data["mmd_toward_B"]) == n
assert n >= 3   # at least start, one mid-point, end

# P5-C: directional trend (soft — log warning, not hard fail)
# mmd_toward_B should generally decrease from start to end
# mmd_toward_A should generally increase from start to end
delta_B = pair_data["mmd_toward_B"][-1] - pair_data["mmd_toward_B"][0]
delta_A = pair_data["mmd_toward_A"][-1] - pair_data["mmd_toward_A"][0]
print(f"delta_B={delta_B:.4f} (want <0), delta_A={delta_A:.4f} (want >0)")
```

**Exit gate:** P5-A and P5-B pass. P5-C is a research result, not a code correctness check.

---

## Phase 6 — Report + CLI

**Files:** `eval/report.py`, `eval/run.py`

**Pass criteria:**

```python
# P6-A: save_exp1_table writes JSON and TXT
save_exp1_table(results_exp1, output_path=f"{RESULTS_DIR}/exp1")
assert os.path.exists(f"{RESULTS_DIR}/exp1.json")
assert os.path.exists(f"{RESULTS_DIR}/exp1.txt")

# P6-B: plot_exp2_trajectory writes PNGs
plot_exp2_trajectory(results_exp2, output_path=f"{RESULTS_DIR}/exp2")
# one PNG per pair
assert os.path.exists(f"{RESULTS_DIR}/exp2/warm_to_bright.png")

# P6-C: CLI end-to-end smoke test (one pair, one instrument, low n_iter)
# python eval/run.py --exp all --method ours
# Exits 0, produces exp1.json + exp2/warm_to_bright.png
```

**Exit gate:** P6-A and P6-B pass. CLI exits 0 with small test config.

---

## Dependency Graph

```
Phase 0 (verify deps)
    ↓
Phase 1 (config + data layer)
    ↓
Phase 2 (GT bank)     Phase 3 (system runner)
    ↓                      ↓
    └──────────┬────────────┘
               ↓
         Phase 4 (Exp 1)   Phase 5 (Exp 2)
               └──────┬────────┘
                       ↓
                  Phase 6 (report + CLI)
```

Phases 2 and 3 can be developed in parallel after Phase 1 completes.
Phases 4 and 5 can be developed in parallel after Phases 2 and 3 complete.

---

## Open Items (must resolve before coding)

| Item | Blocking Phase | Action |
|---|---|---|
| Exact HF field names for `socialfx-gen-eval` and `socialfx-original` | Phase 1 | Inspect dataset schema in Phase 0 |
| Final 20 word pairs | Phase 1 / config.py | Choose from 7 EQ words; `data/socialfx/final_eq_words.json` has counts |
| Dry audio source (which instrument dirs) | Phase 1 / config.py | Confirm paths in `DRY_AUDIO_DIR` |
| `apply_eq.js` exact CLI signature | Phase 1 / get_gt.py | Run `node apply_eq.js --help` in Phase 0 |
