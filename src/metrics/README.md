# Metrics

Short guide to the metric modules and how to run them.

---

## Files

| File | Purpose |
|------|--------|
| **metric.py** | Base `Metric` interface (`compute(original_audio, target_audio, prompt)`). |
| **dsp_features.py** | 35-D DSP feature extraction (spectral, MFCC, RMS, ZCR, etc.). Used by MMD and DSP distance. |
| **clap_metric.py** | CLAP text–audio similarity (laion_clap). Higher = better match to prompt. |
| **mmd_metric.py** | MMD (Gaussian kernel) over DSP features between two sets of audio. **Needs GT.** Lower = closer distribution. |
| **dsp_distance_metric.py** | Euclidean distance between mean DSP feature vectors of two folders. **Needs GT.** Lower = closer. |
| **llm2fx.py** | LLM2Fx-style evaluation (param MAE, CLAP, MMD); imports from the modules above. |
| **metric_fxsearcher.py** | FxSearcher-style metrics (CLAP, WER, PESQ, FAD, etc.) for other pipelines. |

---

## How to run

**From repo root.**

### CLAP (no ground truth)

Compares each audio file to a text prompt. **Input:** folder of audio + prompt(s). **Output:** `clap_per_file`, `clap_mean` (higher = better alignment).

```bash
# One prompt for all files
python clap_metrics.py --pred_dir data/pred --prompt "a warm, lush ambient pad with gentle reverb"

# Or per-file prompts from JSON
python clap_metrics.py --pred_dir data/pred --prompts path/to/prompts.json
```

Result: `result/clap_results.json` → `clap_per_file`, `clap_mean`. Optional `--breakdown word` or `--breakdown phrase` adds interpretability fields.

---

### MMD (needs ground truth)

Compares distribution of DSP features: GT folder vs predicted folder. **Input:** `gt_dir`, `pred_dir`. **Output:** one number (lower = pred closer to GT).

```bash
python -m src.run_mmd_demo --gt_dir data/gt --pred_dir data/pred --sr 22050
```

Result: `result/mmd_results.json` → `mmd_dsp`.

---

### DSP feature distance (needs ground truth)

Euclidean distance between the **mean** DSP feature vector of GT files and of predicted files. **Input:** `gt_dir`, `pred_dir`. **Output:** one number (lower = closer).

```bash
python -m src.run_dsp_distance_demo --gt_dir data/gt --pred_dir data/pred --sr 22050
```

Result: `result/dsp_distance_results.json` → `dsp_feat_dist`.

---

## When to use which

- **No reference audio:** use **CLAP** only (`clap_metrics.py`). Compare configs by `clap_mean`.
- **Have reference (GT) audio:** add **MMD** and/or **DSP distance** with the same `gt_dir` / `pred_dir` to measure how close predictions are to GT in feature space.

See `docs/EVALUATION_WITHOUT_GT.md` for more detail.
