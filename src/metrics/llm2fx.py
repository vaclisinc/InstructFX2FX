import os
import json
import argparse
import warnings
import numpy as np
from typing import Dict, Tuple, Optional, Any

from .dsp_features import (
    FEATURE_NAMES,
    extract_dsp_features,
    extract_dsp_features_from_array,
    extract_features_batch,
)
from .mmd_metric import compute_mmd
from .dsp_distance_metric import compute_dsp_feature_distance
from .clap_metric import (
    compute_clap_score,
    compute_clap_score_from_array,
    compute_guided_clap_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════
# 4. PARAMETER MAE
# ═══════════════════════════════════════════════════════════════

# Ranges for normalize-to-[0,1] MAE. Includes generate_test_data.py keys.
DEFAULT_PARAM_RANGES = {
    # Generic LLM2Fx-style
    "low_gain_db": (-6, 6), "low_cutoff_freq": (60, 120), "low_q": (0.5, 3),
    "mid_gain_db": (-6, 6), "mid_cutoff_freq": (250, 1000), "mid_q": (0.5, 3),
    "high_gain_db": (-6, 6), "high_cutoff_freq": (4000, 8000), "high_q": (0.5, 3),
    "threshold_db": (-20, -10), "ratio": (2, 8),
    "attack_ms": (1, 30), "release_ms": (0, 500),
    "room_size": (0.3, 0.6), "damping": (0.3, 0.6),
    "width": (0.3, 0.6), "mix_ratio": (0.1, 1.0),
    "drive_db": (1, 5), "delay_seconds": (0.01, 0.2),
    "feedback": (0.01, 0.2), "delay_mix": (0.1, 1.0),
    "gain_db": (-6, 6), "pan": (-0.6, 0.6),
    # generate_test_data.py / Pedalboard test data
    "eq_low_gain_db": (-6, 6), "eq_low_freq": (50, 500),
    "eq_high_gain_db": (-6, 6), "eq_high_freq": (1000, 12000),
    "compressor_threshold": (-24, -6), "compressor_ratio": (1.0, 10.0),
    "reverb_room_size": (0.0, 1.0), "reverb_damping": (0.0, 1.0),
    "reverb_wet": (0.0, 1.0),
}


def compute_parameter_mae(
    gt_params: Dict[str, float],
    pred_params: Dict[str, float],
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Normalized MAE. Reference: LLM2Fx-Tools=0.23, Regression=0.20"""
    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES

    common = set(gt_params) & set(pred_params)
    if not common:
        return float("nan")

    errors = []
    for k in sorted(common):
        g, p = gt_params[k], pred_params[k]
        if k in param_ranges:
            lo, hi = param_ranges[k]
            r = hi - lo
            if r > 0:
                g, p = (g - lo) / r, (p - lo) / r
        errors.append(abs(g - p))
    return float(np.mean(errors))


# ═══════════════════════════════════════════════════════════════
# 5. CLAP SCORE + GUIDED CLAP SCORE
# ═══════════════════════════════════════════════════════════════

_clap_model = None


def _get_clap():
    global _clap_model
    if _clap_model is None:
        import laion_clap
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        _clap_model.load_ckpt()
    return _clap_model


def compute_clap_score(audio_path: str, text_prompt: str) -> float:
    """CLAP cosine similarity. Reference: Text2FX=0.527, FxSearcher=0.447"""
    from scipy.spatial.distance import cosine
    m = _get_clap()
    a = m.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
    t = m.get_text_embedding([f"this sound is {text_prompt}"], use_tensor=False)
    return float(1.0 - cosine(a.flatten(), t.flatten()))


def compute_clap_score_from_array(
    audio: np.ndarray, sr: int, text_prompt: str
) -> float:
    """CLAP score from in-memory audio. audio: (samples,) or (channels, samples)."""
    import tempfile
    import soundfile as sf
    from scipy.spatial.distance import cosine

    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, sr)
        return compute_clap_score(f.name, text_prompt)


def compute_guided_clap_score(
    audio_path: str,
    target_prompt: str,
    guide_prompt: str = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound",
) -> Tuple[float, float, float]:
    """FxSearcher's S_final = S_target - S_guide. Returns (s_tgt, s_gd, s_final)."""
    from scipy.spatial.distance import cosine
    m = _get_clap()
    a = m.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False).flatten()
    tt = m.get_text_embedding([f"this sound is {target_prompt}"], use_tensor=False).flatten()
    tg = m.get_text_embedding([guide_prompt], use_tensor=False).flatten()
    st = float(1.0 - cosine(a, tt))
    sg = float(1.0 - cosine(a, tg))
    return st, sg, st - sg


# ═══════════════════════════════════════════════════════════════
# 6. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_evaluation(
    gt_dir: str,
    pred_dir: str,
    prompts_file: Optional[str] = None,
    gt_params_file: Optional[str] = None,
    pred_params_file: Optional[str] = None,
    sr: int = 22050,
) -> Dict[str, float]:
    results = {}
    bar = "=" * 62

    print(f"\n{bar}")
    print("  PWFX — LLM2Fx Metrics Evaluation")
    print(bar)

    # 1. Features
    print("\n[1/5] Extracting DSP features...")
    gt_f, gt_n = extract_features_batch(gt_dir, sr=sr)
    pr_f, pr_n = extract_features_batch(pred_dir, sr=sr)
    print(f"  GT:   {gt_f.shape[0]} files x {gt_f.shape[1]} features")
    print(f"  Pred: {pr_f.shape[0]} files x {pr_f.shape[1]} features")

    # 2. MMD
    print("\n[2/5] Computing MMD...")
    mmd = compute_mmd(gt_f, pr_f)
    results["mmd_dsp"] = mmd
    print(f"  MMD = {mmd:.4f}")
    print(f"  Ref: LLM2Fx EQ=0.17-0.22, Reverb=0.26-0.27, Random=0.53-0.75")

    # 3. Feature distance
    print("\n[3/5] Computing DSP feature distance...")
    af = compute_dsp_feature_distance(gt_f, pr_f)
    results["dsp_feat_dist"] = af
    print(f"  AF Distance = {af:.4f}")
    print(f"  Ref: LLM2Fx-Tools=8.29, No FX=14.82")

    # 4. Parameter MAE
    if gt_params_file and pred_params_file:
        print("\n[4/5] Computing parameter MAE...")
        gt_p = json.load(open(gt_params_file))
        pr_p = json.load(open(pred_params_file))
        maes = [
            compute_parameter_mae(gt_p[k], pr_p[k])
            for k in gt_p if k in pr_p
        ]
        maes = [m for m in maes if not np.isnan(m)]
        if maes:
            results["param_mae"] = float(np.mean(maes))
            print(f"  MAE = {results['param_mae']:.4f} ({len(maes)} pairs)")
            print(f"  Ref: LLM2Fx-Tools=0.23, Regression=0.20")
    else:
        print("\n[4/5] Skipping parameter MAE (no param files)")

    # 5. CLAP
    if prompts_file:
        print("\n[5/5] Computing CLAP scores...")
        try:
            prompts = json.load(open(prompts_file))
            cs_list, gs_list = [], []
            for fname, prompt in prompts.items():
                p = os.path.join(pred_dir, fname)
                if not os.path.exists(p):
                    continue
                cs = compute_clap_score(p, prompt)
                _, _, gf = compute_guided_clap_score(p, prompt)
                cs_list.append(cs)
                gs_list.append(gf)
                print(f"  {fname}: CLAP={cs:.3f} Guided={gf:.3f}")

            if cs_list:
                results["clap_score"] = float(np.mean(cs_list))
                results["guided_score"] = float(np.mean(gs_list))
                print(f"\n  Avg CLAP   = {results['clap_score']:.4f}")
                print(f"  Avg Guided = {results['guided_score']:.4f}")
                print(f"  Ref: Text2FX=0.527, FxSearcher=0.447, LLM2Fx=0.232")
        except ImportError:
            print("  Install: pip install laion-clap")
    else:
        print("\n[5/5] Skipping CLAP (no prompts file)")

    print(f"\n{bar}")
    print("  RESULTS SUMMARY")
    print(bar)
    for k, v in results.items():
        print(f"  {k:20s} = {v:.4f}")
    print()
    return results


# ═══════════════════════════════════════════════════════════════
# Metric base-class integration (optional)
# ═══════════════════════════════════════════════════════════════

def _prompt_text(prompt: Any) -> str:
    """Get text from Prompt dataclass or plain string."""
    if prompt is None:
        return ""
    if hasattr(prompt, "text"):
        return getattr(prompt, "text") or ""
    return str(prompt)


class LLM2FxMMD:
    """MMD over DSP features. Lower is better."""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        gt_feat = extract_dsp_features_from_array(
            np.asarray(target_audio), sr=self.sr
        )
        pred_feat = extract_dsp_features_from_array(
            np.asarray(original_audio), sr=self.sr
        )
        return compute_mmd(gt_feat[np.newaxis, :], pred_feat[np.newaxis, :])


class LLM2FxCLAP:
    """CLAP text-audio similarity. Higher is better."""

    def __init__(self, sr: int = 48000):
        self.sr = sr

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        text = _prompt_text(prompt)
        if not text:
            return float("nan")
        audio = np.asarray(original_audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        return compute_clap_score_from_array(audio, self.sr, text)


class LLM2FxGuidedCLAP:
    """Guided CLAP (target - guide). Higher is better."""

    def __init__(
        self,
        sr: int = 48000,
        guide_prompt: str = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound",
    ):
        self.sr = sr
        self.guide_prompt = guide_prompt

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        text = _prompt_text(prompt)
        if not text:
            return float("nan")
        import tempfile
        import soundfile as sf
        audio = np.asarray(original_audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, self.sr)
            _, _, s_final = compute_guided_clap_score(
                f.name, text, self.guide_prompt
            )
        return s_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PWFX - LLM2Fx Evaluation Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.metrics.llm2fx --gt_dir data/gt/ --pred_dir data/pred/
  python -m src.metrics.llm2fx --gt_dir data/gt/ --pred_dir data/pred/ --prompts prompts.json
  python -m src.metrics.llm2fx --gt_dir data/gt/ --pred_dir data/pred/ \\
    --prompts prompts.json --gt_params gt.json --pred_params pred.json""",
    )
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--gt_params", default=None)
    parser.add_argument("--pred_params", default=None)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate for feature extraction")
    args = parser.parse_args()

    results = run_evaluation(
        args.gt_dir, args.pred_dir,
        args.prompts, args.gt_params, args.pred_params,
        sr=args.sr,
    )
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")
