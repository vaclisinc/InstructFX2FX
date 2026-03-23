import argparse
import json
import os
from typing import Dict, Any

from src.metrics.metric_fxsearcher import run_fxsearcher_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FxSearcher metrics (CLAP/FAD/LUFS/etc.) for a folder of audio."
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing predicted/transformed audio files",
    )
    parser.add_argument(
        "--gt_dir",
        default=None,
        help="Optional ground-truth audio directory (needed for distribution metrics like FAD).",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Optional JSON mapping filename -> text prompt (needed for CLAP-based metrics).",
    )
    parser.add_argument(
        "--outdir",
        default="result",
        help="Directory to save the JSON result (default: result)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Target sample rate for metrics (default: 48000)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional: only evaluate first N files (for quick demo).",
    )
    args = parser.parse_args()

    prompts_map: Dict[str, str] = {}
    if args.prompts is not None:
        with open(args.prompts, "r") as f:
            prompts_map = json.load(f)

    results: Dict[str, Any] = run_fxsearcher_evaluation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        prompts_map=prompts_map if prompts_map else None,
        target_sr=args.sr,
        max_files=args.max_files,
    )

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "fxsearcher_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved FxSearcher metrics to {out_path}")


if __name__ == "__main__":
    main()