import argparse
import json
import os

from src.metrics.mmd_metric import run_mmd_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MMD over DSP features between two folders of audio."
    )
    parser.add_argument("--gt_dir", required=True, help="Ground-truth audio directory")
    parser.add_argument("--pred_dir", required=True, help="Predicted audio directory")
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate for DSP feature extraction",
    )
    parser.add_argument(
        "--outdir",
        default="result",
        help="Directory to save the JSON result (default: result)",
    )
    args = parser.parse_args()

    value = run_mmd_evaluation(args.gt_dir, args.pred_dir, sr=args.sr)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "mmd_results.json")
    with open(out_path, "w") as f:
        json.dump({"mmd_dsp": value}, f, indent=2)
    print(f"Saved MMD result to {out_path}")


if __name__ == "__main__":
    main()

