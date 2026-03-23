import argparse
import json
import os

from src.metrics.dsp_distance_metric import run_dsp_distance_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DSP feature distance between two folders of audio."
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
        default="result-demo",
        help="Directory to save the JSON result (default: result)",
    )
    args = parser.parse_args()

    value = run_dsp_distance_evaluation(args.gt_dir, args.pred_dir, sr=args.sr)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "dsp_distance_results.json")
    with open(out_path, "w") as f:
        json.dump({"dsp_feat_dist": value}, f, indent=2)
    print(f"Saved DSP distance result to {out_path}")


if __name__ == "__main__":
    main()
