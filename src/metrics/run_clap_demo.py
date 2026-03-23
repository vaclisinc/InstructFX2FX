import argparse
import json
import os

from src.metrics.clap_metric import compute_clap_score, compute_guided_clap_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CLAP and guided CLAP scores for a folder of audio."
    )
    parser.add_argument(
        "--pred_dir", required=True, help="Directory containing predicted audio files"
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to JSON file mapping filename -> text prompt",
    )
    parser.add_argument(
        "--outdir",
        default="result-demo",
        help="Directory to save the JSON result (default: result)",
    )
    args = parser.parse_args()

    with open(args.prompts, "r") as f:
        prompts = json.load(f)

    scores = {}
    guided_scores = {}

    for fname, prompt in prompts.items():
        audio_path = os.path.join(args.pred_dir, fname)
        if not os.path.exists(audio_path):
            continue
        s = compute_clap_score(audio_path, prompt)
        _, _, g = compute_guided_clap_score(audio_path, prompt)
        scores[fname] = s
        guided_scores[fname] = g

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "clap_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "clap_per_file": scores,
                "guided_clap_per_file": guided_scores,
                "clap_mean": float(sum(scores.values()) / len(scores)) if scores else None,
                "guided_clap_mean": float(sum(guided_scores.values()) / len(guided_scores))
                if guided_scores
                else None,
            },
            f,
            indent=2,
        )
    print(f"Saved CLAP results to {out_path}")


if __name__ == "__main__":
    main()
