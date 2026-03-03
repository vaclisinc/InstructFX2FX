import argparse
import json
import os

from src.metrics.clap_metric import compute_clap_score, compute_guided_clap_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CLAP and guided CLAP scores for a folder of audio."
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing predicted / generated audio files",
    )
    parser.add_argument(
        "--prompts",
        help="Path to JSON file mapping filename -> text prompt",
    )
    parser.add_argument(
        "--prompt",
        help="Single text prompt to use for all audio files in pred_dir",
    )
    parser.add_argument(
        "--outdir",
        default="result",
        help="Directory to save the JSON result (default: result)",
    )
    args = parser.parse_args()

    if not args.prompts and not args.prompt:
        raise SystemExit("Provide either --prompts JSON or a single --prompt string.")

    if args.prompts:
        with open(args.prompts, "r") as f:
            prompts = json.load(f)
    else:
        # Use the same prompt for every audio file in pred_dir
        prompts = {}
        exts = {".wav", ".mp3", ".flac", ".ogg"}
        for fname in sorted(os.listdir(args.pred_dir)):
            if os.path.splitext(fname)[1].lower() in exts:
                prompts[fname] = args.prompt

    scores: dict[str, float] = {}
    guided_scores: dict[str, float] = {}

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

    mean_clap = float(sum(scores.values()) / len(scores)) if scores else None
    mean_guided = (
        float(sum(guided_scores.values()) / len(guided_scores))
        if guided_scores
        else None
    )

    with open(out_path, "w") as f:
        json.dump(
            {
                "clap_per_file": scores,
                "guided_clap_per_file": guided_scores,
                "clap_mean": mean_clap,
                "guided_clap_mean": mean_guided,
            },
            f,
            indent=2,
        )

    print(f"Saved CLAP results to {out_path}")


if __name__ == "__main__":
    main()

