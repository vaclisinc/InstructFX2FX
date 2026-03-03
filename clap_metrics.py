import argparse
import json
import os
import re

from src.metrics.clap_metric import compute_clap_score


def _split_words(prompt: str) -> list[str]:
    """Split prompt into words (for optional per-word CLAP breakdown)."""
    return [w.strip() for w in prompt.split() if w.strip()]


def _split_phrases(prompt: str) -> list[str]:
    """Split prompt into phrases on commas and ' and ', for optional per-phrase CLAP breakdown."""
    parts = re.split(r",|\s+and\s+", prompt, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CLAP score (audio vs input prompt) for a folder of audio. "
        "See docs/EVALUATION_WITHOUT_GT.md."
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
    parser.add_argument(
        "--breakdown",
        choices=("none", "word", "phrase"),
        default="none",
        help="Optional: also compute CLAP per word or per phrase for interpretability (default: none; use full-sentence as primary metric)",
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
    clap_per_word: dict[str, dict[str, float]] = {}
    clap_per_phrase: dict[str, dict[str, float]] = {}

    for fname, prompt in prompts.items():
        audio_path = os.path.join(args.pred_dir, fname)
        if not os.path.exists(audio_path):
            continue
        scores[fname] = compute_clap_score(audio_path, prompt)

        if args.breakdown == "word":
            words = _split_words(prompt)
            clap_per_word[fname] = {
                w: compute_clap_score(audio_path, w) for w in words
            }
        elif args.breakdown == "phrase":
            phrases = _split_phrases(prompt)
            clap_per_phrase[fname] = {
                p: compute_clap_score(audio_path, p) for p in phrases
            }

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "clap_results.json")

    mean_clap = float(sum(scores.values()) / len(scores)) if scores else None

    result: dict = {
        "clap_per_file": scores,
        "clap_mean": mean_clap,
    }
    if args.breakdown == "word" and clap_per_word:
        result["clap_per_word"] = clap_per_word
    if args.breakdown == "phrase" and clap_per_phrase:
        result["clap_per_phrase"] = clap_per_phrase

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved CLAP results to {out_path}")
    if args.breakdown != "none":
        print(f"Breakdown by {args.breakdown} included (interpretability only; use full-sentence as primary metric).")


if __name__ == "__main__":
    main()

