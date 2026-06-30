import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_JSON = ROOT / "eval" / "results" / "archive" / "metrics_iterations_2026-06-29_19-28-54.json"

with ARCHIVE_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)
trajectory = data["llm_clap"]["trajectory"]

iterations = [step["iteration"] for step in trajectory]

# Overall MMD scores only (no per-instrument breakdown)
metric_specs = [
    ("mmd_dsp_overall",       "MMD DSP",       "#2196F3"),
    ("mmd_fxenc_emb_overall", "MMD FXEnc Emb", "#FF5722"),
    ("mmd_clap_emb_overall",  "MMD CLAP Emb",  "#4CAF50"),
    ("mmd_vggish_overall",    "MMD VGGish",    "#9C27B0"),
]

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
for metric, label, colour in metric_specs:
    values = [step.get(metric) for step in trajectory]
    ax.plot(iterations, values, marker="o", linewidth=2, color=colour, label=label)

ax.set_title("MMD Score")
ax.set_xlabel("Iteration")
ax.set_ylabel("MMD")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
out_png = ROOT / "eval" / "results" / "exp5" / f"trajectory_plot_{ARCHIVE_JSON.stem}_mmd_only.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png)
plt.close(fig)
print(f"Saved: {out_png}")
