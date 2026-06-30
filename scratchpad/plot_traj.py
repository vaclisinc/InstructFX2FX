import importlib.util
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("rec", "eval/recreate_exp5_original_trajectory_plot.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import json

ARCHIVE_JSON = m.ROOT / "eval" / "results" / "archive" / "metrics_iterations_2026-06-29_19-28-54.json"
with ARCHIVE_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)
trajectory = data["llm_clap"]["trajectory"]

# (overall_metric, title, per_inst_metric or None) — afx_rep + CLAP removed, "Overall" stripped
metric_specs = [
    ("mmd_dsp_overall",       "MMD DSP",       "mmd_dsp_per_inst"),
    ("mmd_fxenc_emb_overall", "MMD FXEnc Emb", "mmd_fxenc_emb_per_inst"),
    ("mmd_clap_emb_overall",  "MMD CLAP Emb",  "mmd_clap_emb_per_inst"),
    ("mmd_vggish_overall",    "MMD VGGish",    "mmd_vggish_per_inst"),
]

# Layout 2, 2 per row
fig, axs = plt.subplots(2, 2, figsize=(14, 2 * 3.5), dpi=150)
axes = list(axs.flat)

for ax, (overall_metric, title, per_inst_metric) in zip(axes, metric_specs):
    m._plot_metric(ax, trajectory, overall_metric, title, per_inst_metric)

plt.tight_layout()
out_png = m.ROOT / "eval" / "results" / "exp5" / f"trajectory_plot_{ARCHIVE_JSON.stem}_noafx.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png)
plt.close(fig)
print(f"Saved: {out_png}")
