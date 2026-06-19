import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Load data (trying both dash and underscore naming conventions based on context)
file_path = (
    "results-median.json"
    if os.path.exists("results-median.json")
    else "results_median.json"
)

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Filter for 24-layer variants across ALL sequence lengths
target_exps = [
    "Exp 1: Vanilla MHA (24 Layers)",
    "Exp 2: GQA + RoPE (24 Layers)",
    "Exp 3: GTA + ALiBi (24 Layers)",
    "Exp 12: GGTA + ALiBi (24 Layers)",
    "Exp 9: MHA + RoPE (24 Layers)",
]
short_labels = [
    "Vanilla MHA",
    "GQA + RoPE",
    "GTA + ALiBi",
    "GGTA + ALiBi",
    "MHA + RoPE",
]

df_filtered = df[df["exp_name"].isin(target_exps)].copy()

# Enforce specific order for the narrative
df_filtered["exp_name"] = pd.Categorical(
    df_filtered["exp_name"], categories=target_exps, ordered=True
)
df_filtered = df_filtered.sort_values(["exp_name", "prompt_len"])

# Setup Plot - 3 subplots sharing the x-axis
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

x = np.arange(len(target_exps))
width = 0.20

prompt_lengths = [512, 1024, 2048, 4096]
alphas = [0.4, 0.6, 0.8, 1.0]  # Gradient intensities for seq lengths

color_vram = "#1f77b4"  # Tab:blue
color_ttft = "#2ca02c"  # Tab:green
color_tpot = "#ff7f0e"  # Tab:orange

metrics = [
    ("vram", "Peak VRAM (MB)", color_vram, "{:.0f}"),
    ("ttft", "TTFT (s)", color_ttft, "{:.4f}"),
    ("tpot", "TPOT (s/token)", color_tpot, "{:.4f}"),
]

for ax, (col, ylabel, color, fmt) in zip(axes, metrics):
    for i, p_len in enumerate(prompt_lengths):
        # Safely align data values to target_exps index
        y_vals = []
        for exp in target_exps:
            val = df_filtered[
                (df_filtered["exp_name"] == exp) & (df_filtered["prompt_len"] == p_len)
            ][col]
            y_vals.append(val.values[0] if len(val) > 0 else 0)

        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset,
            y_vals,
            width,
            label=f"{p_len} tokens",
            color=color,
            alpha=alphas[i],
        )

        # Explicitly label all bars to depict absolute gain/loss across all lengths
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

    ax.set_ylabel(ylabel, color=color, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Sequence Length", loc="upper left", fontsize=9)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(short_labels)
axes[0].set_title(
    "Baseline Comparison: 24 Layers across Sequence Lengths", fontweight="bold", pad=15
)

plt.tight_layout()
plt.savefig("baseline_24_layers_plot.png", dpi=300)
print("✅ Baseline plot saved to baseline_24_layers_plot.png")

# --- Print the selected numbers as markdown pivot tables ---
print("\n### Baseline Comparison (24 Layers across Sequence Lengths) ###")
for metric, name in [("ttft", "TTFT (s)"), ("tpot", "TPOT (s)"), ("vram", "VRAM (MB)")]:
    pivot = df_filtered.pivot(index="exp_name", columns="prompt_len", values=metric)
    pivot.index = short_labels
    if metric == "vram":
        pivot = pivot.round(0)
    else:
        pivot = pivot.round(4)
    print(f"\n#### {name} ####")
    print(pivot.to_markdown())
