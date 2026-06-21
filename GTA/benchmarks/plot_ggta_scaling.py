import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Load data
file_path = (
    "results-median.json"
    if os.path.exists("results-median.json")
    else "results_median.json"
)

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["params_m"] = df["params"] / 1e6

# Define the experiments for the Scaling/Depth Narrative
target_exps = [
    "Exp 1: Vanilla MHA (24 Layers)",
    "Exp 11: MHA + RoPE (30 Layers)",
    "Exp 2: GQA + RoPE (24 Layers)",
    "Exp 5: GQA + RoPE (30 Layers)",
    "Exp 14: GTA + ALiBi (30 Layers)",
]

short_labels = [
    "Vanilla MHA\n(24 Layers)",
    "MHA + RoPE\n(30 Layers)",
    "GQA + RoPE\n(24 Layers)",
    "GQA + RoPE\n(30 Layers)",
    "GTA + ALiBi\n(30 Layers)",
]

df_filtered = df[df["exp_name"].isin(target_exps)].copy()

# Enforce explicit order
df_filtered["exp_name"] = pd.Categorical(
    df_filtered["exp_name"], categories=target_exps, ordered=True
)
df_filtered = df_filtered.sort_values(["exp_name", "prompt_len"])

# Dynamically append max param counts (at 4096 context) to labels to prove the parameter growth
final_labels = []
for exp, label in zip(target_exps, short_labels):
    param_val = df_filtered[
        (df_filtered["exp_name"] == exp) & (df_filtered["prompt_len"] == 4096)
    ]["params_m"].iloc[0]
    final_labels.append(f"{label}\n(~{param_val:.1f}M)")

# Setup Plot - 3 subplots sharing the X-axis
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

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
axes[-1].set_xticklabels(final_labels)
axes[0].set_title(
    "The Depth Narrative: GTA 30L vs 24L and 30L Baselines",
    fontweight="bold",
    fontsize=14,
    pad=15,
)

plt.tight_layout()
plt.savefig("ggta_scaling_plot.png", dpi=300)
print("✅ GTA Scaling plot saved to ggta_scaling_plot.png")
