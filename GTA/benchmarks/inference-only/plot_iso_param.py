import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Load data (handling both dash and underscore naming conventions based on context)
file_path = (
    "results-median.json"
    if os.path.exists("results-median.json")
    else "results_median.json"
)

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Calculate parameters in millions for a cleaner X-axis
df["params_m"] = df["params"] / 1e6

# Define the Iso-Parameter Zone (Models around ~350M - ~360M parameters)
iso_exps = [
    "Exp 1: Vanilla MHA (24 Layers)",
    "Exp 9: MHA + RoPE (24 Layers)",
    "Exp 4: GQA + RoPE (28 Layers)",
    "Exp 13: GTA + ALiBi (28 Layers)",
]

short_labels = [
    "Vanilla MHA\n24 Layers",
    "MHA + RoPE\n24 Layers",
    "GQA + RoPE\n28 Layers",
    "GTA + ALiBi\n28 Layers",
]

df_iso = df[df["exp_name"].isin(iso_exps)].copy()

# Enforce explicit order
df_iso["exp_name"] = pd.Categorical(
    df_iso["exp_name"], categories=iso_exps, ordered=True
)
df_iso = df_iso.sort_values(["exp_name", "prompt_len"])

# Dynamically append param counts to labels to preserve the Iso-Parameter context
final_labels = []
for exp, label in zip(iso_exps, short_labels):
    param_val = df_iso[df_iso["exp_name"] == exp]["params_m"].iloc[0]
    final_labels.append(f"{label}\n(~{param_val:.1f}M)")

# Setup Plot - 3 subplots sharing the X-axis
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

x = np.arange(len(iso_exps))
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
        for exp in iso_exps:
            val = df_iso[(df_iso["exp_name"] == exp) & (df_iso["prompt_len"] == p_len)][
                col
            ]
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
    "The Iso-Parameter Narrative: Computing Trade-offs (~354M Parameter Budget)",
    fontweight="bold",
    fontsize=14,
    pad=15,
)

plt.tight_layout()
plt.savefig("iso_param_plot.png", dpi=300)
print("✅ Iso-Parameter plot saved to iso_param_plot.png")

# --- Print markdown pivot tables for reference ---
print("\n### Iso-Parameter Comparison (Subset matching ~352M to ~360M Params) ###")
for metric, name in [
    ("params_m", "Parameter Count (M)"),
    ("vram", "Peak VRAM (MB)"),
    ("ttft", "TTFT (s)"),
    ("tpot", "TPOT (s/token)"),
]:
    pivot = df_iso.pivot(index="exp_name", columns="prompt_len", values=metric)
    pivot = (
        pivot.round(0)
        if metric == "vram"
        else pivot.round(2) if metric == "params_m" else pivot.round(4)
    )
    print(f"\n#### {name} ####")
    print(pivot.to_markdown())
