import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open("results.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Aggregate data (mean over the 5 runs)
# Grouping by 'exp_name' and 'prompt_len'
agg_df = (
    df.groupby(["exp_name", "attn_type", "layers", "prompt_len"])
    .agg({"params": "median", "vram": "median", "ttft": "median", "tpot": "median"})
    .reset_index()
)

# Sort by exp_name appropriately to maintain order
agg_df = agg_df.sort_values(by=["exp_name", "prompt_len"])

# Create pivot tables for clearer presentation
pivot_ttft = agg_df.pivot(index="exp_name", columns="prompt_len", values="ttft").round(
    3
)
pivot_tpot = agg_df.pivot(index="exp_name", columns="prompt_len", values="tpot").round(
    3
)
pivot_vram = agg_df.pivot(index="exp_name", columns="prompt_len", values="vram").round(
    2
)
params = agg_df.pivot(index="exp_name", columns="prompt_len", values="params").round(2)

print("### Averaged Time To First Token (TTFT) in Seconds ###")
print(pivot_ttft.to_markdown())
print("\n### Averaged Time Per Output Token (TPOT) in Seconds ###")
print(pivot_tpot.to_markdown())
print("\n### VRAM Usage in MB ###")
print(pivot_vram.to_markdown())
print("\n### Parameter Count ###")
print(params.to_markdown())

# Setup plotting style
sns.set_theme(style="whitegrid")

# 1. TTFT vs Prompt Length
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="prompt_len",
    y="ttft",
    hue="exp_name",
    marker="o",
    errorbar="sd",
    style="attn_type",
    markers=True,
)
plt.title("Time To First Token (TTFT) vs Prompt Length")
plt.ylabel("TTFT (seconds)")
plt.xlabel("Prompt Length (tokens)")
plt.xticks([512, 1024, 2048, 4096])
plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("ttft_plot.png")
plt.close()

# 2. TPOT vs Prompt Length
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="prompt_len",
    y="tpot",
    hue="exp_name",
    marker="o",
    errorbar="sd",
    style="attn_type",
    markers=True,
)
plt.title("Time Per Output Token (TPOT) vs Prompt Length")
plt.ylabel("TPOT (seconds)")
plt.xlabel("Prompt Length (tokens)")
plt.xticks([512, 1024, 2048, 4096])
plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("tpot_plot.png")
plt.close()

# 3. VRAM vs Prompt Length
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="prompt_len",
    y="vram",
    hue="exp_name",
    marker="o",
    errorbar=None,
    style="attn_type",
    markers=True,
)
plt.title("VRAM Usage vs Prompt Length")
plt.ylabel("VRAM (MB)")
plt.xlabel("Prompt Length (tokens)")
plt.xticks([512, 1024, 2048, 4096])
plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("vram_plot.png")
plt.close()
