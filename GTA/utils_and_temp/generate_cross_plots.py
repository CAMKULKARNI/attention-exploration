import os
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme aesthetic
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": "#30363d",
        "axes.linewidth": 1.2,
        "grid.color": "#21262d",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "savefig.facecolor": "#0d1117",
        "savefig.edgecolor": "#0d1117",
    }
)

OUTPUT_DIR = "./blog_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Consolidated Data Directory (All 29 Experiments)
# -----------------------------------------------------------------------------
data = {
    1: {
        "name": "Vanilla MHA 24L",
        "attn": "mha",
        "pe": "PE",
        "layers": 24,
        "ppl": 60.22,
        "vram_4096": 2563.04,
        "tpot_4096": 0.0066,
        "vram_scale": [530.67, 630.92, 1046.12, 2563.04],
    },
    2: {
        "name": "GQA RoPE 24L",
        "attn": "gqa",
        "pe": "RoPE",
        "layers": 24,
        "ppl": 42.17,
        "vram_4096": 2166.16,
        "tpot_4096": 0.0056,
        "vram_scale": [750.28, 738.92, 1116.88, 2166.16],
    },
    3: {
        "name": "TA ALiBi 24L",
        "attn": "ta",
        "pe": "ALiBi",
        "layers": 24,
        "ppl": 41.06,
        "vram_4096": 2400.31,
        "tpot_4096": 0.0063,
        "vram_scale": [676.44, 702.92, 1159.81, 2400.31],
    },
    4: {
        "name": "GQA RoPE 28L",
        "attn": "gqa",
        "pe": "RoPE",
        "layers": 28,
        "ppl": 42.53,
        "vram_4096": 2124.55,
        "tpot_4096": 0.0069,
        "vram_scale": [637.44, 805.76, 1148.56, 2124.55],
    },
    5: {
        "name": "GQA RoPE 30L",
        "attn": "gqa",
        "pe": "RoPE",
        "layers": 30,
        "ppl": 42.46,
        "vram_4096": 2138.06,
        "tpot_4096": 0.0070,
        "vram_scale": [680.76, 805.76, 1143.56, 2138.06],
    },
    6: {
        "name": "TA ALiBi 28L",
        "attn": "ta",
        "pe": "ALiBi",
        "layers": 28,
        "ppl": 41.20,
        "vram_4096": 2392.05,
        "tpot_4096": 0.0075,
        "vram_scale": [612.76, 724.76, 1160.13, 2392.05],
    },
    7: {
        "name": "TA ALiBi 30L",
        "attn": "ta",
        "pe": "ALiBi",
        "layers": 30,
        "ppl": 41.45,
        "vram_4096": 2423.89,
        "tpot_4096": 0.0075,
        "vram_scale": [642.60, 790.76, 1076.39, 2423.89],
    },
    8: {
        "name": "MHA IsoKV 6L",
        "attn": "mha",
        "pe": "PE",
        "layers": 6,
        "ppl": 53.97,
        "vram_4096": 2102.29,
        "tpot_4096": 0.0028,
        "vram_scale": [819.97, 794.97, 1154.76, 2102.29],
    },
    9: {
        "name": "MHA RoPE 24L",
        "attn": "mha",
        "pe": "RoPE",
        "layers": 24,
        "ppl": 43.16,
        "vram_4096": 2954.04,
        "tpot_4096": 0.0067,
        "vram_scale": [528.67, 630.92, 1070.54, 2954.04],
    },
    10: {
        "name": "MHA RoPE 28L",
        "attn": "mha",
        "pe": "RoPE",
        "layers": 28,
        "ppl": 43.75,
        "vram_4096": 3039.31,
        "tpot_4096": 0.0080,
        "vram_scale": [546.60, 627.76, 1148.38, 3039.31],
    },
    11: {
        "name": "MHA RoPE 30L",
        "attn": "mha",
        "pe": "RoPE",
        "layers": 30,
        "ppl": 43.86,
        "vram_4096": 3123.31,
        "tpot_4096": 0.0090,
        "vram_scale": [536.67, 653.58, 1188.17, 3123.31],
    },
    12: {
        "name": "GTA ALiBi 24L",
        "attn": "gta",
        "pe": "ALiBi",
        "layers": 24,
        "ppl": 40.51,
        "vram_4096": 2106.91,
        "tpot_4096": 0.0053,
        "vram_scale": [790.28, 823.28, 1147.97, 2106.91],
    },
    13: {
        "name": "GTA ALiBi 28L",
        "attn": "gta",
        "pe": "ALiBi",
        "layers": 28,
        "ppl": 40.44,
        "vram_4096": 2114.32,
        "tpot_4096": 0.0055,
        "vram_scale": [668.44, 853.76, 1223.56, 2114.32],
    },
    14: {
        "name": "GTA ALiBi 30L",
        "attn": "gta",
        "pe": "ALiBi",
        "layers": 30,
        "ppl": 40.44,
        "vram_4096": 2120.57,
        "tpot_4096": 0.0062,
        "vram_scale": [694.76, 849.26, 1217.31, 2120.57],
    },
    15: {
        "name": "MHA ALiBi 24L",
        "attn": "mha",
        "pe": "ALiBi",
        "layers": 24,
        "ppl": 40.21,
        "vram_4096": 2608.04,
        "tpot_4096": 0.0070,
        "vram_scale": [528.67, 630.92, 1041.12, 2608.04],
    },
    16: {
        "name": "MHA ALiBi 28L",
        "attn": "mha",
        "pe": "ALiBi",
        "layers": 28,
        "ppl": 40.35,
        "vram_4096": 2639.31,
        "tpot_4096": 0.0082,
        "vram_scale": [546.60, 627.76, 946.88, 2639.31],
    },
    17: {
        "name": "MHA ALiBi 30L",
        "attn": "mha",
        "pe": "ALiBi",
        "layers": 30,
        "ppl": 40.73,
        "vram_4096": 2684.31,
        "tpot_4096": 0.0090,
        "vram_scale": [536.67, 594.92, 967.67, 2684.31],
    },
    18: {
        "name": "GQA ALiBi 24L",
        "attn": "gqa",
        "pe": "ALiBi",
        "layers": 24,
        "ppl": 39.12,
        "vram_4096": 2219.16,
        "tpot_4096": 0.0056,
        "vram_scale": [750.28, 738.92, 1116.88, 2219.16],
    },
    19: {
        "name": "GQA ALiBi 28L",
        "attn": "gqa",
        "pe": "ALiBi",
        "layers": 28,
        "ppl": 39.15,
        "vram_4096": 2177.55,
        "tpot_4096": 0.0067,
        "vram_scale": [637.44, 805.76, 1148.56, 2177.55],
    },
    20: {
        "name": "GQA ALiBi 30L",
        "attn": "gqa",
        "pe": "ALiBi",
        "layers": 30,
        "ppl": 39.30,
        "vram_4096": 2191.06,
        "tpot_4096": 0.0069,
        "vram_scale": [680.76, 805.76, 1143.56, 2191.06],
    },
    21: {
        "name": "TA PE 24L",
        "attn": "ta",
        "pe": "PE",
        "layers": 24,
        "ppl": 56.37,
        "vram_4096": 2346.31,
        "tpot_4096": 0.0063,
        "vram_scale": [670.44, 702.92, 1159.81, 2346.31],
    },
    22: {
        "name": "TA PE 28L",
        "attn": "ta",
        "pe": "PE",
        "layers": 28,
        "ppl": 57.83,
        "vram_4096": 2342.05,
        "tpot_4096": 0.0074,
        "vram_scale": [622.76, 727.76, 1160.13, 2342.05],
    },
    23: {
        "name": "TA PE 30L",
        "attn": "ta",
        "pe": "PE",
        "layers": 30,
        "ppl": 58.65,
        "vram_4096": 2369.89,
        "tpot_4096": 0.0075,
        "vram_scale": [644.60, 790.76, 1071.39, 2369.89],
    },
    24: {
        "name": "GTA PE 24L",
        "attn": "gta",
        "pe": "PE",
        "layers": 24,
        "ppl": 55.41,
        "vram_4096": 2051.31,
        "tpot_4096": 0.0053,
        "vram_scale": [784.28, 821.28, 1147.97, 2051.31],
    },
    25: {
        "name": "GTA PE 28L",
        "attn": "gta",
        "pe": "PE",
        "layers": 28,
        "ppl": 55.17,
        "vram_4096": 2060.32,
        "tpot_4096": 0.0053,
        "vram_scale": [668.44, 852.26, 1222.31, 2060.32],
    },
    26: {
        "name": "GTA PE 30L",
        "attn": "gta",
        "pe": "PE",
        "layers": 30,
        "ppl": 55.61,
        "vram_4096": 2064.82,
        "tpot_4096": 0.0060,
        "vram_scale": [696.76, 849.26, 1217.31, 2064.82],
    },
    27: {
        "name": "GQA PE 24L",
        "attn": "gqa",
        "pe": "PE",
        "layers": 24,
        "ppl": 51.65,
        "vram_4096": 2174.14,
        "tpot_4096": 0.0055,
        "vram_scale": [744.28, 738.92, 1117.38, 2174.14],
    },
    28: {
        "name": "GQA PE 28L",
        "attn": "gqa",
        "pe": "PE",
        "layers": 28,
        "ppl": 53.26,
        "vram_4096": 2123.32,
        "tpot_4096": 0.0066,
        "vram_scale": [646.76, 808.76, 1154.81, 2123.32],
    },
    29: {
        "name": "GQA PE 30L",
        "attn": "gqa",
        "pe": "PE",
        "layers": 30,
        "ppl": 53.43,
        "vram_4096": 2132.56,
        "tpot_4096": 0.0069,
        "vram_scale": [690.76, 808.76, 1142.31, 2132.56],
    },
}


# -----------------------------------------------------------------------------
# PLOT 1: The Pareto Frontier (Peak VRAM vs Val Perplexity @ 24 Layers)
# -----------------------------------------------------------------------------
def plot_pareto_frontier():
    fig, ax = plt.subplots(figsize=(10, 6))

    exps_24l = [1, 2, 3, 9, 12, 15, 18, 21, 24, 27]
    pe_colors = {"ALiBi": "#58a6ff", "RoPE": "#ffa657", "PE": "#f85149"}
    pe_markers = {"ALiBi": "o", "RoPE": "s", "PE": "^"}

    for eid in exps_24l:
        item = data[eid]
        c = pe_colors[item["pe"]]
        m = pe_markers[item["pe"]]

        ax.scatter(item["vram_4096"], item["ppl"], color=c, marker=m, s=140, edgecolors="#ffffff", zorder=4)
        ax.annotate(
            item["name"],
            (item["vram_4096"], item["ppl"]),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=9,
            color="#c9d1d9",
        )

    # Highlight Pareto Optimal Zone
    ax.axvspan(2000, 2250, color="#3fb950", alpha=0.12, label="Pareto Sweet Spot (High Speed / Low VRAM)")

    ax.set_title(
        "The Pareto Frontier: Hardware Cost vs Model Quality (24-Layer Variants)",
        fontsize=14,
        pad=15,
        fontweight="bold",
        color="#f0f6fc",
    )
    ax.set_xlabel("Peak Incremental VRAM @ 4096 Tokens (MB)", fontsize=12, labelpad=10)
    ax.set_ylabel("Validation Perplexity (Lower is Better)", fontsize=12, labelpad=10)
    ax.grid(True)

    # Custom Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="ALiBi Variants", markerfacecolor="#58a6ff", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="RoPE Variants", markerfacecolor="#ffa657", markersize=10),
        Line2D([0], [0], marker="^", color="w", label="Learned PE Variants", markerfacecolor="#f85149", markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_1_pareto_frontier.png", dpi=300)
    plt.close()
    print("✅ Plot 1 saved: plot_1_pareto_frontier.png")


# -----------------------------------------------------------------------------
# PLOT 2: Speed vs. Quality (TPOT Mean vs Val Perplexity)
# -----------------------------------------------------------------------------
def plot_speed_vs_quality():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Focus on top performers (ALiBi and RoPE) across 24L, 28L, and 30L
    selected_exps = [2, 3, 4, 5, 6, 7, 12, 13, 14, 18, 19, 20]
    attn_colors = {"gqa": "#58a6ff", "ta": "#d2a8ff", "gta": "#3fb950"}

    for eid in selected_exps:
        item = data[eid]
        c = attn_colors[item["attn"]]

        # Scale TPOT to milliseconds for cleaner visualization
        tpot_ms = item["tpot_4096"] * 1000
        ax.scatter(tpot_ms, item["ppl"], color=c, s=120, edgecolors="#ffffff", zorder=4)

        label = f"{item['attn'].upper()} {item['pe']} ({item['layers']}L)"
        ax.annotate(
            label,
            (tpot_ms, item["ppl"]),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=8.5,
            color="#8b949e",
        )

    ax.set_title(
        "Inference Speed vs. Perplexity (24L - 30L Depth Sweeps @ 4096 Tokens)",
        fontsize=14,
        pad=15,
        fontweight="bold",
        color="#f0f6fc",
    )
    ax.set_xlabel("Time Per Output Token (TPOT) in Milliseconds (Lower is Faster)", fontsize=12, labelpad=10)
    ax.set_ylabel("Validation Perplexity (Lower is Better)", fontsize=12, labelpad=10)
    ax.grid(True)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Grouped Query Attention (GQA)",
            markerfacecolor="#58a6ff",
            markersize=10,
        ),
        Line2D([0], [0], marker="o", color="w", label="Tied Attention (TA)", markerfacecolor="#d2a8ff", markersize=10),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Group Tied Attention (GTA)",
            markerfacecolor="#3fb950",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_2_speed_vs_quality.png", dpi=300)
    plt.close()
    print("✅ Plot 2 saved: plot_2_speed_vs_quality.png")


# -----------------------------------------------------------------------------
# PLOT 3: Context Scaling Curve (Memory Flattening)
# -----------------------------------------------------------------------------
def plot_context_scaling():
    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = [512, 1024, 2048, 4096]
    scaling_exps = {
        1: ("Vanilla MHA 24L", "#f85149", "o", "-"),
        2: ("GQA RoPE 24L", "#ffa657", "s", "--"),
        3: ("TA ALiBi 24L", "#d2a8ff", "^", "-."),
        12: ("GTA ALiBi 24L", "#3fb950", "D", "-"),
    }

    for eid, (label, color, marker, linestyle) in scaling_exps.items():
        vram_series = data[eid]["vram_scale"]
        ax.plot(
            lengths,
            vram_series,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            markersize=8,
        )

    ax.set_title(
        "KV Cache Memory Scaling Across Prompt Lengths", fontsize=14, pad=15, fontweight="bold", color="#f0f6fc"
    )
    ax.set_xlabel("Prompt Length (Tokens)", fontsize=12, labelpad=10)
    ax.set_ylabel("Peak Incremental VRAM (MB)", fontsize=12, labelpad=10)
    ax.set_xticks(lengths)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, facecolor="#161b22", edgecolor="#30363d")

    # Callout annotation for memory savings
    ax.annotate(
        "GTA & GQA prevent O(N) VRAM explosion\nat 4096 tokens",
        xy=(4096, 2106.91),
        xytext=(2500, 1500),
        arrowprops=dict(facecolor="#3fb950", edgecolor="#3fb950", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        color="#3fb950",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_3_context_scaling.png", dpi=300)
    plt.close()
    print("✅ Plot 3 saved: plot_3_context_scaling.png")


# -----------------------------------------------------------------------------
# PLOT 4: The Iso-KV Cache Showdown (Exp 8 vs Exp 12)
# -----------------------------------------------------------------------------
def plot_isokv_showdown():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    categories = [
        "Exp 8:\nMHA IsoKV (6L)",
        "Exp 12:\nGTA ALiBi (24L)",
        "Exp 1:\nVanilla MHA (24L)",
        "Exp 2:\nGQA RoPE (24L)",
    ]
    vram_vals = [data[8]["vram_4096"], data[12]["vram_4096"], data[1]["vram_4096"], data[2]["vram_4096"]]
    ppl_vals = [data[8]["ppl"], data[12]["ppl"], data[1]["ppl"], data[2]["ppl"]]

    x = np.arange(len(categories))
    width = 0.35

    color_vram = "#58a6ff"
    color_ppl = "#f85149"

    rects1 = ax1.bar(x - width / 2, vram_vals, width, label="Peak VRAM @ 4096 (MB)", color=color_vram, alpha=0.85)
    ax1.set_ylabel("Peak VRAM (MB)", color=color_vram, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_vram)
    ax1.set_ylim(0, 3500)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, ppl_vals, width, label="Validation Perplexity", color=color_ppl, alpha=0.85)
    ax2.set_ylabel("Validation Perplexity (Lower is Better)", color=color_ppl, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_ppl)
    ax2.set_ylim(0, 70)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax1.set_title(
        "Iso-KV Cache Showdown: Depth Reinvestment vs. Truncation",
        fontsize=14,
        pad=15,
        fontweight="bold",
        color="#f0f6fc",
    )
    ax1.grid(False)

    # Value Labels
    for bar in rects1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.0f} MB",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color_vram,
        )

    for bar in rects2:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color_ppl,
        )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_4_isokv_showdown.png", dpi=300)
    plt.close()
    print("✅ Plot 4 saved: plot_4_isokv_showdown.png")


# -----------------------------------------------------------------------------
# PLOT 5: Positional Encoding Ablation (GQA Family)
# -----------------------------------------------------------------------------
def plot_pe_ablation():
    fig, ax = plt.subplots(figsize=(10, 6))

    pe_exps = {
        18: ("GQA + ALiBi (24L)", "#58a6ff", "ALiBi gives strong locality inductive bias"),
        2: ("GQA + RoPE (24L)", "#ffa657", "RoPE requires position rotations"),
        27: ("GQA + PE (24L)", "#f85149", "Learned PE struggles at 512 context"),
    }

    names = [pe_exps[e][0] for e in pe_exps]
    ppl_values = [data[e]["ppl"] for e in pe_exps]
    colors = [pe_exps[e][1] for e in pe_exps]

    bars = ax.barh(names, ppl_values, color=colors, height=0.55, alpha=0.9, edgecolor="#ffffff")

    ax.set_title(
        "Impact of Positional Encoding Choice on GQA (24-Layer Models)",
        fontsize=14,
        pad=15,
        fontweight="bold",
        color="#f0f6fc",
    )
    ax.set_xlabel("Validation Perplexity after 1 Epoch (Lower is Better)", fontsize=12, labelpad=10)
    ax.set_xlim(30, 60)
    ax.grid(True, axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"  {width:.2f} PPL",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#f0f6fc",
        )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_5_pe_ablation.png", dpi=300)
    plt.close()
    print("✅ Plot 5 saved: plot_5_pe_ablation.png")


if __name__ == "__main__":
    print("🚀 Generating blog post visual assets...")
    plot_pareto_frontier()
    plot_speed_vs_quality()
    plot_context_scaling()
    plot_isokv_showdown()
    plot_pe_ablation()
    print(f"\n🎉 All 5 plots successfully saved to '{OUTPUT_DIR}/' directory!")
