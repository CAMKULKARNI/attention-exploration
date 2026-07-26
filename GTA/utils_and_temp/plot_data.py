import os
import json
import matplotlib.pyplot as plt

# Apply Dark Mode Style Globally
plt.style.use("dark_background")

# Configuration
LOG_DIR = "./training_logs"
OUTPUT_BASE_DIR = "./plots_final_sweep_dark"

METRICS = ["train_loss", "val_loss", "train_perplexity", "val_perplexity", "grad_norm"]
TITLES = ["Training Loss", "Validation Loss", "Training Perplexity", "Validation Perplexity", "Gradient Norm"]

# ---------------------------------------------------------
# STANDARD PLOTTING GROUPS (Plots 1 through 7)
# ---------------------------------------------------------
STANDARD_GROUPS = {
    "Plot_1_Positional_Confounder": [
        "MHA ALiBi 24L",
        "Vanilla MHA 24L",
        "MHA RoPE 24L",
        "GQA ALiBi 24L",
        "GQA RoPE 24L",
        "GQA PE 24L",
    ],
    "Plot_2_Cost_of_Compression": [
        "MHA ALiBi 24L",
        "TA ALiBi 24L",
        "GQA ALiBi 24L",
        "GTA ALiBi 24L",
        "Vanilla MHA 24L",
        "TA PE 24L",
        "GQA PE 24L",
        "GTA PE 24L",
    ],
    "Plot_3_Iso_Parameter_Showdown": [
        "MHA ALiBi 24L",
        "GQA ALiBi 28L",
        "GTA ALiBi 28L",
        "GQA RoPE 28L",
        "MHA RoPE 24L",
        "GQA PE 28L",
        "GTA PE 28L",
        "Vanilla MHA 24L",
    ],
    "Plot_4_Edge_Compute_Lightweight": [
        "TA ALiBi 24L",
        "GQA ALiBi 24L",
        "GTA ALiBi 24L",
        "GQA RoPE 24L",
        "GQA PE 24L",
        "GTA PE 24L",
        "TA PE 24L",
    ],
    "Plot_5_The_Deep_End_30L": [
        "MHA ALiBi 30L",
        "TA ALiBi 30L",
        "GQA ALiBi 30L",
        "GTA ALiBi 30L",
        "MHA RoPE 30L",
        "GQA RoPE 30L",
    ],
    "Plot_6_Iso_KV_Cache": ["MHA IsoKV 6L", "GQA RoPE 24L", "GTA ALiBi 24L"],
    "Plot_7_Marginal_Utility_of_Depth": [
        "GTA ALiBi 24L",
        "GTA ALiBi 28L",
        "GTA ALiBi 30L",
        "GQA RoPE 24L",
        "GQA RoPE 28L",
        "GQA RoPE 30L",
    ],
}

# ---------------------------------------------------------
# TRAIN VS. VAL GAP PLOTTING GROUPS (Plots 8 and 9)
# ---------------------------------------------------------
GAP_GROUPS = {
    "Plot_8_Regularization_Proof_24L": ["Vanilla MHA 24L", "MHA ALiBi 24L", "GQA ALiBi 24L", "GTA ALiBi 24L"],
    "Plot_9_Regularization_Proof_30L": ["MHA ALiBi 30L", "GQA ALiBi 30L", "GTA ALiBi 30L"],
}


def get_exp_number(filename):
    try:
        return int(filename.split("_")[1])
    except:
        return 999


def get_clean_name(filename):
    base = filename.replace("_logs.json", "")
    parts = base.split("_", 2)
    if len(parts) == 3 and parts[0] == "Exp":
        return parts[2].replace("_", " ")
    return base.replace("_", " ")


def main():
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith("_logs.json")]
    log_files.sort(key=get_exp_number)

    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return

    # 1. Load all data into memory
    all_data = {}
    for log_file in log_files:
        clean_name = get_clean_name(log_file)
        filepath = os.path.join(LOG_DIR, log_file)

        exp_data = {"steps": []}
        for m in METRICS:
            exp_data[m] = []

        with open(filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                exp_data["steps"].append(row["opt_step"])
                for m in METRICS:
                    exp_data[m].append(row.get(m, None))

        all_data[clean_name] = exp_data

    cmap = plt.get_cmap("tab20")

    # ---------------------------------------------------------
    # ENGINE 1: Standard Metric Generation (Plots 1-7)
    # ---------------------------------------------------------
    for group_name, models_to_plot in STANDARD_GROUPS.items():
        group_out_dir = os.path.join(OUTPUT_BASE_DIR, group_name)
        os.makedirs(group_out_dir, exist_ok=True)

        print(f"\n📊 Generating standard dark plots for: {group_name.replace('_', ' ')}")

        for metric, title in zip(METRICS, TITLES):
            # Slightly darker facecolor for the figure itself to match blog backgrounds
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")

            for idx, model_name in enumerate(models_to_plot):
                if model_name not in all_data:
                    continue

                data = all_data[model_name]
                color = cmap(idx % 20)

                # Make LLaMA defaults and MHA baselines stand out slightly
                is_baseline = "Vanilla MHA" in model_name or "GQA RoPE" in model_name
                linewidth = 2.5 if is_baseline else 1.75
                linestyle = "--" if is_baseline else "-"

                ax.plot(
                    data["steps"],
                    data[metric],
                    label=model_name,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=0.9 if is_baseline else 0.85,  # Bumped alpha slightly for dark mode visibility
                )

            # Formatting
            ax.set_title(f"{group_name.replace('_', ' ')}: {title}", fontsize=16, fontweight="bold", pad=15)
            ax.set_xlabel("Optimizer Steps", fontsize=12)
            # Subtle grid for dark mode
            ax.grid(True, linestyle="--", alpha=0.3, color="gray")

            ax.set_yscale("log")
            ax.set_ylabel(f"{title} (Log Scale)", fontsize=12)

            ax.legend(
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
                fontsize=10,
                title="Architectures",
                title_fontsize=12,
                facecolor="#161b22",
                edgecolor="gray",  # Styled legend box
            )
            plt.tight_layout()

            output_filename = os.path.join(group_out_dir, f"{metric}.png")
            plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()

    # ---------------------------------------------------------
    # ENGINE 2: Train vs. Validation Gap (Plots 8 & 9)
    # ---------------------------------------------------------
    for group_name, models_to_plot in GAP_GROUPS.items():
        group_out_dir = os.path.join(OUTPUT_BASE_DIR, group_name)
        os.makedirs(group_out_dir, exist_ok=True)

        print(f"\n📈 Generating Train/Val Gap dark plots for: {group_name.replace('_', ' ')}")

        for base_metric in ["loss", "perplexity"]:
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")

            train_key = f"train_{base_metric}"
            val_key = f"val_{base_metric}"

            for idx, model_name in enumerate(models_to_plot):
                if model_name not in all_data:
                    continue

                data = all_data[model_name]
                color = cmap((idx * 2) % 20)

                # Plot Training (Dashed)
                ax.plot(
                    data["steps"],
                    data[train_key],
                    label=f"{model_name} (Train)",
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

                # Plot Validation (Solid, thick)
                ax.plot(
                    data["steps"],
                    data[val_key],
                    label=f"{model_name} (Val)",
                    color=color,
                    linestyle="-",
                    linewidth=2.5,
                    alpha=1.0,  # Fully opaque validation line for dark mode
                )

            # Formatting
            ax.set_title(
                f"{group_name.replace('_', ' ')}: Train vs. Val Gap ({base_metric.title()})",
                fontsize=16,
                fontweight="bold",
                pad=15,
            )
            ax.set_xlabel("Optimizer Steps", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.3, color="gray")

            ax.set_yscale("log")
            ax.set_ylabel(f"{base_metric.title()} (Log Scale)", fontsize=12)

            ax.legend(
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
                fontsize=10,
                title="Data Split",
                title_fontsize=12,
                facecolor="#161b22",
                edgecolor="gray",
            )
            plt.tight_layout()

            output_filename = os.path.join(group_out_dir, f"gap_{base_metric}.png")
            plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()

    print("\n🎉 All 9 dark plotting groups generated successfully!")


if __name__ == "__main__":
    main()
