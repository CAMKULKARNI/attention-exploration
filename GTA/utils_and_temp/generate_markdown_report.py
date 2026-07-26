import os
import json
from collections import defaultdict

LOG_DIR = "./training_logs"
OUTPUT_FILE = "training_report.md"


def get_clean_name(filename):
    """
    Truncates 'Exp_1_Vanilla_MHA_24L_logs.json' -> 'Vanilla MHA 24L'
    """
    base = filename.replace("_logs.json", "")
    parts = base.split("_", 2)  # Splits into ['Exp', '1', 'Vanilla_MHA_24L']
    if len(parts) == 3 and parts[0] == "Exp":
        return parts[2].replace("_", " ")
    return base.replace("_", " ")


def get_exp_number(filename):
    """Sort chronologically based on the experiment number."""
    try:
        return int(filename.split("_")[1])
    except (IndexError, ValueError):
        return 999


def main():
    """
    Reads all experiment log files and generates a markdown report
    with a table for each metric.
    """
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith("_logs.json")]
    log_files.sort(key=get_exp_number)

    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return

    # Structure: {metric: {exp_name: {step: value}}}
    all_data = defaultdict(lambda: defaultdict(dict))
    all_steps = set()
    all_metrics = set()
    exp_names = []

    print("Processing log files...")
    for log_file in log_files:
        clean_name = get_clean_name(log_file)
        exp_names.append(clean_name)
        filepath = os.path.join(LOG_DIR, log_file)

        with open(filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    step = row.pop("opt_step")
                    all_steps.add(step)

                    for metric, value in row.items():
                        all_metrics.add(metric)
                        all_data[metric][clean_name][step] = value
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping malformed line in {log_file}: {e}")

    sorted_steps = sorted(list(all_steps))
    sorted_metrics = sorted(list(all_metrics))

    print(f"Generating markdown report to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for metric in sorted_metrics:
            # Metric Header
            f.write(f"# {metric.replace('_', ' ').title()}\n\n")

            # Table Header
            header = "| Experiment Name | " + " | ".join(map(str, sorted_steps)) + " |\n"
            f.write(header)

            # Table Separator
            separator = "|---|" + "---|" * len(sorted_steps) + "\n"
            f.write(separator)

            # Table Rows
            for exp_name in exp_names:
                row_str = f"| {exp_name} |"
                for step in sorted_steps:
                    value = all_data[metric].get(exp_name, {}).get(step)
                    if value is not None:
                        # Format floats to have fewer decimal places
                        if isinstance(value, float):
                            row_str += f" {value:.4f} |"
                        else:
                            row_str += f" {value} |"
                    else:
                        row_str += " N/A |"
                f.write(row_str + "\n")

            f.write("\n")

    print("✅ Report generation complete!")


if __name__ == "__main__":
    main()
