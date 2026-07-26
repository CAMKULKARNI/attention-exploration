import json
import statistics
from collections import defaultdict


def generate_report(json_file_path, output_md_file):
    """
    Reads experiment results from a JSON file, calculates the median of the
    metrics across runs for each experiment, and writes a summary report
    in Markdown format.

    Args:
        json_file_path (str): The path to the input JSON file.
        output_md_file (str): The path to the output Markdown file.
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}.")
        return

    # Group experiments by name and prompt length
    experiments = defaultdict(list)
    for record in data:
        # Create a unique key for each experiment configuration
        exp_key = (record["exp_name"], record["prompt_len"])
        experiments[exp_key].append(record)

    with open(output_md_file, "w") as f:
        f.write("# Experiment Results Summary\n\n")

        # Sort experiments for consistent output
        # Sort by experiment number (as an integer) and then by prompt length.
        sorted_keys = sorted(experiments.keys(), key=lambda k: (int(k[0].split(":")[0].split(" ")[1]), k[1]))
        for exp_key in sorted_keys:
            runs = experiments[exp_key]
            f.write(f"## {exp_key[0]} - Prompt Length: {exp_key[1]}\n\n")

            if not runs:
                f.write("No data for this experiment.\n\n")
                continue

            # Use the first run for static data and headers
            first_run = runs[0]
            headers = [key for key in first_run.keys() if key != "tpot_all"]

            f.write("| Metric | Median Value |\n")
            f.write("|---|---|\n")

            for header in headers:
                # For non-numeric fields, just show the value from the first run
                if isinstance(first_run[header], str):
                    f.write(f"| {header} | {first_run[header]} |\n")
                else:
                    # Calculate median for numeric fields
                    values = [run[header] for run in runs if header in run]
                    median_value = statistics.median(values)
                    f.write(f"| {header} | {median_value:.4f} |\n")
            f.write("\n")


if __name__ == "__main__":
    generate_report("/home/Development/attention-exploration/GTA/results.json", "results_summary.md")
    print("Report 'results_summary.md' generated successfully.")
