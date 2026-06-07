import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_FILE = "results.json"

with open(INPUT_FILE, "r") as f:
    raw_data = json.load(f)

# 1. Group data by (exp_name, prompt_len)
# Using a nested dictionary structure for easy median calculation
grouped_data = defaultdict(lambda: defaultdict(lambda: {"vram": [], "ttft": [], "tpot": []}))

for entry in raw_data:
    name = entry["exp_name"]
    length = entry["prompt_len"]
    
    grouped_data[name][length]["vram"].append(entry["vram"])
    grouped_data[name][length]["ttft"].append(entry["ttft"])
    grouped_data[name][length]["tpot"].append(entry["tpot"])

# 2. Calculate Median and structure for plotting
experiments = {}
for name, lengths_dict in grouped_data.items():
    experiments[name] = {"lengths": [], "vram": [], "ttft": [], "tpot": []}
    
    # Sort lengths to ensure the plot lines draw correctly left to right
    for length in sorted(lengths_dict.keys()):
        metrics = lengths_dict[length]
        experiments[name]["lengths"].append(length)
        
        # Calculate the median for the 5 runs
        experiments[name]["vram"].append(np.median(metrics["vram"]))
        experiments[name]["ttft"].append(np.median(metrics["ttft"]))
        experiments[name]["tpot"].append(np.median(metrics["tpot"]))

print("Generating median-aggregated plots...")

# 1. VRAM Plot
plt.figure(figsize=(10, 6))
for name, data in experiments.items():
    plt.plot(data["lengths"], data["vram"], marker="o", label=name)
plt.title("Peak VRAM Footprint vs Sequence Length (5-Run Median)")
plt.xlabel("Initial Prompt Length")
plt.ylabel("VRAM (MB)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("benchmark_vram.png", dpi=300)
plt.close()

# 2. TTFT Plot
plt.figure(figsize=(10, 6))
for name, data in experiments.items():
    plt.plot(data["lengths"], data["ttft"], marker="s", label=name)
plt.title("TTFT (Prefill Latency) vs Sequence Length (5-Run Median)")
plt.xlabel("Initial Prompt Length")
plt.ylabel("Seconds")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("benchmark_ttft.png", dpi=300)
plt.close()

# 3. TPOT Plot
plt.figure(figsize=(10, 6))
for name, data in experiments.items():
    plt.plot(data["lengths"], data["tpot"], marker="^", label=name)
plt.title("TPOT (Decoding Latency) vs Sequence Length (5-Run Median)")
plt.xlabel("Initial Prompt Length")
plt.ylabel("Seconds per Token")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("benchmark_tpot.png", dpi=300)
plt.close()

print("🎉 Complete! Check the .png files.")