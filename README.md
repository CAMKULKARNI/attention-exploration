## Squeezing Transformers onto Edge Hardware 🧠📱A 29-Experiment Ablation on Attention, Memory, and Depth Reinvestment
This codebase contains a custom JAX/Flax pipeline used to execute a highly controlled, 29-experiment ablation sweep.
📂 Repository Structure
- operators.py: Contains the custom JAX/Flax implementations for Vanilla MHA, GQA, TA, and GTA.
- train.py: The core training loop used to process the WikiText-103 dataset.
- run_training.sh: Shell script to execute controlled training of all the variants.
- benchmark.py: The deterministic profiling suite used to measure Peak Incremental VRAM, TTFT, and TPOT across expanding context windows.
- run_benchmarks.sh: Shell script to execute randomized profiling loops for statistical validation.
