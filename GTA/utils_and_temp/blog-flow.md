### Squeezing Transformers onto Edge Hardware: A 29-Experiment Ablation on Attention, Memory, and Depth Reinvestment
## The Introduction: The Edge Computing Bottleneck

The Narrative: Introduce the memory crisis in LLMs (the KV cache bottleneck) and the goal of the 29-experiment sweep: finding the optimal architecture for edge devices.

The Hook: Give a sneak peek of the winner—Group Tied Attention (GTA) with ALiBi.

## The Cost of Compression: Shrinking Memory Without Losing Minds

The Narrative: Address the immediate fear every ML engineer has: If I compress my attention heads, will my model get stupider?

The Visuals:

val_perplexity_2.jpg (Cost of Compression): Shows that TA, GQA, and GTA actually match or outperform Vanilla MHA during training.

Plot 3 (Context Scaling Curve - from our previous script): Proves how this compression flattens the KV cache growth at 4096 tokens.

## The Positional Puzzle: Why ALiBi is the Edge Champion

The Narrative: Discuss the impact of positional encodings. Explain why Learned PE struggles and why RoPE's rotational overhead is beaten by ALiBi's linear bias in this setup.

The Visuals:

val_perplexity.jpg (Positional Confounder): Show the stark difference in learning curves between ALiBi (bottom/best) and Learned PE (top/worst).

Plot 5 (PE Ablation - from our script): For a clean final-number comparison.

## The Trade-Off Matrix: The Pareto Sweet Spot

The Narrative: Time to look at the intersection of hardware constraints and model quality. How fast are these models, and how much VRAM do they use?

The Visuals:

plot_1_pareto_frontier.png: Highlights the green "sweet spot" where GTA and GQA live.

plot_2_speed_vs_quality.jpg: Proves inference speed (TPOT) remains viable.

val_perplexity_4.jpg (Edge Compute Lightweight): Shows the learning dynamics of these specific Pareto-optimal 24L variants.

## Pushing the Limits: Depth Reinvestment (28L to 30L)

The Narrative: If we save VRAM by tying matrices, what happens if we spend that saved memory on more layers instead?

The Visuals:

val_perplexity_3.jpg (Iso Parameter Showdown): Compares deeper efficient models to shallower vanilla models.

val_perplexity_5.jpg (The Deep End 30L): Shows the absolute limit of performance reached in the sweep.

## The Mic Drop: The Iso-KV Cache Showdown

The Narrative: The ultimate proof of concept. If you have a strict hardware budget of ~2100 MB for KV cache, what is the best model you can deploy?

The Visuals:

val_perplexity_6.jpg (Iso KV Cache): The massive gap between the 6L Vanilla model and the 24L GTA model.

plot_4_isokv_showdown.png: The dual-axis bar chart that puts the final nail in the coffin, proving truncation (6L) is inferior to architectural compression (24L GTA).

## Conclusion & Open Source Release

The Narrative: Summarize the blueprint for Edge AI (GTA + ALiBi + Maximize Layers). Provide links to the GitHub repo, the Flax code, and the benchmarking scripts.