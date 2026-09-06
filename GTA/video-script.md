This is an excellent pivot. If you are aiming for a 1.5-hour, "Karpathy-style" deep dive, you have the perfect amount of technical material to fill that time without adding filler. An extended video allows you to transition from high-level theory to low-level JAX implementation seamlessly.

To pull off a 90-minute technical masterclass, you need to manage your screen real estate and keep the visual context engaging. Here is a comprehensive guide on how to structure the video, what images to use, and how to seamlessly switch between your sources.

### I. The Recording Setup (How to Switch Sources)

To smoothly transition between your face, the code, and the blogs without awkward pauses, you should use **OBS Studio** (Open Broadcaster Software). Set up the following "Scenes" before you hit record:

* **Scene 1: The Virtual Whiteboard (iPad / Excalidraw).** Use this to manually draw the KV cache memory blocks and explain the mathematical difference between RoPE and ALiBi.
* **Scene 2: The Jupyter Notebook & Terminal.** Have VS Code open with `experiment.ipynb` taking up the main screen.


* **Scene 3: The Code Deep-Dive.** Have your source code (`operators.py`, `train.py`, `benchmark.py`) open in VS Code with a slightly larger font size for readability.


* **Scene 4: The Research Paper / Blog.** Have your PDF (`GTA-training-blog_4.pdf`) open to reference the final graphs and learning curves.



---

### II. The 90-Minute Masterclass Timeline

Here is how you can stretch and pace your material to keep viewers hooked for an hour and a half.

#### Phase 1: The Physics of the Memory Wall (0:00 - 15:00)

* **The Hook:** Start with the SpongeBob waiting meme from your inference blog to illustrate the GPU starving for data during autoregressive decoding.


* **The Theory:** Switch to your whiteboard. Explain that inference is overwhelmingly memory-bandwidth bound, not compute-bound. Draw how Vanilla MHA allocates a Key and Value tensor for every single head.


* **The Notebook Proof:** Switch to `experiment.ipynb`. Run the cells defining the 4096 context window and show the audience the 16.00 MB per layer cost of Vanilla MHA.



#### Phase 2: Architecting the Solution (15:00 - 35:00)

* **Visualizing Compression:** Open the inference blog PDF and show the block diagram comparing MHA, MHA+RoPE, GQA+RoPE, TA+ALiBi, and GTA+ALiBi.


* **The Code:** Switch to `operators.py` (or the equivalent code on screen). Walk through the `GroupTiedAttention` class. Explain how you project a single unified `A` tensor instead of separate Keys and Values.


* **The Zero-Copy Trick:** Highlight the `jnp.broadcast_to` lines in your code. Explain how this edits the tensor's strides rather than allocating new physical memory.


* **The Live Proof:** Switch back to the Jupyter notebook and run the GTA cache cell, proving the 4x reduction down to 4.00 MB.



#### Phase 3: The Positional Encoding Dilemma (35:00 - 55:00)

* **The Spider-Man Meme:** Show the Spider-Man pointing meme to visually explain why RoPE ruins a unified tensor (rotating the Key effectively rotates the Value).


* **The ALiBi Solution:** Switch to your code and show the `get_alibi_slopes` function. Walk through how ALiBi applies a static linear penalty directly to the attention logits based on relative distance.


* **The Notebook Math:** Run the ALiBi cell in `experiment.ipynb` to prove that the ALiBi mask is dynamically computed and requires practically zero additional cache memory (0.06 MB).



#### Phase 4: Systems Engineering & The Training Loop (55:00 - 1:15:00)

* **The Noise Problem:** Open `benchmark.py` and explain the importance of isolating the XLA compiler memory allocations. Walk through how you subtract `post_warmup_bytes` from the peak bytes to get strict incremental VRAM.


* **The Async Trap:** Show the `jax.block_until_ready((logits, active_caches))` lines in both `benchmark.py` and your notebook. Explain that without this, Python timers will falsely report lightning-fast TPOTs because JAX dispatches asynchronously.


* **OOM Survival:** Open `train.py` and talk about the hardware traps you faced. Show the `release_ram()` function and the `gc.collect()` calls used to prevent memory double-booking during checkpoint restoration.



#### Phase 5: The Grand Finale (1:15:00 - 1:30:00)

* **The Iso-KV Showdown:** Run the final cell in your Jupyter notebook showing that 6 layers of MHA take the exact same memory as 24 layers of GTA (96.00 MB).


* **The Graphs:** Switch to the training blog PDF (`GTA-training-blog_4.pdf`). Show the "Iso-KV Cache Showdown" bar chart to prove that the 24-layer GTA model destroys the truncated 6-layer model in perplexity.


* **The Pareto Frontier:** Finally, show the "Hardware Cost vs Model Quality" scatter plot. Conclude with your Edge AI Blueprint, stating that the future of edge inference is trading memory-bound cache footprints for compute-bound depth.



---

### III. Key Images to Pre-Load

Have these specific images from your PDFs saved as standalone image files (PNG/JPG) so you can pop them up on screen easily via OBS:

1. **The Architecture Diagram:** From the inference blog, showing the flow from MHA down to GTA+ALiBi.


2. **KV Cache Scaling Curve:** From the training blog, visually proving the curve flattening at 4096 tokens.


3. **The Pareto Frontier Scatter Plot:** To show where MHA, GQA, and GTA sit regarding VRAM vs. Quality.


4. **The Dual-Axis Iso-KV Bar Chart:** The ultimate visual proof of depth reinvestment.



By structuring the video this way, you take the viewer on the exact same journey you went on: discovering the bottleneck, doing the math, writing the low-level JAX code, surviving the compiler traps, and ultimately proving the architecture.

Are you familiar with setting up OBS scenes for this kind of multi-window switching, or would you like some quick tips on how to configure your recording layout?