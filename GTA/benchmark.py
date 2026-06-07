import os
import argparse
import json
import gc
import time
import numpy as np

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
from keras import ops
from transformers import AutoTokenizer
from operators import CausalLM, Config

keras.config.set_floatx("bfloat16")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--attn_type", type=str, required=True)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--prompt_len", type=int, required=True)
parser.add_argument("--decode_tokens", type=int, default=100)
parser.add_argument("--output_file", type=str, default="results.json")
args = parser.parse_args()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
configuration = Config(tokenizer)


def run_isolated_profile():
    print(
        f"\n🚀 [PID: {os.getpid()}] Running {args.exp_name} at Length {args.prompt_len}..."
    )

    model = CausalLM(
        vocab_size=configuration.vocab_size,
        max_seq_len=args.prompt_len + args.decode_tokens,
        latent_dim=configuration.d_model,
        num_heads=configuration.num_heads,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
    )

    # 1. Force parameter initialization to get param count
    _ = model(ops.zeros((1, 1), dtype="int32"))
    total_params = model.count_params()

    # 2. WARMUP
    print("      Warming up GPU kernels (Eager execution)...")
    warmup_prompt = ops.ones((1, args.prompt_len), dtype="int32")

    w_logits, w_caches = model(warmup_prompt, use_causal_mask=True, current_pos=0)
    w_token = ops.cast(ops.argmax(w_logits[:, -1, :], axis=-1), dtype="int32")
    w_token = ops.reshape(w_token, (1, 1))

    for i in range(2):  # Short warmup for eager mode
        pos = ops.convert_to_tensor(args.prompt_len + i, dtype="int32")
        w_logits, w_caches = model(
            w_token, use_causal_mask=False, current_pos=pos, caches=w_caches
        )
        w_token = ops.cast(ops.argmax(w_logits[:, -1, :], axis=-1), dtype="int32")
        w_token = ops.reshape(w_token, (1, 1))

    del warmup_prompt, w_token, w_logits, w_caches
    gc.collect()

    # 3. ACTUAL PROFILING
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        prompt_input = ops.ones((1, args.prompt_len), dtype="int32")

        # TTFT (Prefill)
        start_time = time.perf_counter()
        logits, active_caches = model(prompt_input, use_causal_mask=True, current_pos=0)
        next_token = ops.cast(ops.argmax(logits[:, -1, :], axis=-1), dtype="int32")
        next_token = ops.reshape(next_token, (1, 1))
        _ = next_token.numpy()
        ttft = time.perf_counter() - start_time

        # TPOT (Decode)
        decode_times = []
        for i in range(args.decode_tokens):
            pos = ops.convert_to_tensor(args.prompt_len + i, dtype="int32")

            start_step = time.perf_counter()
            logits, active_caches = model(
                next_token, use_causal_mask=False, current_pos=pos, caches=active_caches
            )
            next_token = ops.cast(ops.argmax(logits[:, -1, :], axis=-1), dtype="int32")
            next_token = ops.reshape(next_token, (1, 1))
            _ = next_token.numpy()
            decode_times.append(time.perf_counter() - start_step)

        tpot = np.mean(decode_times)

        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        peak_vram_mb = mem_info["peak"] / (1024**2)

        print(
            f"      Params: {total_params / 1e6:.2f}M | VRAM: {peak_vram_mb:.2f} MB | TTFT: {ttft:.4f}s | TPOT: {tpot:.4f}s"
        )

        # 4. APPEND TO JSON
        result_data = {
            "exp_name": args.exp_name,
            "attn_type": args.attn_type,
            "layers": args.num_layers,
            "prompt_len": args.prompt_len,
            "params": total_params,
            "vram": peak_vram_mb,
            "ttft": ttft,
            "tpot": tpot,
        }

        if os.path.exists(args.output_file):
            with open(args.output_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(result_data)
        with open(args.output_file, "w") as f:
            json.dump(data, f, indent=4)

    except tf.errors.ResourceExhaustedError:
        print(f"      [!] OOM Crash. Skipping JSON append.")


if __name__ == "__main__":
    run_isolated_profile()
