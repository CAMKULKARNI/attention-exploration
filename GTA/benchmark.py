import os
import argparse
import json
import time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from operators import CausalLM, Config

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--attn_type", type=str, required=True)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--prompt_len", type=int, required=True)
parser.add_argument("--decode_tokens", type=int, default=512)
parser.add_argument("--output_file", type=str, default="results.json")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
configuration = Config(tokenizer)


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# Cache initialization handles TA's single tensor vs MHA/GQA's paired tensors
def create_empty_caches(attn_type, batch_size, num_heads, num_kv_heads, max_len, depth, num_layers):
    caches = []
    for _ in range(num_layers):
        # Tied Attention variants (Single Tensor, Full Heads)
        if attn_type in ("ta", "ta_pe"):
            a = jnp.zeros((batch_size, num_heads, max_len, depth), dtype=jnp.bfloat16)
            caches.append(a)

        # Grouped Tied Attention variants (Single Tensor, Reduced Heads)
        elif attn_type in ("gta", "gta_pe"):
            a = jnp.zeros((batch_size, num_kv_heads, max_len, depth), dtype=jnp.bfloat16)
            caches.append(a)

        # Grouped Query Attention variants (Paired K/V, Reduced Heads)
        elif attn_type in ("gqa", "gqa_alibi", "gqa_pe"):
            k = jnp.zeros((batch_size, num_kv_heads, max_len, depth), dtype=jnp.bfloat16)
            v = jnp.zeros((batch_size, num_kv_heads, max_len, depth), dtype=jnp.bfloat16)
            caches.append((k, v))

        # Vanilla MHA variants (Paired K/V, Full Heads)
        elif attn_type in ("mha", "mha_rope", "mha_alibi"):
            k = jnp.zeros((batch_size, num_heads, max_len, depth), dtype=jnp.bfloat16)
            v = jnp.zeros((batch_size, num_heads, max_len, depth), dtype=jnp.bfloat16)
            caches.append((k, v))

        else:
            raise ValueError(f"Unknown attn_type '{attn_type}'. Check your bash script mapping.")
    return caches


def run_isolated_profile():
    print(f"\n🚀 [PID: {os.getpid()}] Running {args.exp_name} at Length {args.prompt_len}...")

    max_seq_len = args.prompt_len + args.decode_tokens
    depth = configuration.d_model // configuration.num_heads

    model = CausalLM(
        vocab_size=configuration.vocab_size,
        max_seq_len=max_seq_len,
        latent_dim=configuration.d_model,
        num_heads=configuration.num_heads,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
        num_kv_heads=configuration.num_kv_heads,
        use_causal_mask=True,
        deterministic=True,
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)
    print("      Initializing model weights...")
    variables = model.init(rng, dummy_input, current_pos=0)

    def cast_to_bf16_except_layernorm(path, leaf):
        is_layernorm = any("LayerNorm" in str(key) for key in path)
        if is_layernorm:
            return leaf
        if getattr(leaf, "dtype", None) == jnp.float32:
            return leaf.astype(jnp.bfloat16)
        return leaf

    variables = jax.tree_util.tree_map_with_path(cast_to_bf16_except_layernorm, variables)
    total_params = count_params(variables)

    # Cleaned signatures to match operators.py exactly
    @jax.jit
    def prefill_step(params, prompt, caches):
        return model.apply(params, prompt, current_pos=0, caches=caches)

    @jax.jit
    def decode_step(params, token, pos, caches):
        return model.apply(params, token, current_pos=pos, caches=caches)

    try:
        jax.clear_caches()

        rng_np = np.random.default_rng(42)
        prompt_input = jnp.array(
            rng_np.integers(0, configuration.vocab_size, size=(1, args.prompt_len), dtype=np.int32)
        )
        active_caches = create_empty_caches(
            args.attn_type, 1, configuration.num_heads, configuration.num_kv_heads, max_seq_len, depth, args.num_layers
        )

        print("      Compiling XLA Graphs (This will take a moment, but only once!)...")

        _logits, _caches = prefill_step(variables, prompt_input, active_caches)
        jax.block_until_ready((_logits, _caches))

        warmup_token = jnp.argmax(_logits[:, -1, :], axis=-1).astype(jnp.int32).reshape(1, 1)
        _logits, _caches = decode_step(variables, warmup_token, args.prompt_len, _caches)
        jax.block_until_ready((_logits, _caches))

        post_warmup_bytes = jax.local_devices()[0].memory_stats().get("bytes_in_use", 0)

        # -----------------------------------------------------------------
        # ACTUAL PROFILING
        # -----------------------------------------------------------------
        start_time = time.perf_counter()
        logits, active_caches = prefill_step(variables, prompt_input, active_caches)
        jax.block_until_ready(active_caches)
        ttft = time.perf_counter() - start_time

        next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32).reshape(1, 1)

        decode_times = []
        for i in range(args.decode_tokens):
            pos = args.prompt_len + i
            start_step = time.perf_counter()
            logits, active_caches = decode_step(variables, next_token, pos, active_caches)
            jax.block_until_ready((logits, active_caches))
            decode_times.append(time.perf_counter() - start_step)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32).reshape(1, 1)

        mem_stats = jax.local_devices()[0].memory_stats()
        peak_vram_mb = (mem_stats.get("peak_bytes_in_use", 0) - post_warmup_bytes) / (1024**2)
        steady_vram_mb = (mem_stats.get("bytes_in_use", 0) - post_warmup_bytes) / (1024**2)

        warmup_steps = 10
        stable_times = decode_times[warmup_steps:]
        tpot_mean = float(np.mean(stable_times))
        tpot_std = float(np.std(stable_times))
        tpot_p95 = float(np.percentile(stable_times, 95))

        print(
            f"      Params: {total_params / 1e6:.2f}M | VRAM: {peak_vram_mb:.2f} MB | TTFT: {ttft:.4f}s | TPOT: {tpot_mean:.4f}s"
        )

        result_data = {
            "exp_name": args.exp_name,
            "attn_type": args.attn_type,
            "layers": args.num_layers,
            "prompt_len": args.prompt_len,
            "params": total_params,
            "vram_incremental_peak_mb": peak_vram_mb,
            "vram_incremental_steady_mb": steady_vram_mb,
            "ttft": ttft,
            "tpot_mean": tpot_mean,
            "tpot_std": tpot_std,
            "tpot_p95": tpot_p95,
            "tpot_all": [float(t) for t in decode_times],
        }

        if os.path.exists(args.output_file):
            with open(args.output_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(result_data)
        with open(args.output_file, "w") as f:
            json.dump(data, f, indent=4)

    except RuntimeError as e:
        print(f"      [!] OOM Crash or Runtime Error: {e}. Writing failure record.")
        failure_data = {
            "exp_name": args.exp_name,
            "attn_type": args.attn_type,
            "layers": args.num_layers,
            "prompt_len": args.prompt_len,
            "status": "FAILED",
            "error": str(e),
        }
        if os.path.exists(args.output_file):
            with open(args.output_file, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(failure_data)
        with open(args.output_file, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    run_isolated_profile()
