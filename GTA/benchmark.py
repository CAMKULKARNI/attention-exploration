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
parser.add_argument("--decode_tokens", type=int, default=100)
parser.add_argument("--output_file", type=str, default="results.json")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
configuration = Config(tokenizer)


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# Cache initialization handles GTA's single tensor vs MHA/GQA's paired tensors
def create_empty_caches(
    attn_type, batch_size, num_heads, num_kv_heads, max_len, depth, num_layers
):
    caches = []
    for _ in range(num_layers):
        if attn_type == "gta":
            a = jnp.zeros((batch_size, num_heads, max_len, depth), dtype=jnp.bfloat16)
            caches.append(a)
        elif attn_type == "gqa":
            k = jnp.zeros(
                (batch_size, num_kv_heads, max_len, depth), dtype=jnp.bfloat16
            )
            v = jnp.zeros(
                (batch_size, num_kv_heads, max_len, depth), dtype=jnp.bfloat16
            )
            caches.append((k, v))
        else:  # mha
            k = jnp.zeros((batch_size, num_heads, max_len, depth), dtype=jnp.bfloat16)
            v = jnp.zeros((batch_size, num_heads, max_len, depth), dtype=jnp.bfloat16)
            caches.append((k, v))
    return caches


def run_isolated_profile():
    print(
        f"\n🚀 [PID: {os.getpid()}] Running {args.exp_name} at Length {args.prompt_len}..."
    )

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
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)
    print("      Initializing model weights...")
    variables = model.init(rng, dummy_input, use_causal_mask=False, current_pos=0)

    # -----------------------------------------------------------------
    # FIX: RESTORE GLOBAL BFLOAT16 POLICY
    # Cast all parameters to 16-bit to match the Keras footprint
    # -----------------------------------------------------------------
    variables = jax.tree_util.tree_map(
        lambda x: (
            x.astype(jnp.bfloat16) if getattr(x, "dtype", None) == jnp.float32 else x
        ),
        variables,
    )

    total_params = count_params(variables)

    # -----------------------------------------------------------------
    # JIT COMPILATION & ALIAS ANALYSIS
    # XLA fusion prevents the 1 GB attention matrices from materializing.
    # dynamic_update_slice ensures O(1) in-place memory mutations.
    # -----------------------------------------------------------------
    @jax.jit
    def prefill_step(params, prompt, caches):
        return model.apply(
            params, prompt, use_causal_mask=True, current_pos=0, caches=caches
        )

    @jax.jit
    def decode_step(params, token, pos, caches):
        return model.apply(
            params, token, use_causal_mask=False, current_pos=pos, caches=caches
        )

    try:
        jax.clear_caches()

        prompt_input = jnp.ones((1, args.prompt_len), dtype=jnp.int32)
        active_caches = create_empty_caches(
            args.attn_type,
            1,
            configuration.num_heads,
            configuration.num_kv_heads,
            max_seq_len,
            depth,
            args.num_layers,
        )

        # -----------------------------------------------------------------
        # FIX: WARMUP BOTH COMPILED GRAPHS
        # We must compile BOTH the prefill and the decode kernels before
        # starting any timers.
        # -----------------------------------------------------------------
        print("      Compiling XLA Graphs (This will take a moment, but only once!)...")

        # 1. Warm up Prefill Graph (Absorbs the ~18 second compile tax)
        _logits, _caches = prefill_step(variables, prompt_input, active_caches)
        _logits.block_until_ready()

        # 2. Warm up Decode Graph (Absorbs the ~3 second compile tax)
        warmup_token = jnp.ones((1, 1), dtype=jnp.int32)
        _logits, _ = decode_step(variables, warmup_token, args.prompt_len, _caches)
        _logits.block_until_ready()

        # -----------------------------------------------------------------
        # ACTUAL PROFILING
        # -----------------------------------------------------------------

        # TTFT (Prefill)
        start_time = time.perf_counter()
        logits, active_caches = prefill_step(variables, prompt_input, active_caches)
        next_token = (
            jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32).reshape(1, 1)
        )
        next_token.block_until_ready()
        ttft = time.perf_counter() - start_time

        # TPOT (Decode)
        decode_times = []
        for i in range(args.decode_tokens):
            pos = jnp.array(args.prompt_len + i, dtype=jnp.int32)

            start_step = time.perf_counter()
            logits, active_caches = decode_step(
                variables, next_token, pos, active_caches
            )
            next_token = (
                jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32).reshape(1, 1)
            )
            next_token.block_until_ready()
            decode_times.append(time.perf_counter() - start_step)

        tpot = np.mean(decode_times)

        mem_stats = jax.local_devices()[0].memory_stats()
        peak_vram_mb = mem_stats.get("peak_bytes_in_use", 0) / (1024**2)

        print(
            f"      Params: {total_params / 1e6:.2f}M | VRAM: {peak_vram_mb:.2f} MB | TTFT: {ttft:.4f}s | TPOT: {tpot:.4f}s"
        )

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

    except RuntimeError as e:
        print(f"      [!] OOM Crash or Runtime Error: {e}. Skipping JSON append.")


if __name__ == "__main__":
    run_isolated_profile()
