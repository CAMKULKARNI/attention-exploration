# train.py
import os
import argparse
import json
import math
import numpy as np
import gc
import resource
import functools
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from transformers import AutoTokenizer

from operators import CausalLM, Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--attn_type", type=str, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--save_interval", type=int, default=148)
    parser.add_argument("--output_dir", type=str, default="./training_logs")
    parser.add_argument("--train_data_file", type=str, default="wikitext103_train.bin")
    parser.add_argument("--val_data_file", type=str, default="wikitext103_val.bin")
    return parser.parse_args()


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def release_ram():
    gc.collect()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def log_runtime_state(tag):
    try:
        stats = jax.devices()[0].memory_stats()
        in_use = stats.get("bytes_in_use", 0) / 1e9
        peak = stats.get("peak_bytes_in_use", 0) / 1e9
        print(f"      📈 [{tag}] GPU bytes_in_use={in_use:.2f}GB peak={peak:.2f}GB")
    except Exception as e:
        pass
    print(f"      📈 [{tag}] Host RSS={rss_gb():.2f}GiB")


class ResumableTrainState(train_state.TrainState):
    data_index: jnp.ndarray


def create_train_state(rng, model, args, max_opt_steps):
    dummy_input = jnp.zeros((args.micro_batch_size, args.seq_len), dtype=jnp.int32)
    print("      Initializing model weights...")
    variables = model.init(rng, dummy_input, current_pos=0)

    def cast_to_bf16_except_layernorm(path, leaf):
        is_layernorm = any("LayerNorm" in str(key) for key in path)
        if is_layernorm:
            return leaf
        if getattr(leaf, "dtype", None) == jnp.float32:
            return leaf.astype(jnp.bfloat16)
        return leaf

    variables = jax.tree_util.tree_map_with_path(
        cast_to_bf16_except_layernorm, variables
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=int(max_opt_steps * 0.05),
        decay_steps=max_opt_steps,
        end_value=args.learning_rate * 0.1,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1, mu_dtype=jnp.bfloat16),
    )

    return ResumableTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        data_index=jnp.array(0, dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnums=(1,))
def train_step(state, model, macro_batch, dropout_rng):
    def body_fn(carry, micro_batch):
        rng, accum_loss, accum_grads = carry
        rng, step_rng = jax.random.split(rng)

        inputs = micro_batch[:, :-1]
        targets = micro_batch[:, 1:]

        def loss_fn(params):
            logits, _ = model.apply(
                {"params": params},
                inputs,
                current_pos=0,
                caches=None,
                rngs={"dropout": step_rng},
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets
            )
            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        accum_loss = accum_loss + loss
        accum_grads = jax.tree_util.tree_map(
            lambda a, b: a + b.astype(jnp.float32), accum_grads, grads
        )

        return (rng, accum_loss, accum_grads), None

    init_loss = jnp.array(0.0)
    init_grads = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=jnp.float32), state.params
    )
    init_carry = (dropout_rng, init_loss, init_grads)

    final_carry, _ = jax.lax.scan(body_fn, init_carry, macro_batch)
    _, total_loss, total_grads = final_carry

    num_micro_batches = macro_batch.shape[0]
    avg_loss = total_loss / num_micro_batches
    avg_grads = jax.tree_util.tree_map(
        lambda x: (x / num_micro_batches).astype(jnp.bfloat16), total_grads
    )

    state = state.apply_gradients(grads=avg_grads)

    grad_norm = jnp.sqrt(
        sum(
            [
                jnp.sum(jnp.square(x.astype(jnp.float32)))
                for x in jax.tree_util.tree_leaves(avg_grads)
            ]
        )
    )

    return state, avg_loss, grad_norm


@functools.partial(jax.jit, static_argnums=(1,))
def eval_step(state, model, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits, _ = model.apply(
        {"params": state.params},
        inputs,
        current_pos=0,
        caches=None,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=targets
    )
    return jnp.mean(loss)


def run_validation(state, model_eval, val_data_map, args):
    tokens_per_batch = args.micro_batch_size * (args.seq_len + 1)
    num_batches = len(val_data_map) // tokens_per_batch
    total_val_loss = 0.0

    pbar = tqdm(range(num_batches), desc="      🔬 Validating", leave=False)
    for i in pbar:
        start_idx = i * tokens_per_batch
        end_idx = start_idx + tokens_per_batch
        raw_batch = np.array(val_data_map[start_idx:end_idx], dtype=np.int32).reshape(
            args.micro_batch_size, args.seq_len + 1
        )
        batch = jnp.asarray(raw_batch)
        del raw_batch

        loss_val = float(eval_step(state, model_eval, batch))
        total_val_loss += loss_val
        del batch
        pbar.set_postfix({"Loss": f"{loss_val:.4f}"})

    avg_val_loss = total_val_loss / num_batches
    val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
    return avg_val_loss, val_ppl


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.train_data_file) or not os.path.exists(
        args.val_data_file
    ):
        raise FileNotFoundError("Missing .bin files. Run prepare_data.py first.")

    print("      Loading datasets entirely into RAM...")
    train_data_map = np.fromfile(args.train_data_file, dtype=np.uint16)
    val_data_map = np.fromfile(args.val_data_file, dtype=np.uint16)

    tokens_per_macro_batch = (
        args.grad_accum_steps * args.micro_batch_size * (args.seq_len + 1)
    )

    total_train_tokens = len(train_data_map)
    max_opt_steps = total_train_tokens // tokens_per_macro_batch

    print(f"      📊 Train Dataset: {total_train_tokens:,} tokens")
    print(f"      🎯 1 Epoch requires: {max_opt_steps} optimizer steps.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config_obj = Config(tokenizer)

    model_train = CausalLM(
        vocab_size=config_obj.vocab_size,
        max_seq_len=args.seq_len,
        latent_dim=config_obj.d_model,
        num_heads=config_obj.num_heads,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
        num_kv_heads=config_obj.num_kv_heads,
        use_causal_mask=True,
        deterministic=False,
    )

    model_eval = CausalLM(
        vocab_size=config_obj.vocab_size,
        max_seq_len=args.seq_len,
        latent_dim=config_obj.d_model,
        num_heads=config_obj.num_heads,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
        num_kv_heads=config_obj.num_kv_heads,
        use_causal_mask=True,
        deterministic=True,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, dropout_rng = jax.random.split(rng)

    state = create_train_state(init_rng, model_train, args, max_opt_steps)

    print(f"\n      🔥 Warming up XLA compilation and aligning memory layouts...")
    dummy_macro_batch = jnp.zeros(
        (args.grad_accum_steps, args.micro_batch_size, args.seq_len + 1),
        dtype=jnp.int32,
    )
    dummy_eval_batch = jnp.zeros(
        (args.micro_batch_size, args.seq_len + 1), dtype=jnp.int32
    )

    _warmup_state, _, _ = train_step(state, model_train, dummy_macro_batch, dropout_rng)
    del _warmup_state
    _ = eval_step(state, model_eval, dummy_eval_batch)

    jax.block_until_ready(state)
    print(f"      ✅ Warmup complete. Fast kernels locked.\n")

    start_opt_step = 0
    current_data_idx = 0

    print(f"🚀 [PID: {os.getpid()}] Starting Training: {args.exp_name}")
    print(f"      RSS at training start: {rss_gb():.2f} GiB")

    log_file = os.path.join(
        args.output_dir, f"{args.exp_name.replace(':', '').replace(' ', '_')}_logs.json"
    )

    pbar = tqdm(
        range(start_opt_step + 1, max_opt_steps + 1),
        desc=f"Training {args.exp_name}",
        initial=start_opt_step,
        total=max_opt_steps,
    )

    for opt_step in pbar:
        end_idx = current_data_idx + tokens_per_macro_batch
        if end_idx > len(train_data_map):
            tqdm.write("      ⚠️ Reached end of dataset! Ending epoch.")
            break

        # Slicing from np.fromfile is an instant O(1) RAM operation
        raw_batch = np.array(
            train_data_map[current_data_idx:end_idx], dtype=np.int32
        ).reshape(args.grad_accum_steps, args.micro_batch_size, args.seq_len + 1)
        macro_batch = jnp.asarray(raw_batch)
        del raw_batch

        current_data_idx = end_idx
        state = state.replace(data_index=jnp.array(current_data_idx, dtype=jnp.int32))

        dropout_rng, step_dropout_rng = jax.random.split(dropout_rng)

        state, loss, grad_norm = train_step(
            state, model_train, macro_batch, step_dropout_rng
        )
        del macro_batch

        avg_train_loss = float(loss)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
        gnorm = float(grad_norm)

        current_rss = rss_gb()
        pbar.set_postfix(
            {
                "Loss": f"{avg_train_loss:.4f}",
                "PPL": f"{train_ppl:.2f}",
                "RSS": f"{current_rss:.1f}G",
            }
        )

        if opt_step % args.save_interval == 0 or opt_step == max_opt_steps:
            val_loss, val_ppl = run_validation(state, model_eval, val_data_map, args)
            tqdm.write(
                f"      ⭐ Validation Complete | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}\n"
            )

            jax.block_until_ready((val_loss, val_ppl))
            release_ram()

            log_dict = {
                "opt_step": opt_step,
                "train_loss": avg_train_loss,
                "train_perplexity": train_ppl,
                "grad_norm": gnorm,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            }
            with open(log_file, "a") as f:
                json.dump(log_dict, f)
                f.write("\n")

    print(f"✅ Training for {args.exp_name} completed successfully.")

    # -----------------------------------------------------------------
    # ONE-TIME FINAL SAVE
    # No periodic checkpointing anymore (that was the whole investigation) -
    # save weights exactly once, here, at the very end of the run.
    # -----------------------------------------------------------------
    safe_name = args.exp_name.replace(":", "").replace(" ", "_")
    final_ckpt_dir = os.path.join(args.output_dir, f"{safe_name}_final_weights")
    tqdm.write(f"      💾 Saving final weights to {final_ckpt_dir} ...")
    jax.block_until_ready(state)
    ocp.StandardCheckpointer().save(
        os.path.abspath(final_ckpt_dir), state.params, force=True
    )
    tqdm.write(f"      ✅ Final weights saved.")

    # Write completion marker for the bash script
    done_file = os.path.join(args.output_dir, f"{safe_name}_DONE.txt")
    with open(done_file, "w") as f:
        f.write("COMPLETED\n")


if __name__ == "__main__":
    main()
