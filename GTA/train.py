# train.py
import os
import argparse
import json
import time
import math
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.linen as nn
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
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./training_logs")
    parser.add_argument("--data_file", type=str, default="wikitext103_train.bin")
    return parser.parse_args()


# -----------------------------------------------------------------
# FAULT-TOLERANT STATE
# We subclass TrainState to track our exact position in the dataset.
# -----------------------------------------------------------------
class ResumableTrainState(train_state.TrainState):
    data_index: jnp.ndarray


def create_train_state(rng, model, config, args):
    dummy_input = jnp.zeros((args.micro_batch_size, args.seq_len), dtype=jnp.int32)

    print("      Initializing model weights...")
    variables = model.init(
        rng, dummy_input, use_causal_mask=True, current_pos=0, deterministic=True
    )

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
        warmup_steps=int(args.max_steps * 0.05),
        decay_steps=args.max_steps,
        end_value=args.learning_rate * 0.1,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1),
    )
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=args.grad_accum_steps)

    return ResumableTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        data_index=jnp.array(0, dtype=jnp.int32),
    )


@jax.jit
def train_step(state, batch, dropout_rng):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    def loss_fn(params):
        logits, _ = state.apply_fn(
            {"params": params},
            inputs,
            use_causal_mask=True,
            current_pos=0,
            caches=None,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=targets
        )
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    state = state.apply_updates(grads=grads)
    grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)])
    )

    return state, loss, grad_norm


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # O(1) MEMORY-MAPPED DATALOADER & MAX_STEPS CALCULATION
    # This MUST happen before create_train_state so the Optax schedule
    # can build its cosine decay curve using the exact 1-epoch step count.
    # -----------------------------------------------------------------
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Missing {args.data_file}. Run prepare_data.py first.")

    # Open array in Read-Only mode. Doesn't consume RAM until sliced.
    data_map = np.memmap(args.data_file, dtype=np.uint16, mode="r")

    tokens_per_micro_batch = args.micro_batch_size * (args.seq_len + 1)
    tokens_per_step = tokens_per_micro_batch * args.grad_accum_steps

    total_tokens = len(data_map)
    args.max_steps = total_tokens // tokens_per_step

    print(f"\n      📊 Dataset Size: {total_tokens:,} tokens")
    print(
        f"      🎯 Auto-configured max_steps to {args.max_steps} for exactly 1 Epoch."
    )

    # -----------------------------------------------------------------
    # CHECKPOINT MANAGER SETUP
    # -----------------------------------------------------------------
    ckpt_dir = os.path.join(
        args.output_dir, args.exp_name.replace(":", "").replace(" ", "_"), "checkpoints"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    ckpt_manager = ocp.CheckpointManager(os.path.abspath(ckpt_dir), options=options)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config_obj = Config(tokenizer)

    model = CausalLM(
        vocab_size=config_obj.vocab_size,
        max_seq_len=args.seq_len,
        latent_dim=config_obj.d_model,
        num_heads=config_obj.num_heads,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
        num_kv_heads=config_obj.num_kv_heads,
    )

    # Init RNGs
    rng = jax.random.PRNGKey(42)
    init_rng, dropout_rng = jax.random.split(rng)

    # Optax schedule is now created with the mathematically perfect max_steps
    state = create_train_state(init_rng, model, config_obj, args)

    # -----------------------------------------------------------------
    # RESUME LOGIC
    # -----------------------------------------------------------------
    if ckpt_manager.latest_step() is not None:
        print(f"      🔄 Resuming from step {ckpt_manager.latest_step()}...")
        state = ckpt_manager.restore(
            ckpt_manager.latest_step(), args=ocp.args.StandardRestore(state)
        )

    start_step = int(state.step)
    current_data_idx = int(state.data_index)

    print(f"\n🚀 [PID: {os.getpid()}] Starting Training: {args.exp_name}")
    print(f"      Resuming at Step: {start_step} | Data Index: {current_data_idx}")

    log_data = []
    log_file = os.path.join(
        args.output_dir, f"{args.exp_name.replace(':', '').replace(' ', '_')}_logs.json"
    )
    start_time = time.perf_counter()
    step_loss = 0.0

    for step in range(start_step + 1, args.max_steps + 1):
        # Slice exactly what we need
        end_idx = current_data_idx + tokens_per_micro_batch
        if end_idx > len(data_map):
            print("      ⚠️ Reached end of dataset! Ending epoch.")
            break

        raw_batch = data_map[current_data_idx:end_idx].astype(np.int32)
        batch = jnp.array(raw_batch.reshape(args.micro_batch_size, args.seq_len + 1))

        # Advance the index
        current_data_idx = end_idx
        state = state.replace(data_index=jnp.array(current_data_idx, dtype=jnp.int32))

        # Split dropout RNG for this specific step to ensure stochasticity
        dropout_rng, step_dropout_rng = jax.random.split(dropout_rng)
        state, loss, grad_norm = train_step(state, batch, step_dropout_rng)

        step_loss += loss

        if step % args.log_interval == 0:
            avg_loss = float(step_loss / args.log_interval)
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            gnorm = float(grad_norm)
            elapsed = time.perf_counter() - start_time
            steps_per_sec = args.log_interval / elapsed

            print(
                f"      Step {step:05d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | GradNorm: {gnorm:.2f} | Steps/s: {steps_per_sec:.2f}"
            )
            log_data.append(
                {"step": step, "loss": avg_loss, "perplexity": ppl, "grad_norm": gnorm}
            )

            step_loss = 0.0
            start_time = time.perf_counter()
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)

        if step % args.save_interval == 0 or step == args.max_steps:
            print(
                f"      💾 Saving checkpoint at step {step} (Data Index: {current_data_idx})..."
            )
            ckpt_manager.save(step, args=ocp.args.StandardSave(state))
            ckpt_manager.wait_until_finished()

    print(f"✅ Training for {args.exp_name} completed.")


if __name__ == "__main__":
    main()
