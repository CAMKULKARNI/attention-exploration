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
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./training_logs")
    parser.add_argument("--train_data_file", type=str, default="wikitext103_train.bin")
    parser.add_argument("--val_data_file", type=str, default="wikitext103_val.bin")
    return parser.parse_args()


class ResumableTrainState(train_state.TrainState):
    data_index: jnp.ndarray


def create_train_state(rng, model, args, max_opt_steps):
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
        warmup_steps=int(max_opt_steps * 0.05),
        decay_steps=max_opt_steps,
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

    state = state.apply_gradients(grads=grads)

    # Grad norm computed on the per-micro-batch gradient for observability.
    # Note: this reports the variance of individual micro-batch gradients,
    # not the accumulated gradient that AdamW actually applies once every
    # grad_accum_steps calls -- expect this to look noisier than a true
    # per-optimizer-step norm, by design.
    grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)])
    )

    return state, loss, grad_norm


@jax.jit
def eval_step(state, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits, _ = state.apply_fn(
        {"params": state.params},
        inputs,
        use_causal_mask=True,
        current_pos=0,
        caches=None,
        deterministic=True,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=targets
    )
    return jnp.mean(loss)


def run_validation(state, val_data_map, args):
    tokens_per_batch = args.micro_batch_size * (args.seq_len + 1)
    num_batches = len(val_data_map) // tokens_per_batch
    total_val_loss = 0.0

    for i in range(num_batches):
        start_idx = i * tokens_per_batch
        end_idx = start_idx + tokens_per_batch
        raw_batch = val_data_map[start_idx:end_idx].astype(np.int32)
        batch = jnp.array(raw_batch.reshape(args.micro_batch_size, args.seq_len + 1))

        # -----------------------------------------------------------------
        # MEMORY LEAK FIX (Validation)
        # -----------------------------------------------------------------
        total_val_loss += float(eval_step(state, batch))

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

    train_data_map = np.memmap(args.train_data_file, dtype=np.uint16, mode="r")
    val_data_map = np.memmap(args.val_data_file, dtype=np.uint16, mode="r")

    tokens_per_micro_batch = args.micro_batch_size * (args.seq_len + 1)

    total_train_tokens = len(train_data_map)
    max_micro_steps = total_train_tokens // tokens_per_micro_batch
    max_opt_steps = max_micro_steps // args.grad_accum_steps

    print(f"\n      📊 Train Dataset: {total_train_tokens:,} tokens")
    print(
        f"      🎯 1 Epoch requires: {max_opt_steps} optimizer steps ({max_micro_steps} micro-batches)."
    )

    # -----------------------------------------------------------------
    # METHODOLOGY NOTE (Causal Transitions)
    # The dataloader dynamically slices 1D arrays into 2D chunks of shape
    # (batch_size, seq_len + 1). Consequently, the causal link between the
    # last token of Row N and the first token of Row N+1 is discarded.
    # We drop (batch_size - 1) causal edges per micro-batch. Across 103M
    # tokens, this is statistically negligible and preserves the O(1) speed
    # of the memory-mapped loader.
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

    rng = jax.random.PRNGKey(42)
    init_rng, dropout_rng = jax.random.split(rng)

    state = create_train_state(init_rng, model, args, max_opt_steps)

    if ckpt_manager.latest_step() is not None:
        print(f"      🔄 Resuming from optimizer step {ckpt_manager.latest_step()}...")
        state = ckpt_manager.restore(
            ckpt_manager.latest_step(), args=ocp.args.StandardRestore(state)
        )

    start_micro_step = int(state.step)
    current_data_idx = int(state.data_index)

    print(f"\n🚀 [PID: {os.getpid()}] Starting Training: {args.exp_name}")
    print(
        f"      Resuming at Micro-Step: {start_micro_step} | Data Index: {current_data_idx}"
    )

    log_data = []
    log_file = os.path.join(
        args.output_dir, f"{args.exp_name.replace(':', '').replace(' ', '_')}_logs.json"
    )
    start_time = time.perf_counter()
    accumulated_loss = 0.0

    for micro_step in range(start_micro_step + 1, max_micro_steps + 1):
        end_idx = current_data_idx + tokens_per_micro_batch
        if end_idx > len(train_data_map):
            print("      ⚠️ Reached end of dataset! Ending epoch.")
            break

        raw_batch = train_data_map[current_data_idx:end_idx].astype(np.int32)
        batch = jnp.array(raw_batch.reshape(args.micro_batch_size, args.seq_len + 1))

        current_data_idx = end_idx
        state = state.replace(data_index=jnp.array(current_data_idx, dtype=jnp.int32))

        dropout_rng, step_dropout_rng = jax.random.split(dropout_rng)

        # JIT-compiled training step
        state, loss, grad_norm = train_step(state, batch, step_dropout_rng)

        # -----------------------------------------------------------------
        # MEMORY LEAK FIX (Training)
        # -----------------------------------------------------------------
        accumulated_loss += float(loss)

        if micro_step % args.grad_accum_steps == 0:
            opt_step = micro_step // args.grad_accum_steps

            if opt_step % args.log_interval == 0:
                avg_train_loss = accumulated_loss / (
                    args.log_interval * args.grad_accum_steps
                )
                train_ppl = (
                    math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
                )
                gnorm = float(grad_norm)

                elapsed = time.perf_counter() - start_time
                steps_per_sec = args.log_interval / elapsed

                print(
                    f"      Opt Step {opt_step:05d}/{max_opt_steps} | "
                    f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
                    f"GradNorm: {gnorm:.2f} | Steps/s: {steps_per_sec:.2f}"
                )

                log_data.append(
                    {
                        "opt_step": opt_step,
                        "train_loss": avg_train_loss,
                        "train_perplexity": train_ppl,
                        "grad_norm": gnorm,
                    }
                )

                accumulated_loss = 0.0
                start_time = time.perf_counter()

                with open(log_file, "w") as f:
                    json.dump(log_data, f, indent=4)

            if opt_step % args.save_interval == 0 or opt_step == max_opt_steps:
                print(f"      🔬 Running validation over full hold-out set...")
                val_loss, val_ppl = run_validation(state, val_data_map, args)
                print(
                    f"      ⭐ Validation Complete | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                )

                log_data[-1]["val_loss"] = val_loss
                log_data[-1]["val_perplexity"] = val_ppl
                with open(log_file, "w") as f:
                    json.dump(log_data, f, indent=4)

                print(
                    f"      💾 Saving checkpoint at Opt Step {opt_step} (Data Index: {current_data_idx})..."
                )
                ckpt_manager.save(opt_step, args=ocp.args.StandardSave(state))
                ckpt_manager.wait_until_finished()

                start_time = time.perf_counter()

    print(f"✅ Training for {args.exp_name} completed successfully.")


if __name__ == "__main__":
    main()
