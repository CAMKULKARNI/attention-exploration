# train.py
import os
import argparse
import json
import time
import math
import numpy as np
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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

    # REMOVED optax.MultiSteps to save VRAM and fix the XLA I/O limit.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1),
    )

    return ResumableTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        data_index=jnp.array(0, dtype=jnp.int32),
    )


@jax.jit
def train_step(state, macro_batch, dropout_rng):
    """
    Executes a full macro-batch using jax.lax.scan to accumulate gradients
    strictly within the compiled XLA graph, bypassing I/O memory limits.
    """

    def body_fn(carry, micro_batch):
        rng, accum_loss, accum_grads = carry
        rng, step_rng = jax.random.split(rng)

        inputs = micro_batch[:, :-1]
        targets = micro_batch[:, 1:]

        def loss_fn(params):
            logits, _ = state.apply_fn(
                {"params": params},
                inputs,
                use_causal_mask=True,
                current_pos=0,
                caches=None,
                deterministic=False,
                rngs={"dropout": step_rng},
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets
            )
            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        # Accumulate loss and gradients in float32 to prevent bfloat16 underflow
        accum_loss = accum_loss + loss
        accum_grads = jax.tree_util.tree_map(
            lambda a, b: a + b.astype(jnp.float32), accum_grads, grads
        )

        return (rng, accum_loss, accum_grads), None

    # Initialize accumulators
    init_loss = jnp.array(0.0)
    init_grads = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=jnp.float32), state.params
    )
    init_carry = (dropout_rng, init_loss, init_grads)

    # jax.lax.scan loops over the 1st dimension of macro_batch (grad_accum_steps)
    final_carry, _ = jax.lax.scan(body_fn, init_carry, macro_batch)
    _, total_loss, total_grads = final_carry

    # Calculate exact averages
    num_micro_batches = macro_batch.shape[0]
    avg_loss = total_loss / num_micro_batches
    avg_grads = jax.tree_util.tree_map(lambda x: x / num_micro_batches, total_grads)

    # Apply the perfectly averaged macro-gradient
    state = state.apply_gradients(grads=avg_grads)

    grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(avg_grads)])
    )

    return state, avg_loss, grad_norm


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

    pbar = tqdm(range(num_batches), desc="      🔬 Validating", leave=False)
    for i in pbar:
        start_idx = i * tokens_per_batch
        end_idx = start_idx + tokens_per_batch
        raw_batch = val_data_map[start_idx:end_idx].astype(np.int32)
        batch = jnp.array(raw_batch.reshape(args.micro_batch_size, args.seq_len + 1))

        loss_val = float(eval_step(state, batch))
        total_val_loss += loss_val
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

    train_data_map = np.memmap(args.train_data_file, dtype=np.uint16, mode="r")
    val_data_map = np.memmap(args.val_data_file, dtype=np.uint16, mode="r")

    # Tokens required for one full optimizer step (Macro-Batch)
    tokens_per_macro_batch = (
        args.grad_accum_steps * args.micro_batch_size * (args.seq_len + 1)
    )

    total_train_tokens = len(train_data_map)
    max_opt_steps = total_train_tokens // tokens_per_macro_batch

    print(f"\n      📊 Train Dataset: {total_train_tokens:,} tokens")
    print(f"      🎯 1 Epoch requires: {max_opt_steps} optimizer steps.")

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

    start_opt_step = int(state.step)
    current_data_idx = int(state.data_index)

    print(f"\n🚀 [PID: {os.getpid()}] Starting Training: {args.exp_name}")
    print(
        f"      Resuming at Opt-Step: {start_opt_step} | Data Index: {current_data_idx}"
    )

    log_data = []
    log_file = os.path.join(
        args.output_dir, f"{args.exp_name.replace(':', '').replace(' ', '_')}_logs.json"
    )

    # -----------------------------------------------------------------
    # MAIN TRAINING LOOP WITH TQDM
    # -----------------------------------------------------------------
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

        raw_batch = train_data_map[current_data_idx:end_idx].astype(np.int32)
        # Reshape into (grad_accum_steps, micro_batch_size, seq_len + 1)
        macro_batch = jnp.array(
            raw_batch.reshape(
                args.grad_accum_steps, args.micro_batch_size, args.seq_len + 1
            )
        )

        current_data_idx = end_idx
        state = state.replace(data_index=jnp.array(current_data_idx, dtype=jnp.int32))

        dropout_rng, step_dropout_rng = jax.random.split(dropout_rng)

        # JIT-compiled macro step executed entirely on GPU
        state, loss, grad_norm = train_step(state, macro_batch, step_dropout_rng)

        # Float cast forces CPU sync, freeing GPU memory instantly
        avg_train_loss = float(loss)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
        gnorm = float(grad_norm)

        # Update progress bar
        pbar.set_postfix({"Loss": f"{avg_train_loss:.4f}", "PPL": f"{train_ppl:.2f}"})

        if opt_step % args.log_interval == 0:
            log_data.append(
                {
                    "opt_step": opt_step,
                    "train_loss": avg_train_loss,
                    "train_perplexity": train_ppl,
                    "grad_norm": gnorm,
                }
            )
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)

        if opt_step % args.save_interval == 0 or opt_step == max_opt_steps:
            tqdm.write(
                f"\n      💾 Saving checkpoint at Opt Step {opt_step} (Data Index: {current_data_idx})..."
            )
            ckpt_manager.save(opt_step, args=ocp.args.StandardSave(state))
            ckpt_manager.wait_until_finished()

            val_loss, val_ppl = run_validation(state, val_data_map, args)
            tqdm.write(
                f"      ⭐ Validation Complete | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}\n"
            )

            log_data[-1]["val_loss"] = val_loss
            log_data[-1]["val_perplexity"] = val_ppl
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)

    print(f"✅ Training for {args.exp_name} completed successfully.")


if __name__ == "__main__":
    main()
