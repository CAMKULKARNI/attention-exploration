import os
import argparse
import json
import math
import numpy as np
import functools
from tqdm import tqdm

# Maintain fast async allocation pools without runtime dynamic mmap fragmentation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from transformers import AutoTokenizer

from operators import CausalLM, Config


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--is_gqa", type=str2bool, required=True)
    parser.add_argument("--use_q_proj", type=str2bool, required=True)
    parser.add_argument("--use_k_proj", type=str2bool, required=True)
    parser.add_argument("--use_v_proj", type=str2bool, required=True)
    parser.add_argument("--q_act", type=str, default="none")
    parser.add_argument("--k_act", type=str, default="none")
    parser.add_argument("--v_act", type=str, default="none")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_decay_fraction", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=148)
    parser.add_argument("--output_dir", type=str, default="./training_logs")
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--val_data_file", type=str, required=True)
    return parser.parse_args()


class ResumableTrainState(train_state.TrainState):
    data_index: jnp.ndarray


def create_train_state(rng, model, args, max_opt_steps):
    dummy_input = jnp.zeros((args.micro_batch_size, args.seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, current_pos=0)

    # Master parameters maintained strictly in FP32
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), variables["params"])

    warmup_steps = int(max_opt_steps * 0.05)
    decay_steps = max(1, max_opt_steps - warmup_steps)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=args.learning_rate * args.lr_decay_fraction,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1, mu_dtype=jnp.float32),
    )

    return ResumableTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        data_index=jnp.array(0, dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnums=(1,))
def train_step(state, model, macro_batch, dropout_rng):
    def body_fn(carry, micro_batch):
        rng, accum_loss, accum_grads = carry
        rng, step_rng = jax.random.split(rng)
        inputs, targets = micro_batch[:, :-1], micro_batch[:, 1:]

        def loss_fn(params):
            logits, _ = model.apply({"params": params}, inputs, current_pos=0, caches=None, rngs={"dropout": step_rng})
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        accum_loss += loss
        accum_grads = jax.tree_util.tree_map(lambda a, b: a + b.astype(jnp.float32), accum_grads, grads)
        return (rng, accum_loss, accum_grads), None

    init_carry = (
        dropout_rng,
        jnp.array(0.0, dtype=jnp.float32),
        jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=jnp.float32), state.params),
    )
    final_carry, _ = jax.lax.scan(body_fn, init_carry, macro_batch)
    _, total_loss, total_grads = final_carry

    num_micro_batches = macro_batch.shape[0]
    avg_loss = total_loss / num_micro_batches
    avg_grads = jax.tree_util.tree_map(lambda x: x / num_micro_batches, total_grads)

    grad_norm = optax.global_norm(avg_grads)
    new_state = state.apply_gradients(grads=avg_grads)
    return new_state, avg_loss, grad_norm


@functools.partial(jax.jit, static_argnums=(1,))
def eval_step(state, model, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits, _ = model.apply({"params": state.params}, inputs, current_pos=0, caches=None)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
    return jnp.sum(loss)


def run_validation(state, model_eval, val_data_map, args):
    tokens_per_batch = args.eval_batch_size * (args.seq_len + 1)
    num_batches = len(val_data_map) // tokens_per_batch
    total_tokens = num_batches * args.eval_batch_size * args.seq_len

    # On-device scalar reduction eliminates 250+ serialized Host-Device sync points
    total_loss = jnp.array(0.0, dtype=jnp.float32)
    for i in tqdm(range(num_batches)):
        start_idx = i * tokens_per_batch
        end_idx = start_idx + tokens_per_batch
        batch = jnp.asarray(
            val_data_map[start_idx:end_idx].reshape(args.eval_batch_size, args.seq_len + 1).astype(np.int32)
        )
        total_loss = total_loss + eval_step(state, model_eval, batch)

    total_loss_cpu = float(jax.block_until_ready(total_loss))
    avg_val_loss = total_loss_cpu / total_tokens
    val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
    return avg_val_loss, val_ppl


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    safe_name = args.exp_name.replace(":", "").replace(" ", "_")
    done_file = os.path.join(args.output_dir, f"{safe_name}_DONE.txt")
    log_file = os.path.join(args.output_dir, f"{safe_name}_logs.json")

    if os.path.exists(done_file):
        print(f"[SKIP] Experiment '{args.exp_name}' is already complete.")
        return

    # Wikitext-103 fits entirely in memory (~206MB); load directly to eliminate page-fault spikes
    train_data_map = np.fromfile(args.train_data_file, dtype=np.uint16)
    val_data_map = np.fromfile(args.val_data_file, dtype=np.uint16)

    tokens_per_macro = args.grad_accum_steps * args.micro_batch_size * (args.seq_len + 1)
    max_opt_steps = len(train_data_map) // tokens_per_macro

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config_obj = Config(tokenizer)

    common_kwargs = dict(
        vocab_size=config_obj.vocab_size,
        max_seq_len=args.seq_len,
        latent_dim=config_obj.d_model,
        num_heads=config_obj.num_heads,
        num_layers=args.num_layers,
        num_kv_heads=config_obj.num_kv_heads,
        is_gqa=args.is_gqa,
        use_q_proj=args.use_q_proj,
        use_k_proj=args.use_k_proj,
        use_v_proj=args.use_v_proj,
        q_act=args.q_act,
        k_act=args.k_act,
        v_act=args.v_act,
        use_causal_mask=True,
    )

    model_train = CausalLM(**common_kwargs, deterministic=False)
    model_eval = CausalLM(**common_kwargs, deterministic=True)

    rng = jax.random.PRNGKey(42)
    init_rng, dropout_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model_train, args, max_opt_steps)

    # Warm up JIT execution kernels
    dummy_macro_batch = jnp.zeros((args.grad_accum_steps, args.micro_batch_size, args.seq_len + 1), dtype=jnp.int32)
    dummy_eval_batch = jnp.zeros((args.eval_batch_size, args.seq_len + 1), dtype=jnp.int32)
    warm_state, warm_loss, warm_norm = train_step(state, model_train, dummy_macro_batch, dropout_rng)
    jax.block_until_ready((warm_state.params, warm_loss, warm_norm))
    del warm_state, warm_loss, warm_norm
    eval_warm = eval_step(state, model_eval, dummy_eval_batch)
    jax.block_until_ready(eval_warm)

    current_data_idx = 0
    pbar = tqdm(range(1, max_opt_steps + 1), desc=f"Training {args.exp_name}", total=max_opt_steps)

    BENCH_WINDOW = 50

    for opt_step in pbar:
        end_idx = current_data_idx + tokens_per_macro
        if end_idx > len(train_data_map):
            break

        macro_batch = jnp.asarray(
            train_data_map[current_data_idx:end_idx]
            .reshape(args.grad_accum_steps, args.micro_batch_size, args.seq_len + 1)
            .astype(np.int32)
        )

        current_data_idx = end_idx
        dropout_rng, step_rng = jax.random.split(dropout_rng)

        # Start non-intrusive timing window
        if (opt_step - 1) % BENCH_WINDOW == 0:
            jax.block_until_ready(state.params)

        state, loss, grad_norm = train_step(state, model_train, macro_batch, step_rng)

        if opt_step % BENCH_WINDOW == 0 or opt_step == max_opt_steps:
            jax.block_until_ready(state.params)

            avg_train_loss = float(loss)
            train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
            gnorm = float(grad_norm)

            pbar.set_postfix(
                {
                    "Loss": f"{avg_train_loss:.4f}",
                    "PPL": f"{train_ppl:.2f}",
                    "Norm": f"{gnorm:.2f}",
                }
            )

            val_loss, val_ppl = run_validation(state, model_eval, val_data_map, args)
            tqdm.write(f"Step {opt_step}/{max_opt_steps} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

            log_dict = {
                "opt_step": opt_step,
                "train_loss": float(loss),
                "train_perplexity": math.exp(float(loss)) if float(loss) < 20 else float("inf"),
                "grad_norm": float(grad_norm),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            }
            with open(log_file, "a") as f:
                json.dump(log_dict, f)
                f.write("\n")

    jax.block_until_ready(state.params)
    with open(done_file, "w") as f:
        f.write("COMPLETED\n")


if __name__ == "__main__":
    main()
