import os
import argparse
import json
import time
import math
import numpy as np

# Prevent JAX from preallocating all VRAM so we don't crash the WSL container instantly
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from datasets import load_dataset
from transformers import AutoTokenizer

from operators import CausalLM, Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--attn_type", type=str, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Batch size per forward pass"
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=16, help="Accumulation steps"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./training_logs")
    return parser.parse_args()


class TrainState(train_state.TrainState):
    # Custom TrainState can be expanded here if we need custom dropout/batchnorm rngs
    pass


def create_train_state(rng, model, config, args):
    dummy_input = jnp.zeros((args.micro_batch_size, args.seq_len), dtype=jnp.int32)

    print("Initializing model weights...")
    variables = model.init(rng, dummy_input, use_causal_mask=True, current_pos=0)

    # BFLOAT16 POLICY -- LAYERNORM-AWARE (From benchmark.py)
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

    # Setup Optax Optimizer with Gradient Accumulation
    # We use a Cosine Decay schedule with warmup
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

    # Wrap optimizer in MultiSteps for gradient accumulation
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=args.grad_accum_steps)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )


@jax.jit
def train_step(state, batch):
    """Executes a single micro-batch step."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    def loss_fn(params):
        # We pass caches=None to ignore KV caching during training
        logits, _ = state.apply_fn(
            {"params": params}, inputs, use_causal_mask=True, current_pos=0, caches=None
        )
        # Calculate Cross Entropy Loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=targets
        )
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # state.apply_updates automatically handles the accumulation logic via optax.MultiSteps
    state = state.apply_updates(grads=grads)

    # Calculate gradient norm for telemetry
    grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)])
    )

    return state, loss, grad_norm


def get_dataloader(tokenizer, seq_len, batch_size):
    # Load TinyStories
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def token_generator():
        buffer = []
        for example in dataset:
            tokens = tokenizer.encode(example["text"])
            tokens.append(tokenizer.eos_token_id)
            buffer.extend(tokens)

            # Yield chunks of (seq_len + 1) to allow shifting for next-token prediction
            while len(buffer) >= seq_len + 1:
                yield buffer[: seq_len + 1]
                buffer = buffer[seq_len + 1 :]

    gen = token_generator()
    while True:
        batch = [next(gen) for _ in range(batch_size)]
        yield jnp.array(batch, dtype=jnp.int32)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
    state = create_train_state(rng, model, config_obj, args)

    dataloader = get_dataloader(tokenizer, args.seq_len, args.micro_batch_size)

    print(f"\n🚀 Starting Training: {args.exp_name}")
    print(f"   Effective Batch Size: {args.micro_batch_size * args.grad_accum_steps}")
    print(f"   Target Steps: {args.max_steps}")

    log_data = []
    log_file = os.path.join(
        args.output_dir, f"{args.exp_name.replace(':', '').replace(' ', '_')}.json"
    )

    start_time = time.perf_counter()
    step_loss = 0.0

    for step in range(1, args.max_steps + 1):
        batch = next(dataloader)
        state, loss, grad_norm = train_step(state, batch)
        step_loss += loss

        if step % args.log_interval == 0:
            avg_loss = float(step_loss / args.log_interval)
            ppl = (
                math.exp(avg_loss) if avg_loss < 20 else float("inf")
            )  # Prevent overflow
            gnorm = float(grad_norm)

            elapsed = time.perf_counter() - start_time
            steps_per_sec = args.log_interval / elapsed

            print(
                f"Step {step:05d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | GradNorm: {gnorm:.2f} | Steps/s: {steps_per_sec:.2f}"
            )

            log_data.append(
                {"step": step, "loss": avg_loss, "perplexity": ppl, "grad_norm": gnorm}
            )

            step_loss = 0.0
            start_time = time.perf_counter()

            # Save logs incrementally
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)


if __name__ == "__main__":
    main()
