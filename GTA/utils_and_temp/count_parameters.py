import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from operators import CausalLM, Config


def count_params(params):
    """Counts the total number of parameters in a PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def main():
    """
    Initializes each of the 29 experiment models to count and report
    the total number of trainable parameters.
    """
    # These definitions are copied from run_training.sh
    experiments = [
        {"id": 1, "name": "Vanilla MHA 24L", "attn": "mha", "layers": 24},
        {"id": 2, "name": "GQA RoPE 24L", "attn": "gqa", "layers": 24},
        {"id": 3, "name": "TA ALiBi 24L", "attn": "ta", "layers": 24},
        {"id": 4, "name": "GQA RoPE 28L", "attn": "gqa", "layers": 28},
        {"id": 5, "name": "GQA RoPE 30L", "attn": "gqa", "layers": 30},
        {"id": 6, "name": "TA ALiBi 28L", "attn": "ta", "layers": 28},
        {"id": 7, "name": "TA ALiBi 30L", "attn": "ta", "layers": 30},
        {"id": 8, "name": "MHA IsoKV 6L", "attn": "mha", "layers": 6},
        {"id": 9, "name": "MHA RoPE 24L", "attn": "mha_rope", "layers": 24},
        {"id": 10, "name": "MHA RoPE 28L", "attn": "mha_rope", "layers": 28},
        {"id": 11, "name": "MHA RoPE 30L", "attn": "mha_rope", "layers": 30},
        {"id": 12, "name": "GTA ALiBi 24L", "attn": "gta", "layers": 24},
        {"id": 13, "name": "GTA ALiBi 28L", "attn": "gta", "layers": 28},
        {"id": 14, "name": "GTA ALiBi 30L", "attn": "gta", "layers": 30},
        {"id": 15, "name": "MHA ALiBi 24L", "attn": "mha_alibi", "layers": 24},
        {"id": 16, "name": "MHA ALiBi 28L", "attn": "mha_alibi", "layers": 28},
        {"id": 17, "name": "MHA ALiBi 30L", "attn": "mha_alibi", "layers": 30},
        {"id": 18, "name": "GQA ALiBi 24L", "attn": "gqa_alibi", "layers": 24},
        {"id": 19, "name": "GQA ALiBi 28L", "attn": "gqa_alibi", "layers": 28},
        {"id": 20, "name": "GQA ALiBi 30L", "attn": "gqa_alibi", "layers": 30},
        {"id": 21, "name": "TA PE 24L", "attn": "ta_pe", "layers": 24},
        {"id": 22, "name": "TA PE 28L", "attn": "ta_pe", "layers": 28},
        {"id": 23, "name": "TA PE 30L", "attn": "ta_pe", "layers": 30},
        {"id": 24, "name": "GTA PE 24L", "attn": "gta_pe", "layers": 24},
        {"id": 25, "name": "GTA PE 28L", "attn": "gta_pe", "layers": 28},
        {"id": 26, "name": "GTA PE 30L", "attn": "gta_pe", "layers": 30},
        {"id": 27, "name": "GQA PE 24L", "attn": "gqa_pe", "layers": 24},
        {"id": 28, "name": "GQA PE 28L", "attn": "gqa_pe", "layers": 28},
        {"id": 29, "name": "GQA PE 30L", "attn": "gqa_pe", "layers": 30},
    ]

    # These configs are fixed across all experiments
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config_obj = Config(tokenizer)
    seq_len = 512  # Dummy value for initialization

    print("Calculating parameters for all 29 experiments...")

    results = []
    for exp in experiments:
        model = CausalLM(
            vocab_size=config_obj.vocab_size,
            max_seq_len=seq_len,
            latent_dim=config_obj.d_model,
            num_heads=config_obj.num_heads,
            num_layers=exp["layers"],
            attn_type=exp["attn"],
            num_kv_heads=config_obj.num_kv_heads,
            deterministic=True,  # No need for dropout
        )

        # Initialize the model to get the parameters
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, seq_len), dtype=jnp.int32)
        variables = model.init(rng, dummy_input)
        params = variables["params"]

        total_params = count_params(params)
        results.append(
            {
                "id": exp["id"],
                "name": exp["name"],
                "layers": exp["layers"],
                "attn_type": exp["attn"],
                "params": total_params,
            }
        )

    # Print the results in a markdown table
    print("\n| Exp ID | Experiment Name         | Layers | Attention Type | Parameters (M) |")
    print("|:-------|:------------------------|:-------|:---------------|---------------:|")

    for res in sorted(results, key=lambda x: x["id"]):
        params_in_millions = res["params"] / 1_000_000
        print(
            f"| {res['id']:<6} | {res['name']:<23} | {res['layers']:<6} | {res['attn_type']:<14} | {params_in_millions:>14.2f} |"
        )


if __name__ == "__main__":
    main()
