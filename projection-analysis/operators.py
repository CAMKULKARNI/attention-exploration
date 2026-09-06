# REPLACEMENT CODE FOR: operators.py
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


class Config:
    def __init__(self, tokenizer):
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.d_model = 1024
        self.num_heads = 16
        self.num_kv_heads = 4


def compute_inv_freq(depth: int) -> jnp.ndarray:
    assert depth % 2 == 0, f"Head depth must be even, got {depth}"
    return 1.0 / (10000.0 ** (jnp.arange(0, depth, 2, dtype=jnp.float32) / depth)).reshape(1, -1)


def apply_rope(x: jnp.ndarray, start_pos: int, inv_freq: jnp.ndarray) -> jnp.ndarray:
    seq_len = x.shape[2]
    depth = x.shape[3]
    half_depth = depth // 2

    positions = start_pos + jnp.arange(seq_len, dtype=jnp.float32).reshape(-1, 1)
    angles = jnp.matmul(positions, inv_freq)
    sin = jnp.sin(angles).reshape(1, 1, seq_len, half_depth)
    cos = jnp.cos(angles).reshape(1, 1, seq_len, half_depth)

    x_f32 = x.astype(jnp.float32)
    x1, x2 = x_f32[..., :half_depth], x_f32[..., half_depth:]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return jnp.concatenate([rotated_x1, rotated_x2], axis=-1).astype(x.dtype)


class Attention(nn.Module):
    latent_dim: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int
    is_gqa: bool = False

    use_q_proj: bool = True
    use_k_proj: bool = True
    use_v_proj: bool = True
    q_act: str = "none"
    k_act: str = "none"
    v_act: str = "none"

    use_causal_mask: bool = True
    deterministic: bool = True

    def setup(self):
        depth = self.latent_dim // self.num_heads
        self.inv_freq = compute_inv_freq(depth)

    def bottleneck_proj(self, x, out_dim, name):
        in_dim = x.shape[-1]
        r = max(1, (in_dim * out_dim) // (in_dim + out_dim))

        rms = nn.RMSNorm(epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, name=f"{name}_rms")(x)
        w1 = nn.Dense(r, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32, name=f"{name}_w1")(rms)
        act = nn.gelu(w1, approximate=False)
        ln = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, name=f"{name}_ln")(act)
        f_x = nn.Dense(out_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32, name=f"{name}_w2")(ln)

        # Apply your exact instructions: Drop skip if mismatched, keep 0.5 scaling if matched
        if in_dim == out_dim:
            return (x + f_x) / 2.0
        return f_x

    def _apply_activation_proj(self, x: jnp.ndarray, out_dim: int, act_type: str, proj_name: str) -> jnp.ndarray:
        if act_type == "bottleneck":
            return self.bottleneck_proj(x, out_dim, proj_name)
        linear = nn.Dense(out_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32, name=proj_name)(x)
        if act_type == "gelu":
            return nn.gelu(linear, approximate=False)
        return linear

    @nn.compact
    def __call__(self, x: jnp.ndarray, start_pos: int = 0, cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        depth = self.latent_dim // self.num_heads
        batch_size, q_seq_len, in_dim = x.shape

        eff_kv_heads = self.num_kv_heads if self.is_gqa else self.num_heads
        assert self.num_heads % eff_kv_heads == 0, "Query heads must be a multiple of KV heads."
        num_queries_per_kv = self.num_heads // eff_kv_heads
        kv_dim = eff_kv_heads * depth

        # Projections
        if self.use_q_proj:
            q = self._apply_activation_proj(x, self.latent_dim, self.q_act, "q_proj")
        else:
            assert in_dim == self.latent_dim, "Cannot skip Q projection when input dimension does not match latent_dim"
            q = x

        if self.use_k_proj:
            k = self._apply_activation_proj(x, kv_dim, self.k_act, "k_proj")
        else:
            assert in_dim == kv_dim, "Cannot skip K projection when input dimension does not match kv_dim in GQA"
            k = x

        if self.use_v_proj:
            v = self._apply_activation_proj(x, kv_dim, self.v_act, "v_proj")
        else:
            assert in_dim == kv_dim, "Cannot skip V projection when input dimension does not match kv_dim in GQA"
            v = x

        q = q.reshape(batch_size, q_seq_len, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_k = k.reshape(batch_size, -1, eff_kv_heads, depth).transpose(0, 2, 1, 3)
        new_v = v.reshape(batch_size, -1, eff_kv_heads, depth).transpose(0, 2, 1, 3)

        q = apply_rope(q, start_pos, self.inv_freq)
        new_k = apply_rope(new_k, start_pos, self.inv_freq)

        if cache is not None:
            k_cache, v_cache = cache
            k_use = jax.lax.dynamic_update_slice(k_cache, new_k, (0, 0, start_pos, 0))
            v_use = jax.lax.dynamic_update_slice(v_cache, new_v, (0, 0, start_pos, 0))
            new_cache = (k_use, v_use)
        else:
            k_use, v_use = new_k, new_v
            new_cache = None

        kv_seq_len = k_use.shape[2]

        # Reshape Q for memory-efficient GQA dot-product without replicating K/V
        q = q.reshape(batch_size, eff_kv_heads, num_queries_per_kv, q_seq_len, depth)
        q = q * (1.0 / math.sqrt(depth))

        # Scaled Dot-Product: (B, H_kv, G, L_q, D) x (B, H_kv, L_k, D) -> (B, H_kv, G, L_q, L_k)
        scaled_logits = jnp.einsum("bhgqd,bhkd->bhgqk", q.astype(jnp.float32), k_use.astype(jnp.float32))

        q_idx = start_pos + jnp.arange(q_seq_len, dtype=jnp.int32).reshape(1, 1, 1, -1, 1)
        kv_range = jnp.arange(kv_seq_len, dtype=jnp.int32).reshape(1, 1, 1, 1, -1)

        mask = kv_range < (start_pos + q_seq_len)
        if self.use_causal_mask:
            mask = mask & (kv_range <= q_idx)

        # Numerically stable masking without overflow
        scaled_logits = jnp.where(mask, scaled_logits, -1e4)
        attention_weights = jax.nn.softmax(scaled_logits, axis=-1).astype(x.dtype)
        attention_weights = nn.Dropout(rate=0.1)(attention_weights, deterministic=self.deterministic)

        # (B, H_kv, G, L_q, L_k) x (B, H_kv, L_k, D) -> (B, H_kv, G, L_q, D)
        output = jnp.einsum("bhgqk,bhkd->bhgqd", attention_weights, v_use)
        output = output.reshape(batch_size, self.num_heads, q_seq_len, depth).transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, q_seq_len, self.latent_dim)

        out = nn.Dense(self.latent_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32, name="out_proj")(
            output
        )
        return out, new_cache


class DecoderBlock(nn.Module):
    latent_dim: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int
    is_gqa: bool = False
    use_q_proj: bool = True
    use_k_proj: bool = True
    use_v_proj: bool = True
    q_act: str = "none"
    k_act: str = "none"
    v_act: str = "none"
    use_causal_mask: bool = True
    deterministic: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, start_pos: int = 0, cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        x_norm = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, name="attn_ln")(x)
        attn_out, new_cache = Attention(
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_seq_len=self.max_seq_len,
            is_gqa=self.is_gqa,
            use_q_proj=self.use_q_proj,
            use_k_proj=self.use_k_proj,
            use_v_proj=self.use_v_proj,
            q_act=self.q_act,
            k_act=self.k_act,
            v_act=self.v_act,
            use_causal_mask=self.use_causal_mask,
            deterministic=self.deterministic,
            name="self_attention",
        )(x_norm, start_pos, cache)

        x = x + attn_out
        ffn_norm = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, name="ffn_ln")(x)
        ffn_out = nn.Dense(
            self.latent_dim * 4, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32, name="ffn_w1"
        )(ffn_norm)
        ffn_out = nn.gelu(ffn_out, approximate=False)
        ffn_out = nn.Dropout(rate=0.1)(ffn_out, deterministic=self.deterministic)
        ffn_out = nn.Dense(self.latent_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32, name="ffn_w2")(
            ffn_out
        )
        return x + ffn_out, new_cache


class CausalLM(nn.Module):
    vocab_size: int
    max_seq_len: int
    latent_dim: int
    num_heads: int
    num_layers: int
    num_kv_heads: int
    is_gqa: bool = False
    use_q_proj: bool = True
    use_k_proj: bool = True
    use_v_proj: bool = True
    q_act: str = "none"
    k_act: str = "none"
    v_act: str = "none"
    use_causal_mask: bool = True
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, current_pos: int = 0, caches: Optional[list] = None):
        token_emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.latent_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            name="token_embed",
        )
        x = token_emb(inputs)

        new_caches = []
        remat_block = nn.remat(DecoderBlock)

        for i in range(self.num_layers):
            layer_cache = caches[i] if caches is not None else None
            x, layer_new_cache = remat_block(
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                max_seq_len=self.max_seq_len,
                is_gqa=self.is_gqa,
                use_q_proj=self.use_q_proj,
                use_k_proj=self.use_k_proj,
                use_v_proj=self.use_v_proj,
                q_act=self.q_act,
                k_act=self.k_act,
                v_act=self.v_act,
                use_causal_mask=self.use_causal_mask,
                deterministic=self.deterministic,
                name=f"layer_{i}",
            )(x, current_pos, layer_cache)
            new_caches.append(layer_new_cache)

        x = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, name="final_ln")(x)
        logits = token_emb.attend(x).astype(jnp.float32)
        return logits, new_caches
