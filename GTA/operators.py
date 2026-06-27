import math
import jax
import jax.numpy as jnp
import flax.linen as nn


class Config:
    def __init__(self, tokenizer):
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.num_layers = 24
        self.d_model = 1024
        self.num_heads = 16
        self.num_kv_heads = 4


def get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [math.pow(base, i) for i in range(1, closest_power_of_2 + 1)]
    if closest_power_of_2 < num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(closest_power_of_2 * 2) - 3)))
        slopes.extend(
            [
                math.pow(extra_base, i)
                for i in range(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
            ]
        )
    return jnp.array(slopes, dtype=jnp.float32)


def apply_rope(x, start_pos, inv_freq):
    seq_len = x.shape[2]
    depth = x.shape[3]

    # Use base array + dynamic offset to prevent JIT tracing errors
    positions = start_pos + jnp.arange(seq_len, dtype=jnp.float32)
    positions = positions.reshape(-1, 1)

    angles = jnp.matmul(positions, inv_freq)
    sin = jnp.sin(angles).reshape(1, 1, seq_len, depth // 2)
    cos = jnp.cos(angles).reshape(1, 1, seq_len, depth // 2)

    x_f32 = x.astype(jnp.float32)
    half_depth = depth // 2
    x1, x2 = x_f32[..., :half_depth], x_f32[..., half_depth:]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    rotated_out = jnp.concatenate([rotated_x1, rotated_x2], axis=-1)
    return rotated_out.astype(x.dtype)


class TiedAttention(nn.Module):
    latent_dim: int
    num_heads: int
    max_seq_len: int

    @nn.compact
    def __call__(
        self, x, use_causal_mask=False, start_pos=0, cache=None, deterministic=True
    ):
        depth = self.latent_dim // self.num_heads
        batch_size = x.shape[0]
        q_seq_len = x.shape[1]

        wq = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wa = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)

        q = wq.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_a = wa.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)

        if cache is not None:
            a_use = jax.lax.dynamic_update_slice(cache, new_a, (0, 0, start_pos, 0))
            new_cache = a_use
        else:
            a_use = new_a
            new_cache = None

        kv_seq_len = a_use.shape[2]
        a_t = a_use.transpose(0, 1, 3, 2)

        matmul_qa = jnp.matmul(q, a_t)
        inv_sqrt_depth = 1.0 / math.sqrt(depth)
        scaled_attention_logits = matmul_qa.astype(jnp.float32) * inv_sqrt_depth

        # num_heads is a compile-time module attribute; XLA constant-folds this
        # across all decode steps, so no per-step allocation occurs at runtime.
        alibi_slopes = get_alibi_slopes(self.num_heads)

        q_idx = start_pos + jnp.arange(q_seq_len, dtype=jnp.float32)
        q_idx = q_idx.reshape(1, 1, -1, 1)
        kv_range = jnp.arange(kv_seq_len, dtype=jnp.float32).reshape(1, 1, 1, -1)

        distance = kv_range - q_idx
        slopes = alibi_slopes.reshape(1, self.num_heads, 1, 1)
        scaled_attention_logits += slopes * (-jnp.abs(distance))

        valid_cache_mask = (kv_range < (start_pos + q_seq_len)).astype(jnp.float32)
        scaled_attention_logits += (1.0 - valid_cache_mask) * -1e9

        if use_causal_mask:
            causal_mask = (kv_range <= q_idx).astype(jnp.float32)
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1).astype(
            x.dtype
        )

        attention_weights = nn.Dropout(rate=0.1)(
            attention_weights, deterministic=deterministic
        )

        output = jnp.matmul(attention_weights, a_use).transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, -1, self.latent_dim)

        return (
            nn.Dense(
                self.latent_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )(output),
            new_cache,
        )


class GroupTiedAttention(nn.Module):
    """Group Tied Attention (GTA).

    Extends TiedAttention by reducing the A-tensor head count from
    num_heads down to num_a_heads (analogous to num_kv_heads in GQA).
    This produces a KV cache of shape [batch, num_a_heads, max_len, depth],
    which is 4x smaller than standard GTA and smaller than GQA's paired
    (K, V) cache at the same head count.

    Design space position:
        Full-heads tied  → TiedAttention          (TA)
        Reduced-heads tied → GroupTiedAttention (GTA)   ← this class
        Reduced-heads separate KV → GroupedQueryAttention (GQA)

    The Q projection uses num_heads (full query heads) throughout.
    The A projection uses num_a_heads, then the cache is expanded to
    num_heads via broadcast_to before the attention matmul — identical
    to GQA's K/V head expansion pattern.
    ALiBi slopes are computed for num_heads so every query head gets
    a correctly-scaled positional bias.
    """

    latent_dim: int
    num_heads: int
    num_a_heads: int  # reduced head count for the A tensor (e.g. 4)
    max_seq_len: int

    @nn.compact
    def __call__(
        self, x, use_causal_mask=False, start_pos=0, cache=None, deterministic=True
    ):
        depth = self.latent_dim // self.num_heads
        num_queries_per_a = self.num_heads // self.num_a_heads
        batch_size = x.shape[0]
        q_seq_len = x.shape[1]

        # Q projects to full latent_dim; A projects to num_a_heads * depth.
        # Both use bfloat16 natively — symmetric with every other variant.
        wq = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wa = nn.Dense(
            self.num_a_heads * depth,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)

        q = wq.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_a = wa.reshape(batch_size, -1, self.num_a_heads, depth).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            a_use = jax.lax.dynamic_update_slice(cache, new_a, (0, 0, start_pos, 0))
            new_cache = a_use
        else:
            a_use = new_a
            new_cache = None

        kv_seq_len = a_use.shape[2]

        # Expand A from num_a_heads to num_heads via broadcast_to.
        # Identical pattern to GQA's K/V head expansion — zero-copy view.
        a_use_exp = a_use.reshape(batch_size, self.num_a_heads, 1, kv_seq_len, depth)
        a_use_rep = jnp.broadcast_to(
            a_use_exp,
            (batch_size, self.num_a_heads, num_queries_per_a, kv_seq_len, depth),
        ).reshape(batch_size, self.num_heads, kv_seq_len, depth)

        # Q * A^T attention logits — symmetric with GTA
        a_t = a_use_rep.transpose(0, 1, 3, 2)
        matmul_qa = jnp.matmul(q, a_t)
        inv_sqrt_depth = 1.0 / math.sqrt(depth)
        scaled_attention_logits = matmul_qa.astype(jnp.float32) * inv_sqrt_depth

        # ALiBi slopes over num_heads query heads — same as GTA
        # num_heads is compile-time; XLA constant-folds this.
        alibi_slopes = get_alibi_slopes(self.num_heads)

        q_idx = start_pos + jnp.arange(q_seq_len, dtype=jnp.float32)
        q_idx = q_idx.reshape(1, 1, -1, 1)
        kv_range = jnp.arange(kv_seq_len, dtype=jnp.float32).reshape(1, 1, 1, -1)

        distance = kv_range - q_idx
        slopes = alibi_slopes.reshape(1, self.num_heads, 1, 1)
        scaled_attention_logits += slopes * (-jnp.abs(distance))

        valid_cache_mask = (kv_range < (start_pos + q_seq_len)).astype(jnp.float32)
        scaled_attention_logits += (1.0 - valid_cache_mask) * -1e9

        if use_causal_mask:
            causal_mask = (kv_range <= q_idx).astype(jnp.float32)
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1).astype(
            x.dtype
        )

        attention_weights = nn.Dropout(rate=0.1)(
            attention_weights, deterministic=deterministic
        )

        # Output matmul uses the expanded a_use_rep — symmetric with TA
        output = jnp.matmul(attention_weights, a_use_rep).transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, -1, self.latent_dim)

        return (
            nn.Dense(
                self.latent_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )(output),
            new_cache,
        )


class GroupedQueryAttention(nn.Module):
    latent_dim: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int

    @nn.compact
    def __call__(
        self, x, use_causal_mask=False, start_pos=0, cache=None, deterministic=True
    ):
        depth = self.latent_dim // self.num_heads
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        batch_size = x.shape[0]
        q_seq_len = x.shape[1]

        # -----------------------------------------------------------------
        # OPTIMIZATION: STRICT NATIVE DTYPES
        # Forces native 16-bit execution to avoid defensive casts later.
        # -----------------------------------------------------------------
        wq = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wk = nn.Dense(
            self.num_kv_heads * depth,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wv = nn.Dense(
            self.num_kv_heads * depth,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)

        inv_freq = (
            1.0 / (10000.0 ** (jnp.arange(0, depth, 2, dtype=jnp.float32) / depth))
        ).reshape(1, -1)

        q = wq.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_k = wk.reshape(batch_size, -1, self.num_kv_heads, depth).transpose(
            0, 2, 1, 3
        )
        new_v = wv.reshape(batch_size, -1, self.num_kv_heads, depth).transpose(
            0, 2, 1, 3
        )

        q = apply_rope(q, start_pos, inv_freq)
        new_k = apply_rope(new_k, start_pos, inv_freq)

        if cache is not None:
            k_cache, v_cache = cache
            # Defensive casts removed; layers inherently match the cache dtype
            k_use = jax.lax.dynamic_update_slice(k_cache, new_k, (0, 0, start_pos, 0))
            v_use = jax.lax.dynamic_update_slice(v_cache, new_v, (0, 0, start_pos, 0))
            new_cache = (k_use, v_use)
        else:
            k_use, v_use = new_k, new_v
            new_cache = None

        kv_seq_len = k_use.shape[2]

        # Replace jnp.repeat with broadcast_to for true zero-copy expansion
        k_use_exp = k_use.reshape(batch_size, self.num_kv_heads, 1, kv_seq_len, depth)
        k_use_rep = jnp.broadcast_to(
            k_use_exp,
            (batch_size, self.num_kv_heads, num_queries_per_kv, kv_seq_len, depth),
        ).reshape(batch_size, self.num_heads, kv_seq_len, depth)

        v_use_exp = v_use.reshape(batch_size, self.num_kv_heads, 1, kv_seq_len, depth)
        v_use_rep = jnp.broadcast_to(
            v_use_exp,
            (batch_size, self.num_kv_heads, num_queries_per_kv, kv_seq_len, depth),
        ).reshape(batch_size, self.num_heads, kv_seq_len, depth)

        k_t = k_use_rep.transpose(0, 1, 3, 2)

        # Standard cuBLAS Matmul replaces the slow einsum
        matmul_qk = jnp.matmul(q, k_t)

        # -----------------------------------------------------------------
        # OPTIMIZATION: ARITHMETIC INTENSITY
        # -----------------------------------------------------------------
        inv_sqrt_depth = 1.0 / math.sqrt(depth)
        scaled_attention_logits = matmul_qk.astype(jnp.float32) * inv_sqrt_depth

        q_idx = start_pos + jnp.arange(q_seq_len, dtype=jnp.float32)
        q_idx = q_idx.reshape(1, 1, -1, 1)
        kv_range = jnp.arange(kv_seq_len, dtype=jnp.float32).reshape(1, 1, 1, -1)

        valid_cache_mask = (kv_range < (start_pos + q_seq_len)).astype(jnp.float32)
        scaled_attention_logits += (1.0 - valid_cache_mask) * -1e9

        if use_causal_mask:
            causal_mask = (kv_range <= q_idx).astype(jnp.float32)
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1).astype(
            x.dtype
        )

        attention_weights = nn.Dropout(rate=0.1)(
            attention_weights, deterministic=deterministic
        )

        # Standard cuBLAS Matmul replaces the output einsum
        output = jnp.matmul(attention_weights, v_use_rep).transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, -1, self.latent_dim)

        return (
            nn.Dense(
                self.latent_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )(output),
            new_cache,
        )


class MultiHeadAttention(nn.Module):
    latent_dim: int
    num_heads: int
    max_seq_len: int

    @nn.compact
    def __call__(
        self, x, use_causal_mask=False, start_pos=0, cache=None, deterministic=True
    ):
        depth = self.latent_dim // self.num_heads
        batch_size = x.shape[0]
        q_seq_len = x.shape[1]

        wq = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wk = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wv = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)

        q = wq.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_k = wk.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_v = wv.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            k_use = jax.lax.dynamic_update_slice(k_cache, new_k, (0, 0, start_pos, 0))
            v_use = jax.lax.dynamic_update_slice(v_cache, new_v, (0, 0, start_pos, 0))
            new_cache = (k_use, v_use)
        else:
            k_use, v_use = new_k, new_v
            new_cache = None

        kv_seq_len = k_use.shape[2]
        k_t = k_use.transpose(0, 1, 3, 2)

        matmul_qk = jnp.matmul(q, k_t)
        inv_sqrt_depth = 1.0 / math.sqrt(depth)
        scaled_attention_logits = matmul_qk.astype(jnp.float32) * inv_sqrt_depth

        q_idx = start_pos + jnp.arange(q_seq_len, dtype=jnp.float32)
        q_idx = q_idx.reshape(1, 1, -1, 1)
        kv_range = jnp.arange(kv_seq_len, dtype=jnp.float32).reshape(1, 1, 1, -1)

        valid_cache_mask = (kv_range < (start_pos + q_seq_len)).astype(jnp.float32)
        scaled_attention_logits += (1.0 - valid_cache_mask) * -1e9

        if use_causal_mask:
            causal_mask = (kv_range <= q_idx).astype(jnp.float32)
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1).astype(
            x.dtype
        )

        attention_weights = nn.Dropout(rate=0.1)(
            attention_weights, deterministic=deterministic
        )

        output = jnp.matmul(attention_weights, v_use).transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, -1, self.latent_dim)

        return (
            nn.Dense(
                self.latent_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )(output),
            new_cache,
        )


class MultiHeadAttentionRoPE(nn.Module):
    latent_dim: int
    num_heads: int
    max_seq_len: int

    @nn.compact
    def __call__(
        self, x, use_causal_mask=False, start_pos=0, cache=None, deterministic=True
    ):
        depth = self.latent_dim // self.num_heads
        batch_size = x.shape[0]
        q_seq_len = x.shape[1]

        wq = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wk = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)
        wv = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(x)

        # Identical RoPE construction to GQA — this is the symmetry requirement
        inv_freq = (
            1.0 / (10000.0 ** (jnp.arange(0, depth, 2, dtype=jnp.float32) / depth))
        ).reshape(1, -1)

        q = wq.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_k = wk.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)
        new_v = wv.reshape(batch_size, -1, self.num_heads, depth).transpose(0, 2, 1, 3)

        # Apply RoPE — identical call signature to GQA
        q = apply_rope(q, start_pos, inv_freq)
        new_k = apply_rope(new_k, start_pos, inv_freq)

        if cache is not None:
            k_cache, v_cache = cache
            k_use = jax.lax.dynamic_update_slice(k_cache, new_k, (0, 0, start_pos, 0))
            v_use = jax.lax.dynamic_update_slice(v_cache, new_v, (0, 0, start_pos, 0))
            new_cache = (k_use, v_use)
        else:
            k_use, v_use = new_k, new_v
            new_cache = None

        kv_seq_len = k_use.shape[2]
        k_t = k_use.transpose(0, 1, 3, 2)
        matmul_qk = jnp.matmul(q, k_t)
        inv_sqrt_depth = 1.0 / math.sqrt(depth)
        scaled_attention_logits = matmul_qk.astype(jnp.float32) * inv_sqrt_depth

        # Identical masking logic to MHA — symmetric
        q_idx = start_pos + jnp.arange(q_seq_len, dtype=jnp.float32)
        q_idx = q_idx.reshape(1, 1, -1, 1)
        kv_range = jnp.arange(kv_seq_len, dtype=jnp.float32).reshape(1, 1, 1, -1)

        valid_cache_mask = (kv_range < (start_pos + q_seq_len)).astype(jnp.float32)
        scaled_attention_logits += (1.0 - valid_cache_mask) * -1e9

        if use_causal_mask:
            causal_mask = (kv_range <= q_idx).astype(jnp.float32)
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1).astype(
            x.dtype
        )

        attention_weights = nn.Dropout(rate=0.1)(
            attention_weights, deterministic=deterministic
        )
        output = jnp.matmul(attention_weights, v_use).transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, -1, self.latent_dim)

        return (
            nn.Dense(
                self.latent_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )(output),
            new_cache,
        )


class DecoderBlock(nn.Module):
    latent_dim: int
    num_heads: int
    attn_type: str
    max_seq_len: int
    num_kv_heads: int = 4

    @nn.compact
    def __call__(
        self, x, use_causal_mask=False, start_pos=0, cache=None, deterministic=True
    ):
        x_norm = nn.LayerNorm(epsilon=1e-6)(x)

        if self.attn_type == "ta":
            attn_out, new_cache = TiedAttention(
                self.latent_dim, self.num_heads, self.max_seq_len
            )(
                x_norm,
                use_causal_mask=use_causal_mask,
                start_pos=start_pos,
                cache=cache,
                deterministic=deterministic,
            )
        elif self.attn_type == "gta":
            attn_out, new_cache = GroupTiedAttention(
                self.latent_dim, self.num_heads, self.num_kv_heads, self.max_seq_len
            )(
                x_norm,
                use_causal_mask=use_causal_mask,
                start_pos=start_pos,
                cache=cache,
                deterministic=deterministic,
            )
        elif self.attn_type == "gqa":
            attn_out, new_cache = GroupedQueryAttention(
                self.latent_dim, self.num_heads, self.num_kv_heads, self.max_seq_len
            )(
                x_norm,
                use_causal_mask=use_causal_mask,
                start_pos=start_pos,
                cache=cache,
                deterministic=deterministic,
            )
        elif self.attn_type == "mha":
            attn_out, new_cache = MultiHeadAttention(
                self.latent_dim, self.num_heads, self.max_seq_len
            )(
                x_norm,
                use_causal_mask=use_causal_mask,
                start_pos=start_pos,
                cache=cache,
                deterministic=deterministic,
            )
        elif self.attn_type == "mha_rope":
            attn_out, new_cache = MultiHeadAttentionRoPE(
                self.latent_dim, self.num_heads, self.max_seq_len
            )(
                x_norm,
                use_causal_mask=use_causal_mask,
                start_pos=start_pos,
                cache=cache,
                deterministic=deterministic,
            )
        else:
            raise ValueError(
                f"Unknown attn_type '{self.attn_type}'. "
                f"Expected one of: 'ta', 'gta', 'gqa', 'mha', 'mha_rope'."
            )

        x = x + attn_out

        ffn_norm = nn.LayerNorm(epsilon=1e-6)(x)

        ffn_out = nn.Dense(
            self.latent_dim * 4,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(ffn_norm)

        ffn_out = nn.gelu(ffn_out)

        ffn_out = nn.Dropout(rate=0.1)(ffn_out, deterministic=deterministic)

        ffn_out = nn.Dense(
            self.latent_dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(ffn_out)

        x = x + ffn_out

        return x, new_cache


class CausalLM(nn.Module):
    vocab_size: int
    max_seq_len: int
    latent_dim: int
    num_heads: int
    num_layers: int
    attn_type: str
    num_kv_heads: int = 4

    @nn.compact
    def __call__(self, inputs, use_causal_mask=False, current_pos=0, caches=None, deterministic=True):
        seq_len = inputs.shape[-1]

        token_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.latent_dim)
        x = token_emb(inputs)

        if self.attn_type == "mha":
            pos_emb = nn.Embed(
                num_embeddings=self.max_seq_len, features=self.latent_dim
            )
            positions = current_pos + jnp.arange(seq_len, dtype=jnp.int32)
            positions = positions.reshape(1, -1)
            x = x + pos_emb(positions)

        new_caches = []
        for i in range(self.num_layers):
            layer_cache = caches[i] if caches is not None else None
            x, layer_new_cache = DecoderBlock(
                self.latent_dim,
                self.num_heads,
                self.attn_type,
                self.max_seq_len,
                self.num_kv_heads,
            )(
                x,
                use_causal_mask=use_causal_mask,
                start_pos=current_pos,
                cache=layer_cache,
                deterministic=deterministic,
            )
            new_caches.append(layer_new_cache)

        x = nn.LayerNorm(epsilon=1e-6)(x)
        # Cast to float32 at the model boundary. token_emb.attend(x) produces
        # bfloat16 logits over a 50k-class vocabulary. Upcasting here costs
        # one cheap elementwise op and ensures downstream consumers (argmax,
        # softmax sampling, cross-entropy loss) operate with full precision.
        logits = token_emb.attend(x).astype(jnp.float32)

        return logits, new_caches
