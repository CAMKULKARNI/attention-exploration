import math
import keras
from keras import ops


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
    return slopes


def apply_rope(x, start_pos, inv_freq):
    seq_len = ops.shape(x)[2]
    depth = ops.shape(x)[3]

    start_pos_f32 = ops.cast(start_pos, "float32")
    seq_len_f32 = ops.cast(seq_len, "float32")

    positions = ops.arange(start_pos_f32, start_pos_f32 + seq_len_f32, dtype="float32")
    positions = ops.reshape(positions, (-1, 1))

    angles = ops.matmul(positions, inv_freq)
    sin = ops.reshape(ops.sin(angles), (1, 1, seq_len, depth // 2))
    cos = ops.reshape(ops.cos(angles), (1, 1, seq_len, depth // 2))

    x_fp32 = ops.cast(x, "float32")
    half_depth = depth // 2
    x1, x2 = x_fp32[..., :half_depth], x_fp32[..., half_depth:]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    rotated_out = ops.concatenate([rotated_x1, rotated_x2], axis=-1)
    return ops.cast(rotated_out, x.dtype)


class GroupTiedAttention(keras.layers.Layer):
    def __init__(self, latent_dim: int, num_heads: int, max_seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.depth = latent_dim // num_heads

        self.wq = keras.layers.Dense(latent_dim, use_bias=False)
        self.wa = keras.layers.Dense(latent_dim, use_bias=False)
        self.dense = keras.layers.Dense(latent_dim, use_bias=False)

        slopes = get_alibi_slopes(num_heads)
        self.alibi_slopes = ops.convert_to_tensor(slopes, dtype="float32")

    def split_heads(self, x, batch_size):
        x = ops.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return ops.transpose(x, axes=[0, 2, 1, 3])

    def call(self, x, use_causal_mask=False, start_pos=0, cache=None):
        batch_size = ops.shape(x)[0]
        q_seq_len = ops.shape(x)[1]

        q = self.split_heads(self.wq(x), batch_size)
        new_a = self.split_heads(self.wa(x), batch_size)

        if cache is not None:
            a_use = ops.concatenate([cache, new_a], axis=2)
        else:
            a_use = new_a

        kv_seq_len = ops.shape(a_use)[2]
        a_t = ops.transpose(a_use, axes=[0, 1, 3, 2])

        matmul_qa = ops.matmul(q, a_t)
        scaled_attention_logits = ops.cast(matmul_qa, "float32") / math.sqrt(self.depth)

        q_idx = ops.reshape(
            ops.arange(start_pos, start_pos + q_seq_len, dtype="float32"), (1, 1, -1, 1)
        )
        kv_range = ops.reshape(ops.arange(kv_seq_len, dtype="float32"), (1, 1, 1, -1))

        distance = kv_range - q_idx
        slopes = ops.reshape(self.alibi_slopes, (1, self.num_heads, 1, 1))
        scaled_attention_logits += slopes * distance

        if use_causal_mask:
            causal_mask = ops.cast(kv_range <= q_idx, "float32")
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)
        attention_weights = ops.cast(attention_weights, x.dtype)

        output = ops.transpose(ops.matmul(attention_weights, a_use), axes=[0, 2, 1, 3])
        return self.dense(ops.reshape(output, (batch_size, -1, self.latent_dim))), a_use


class GroupedQueryAttention(keras.layers.Layer):
    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.latent_dim = latent_dim
        self.depth = latent_dim // num_heads

        self.wq = keras.layers.Dense(latent_dim, use_bias=False)
        self.wk = keras.layers.Dense(num_kv_heads * self.depth, use_bias=False)
        self.wv = keras.layers.Dense(num_kv_heads * self.depth, use_bias=False)
        self.dense = keras.layers.Dense(latent_dim, use_bias=False)

        freqs = ops.arange(0, self.depth, 2, dtype="float32") / self.depth
        self.inv_freq = ops.reshape(1.0 / (10000.0**freqs), (1, -1))

    def split_heads(self, x, batch_size, heads):
        x = ops.reshape(x, (batch_size, -1, heads, self.depth))
        return ops.transpose(x, axes=[0, 2, 1, 3])

    def call(self, x, use_causal_mask=False, start_pos=0, cache=None):
        batch_size = ops.shape(x)[0]
        q_seq_len = ops.shape(x)[1]

        q = self.split_heads(self.wq(x), batch_size, self.num_heads)
        new_k = self.split_heads(self.wk(x), batch_size, self.num_kv_heads)
        new_v = self.split_heads(self.wv(x), batch_size, self.num_kv_heads)

        q = apply_rope(q, start_pos, self.inv_freq)
        new_k = apply_rope(new_k, start_pos, self.inv_freq)

        if cache is not None:
            k_cache, v_cache = cache
            k_use = ops.concatenate([k_cache, new_k], axis=2)
            v_use = ops.concatenate([v_cache, new_v], axis=2)
        else:
            k_use, v_use = new_k, new_v

        kv_seq_len = ops.shape(k_use)[2]

        q_reshaped = ops.reshape(
            q,
            (
                batch_size,
                self.num_kv_heads,
                self.num_queries_per_kv,
                q_seq_len,
                self.depth,
            ),
        )
        matmul_qk = ops.einsum("b h g q d, b h k d -> b h g q k", q_reshaped, k_use)
        matmul_qk = ops.reshape(
            matmul_qk, (batch_size, self.num_heads, q_seq_len, kv_seq_len)
        )

        scaled_attention_logits = ops.cast(matmul_qk, "float32") / math.sqrt(self.depth)

        if use_causal_mask:
            q_idx = ops.reshape(
                ops.arange(start_pos, start_pos + q_seq_len, dtype="float32"),
                (1, 1, -1, 1),
            )
            kv_range = ops.reshape(
                ops.arange(kv_seq_len, dtype="float32"), (1, 1, 1, -1)
            )
            causal_mask = ops.cast(kv_range <= q_idx, "float32")
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)
        attention_weights = ops.cast(attention_weights, x.dtype)

        attention_weights_reshaped = ops.reshape(
            attention_weights,
            (
                batch_size,
                self.num_kv_heads,
                self.num_queries_per_kv,
                q_seq_len,
                kv_seq_len,
            ),
        )
        output = ops.einsum(
            "b h g q k, b h k d -> b h g q d", attention_weights_reshaped, v_use
        )
        output = ops.reshape(
            output, (batch_size, self.num_heads, q_seq_len, self.depth)
        )

        output = ops.transpose(output, axes=[0, 2, 1, 3])
        return self.dense(ops.reshape(output, (batch_size, -1, self.latent_dim))), (
            k_use,
            v_use,
        )


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, latent_dim: int, num_heads: int, max_seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.depth = latent_dim // num_heads

        self.wq = keras.layers.Dense(latent_dim, use_bias=False)
        self.wk = keras.layers.Dense(latent_dim, use_bias=False)
        self.wv = keras.layers.Dense(latent_dim, use_bias=False)
        self.dense = keras.layers.Dense(latent_dim, use_bias=False)

    def split_heads(self, x, batch_size):
        x = ops.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return ops.transpose(x, axes=[0, 2, 1, 3])

    def call(self, x, use_causal_mask=False, start_pos=0, cache=None):
        batch_size = ops.shape(x)[0]
        q_seq_len = ops.shape(x)[1]

        q = self.split_heads(self.wq(x), batch_size)
        new_k = self.split_heads(self.wk(x), batch_size)
        new_v = self.split_heads(self.wv(x), batch_size)

        if cache is not None:
            k_cache, v_cache = cache
            k_use = ops.concatenate([k_cache, new_k], axis=2)
            v_use = ops.concatenate([v_cache, new_v], axis=2)
        else:
            k_use, v_use = new_k, new_v

        kv_seq_len = ops.shape(k_use)[2]
        k_t = ops.transpose(k_use, axes=[0, 1, 3, 2])

        matmul_qk = ops.matmul(q, k_t)
        matmul_qk = ops.reshape(
            matmul_qk, (batch_size, self.num_heads, q_seq_len, kv_seq_len)
        )
        scaled_attention_logits = ops.cast(matmul_qk, "float32") / math.sqrt(self.depth)

        if use_causal_mask:
            q_idx = ops.reshape(
                ops.arange(start_pos, start_pos + q_seq_len, dtype="float32"),
                (1, 1, -1, 1),
            )
            kv_range = ops.reshape(
                ops.arange(kv_seq_len, dtype="float32"), (1, 1, 1, -1)
            )
            causal_mask = ops.cast(kv_range <= q_idx, "float32")
            scaled_attention_logits += (1.0 - causal_mask) * -1e9

        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)
        attention_weights = ops.cast(attention_weights, x.dtype)

        output = ops.transpose(ops.matmul(attention_weights, v_use), axes=[0, 2, 1, 3])
        return self.dense(ops.reshape(output, (batch_size, -1, self.latent_dim))), (
            k_use,
            v_use,
        )


class DecoderBlock(keras.layers.Layer):
    def __init__(
        self, latent_dim, num_heads, attn_type, max_seq_len, num_kv_heads=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-6)

        if attn_type == "gta":
            self.attn = GroupTiedAttention(latent_dim, num_heads, max_seq_len)
        elif attn_type == "gqa":
            self.attn = GroupedQueryAttention(
                latent_dim, num_heads, num_kv_heads, max_seq_len
            )
        elif attn_type == "mha":
            self.attn = MultiHeadAttention(latent_dim, num_heads, max_seq_len)

        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(latent_dim * 4, activation="gelu"),
                keras.layers.Dense(latent_dim),
            ]
        )

    def call(self, x, use_causal_mask=False, start_pos=0, cache=None):
        x_norm = self.ln1(x)
        attn_out, new_cache = self.attn(
            x_norm, use_causal_mask=use_causal_mask, start_pos=start_pos, cache=cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class CausalLM(keras.Model):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        latent_dim,
        num_heads,
        num_layers,
        attn_type,
        num_kv_heads=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.attn_type = attn_type

        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=latent_dim
        )
        if self.attn_type == "mha":
            self.pos_emb = keras.layers.Embedding(
                input_dim=max_seq_len, output_dim=latent_dim
            )

        self.decoder_blocks = [
            DecoderBlock(latent_dim, num_heads, attn_type, max_seq_len, num_kv_heads)
            for _ in range(num_layers)
        ]
        self.final_ln = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, use_causal_mask=False, current_pos=0, caches=None):
        seq_len = ops.shape(inputs)[-1]
        x = self.token_emb(inputs)

        if self.attn_type == "mha":
            start_pos_int = ops.cast(current_pos, "int32")
            seq_len_int = ops.cast(seq_len, "int32")
            positions = ops.reshape(
                ops.arange(start_pos_int, start_pos_int + seq_len_int, dtype="int32"),
                (1, -1),
            )
            x = x + self.pos_emb(positions)

        new_caches = []
        for i in range(self.num_layers):
            layer_cache = caches[i] if caches is not None else None
            x, layer_new_cache = self.decoder_blocks[i](
                x,
                use_causal_mask=use_causal_mask,
                start_pos=current_pos,
                cache=layer_cache,
            )
            new_caches.append(layer_new_cache)

        x = self.final_ln(x)
        lm_head_kernel = ops.transpose(self.token_emb.embeddings, axes=[1, 0])
        logits = ops.matmul(x, lm_head_kernel)

        return logits, new_caches
