from typing import Callable, Optional, Tuple

import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from pluto.trm.models.common import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]

# "sdpa" (default) or "flash_attn" — set via set_attention_backend() before building the model.
_ATTENTION_BACKEND: str = "sdpa"


def set_attention_backend(backend: str) -> None:
    if backend not in ("sdpa", "flash_attn"):
        raise ValueError(f"attention backend must be 'sdpa' or 'flash_attn', got {backend!r}")
    global _ATTENTION_BACKEND
    _ATTENTION_BACKEND = backend


def get_attention_backend() -> str:
    return _ATTENTION_BACKEND


def _sdpa_attn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, causal: bool) -> torch.Tensor:
    return scaled_dot_product_attention(query=query, key=key, value=value, is_causal=causal)


def _flash_attn_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, causal: bool) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_func  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "flash_attn is not installed. Install a matching wheel or use --attention-backend sdpa (default)."
        ) from e
    # flash_attn_func expects (batch, seqlen, nheads, headdim); we have (B, H, S, D)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
    return out.transpose(1, 2)


def _resolve_attn_impl() -> Callable[..., torch.Tensor]:
    return _flash_attn_fn if _ATTENTION_BACKEND == "flash_attn" else _sdpa_attn


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class RotaryEmbedding2d(nn.Module):
    """2D RoPE for square grids (ml-jku/SE-RRM ``rope2d``); sequence index is row-major over sqrt(seq_len)."""

    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 4, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        width = int(max_position_embeddings**0.5)
        t_row = t // width
        t_col = t % width
        freqs_row = torch.outer(t_row, inv_freq)
        freqs_col = torch.outer(t_col, inv_freq)
        emb = torch.cat((freqs_row, freqs_col, freqs_row, freqs_col), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query, key, value = map(lambda t: einops.rearrange(t, "B S H D -> B H S D"), (query, key, value))
        attn_impl = _resolve_attn_impl()
        attn_output = attn_impl(query, key, value, causal=self.causal)
        attn_output = einops.rearrange(attn_output, "B H S D -> B S H D")
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
