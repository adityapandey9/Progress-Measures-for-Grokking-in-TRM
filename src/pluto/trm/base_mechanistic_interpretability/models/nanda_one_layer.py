"""One-layer transformer baseline for Nanda progress measures (arXiv:2301.05217)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from pluto.trm.base_mechanistic_interpretability.config import NandaBaselineConfig
from pluto.trm.models.layers import Attention, CastedEmbedding, CastedLinear, CosSin, SwiGLU, rms_norm


class NandaTransformerBlock(nn.Module):
    def __init__(self, cfg: NandaBaselineConfig) -> None:
        super().__init__()
        head_dim = cfg.hidden_size // cfg.num_heads
        self.self_attn = Attention(
            hidden_size=cfg.hidden_size,
            head_dim=head_dim,
            num_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_heads,
            causal=False,
        )
        self.mlp = None if cfg.attn_only else SwiGLU(hidden_size=cfg.hidden_size, expansion=cfg.expansion)
        self.norm_eps = 1e-5

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        if self.mlp is not None:
            hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class NandaOneLayerTransformer(nn.Module):
    def __init__(self, cfg: NandaBaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        embed_init = 1.0
        self.embed_tokens = CastedEmbedding(
            cfg.vocab_size, cfg.hidden_size, init_std=embed_init, cast_to=torch.float32
        )
        self.embed_pos = CastedEmbedding(cfg.seq_len, cfg.hidden_size, init_std=embed_init, cast_to=torch.float32)
        self.embed_scale = math.sqrt(cfg.hidden_size)
        self.block = NandaTransformerBlock(cfg)
        self.lm_head = CastedLinear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_scale * self.embed_tokens(input_ids)
        pos = self.embed_pos.embedding_weight.unsqueeze(0).expand(emb.shape[0], -1, -1)
        return 0.707106781 * (emb + pos)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(input_ids)
        x = self.block(None, x)
        return self.lm_head(x)

    def logits_at_equals(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)[:, 2, : self.cfg.p]


class _NandaFaithfulAttention(nn.Module):
    """Causal attention matching the official Nanda repo (transformers.py)."""

    def __init__(self, d_model: int, num_heads: int, n_ctx: int) -> None:
        super().__init__()
        d_head = d_model // num_heads
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / math.sqrt(d_model))
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-2]
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:n, :n])
        attn = torch.softmax(attn_scores_masked / math.sqrt(self.d_head), dim=-1)
        z = torch.einsum("biph,biqp->biqh", v, attn)
        z_flat = z.permute(0, 2, 1, 3).reshape(x.shape[0], n, self.num_heads * self.d_head)
        return torch.einsum("df,bqf->bqd", self.W_O, z_flat)


class _NandaFaithfulMLP(nn.Module):
    """ReLU MLP with biases (d_mlp = 4*d_model), matching the official repo."""

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        h = torch.relu(h)
        return torch.einsum("dm,bpm->bpd", self.W_out, h) + self.b_out


class NandaFaithfulTransformer(nn.Module):
    """Faithful reproduction of Nanda et al. one-layer transformer (arXiv:2301.05217).

    No normalization (``use_ln=False``), ReLU MLP with biases, separate
    ``randn/sqrt`` embeddings/unembeddings, additive causal attention. This
    mirrors the official ``mechanistic-interpretability-grokking`` repo so the
    progress-measure stack can calibrate against the reported ~95%+ trig-FVE.
    """

    def __init__(self, cfg: NandaBaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_size
        d_vocab = cfg.vocab_size
        d_mlp = int(cfg.expansion * d_model)
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(cfg.seq_len, d_model) / math.sqrt(d_model))
        self.attn = _NandaFaithfulAttention(d_model, cfg.num_heads, cfg.seq_len)
        self.mlp = _NandaFaithfulMLP(d_model, d_mlp)
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / math.sqrt(d_vocab))

    @property
    def embed_tokens(self):  # compat shim for analysis code reading embedding_weight
        outer = self

        class _W:
            embedding_weight = outer.W_E.T

        return _W()

    @property
    def lm_head(self):  # compat shim: [vocab, hidden] like CastedLinear.weight
        outer = self

        class _Head:
            weight = outer.W_U.T

        return _Head()

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.W_E[:, input_ids].permute(1, 2, 0)
        x = x + self.W_pos[: x.shape[-2]]
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids) @ self.W_U

    def logits_at_equals(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)[:, 2, : self.cfg.p]
