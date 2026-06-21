"""Direct TRM training (no ACT / q_halt) for mod-add grokking ablations."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from pluto.trm.models.common import trunc_normal_init_
from pluto.trm.models.layers import Attention, CastedEmbedding, CastedLinear, CosSin, SwiGLU, rms_norm
from pluto.trm.models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
)


class TrmDirectBlock(nn.Module):
    """TRM encoder block (RMSNorm + SwiGLU) with configurable causal attention."""

    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, *, causal: bool) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class TrmDirectModel(nn.Module):
    """TRM minimal encoder trained without ACT: flat or z_H/z_L recursive forward."""

    def __init__(self, act_cfg: TinyRecursiveReasoningModel_ACTV1Config, *, mode: str) -> None:
        super().__init__()
        if mode not in ("flat_causal", "flat_bidir", "recursive"):
            raise ValueError(f"unknown trm direct mode: {mode}")
        self.mode = mode
        self.config = act_cfg
        self.forward_dtype = getattr(torch, act_cfg.forward_dtype)
        self.embed_scale = math.sqrt(act_cfg.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(
            act_cfg.vocab_size, act_cfg.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(act_cfg.hidden_size, act_cfg.vocab_size, bias=False)
        self.embed_pos = CastedEmbedding(
            act_cfg.seq_len, act_cfg.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        causal = mode == "flat_causal"
        layers = [TrmDirectBlock(act_cfg, causal=causal if mode.startswith("flat") else False)]
        self.encoder = nn.ModuleList(layers)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TrmDirectBlock(act_cfg, causal=False) for _ in range(act_cfg.L_layers)]
        )
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(act_cfg.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(act_cfg.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def _flat_forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        hidden = input_embeddings
        for layer in self.encoder:
            hidden = layer(None, hidden)
        return self.lm_head(hidden)

    def _recursive_forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input_embeddings.shape
        device = input_embeddings.device
        z_H = self.H_init.view(1, 1, -1).expand(batch_size, seq_len, -1)
        z_L = self.L_init.view(1, 1, -1).expand(batch_size, seq_len, -1)
        seq_info = dict(cos_sin=None)
        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _l in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        return self.lm_head(z_H)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeddings = self._input_embeddings(batch["inputs"])
        if self.mode.startswith("flat"):
            return self._flat_forward(input_embeddings)
        return self._recursive_forward(input_embeddings)
