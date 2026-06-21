from typing import Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel, ConfigDict

from pluto.trm.models.common import trunc_normal_init_
from pluto.trm.models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from pluto.trm.models.sparse_embedding import CastedSparseEmbedding


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    z_H_streams: Optional[torch.Tensor] = None
    z_L_streams: Optional[torch.Tensor] = None


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True
    # Optional: shared with optimize_trm; ignored by this module's blocks.
    attention_type: str = "mla"
    kv_lora_rank: int = 512
    # optimize_trm only: Full AttnRes over L-layer sublayer deltas (default False).
    attention_residual: bool = False
    # optimize_trm only: accumulate those deltas across H/L recurrent L_level calls into one depth axis (default False).
    recurrent_cycles_as_attn_depth: bool = False
    # optimize_trm only: store AttnRes sublayer deltas in low rank r (MLA-style); default False = full D per slot.
    attention_residual_latent: bool = False
    attention_residual_latent_rank: int = 128
    # attn_residual_base only: multi-head depth AttnRes (softmax over depth per head).
    attention_residual_multihead: bool = False
    attention_residual_num_heads: int = 8
    # attn_residual_base only: stack multiple depth-AttnRes layers with residual + RMSNorm.
    attention_residual_multilayer: bool = False
    attention_residual_num_layers: int = 2
    # attn_residual_base only: cross-task depth memory (default off; see spec 2026-06-07).
    attention_residual_xtask: bool = False
    attention_residual_xtask_mode: str = "proto"  # {proto, pos, full}
    attention_residual_xtask_capacity: int = 64
    attention_residual_xtask_update: str = "fifo"  # {fifo, ema}
    attention_residual_xtask_ema_decay: float = 0.99
    attention_residual_xtask_gate_init: float = -2.0
    attention_residual_xtask_at_inference: bool = False
    attention_residual_xtask_topk: int = 0  # v2 only; rejected if > 0
    attention_residual_xtask_utility_write: bool = False  # v2 only; rejected if True
    attention_residual_xtask_mem_dropout: float = 0.0  # train-time prob of dropping the memory read (robust use)
    attention_residual_xtask_norm_mem: bool = True  # rms-norm the memory readout before gating
    # optimize_trm only: Hyperloop-style loop-level hyper-connections (middle block looped R times; mutually exclusive with attention_residual).
    hyperloop: bool = False
    hyperloop_n_streams: int = 4
    hyperloop_middle_loops: int = 3
    hyperloop_pre_layers: int = 0
    hyperloop_mid_layers: int = 0
    hyperloop_post_layers: int = 0
    hyperloop_pre_frac: float = 0.25
    hyperloop_post_frac: float = 0.25
    # Paper eq. 36: sigmoid on H_pre (mHC reference codebases often use softmax).
    hyperloop_pre_mix: str = "sigmoid"
    # Paper eq. 38 diagonal sigmoid (Hyperloop) vs eq. 35 Sinkhorn (mHC baseline).
    hyperloop_h_res_mode: str = "diagonal"
    hyperloop_sinkhorn_iters: int = 20
    # optimize_trm: optional multi-stream recurrent bridge after each L_level on z_H / z_L (requires hyperloop=True).
    hyperloop_recurrent_z_h: bool = False
    hyperloop_recurrent_z_l: bool = False
    # optimize_trm only: inner-loop ACT head on z_H (per-example halt of further H/L refinement).
    inner_act: bool = False
    # optimize_trm only: training-time random cap on effective depth (see pluto/trm/optimize_trm/trm.py).
    train_depth_cap: bool = False
    train_depth_cap_low: int = 10
    train_depth_cap_high: int = 15
    train_depth_cap_rng_seed: int = 0
    # optimize_trm only: sample H_cycles / L_cycles from cap bounds (train+val; not clamped to H_cycles/L_cycles).
    train_cycle_cap: bool = False
    train_h_cycles_cap_low: int = 1
    train_h_cycles_cap_high: int = 0  # 0 = use H_cycles at runtime
    train_l_cycles_cap_low: int = 1
    train_l_cycles_cap_high: int = 0  # 0 = use L_cycles at runtime
    train_cycle_cap_at_inference: bool = False
    train_depth_cap_at_inference: bool = False
    # cfg_trm (DIS-style): classifier-free dropout on timestep embed; inference guidance scale on logits.
    cf_uncond_prob: float = 0.0
    cfg_guidance_w: float = 1.0
    # parcae_trm: stable diagonal injection on recurrent state + normalized prelude injection.
    parcae_prelude_rms: bool = True
    parcae_prelude_ln: bool = False
    parcae_init_log_a: float = 0.0
    parcae_init_dt_bias: float = 0.0
    parcae_b_cross_init: float = 1.0
    parcae_use_readout: bool = False
    # parcae_trm: Parcae-style truncated BPTT (no_grad warmup + grad tail) on each L_level call.
    parcae_truncated_bptt: bool = False
    parcae_mean_recurrence: int = 12
    parcae_mean_backprop_depth: int = 2
    parcae_sampling_scheme: str = "fixed"  # "fixed" | "poisson_truncated_full"
    # parcae_trm: on ACT puzzle reset, draw z_H/z_L like embedding init (Parcae "like-init") instead of fixed buffers.
    parcae_like_state_init: bool = False
    # mg_trm (Model-Guidance): scale s conditioning + training mixture (Diffusion-wo-CFG / ScaleAware style).
    mg_mgw_low: float = 1.45
    mg_mgw_high: float = 1.45
    mg_data_ratio_mg: float = 0.2
    mg_data_ratio_drop: float = 0.1
    mg_inference_scale: float = 1.45
    # cfg_trm / mg_trm: DIS flow-style training time (optional; inference unchanged).
    dis_flow_supervision: Literal["none", "flow_matching", "rectified_flow"] = "none"
    dis_flow_embed_lerp: bool = False
    dis_flow_loss_weight: float = 1.0
    dis_flow_velocity_only: bool = False
    dis_flow_rf_x0_noise: bool = True
    # se_rrm: symbol-equivariant RRM (axial attention on positions + symbol plane); see pluto/trm/se_rrm/
    symbol_equivariant: bool = False
    equivariant_symbols: bool = True
    num_symbol_slots: int = 0  # 0 -> use vocab_size at runtime
    se_rrm_legacy: bool = False  # dual-z pluto port; False = ml-jku trm_equi single-z
    dis_strict_inner: bool = True  # cfg_trm/mg_trm: DIS MHA inner (False = optimize_trm stack)
    add_tokens: int = 0
    se_rrm_dropout: float = 0.0
    num_heads_t: int = 0  # 0 -> num_heads
    head_dim: int = 0  # 0 -> hidden_size // num_heads
    head_dim_t: int = 0  # 0 -> head_dim
    # ras_rrm: role-aware symbolic reasoning with [B,T,D] state (see pluto/trm/ras_rrm/)
    role_aware: bool = False
    ras_symbol_candidates: int = 0  # 0 = full vocab when K is small
    ras_relational_bias: bool = True
    # ras_rrm_v2: budgeted union, dynamic roles, candidate memory, prototypes
    ras_v2: bool = False
    ras_num_role_slots: int = 8
    ras_prototypes_per_role: int = 4
    ras_candidate_memory: int = 0  # 0 = auto (C_total // 8)
    ras_v2_prototypes: bool = True
    ras_v2_candidate_memory: bool = True
    ras_gold_force_prob: float = 0.0
    ras_candidate_recall_loss_weight: float = 0.0
    ras_role_entropy_weight: float = 0.01


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(hidden_size=self.config.seq_len + self.puzzle_emb_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
        else:
            self.puzzle_emb = None
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
            )
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)]
        )
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: Union[torch.device, str]):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        assert self.inner.puzzle_emb is not None
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()
        }
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    )
        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
