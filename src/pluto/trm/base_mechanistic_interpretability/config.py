"""Minimal TRM config for modular-addition grokking (Progress Measures setup)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ModAddGrokkingConfig:
    """Matches arXiv:2301.05217 mainline experiment (mod addition mod P)."""

    p: int = 113
    frac_train: float = 0.3
    seed: int = 0
    eq_token: int = 113  # vocab index for '=' token

    # Optimizer (Nanda et al. §3)
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    max_steps: int = 20_000
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 2000

    # TRM architecture (minimal base, grokking-friendly)
    hidden_size: int = 128
    num_heads: int = 4
    expansion: float = 4.0
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 2
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    pos_encodings: str = "learned"
    forward_dtype: str = "float32"
    loss_type: str = "stablemax_cross_entropy"
    fidelity: str = "B"
    # When True, use Nanda-faithful one-layer encoder (no z_H/z_L, ReLU MLP, no norm).
    nanda_bypass: bool = False
    warmup_steps: int = 10
    # Direct TRM training (no ACT): flat_causal | flat_bidir | recursive
    trm_direct_mode: str = ""

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def seq_len(self) -> int:
        return 3  # a, b, =

    @property
    def num_pairs(self) -> int:
        return self.p * self.p

    def to_model_dict(self, batch_size: int) -> Dict[str, Any]:
        return {
            "batch_size": batch_size,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "num_puzzle_identifiers": 1,
            "puzzle_emb_ndim": 0,
            "H_cycles": self.H_cycles,
            "L_cycles": self.L_cycles,
            "H_layers": 0,
            "L_layers": self.L_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "expansion": self.expansion,
            "pos_encodings": self.pos_encodings,
            "forward_dtype": self.forward_dtype,
            "halt_max_steps": self.halt_max_steps,
            "halt_exploration_prob": self.halt_exploration_prob,
            "mlp_t": False,
            "puzzle_emb_len": 0,
            "no_ACT_continue": True,
        }


@dataclass(frozen=True)
class NandaBaselineConfig:
    """One-layer Nanda-style baseline (fidelity B default)."""

    p: int = 113
    frac_train: float = 0.3
    seed: int = 0
    eq_token: int = 113
    hidden_size: int = 128
    num_heads: int = 4
    n_layers: int = 1
    attn_only: bool = False
    expansion: float = 4.0
    pos_encodings: str = "learned"
    forward_dtype: str = "float32"
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    max_steps: int = 20_000
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 2000
    fidelity: str = "B"
    # Faithful reproduction of Nanda et al. (arXiv:2301.05217) one-layer transformer:
    # no normalization, ReLU MLP with biases, randn/sqrt init, LR warmup, high-precision CE.
    faithful: bool = False
    grokking_ce: bool = False
    warmup_steps: int = 10

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def seq_len(self) -> int:
        return 3


def trm_minimal_config(**kwargs: Any) -> ModAddGrokkingConfig:
    defaults = dict(
        L_layers=1,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=1,
        max_steps=20_000,
        save_every=2000,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def nanda_config_from_modadd(cfg: ModAddGrokkingConfig) -> NandaBaselineConfig:
    """Map modular-add TRM config to Nanda-faithful one-layer hyperparameters."""
    return NandaBaselineConfig(
        p=cfg.p,
        frac_train=cfg.frac_train,
        seed=cfg.seed,
        eq_token=cfg.eq_token,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        expansion=cfg.expansion,
        n_layers=1,
        attn_only=False,
        faithful=True,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        max_steps=cfg.max_steps,
        log_every=cfg.log_every,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        warmup_steps=cfg.warmup_steps,
    )


def trm_direct_flat_causal_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """SwiGLU + causal attn, flat (no z_H/z_L), hi-prec CE, no ACT."""
    defaults = dict(
        L_layers=1,
        trm_direct_mode="flat_causal",
        max_steps=50_000,
        save_every=2000,
        warmup_steps=10,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_direct_flat_bidir_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """SwiGLU + bidirectional attn, flat (no z_H/z_L), hi-prec CE, no ACT."""
    defaults = dict(
        L_layers=1,
        trm_direct_mode="flat_bidir",
        max_steps=50_000,
        save_every=2000,
        warmup_steps=10,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_direct_recursive_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """SwiGLU + bidirectional attn + z_H/z_L (H=1,L=1), hi-prec CE, no ACT."""
    defaults = dict(
        L_layers=1,
        H_cycles=1,
        L_cycles=1,
        trm_direct_mode="recursive",
        max_steps=50_000,
        save_every=2000,
        warmup_steps=10,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_minimal_hiprec_act_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """TRM minimal (bidir + z_H/z_L + ACT) with hi-prec float64 CE on lm_loss."""
    defaults = dict(
        L_layers=1,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=1,
        max_steps=50_000,
        save_every=2000,
        loss_type="cross_entropy_high_precision",
        fidelity="A",
        warmup_steps=10,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def nanda_faithful_50k_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """Nanda-faithful 50k protocol: float64 CE, wd=1.0, 30% train, P=113.

    Used for all hero runs (Nanda 1L, TRM minimal, TRM full contrast).
    Matches train_nanda_baseline.py --faithful hi-prec CE.
    """
    return trm_minimal_hiprec_act_config(**kwargs)


def trm_direct_full_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """Full TRM depth (L=2, H=3, L=6) direct training, hi-prec CE, no ACT."""
    defaults = dict(
        L_layers=2,
        H_cycles=3,
        L_cycles=6,
        trm_direct_mode="recursive",
        max_steps=50_000,
        save_every=2000,
        warmup_steps=10,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_minimal_nanda_bypass_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """Minimal TRM shell with Nanda-faithful encoder (ablation: where does 5/5 FVE form?)."""
    defaults = dict(
        L_layers=1,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=1,
        nanda_bypass=True,
        max_steps=50_000,
        save_every=2000,
        loss_type="cross_entropy",
        fidelity="A",
        warmup_steps=10,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_minimal_l2_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """Minimal TRM with two L-level blocks (ablation vs Nanda one-layer depth)."""
    defaults = dict(
        L_layers=2,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=1,
        max_steps=50_000,
        save_every=2000,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_full_config(**kwargs: Any) -> ModAddGrokkingConfig:
    defaults = dict(
        L_layers=2,
        H_cycles=2,
        L_cycles=2,
        halt_max_steps=8,
        max_steps=20_000,
        save_every=2000,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def trm_fidelity_a_config(**kwargs: Any) -> ModAddGrokkingConfig:
    """Strict Nanda appendix: standard CE, fidelity A."""
    defaults = dict(
        loss_type="cross_entropy",
        fidelity="A",
        max_steps=20_000,
        save_every=2000,
    )
    defaults.update(kwargs)
    return ModAddGrokkingConfig(**defaults)


def mod_add_dataset_config(cfg: ModAddGrokkingConfig | NandaBaselineConfig) -> ModAddGrokkingConfig:
    if isinstance(cfg, ModAddGrokkingConfig):
        return cfg
    return ModAddGrokkingConfig(p=cfg.p, frac_train=cfg.frac_train, seed=cfg.seed, eq_token=cfg.eq_token)


def gen_train_test_indices(
    p: int, frac_train: float, seed: int
) -> Tuple[List[Tuple[int, int, int]], List[bool], List[bool]]:
    """Build train/test masks over all (a,b) pairs (Nanda paper ``gen_train_test``)."""
    import random

    rng = random.Random(seed)
    pairs: List[Tuple[int, int, int]] = []
    is_train: List[bool] = []
    is_test: List[bool] = []
    eq = p
    n_train = int(round(frac_train * p * p))
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)
    train_set = set(all_pairs[:n_train])
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            pairs.append((a, b, eq))
            in_train = (a, b) in train_set
            is_train.append(in_train)
            is_test.append(not in_train)
    return pairs, is_train, is_test
