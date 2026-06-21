"""Modular addition dataset for grokking (arXiv:2301.05217 §3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig, gen_train_test_indices
from pluto.trm.models.losses import IGNORE_LABEL_ID


@dataclass
class ModAddBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    puzzle_identifiers: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


def _build_tensors(cfg: ModAddGrokkingConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[bool], List[bool]]:
    pairs, is_train, is_test = gen_train_test_indices(cfg.p, cfg.frac_train, cfg.seed)
    n = len(pairs)
    inputs = torch.zeros(n, cfg.seq_len, dtype=torch.long)
    labels = torch.full((n, cfg.seq_len), IGNORE_LABEL_ID, dtype=torch.long)
    a_vals = torch.zeros(n, dtype=torch.long)
    b_vals = torch.zeros(n, dtype=torch.long)
    c_vals = torch.zeros(n, dtype=torch.long)
    for i, (a, b, eq) in enumerate(pairs):
        c = (a + b) % cfg.p
        inputs[i] = torch.tensor([a, b, eq], dtype=torch.long)
        labels[i, 2] = c
        a_vals[i] = a
        b_vals[i] = b
        c_vals[i] = c
    return inputs, labels, torch.stack([a_vals, b_vals, c_vals], dim=1), is_train, is_test


class ModAddFullDataset(Dataset):
    """All p² pairs; index with train/test masks."""

    def __init__(self, cfg: ModAddGrokkingConfig):
        self.cfg = cfg
        self.inputs, self.labels, self.abc, self.is_train, self.is_test = _build_tensors(cfg)
        self.puzzle_ids = torch.zeros(len(self.inputs), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx],
            "puzzle_identifiers": self.puzzle_ids[idx],
        }

    @property
    def train_mask(self) -> torch.Tensor:
        return torch.tensor(self.is_train, dtype=torch.bool)

    @property
    def test_mask(self) -> torch.Tensor:
        return torch.tensor(self.is_test, dtype=torch.bool)


class ModAddTrainIterable(IterableDataset):
    """Full-batch shuffle iterator for grokking training."""

    def __init__(self, cfg: ModAddGrokkingConfig):
        self.ds = ModAddFullDataset(cfg)
        self.train_idx = torch.where(self.ds.train_mask)[0]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        perm = self.train_idx[torch.randperm(len(self.train_idx))]
        batch = {
            "inputs": self.ds.inputs[perm],
            "labels": self.ds.labels[perm],
            "puzzle_identifiers": self.ds.puzzle_ids[perm],
        }
        yield batch


def all_pairs_batch(cfg: ModAddGrokkingConfig, *, train_only: bool = False, test_only: bool = False) -> Dict[str, torch.Tensor]:
    ds = ModAddFullDataset(cfg)
    mask = torch.ones(len(ds), dtype=torch.bool)
    if train_only:
        mask = ds.train_mask
    elif test_only:
        mask = ds.test_mask
    idx = torch.where(mask)[0]
    return {
        "inputs": ds.inputs[idx],
        "labels": ds.labels[idx],
        "puzzle_identifiers": ds.puzzle_ids[idx],
    }


def save_dataset_artifacts(cfg: ModAddGrokkingConfig, out_dir: str) -> None:
    """Persist train/test split and tensors for reproducibility."""
    import json
    from pathlib import Path

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ds = ModAddFullDataset(cfg)
    torch.save(
        {
            "inputs": ds.inputs,
            "labels": ds.labels,
            "train_mask": ds.train_mask,
            "test_mask": ds.test_mask,
            "config": cfg,
        },
        root / "mod_add_split.pt",
    )
    meta = {
        "p": cfg.p,
        "frac_train": cfg.frac_train,
        "seed": cfg.seed,
        "n_train": int(ds.train_mask.sum()),
        "n_test": int(ds.test_mask.sum()),
        "vocab_size": cfg.vocab_size,
        "seq_len": cfg.seq_len,
    }
    (root / "mod_add_meta.json").write_text(json.dumps(meta, indent=2))
