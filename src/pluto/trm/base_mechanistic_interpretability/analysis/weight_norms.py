#!/usr/bin/env python3
"""Weight norm phase metrics across checkpoints."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json


def _sq_norm(t: torch.Tensor) -> float:
    return float((t.detach().float() ** 2).sum().item())


def summarize_state_dict_norms(state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    groups = {
        "total_sq_norm": 0.0,
        "embedding_sq_norm": 0.0,
        "unembedding_sq_norm": 0.0,
        "attention_sq_norm": 0.0,
        "mlp_sq_norm": 0.0,
        "halt_sq_norm": 0.0,
        "other_sq_norm": 0.0,
    }
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        value = _sq_norm(tensor)
        groups["total_sq_norm"] += value
        lower = name.lower()
        if "embed" in lower and "lm_head" not in lower:
            groups["embedding_sq_norm"] += value
        elif "lm_head" in lower or "unembed" in lower:
            groups["unembedding_sq_norm"] += value
        elif "attn" in lower or "qkv" in lower:
            groups["attention_sq_norm"] += value
        elif "mlp" in lower or "gate_up" in lower or "down_proj" in lower:
            groups["mlp_sq_norm"] += value
        elif "halt" in lower:
            groups["halt_sq_norm"] += value
        else:
            groups["other_sq_norm"] += value
    return groups


def _step_from_checkpoint(path: Path) -> int:
    if path.stem == "checkpoint_final":
        return 10**12
    match = re.search(r"checkpoint_step(\d+)", path.name)
    return int(match.group(1)) if match else -1


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir)
    rows: List[Dict[str, Any]] = []
    for ckpt_path in sorted(run_dir.glob("checkpoint_step*.pt"), key=_step_from_checkpoint):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        row = {"checkpoint": ckpt_path.name, "step": _step_from_checkpoint(ckpt_path)}
        row.update(summarize_state_dict_norms(state))
        rows.append(row)
    final = run_dir / "checkpoint_final.pt"
    if final.exists():
        ckpt = torch.load(final, map_location="cpu", weights_only=False)
        row = {"checkpoint": final.name, "step": "final"}
        row.update(summarize_state_dict_norms(ckpt.get("model", ckpt)))
        rows.append(row)

    results = {"run_dir": str(run_dir), "weight_norms": rows}
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "weight_norms.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/weight_norms")
    args = p.parse_args()
    r = run(args)
    print(f"wrote {len(r['weight_norms'])} weight-norm rows")


if __name__ == "__main__":
    main()
