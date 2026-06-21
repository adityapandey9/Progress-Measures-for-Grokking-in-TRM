"""Faithful progress measures from arXiv:2301.05217 (Nanda et al.)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def fourier_basis(p: int, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    xs = torch.arange(p, device=device, dtype=dtype)
    rows: List[torch.Tensor] = []
    for k in range(p // 2):
        w = 2.0 * math.pi * k / p
        rows.append(torch.cos(w * xs))
        rows.append(torch.sin(w * xs))
    return torch.stack(rows, dim=0)


def fourier_2d_basis_term(x_index: int, y_index: int, basis: torch.Tensor) -> torch.Tensor:
    return (basis[x_index][:, None] * basis[y_index][None, :]).flatten()


def fft1d(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return tensor.to(basis.dtype) @ basis.T


def logits_grid(logits_flat: torch.Tensor, p: int) -> torch.Tensor:
    """[p*p, p] -> [p, p, p] where [:,:,c] is logit for output c."""
    return logits_flat.reshape(p, p, p)


def logits_flat(grid: torch.Tensor) -> torch.Tensor:
    p = grid.shape[0]
    return grid.reshape(p * p, p)


def trig_key_reconstruction(grid: torch.Tensor, key_freqs: Sequence[int]) -> torch.Tensor:
    """Restricted logits: per (a,b) project onto cos/sin(w_k(a+b-c)) output basis (vectorized)."""
    p = grid.shape[0]
    device = grid.device
    a = torch.arange(p, device=device, dtype=torch.float64)
    b = torch.arange(p, device=device, dtype=torch.float64)
    c = torch.arange(p, device=device, dtype=torch.float64)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    row = grid.double()
    recon = torch.zeros_like(row)
    for k in key_freqs:
        w = 2.0 * math.pi * k / p
        phase = w * (aa.unsqueeze(-1) + bb.unsqueeze(-1) - c.view(1, 1, p))
        cos_b = torch.cos(phase)
        sin_b = torch.sin(phase)
        denom_c = cos_b.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
        denom_s = sin_b.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
        recon += (row * cos_b).sum(-1, keepdim=True) / denom_c * cos_b
        recon += (row * sin_b).sum(-1, keepdim=True) / denom_s * sin_b
    return recon.to(grid.dtype)


def subtract_key_components(grid: torch.Tensor, key_freqs: Sequence[int]) -> torch.Tensor:
    """Excluded logits: per (a,b) remove cos/sin(w_k(a+b-c)) projections (vectorized)."""
    p = grid.shape[0]
    device = grid.device
    a = torch.arange(p, device=device, dtype=torch.float64)
    b = torch.arange(p, device=device, dtype=torch.float64)
    c = torch.arange(p, device=device, dtype=torch.float64)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    current = grid.double()
    for k in key_freqs:
        w = 2.0 * math.pi * k / p
        phase = w * (aa.unsqueeze(-1) + bb.unsqueeze(-1) - c.view(1, 1, p))
        for basis in (torch.cos(phase), torch.sin(phase)):
            denom = basis.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
            coeff = (current * basis).sum(-1, keepdim=True) / denom
            current = current - coeff * basis
    return current.to(grid.dtype)


def get_component_cos_xpy(grid: torch.Tensor, freq: int, basis: torch.Tensor, *, collapse: bool = False) -> torch.Tensor:
    """Legacy 2D cos(w(a+b)) component; kept for embedding-style maps."""
    p = grid.shape[0]
    if grid.shape[-1] != p:
        raise ValueError("get_component_cos_xpy expects last dim p (logit outputs)")
    flat = grid.reshape(p, p, p).reshape(p * p, p)
    cosx_cosy = fourier_2d_basis_term(2 * freq - 1, 2 * freq - 1, basis)
    sinx_siny = fourier_2d_basis_term(2 * freq, 2 * freq, basis)
    direction = (cosx_cosy - sinx_siny) / math.sqrt(2)
    denom = direction.pow(2).sum().clamp_min(1e-12)
    proj = (direction[:, None] * (direction[None, :] @ flat.to(basis.dtype))) / denom
    if collapse:
        return proj
    return proj.reshape(p, p, p)


def get_component_sin_xpy(grid: torch.Tensor, freq: int, basis: torch.Tensor, *, collapse: bool = False) -> torch.Tensor:
    p = grid.shape[0]
    flat = grid.reshape(p, p, p).reshape(p * p, p)
    sinx_cosy = fourier_2d_basis_term(2 * freq, 2 * freq - 1, basis)
    cosx_siny = fourier_2d_basis_term(2 * freq - 1, 2 * freq, basis)
    direction = (sinx_cosy + cosx_siny) / math.sqrt(2)
    denom = direction.pow(2).sum().clamp_min(1e-12)
    proj = (direction[:, None] * (direction[None, :] @ flat.to(basis.dtype))) / denom
    if collapse:
        return proj
    return proj.reshape(p, p, p)


def trig_component_sum(grid: torch.Tensor, key_freqs: Sequence[int], basis: torch.Tensor | None = None) -> torch.Tensor:
    _ = basis  # unused; key reconstruction uses 3D cos(w(a+b-c)) basis
    return trig_key_reconstruction(grid, key_freqs)


def cross_entropy_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    sel = mask.nonzero(as_tuple=False).squeeze(-1)
    if sel.numel() == 0:
        return float("nan")
    return F.cross_entropy(logits[sel].float(), labels[sel], reduction="mean").item()


def test_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    mode: str = "all",
    bias_correction: bool = False,
    original_logits: torch.Tensor | None = None,
) -> float:
    if bias_correction and original_logits is not None:
        delta = (original_logits - logits).mean(dim=0, keepdim=True)
        logits = logits + delta
    if mode == "train":
        return cross_entropy_logits(logits, labels, train_mask)
    if mode == "test":
        return cross_entropy_logits(logits, labels, test_mask)
    full = train_mask | test_mask
    return cross_entropy_logits(logits, labels, full)


def calculate_excluded_loss(
    logits_grid: torch.Tensor,
    key_freqs: Sequence[int],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    basis: torch.Tensor,
    *,
    mode: str = "test",
) -> Tuple[float, List[float]]:
    """Remove cos/sin(w_k(a+b-c)) sequentially per frequency; return final + per-freq losses."""
    current = logits_grid.clone()
    per_freq: List[float] = []
    for freq in key_freqs:
        p = logits_grid.shape[0]
        device = logits_grid.device
        a = torch.arange(p, device=device, dtype=torch.float64)
        b = torch.arange(p, device=device, dtype=torch.float64)
        c = torch.arange(p, device=device, dtype=torch.float64)
        aa, bb = torch.meshgrid(a, b, indexing="ij")
        w = 2.0 * math.pi * freq / p
        phase = w * (aa.unsqueeze(-1) + bb.unsqueeze(-1) - c.view(1, 1, p))
        cur = current.double()
        for basis_fn in (torch.cos(phase), torch.sin(phase)):
            denom = basis_fn.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
            coeff = (cur * basis_fn).sum(-1, keepdim=True) / denom
            cur = cur - coeff * basis_fn
        current = cur.to(logits_grid.dtype)
        per_freq.append(test_logits(logits_flat(current), labels, train_mask, test_mask, mode=mode))
    final = test_logits(logits_flat(current), labels, train_mask, test_mask, mode=mode)
    return final, per_freq


def calculate_trig_loss(
    logits_grid: torch.Tensor,
    key_freqs: Sequence[int],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    basis: torch.Tensor,
    *,
    mode: str = "test",
) -> float:
    trig = trig_component_sum(logits_grid, key_freqs, basis)
    flat_trig = logits_flat(trig)
    flat_orig = logits_flat(logits_grid)
    return test_logits(
        flat_trig,
        labels,
        train_mask,
        test_mask,
        mode=mode,
        bias_correction=True,
        original_logits=flat_orig,
    )


def embedding_fourier_norms(w_e: torch.Tensor, p: int) -> torch.Tensor:
    basis = fourier_basis(p, w_e.device)
    w = w_e[:p].T
    coeffs = fft1d(w, basis)
    return coeffs.pow(2).sum(0).sqrt()


def unembed_fourier_norms(w_u: torch.Tensor, p: int) -> torch.Tensor:
    """DFT along output/logit axis of lm_head [vocab, hidden] -> [hidden, vocab]."""
    basis = fourier_basis(p, w_u.device)
    w = w_u[:p].T  # [hidden, p]
    coeffs = fft1d(w, basis)
    return coeffs.pow(2).sum(0).sqrt()


def logits_fourier_norm_map(grid: torch.Tensor, p: int) -> torch.Tensor:
    """2D DFT energy map over (a,b) for each output logit, summed over c."""
    basis = fourier_basis(p, grid.device)
    m = grid.to(basis.dtype)
    fa = torch.einsum("abc,fx->fbc", m, basis)
    fab = torch.einsum("fbc,Fy->fFc", fa, basis)
    return fab.pow(2).sum(-1).sqrt()


def logits_fourier_norms(grid: torch.Tensor, p: int) -> torch.Tensor:
    """Per-frequency norm vector (length p) aggregated over 2D logit map."""
    fab = logits_fourier_norm_map(grid, p)
    return fab.sum(dim=1).sqrt()


def fft2d(mat: torch.Tensor, p: int, basis: torch.Tensor) -> torch.Tensor:
    """2D DFT of ``[p*p, ...]`` along the (a,b) axes (Nanda ``helpers.fft2d``)."""
    n_f = basis.shape[0]
    tail = mat.shape[1:]
    grid = mat.reshape(p, p, *tail).to(basis.dtype)
    fa = torch.einsum("xy...,fx->fy...", grid, basis)
    fab = torch.einsum("fy...,Fy->fF...", fa, basis)
    return fab.reshape(n_f * n_f, *tail)


def extract_freq_2d(tensor: torch.Tensor, freq: int, p: int) -> torch.Tensor:
    """Linear+quadratic 2D Fourier block at ``freq`` (cos/sin pair; DC omitted)."""
    if freq <= 0:
        tail = tensor.shape[2:]
        return torch.zeros(2, 2, *tail, device=tensor.device, dtype=tensor.dtype)
    ci = 2 * (freq - 1)
    si = ci + 1
    idx = [ci, si]
    return tensor[idx][:, idx]


def calculate_key_freqs_from_mlp_acts(neuron_acts: torch.Tensor, p: int) -> List[int]:
    """Nanda ``calculate_key_freqs``: unique dominant frequencies across MLP neurons.

    ``neuron_acts`` is ``[p*p, n_neurons]`` at the ``='' token, centered over the batch.
    """
    device = neuron_acts.device
    centered = neuron_acts - neuron_acts.mean(dim=0, keepdim=True)
    basis = fourier_basis(p, device)
    fourier_neuron = fft2d(centered, p, basis)
    n_f = basis.shape[0]
    fourier_square = fourier_neuron.reshape(n_f, n_f, -1)
    n_neurons = fourier_square.shape[-1]
    neuron_freqs: List[int] = []
    for ni in range(n_neurons):
        best_frac = -1.0
        best_freq = 1
        plane = fourier_square[:, :, ni]
        denom = plane.pow(2).sum().item()
        if denom < 1e-12:
            continue
        for freq in range(1, p // 2):
            numer = extract_freq_2d(plane, freq, p).pow(2).sum().item()
            frac = numer / denom
            if frac > best_frac:
                best_frac = frac
                best_freq = freq
        neuron_freqs.append(best_freq)
    return sorted(set(neuron_freqs))


def identify_key_frequencies(norm_vec: torch.Tensor, top_k: int = 5) -> List[int]:
    pair_energy = []
    n_pairs = len(norm_vec) // 2
    for k in range(n_pairs):
        e = norm_vec[2 * k].item() + norm_vec[2 * k + 1].item()
        pair_energy.append((e, k))
    pair_energy.sort(reverse=True)
    return [k for _, k in pair_energy[:top_k]]


def identify_key_frequencies_by_excluded(
    grid: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    p: int,
    top_k: int = 10,
) -> List[int]:
    """Model-independent key freqs: top k single-frequency excluded loss increases (Nanda footnote)."""
    device = grid.device
    basis = fourier_basis(p, device)
    full = test_logits(logits_flat(grid), labels, train_mask, test_mask, mode="test")
    scores: List[tuple[float, int]] = []
    for k in range(p // 2):
        excl, _ = calculate_excluded_loss(grid, [k], labels, train_mask, test_mask, basis, mode="test")
        scores.append((excl - full, k))
    scores.sort(reverse=True)
    return [k for _, k in scores[:top_k]]


def identify_key_frequencies_adaptive(
    grid: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    p: int,
    *,
    max_k: int = 12,
    tol: float = 0.01,
    target: float = 0.99,
) -> List[int]:
    """Data-driven key-frequency count, mirroring Nanda's methodology.

    Nanda's ``calculate_key_freqs`` sets the number of key frequencies from the
    *number of distinct frequencies the MLP neurons specialize in* -- it is not a
    fixed constant. A hard-coded ``top_k=5`` under-counts models whose circuit
    spreads over 6-8 frequencies (it reports e.g. FVE 0.83 when the true sparse
    reconstruction reaches 0.99). Here we rank frequencies by excluded-loss
    increase, then add them until the bias-corrected FVE plateaus (marginal gain
    ``< tol``) or reaches ``target``, capped at ``max_k``.
    """
    ranked = identify_key_frequencies_by_excluded(grid, labels, train_mask, test_mask, p, top_k=max_k)
    selected: List[int] = []
    prev = 0.0
    for freq in ranked:
        candidate = selected + [freq]
        fve = fit_trig_logits_fve_bias_corrected(grid, candidate, p)["fve_mean"]
        selected = candidate
        if fve >= target or (fve - prev) < tol:
            break
        prev = fve
    return selected


def gini_coefficient(values: torch.Tensor) -> float:
    x = values.flatten().float()
    x = x[x >= 0]
    if x.numel() == 0:
        return 0.0
    sorted_x, _ = torch.sort(x)
    n = sorted_x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return float((2 * (idx * sorted_x).sum() / (n * sorted_x.sum())) - (n + 1) / n)


def cos_xpymz_basis(p: int, freq: int, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Normalized cos(w*(a+b-c)) basis over flattened (a,b,c) with shape [p^3, p^3] projection helper."""
    a = torch.arange(p, device=device, dtype=dtype)[None, :, None, None]
    b = torch.arange(p, device=device, dtype=dtype)[None, None, :, None]
    c = torch.arange(p, device=device, dtype=dtype)[None, None, None, :]
    w = 2.0 * math.pi * freq / p
    vals = torch.cos(w * (a + b - c))
    flat = vals.reshape(p * p, p)
    flat = flat / flat.pow(2).sum().sqrt().clamp_min(1e-12)
    return flat


def fit_trig_logits_fve(logits_grid: torch.Tensor, key_freqs: Sequence[int], p: int) -> Dict[str, float]:
    """Fraction of variance explained by sum_k alpha_k cos(w_k(a+b-c))."""
    device = logits_grid.device
    flat = logits_flat(logits_grid).double()
    labels = torch.arange(p, device=device).repeat(p * p)
    # one-hot target directions for supervised fit: use actual argmax structure
    y = flat
    y_mean = y.mean(0, keepdim=True)
    y_centered = y - y_mean
    total_var = y_centered.pow(2).sum().item()
    basis_cols = []
    for k in key_freqs:
        basis_cols.append(cos_xpymz_basis(p, k, device).reshape(-1))
    if not basis_cols:
        return {"fve": 0.0, "residual_frac": 1.0}
    B = torch.stack(basis_cols, dim=1)  # [p*p*p?,] actually [p*p, p] per col - wrong

    # Build design matrix over (a,b) with target logit vector per pair
    design = []
    for k in key_freqs:
        comp = get_component_cos_xpy(logits_grid, k, fourier_basis(p, device)).reshape(-1)
        design.append(comp)
    X = torch.stack(design, dim=1).double()  # [p*p, p, n_freq] - wrong shape

    # Simpler: per output dimension c, fit logits[:,c] ~ sum alpha_k cos(w_k(a+b-c))
    fves = []
    for c in range(p):
        target = flat[:, c]
        cols = []
        for k in key_freqs:
            w = 2.0 * math.pi * k / p
            a = torch.arange(p, device=device, dtype=torch.float64)
            b = torch.arange(p, device=device, dtype=torch.float64)
            aa, bb = torch.meshgrid(a, b, indexing="ij")
            cols.append(torch.cos(w * (aa + bb - c)).reshape(-1))
        Xc = torch.stack(cols, dim=1)
        coef, _, _, _ = torch.linalg.lstsq(Xc, target.double())
        pred = Xc @ coef
        resid = target.double() - pred
        var = target.double().var(unbiased=False).item()
        fves.append(1.0 - resid.var(unbiased=False).item() / var if var > 1e-12 else 1.0)
    return {"fve_mean": float(np.mean(fves)), "fve_per_output_min": float(np.min(fves)), "fve_per_output_max": float(np.max(fves))}


def fit_trig_logits_fve_faithful(logits_grid: torch.Tensor, key_freqs: Sequence[int], p: int) -> Dict[str, float]:
    """FVE using trig_key_reconstruction subspace (cos+sin), aligned with restricted loss."""
    recon = trig_key_reconstruction(logits_grid, key_freqs)
    flat = logits_flat(logits_grid).double()
    recon_flat = logits_flat(recon).double()
    resid = flat - recon_flat
    var = flat.var(unbiased=False).item()
    fve = 1.0 - resid.var(unbiased=False).item() / var if var > 1e-12 else 1.0
    return {"fve_mean": float(fve), "fve_faithful": float(fve)}


def fit_trig_logits_fve_bias_corrected(
    logits_grid: torch.Tensor, key_freqs: Sequence[int], p: int
) -> Dict[str, float]:
    """Bias-corrected faithful FVE (matches Nanda ``bias_correction=True``).

    The cos/sin(w_k(a+b-c)) directions are mean-zero over the output axis c, so a
    grokked model's *constant* logit offset (the input-independent bias term that
    Nanda restores via bias correction) is structurally invisible to the plain
    reconstruction and is wrongly counted as residual. Here we remove the
    per-(a,b) constant-over-c offset from the logits before measuring variance
    explained, so FVE reflects the *input-dependent* trig structure only --
    the quantity Nanda reports as ~95%+ for the one-layer transformer.
    """
    grid = logits_grid.double()
    centered = grid - grid.mean(dim=-1, keepdim=True)
    recon = trig_key_reconstruction(centered, key_freqs).double()
    resid = centered - recon
    var = centered.var(unbiased=False).item()
    fve = 1.0 - resid.var(unbiased=False).item() / var if var > 1e-12 else 1.0
    return {"fve_mean": float(fve), "fve_bias_corrected": float(fve)}


def latent_fve_along_freq(z_flat: torch.Tensor, key_freqs: Sequence[int], p: int) -> Dict[str, float]:
    """R^2 between ||z|| at (a,b) and cos/sin(w(a+b)) scalars (latent readout probe)."""
    device = z_flat.device
    a = torch.arange(p, device=device, dtype=torch.float64)
    b = torch.arange(p, device=device, dtype=torch.float64)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    z_norm = z_flat.double().norm(dim=-1)
    results: Dict[str, float] = {}
    for k in key_freqs:
        w = 2.0 * math.pi * k / p
        for name, feat in (("cos", torch.cos(w * (aa + bb))), ("sin", torch.sin(w * (aa + bb)))):
            f = feat.reshape(-1)
            f = f - f.mean()
            z_c = z_norm - z_norm.mean()
            denom = (f.pow(2).sum() * z_c.pow(2).sum()).clamp_min(1e-12)
            r2 = float((f @ z_c).pow(2).item() / denom.item())
            results[f"latent_fve_k{k}_{name}"] = r2
    results["latent_fve_mean"] = float(np.mean(list(results.values()))) if results else 0.0
    return results


def progress_measure_bundle(
    logits_grid: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    key_freqs: Sequence[int],
    w_e: torch.Tensor,
    w_u: torch.Tensor,
    *,
    mlp_neuron_acts: torch.Tensor | None = None,
) -> Dict[str, float | List[float] | Dict[str, float]]:
    p = logits_grid.shape[0]
    device = logits_grid.device
    basis = fourier_basis(p, device)
    emb_norms = embedding_fourier_norms(w_e, p)
    un_norms = unembed_fourier_norms(w_u, p)
    log_norms = logits_fourier_norm_map(logits_grid, p)
    key_freqs_emb = identify_key_frequencies(emb_norms, top_k=5)
    key_freqs_excl = identify_key_frequencies_by_excluded(
        logits_grid, labels, train_mask, test_mask, p, top_k=5
    )
    key_freqs_adaptive = identify_key_frequencies_adaptive(
        logits_grid, labels, train_mask, test_mask, p
    )
    key_freqs_neuron: List[int] = []
    if mlp_neuron_acts is not None:
        key_freqs_neuron = calculate_key_freqs_from_mlp_acts(mlp_neuron_acts, p)
    key_freqs = key_freqs_excl

    excluded_test, excluded_per = calculate_excluded_loss(
        logits_grid, key_freqs, labels, train_mask, test_mask, basis, mode="test"
    )
    excluded_train, _ = calculate_excluded_loss(
        logits_grid, key_freqs, labels, train_mask, test_mask, basis, mode="train"
    )
    trig_test = calculate_trig_loss(logits_grid, key_freqs, labels, train_mask, test_mask, basis, mode="test")
    trig_train = calculate_trig_loss(logits_grid, key_freqs, labels, train_mask, test_mask, basis, mode="train")
    full_test = test_logits(logits_flat(logits_grid), labels, train_mask, test_mask, mode="test")
    full_train = test_logits(logits_flat(logits_grid), labels, train_mask, test_mask, mode="train")

    return {
        "key_frequencies": list(key_freqs),
        "key_frequencies_embedding": list(key_freqs_emb),
        "key_frequencies_excluded": list(key_freqs_excl),
        "full_loss_train": full_train,
        "full_loss_test": full_test,
        "excluded_loss_train": excluded_train,
        "excluded_loss_test": excluded_test,
        "excluded_loss_test_per_freq": excluded_per,
        "trig_loss_train": trig_train,
        "trig_loss_test": trig_test,
        "embedding_fourier_norms": emb_norms.cpu().tolist(),
        "unembed_fourier_norms": un_norms.cpu().tolist(),
        "logits_fourier_norms": log_norms.cpu().tolist(),
        "embedding_gini": gini_coefficient(emb_norms),
        "unembed_gini": gini_coefficient(un_norms),
        "logits_fourier_gini": gini_coefficient(log_norms),
        "logit_trig_fve": fit_trig_logits_fve(logits_grid, key_freqs, p),
        "logit_trig_fve_faithful": fit_trig_logits_fve_faithful(logits_grid, key_freqs, p),
        "logit_trig_fve_bias_corrected": fit_trig_logits_fve_bias_corrected(logits_grid, key_freqs, p),
        "logit_trig_fve_adaptive": fit_trig_logits_fve_bias_corrected(logits_grid, key_freqs_adaptive, p),
        "key_frequencies_adaptive": list(key_freqs_adaptive),
        "n_key_frequencies_adaptive": len(key_freqs_adaptive),
        "key_frequencies_neuron": list(key_freqs_neuron),
        "n_key_frequencies_neuron": len(key_freqs_neuron),
        "logit_trig_fve_neuron_keys": fit_trig_logits_fve_bias_corrected(logits_grid, key_freqs_neuron, p)
        if key_freqs_neuron
        else {"fve_mean": 0.0, "fve_bias_corrected": 0.0},
    }
