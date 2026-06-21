import math

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.paper_v2.head_trig_fit import fit_head_trig_poly


def test_fit_head_trig_poly_synthetic():
    p = 113
    k = 14
    w = 2 * math.pi * k / p
    a = torch.arange(p).repeat_interleave(p).double()
    b = torch.arange(p).repeat(p).double()
    y = torch.cos(w * (a + b)) + 0.1 * torch.sin(w * (a + b))
    row = fit_head_trig_poly(y, a, b, p)
    assert row["dominant_frequency"] == k
    assert row["fve_pct"] >= 85.0
