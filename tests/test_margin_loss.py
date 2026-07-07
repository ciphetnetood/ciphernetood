from __future__ import annotations

import torch

from ciphernet_ood.losses.traffic_margin import traffic_conditioned_margin


def test_traffic_conditioned_margin_shape_and_bounds():
    config = {
        "cfml": {
            "base_margin": 0.2,
            "beta": 0.75,
            "gamma": 0.76,
            "eta": 0.85,
            "margin_min": 0.05,
            "margin_max": 0.80,
        }
    }
    freq = torch.tensor([1.0, 0.25])
    dispersion = torch.zeros(2, 3)
    confusability = torch.zeros(2)
    margin = traffic_conditioned_margin(freq, dispersion, confusability, config)
    assert margin.shape == (2, 3)
    assert torch.all(margin >= 0.05)
    assert torch.all(margin <= 0.80)
    assert torch.all(margin[1] >= margin[0])
