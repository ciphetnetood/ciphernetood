from __future__ import annotations

import torch

from ciphernet_ood.ood.gram import gram_vectors


def test_gram_vectors_upper_triangle():
    hidden = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1])
    vectors = gram_vectors(hidden, batch)
    assert vectors.shape == (2, 10)
    assert torch.isfinite(vectors).all()
