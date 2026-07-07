from __future__ import annotations

from ciphernet_ood.evaluation.metrics import classification_metrics, imbalance_metrics, ood_metrics


def test_metrics_return_expected_keys():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    cls = classification_metrics(y_true, y_pred)
    imb = imbalance_metrics(y_true, y_pred)
    ood = ood_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert "macro_f1" in cls
    assert "f1_gap" in imb
    assert "auroc" in ood
