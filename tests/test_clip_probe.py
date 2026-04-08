"""Tests for the CLIP linear probe training script."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression


def test_train_probe_returns_fitted_model():
    """train_probe() returns a fitted LogisticRegression given embeddings + labels."""
    from model.train_clip_probe import train_probe

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, 512)).astype(np.float32)
    labels = np.array([1, 0] * 10)

    probe = train_probe(embeddings, labels)

    assert isinstance(probe, LogisticRegression)
    assert hasattr(probe, "coef_"), "Probe must be fitted"
    preds = probe.predict(embeddings)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_train_probe_raises_on_single_class():
    """train_probe() raises ValueError when all labels are the same class."""
    from model.train_clip_probe import train_probe

    embeddings = np.ones((10, 512), dtype=np.float32)
    labels = np.ones(10, dtype=int)

    with pytest.raises(ValueError, match="at least two classes"):
        train_probe(embeddings, labels)


def test_aggregate_probe_scores_to_hex():
    """aggregate_to_hex() returns mean clip_risk_prob per h3_index."""
    from model.train_clip_probe import aggregate_to_hex
    import pandas as pd

    rows = pd.DataFrame({
        "h3_index": ["aaa", "aaa", "bbb"],
        "clip_risk_prob": [0.8, 0.6, 0.3],
    })

    result = aggregate_to_hex(rows)

    assert set(result.columns) == {"h3_index", "clip_risk_prob"}
    assert result.set_index("h3_index").loc["aaa", "clip_risk_prob"] == pytest.approx(0.7)
    assert result.set_index("h3_index").loc["bbb", "clip_risk_prob"] == pytest.approx(0.3)
