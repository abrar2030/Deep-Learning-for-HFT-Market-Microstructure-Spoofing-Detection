"""Unit tests for utils.data_loading (real LOB data loading)."""

import numpy as np
import pandas as pd
import pytest
from utils.data_loading import load_real_data

WINDOW = 20
DIM = 12


def _make_lob_frame(n_rows: int = 100, with_labels: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n_rows))
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=float),
            "best_bid": mid - 0.02,
            "best_ask": mid + 0.02,
            "bid_volume": rng.integers(100, 1000, n_rows).astype(float),
            "ask_volume": rng.integers(100, 1000, n_rows).astype(float),
        }
    )
    if with_labels:
        labels = np.zeros(n_rows, dtype=int)
        labels[40:60] = 1  # one spoofing burst
        df["label"] = labels
    return df


def test_csv_loading(tmp_path):
    path = tmp_path / "lob.csv"
    _make_lob_frame().to_csv(path, index=False)

    seq, labels, dt = load_real_data(str(path), window_size=WINDOW, input_dim=DIM)

    assert seq.ndim == 3 and seq.shape[1:] == (WINDOW, DIM)
    assert labels.shape == (seq.shape[0],)
    assert dt.shape == (seq.shape[0], WINDOW, 1)
    assert set(np.unique(labels)).issubset({0, 1})
    assert (labels == 1).any(), "spoofing burst should label some windows 1"
    assert np.all(dt > 0), "time deltas must be strictly positive"


def test_csv_without_labels(tmp_path):
    path = tmp_path / "lob_nolabel.csv"
    _make_lob_frame(with_labels=False).to_csv(path, index=False)
    _, labels, _ = load_real_data(str(path), window_size=WINDOW, input_dim=DIM)
    assert (labels == 0).all()


def test_npz_loading(tmp_path):
    n, t, f = 8, WINDOW, DIM
    rng = np.random.default_rng(1)
    path = tmp_path / "windows.npz"
    np.savez(
        path,
        sequences=rng.normal(size=(n, t, f)).astype(np.float32),
        labels=rng.integers(0, 2, n),
        time_deltas=np.ones((n, t), dtype=np.float32),
    )
    seq, labels, dt = load_real_data(str(path), window_size=t, input_dim=f)
    assert seq.shape == (n, t, f)
    assert dt.shape == (n, t, 1)


def test_npz_shape_adaptation(tmp_path):
    """Sequences with different T / F are padded or truncated to fit."""
    path = tmp_path / "odd.npz"
    np.savez(
        path,
        sequences=np.ones((4, WINDOW + 5, DIM - 4), dtype=np.float32),
        labels=np.zeros(4, dtype=np.int64),
    )
    seq, _, dt = load_real_data(str(path), window_size=WINDOW, input_dim=DIM)
    assert seq.shape == (4, WINDOW, DIM)
    assert dt.shape == (4, WINDOW, 1)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_real_data("/nonexistent/lob.csv")


def test_bad_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required column"):
        load_real_data(str(path))


def test_unsupported_format(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not lob data")
    with pytest.raises(ValueError, match="Unsupported data format"):
        load_real_data(str(path))


def test_too_few_rows(tmp_path):
    path = tmp_path / "short.csv"
    _make_lob_frame(n_rows=5).to_csv(path, index=False)
    with pytest.raises(ValueError, match="window_size"):
        load_real_data(str(path), window_size=WINDOW, input_dim=DIM)
