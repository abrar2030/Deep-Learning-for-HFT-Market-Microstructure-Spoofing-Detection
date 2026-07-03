"""
Real LOB data loading for TEN-GNN training.

Supports three file formats:

* ``.npz``      - pre-windowed arrays: sequences (N, T, F), labels (N,),
                  and optionally time_deltas (N, T, 1).
* ``.csv``      - raw event-level LOB snapshots (long format, see schema).
* ``.parquet``  - same schema as CSV.

CSV / Parquet schema
--------------------
Required columns::

    timestamp   float   event time (ms or s, monotonically increasing)
    best_bid    float   best bid price
    best_ask    float   best ask price
    bid_volume  float   volume at best bid
    ask_volume  float   volume at best ask

Optional columns::

    label       int     1 = spoofing window, 0 = clean (per-row; the label of
                        a window is the max over its rows). If absent, all
                        windows are labelled 0 (inference-style loading).
    bid_price_{i}, ask_price_{i}, bid_volume_{i}, ask_volume_{i}
                        deeper book levels, i in [2, num_levels]

Rows are windowed into overlapping sequences of ``window_size`` events with a
stride of ``window_size // 2``. Per-window features are engineered from the
snapshot columns and padded / truncated to ``input_dim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = ("timestamp", "best_bid", "best_ask", "bid_volume", "ask_volume")


def load_real_data(
    data_path: str,
    window_size: int = 100,
    input_dim: int = 47,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real LOB data and return model-ready arrays.

    Args:
        data_path: Path to a .npz, .csv, or .parquet file.
        window_size: Sequence length in events.
        input_dim: Feature dimension expected by the model.

    Returns:
        Tuple of (sequences, labels, time_deltas) with shapes
        (N, window_size, input_dim), (N,), (N, window_size, 1).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the format is unsupported or the schema is invalid.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path, window_size, input_dim)
    if suffix == ".csv":
        df = pd.read_csv(path)
        return _windows_from_dataframe(df, window_size, input_dim)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return _windows_from_dataframe(df, window_size, input_dim)

    raise ValueError(
        f"Unsupported data format '{suffix}'. Use .npz, .csv, or .parquet."
    )


# ---------------------------------------------------------------------------
# NPZ (pre-windowed) loader
# ---------------------------------------------------------------------------


def _load_npz(
    path: Path, window_size: int, input_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)

    if "sequences" not in data or "labels" not in data:
        raise ValueError(
            "NPZ file must contain 'sequences' and 'labels' arrays "
            f"(found: {sorted(data.files)})."
        )

    sequences = np.asarray(data["sequences"], dtype=np.float32)
    labels = np.asarray(data["labels"]).astype(np.int64).reshape(-1)

    if sequences.ndim != 3:
        raise ValueError(
            f"'sequences' must be 3-D (N, T, F); got shape {sequences.shape}."
        )
    if sequences.shape[0] != labels.shape[0]:
        raise ValueError(
            f"sequences ({sequences.shape[0]}) and labels ({labels.shape[0]}) "
            "have mismatched sample counts."
        )

    sequences = _fit_time_axis(sequences, window_size)
    sequences = _fit_feature_axis(sequences, input_dim)

    if "time_deltas" in data:
        time_deltas = np.asarray(data["time_deltas"], dtype=np.float32)
        if time_deltas.ndim == 2:
            time_deltas = time_deltas[..., np.newaxis]
        time_deltas = _fit_time_axis(time_deltas, window_size)
    else:
        time_deltas = np.ones((sequences.shape[0], window_size, 1), dtype=np.float32)

    return sequences, labels, time_deltas


# ---------------------------------------------------------------------------
# CSV / Parquet (event-level) loader
# ---------------------------------------------------------------------------


def _windows_from_dataframe(
    df: pd.DataFrame, window_size: int, input_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Required schema: {list(_REQUIRED_COLUMNS)}."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < window_size:
        raise ValueError(
            f"Need at least window_size={window_size} rows; got {len(df)}."
        )

    features = _engineer_features(df, input_dim)  # (n_rows, input_dim)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    has_labels = "label" in df.columns
    row_labels = (
        df["label"].to_numpy(dtype=np.int64)
        if has_labels
        else np.zeros(len(df), dtype=np.int64)
    )

    stride = max(window_size // 2, 1)
    starts = range(0, len(df) - window_size + 1, stride)

    sequences, labels, time_deltas = [], [], []
    for s in starts:
        e = s + window_size
        sequences.append(features[s:e])
        labels.append(int(row_labels[s:e].max()))

        dt = np.diff(timestamps[s:e], prepend=timestamps[s])
        # Guard against zero / negative deltas from duplicated timestamps.
        dt = np.clip(dt, 1e-6, None).astype(np.float32)
        time_deltas.append(dt[:, np.newaxis])

    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.stack(time_deltas).astype(np.float32),
    )


def _engineer_features(df: pd.DataFrame, input_dim: int) -> np.ndarray:
    """Build a per-row feature matrix from LOB snapshot columns."""
    eps = 1e-8
    best_bid = df["best_bid"].to_numpy(dtype=np.float64)
    best_ask = df["best_ask"].to_numpy(dtype=np.float64)
    bid_vol = df["bid_volume"].to_numpy(dtype=np.float64)
    ask_vol = df["ask_volume"].to_numpy(dtype=np.float64)

    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
    weighted_mid = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol + eps)

    mid_returns = np.diff(mid, prepend=mid[0]) / (mid + eps)
    log_bid_vol = np.log1p(np.abs(bid_vol))
    log_ask_vol = np.log1p(np.abs(ask_vol))

    columns = [
        mid_returns,
        spread / (mid + eps),
        imbalance,
        (weighted_mid - mid) / (mid + eps),
        log_bid_vol,
        log_ask_vol,
    ]

    # Any deeper book levels present in the file are appended in order.
    level = 2
    while True:
        cols = [
            f"bid_price_{level}",
            f"ask_price_{level}",
            f"bid_volume_{level}",
            f"ask_volume_{level}",
        ]
        if not all(c in df.columns for c in cols):
            break
        bp = df[cols[0]].to_numpy(dtype=np.float64)
        ap = df[cols[1]].to_numpy(dtype=np.float64)
        bv = df[cols[2]].to_numpy(dtype=np.float64)
        av = df[cols[3]].to_numpy(dtype=np.float64)
        columns.append((bp - mid) / (mid + eps))
        columns.append((ap - mid) / (mid + eps))
        columns.append(np.log1p(np.abs(bv)))
        columns.append(np.log1p(np.abs(av)))
        level += 1

    features = np.stack(columns, axis=1)

    # Z-score normalisation per feature column.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + eps
    features = (features - mean) / std

    return _fit_feature_axis(features[np.newaxis, ...], input_dim)[0]


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _fit_time_axis(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Pad (with zeros) or truncate axis 1 to exactly window_size."""
    t = arr.shape[1]
    if t == window_size:
        return arr
    if t > window_size:
        return arr[:, :window_size]
    pad = np.zeros((arr.shape[0], window_size - t, *arr.shape[2:]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _fit_feature_axis(arr: np.ndarray, input_dim: int) -> np.ndarray:
    """Pad (with zeros) or truncate the last axis to exactly input_dim."""
    f = arr.shape[-1]
    if f == input_dim:
        return arr
    if f > input_dim:
        return arr[..., :input_dim]
    pad_shape = (*arr.shape[:-1], input_dim - f)
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=-1)
