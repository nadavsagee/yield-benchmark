"""Unsupervised pre-step — Isolation Forest + PCA over per-lot feature vectors."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from schema import INLINE_METRICS, INLINE_STEPS, WAT_PARAMS

from agent.tools._baseline import EDGE_RADIUS, CENTER_RADIUS

# Strong flag when robust z-score of isolation raw score exceeds this.
# Calibrated: clean datasets peak ~5; injected excursions ~8+.
STRONG_ANOMALY_THRESHOLD = 6.0
TOP_FEATURES_K = 5
PCA_COMPONENTS = 3

# Sort-only excursions (correlation_break) need visible sort signal in feature space.
SORT_FEATURE_WEIGHT = 2.5

# Trailing window (lots) for inline temporal drift features.
TREND_WINDOW = 4

# Prior lots used for inline rolling-baseline overlay residual composite.
INLINE_RESIDUAL_LOOKBACK = 10


def run_prestep(
    tables: dict[str, pd.DataFrame],
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Flag suspect lots via Isolation Forest on per-lot features (no ground truth).

    Features: WAT mean / edge / radial-slope, inline step means, inline temporal
    trend slopes, sort spatial aggregates (sort features up-weighted). Returns
    ranked suspects, scores, PCA view, and top drivers.
    """
    feature_df = build_lot_features(tables)
    lot_ids = feature_df.index.tolist()
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=float)
    lots = _ordered_lots(tables["sort"])
    inline_means = _inline_lot_means(tables["inline"], lots)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = float(np.clip(1.0 / len(lot_ids), 0.02, 0.08))
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    iso.fit(X_scaled)
    raw = -iso.score_samples(X_scaled)
    if_scores = _normalize_scores(raw)
    drift_scores = _overlay_drift_scores(X, feature_names, lots, inline_means)
    anomaly_scores = np.maximum(if_scores, drift_scores)

    n_pca = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    suspects = []
    order = np.argsort(-anomaly_scores)
    for rank, idx in enumerate(order, start=1):
        lot = lot_ids[idx]
        suspects.append(
            {
                "lot": lot,
                "rank": rank,
                "anomaly_score": round(float(anomaly_scores[idx]), 4),
                "isolation_raw": round(float(raw[idx]), 4),
                "is_anomaly": bool(iso.predict(X_scaled[idx : idx + 1])[0] == -1),
                "top_features": _top_driving_features(
                    X, idx, feature_names, top_k=TOP_FEATURES_K
                ),
                "pca": {
                    f"pc{i + 1}": round(float(pca_coords[idx, i]), 4)
                    for i in range(n_pca)
                },
            }
        )

    return {
        "n_lots": len(lot_ids),
        "feature_count": len(feature_names),
        "contamination": contamination,
        "strong_threshold": STRONG_ANOMALY_THRESHOLD,
        "suspects": suspects,
        "pca": {
            "n_components": n_pca,
            "explained_variance_ratio": [
                round(float(v), 4) for v in pca.explained_variance_ratio_
            ],
            "top_loadings": _pca_top_loadings(pca, feature_names, pc=0),
        },
    }


def _lot_order(lot: str) -> int:
    return int(lot.split("_")[1])


def _ordered_lots(sort_df: pd.DataFrame) -> list[str]:
    return sorted(sort_df["lot"].unique(), key=_lot_order)


def _inline_lot_means(inline_df: pd.DataFrame, lots: list[str]) -> dict[tuple[str, str, str], float]:
    means: dict[tuple[str, str, str], float] = {}
    for lot in lots:
        lot_df = inline_df[inline_df["lot"] == lot]
        for step in INLINE_STEPS:
            step_df = lot_df[lot_df["step"] == step]
            for metric in INLINE_METRICS:
                means[(lot, step, metric)] = (
                    float(step_df[metric].mean()) if len(step_df) else 0.0
                )
    return means


def build_lot_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per lot; columns are engineered aggregates + temporal trends."""
    sort_df = tables["sort"]
    wat_df = tables["wat"]
    inline_df = tables["inline"]

    lots = _ordered_lots(sort_df)
    inline_means = _inline_lot_means(inline_df, lots)

    rows: list[dict[str, float]] = []
    for lot_index, lot in enumerate(lots):
        row: dict[str, float] = {}
        row.update(_sort_features(sort_df, lot, weight=SORT_FEATURE_WEIGHT))
        row.update(_wat_features(wat_df, lot))
        row.update(_inline_features(inline_df, lot))
        row.update(_inline_trend_features(lots, lot_index, inline_means))
        rows.append(row)

    return pd.DataFrame(rows, index=lots).fillna(0.0)


def _sort_features(
    sort_df: pd.DataFrame,
    lot: str,
    *,
    weight: float = 1.0,
) -> dict[str, float]:
    lot_df = sort_df[sort_df["lot"] == lot]
    edge = lot_df[lot_df["radius"] >= EDGE_RADIUS]
    center = lot_df[lot_df["radius"] <= CENTER_RADIUS]
    wafer_yield = lot_df.groupby("wafer")["pass"].mean()
    fail_ind = (~lot_df["pass"]).astype(float)

    gross_rate = float((lot_df["soft_bin"] == "gross").mean())
    leakage_rate = float((lot_df["soft_bin"] == "leakage").mean())

    feats = {
        "sort_yield": float(lot_df["pass"].mean()),
        "sort_fail_rate": float((~lot_df["pass"]).mean()),
        "sort_edge_fail_rate": float((~edge["pass"]).mean()) if len(edge) else 0.0,
        "sort_center_fail_rate": float((~center["pass"]).mean()) if len(center) else 0.0,
        "sort_fail_radial_slope": _radial_slope(lot_df["radius"], fail_ind),
        "sort_speed_rate": float((lot_df["soft_bin"] == "speed").mean()),
        "sort_gross_rate": gross_rate,
        "sort_leakage_rate": leakage_rate,
        "sort_wafer_yield_std": float(wafer_yield.std(ddof=0)) if len(wafer_yield) > 1 else 0.0,
        "sort_fail_only_signal": float((~lot_df["pass"]).mean()),
    }
    if weight != 1.0:
        return {k: v * weight for k, v in feats.items()}
    return feats


def _wat_features(wat_df: pd.DataFrame, lot: str) -> dict[str, float]:
    lot_df = wat_df[wat_df["lot"] == lot]
    features: dict[str, float] = {}
    for param in WAT_PARAMS:
        prefix = f"wat_{param}"
        features[f"{prefix}_mean"] = float(lot_df[param].mean())
        edge = lot_df[lot_df["radius"] >= EDGE_RADIUS][param]
        features[f"{prefix}_edge"] = float(edge.mean()) if len(edge) else features[f"{prefix}_mean"]
        features[f"{prefix}_radial_slope"] = _radial_slope(lot_df["radius"], lot_df[param])
    return features


def _inline_features(inline_df: pd.DataFrame, lot: str) -> dict[str, float]:
    lot_df = inline_df[inline_df["lot"] == lot]
    features: dict[str, float] = {}
    for step in INLINE_STEPS:
        step_df = lot_df[lot_df["step"] == step]
        for metric in INLINE_METRICS:
            key = f"inline_{step}_{metric}_mean"
            features[key] = float(step_df[metric].mean()) if len(step_df) else 0.0
    return features


def _inline_trend_features(
    lots: list[str],
    lot_index: int,
    inline_means: dict[tuple[str, str, str], float],
) -> dict[str, float]:
    """
    Trailing-window slope / sustained drift for each inline metric.

    Captures slow multi-lot inline drift (early_detection) even when no single
    lot snapshot is an outlier.
    """
    features: dict[str, float] = {}
    start = max(0, lot_index - TREND_WINDOW + 1)
    window_lots = lots[start : lot_index + 1]

    sustained_vals: list[float] = []
    slope_vals: list[float] = []

    for step in INLINE_STEPS:
        for metric in INLINE_METRICS:
            slope_key = f"trend_{step}_{metric}_slope"
            sust_key = f"trend_{step}_{metric}_sustained"

            if len(window_lots) < 3:
                features[slope_key] = 0.0
                features[sust_key] = 0.0
                continue

            y = np.array(
                [inline_means[(lot, step, metric)] for lot in window_lots],
                dtype=float,
            )
            x = np.arange(len(window_lots), dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            features[slope_key] = slope
            slope_vals.append(abs(slope))

            diffs = np.diff(y)
            if len(diffs) == 0:
                mono = 0.0
            elif slope >= 0:
                mono = float(np.mean(diffs >= 0))
            else:
                mono = float(np.mean(diffs <= 0))
            sustained = abs(slope) * mono * len(window_lots)
            features[sust_key] = sustained
            sustained_vals.append(sustained)

    features["trend_max_slope"] = max(slope_vals) if slope_vals else 0.0
    features["trend_max_sustained"] = max(sustained_vals) if sustained_vals else 0.0
    return features


def _radial_slope(radius: pd.Series, values: pd.Series) -> float:
    if len(radius) < 3:
        return 0.0
    r = radius.to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    if np.allclose(r.std(), 0.0):
        return 0.0
    return float(np.polyfit(r, v, 1)[0])


def _normalize_scores(raw: np.ndarray) -> np.ndarray:
    """Robust z-scores — comparable across datasets, no forced max at 1.0."""
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))
    sigma = 1.4826 * mad if mad > 1e-12 else float(np.std(raw))
    if sigma < 1e-12:
        return np.zeros_like(raw)
    return (raw - med) / sigma


_TREND_PAIR_RE = re.compile(
    r"^trend_(?P<step>.+)_(?P<metric>.+)_(?P<kind>slope|sustained)$"
)


def _robust_z(col: np.ndarray, val: float) -> float:
    med = float(np.median(col))
    mad = float(np.median(np.abs(col - med)))
    sigma = 1.4826 * mad if mad > 1e-12 else 1.0
    return abs((val - med) / sigma)


def _residual_overlay_values(
    lots: list[str],
    inline_means: dict[tuple[str, str, str], float],
) -> np.ndarray:
    values = np.zeros(len(lots))
    for lot_index, lot in enumerate(lots):
        overlay_vals: list[float] = []
        for step in INLINE_STEPS:
            current = inline_means[(lot, step, "overlay_nm")]
            start = max(0, lot_index - INLINE_RESIDUAL_LOOKBACK)
            if start == lot_index:
                continue
            prior = [
                inline_means[(lots[j], step, "overlay_nm")]
                for j in range(start, lot_index)
            ]
            overlay_vals.append(abs(current - float(np.median(prior))))
        values[lot_index] = max(overlay_vals) if overlay_vals else 0.0
    return values


def _overlay_drift_scores(
    X: np.ndarray,
    feature_names: list[str],
    lots: list[str],
    inline_means: dict[tuple[str, str, str], float],
) -> np.ndarray:
    """
    Pair sustained + slope overlay trends; require both to be elevated.

    Catches slow multi-lot inline drift (early_detection) while ignoring
    spurious single-feature noise on clean lots.
    """
    pairs: dict[tuple[str, str], dict[str, int]] = {}
    for j, name in enumerate(feature_names):
        match = _TREND_PAIR_RE.match(name)
        if not match or match.group("metric") != "overlay_nm":
            continue
        key = (match.group("step"), match.group("metric"))
        pairs.setdefault(key, {})[match.group("kind")] = j

    residual_vals = _residual_overlay_values(lots, inline_means)

    scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        best = 0.0
        for idxs in pairs.values():
            if "slope" not in idxs or "sustained" not in idxs:
                continue
            z_slope = _robust_z(X[:, idxs["slope"]], float(X[i, idxs["slope"]]))
            z_sust = _robust_z(X[:, idxs["sustained"]], float(X[i, idxs["sustained"]]))
            best = max(best, min(z_slope, z_sust))
        best = max(best, _robust_z(residual_vals, float(residual_vals[i])))
        scores[i] = best
    return scores


def _top_driving_features(
    X: np.ndarray,
    row_idx: int,
    feature_names: list[str],
    top_k: int = TOP_FEATURES_K,
) -> list[dict[str, Any]]:
    row = X[row_idx]
    scored: list[tuple[str, float]] = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med)))
        sigma = 1.4826 * mad if mad > 1e-12 else 1.0
        z = (float(row[j]) - med) / sigma
        scored.append((name, z))
    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    return [
        {
            "feature": name,
            "z_score": round(z, 3),
            "direction": "high" if z > 0 else "low",
        }
        for name, z in scored[:top_k]
    ]


def _pca_top_loadings(
    pca: PCA,
    feature_names: list[str],
    pc: int = 0,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    loadings = [
        (feature_names[i], float(pca.components_[pc, i]))
        for i in range(len(feature_names))
    ]
    loadings.sort(key=lambda t: abs(t[1]), reverse=True)
    return [
        {"feature": name, "loading": round(val, 4)}
        for name, val in loadings[:top_k]
    ]
