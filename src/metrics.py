"""Métriques, seuil de validation et agrégation des résultats."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choisit sur validation le seuil maximisant le F1 illicite."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates]
    # En cas d'égalité, retient le seuil le plus élevé pour limiter les faux positifs.
    best = max(score for score in scores)
    return float(max(threshold for threshold, score in zip(candidates, scores) if score == best))


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calcule les métriques positives pour les seuls nœuds étiquetés fournis."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= threshold
    return {
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "rappel": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def f1_by_time_step(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    time_steps: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Retourne le F1 et le support illicite par pas de temps étiqueté."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    time_steps = np.asarray(time_steps)
    selected = (y_true >= 0) if mask is None else np.asarray(mask, dtype=bool)
    rows = []
    for step in range(35, 50):
        current = selected & (time_steps == step)
        labels = y_true[current]
        rows.append(
            {
                "time_step": step,
                "f1": float(f1_score(labels, probabilities[current] >= threshold, zero_division=0))
                if len(labels)
                else float("nan"),
                "support_illicite": int(np.sum(labels == 1)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(results: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Agrège les métriques répétées en moyenne et écart-type (population)."""
    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    metric_columns = [name for name in ("precision", "rappel", "f1", "pr_auc") if name in frame]
    group_columns = [name for name in ("modele", "partition") if name in frame]
    if not group_columns:
        group_columns = ["modele"] if "modele" in frame else []
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns]
    summary = grouped.agg(["mean", "std"]).fillna(0.0)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
