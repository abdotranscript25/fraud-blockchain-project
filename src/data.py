"""Chargement et préparation commune des données Elliptic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected


@dataclass
class EllipticData:
    """Données alignées par transaction, masques et graphe PyG."""

    x: np.ndarray
    y: np.ndarray
    time_steps: np.ndarray
    tx_ids: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    labeled_mask: np.ndarray
    edge_index: torch.Tensor
    scaler: StandardScaler
    features: pd.DataFrame
    edges: pd.DataFrame
    classes: pd.DataFrame


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    """Détermine le dossier des CSV à partir du dossier courant ou du dépôt."""
    if data_dir is not None:
        directory = Path(data_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Dossier de données introuvable : {directory}")
        return directory

    candidats = [Path.cwd() / "../data/raw", Path.cwd() / "data/raw"]
    candidats.extend([Path(__file__).resolve().parents[1] / "data/raw"])
    for candidat in candidats:
        directory = candidat.resolve()
        if all((directory / nom).is_file() for nom in _CSV_NAMES):
            return directory
    raise FileNotFoundError("CSV Elliptic introuvables dans ../data/raw ou data/raw.")


_CSV_NAMES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)


def load_data(data_dir: str | Path | None = None) -> EllipticData:
    """Charge les CSV, prépare les masques temporels, le scaler et edge_index."""
    directory = _resolve_data_dir(data_dir)
    features = pd.read_csv(directory / _CSV_NAMES[0], header=None)
    if features.shape[1] != 167:
        raise ValueError(f"167 colonnes attendues dans features, reçu : {features.shape[1]}")
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]
    edges = pd.read_csv(directory / _CSV_NAMES[1])
    classes = pd.read_csv(directory / _CSV_NAMES[2], dtype={"class": str})

    # L'ordre des lignes features définit l'index des nœuds dans tout le protocole.
    joined = features.merge(classes, on="txId", how="left", validate="one_to_one")
    raw_class = joined["class"].fillna("unknown").astype(str)
    y = np.full(len(joined), -1, dtype=np.int64)
    y[raw_class.to_numpy() == "1"] = 1
    y[raw_class.to_numpy() == "2"] = 0

    time_steps = joined["time_step"].to_numpy(dtype=np.int16)
    labeled = y >= 0
    train = (time_steps >= 1) & (time_steps <= 24) & labeled
    val = (time_steps >= 25) & (time_steps <= 34) & labeled
    test = (time_steps >= 35) & (time_steps <= 49) & labeled

    feature_columns = ["time_step"] + [f"f{i}" for i in range(1, 166)]
    x_raw = joined[feature_columns].to_numpy(dtype=np.float32, copy=True)
    # Le scaler est ajusté sur tous les nœuds des pas d'entraînement, étiquetés ou non.
    scaler_fit_mask = (time_steps >= 1) & (time_steps <= 24)
    scaler = StandardScaler()
    scaler.fit(x_raw[scaler_fit_mask])
    x = scaler.transform(x_raw).astype(np.float32, copy=False)

    node_index = pd.Series(np.arange(len(features), dtype=np.int64), index=features["txId"])
    source = edges["txId1"].map(node_index)
    target = edges["txId2"].map(node_index)
    if source.isna().any() or target.isna().any():
        raise ValueError("Certaines extrémités d'arêtes ne figurent pas dans les features.")
    edge_array = np.vstack([source.to_numpy(dtype=np.int64), target.to_numpy(dtype=np.int64)])
    edge_index = to_undirected(torch.from_numpy(edge_array).long(), num_nodes=len(features))

    return EllipticData(
        x=x,
        y=y,
        time_steps=time_steps,
        tx_ids=joined["txId"].to_numpy(),
        train_mask=train,
        val_mask=val,
        test_mask=test,
        labeled_mask=labeled,
        edge_index=edge_index,
        scaler=scaler,
        features=features,
        edges=edges,
        classes=classes,
    )
