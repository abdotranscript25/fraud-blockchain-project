"""Implémentation des six modèles du protocole commun."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier

from .data import EllipticData
from .metrics import evaluate_probabilities, f1_by_time_step, select_threshold
from .utils import set_seed


GCN_EPOCHS = 300
"""Nombre d'époques GCN par défaut ; peut être remplacé dans run_model."""


def _torch_masks(data: EllipticData) -> tuple[torch.Tensor, ...]:
    """Convertit les tenseurs du jeu complet une seule fois par exécution."""
    return (
        torch.from_numpy(data.x),
        torch.from_numpy(data.y).long(),
        torch.from_numpy(data.train_mask),
        torch.from_numpy(data.val_mask),
        torch.from_numpy(data.test_mask),
        data.edge_index,
    )


def _gcn_probabilities(
    data: EllipticData,
    seed: int,
    epochs: int,
    skip: bool,
    patience: int,
    save_embeddings: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Entraîne un GCN et retient l'époque à meilleure PR-AUC de validation."""
    set_seed(seed)
    x, y, train_mask, val_mask, test_mask, edge_index = _torch_masks(data)
    x = x.float()

    class GraphModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GCNConv(x.shape[1], 100)
            self.conv2 = GCNConv(100, 2)
            self.skip_layer = nn.Linear(x.shape[1], 2, bias=False) if skip else None

        def forward(self, values: torch.Tensor, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = F.relu(self.conv1(values, edges))
            dropped = F.dropout(hidden, p=0.2, training=self.training)
            logits = self.conv2(dropped, edges)
            if self.skip_layer is not None:
                logits = logits + self.skip_layer(values)
            return logits, hidden

    model = GraphModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weights = torch.tensor([0.3, 0.7], dtype=torch.float32)
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x, edge_index)
            val_probabilities = torch.softmax(val_logits[val_mask], dim=1)[:, 1].cpu().numpy()
        score = float(average_precision_score(y[val_mask].cpu().numpy(), val_probabilities))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Aucun état GCN n'a été évalué sur la validation.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, hidden = model(x, edge_index)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        embedding_values = hidden.cpu().numpy() if save_embeddings else None
    # La topologie est complète mais aucun lien ne traverse les pas temporels : les masques
    # séparent donc les composantes temporelles et l'évaluation correspond à un cadre inductif.
    return (
        probabilities[val_mask.numpy()],
        probabilities[test_mask.numpy()],
        embedding_values,
        best_epoch,
    )


def run_model(
    name: str,
    data: EllipticData,
    seed: int,
    *,
    mlp_epochs: int = 200,
    gcn_epochs: int = GCN_EPOCHS,
    gcn_patience: int = 40,
    save_embeddings: bool = False,
) -> dict[str, Any]:
    """Exécute un modèle et renvoie probabilités et scores validation/test."""
    started = time.perf_counter()
    set_seed(seed)
    normalized_name = name.strip().lower().replace("_", "-")
    train_indices = np.flatnonzero(data.train_mask)
    val_indices = np.flatnonzero(data.val_mask)
    test_indices = np.flatnonzero(data.test_mask)
    y_train = data.y[train_indices]
    x_train = data.x[train_indices]
    x_val = data.x[val_indices]
    x_test = data.x[test_indices]
    embeddings = None
    best_epoch = None

    if normalized_name in {"lr", "logistic", "logistic-regression", "régression-logistique"}:
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
        model_name = "Logistic Regression"
    elif normalized_name in {"rf", "random-forest", "random forest"}:
        model = RandomForestClassifier(
            n_estimators=50,
            max_features=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
        model_name = "Random Forest"
    elif normalized_name == "mlp":
        set_seed(seed)
        x_all, y_all, train_mask, _, _, _ = _torch_masks(data)
        x_all = x_all.float()
        y_all = y_all.long()
        network = nn.Sequential(nn.Linear(x_all.shape[1], 50), nn.ReLU(), nn.Linear(50, 2))
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        weights = torch.tensor([0.3, 0.7], dtype=torch.float32)
        for _ in range(mlp_epochs):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(network(x_all[train_mask]), y_all[train_mask], weight=weights)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(network(x_all), dim=1)[:, 1].numpy()
        val_prob = probabilities[val_indices]
        test_prob = probabilities[test_indices]
        model_name = "MLP"
    elif normalized_name in {"xgb", "xgboost"}:
        negatives = int(np.sum(y_train == 0))
        positives = int(np.sum(y_train == 1))
        if positives == 0:
            raise ValueError("Aucun positif illicite dans l'entraînement.")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=negatives / positives,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
        model_name = "XGBoost"
    elif normalized_name in {"gcn", "skip-gcn", "skipgcn"}:
        is_skip = normalized_name != "gcn"
        val_prob, test_prob, embeddings, best_epoch = _gcn_probabilities(
            data,
            seed,
            epochs=gcn_epochs,
            skip=is_skip,
            patience=gcn_patience,
            save_embeddings=save_embeddings,
        )
        model_name = "Skip-GCN" if is_skip else "GCN"
    else:
        raise ValueError(f"Modèle inconnu : {name}")

    y_val = data.y[val_indices]
    y_test = data.y[test_indices]
    threshold = select_threshold(y_val, val_prob)
    validation_scores = evaluate_probabilities(y_val, val_prob, threshold)
    # Seul appel de métriques test : le seuil provient exclusivement de la validation.
    test_scores = evaluate_probabilities(y_test, test_prob, threshold)
    return {
        "modele": model_name,
        "seed": int(seed),
        "threshold": threshold,
        "validation": validation_scores,
        "test": test_scores,
        "y_validation": y_val,
        "probabilities_validation": np.asarray(val_prob),
        "y_test": y_test,
        "probabilities_test": np.asarray(test_prob),
        "time_steps_test": data.time_steps[test_indices],
        "support_test": int(np.sum(y_test == 1)),
        "best_epoch": best_epoch,
        "embeddings": embeddings,
        "duration_seconds": time.perf_counter() - started,
    }


def run_model_final(
    name: str,
    data: EllipticData,
    seed: int,
    mlp_epochs: int = 200,
    gcn_epochs: int = GCN_EPOCHS,
) -> dict[str, Any]:
    """Sélectionne seuil/époque, réentraîne sur 1–34, puis évalue le test une fois.

    La phase de sélection appelle ``run_model`` comme dans le protocole standard.
    Comme cette fonction calcule aussi des métriques sur son masque ``test``, une
    vue temporaire lui donne le masque validation à cette place. Le véritable masque
    test reste intact dans ``data`` et n'est utilisé qu'après le réentraînement final.
    """
    from dataclasses import replace
    from sklearn.preprocessing import StandardScaler

    selection_started = time.perf_counter()
    # Cette vue aveugle les features et labels test ; son masque ``test`` pointe
    # sur la validation car run_model calcule toujours des métriques pour ce masque.
    # Le graphe n'a pas d'arête entre périodes : masquer les features test ne change
    # donc pas les prédictions de validation du GCN.
    selection_x = data.x.copy()
    selection_x[data.test_mask] = 0.0
    selection_y = data.y.copy()
    selection_y[data.test_mask] = -1
    selection_data = replace(
        data,
        x=selection_x,
        y=selection_y,
        test_mask=data.val_mask.copy(),
    )
    selection = run_model(
        name,
        selection_data,
        seed,
        mlp_epochs=mlp_epochs,
        gcn_epochs=gcn_epochs,
    )
    selection_duration = time.perf_counter() - selection_started
    threshold = float(selection["threshold"])

    normalized_name = name.strip().lower().replace("_", "-")
    time_steps = data.time_steps
    final_train_mask = (time_steps >= 1) & (time_steps <= 34) & (data.y >= 0)
    y_train = data.y[final_train_mask]
    if len(np.unique(y_train)) != 2:
        raise ValueError("L'entraînement final doit contenir les deux classes étiquetées.")

    # Ajuste une nouvelle normalisation sur toutes les lignes 1–34, connues ou non.
    feature_columns = ["time_step"] + [f"f{i}" for i in range(1, 166)]
    x_raw = data.features[feature_columns].to_numpy(dtype=np.float32, copy=True)
    scaler_fit_mask = (time_steps >= 1) & (time_steps <= 34)
    final_scaler = StandardScaler()
    final_scaler.fit(x_raw[scaler_fit_mask])
    x_final = final_scaler.transform(x_raw).astype(np.float32, copy=False)

    final_started = time.perf_counter()
    set_seed(seed)
    epochs_used: int | None = None

    if normalized_name in {"lr", "logistic", "logistic-regression", "régression-logistique"}:
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(x_final[final_train_mask], y_train)
        test_indices = np.flatnonzero(data.test_mask)
        test_probabilities = model.predict_proba(x_final[test_indices])[:, 1]
        model_name = "Logistic Regression"
    elif normalized_name in {"rf", "random-forest", "random forest"}:
        model = RandomForestClassifier(
            n_estimators=50,
            max_features=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_final[final_train_mask], y_train)
        test_indices = np.flatnonzero(data.test_mask)
        test_probabilities = model.predict_proba(x_final[test_indices])[:, 1]
        model_name = "Random Forest"
    elif normalized_name == "mlp":
        set_seed(seed)
        x_tensor = torch.from_numpy(x_final).float()
        y_tensor = torch.from_numpy(data.y).long()
        train_tensor_mask = torch.from_numpy(final_train_mask)
        network = nn.Sequential(nn.Linear(x_tensor.shape[1], 50), nn.ReLU(), nn.Linear(50, 2))
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        weights = torch.tensor([0.3, 0.7], dtype=torch.float32)
        for _ in range(mlp_epochs):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                network(x_tensor[train_tensor_mask]), y_tensor[train_tensor_mask], weight=weights
            )
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(network(x_tensor), dim=1)[:, 1].numpy()
        test_indices = np.flatnonzero(data.test_mask)
        test_probabilities = probabilities[test_indices]
        epochs_used = int(mlp_epochs)
        model_name = "MLP"
    elif normalized_name in {"xgb", "xgboost"}:
        negatives = int(np.sum(y_train == 0))
        positives = int(np.sum(y_train == 1))
        if positives == 0:
            raise ValueError("Aucun positif illicite dans l'entraînement final.")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=negatives / positives,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(x_final[final_train_mask], y_train)
        test_indices = np.flatnonzero(data.test_mask)
        test_probabilities = model.predict_proba(x_final[test_indices])[:, 1]
        model_name = "XGBoost"
    elif normalized_name in {"gcn", "skip-gcn", "skipgcn"}:
        best_epoch = selection["best_epoch"]
        if best_epoch is None or int(best_epoch) < 1:
            raise RuntimeError("L'étape de sélection n'a pas fourni d'époque GCN valide.")
        epochs_used = int(best_epoch)
        x_tensor = torch.from_numpy(x_final).float()
        y_tensor = torch.from_numpy(data.y).long()
        train_tensor_mask = torch.from_numpy(final_train_mask)
        edge_index = data.edge_index
        is_skip = normalized_name != "gcn"

        class FinalGraphModel(nn.Module):
            """GCN final sans calcul de validation durant le réentraînement."""

            def __init__(self) -> None:
                super().__init__()
                self.conv1 = GCNConv(x_tensor.shape[1], 100)
                self.conv2 = GCNConv(100, 2)
                self.skip_layer = nn.Linear(x_tensor.shape[1], 2, bias=False) if is_skip else None

            def forward(self, values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
                hidden = F.relu(self.conv1(values, edges))
                hidden = F.dropout(hidden, p=0.2, training=self.training)
                logits = self.conv2(hidden, edges)
                if self.skip_layer is not None:
                    logits = logits + self.skip_layer(values)
                return logits

        network = FinalGraphModel()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        weights = torch.tensor([0.3, 0.7], dtype=torch.float32)
        for _ in range(epochs_used):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            logits = network(x_tensor, edge_index)
            loss = F.cross_entropy(logits[train_tensor_mask], y_tensor[train_tensor_mask], weight=weights)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(network(x_tensor, edge_index), dim=1)[:, 1].numpy()
        test_indices = np.flatnonzero(data.test_mask)
        test_probabilities = probabilities[test_indices]
        model_name = "Skip-GCN" if is_skip else "GCN"
    else:
        raise ValueError(f"Modèle inconnu : {name}")

    final_duration = time.perf_counter() - final_started
    # Première et unique évaluation sur les labels du véritable test.
    y_test = data.y[test_indices]
    test_scores = evaluate_probabilities(y_test, test_probabilities, threshold)
    per_step = f1_by_time_step(
        y_test,
        test_probabilities,
        time_steps[test_indices],
        threshold,
    )
    return {
        "modele": model_name,
        "seed": int(seed),
        "threshold": threshold,
        "test": test_scores,
        "f1_par_pas": per_step,
        "epochs_used": epochs_used,
        "selection_best_epoch": selection["best_epoch"],
        "selection_duration_seconds": selection_duration,
        "final_duration_seconds": final_duration,
        "duration_seconds": selection_duration + final_duration,
        "scaler_fit_n_samples": int(final_scaler.n_samples_seen_),
        "scaler_fit_max_time_step": 34,
        "support_test": int(np.sum(y_test == 1)),
    }
