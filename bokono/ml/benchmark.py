# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.benchmark — Validation empirique FaNet v1.0
======================================================
Comparaison rigoureuse de FaNet v3.0 (PyTorch) contre les modèles
standard de scikit-learn sur des datasets réels.

Datasets utilisés :
  1. Breast Cancer Wisconsin — classification binaire (santé)
  2. Diabetes — régression (santé)
  3. Heart Disease (Statlog) — classification binaire (santé)

Modèles comparés :
  - FaNet v3.0 (PyTorch) — le nôtre
  - RandomForest (scikit-learn)
  - GradientBoosting (scikit-learn)
  - MLP (scikit-learn)
  - LogisticRegression / Ridge (scikit-learn)

Métriques :
  Classification : accuracy, F1-score, AUC-ROC
  Régression     : MSE, MAE, R²

Protocole :
  - 5-fold cross-validation (reproductible, seed=42)
  - Normalisation MinMaxScaler sur [0, 1]
  - Même split train/test pour tous les modèles

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

# Sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

# FaNet
from bokono.ml.torch_network import TorchFaNet


# ─── Structures de résultat ───────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Résultat d'un modèle sur un dataset."""
    model_name: str
    dataset_name: str
    task: str  # "classification" ou "regression"
    metrics: dict[str, list[float]] = field(default_factory=dict)
    train_times: list[float] = field(default_factory=list)

    @property
    def mean_metrics(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.metrics.items()}

    @property
    def std_metrics(self) -> dict[str, float]:
        return {k: float(np.std(v)) for k, v in self.metrics.items()}

    @property
    def mean_train_time(self) -> float:
        return float(np.mean(self.train_times))


@dataclass
class BenchmarkReport:
    """Rapport complet de benchmark."""
    results: list[ModelResult] = field(default_factory=list)

    def print_report(self) -> None:
        """Affiche le rapport formaté."""
        datasets = sorted(set(r.dataset_name for r in self.results))
        for ds in datasets:
            ds_results = [r for r in self.results if r.dataset_name == ds]
            if not ds_results:
                continue

            task = ds_results[0].task
            print()
            print("=" * 80)
            print(f"  Dataset : {ds} ({task})")
            print("=" * 80)

            if task == "classification":
                print(f"  {'Modèle':<25} {'Accuracy':>12} {'F1-Score':>12} {'AUC-ROC':>12} {'Temps (s)':>12}")
            else:
                print(f"  {'Modèle':<25} {'MSE':>12} {'MAE':>12} {'R²':>12} {'Temps (s)':>12}")
            print("  " + "-" * 75)

            # Trier par la métrique principale (accuracy ou R²)
            key_metric = "accuracy" if task == "classification" else "r2"
            ds_results.sort(key=lambda r: r.mean_metrics.get(key_metric, 0), reverse=True)

            for r in ds_results:
                m = r.mean_metrics
                s = r.std_metrics
                t = r.mean_train_time

                if task == "classification":
                    acc = f"{m.get('accuracy', 0):.4f}±{s.get('accuracy', 0):.3f}"
                    f1 = f"{m.get('f1', 0):.4f}±{s.get('f1', 0):.3f}"
                    auc = f"{m.get('auc', 0):.4f}±{s.get('auc', 0):.3f}"
                    print(f"  {r.model_name:<25} {acc:>12} {f1:>12} {auc:>12} {t:>11.3f}s")
                else:
                    mse = f"{m.get('mse', 0):.4f}±{s.get('mse', 0):.3f}"
                    mae = f"{m.get('mae', 0):.4f}±{s.get('mae', 0):.3f}"
                    r2 = f"{m.get('r2', 0):.4f}±{s.get('r2', 0):.3f}"
                    print(f"  {r.model_name:<25} {mse:>12} {mae:>12} {r2:>12} {t:>11.3f}s")

            # Identifier FaNet et le meilleur concurrent
            fanet_r = next((r for r in ds_results if "FaNet" in r.model_name), None)
            best_r = ds_results[0]

            if fanet_r and best_r:
                print()
                if fanet_r.model_name == best_r.model_name:
                    print(f"  ✅ FaNet est le MEILLEUR modèle sur ce dataset")
                else:
                    fm = fanet_r.mean_metrics.get(key_metric, 0)
                    bm = best_r.mean_metrics.get(key_metric, 0)
                    diff = abs(fm - bm)
                    print(f"  📊 FaNet vs {best_r.model_name} : Δ{key_metric} = {diff:.4f}")
                    if diff < 0.02:
                        print(f"  ✅ FaNet est COMPÉTITIF (écart < 2%)")
                    elif diff < 0.05:
                        print(f"  ⚠️  FaNet est proche (écart < 5%)")
                    else:
                        print(f"  📉 FaNet est inférieur (écart ≥ 5%)")

                # L'avantage unique de FaNet
                print(f"  💡 Avantage FaNet : chaque prédiction inclut un du Fâ interprétatif")

        print()
        print("=" * 80)


# ─── Chargement des datasets ──────────────────────────────────────────────────

def load_breast_cancer_data() -> tuple[np.ndarray, np.ndarray, str]:
    """Breast Cancer Wisconsin — classification binaire (santé)."""
    data = load_breast_cancer()
    return data.data, data.target, "Breast Cancer (classification)"


def load_diabetes_data() -> tuple[np.ndarray, np.ndarray, str]:
    """Diabetes — régression (santé)."""
    data = load_diabetes()
    return data.data, data.target, "Diabetes (régression)"


def load_heart_data() -> tuple[np.ndarray, np.ndarray, str]:
    """
    Heart Disease Statlog — classification binaire (santé).

    Si fetch_openml échoue (réseau), on génère un jeu de données
    de remplacement avec les mêmes caractéristiques statistiques.
    """
    try:
        from sklearn.datasets import fetch_openml
        heart = fetch_openml(name="heart-statlog", version=1, as_frame=False,
                             parser="auto")
        X = heart.data.astype(np.float64)
        y = (heart.target.astype(int) - 1).astype(np.float64)  # {1,2} → {0,1}
        if y.max() > 1:
            y = (y > 0).astype(np.float64)
        return X, y, "Heart Disease (classification)"
    except Exception:
        # Fallback : Wine dataset (sklearn built-in, classification binaire)
        from sklearn.datasets import load_wine
        data = load_wine()
        X = data.data
        y = (data.target > 0).astype(np.float64)  # 3 classes → binaire
        return X, y, "Wine (classification, fallback)"


# ─── Entraînement FaNet ───────────────────────────────────────────────────────

def _train_eval_fanet_classification(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 100,
    lr: float = 0.005,
) -> tuple[dict[str, float], float]:
    """Entraîne et évalue FaNet pour la classification."""
    t0 = time.time()

    net = TorchFaNet(
        n_features=X_train.shape[1],
        hidden_dim=hidden_dim,
        n_outputs=1,
        activation="fa_sigma",
        seed=42,
    )
    net.fit(X_train, y_train, epochs=epochs, lr=lr, verbose=False)
    train_time = time.time() - t0

    pred_proba = net.predict_proba(X_test).flatten()
    pred_class = (pred_proba >= 0.5).astype(int)

    y_int = y_test.astype(int)
    acc = accuracy_score(y_int, pred_class)
    f1 = f1_score(y_int, pred_class, zero_division=0)
    try:
        auc = roc_auc_score(y_int, pred_proba)
    except ValueError:
        auc = 0.5

    return {"accuracy": acc, "f1": f1, "auc": auc}, train_time


def _train_eval_fanet_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 150,
    lr: float = 0.005,
) -> tuple[dict[str, float], float]:
    """Entraîne et évalue FaNet pour la régression."""
    t0 = time.time()

    # Normaliser la cible dans [0, 1] pour FaNet (sortie sigmoid)
    y_min, y_max = y_train.min(), y_train.max()
    y_range = y_max - y_min + 1e-10
    y_train_norm = (y_train - y_min) / y_range
    y_test_norm = (y_test - y_min) / y_range

    net = TorchFaNet(
        n_features=X_train.shape[1],
        hidden_dim=hidden_dim,
        n_outputs=1,
        activation="fa_sigma",
        seed=42,
    )
    net.fit(X_train, y_train_norm, epochs=epochs, lr=lr, verbose=False)
    train_time = time.time() - t0

    pred_norm = net.predict_proba(X_test).flatten()
    pred = pred_norm * y_range + y_min  # Dénormaliser

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return {"mse": mse, "mae": mae, "r2": r2}, train_time


# ─── Entraînement sklearn ─────────────────────────────────────────────────────

def _train_eval_sklearn(
    model, model_name: str, task: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Entraîne et évalue un modèle sklearn."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    if task == "classification":
        pred = model.predict(X_test)
        y_int = y_test.astype(int)
        pred_int = pred.astype(int)
        acc = accuracy_score(y_int, pred_int)
        f1 = f1_score(y_int, pred_int, zero_division=0)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.decision_function(X_test)
            auc = roc_auc_score(y_int, proba)
        except (ValueError, AttributeError):
            auc = 0.5
        return {"accuracy": acc, "f1": f1, "auc": auc}, train_time
    else:
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        return {"mse": mse, "mae": mae, "r2": r2}, train_time


# ─── Benchmark principal ──────────────────────────────────────────────────────

def run_benchmark(
    n_folds: int = 5,
    seed: int = 42,
    fanet_epochs: int = 100,
    fanet_lr: float = 0.005,
    fanet_hidden: int = 32,
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Exécute le benchmark complet.

    Protocole :
      1. Chaque dataset est normalisé avec MinMaxScaler
      2. 5-fold CV stratifié (classification) ou standard (régression)
      3. Même split pour tous les modèles
      4. Résultats moyennés sur les 5 folds

    Returns:
        BenchmarkReport contenant tous les résultats
    """
    report = BenchmarkReport()

    # ── Datasets ──────────────────────────────────────────────────────────
    datasets = [
        (*load_breast_cancer_data(), "classification"),
        (*load_diabetes_data(), "regression"),
        (*load_heart_data(), "classification"),
    ]

    # ── Modèles sklearn ───────────────────────────────────────────────────
    classifiers = [
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=100, random_state=seed)),
        ("GradientBoosting", lambda: GradientBoostingClassifier(n_estimators=100, random_state=seed)),
        ("MLP (sklearn)", lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                                                 random_state=seed)),
        ("LogisticRegression", lambda: LogisticRegression(max_iter=1000, random_state=seed)),
    ]

    regressors = [
        ("RandomForest", lambda: RandomForestRegressor(n_estimators=100, random_state=seed)),
        ("GradientBoosting", lambda: GradientBoostingRegressor(n_estimators=100, random_state=seed)),
        ("MLP (sklearn)", lambda: MLPRegressor(hidden_layer_sizes=(32,), max_iter=500,
                                                random_state=seed)),
        ("Ridge", lambda: Ridge(alpha=1.0)),
    ]

    for X, y, ds_name, task in datasets:
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Benchmarking : {ds_name}")
            print(f"  Samples={X.shape[0]}, Features={X.shape[1]}, Task={task}")
            print(f"{'─' * 60}")

        # Cross-validation
        if task == "classification":
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            model_list = classifiers
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            model_list = regressors

        # Résultats par modèle
        fanet_result = ModelResult("FaNet v3.0", ds_name, task)
        sklearn_results = {name: ModelResult(name, ds_name, task) for name, _ in model_list}

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Normalisation [0, 1]
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            fold_label = f"Fold {fold_idx + 1}/{n_folds}"

            # ── FaNet ────────────────────────────────────────────────────
            if task == "classification":
                metrics, t = _train_eval_fanet_classification(
                    X_train, y_train, X_test, y_test,
                    hidden_dim=fanet_hidden, epochs=fanet_epochs, lr=fanet_lr,
                )
            else:
                metrics, t = _train_eval_fanet_regression(
                    X_train, y_train, X_test, y_test,
                    hidden_dim=fanet_hidden, epochs=fanet_epochs * 2, lr=fanet_lr,
                )

            for k, v in metrics.items():
                fanet_result.metrics.setdefault(k, []).append(v)
            fanet_result.train_times.append(t)

            if verbose:
                key = "accuracy" if task == "classification" else "r2"
                print(f"    {fold_label} | FaNet {key}={metrics[key]:.4f} ({t:.2f}s)")

            # ── Sklearn models ───────────────────────────────────────────
            for name, model_fn in model_list:
                model = model_fn()
                metrics_sk, t_sk = _train_eval_sklearn(
                    model, name, task, X_train, y_train, X_test, y_test,
                )
                for k, v in metrics_sk.items():
                    sklearn_results[name].metrics.setdefault(k, []).append(v)
                sklearn_results[name].train_times.append(t_sk)

        report.results.append(fanet_result)
        for r in sklearn_results.values():
            report.results.append(r)

    return report


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("  FaNet v1.0 — Benchmark empirique contre les standards ML")
    print("  Projet Matrice-256 / bokono")
    print("=" * 80)

    report = run_benchmark(
        n_folds=5,
        fanet_epochs=150,
        fanet_lr=0.005,
        fanet_hidden=32,
    )
    report.print_report()
