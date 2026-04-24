# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests unitaires — FaNet v1.0 (Validation empirique)
Projet bokono / Matrice-256

Valide que le benchmark tourne sur des datasets réels et que FaNet
produit des résultats compétitifs et reproductibles.

Datasets réels utilisés dans les tests :
  1. Breast Cancer Wisconsin — sklearn.datasets.load_breast_cancer
     - 569 échantillons, 30 features
     - Classification binaire : tumeur maligne (0) / bénigne (1)
     - Domaine : Santé — dépistage cancer du sein
     - Pertinence Bénin : screening médical dans les zones rurales

  2. Diabetes — sklearn.datasets.load_diabetes
     - 442 échantillons, 10 features
     - Régression : progression du diabète à 1 an
     - Domaine : Santé — prédiction glycémique
     - Pertinence Bénin : le diabète est un enjeu sanitaire majeur en Afrique

  3. Heart Disease (Statlog) ou Wine (fallback) — sklearn
     - 270 échantillons, 13 features
     - Classification binaire : présence/absence de maladie cardiaque
     - Domaine : Santé — diagnostic cardiaque précoce
     - Pertinence Bénin : accès limité aux cardiologues → IA de triage

Exécuter avec : python -m pytest tests/test_v10.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.benchmark import (
    run_benchmark,
    BenchmarkReport,
    ModelResult,
    load_breast_cancer_data,
    load_diabetes_data,
    load_heart_data,
    _train_eval_fanet_classification,
    _train_eval_fanet_regression,
)
from bokono.ml.torch_network import TorchFaNet
from sklearn.preprocessing import MinMaxScaler


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def breast_cancer():
    """Breast Cancer Wisconsin — dataset de santé réel."""
    X, y, name = load_breast_cancer_data()
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, y, name

@pytest.fixture(scope="module")
def diabetes():
    """Diabetes — dataset de régression réel."""
    X, y, name = load_diabetes_data()
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, y, name

@pytest.fixture(scope="module")
def heart():
    """Heart Disease — dataset de classification réel."""
    X, y, name = load_heart_data()
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, y, name


# ═══════════════════════════════════════════════════════════════════════════════
# T_v10.1 : Chargement des datasets
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetLoading:
    """Vérification que les datasets réels se chargent correctement."""

    def test_breast_cancer_shape(self, breast_cancer):
        X, y, name = breast_cancer
        assert X.shape == (569, 30), f"Shape attendu (569, 30), reçu {X.shape}"
        assert y.shape == (569,)
        assert "Breast Cancer" in name

    def test_breast_cancer_classes(self, breast_cancer):
        X, y, _ = breast_cancer
        unique = np.unique(y)
        assert len(unique) == 2, f"Binaire attendu, reçu {unique}"
        assert 0 in unique and 1 in unique

    def test_breast_cancer_normalized(self, breast_cancer):
        X, _, _ = breast_cancer
        assert X.min() >= 0.0 - 1e-6
        assert X.max() <= 1.0 + 1e-6

    def test_diabetes_shape(self, diabetes):
        X, y, name = diabetes
        assert X.shape == (442, 10)
        assert y.shape == (442,)
        assert "Diabetes" in name

    def test_diabetes_is_regression(self, diabetes):
        _, y, _ = diabetes
        unique = np.unique(y)
        assert len(unique) > 10, "Dataset de régression — beaucoup de valeurs uniques"

    def test_heart_shape(self, heart):
        X, y, name = heart
        assert X.shape[0] > 100, f"Trop peu d'échantillons : {X.shape[0]}"
        assert X.shape[1] >= 10, f"Trop peu de features : {X.shape[1]}"

    def test_heart_classes(self, heart):
        _, y, _ = heart
        unique = np.unique(y)
        assert len(unique) == 2, f"Binaire attendu, reçu {unique}"


# ═══════════════════════════════════════════════════════════════════════════════
# T_v10.2 : FaNet sur Breast Cancer (classification réelle)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaNetBreastCancer:
    """Valide FaNet sur le dataset de cancer du sein — données réelles."""

    def test_fanet_trains_on_real_data(self, breast_cancer):
        """FaNet peut s'entraîner sur des données réelles sans crash."""
        X, y, _ = breast_cancer
        n_train = int(0.8 * len(X))
        X_train, y_train = X[:n_train], y[:n_train]

        net = TorchFaNet(n_features=30, hidden_dim=32, seed=42)
        history = net.fit(X_train, y_train, epochs=30, lr=0.005, verbose=False)
        assert len(history["loss"]) == 30
        assert all(np.isfinite(l) for l in history["loss"])

    def test_fanet_accuracy_above_chance(self, breast_cancer):
        """FaNet dépasse le hasard (>50%) sur Breast Cancer."""
        X, y, _ = breast_cancer
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        metrics, _ = _train_eval_fanet_classification(
            X_train, y_train, X_test, y_test,
            hidden_dim=32, epochs=80, lr=0.005,
        )
        assert metrics["accuracy"] > 0.50, \
            f"Accuracy {metrics['accuracy']:.2%} doit dépasser le hasard"

    def test_fanet_accuracy_respectable(self, breast_cancer):
        """FaNet atteint une accuracy respectable (>85%) sur Breast Cancer."""
        X, y, _ = breast_cancer
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        metrics, _ = _train_eval_fanet_classification(
            X_train, y_train, X_test, y_test,
            hidden_dim=32, epochs=150, lr=0.005,
        )
        assert metrics["accuracy"] > 0.85, \
            f"Accuracy {metrics['accuracy']:.2%} doit être >85% pour être compétitif"

    def test_fanet_predictions_include_fa_interpretation(self, breast_cancer):
        """Chaque prédiction inclut un du Fâ — l'avantage unique de FaNet."""
        X, y, _ = breast_cancer
        net = TorchFaNet(n_features=30, hidden_dim=16, seed=42)
        net.fit(X[:100], y[:100], epochs=10, verbose=False)
        preds, dus = net.predict(X[:5])
        assert len(dus) == 5
        for du in dus:
            assert "nom_fon" in du, "Chaque prédiction doit avoir une interprétation Fâ"
            assert "id" in du


# ═══════════════════════════════════════════════════════════════════════════════
# T_v10.3 : FaNet sur Diabetes (régression réelle)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaNetDiabetes:
    """Valide FaNet sur le dataset de diabète — régression réelle."""

    def test_fanet_regression_trains(self, diabetes):
        """FaNet peut s'entraîner en régression sur des données réelles."""
        X, y, _ = diabetes
        n_train = int(0.8 * len(X))
        y_norm = (y - y.min()) / (y.max() - y.min())

        net = TorchFaNet(n_features=10, hidden_dim=32, seed=42)
        history = net.fit(X[:n_train], y_norm[:n_train], epochs=30, verbose=False)
        assert all(np.isfinite(l) for l in history["loss"])

    def test_fanet_regression_r2_positive(self, diabetes):
        """FaNet produit un R² > 0 (meilleur que la moyenne)."""
        X, y, _ = diabetes
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        metrics, _ = _train_eval_fanet_regression(
            X_train, y_train, X_test, y_test,
            hidden_dim=32, epochs=200, lr=0.005,
        )
        assert metrics["r2"] > 0.0, \
            f"R² = {metrics['r2']:.4f} doit être positif (meilleur que la moyenne)"


# ═══════════════════════════════════════════════════════════════════════════════
# T_v10.4 : FaNet sur Heart Disease (classification réelle)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaNetHeart:
    """Valide FaNet sur Heart Disease — classification réelle."""

    def test_fanet_heart_above_chance(self, heart):
        """FaNet dépasse le hasard sur Heart Disease."""
        X, y, _ = heart
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        metrics, _ = _train_eval_fanet_classification(
            X_train, y_train, X_test, y_test,
            hidden_dim=32, epochs=100, lr=0.005,
        )
        assert metrics["accuracy"] > 0.50, \
            f"Accuracy {metrics['accuracy']:.2%} doit dépasser le hasard"


# ═══════════════════════════════════════════════════════════════════════════════
# T_v10.5 : Benchmark complet (5-fold CV)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmark:
    """Validation du système de benchmark complet."""

    def test_benchmark_produces_report(self):
        """Le benchmark retourne un BenchmarkReport non-vide."""
        report = run_benchmark(
            n_folds=2,  # 2 folds pour la vitesse en test
            fanet_epochs=30,
            fanet_hidden=16,
            verbose=False,
        )
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) > 0

    def test_all_models_present(self):
        """Le rapport contient FaNet et les modèles sklearn."""
        report = run_benchmark(n_folds=2, fanet_epochs=20, fanet_hidden=16, verbose=False)
        model_names = {r.model_name for r in report.results}
        assert "FaNet v3.0" in model_names
        assert "RandomForest" in model_names
        assert "GradientBoosting" in model_names

    def test_all_datasets_present(self):
        """Le rapport couvre les 3 datasets."""
        report = run_benchmark(n_folds=2, fanet_epochs=20, fanet_hidden=16, verbose=False)
        ds_names = {r.dataset_name for r in report.results}
        assert len(ds_names) >= 3, f"3 datasets attendus, reçu {ds_names}"

    def test_metrics_are_finite(self):
        """Toutes les métriques sont finies (pas de NaN/Inf)."""
        report = run_benchmark(n_folds=2, fanet_epochs=20, fanet_hidden=16, verbose=False)
        for r in report.results:
            for k, vals in r.metrics.items():
                for v in vals:
                    assert np.isfinite(v), f"NaN/Inf dans {r.model_name}.{k}: {v}"

    def test_report_printable(self):
        """Le rapport s'affiche sans erreur."""
        report = run_benchmark(n_folds=2, fanet_epochs=20, fanet_hidden=16, verbose=False)
        report.print_report()  # Ne doit pas crash


# ═══════════════════════════════════════════════════════════════════════════════
# T_v10.6 : Propriétés v1.0
# ═══════════════════════════════════════════════════════════════════════════════

class TestV10Properties:
    """Vérifications des propriétés structurelles de v1.0."""

    def test_model_result_dataclass(self):
        """ModelResult calcule les moyennes correctement."""
        r = ModelResult("test", "ds", "classification")
        r.metrics = {"accuracy": [0.8, 0.9, 0.85]}
        r.train_times = [1.0, 1.5, 1.2]
        assert abs(r.mean_metrics["accuracy"] - 0.85) < 1e-6
        assert abs(r.mean_train_time - (1.0 + 1.5 + 1.2) / 3) < 1e-6

    def test_fanet_version_is_30(self):
        """TorchFaNet est en v3.0.0 (le réseau sous-jacent)."""
        assert TorchFaNet.VERSION == "3.0.0"

    def test_fanet_predict_includes_du(self):
        """Chaque prédiction FaNet inclut un du Fâ — le différenciateur."""
        net = TorchFaNet(n_features=4, hidden_dim=8, seed=42)
        _, dus = net.predict(np.random.rand(3, 4))
        for du in dus:
            assert isinstance(du, dict)
            assert "nom_fon" in du
