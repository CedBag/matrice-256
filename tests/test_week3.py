# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests unitaires — Semaine 3
Projet bokono / Matrice-256

Valide les couches FaNet et le réseau complet.
Exécuter avec : python -m pytest tests/test_week3.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.layers import (
    FaEmbeddingLayer, FaHiddenLayer, FaDecoderLayer,
    _xavier_init, _fa_weight_init,
)
from bokono.ml.network import (
    FaNet, mse_loss, mse_gradient, binary_cross_entropy, bce_gradient,
)
from bokono.core.fa_matrix import FaMatrix


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fa_tensor():
    """Tenseur Matrice-256 — chargé une seule fois."""
    return FaMatrix().to_weight_tensor()

@pytest.fixture(scope="module")
def X_small():
    """Données synthétiques : 20 échantillons, 4 features dans [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (20, 4))

@pytest.fixture(scope="module")
def y_small(X_small):
    """Cibles linéaires simples pour tester l'entraînement."""
    return X_small[:, 0] * 0.5 + X_small[:, 1] * 0.3

@pytest.fixture
def net_small():
    """FaNet minimal pour les tests rapides."""
    return FaNet(n_features=4, hidden_dim=16, n_outputs=1, seed=42)


# ─── T19.1 : Utilitaires d'initialisation ─────────────────────────────────────
class TestInit:

    def test_xavier_shape(self):
        """_xavier_init retourne la bonne forme."""
        W = _xavier_init((8, 16), seed=0)
        assert W.shape == (8, 16)

    def test_xavier_values_bounded(self):
        """Valeurs Xavier bornées dans [-√(6/(m+n)), +√(6/(m+n))]."""
        W = _xavier_init((8, 16), seed=0)
        limit = np.sqrt(6 / (8 + 16))
        assert np.all(np.abs(W) <= limit + 1e-9)

    def test_fa_weight_init_shape(self, fa_tensor):
        """_fa_weight_init retourne la bonne forme."""
        W = _fa_weight_init((32, 16), fa_tensor, seed=42)
        assert W.shape == (32, 16)

    def test_fa_weight_init_different_from_xavier(self, fa_tensor):
        """L'initialisation Fâ est différente de Xavier (ancrage Matrice-256)."""
        W_fa = _fa_weight_init((16, 16), fa_tensor, seed=42)
        W_xav = _xavier_init((16, 16), seed=42)
        assert not np.allclose(W_fa, W_xav), \
            "L'init Fâ doit différer de Xavier"


# ─── T19.2 : FaEmbeddingLayer ────────────────────────────────────────────────
class TestFaEmbeddingLayer:

    def test_output_shape_batch(self):
        """Output shape : (batch, n_features × 4)."""
        layer = FaEmbeddingLayer(n_features=3)
        x = np.random.uniform(0, 1, (5, 3))
        out = layer.forward(x)
        assert out.shape == (5, 12)

    def test_output_shape_single(self):
        """Un seul échantillon → (1, n_features × 4)."""
        layer = FaEmbeddingLayer(n_features=4)
        x = np.array([0.2, 0.5, 0.8, 0.1])
        out = layer.forward(x)
        assert out.shape == (1, 16)

    def test_values_in_valid_range(self):
        """Les valeurs d'embedding sont dans [0, 1]."""
        layer = FaEmbeddingLayer(n_features=4)
        x = np.random.uniform(0, 1, (10, 4))
        out = layer.forward(x)
        assert np.all(out >= 0.0) and np.all(out <= 1.0 + 1e-9)

    def test_wrong_n_features_raises(self):
        """Mauvais nombre de features → ValueError."""
        layer = FaEmbeddingLayer(n_features=3)
        with pytest.raises(ValueError, match="features"):
            layer.forward(np.ones((5, 5)))  # 5 features ≠ 3

    def test_no_nan_or_inf(self):
        """Pas de NaN ou Inf dans les embeddings."""
        layer = FaEmbeddingLayer(n_features=6)
        x = np.random.uniform(0, 1, (50, 6))
        out = layer.forward(x)
        assert np.all(np.isfinite(out))

    def test_extreme_values(self):
        """Les valeurs extrêmes (0 et 1) sont bien gérées."""
        layer = FaEmbeddingLayer(n_features=2)
        x = np.array([[0.0, 1.0]])
        out = layer.forward(x)
        assert np.all(np.isfinite(out))

    def test_val_norm_dimension(self):
        """La dimension 0 de chaque feature contient la valeur normalisée."""
        layer = FaEmbeddingLayer(n_features=1)
        x = np.array([[0.5]])
        out = layer.forward(x)
        # La dimension 0 doit être proche de 0.5
        assert abs(out[0, 0] - 0.5) < 0.05

    def test_backward_shape(self):
        """backward() retourne le bon gradient shape."""
        layer = FaEmbeddingLayer(n_features=3)
        x = np.random.uniform(0, 1, (4, 3))
        out = layer.forward(x)
        grad_out = np.ones_like(out)
        grad_x = layer.backward(grad_out)
        assert grad_x.shape == (4, 3)

    def test_no_trainable_params(self):
        """FaEmbeddingLayer n'a pas de paramètres entraînables."""
        layer = FaEmbeddingLayer(n_features=3)
        assert layer.params == {}


# ─── T19.3 : FaHiddenLayer ───────────────────────────────────────────────────
class TestFaHiddenLayer:

    def test_output_shape(self, fa_tensor):
        """Output shape : (batch, hidden_dim)."""
        layer = FaHiddenLayer(12, 32, fa_tensor=fa_tensor)
        x = np.random.randn(5, 12)
        out = layer.forward(x)
        assert out.shape == (5, 32)

    def test_output_nonneg_with_fa_relu(self, fa_tensor):
        """Avec FaReLU, la sortie est toujours >= 0."""
        layer = FaHiddenLayer(12, 32, activation="fa_relu", fa_tensor=fa_tensor)
        x = np.random.randn(10, 12)
        out = layer.forward(x)
        assert np.all(out >= 0.0)

    def test_output_sigma_in_01(self, fa_tensor):
        """Avec FaSigma, la sortie est dans ]0, 1[."""
        layer = FaHiddenLayer(12, 32, activation="fa_sigma", fa_tensor=fa_tensor)
        x = np.random.randn(10, 12)
        out = layer.forward(x)
        assert np.all(out > 0) and np.all(out < 1)

    def test_backward_shapes(self, fa_tensor):
        """backward() retourne les gradients correctement shapés."""
        layer = FaHiddenLayer(12, 32, fa_tensor=fa_tensor)
        x = np.random.randn(5, 12)
        out = layer.forward(x)
        grad_out = np.ones_like(out)
        grad_x = layer.backward(grad_out)
        assert grad_x.shape == (5, 12)
        assert layer.dW.shape == (32, 12)
        assert layer.db.shape == (32,)

    def test_weights_differ_from_xavier(self, fa_tensor):
        """Les poids initiaux diffèrent de Xavier (init Fâ active)."""
        layer_fa = FaHiddenLayer(12, 32, fa_tensor=fa_tensor, seed=42)
        layer_xav = FaHiddenLayer(12, 32, fa_tensor=None, seed=42)
        assert not np.allclose(layer_fa.W, layer_xav.W)

    def test_zero_grad(self, fa_tensor):
        """zero_grad() remet les gradients à zéro."""
        layer = FaHiddenLayer(12, 32, fa_tensor=fa_tensor)
        x = np.random.randn(5, 12)
        layer.forward(x)
        layer.backward(np.ones((5, 32)))
        layer.zero_grad()
        assert np.all(layer.dW == 0.0)
        assert np.all(layer.db == 0.0)

    def test_weights_change_after_update(self, fa_tensor):
        """Les poids changent après update()."""
        layer = FaHiddenLayer(12, 32, fa_tensor=fa_tensor)
        W_before = layer.W.copy()
        x = np.random.randn(5, 12)
        layer.forward(x)
        layer.backward(np.ones((5, 32)))
        layer.update(lr=0.01)
        assert not np.allclose(layer.W, W_before)


# ─── T19.4 : FaDecoderLayer ──────────────────────────────────────────────────
class TestFaDecoderLayer:

    def test_output_shape(self):
        """Output shape : (batch, n_outputs)."""
        layer = FaDecoderLayer(32, n_outputs=1)
        x = np.random.randn(5, 32)
        out = layer.forward(x)
        assert out.shape == (5, 1)

    def test_output_in_01(self):
        """Sortie dans ]0, 1[ (FaDual activation)."""
        layer = FaDecoderLayer(32, n_outputs=1)
        x = np.random.randn(20, 32)
        out = layer.forward(x)
        assert np.all(out > 0) and np.all(out < 1)

    def test_forward_with_interpretation_returns_du(self):
        """forward_with_interpretation retourne prediction + liste de du."""
        layer = FaDecoderLayer(32, n_outputs=1)
        x = np.random.randn(5, 32)
        preds, dus = layer.forward_with_interpretation(x)
        assert preds.shape == (5, 1)
        assert len(dus) == 5
        for du in dus:
            assert "nom_fon" in du
            assert "polarite" in du
            assert "code_binaire" in du

    def test_backward_shape(self):
        """backward() retourne le gradient vers la couche cachée."""
        layer = FaDecoderLayer(32, n_outputs=1)
        x = np.random.randn(5, 32)
        layer.forward(x)
        grad_x = layer.backward(np.ones((5, 1)))
        assert grad_x.shape == (5, 32)


# ─── T19.5 : FaNet complet ────────────────────────────────────────────────────
class TestFaNet:

    def test_summary_runs(self, net_small):
        """summary() s'exécute sans erreur."""
        net_small.summary()  # Ne doit pas lever d'exception

    def test_forward_shape(self, net_small, X_small):
        """forward() retourne la bonne forme."""
        out = net_small.forward(X_small)
        assert out.shape == (20, 1)

    def test_forward_values_in_01(self, net_small, X_small):
        """Sortie de forward() dans ]0, 1[."""
        out = net_small.forward(X_small)
        assert np.all(out > 0) and np.all(out < 1)

    def test_forward_no_nan(self, net_small, X_small):
        """Pas de NaN dans la sortie."""
        out = net_small.forward(X_small)
        assert np.all(np.isfinite(out))

    def test_predict_returns_du_interpretations(self, net_small, X_small):
        """predict() retourne (predictions, list[dict])."""
        preds, dus = net_small.predict(X_small)
        assert len(dus) == len(X_small)
        for du in dus:
            assert "nom_fon" in du

    def test_predict_single_sample(self, net_small):
        """predict() fonctionne avec un seul échantillon."""
        x_single = np.array([0.3, 0.6, 0.1, 0.8])
        preds, dus = net_small.predict(x_single)
        assert preds.shape == (1, 1)
        assert len(dus) == 1

    def test_train_reduces_loss(self, X_small, y_small):
        """L'entraînement réduit la perte."""
        net = FaNet(n_features=4, hidden_dim=16, n_outputs=1, seed=0)
        loss_before = net.evaluate(X_small, y_small)["loss"]
        net.train(X_small, y_small, epochs=50, lr=0.01, verbose=False)
        loss_after = net.evaluate(X_small, y_small)["loss"]
        assert loss_after < loss_before, \
            f"La perte n'a pas diminué : {loss_before:.4f} → {loss_after:.4f}"

    def test_train_returns_history(self, net_small, X_small, y_small):
        """train() retourne l'historique de perte."""
        history = net_small.train(X_small, y_small, epochs=5, verbose=False)
        assert "loss" in history
        assert len(history["loss"]) == 5

    def test_evaluate_returns_metrics(self, net_small, X_small, y_small):
        """evaluate() retourne les métriques attendues."""
        metrics = net_small.evaluate(X_small, y_small)
        assert "loss" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "accuracy" in metrics

    def test_save_load_roundtrip(self, net_small, X_small, tmp_path):
        """save() puis load() → prédictions identiques."""
        save_path = str(tmp_path / "fanet_test.npz")
        pred_before = net_small.predict_proba(X_small)
        net_small.save(save_path)

        net_loaded = FaNet.load(save_path)
        pred_after = net_loaded.predict_proba(X_small)

        np.testing.assert_allclose(
            pred_before, pred_after, rtol=1e-5,
            err_msg="Les prédictions avant/après save/load diffèrent"
        )


# ─── T19.6 : Fonctions de perte ───────────────────────────────────────────────
class TestLossFunctions:

    def test_mse_perfect_prediction(self):
        """MSE = 0 si prédiction parfaite."""
        y = np.array([[0.3], [0.7], [0.1]])
        assert mse_loss(y, y) == pytest.approx(0.0)

    def test_mse_gradient_shape(self):
        """Le gradient MSE a la même forme que y."""
        y_pred = np.array([[0.5], [0.3]])
        y_true = np.array([[0.6], [0.4]])
        g = mse_gradient(y_pred, y_true)
        assert g.shape == y_pred.shape

    def test_bce_in_positive_range(self):
        """BCE est toujours positif."""
        y_pred = np.array([[0.3], [0.7], [0.5]])
        y_true = np.array([[0.0], [1.0], [1.0]])
        assert binary_cross_entropy(y_pred, y_true) > 0

    def test_unknown_loss_raises(self):
        """Perte inconnue → ValueError à la création du FaNet."""
        with pytest.raises(ValueError, match="inconnue"):
            FaNet(n_features=3, loss="invalid_loss")
