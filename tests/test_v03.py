# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests unitaires — FaNet v0.3 (PyTorch)
Projet bokono / Matrice-256

Valide le port PyTorch complet : activations, couches, réseau, autograd,
save/load, interprétation Fâ.

Exécuter avec : python -m pytest tests/test_v03.py -v
"""

import pytest
import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.torch_activations import (
    FaReLU, TorchLearnableFaSigma, TorchFaDual, get_torch_activation,
)
from bokono.ml.torch_layers import TorchFaEmbedding, TorchFaHidden, TorchFaDecoder
from bokono.ml.torch_network import TorchFaNet
from bokono.core.fa_matrix import FaMatrix


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fa_tensor():
    return FaMatrix().to_weight_tensor()

@pytest.fixture(scope="module")
def X_np():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, (40, 4))

@pytest.fixture(scope="module")
def y_np(X_np):
    return 0.4 * X_np[:, 0] + 0.3 * X_np[:, 1] + 0.2 * X_np[:, 2]

@pytest.fixture
def net():
    return TorchFaNet(n_features=4, hidden_dim=16, n_outputs=1, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# T_v03.1 : Activations PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaReLUTorch:

    def test_output_shape(self):
        act = FaReLU()
        x = torch.randn(5, 8, dtype=torch.float64)
        y = act(x)
        assert y.shape == x.shape

    def test_output_nonneg(self):
        act = FaReLU()
        x = torch.randn(20, 8, dtype=torch.float64)
        y = act(x)
        assert torch.all(y >= 0)

    def test_autograd_works(self):
        """Le gradient est calculé automatiquement."""
        act = FaReLU()
        x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        y = act(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_four_zones_distinct(self):
        """Les 4 zones produisent des gains différents."""
        act = FaReLU()
        x = torch.tensor([-1.0, 0.03, 0.3, 0.7, 0.95], dtype=torch.float64)
        y = act(x)
        assert y[0] == 0.0        # zone fermée
        assert 0 < y[1] < 0.03    # semi-ouvert : gain 0.25
        assert 0 < y[2] < 0.3     # pré-ouvert : gain 0.5
        assert y[3] == pytest.approx(0.7, abs=1e-6)  # plein ouvert


class TestTorchLearnableFaSigmaTorch:

    def test_output_in_01(self):
        act = TorchLearnableFaSigma()
        x = torch.randn(20, 8, dtype=torch.float64)
        y = act(x)
        assert torch.all(y > 0) and torch.all(y < 1)

    def test_has_nn_parameter(self):
        """raw est un nn.Parameter → autograd le met à jour."""
        act = TorchLearnableFaSigma()
        params = list(act.parameters())
        assert len(params) == 1
        assert params[0].shape == (16,)
        assert params[0].requires_grad

    def test_polarity_in_01(self):
        act = TorchLearnableFaSigma()
        p = act.polarity_weights
        assert torch.all(p > 0) and torch.all(p < 1)

    def test_gain_in_range(self):
        act = TorchLearnableFaSigma()
        g = act.gain_values
        assert torch.all(g > 0.5) and torch.all(g < 1.5)

    def test_autograd_updates_raw(self):
        """Vérification que le gradient passe jusqu'à raw."""
        act = TorchLearnableFaSigma()
        x = torch.randn(10, 8, dtype=torch.float64)
        y = act(x)
        loss = y.sum()
        loss.backward()
        assert act.raw.grad is not None
        assert torch.any(act.raw.grad != 0)


class TestTorchFaDualTorch:

    def test_output_in_01(self):
        act = TorchFaDual()
        x = torch.randn(10, 4, dtype=torch.float64)
        y = act(x)
        assert torch.all(y > 0) and torch.all(y < 1)

    def test_has_parameters(self):
        """FaDual hérite les paramètres de LearnableFaSigma."""
        act = TorchFaDual()
        assert len(list(act.parameters())) > 0


class TestGetTorchActivation:

    def test_get_fa_relu(self):
        act = get_torch_activation("fa_relu")
        assert isinstance(act, FaReLU)

    def test_get_fa_sigma(self):
        act = get_torch_activation("fa_sigma")
        assert isinstance(act, TorchLearnableFaSigma)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_torch_activation("unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# T_v03.2 : Couches PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

class TestTorchFaEmbedding:

    def test_output_shape(self):
        layer = TorchFaEmbedding(n_features=4)
        x = torch.rand(5, 4, dtype=torch.float64)
        out = layer(x)
        assert out.shape == (5, 16)

    def test_values_bounded(self):
        layer = TorchFaEmbedding(n_features=3)
        x = torch.rand(10, 3, dtype=torch.float64)
        out = layer(x)
        assert torch.all(out >= 0) and torch.all(out <= 1.0 + 1e-6)

    def test_differentiable(self):
        """Innovation v0.3 : l'embedding est DIFFÉRENTIABLE."""
        layer = TorchFaEmbedding(n_features=3)
        x = torch.rand(5, 3, dtype=torch.float64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient doit passer à travers l'embedding"
        assert torch.any(x.grad != 0), "Gradient non-nul attendu"

    def test_extreme_values(self):
        layer = TorchFaEmbedding(n_features=2)
        x = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        out = layer(x)
        assert torch.all(torch.isfinite(out))

    def test_has_trainable_params(self):
        """L'embedding Fâ v3.0 a une projection Query entraînable."""
        layer = TorchFaEmbedding(n_features=4)
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        assert trainable > 0, "Query_proj devrait avoir des paramètres entraînables"


class TestTorchFaHidden:

    def test_output_shape(self, fa_tensor):
        layer = TorchFaHidden(16, 32, fa_tensor=fa_tensor)
        x = torch.randn(5, 16, dtype=torch.float64)
        out = layer(x)
        assert out.shape == (5, 32)

    def test_has_weight_bias(self, fa_tensor):
        layer = TorchFaHidden(16, 32, fa_tensor=fa_tensor)
        assert hasattr(layer, "weight")
        assert hasattr(layer, "bias")
        assert layer.weight.shape == (32, 16)

    def test_autograd_through_hidden(self, fa_tensor):
        layer = TorchFaHidden(16, 32, fa_tensor=fa_tensor)
        x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


class TestTorchFaDecoder:

    def test_output_shape(self):
        layer = TorchFaDecoder(32, n_outputs=1)
        x = torch.randn(5, 32, dtype=torch.float64)
        out = layer(x)
        assert out.shape == (5, 1)

    def test_output_in_01(self):
        layer = TorchFaDecoder(32, n_outputs=1)
        x = torch.randn(20, 32, dtype=torch.float64)
        out = layer(x)
        assert torch.all(out > 0) and torch.all(out < 1)

    def test_interpretation(self):
        layer = TorchFaDecoder(32, n_outputs=1)
        x = torch.randn(3, 32, dtype=torch.float64)
        preds, dus = layer.forward_with_interpretation(x)
        assert len(dus) == 3
        for du in dus:
            assert "nom_fon" in du


# ═══════════════════════════════════════════════════════════════════════════════
# T_v03.3 : TorchFaNet complet
# ═══════════════════════════════════════════════════════════════════════════════

class TestTorchFaNet:

    def test_version_is_30(self, net):
        assert net.VERSION == "3.0.0"

    def test_is_nn_module(self, net):
        assert isinstance(net, nn.Module)

    def test_forward_shape(self, net, X_np):
        x_t = torch.tensor(X_np, dtype=torch.float64)
        out = net(x_t)
        assert out.shape == (40, 1)

    def test_forward_in_01(self, net, X_np):
        x_t = torch.tensor(X_np, dtype=torch.float64)
        out = net(x_t)
        assert torch.all(out > 0) and torch.all(out < 1)

    def test_forward_no_nan(self, net, X_np):
        x_t = torch.tensor(X_np, dtype=torch.float64)
        out = net(x_t)
        assert torch.all(torch.isfinite(out))

    def test_summary_runs(self, net, capsys):
        net.summary()
        captured = capsys.readouterr()
        assert "v3.0.0" in captured.out
        assert "PyTorch" in captured.out

    def test_predict_returns_du(self, net, X_np):
        preds, dus = net.predict(X_np)
        assert preds.shape == (40, 1)
        assert len(dus) == 40
        for du in dus:
            assert "nom_fon" in du

    def test_predict_single_sample(self, net):
        x = np.array([0.3, 0.6, 0.1, 0.8])
        preds, dus = net.predict(x)
        assert preds.shape == (1, 1)
        assert len(dus) == 1

    def test_evaluate_returns_metrics(self, net, X_np, y_np):
        metrics = net.evaluate(X_np, y_np)
        assert "loss" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "accuracy" in metrics


class TestTorchFaNetTraining:

    def test_fit_reduces_loss(self, X_np, y_np):
        net = TorchFaNet(n_features=4, hidden_dim=16, seed=0)
        loss_before = net.evaluate(X_np, y_np)["loss"]
        net.fit(X_np, y_np, epochs=50, lr=0.01, verbose=False)
        loss_after = net.evaluate(X_np, y_np)["loss"]
        assert loss_after < loss_before, \
            f"Perte doit diminuer : {loss_before:.4f} → {loss_after:.4f}"

    def test_fit_returns_history(self, X_np, y_np):
        net = TorchFaNet(n_features=4, hidden_dim=16, seed=1)
        history = net.fit(X_np, y_np, epochs=5, verbose=False)
        assert "loss" in history
        assert len(history["loss"]) == 5

    def test_adam_optimizer(self, X_np, y_np):
        net = TorchFaNet(n_features=4, hidden_dim=16, seed=2)
        net.fit(X_np, y_np, epochs=10, optimizer="adam", verbose=False)

    def test_sgd_optimizer(self, X_np, y_np):
        net = TorchFaNet(n_features=4, hidden_dim=16, seed=3)
        net.fit(X_np, y_np, epochs=10, optimizer="sgd", verbose=False)

    def test_unknown_optimizer_raises(self, X_np, y_np):
        net = TorchFaNet(n_features=4, hidden_dim=16)
        with pytest.raises(ValueError, match="inconnu"):
            net.fit(X_np, y_np, epochs=1, optimizer="invalid", verbose=False)


class TestTorchFaNetAutograd:

    def test_full_gradient_flow(self, X_np):
        """Gradient passe de la sortie jusqu'à l'entrée (embedding différentiable)."""
        net = TorchFaNet(n_features=4, hidden_dim=16)
        x = torch.tensor(X_np[:5], dtype=torch.float64, requires_grad=True)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient doit atteindre l'entrée"
        assert torch.any(x.grad != 0), "Gradient non-nul attendu à l'entrée"

    def test_polarity_weights_receive_gradients(self, X_np, y_np):
        """Les poids de polarité Kpoli reçoivent un gradient."""
        net = TorchFaNet(n_features=4, hidden_dim=16)
        x_t = torch.tensor(X_np[:5], dtype=torch.float64)
        y_t = torch.tensor(y_np[:5], dtype=torch.float64).unsqueeze(1)

        out = net(x_t)
        loss = nn.MSELoss()(out, y_t)
        loss.backward()

        # Trouver le paramètre de polarité
        for name, param in net.named_parameters():
            if "raw" in name:
                assert param.grad is not None, \
                    f"Le gradient de {name} ne doit pas être None"
                assert torch.any(param.grad != 0), \
                    f"Le gradient de {name} doit être non-nul"

    def test_all_parameters_have_gradients(self, X_np, y_np):
        """Tous les paramètres reçoivent un gradient après une passe."""
        net = TorchFaNet(n_features=4, hidden_dim=16)
        x_t = torch.tensor(X_np[:5], dtype=torch.float64)
        y_t = torch.tensor(y_np[:5], dtype=torch.float64).unsqueeze(1)

        out = net(x_t)
        loss = nn.MSELoss()(out, y_t)
        loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient None pour {name}"


class TestTorchFaNetSaveLoad:

    def test_save_load_roundtrip(self, net, X_np, tmp_path):
        """save() puis load() → prédictions identiques."""
        pred_before = net.predict_proba(X_np)
        save_path = str(tmp_path / "fanet_v03.pth")
        net.save(save_path)

        net_loaded = TorchFaNet.load(save_path)
        pred_after = net_loaded.predict_proba(X_np)

        np.testing.assert_allclose(
            pred_before, pred_after, rtol=1e-5,
            err_msg="Prédictions avant/après save/load diffèrent"
        )

    def test_load_preserves_polarity(self, X_np, y_np, tmp_path):
        """Les poids de polarité sont préservés après save/load."""
        net = TorchFaNet(n_features=4, hidden_dim=16, seed=0)
        net.fit(X_np, y_np, epochs=10, verbose=False)

        save_path = str(tmp_path / "fanet_trained.pth")
        net.save(save_path)
        net2 = TorchFaNet.load(save_path)

        pol_before = net.hidden.activation.polarity_weights.detach().numpy()
        pol_after = net2.hidden.activation.polarity_weights.detach().numpy()
        np.testing.assert_allclose(pol_before, pol_after, atol=1e-6)
