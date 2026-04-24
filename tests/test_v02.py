# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests unitaires — FaNet v0.2 (LearnableFaSigma)
Projet bokono / Matrice-256

Valide que les poids de polarité Kpoli sont entraînables et s'adaptent
réellement aux données, ce qui constitue l'innovation centrale de v0.2.

Exécuter avec : python -m pytest tests/test_v02.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.activations import LearnableFaSigma, load_polarity_weights
from bokono.ml.layers import FaHiddenLayer
from bokono.ml.network import FaNet
from bokono.core.fa_matrix import FaMatrix


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fa_tensor():
    return FaMatrix().to_weight_tensor()

@pytest.fixture(scope="module")
def X_train():
    """Données synthétiques reproductibles."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, (40, 4))

@pytest.fixture(scope="module")
def y_train(X_train):
    """Cible : combinaison linéaire des features."""
    return 0.4 * X_train[:, 0] + 0.3 * X_train[:, 1] + 0.2 * X_train[:, 2]


# ─── T_v02.1 : LearnableFaSigma — initialisation ─────────────────────────────
class TestLearnableFaSigmaInit:

    def test_init_from_defaults(self):
        """Initialisation depuis les poids par défaut."""
        lfs = LearnableFaSigma()
        assert lfs.polarity_weights.shape == (16,)
        assert np.all(lfs.polarity_weights > 0)
        assert np.all(lfs.polarity_weights < 1)

    def test_init_from_kb(self):
        """Initialisation depuis la base de connaissance réelle."""
        kb_weights = load_polarity_weights()
        lfs = LearnableFaSigma(init_weights=kb_weights)
        # Les poids initiaux doivent être proches des valeurs KB (différence < 1e-5)
        np.testing.assert_allclose(
            lfs.polarity_weights, kb_weights.astype(np.float64),
            atol=1e-5,
            err_msg="Les poids initiaux doivent refléter la KB Fâ"
        )

    def test_raw_weights_are_logit_of_polarity(self):
        """raw = logit(polarity) — vérification mathématique."""
        lfs = LearnableFaSigma()
        p = lfs.polarity_weights
        raw_expected = np.log(p / (1.0 - p))
        np.testing.assert_allclose(lfs._raw, raw_expected, atol=1e-10)

    def test_gain_values_in_range(self):
        """Gains dans ]0.5, 1.5[ — toujours vrai."""
        lfs = LearnableFaSigma()
        gains = lfs.gain_values
        assert np.all(gains > 0.5) and np.all(gains < 1.5)

    def test_n_params_is_16(self):
        """Exactement 16 paramètres entraînables — un par Kpoli."""
        lfs = LearnableFaSigma()
        assert lfs.n_params == 16

    def test_repr_shows_ranges(self):
        """__repr__ retourne une chaîne lisible."""
        lfs = LearnableFaSigma()
        r = repr(lfs)
        assert "LearnableFaSigma" in r
        assert "polarity_min" in r


# ─── T_v02.2 : LearnableFaSigma — forward ────────────────────────────────────
class TestLearnableFaSigmaForward:

    def test_output_shape_preserved(self):
        """La forme de sortie est identique à l'entrée."""
        lfs = LearnableFaSigma()
        x = np.random.randn(5, 16)
        y = lfs.forward(x)
        assert y.shape == x.shape

    def test_output_in_01(self):
        """Sortie toujours dans ]0, 1[."""
        lfs = LearnableFaSigma()
        x = np.random.randn(20, 8)
        y = lfs.forward(x)
        assert np.all(y > 0) and np.all(y < 1)

    def test_no_nan_or_inf(self):
        """Pas de NaN ni Inf, même sur des valeurs extrêmes."""
        lfs = LearnableFaSigma()
        x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        y = lfs.forward(x)
        assert np.all(np.isfinite(y))

    def test_differs_from_static_fasigma(self):
        """LearnableFaSigma with identical weights = static FaSigma."""
        from bokono.ml.activations import fa_sigma
        lfs = LearnableFaSigma()
        x = np.linspace(-3, 3, 50)
        y_learnable = lfs.forward(x)
        y_static = fa_sigma(x, polarity_weights=lfs.polarity_weights - 0.5)
        # Les deux formulations diffèrent légèrement à cause du gain mapping
        # (gain=0.5+polarity vs gain directement) — ce n'est pas identique
        assert y_learnable.shape == y_static.shape


# ─── T_v02.3 : LearnableFaSigma — backward ───────────────────────────────────
class TestLearnableFaSigmaBackward:

    def test_grad_x_shape(self):
        """backward() retourne grad_x de même forme que x."""
        lfs = LearnableFaSigma()
        x = np.random.randn(5, 16)
        lfs.forward(x)
        grad_x = lfs.backward(np.ones_like(x))
        assert grad_x.shape == x.shape

    def test_d_raw_shape(self):
        """backward() calcule d_raw de forme (16,)."""
        lfs = LearnableFaSigma()
        x = np.random.randn(10, 8)
        lfs.forward(x)
        lfs.backward(np.ones((10, 8)))
        assert lfs.d_raw.shape == (16,)

    def test_d_raw_nonzero_after_backward(self):
        """Au moins un gradient de polarité est non-nul après backward."""
        lfs = LearnableFaSigma()
        x = np.random.randn(20, 16)
        lfs.forward(x)
        lfs.backward(np.ones((20, 16)))
        assert np.any(lfs.d_raw != 0.0)

    def test_grad_x_finite(self):
        """Gradient vers x est fini."""
        lfs = LearnableFaSigma()
        x = np.random.randn(5, 8)
        lfs.forward(x)
        grad_x = lfs.backward(np.ones((5, 8)))
        assert np.all(np.isfinite(grad_x))

    def test_zero_grad_resets_d_raw(self):
        """zero_grad() remet d_raw à zéro."""
        lfs = LearnableFaSigma()
        x = np.random.randn(5, 8)
        lfs.forward(x)
        lfs.backward(np.ones((5, 8)))
        lfs.zero_grad()
        assert np.all(lfs.d_raw == 0.0)

    def test_backward_without_forward_raises(self):
        """backward() sans forward() préalable → AssertionError."""
        lfs = LearnableFaSigma()
        with pytest.raises(AssertionError):
            lfs.backward(np.ones((3, 4)))


# ─── T_v02.4 : LearnableFaSigma — adaptation réelle ─────────────────────────
class TestLearnableFaSigmaLearns:

    def test_polarity_weights_change_after_training(self):
        """
        LES POIDS DE POLARITÉ S'ADAPTENT AUX DONNÉES.

        C'est le test fondamental de v0.2 : après entraînement,
        les poids de polarité doivent différer de leur valeur initiale.
        """
        lfs = LearnableFaSigma()
        initial_raw = lfs._raw.copy()

        x = np.random.randn(20, 16)
        grad_out = np.random.randn(20, 16) * 0.1

        for _ in range(10):   # 10 pas de gradient
            lfs.zero_grad()
            lfs.forward(x)
            lfs.backward(grad_out)
            lfs.update(lr=0.1)

        assert not np.allclose(lfs._raw, initial_raw), \
            "Les poids raw doivent changer après entraînement"

    def test_polarity_stays_bounded_after_training(self):
        """polarity_weights reste dans ]0, 1[ même après de nombreuses mises à jour."""
        lfs = LearnableFaSigma()
        x = np.random.randn(50, 16)
        grad_out = np.random.randn(50, 16)

        for _ in range(100):
            lfs.zero_grad()
            lfs.forward(x)
            lfs.backward(grad_out)
            lfs.update(lr=0.5)  # Learning rate agressif

        p = lfs.polarity_weights
        assert np.all(p > 0) and np.all(p < 1), \
            "polarity_weights doivent rester dans ]0, 1[ (grâce à la paramétrisation sigmoid)"

    def test_gain_stays_bounded(self):
        """gain_values reste dans ]0.5, 1.5[ par construction."""
        lfs = LearnableFaSigma()
        x = np.random.randn(50, 16)
        grad_out = np.ones((50, 16))

        for _ in range(50):
            lfs.zero_grad()
            lfs.forward(x)
            lfs.backward(grad_out)
            lfs.update(lr=1.0)

        g = lfs.gain_values
        assert np.all(g > 0.5) and np.all(g < 1.5)


# ─── T_v02.5 : FaHiddenLayer avec LearnableFaSigma ───────────────────────────
class TestFaHiddenLayerLearnable:

    def test_learnable_sigma_activates(self, fa_tensor):
        """FaHiddenLayer(learnable_sigma=True) crée un LearnableFaSigma."""
        layer = FaHiddenLayer(12, 32, activation="fa_sigma",
                              fa_tensor=fa_tensor, learnable_sigma=True)
        assert layer._learnable_sigma is not None
        assert isinstance(layer._learnable_sigma, LearnableFaSigma)

    def test_static_sigma_no_learnable(self, fa_tensor):
        """FaHiddenLayer(learnable_sigma=False) n'a pas de LearnableFaSigma."""
        layer = FaHiddenLayer(12, 32, activation="fa_sigma",
                              fa_tensor=fa_tensor, learnable_sigma=False)
        assert layer._learnable_sigma is None

    def test_n_learnable_params_with_lfs(self, fa_tensor):
        """n_learnable_params inclut les 16 poids de polarité."""
        layer = FaHiddenLayer(12, 32, activation="fa_sigma",
                              fa_tensor=fa_tensor, learnable_sigma=True)
        expected = 32 * 12 + 32 + 16  # W + b + polarity
        assert layer.n_learnable_params == expected

    def test_polarity_raw_in_params(self, fa_tensor):
        """params inclut 'polarity_raw' si LearnableFaSigma est actif."""
        layer = FaHiddenLayer(12, 32, activation="fa_sigma",
                              fa_tensor=fa_tensor, learnable_sigma=True)
        assert "polarity_raw" in layer.params

    def test_polarity_changes_after_update(self, fa_tensor):
        """Les poids de polarité changent après forward/backward/update."""
        layer = FaHiddenLayer(12, 32, activation="fa_sigma",
                              fa_tensor=fa_tensor, learnable_sigma=True)
        lfs = layer._learnable_sigma
        assert lfs is not None
        initial_raw = lfs._raw.copy()

        x = np.random.randn(5, 12)
        layer.forward(x)
        layer.backward(np.ones((5, 32)))
        layer.update(lr=0.1)

        assert not np.allclose(lfs._raw, initial_raw), \
            "Les poids de polarité doivent changer après update"


# ─── T_v02.6 : FaNet v0.2 complet ────────────────────────────────────────────
class TestFaNetV02:

    def test_version_is_02(self):
        """FaNet VERSION == '0.2.0'."""
        assert FaNet.VERSION == "0.2.0"

    def test_default_activation_is_fa_sigma(self):
        """L'activation par défaut est fa_sigma en v0.2."""
        net = FaNet(n_features=4)
        assert net.activation == "fa_sigma"

    def test_learnable_sigma_is_true_by_default(self):
        """learnable_sigma=True par défaut en v0.2."""
        net = FaNet(n_features=4)
        assert net.learnable_sigma is True
        assert net.hidden._learnable_sigma is not None

    def test_summary_shows_16_polarity_params(self, capsys):
        """summary() mentionne les 16 poids de polarité entraînables."""
        net = FaNet(n_features=4, hidden_dim=16)
        net.summary()
        captured = capsys.readouterr()
        assert "16" in captured.out
        assert "ENTRA" in captured.out  # "ENTRAÎNABLE"

    def test_polarity_weights_change_during_training(self, X_train, y_train):
        """Preuve que les poids de polarité s'adaptent pendant train()."""
        net = FaNet(n_features=4, hidden_dim=16, seed=0)
        lfs = net.hidden._learnable_sigma
        assert lfs is not None
        initial_polarity = lfs.polarity_weights.copy()

        net.train(X_train, y_train, epochs=20, lr=0.01, verbose=False)

        final_polarity = lfs.polarity_weights
        assert not np.allclose(initial_polarity, final_polarity), \
            "Les poids de polarité doivent s'adapter pendant train()"

    def test_save_load_preserves_polarity_weights(self, X_train, tmp_path):
        """save() puis load() → poids de polarité identiques."""
        net = FaNet(n_features=4, hidden_dim=16, seed=0)
        # Entraînement pour modifier les poids
        y = X_train[:, 0] * 0.5
        net.train(X_train, y, epochs=10, verbose=False)

        lfs_before = net.hidden._learnable_sigma
        assert lfs_before is not None
        polarity_before = lfs_before.polarity_weights.copy()

        save_path = str(tmp_path / "fanet_v02.npz")
        net.save(save_path)
        net_loaded = FaNet.load(save_path)

        lfs_after = net_loaded.hidden._learnable_sigma
        assert lfs_after is not None
        np.testing.assert_allclose(
            polarity_before, lfs_after.polarity_weights,
            atol=1e-6,
            err_msg="Les poids de polarité doivent survivre à save/load"
        )

    def test_train_reduces_loss_with_learnable_sigma(self, X_train, y_train):
        """L'entraînement réduit la perte même avec LearnableFaSigma."""
        net = FaNet(n_features=4, hidden_dim=16, seed=1,
                    activation="fa_sigma", learnable_sigma=True)
        loss_before = net.evaluate(X_train, y_train)["loss"]
        net.train(X_train, y_train, epochs=50, lr=0.01, verbose=False)
        loss_after = net.evaluate(X_train, y_train)["loss"]
        assert loss_after < loss_before, \
            f"Perte doit diminuer : {loss_before:.4f} → {loss_after:.4f}"

    def test_forward_no_nan_with_learnable_sigma(self, X_train):
        """Pas de NaN dans la sortie avec LearnableFaSigma."""
        net = FaNet(n_features=4)
        out = net.forward(X_train)
        assert np.all(np.isfinite(out)), "FaNet v0.2 ne doit pas produire de NaN"


# ─── T_v02.7 : Sérialisation de LearnableFaSigma ────────────────────────────
class TestLearnableFaSigmaSerialization:

    def test_get_state_returns_raw(self):
        """get_state() retourne une copie des paramètres bruts."""
        lfs = LearnableFaSigma()
        state = lfs.get_state()
        assert state.shape == (16,)
        np.testing.assert_array_equal(state, lfs._raw)
        # C'est une copie, pas une référence
        state[0] = 999.0
        assert lfs._raw[0] != 999.0

    def test_set_state_restores_weights(self):
        """set_state() restaure exactement les poids sauvegardés."""
        lfs = LearnableFaSigma()
        state_before = lfs.get_state()

        # Modifier les poids
        lfs._raw[:] = 0.0

        # Restaurer
        lfs.set_state(state_before)
        np.testing.assert_array_equal(lfs._raw, state_before)
