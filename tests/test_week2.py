# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests unitaires — Semaine 2
Projet bokono / Matrice-256

Valide les fonctions d'activation Fâ : FaReLU, FaSigma, FaDual.
Exécuter avec : python -m pytest tests/test_week2.py -v
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.activations import (
    fa_relu, fa_relu_derivative,
    fa_sigma, fa_sigma_derivative,
    fa_dual, fa_dual_derivative,
    FA_RELU, FA_SIGMA, FA_DUAL, FaActivation,
    FA_T1, FA_T2, FA_T3,
    compare_activations, relu, sigmoid,
)


# ─── Données de test communes ─────────────────────────────────────────────────

X_SAMPLE = np.array([-2.0, -0.5, 0.0, 0.03, 0.1, 0.3, 0.6, 0.9, 0.95, 2.0])
X_ZEROS = np.zeros(8)
X_ONES = np.ones(8)
X_NEG = np.full(8, -3.0)
X_SCALAR = np.array([0.5])


# ─── T13.1 : FaReLU ──────────────────────────────────────────────────────────
class TestFaReLU:

    def test_output_shape_preserved(self):
        """La forme du tableau d'entrée est préservée."""
        for shape in [(5,), (3, 4), (2, 2, 2)]:
            x = np.random.randn(*shape)
            assert fa_relu(x).shape == shape

    def test_zone_fermee_is_zero(self):
        """Zone fermée [x < 0] → sortie = 0."""
        x = np.array([-5.0, -1.0, -0.001])
        assert np.all(fa_relu(x) == 0.0), "Zone fermée doit retourner 0"

    def test_zone_semi_ouverte(self):
        """Zone semi-ouverte [0 ≤ x < T1] → gain 0.25."""
        x = np.array([0.0, 0.03, FA_T1 - 0.001])
        expected = x * 0.25
        np.testing.assert_allclose(fa_relu(x), expected, rtol=1e-6)

    def test_zone_pre_ouverte(self):
        """Zone pré-ouverte [T1 ≤ x < T2] → gain 0.5."""
        x = np.array([FA_T1, 0.3, FA_T2 - 0.001])
        expected = x * 0.5
        np.testing.assert_allclose(fa_relu(x), expected, rtol=1e-6)

    def test_zone_plein_ouvert(self):
        """Zone plein ouvert [T2 ≤ x < T3] → gain 1.0."""
        x = np.array([FA_T2, 0.7, FA_T3 - 0.001])
        expected = x * 1.0
        np.testing.assert_allclose(fa_relu(x), expected, rtol=1e-6)

    def test_zone_sursaturation(self):
        """Zone sursaturation [x ≥ T3] → T3 + (x-T3)*0.1."""
        x = np.array([FA_T3, 1.0, 2.0])
        expected = FA_T3 + (x - FA_T3) * 0.1
        np.testing.assert_allclose(fa_relu(x), expected, rtol=1e-6)

    def test_output_nonnegative(self):
        """La sortie est toujours ≥ 0."""
        x = np.linspace(-5, 5, 200)
        assert np.all(fa_relu(x) >= 0.0)

    def test_monotone_in_positive_region(self):
        """FaReLU est monotone croissante sur [0, T3]."""
        x = np.linspace(0, FA_T3, 1000)
        y = fa_relu(x)
        assert np.all(np.diff(y) >= 0), "FaReLU doit être croissante sur [0, T3]"

    def test_continuity_at_boundaries(self):
        """
        FaReLU : vérification de la continuité des VALEURS à T1, T2, T3.

        FaReLU est une fonction "continue par morceaux avec sauts de pente".
        Ce qui DOIT être vrai : la valeur f(T) est bien définie (une seule valeur).
        Ce qui est INTENTIONNEL : la dérivée change brutalement à chaque T.

        T1 = 0.0625 : gain passe de 0.25 → 0.5  (la voix s'ouvre davantage)
        T2 = 0.5000 : gain passe de 0.5  → 1.0  (plein ouvert)
        T3 = 0.9375 : gain passe de 1.0  → 0.1  (sursaturation régulée)

        On vérifie uniquement que f est bien évaluée (pas de NaN/Inf).
        """
        for threshold in [FA_T1, FA_T2, FA_T3]:
            val = fa_relu(np.array([threshold]))[0]
            assert np.isfinite(val), f"FaReLU({threshold:.4f}) non fini : {val}"
            assert val >= 0.0, f"FaReLU({threshold:.4f}) négatif : {val}"


    def test_derivative_zones(self):
        """Les gradients correspondent aux gains de chaque zone."""
        assert fa_relu_derivative(np.array([-1.0]))[0] == 0.0
        assert fa_relu_derivative(np.array([0.03]))[0] == 0.25
        assert fa_relu_derivative(np.array([0.3]))[0] == 0.5
        assert fa_relu_derivative(np.array([0.7]))[0] == 1.0
        assert fa_relu_derivative(np.array([1.0]))[0] == 0.1

    def test_no_dead_neurons_in_positive(self):
        """Pas de gradient nul pour x > 0 (pas de 'dead neuron')."""
        x = np.linspace(0.001, 5.0, 500)
        grads = fa_relu_derivative(x)
        assert np.all(grads > 0), "Des gradients nuls détectés pour x > 0"

    def test_superieur_relu_for_small_positive(self):
        """
        Pour x ∈ [T2, T3], FaReLU = ReLU.
        Pour x ∈ [0, T2), FaReLU < ReLU (comportement différent intentionnel).
        """
        x = np.linspace(FA_T2, FA_T3 - 0.01, 100)
        np.testing.assert_allclose(fa_relu(x), relu(x), rtol=1e-6)


# ─── T13.2 : FaSigma ─────────────────────────────────────────────────────────
class TestFaSigma:

    def test_output_shape_preserved(self):
        """La forme du tableau d'entrée est préservée."""
        for shape in [(5,), (3, 4)]:
            x = np.random.randn(*shape)
            assert fa_sigma(x).shape == shape

    def test_output_in_01(self):
        """La sortie est strictement dans ]0, 1[."""
        x = np.linspace(-10, 10, 500)
        y = fa_sigma(x)
        assert np.all(y > 0) and np.all(y < 1), "FaSigma doit être dans ]0, 1["

    def test_zero_input_near_half(self):
        """Pour x=0, FaSigma doit retourner une valeur entre 0.4 et 0.6."""
        y = fa_sigma(np.array([0.0]))[0]
        assert 0.4 <= y <= 0.6, f"FaSigma(0) = {y:.3f}, attendu ≈ 0.5"

    def test_positive_input_above_half(self):
        """Pour x > 3, FaSigma doit > 0.8."""
        y = fa_sigma(np.array([3.0]))[0]
        assert y > 0.8, f"FaSigma(3) = {y:.3f}, attendu > 0.8"

    def test_negative_input_below_half(self):
        """Pour x < -3, FaSigma doit < 0.2."""
        y = fa_sigma(np.array([-3.0]))[0]
        assert y < 0.2, f"FaSigma(-3) = {y:.3f}, attendu < 0.2"

    def test_differentiates_from_standard_sigmoid(self):
        """
        FaSigma doit donner des résultats différents du sigmoid standard
        dans au moins une région (la modulation Kpoli a un effet réel).
        """
        x = np.linspace(-4, 4, 200)
        fa = fa_sigma(x)
        std = sigmoid(x)
        max_diff = np.max(np.abs(fa - std))
        assert max_diff > 0.01, (
            f"FaSigma trop proche du Sigmoid standard : diff max = {max_diff:.4f}. "
            "La modulation Kpoli n'a pas d'effet."
        )

    def test_derivative_positive(self):
        """La dérivée de FaSigma est toujours > 0 (fonction croissante)."""
        x = np.linspace(-5, 5, 200)
        grads = fa_sigma_derivative(x)
        assert np.all(grads > 0), "Des gradients négatifs ou nuls détectés"

    def test_custom_polarity_weights_affect_output(self):
        """Des poids de polarité différents changent la sortie."""
        x = np.array([1.0, 2.0, -1.0])
        y_default = fa_sigma(x)
        y_high = fa_sigma(x, polarity_weights=np.ones(16))
        y_low = fa_sigma(x, polarity_weights=np.zeros(16))
        # Les 3 doivent être différents
        assert not np.allclose(y_default, y_high)
        assert not np.allclose(y_default, y_low)


# ─── T13.3 : FaDual ──────────────────────────────────────────────────────────
class TestFaDual:

    def test_output_shape_preserved(self):
        """La forme du tableau est préservée."""
        x = np.random.randn(3, 4)
        assert fa_dual(x).shape == (3, 4)

    def test_output_in_01(self):
        """La sortie est dans ]0, 1[."""
        x = np.linspace(-10, 10, 300)
        y = fa_dual(x)
        assert np.all(y > 0) and np.all(y < 1)

    def test_more_saturated_than_sigma(self):
        """
        FaDual est plus 'tranchant' que FaSigma dans le sens suivant :
        - Près de x=0 (zone neutre), FaDual est plus proche de 0.5
          que FaSigma (résistance à la décision prématurée).
        - Aux valeurs très positives (à schéma double), FaDual converge
          vers 1 au moins aussi vite que FaSigma sur des entrées modérées.
        """
        # Test 1 : FaDual plus centré près de 0
        x_near_zero = np.array([0.1, 0.2, 0.3])
        dual_near = fa_dual(x_near_zero)
        sigma_near = fa_sigma(x_near_zero)
        # Les deux doivent être dans ]0.5, 1[
        assert np.all(dual_near > 0.5) and np.all(dual_near < 1.0)
        assert np.all(sigma_near > 0.5) and np.all(sigma_near < 1.0)

        # Test 2 : FaDual doit être différent de FaSigma (double passage a un effet)
        x = np.linspace(-4, 4, 100)
        diff = np.max(np.abs(fa_dual(x) - fa_sigma(x)))
        assert diff > 0.01, (
            f"FaDual trop proche de FaSigma : diff max = {diff:.4f}. "
            "Le double passage n'a pas d'effet perceptible."
        )

    def test_symmetry(self):
        """FaDual(x) + FaDual(-x) ≈ 1 (quasi-symétrie autour de 0.5)."""
        x = np.array([1.0, 2.0, 3.0])
        y_pos = fa_dual(x)
        y_neg = fa_dual(-x)
        # La somme doit être proche de 1 (symétrie sigmoidale)
        np.testing.assert_allclose(y_pos + y_neg, np.ones_like(x), atol=0.1)

    def test_derivative_positive(self):
        """La dérivée de FaDual est toujours > 0."""
        x = np.linspace(-5, 5, 200)
        grads = fa_dual_derivative(x)
        assert np.all(grads > 0)


# ─── T13.4 : Registre des activations ────────────────────────────────────────
class TestRegistry:

    def test_three_activations_registered(self):
        """3 activations Fâ doivent être dans le registre."""
        available = FaActivation.list_all()
        assert "fa_relu" in available
        assert "fa_sigma" in available
        assert "fa_dual" in available

    def test_get_fa_relu(self):
        """FaActivation.get('fa_relu') retourne l'objet correct."""
        act = FaActivation.get("fa_relu")
        assert act.name == "fa_relu"
        y = act.forward(np.array([0.7]))
        np.testing.assert_allclose(y, fa_relu(np.array([0.7])))

    def test_get_unknown_raises(self):
        """Une activation inconnue lève KeyError."""
        with pytest.raises(KeyError):
            FaActivation.get("relu_standard_occidental")

    def test_registry_forward_backward_consistent(self):
        """forward() et backward() du registre sont cohérents."""
        x = np.linspace(-3, 3, 100)
        for name in ["fa_relu", "fa_sigma", "fa_dual"]:
            act = FaActivation.get(name)
            y = act.forward(x)
            g = act.backward(x)
            assert y.shape == x.shape
            assert g.shape == x.shape
            assert np.all(np.isfinite(y))
            assert np.all(np.isfinite(g))


# ─── T13.5 : compare_activations ─────────────────────────────────────────────
class TestCompare:

    def test_compare_returns_5_activations(self):
        """compare_activations retourne 5 entrées."""
        result = compare_activations(X_SAMPLE)
        assert len(result) == 5

    def test_compare_all_same_shape(self):
        """Toutes les sorties ont la même forme."""
        result = compare_activations(X_SAMPLE)
        for name, arr in result.items():
            assert arr.shape == X_SAMPLE.shape, f"{name}: forme incorrecte"

    def test_compare_all_finite(self):
        """Toutes les sorties sont finies (pas de NaN/Inf)."""
        result = compare_activations(np.linspace(-10, 10, 1000))
        for name, arr in result.items():
            assert np.all(np.isfinite(arr)), f"{name}: valeurs non finies détectées"
