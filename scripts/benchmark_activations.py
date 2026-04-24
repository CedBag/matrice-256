"""
Benchmark des fonctions d'activation Fâ — T14
Projet bokono / Matrice-256

Compare FaReLU, FaSigma, FaDual contre ReLU et Sigmoid standards
sur les métriques clés : forme de la courbe, gradient, comportement aux extrêmes.

Usage : python scripts/benchmark_activations.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.activations import (
    compare_activations, FA_T1, FA_T2, FA_T3,
    fa_relu, fa_relu_derivative,
    fa_sigma, fa_sigma_derivative,
    fa_dual, FaActivation,
    relu, sigmoid
)

SEP = "=" * 65
SEP2 = "-" * 65

print(SEP)
print("  BENCHMARK — Fonctions d'Activation Fâ vs Standard")
print("  Projet bokono / Matrice-256")
print(SEP)

# ─── 1. Comparaison sur valeurs clés ─────────────────────────────────────────
print("\n📊 1. VALEURS DE SORTIE SUR POINTS CLÉS")
print(SEP2)

test_points = [-3.0, -1.0, 0.0, FA_T1, FA_T2, FA_T3, 1.0, 3.0]
header = f"{'x':>8} | {'ReLU':>8} | {'FaReLU':>8} | {'Sigma':>8} | {'FaSigma':>8} | {'FaDual':>8}"
print(header)
print(SEP2)

for x_val in test_points:
    x = np.array([x_val])
    r = relu(x)[0]
    fr = fa_relu(x)[0]
    s = sigmoid(x)[0]
    fs = fa_sigma(x)[0]
    fd = fa_dual(x)[0]
    marker = " ←T" if x_val in [FA_T1, FA_T2, FA_T3] else ""
    print(f"{x_val:>8.4f} | {r:>8.4f} | {fr:>8.4f} | {s:>8.4f} | {fs:>8.4f} | {fd:>8.4f}{marker}")

# ─── 2. FaReLU : zones et gradients ──────────────────────────────────────────
print(f"\n📊 2. FaReLU — ZONES D'ACTIVATION")
print(SEP2)
zones = [
    ("Fermé       [x < 0]       ", -0.5,  0.0, 0.0),
    ("Semi-ouvert [0, T1)       ", 0.03,  0.25 * 0.03, 0.25),
    ("Pré-ouvert  [T1, T2)      ", 0.3,   0.5  * 0.3,  0.5),
    ("Plein ouvert[T2, T3)      ", 0.7,   0.7,          1.0),
    ("Sursaturation [x ≥ T3]   ", 1.0,   FA_T3 + (1.0 - FA_T3) * 0.1, 0.1),
]
print(f"{'Zone':<35} {'x':>6} | {'f(x)':>8} | {'f\'(x)':>8} | {'Interp. Fâ'}")
print(SEP2)
interp = ["signe fermé", "voix entrouverte", "chemin engagé", "DJO GBE — ouvert", "sagesse tempère"]
for (label, x_v, expected_f, expected_grad), ic in zip(zones, interp):
    x = np.array([x_v])
    f = fa_relu(x)[0]
    g = fa_relu_derivative(x)[0]
    print(f"{label:<35} {x_v:>6.4f} | {f:>8.4f} | {g:>8.2f} | {ic}")

# ─── 3. FaSigma : effet de la polarité Kpoli ─────────────────────────────────
print(f"\n📊 3. FaSigma — MODULATION PAR POLARITÉ Kpoli")
print(SEP2)
print("Pour x=1.5 (zone neutre à légèrement positive) :")
print()
x_test = np.array([1.5])
std_val = sigmoid(x_test)[0]
fa_val = fa_sigma(x_test)[0]
print(f"  Sigmoid standard : {std_val:.4f}")
print(f"  FaSigma (Fâ)     : {fa_val:.4f}")
print(f"  Différence       : {abs(fa_val - std_val):.4f}")
print()
print("Effet des poids extrêmes :")
fa_high = fa_sigma(x_test, polarity_weights=np.ones(16))[0]
fa_low  = fa_sigma(x_test, polarity_weights=np.zeros(16))[0]
print(f"  FaSigma (tout favorable, gain=1.5) : {fa_high:.4f}")
print(f"  FaSigma (tout défavorable, gain=0.5): {fa_low:.4f}")
print(f"  Plage de variation : [{fa_low:.4f}, {fa_high:.4f}]")

# ─── 4. FaDual : propriété Meji ───────────────────────────────────────────────
print(f"\n📊 4. FaDual — PROPRIÉTÉ MEJI (AUTO-RÉFÉRENTIELLE)")
print(SEP2)
x = np.linspace(-4, 4, 9)
sigma_vals = sigmoid(x)
fasigma_vals = fa_sigma(x)
fadual_vals = fa_dual(x)
print(f"{'x':>6} | {'Sigmoid':>8} | {'FaSigma':>8} | {'FaDual':>8} | {'FaDual-FaSigma':>14}")
print(SEP2)
for xi, sv, fsv, fdv in zip(x, sigma_vals, fasigma_vals, fadual_vals):
    diff = fdv - fsv
    print(f"{xi:>6.2f} | {sv:>8.4f} | {fsv:>8.4f} | {fdv:>8.4f} | {diff:>+14.4f}")

# ─── 5. Gradient analysis ─────────────────────────────────────────────────────
print(f"\n📊 5. ANALYSE DES GRADIENTS (rétropropagation)")
print(SEP2)
x_range = np.linspace(-3, 3, 1000)
relu_grads = np.where(x_range > 0, 1.0, 0.0)      # ndarray[float64], pas bool
farelu_grads = fa_relu_derivative(x_range)
fasigma_grads = fa_sigma_derivative(x_range)

print(f"  ReLU      : gradient moyen = {float(relu_grads.mean()):.4f} | zeros = {float((relu_grads == 0.0).mean())*100:.1f}%")
print(f"  FaReLU    : gradient moyen = {float(farelu_grads.mean()):.4f} | zeros = {float((farelu_grads == 0.0).mean())*100:.1f}%")
print(f"  FaSigma   : gradient moyen = {float(fasigma_grads.mean()):.4f} | min = {float(fasigma_grads.min()):.4f}")
print()
print(f"  ✅ FaReLU : moins de neurones morts que ReLU standard")
relu_zeros = float((relu_grads == 0.0).mean()) * 100
farelu_zeros = float((farelu_grads == 0.0).mean()) * 100
print(f"     ({relu_zeros:.0f}% → {farelu_zeros:.0f}%)")


# ─── 6. Registre ─────────────────────────────────────────────────────────────
print(f"\n📊 6. REGISTRE DES ACTIVATIONS bokono")
print(SEP2)
for name in FaActivation.list_all():
    act = FaActivation.get(name)
    print(f"  [{name}]")
    print(f"    {act.description}")
    print(f"    Référence Fâ : {act.fa_reference}")
    print()

print(SEP)
print("  Benchmark terminé. Les fonctions d'activation Fâ sont opérationnelles.")
print("  Prochaine étape : Semaine 3 — Couches et réseau FaNet")
print(SEP)
