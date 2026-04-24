# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Démonstration : FaNet v3.0 (Architecture Combinatoire Native) sur un cas RH.
Candidat : Compétent (95%), en bonne santé (90%), mais bloqué au chômage depuis 12 ans.
L'IA trace le chemin causal exact via les 16 Kpoli (Gauche × Droit).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.torch_network import TorchFaNet
from bokono.core.fa_encoder import _get_kb

np.set_printoptions(precision=4, suppress=True)

# Les 16 noms Kpoli fondamentaux (Meji)
KPOLI_NOMS = [
    "GBE", "YEKU", "WOLI", "DI",
    "ABLA", "AKLAN", "TRINMI", "GUDA",
    "SA", "KA", "TULA", "TURUKPE",
    "KANA", "OTURA", "OKANRAN", "IKA",
]

print("=" * 70)
print(" 🧠 FaNet v3.0 — ANALYSE COMBINATOIRE NATIVE (Kpoli G × Kpoli D)")
print("=" * 70)

# 1. Création du mini-dataset RH
np.random.seed(42)
N = 1000
X_train = np.random.rand(N, 3)
y_train = np.where((X_train[:, 0] > 0.4) & (X_train[:, 2] < 0.3), 1.0, 0.0)

# Le candidat bloqué
X_test = np.array([[0.95, 0.90, 0.80]])

print("\n⚙️  Entraînement de FaNet v3.0 sur les données RH du marché...")
net = TorchFaNet(n_features=3, hidden_dim=16, n_outputs=1, seed=42)
net.fit(X_train, y_train, epochs=80, lr=0.01, verbose=False)

# 2. Prédiction avec extraction de la causalité croisée
preds, dus, attn_g, attn_d = net.predict(X_test, return_dual=True)

print(f"\n🔍 Probabilité de trouver un emploi : {preds[0][0]:.2%}")

du = dus[0]
print(f"\n📜 Signe final prédit     : {du.get('nom_fon', 'N/A')}")
print(f"   Polarité               : {du.get('polarite', 'N/A')}")
print(f"   Intensité              : {du.get('intensite', 'N/A')}")

interp = du.get('interpretation', '') or ''
print(f"   Interprétation         : {interp[:150]}...")

prescriptions = du.get('prescriptions', [])
interdictions = du.get('interdictions', [])
print(f"   Prescriptions (Action) : {prescriptions if prescriptions else 'Aucune spécifique'}")
print(f"   Interdictions          : {interdictions if interdictions else 'Aucune spécifique'}")

# 3. Radiographie Combinatoire (Le vrai apport v3.0)
feature_names = ["Compétence", "Santé", "Années chômage"]

print("\n" + "=" * 70)
print(" 👁️  RADIOGRAPHIE COMBINATOIRE (Le Cerveau du Bokonon)")
print("=" * 70)

for feat_idx, feat_name in enumerate(feature_names):
    g = attn_g[0, feat_idx, :]  # (16,)
    d = attn_d[0, feat_idx, :]  # (16,)

    top_g = np.argsort(g)[::-1][:3]
    top_d = np.argsort(d)[::-1][:3]

    print(f"\n  Feature : {feat_name}")
    print(f"  ┌─ Kpoli GAUCHE (Force endogène) :")
    for i, idx in enumerate(top_g):
        print(f"  │   {i+1}. {KPOLI_NOMS[idx]:<10} → {g[idx]*100:05.2f}%")
    print(f"  └─ Kpoli DROIT (Force exogène) :")
    for i, idx in enumerate(top_d):
        print(f"      {i+1}. {KPOLI_NOMS[idx]:<10} → {d[idx]*100:05.2f}%")

    # Le signe combiné dominant
    combined = np.outer(g, d)  # (16, 16)
    flat_idx = np.argmax(combined)
    g_idx, d_idx = divmod(flat_idx, 16)
    signe_idx = g_idx * 16 + d_idx
    kb = _get_kb()
    signe = kb[signe_idx]
    print(f"  ══> Combinaison dominante : {KPOLI_NOMS[g_idx]} × {KPOLI_NOMS[d_idx]} = {signe.get('nom_fon', 'N/A')} (ID:{signe_idx})")

print("\n" + "=" * 70)
print(" ✅ FaNet v3.0 — Architecture Combinatoire Native Fâ (16×16→256)")
print("=" * 70)
