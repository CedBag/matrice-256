"""
Démonstration : FaNet v2.0 (Soft-Lookup Attention) sur un cas RH Complexe.
Cas du candidat : Compétent, en bonne santé, mais bloqué au chômage depuis 12 ans.
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.ml.torch_network import TorchFaNet
from bokono.core.fa_encoder import _get_kb

np.set_printoptions(precision=4, suppress=True)

print("="*70)
print(" 🧠 ANALYSE NEURO-SYMBOLIQUE (FaNet v2.0) - BLOCAGE PROFESSIONNEL")
print("="*70)

# 1. Création d'un mini-dataset RH (1000 échantillons) pour apprendre la logique
# Features : [competence, sante, annees_chomage (normé sur 15 ans)]
np.random.seed(42)
N = 1000
X_train = np.random.rand(N, 3)

# Logique de la société (Ground Truth) :
# Pour réussir, il faut de la compétence (>0.4) ET peu d'années de chômage (<0.3 c-a-d < 4.5 ans).
# Si chomage est fort, la probabilité de succès s'effondre (blocage systémique).
y_train = np.where((X_train[:, 0] > 0.4) & (X_train[:, 2] < 0.3), 1.0, 0.0)

# L'individu du test :
# Compétence = 95% (0.95)
# Santé = 90% (0.90)
# Années de chômage = 12 ans (12/15 = 0.80)
X_test = np.array([[0.95, 0.90, 0.80]])

print("\n⚙️  Entraînement de FaNet v2.0 sur les données RH du marché...")
net = TorchFaNet(n_features=3, hidden_dim=16, n_outputs=1, seed=42)
# On donne à FaNet le temps d'aligner le QKV Attention
net.fit(X_train, y_train, epochs=80, lr=0.01, verbose=False)

print("\n🔍 1. Prédiction Numérique Brute")
pred, dus, attn = net.predict(X_test, return_attention=True)
reussite_prob = pred[0][0]
print(f"Probabilité de trouver un emploi : {reussite_prob:.2%}")

print("\n📜 2. Le Verdict Fâ Global (La Couche de Sortie)")
du = dus[0]
print(f"Signe final prédit : {du.get('nom_fon', '')} ({du.get('nom_yoruba', '')})")
print(f"Polarité          : {du.get('polarite', '')}")
print(f"Intensité         : {du.get('intensite', '')}")

print(f"Interprétation    : {du.get('interpretation', '')[:200]}...")
print(f"Prescription      : {du.get('prescriptions', [])}")

print("\n👁️  3. Radiographie de l'Attention (Soft-Lookup Matrix)")
# attn est de shape (batch, n_features, 256)
# Regardons sur quels signes le réseau a fixé son "Attention" pour la feature 3 (annees_chomage)
kb = _get_kb()
feature_idx = 2  # annees_chomage
attention_chomage = attn[0, feature_idx, :]

# Top 3 des signes les plus regardés pour comprendre le chômage prolongé
top_3_indices = np.argsort(attention_chomage)[::-1][:3]

print("Pour la feature 'Années de chômage (12 ans)', l'IA s'est focalisée sur :")
for i, idx in enumerate(top_3_indices):
    score = attention_chomage[idx] * 100
    signe = kb[idx]
    
    interp = signe.get('interpretation', '') or ''
    # On prend les 100 premiers caractères max pour la lisibilité
    diag = interp[:100].replace('\n', ' ') + '...' if interp else 'Aucune'
    
    print(f"  {i+1}. {signe.get('nom_fon', ''):<12} (Attention: {score:05.2f}%) -> {diag}")

print("="*70)
