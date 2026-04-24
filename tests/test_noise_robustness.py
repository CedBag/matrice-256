"""
Test 1 : Stress Test — Robustesse au Bruit (Noise Robustness Test)
==================================================================
Protocole Scientifique :

On injecte du bruit gaussien progressif (0% → 50%) dans les données
de test UNIQUEMENT (les modèles sont entraînés sur des données propres).
Cela simule des capteurs IoT défaillants, des relevés médicaux approximatifs,
ou des données terrain corrompues.

Hypothèse FaNet :
    La FaImbalanceLoss agit comme un régularisateur ontologique qui force
    le modèle à converger vers les attracteurs Fâ stables (les 256 états).
    Résultat attendu : FaNet doit dégrader MOINS vite que les modèles
    standards (RF, GBM) quand le bruit augmente.

Auteur : Projet Matrice-256 / Phase Académique v4
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Fix Windows terminal encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bokono.ml.torch_network import TorchFaNet

# ─── Paramètres du Test ────────────────────────────────────────────────────────
TARGET_CROP    = 'rice'
NOISE_LEVELS   = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]  # Ratio du std des données
FANET_EPOCHS   = 40
FANET_LAMBDA   = 0.1   # Contrainte Fâ complète
RANDOM_STATE   = 42

# ─── Injection de Bruit ────────────────────────────────────────────────────────
def add_gaussian_noise(X: np.ndarray, noise_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Injecte un bruit gaussien proportionnel à l'écart-type de chaque feature."""
    if noise_ratio == 0.0:
        return X.copy()
    std_per_feature = X.std(axis=0) + 1e-9
    noise = rng.normal(0, noise_ratio, X.shape) * std_per_feature
    return X + noise

# ─── Main ──────────────────────────────────────────────────────────────────────
def run_stress_test():
    print("=" * 65)
    print("  TEST 1 : STRESS TEST — ROBUSTESSE AU BRUIT (Noise Robustness)")
    print("  FaNet v3.0 vs Random Forest vs Gradient Boosting")
    print("=" * 65)

    # 1. Chargement des données
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data_poc", "agri_dataset.csv"
    )
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path).dropna(axis=1, how='all')
    df['is_target'] = (df['label'] == TARGET_CROP).astype(int)
    X = df.drop(columns=['label', 'is_target']).values.astype(np.float64)
    y = df['is_target'].values

    # 2. Split train / test (propre — seul le test sera bruité)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n[INFO] {X.shape[0]} échantillons | {X.shape[1]} capteurs | Cible : '{TARGET_CROP}'")
    print(f"[INFO] Entraînement sur données PROPRES — Bruitage appliqué au JEU DE TEST UNIQUEMENT\n")

    # 3. Entraîner tous les modèles UNE SEULE FOIS sur données propres
    print("[INFO] Entraînement des modèles (données propres)...")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    # Gradient Boosting
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gbm.fit(X_train_sc, y_train)

    # FaNet SANS contrainte (ablation)
    fanet_no = TorchFaNet(n_features=X_train.shape[1], hidden_dim=16, n_outputs=1, seed=12)
    fanet_no.fit(X_train_sc, y_train, epochs=FANET_EPOCHS, lr=0.01,
                 batch_size=64, optimizer="adam", verbose=False, lambda_fa=0.0)

    # FaNet COMPLET (contrainte Fâ activée)
    fanet_fa = TorchFaNet(n_features=X_train.shape[1], hidden_dim=16, n_outputs=1, seed=12)
    fanet_fa.fit(X_train_sc, y_train, epochs=FANET_EPOCHS, lr=0.01,
                 batch_size=64, optimizer="adam", verbose=False, lambda_fa=FANET_LAMBDA)

    print("[INFO] Modèles entraînés. Lancement du Stress Test...\n")

    # 4. Boucle de bruitage sur le JEU DE TEST
    rng = np.random.default_rng(RANDOM_STATE)
    all_results = []

    for noise in NOISE_LEVELS:
        X_test_noisy_sc = add_gaussian_noise(X_test_sc, noise, rng)

        # RF
        acc_rf  = accuracy_score(y_test, rf.predict(X_test_noisy_sc))
        f1_rf   = f1_score(y_test, rf.predict(X_test_noisy_sc), zero_division=0)

        # GBM
        acc_gbm = accuracy_score(y_test, gbm.predict(X_test_noisy_sc))
        f1_gbm  = f1_score(y_test, gbm.predict(X_test_noisy_sc), zero_division=0)

        # FaNet Sans FaLoss
        pno  = fanet_no.predict_proba(X_test_noisy_sc.astype(np.float32))
        acc_fno = accuracy_score(y_test, (pno > 0.5).astype(int).flatten())
        f1_fno  = f1_score(y_test, (pno > 0.5).astype(int).flatten(), zero_division=0)

        # FaNet Complet
        pfa  = fanet_fa.predict_proba(X_test_noisy_sc.astype(np.float32))
        acc_ffa = accuracy_score(y_test, (pfa > 0.5).astype(int).flatten())
        f1_ffa  = f1_score(y_test, (pfa > 0.5).astype(int).flatten(), zero_division=0)

        all_results.append({
            "noise": noise,
            "rf_acc": acc_rf,   "rf_f1": f1_rf,
            "gbm_acc": acc_gbm, "gbm_f1": f1_gbm,
            "fno_acc": acc_fno, "fno_f1": f1_fno,
            "ffa_acc": acc_ffa, "ffa_f1": f1_ffa,
        })

    # 5. Affichage du tableau scientifique
    print("=" * 95)
    print(f"{'Bruit':>6} | {'RF Acc':>8} | {'GBM Acc':>8} | {'FaNet(0) Acc':>12} | {'FaNet(Fa) Acc':>13} | {'Ecart (RF vs Fa)':>16}")
    print("-" * 95)

    baseline_rf  = all_results[0]["rf_acc"]
    baseline_ffa = all_results[0]["ffa_acc"]

    for r in all_results:
        delta_rf  = (r["rf_acc"]  - baseline_rf)  * 100
        delta_ffa = (r["ffa_acc"] - baseline_ffa) * 100
        gap       = (r["ffa_acc"] - r["rf_acc"])  * 100

        print(
            f"{r['noise']*100:>5.0f}% | "
            f"{r['rf_acc']*100:>7.2f}% | "
            f"{r['gbm_acc']*100:>7.2f}% | "
            f"{r['fno_acc']*100:>11.2f}% | "
            f"{r['ffa_acc']*100:>12.2f}% | "
            f"{gap:>+15.2f}%"
        )

    print("=" * 95)

    # 6. Analyse Scientifique
    r0  = all_results[0]
    r50 = all_results[-1]
    degradation_rf  = (r0["rf_acc"]  - r50["rf_acc"])  * 100
    degradation_ffa = (r0["ffa_acc"] - r50["ffa_acc"]) * 100

    print("\n[ANALYSE SCIENTIFIQUE]")
    print(f"  → Random Forest    : chute de {degradation_rf:+.2f}% entre 0% et 50% de bruit")
    print(f"  → FaNet v3.0 (Fâ) : chute de {degradation_ffa:+.2f}% entre 0% et 50% de bruit")

    if degradation_ffa <= degradation_rf:
        print("\n[VERDICT] [OK] HYPOTHESE CONFIRMEE :")
        print("  La FaImbalanceLoss agit comme un amortisseur ontologique.")
        print("  FaNet se degrade MOINS vite que les modeles standards sous injection de bruit.")
        print("  Consequence : FaNet est plus adapte aux deploiements sur capteurs IoT imparfaits.")
    else:
        gap_deg = degradation_ffa - degradation_rf
        print(f"\n[VERDICT] [!] FaNet se degrade {gap_deg:.2f}% de PLUS que Random Forest sous bruit.")
        print("  Analyse requise : l'architecture d'attention combinatoire amplifie peut-etre")
        print("  certaines features bruyantes via le mecanisme de Query Gauche/Droite.")
        print("  -> Piste d'amelioration : ajouter un Dropout (p=0.2) dans TorchFaEmbedding.")

    print("\n[INFO] Résultats prêts pour insertion dans le Whitepaper (docs/whitepaper_matrice256.md)")


if __name__ == "__main__":
    run_stress_test()
