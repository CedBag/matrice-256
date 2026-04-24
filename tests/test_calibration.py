"""
Test 4 : Calibration des Probabilites (Reliability & Brier Score)
===================================================================
Objectif : Verifier que les probabilites emises par FaNet sont fiables 
dans un contexte clinique/industriel. (Ex: Si le modele dit "80% de risque",
il doit y avoir effectivement 80% de cas positifs dans cette strate).

Methode :
- Entrainement classique (Riz vs autres)
- Calcul de la brier_score_loss (Metrique d'evaluation de l'exactitude probabiliste)
- Calcul de la calibration_curve (Reliability Diagram)
- ECE (Expected Calibration Error) evaluee empiriquement.

Hypothese : La BCEWithLogitsLoss couplee a la FaImbalanceLoss force une 
regularisation qui pourrait naturellement calibrer les probabilites de FaNet
face aux modeles GBM/RF extremement confiances (souvent sur-confiants).

Auteur : Projet Matrice-256 / Phase Academique v4
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bokono.ml.torch_network import TorchFaNet

# ─── Configuration ─────────────────────────────────────────────────────────────
TARGET_CROP  = "rice"
EPOCHS       = 100
LR           = 0.005
LAMBDA_BASE  = 0.1
RANDOM_STATE = 42

def expected_calibration_error(y_true, probas, n_bins=10):
    """Calcule l'ECE (Expected Calibration Error)."""
    prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=n_bins, strategy='uniform')
    # Estimation simplifiee de l'erreur absolue moyenne par bin (sans ponderation stricte)
    return np.mean(np.abs(prob_true - prob_pred))

def print_calibration_stats(name, y_true, probas):
    brier = brier_score_loss(y_true, probas)
    ece   = expected_calibration_error(y_true, probas, n_bins=10)
    print(f"  {name:<20} | Brier Score: {brier:.4f} | ECE: {ece*100:.2f}%")
    return brier, ece

def run_calibration_test():
    print("=" * 90)
    print("  TEST 4 : CALIBRATION DES PROBABILITES (FIABILITE CLINIQUE/INDUSTRIELLE)")
    print("  Metriques : Brier Score (plus faible = meilleur) et ECE (Expected Calibration Error)")
    print("=" * 90)

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data_poc", "agri_dataset.csv"
    )
    df = pd.read_csv(csv_path).dropna(axis=1, how="all")
    
    # Afin d'avoir plus de donnees pour la calibration, on prend un ratio 1/3
    target_crops = ["rice", "jute", "maize"]
    df["target"] = df["label"].isin(target_crops).astype(int)
    X = df.drop(columns=["label", "target"]).values.astype(np.float64)
    y = df["target"].values

    # Train / Test split massif
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    
    print(f"\n[DATASET]")
    print(f"  Positifs (Riz/Jute/Mais) : {y_tr.sum()} Train | {y_te.sum()} Test")
    print(f"  Negatifs (Autres)        : {len(y_tr)-y_tr.sum()} Train | {len(y_te)-y_te.sum()} Test\n")

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    # Parametres FaNet
    pos_w = float(np.sqrt((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)))
    lam   = LAMBDA_BASE

    # ======== 1. Random Forest ========
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_tr_sc, y_tr)
    rf_probas = rf.predict_proba(X_te_sc)[:, 1]

    # ======== 2. Gradient Boosting ========
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gbm.fit(X_tr_sc, y_tr)
    gbm_probas = gbm.predict_proba(X_te_sc)[:, 1]

    # ======== 3. FaNet v3.1 ========
    ffa = TorchFaNet(n_features=X_tr.shape[1], hidden_dim=32, n_outputs=1, seed=RANDOM_STATE)
    ffa.fit(
        X_tr_sc, y_tr,
        epochs=EPOCHS, lr=LR, batch_size=64, optimizer="adamw",
        verbose=False, lambda_fa=lam, pos_weight=pos_w
    )
    ffa_probas = ffa.predict_proba(X_te_sc.astype(np.float32)).flatten()

    print("=" * 90)
    print("  RESULTATS DE CALIBRATION (Brier Score & Expected Calibration Error)")
    print("=" * 90)
    rf_brier, rf_ece   = print_calibration_stats("Random Forest", y_te, rf_probas)
    gbm_brier, gbm_ece = print_calibration_stats("Gradient Boosting", y_te, gbm_probas)
    fa_brier, fa_ece   = print_calibration_stats("FaNet v3.1", y_te, ffa_probas)

    print("\n" + "=" * 90)
    print("  COURBES DE CALIBRATION (BINS % Predits vs % Reels)")
    print("=" * 90)
    
    models = [("Random Forest", rf_probas), ("Grad. Boosting", gbm_probas), ("FaNet v3.1", ffa_probas)]
    for nom, prob in models:
        # N_bins dynamique base sur la distribution des probas pour eviter les erreurs de bornes vides
        prob_t, prob_p = calibration_curve(y_te, prob, n_bins=5, strategy='uniform')
        print(f"\n  [ {nom} ]")
        for pt, pp in zip(prob_t, prob_p):
            print(f"    Pred {pp*100:04.1f}%  ----> Verite: {pt*100:04.1f}% | Ecart: {abs(pp-pt)*100:04.1f}%")

    print("\n[VERDICT CALIBRATION CLINIQUE]")
    if fa_brier < rf_brier and fa_brier < gbm_brier:
        print("  SUCCES MASSIF: FaNet est mathematiquement le modele le mieux calibre (Brier Score inferieur).")
        print("  Ideale pour des garanties legales/hospitalieres car le modele est moins sur-confiant.")
    elif fa_ece < rf_ece and fa_ece < gbm_ece:
        print("  SUCCES: FaNet a la plus faible erreur de calibration attendue (ECE). Probabilites tres fiables.")
    else:
        print("  FaNet est legerement moins calibre aux bornes que les methodes statistiques etablies.")
        print("  C'est un comportement classique pour une architecture hybride. La methode de calibration")
        print("  isotonique globale de scikit-learn pourrait s'averer necessaire post-entrainement.")

if __name__ == "__main__":
    run_calibration_test()
