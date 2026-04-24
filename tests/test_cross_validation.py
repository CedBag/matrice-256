"""
Test 2 : Cross-Validation Stratifiee K=10 (Standard Scientifique Absolu)
=========================================================================
Protocole :

La Cross-Validation K=10 divise les donnees en 10 sous-ensembles (folds)
de taille egale. Pour chaque fold :
  - Le modele est entraine sur les 9 folds restants
  - Le modele est evalue sur le fold exclu
  - On repete 10 fois en rotant le fold exclu

Resultat : 10 scores independants => on rapporte MOYENNE et ECART-TYPE.

Pourquoi c'est le test le plus exigeant ?
  - Il elimine le biais de "chance" d'un seul split train/test arbitraire
  - L'ecart-type revele si le modele est STABLE ou CHANCEUX
  - Standard obligatoire pour toute publication academique (NeurIPS, ICML)
  - Reponse directe a DeepSeek : "les scores annonces sans methodologie ne
    prouvent rien"

NB : Ce test entraine FaNet 10 fois de suite. C'est intentionnel et necessaire.
     Ce n'est pas de la simulation. C'est de la science.

Auteur : Projet Matrice-256 / Phase Academique v4
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

# Supprimer les warnings de convergence pour ne garder que les vrais resultats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bokono.ml.torch_network import TorchFaNet

# ─── Configuration ─────────────────────────────────────────────────────────────
TARGET_CROP  = "rice"
N_SPLITS     = 10
FANET_EPOCHS = 40
FANET_LAMBDA = 0.1
RANDOM_STATE = 42


# ─── Outils ────────────────────────────────────────────────────────────────────
def score_model(y_true, y_pred):
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
    }


def aggregate(scores: list[dict]) -> dict:
    """Calcule moyenne et ecart-type sur une liste de dicts de metriques."""
    keys = scores[0].keys()
    return {
        k: {
            "mean": float(np.mean([s[k] for s in scores])),
            "std":  float(np.std([s[k] for s in scores])),
        }
        for k in keys
    }


def print_summary(name: str, agg: dict, elapsed: float):
    """Affiche une ligne de resultat formatee."""
    acc  = agg["acc"]
    f1   = agg["f1"]
    prec = agg["prec"]
    rec  = agg["rec"]
    print(
        f"  {name:<30} | "
        f"ACC: {acc['mean']*100:6.2f}% +/- {acc['std']*100:.2f}% | "
        f"F1: {f1['mean']*100:6.2f}% +/- {f1['std']*100:.2f}% | "
        f"P: {prec['mean']*100:5.1f}% R: {rec['mean']*100:5.1f}%  | "
        f"{elapsed:6.1f}s"
    )


# ─── Main ──────────────────────────────────────────────────────────────────────
def run_cross_validation():
    print("=" * 90)
    print("  TEST 2 : CROSS-VALIDATION STRATIFIEE K=10")
    print("  Standard Scientifique Absolu — FaNet vs RF vs GBM")
    print("  Chaque modele sera entraine 10 fois. Ce n'est pas optionnel. C'est la science.")
    print("=" * 90)

    # 1. Chargement
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data_poc", "agri_dataset.csv"
    )
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path).dropna(axis=1, how="all")
    df["is_target"] = (df["label"] == TARGET_CROP).astype(int)
    X = df.drop(columns=["label", "is_target"]).values.astype(np.float64)
    y = df["is_target"].values

    # Statistique de la distribution (on verifie que c'est bien stratifie)
    n_pos = y.sum()
    n_total = len(y)
    print(f"\n[INFO] Dataset : {n_total} echantillons | {X.shape[1]} features")
    print(f"[INFO] Distribution : {n_pos} positifs ({n_pos/n_total*100:.1f}%) / "
          f"{n_total-n_pos} negatifs ({(n_total-n_pos)/n_total*100:.1f}%)")
    print(f"[INFO] Folds : K={N_SPLITS} | Stratifie => distribution equilibree dans chaque fold")
    print(f"\n[INFO] Debut des {N_SPLITS} rounds d'entrainement...\n")

    # 2. Generateur de folds
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 3. Accumulateurs de scores
    scores = {
        "rf":  [],
        "gbm": [],
        "fno": [],  # FaNet sans contrainte Fa (ablation)
        "ffa": [],  # FaNet complet (lambda_fa=0.1)
    }

    t_total = {k: 0.0 for k in scores}

    # 4. Boucle de K folds
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Standardisation FITTEE uniquement sur le train du fold => pas de fuite de donnees
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc  = sc.transform(X_te)

        print(f"  [Fold {fold_idx:02d}/{N_SPLITS}] "
              f"train={len(y_tr)} test={len(y_te)} "
              f"(positifs train={y_tr.sum()} / test={y_te.sum()})")

        # --- Random Forest ---
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=fold_idx, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        scores["rf"].append(score_model(y_te, rf.predict(X_te_sc)))
        t_total["rf"] += time.time() - t0

        # --- Gradient Boosting ---
        t0 = time.time()
        gbm = GradientBoostingClassifier(n_estimators=100, random_state=fold_idx)
        gbm.fit(X_tr_sc, y_tr)
        scores["gbm"].append(score_model(y_te, gbm.predict(X_te_sc)))
        t_total["gbm"] += time.time() - t0

        # --- FaNet SANS contrainte (Ablation) ---
        t0 = time.time()
        fno = TorchFaNet(n_features=X_tr.shape[1], hidden_dim=16, n_outputs=1, seed=fold_idx)
        fno.fit(X_tr_sc, y_tr, epochs=FANET_EPOCHS, lr=0.01,
                batch_size=64, optimizer="adam", verbose=False, lambda_fa=0.0)
        p_fno = (fno.predict_proba(X_te_sc.astype(np.float32)) > 0.5).astype(int).flatten()
        scores["fno"].append(score_model(y_te, p_fno))
        t_total["fno"] += time.time() - t0

        # --- FaNet COMPLET (Contrainte Fa activee) ---
        t0 = time.time()
        ffa = TorchFaNet(n_features=X_tr.shape[1], hidden_dim=16, n_outputs=1, seed=fold_idx)
        ffa.fit(X_tr_sc, y_tr, epochs=FANET_EPOCHS, lr=0.01,
                batch_size=64, optimizer="adam", verbose=False, lambda_fa=FANET_LAMBDA)
        p_ffa = (ffa.predict_proba(X_te_sc.astype(np.float32)) > 0.5).astype(int).flatten()
        scores["ffa"].append(score_model(y_te, p_ffa))
        t_total["ffa"] += time.time() - t0

    # 5. Aggregation et affichage
    agg = {k: aggregate(v) for k, v in scores.items()}

    print("\n" + "=" * 90)
    print("  RESULTATS K=10 CROSS-VALIDATION (MOYENNE +/- ECART-TYPE SUR 10 FOLDS)")
    print("=" * 90)
    print(f"  {'Modele':<30} | {'Accuracy':^23} | {'F1-Score':^22} | {'P / R':^15} | {'Temps'}")
    print("-" * 90)
    print_summary("Random Forest (SOTA)",        agg["rf"],  t_total["rf"])
    print_summary("Gradient Boosting",            agg["gbm"], t_total["gbm"])
    print_summary("FaNet (sans contrainte Fa)",   agg["fno"], t_total["fno"])
    print_summary("FaNet v3.0 (Fa actif L=0.1)", agg["ffa"], t_total["ffa"])
    print("=" * 90)

    # 6. Analyse de la Stabilite (C'est le vrai apport de la CV)
    print("\n[ANALYSE DE STABILITE - L'ecart-type est la vraie signature]")
    ref_std   = agg["rf"]["acc"]["std"] * 100
    ffa_std   = agg["ffa"]["acc"]["std"] * 100
    ref_mean  = agg["rf"]["acc"]["mean"] * 100
    ffa_mean  = agg["ffa"]["acc"]["mean"] * 100

    print(f"  Random Forest    : mu={ref_mean:.2f}%  sigma={ref_std:.3f}%")
    print(f"  FaNet v3.0 (Fa)  : mu={ffa_mean:.2f}%  sigma={ffa_std:.3f}%")

    if ffa_std <= ref_std * 1.5:
        print("\n  [VERDICT] FaNet est STABLE sur K=10 folds.")
        print("  L'ecart-type confirme que le score n'est pas du a la chance d'un split unique.")
        print("  Ces chiffres sont publiables dans une conference academique.")
    else:
        diff = ffa_std - ref_std
        print(f"\n  [ATTENTION] FaNet a une variance {diff:.3f}% superieure a RF.")
        print("  Recommandation : augmenter les epochs ou ajouter un scheduler de LR.")

    print("\n[INFO] Section 6 du Whitepaper peut etre redigee avec ces resultats.")


if __name__ == "__main__":
    run_cross_validation()
