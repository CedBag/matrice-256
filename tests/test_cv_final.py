"""
Test 2 (v3 - FINAL) : Cross-Validation K=10 — API Native pos_weight
====================================================================
Correction architecturale definitive :
  - pos_weight est un parametre natif de TorchFaNet.fit()
  - Plus de monkey-patching. L'API est propre et documentee.
  - lambda_fa et pos_weight sont calcules par fold depuis y_train.
  - SMOTE applique sur les donnees d'entrainement de chaque fold uniquement.

C'est le script de reference academique. Ses resultats sont ceux
qui entrent dans le Whitepaper et dans toute soumission a CARI / ICLR.

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bokono.ml.torch_network import TorchFaNet

# ─── Configuration ─────────────────────────────────────────────────────────────
TARGET_CROP  = "rice"
N_SPLITS     = 10
EPOCHS       = 100
LR           = 0.008
LAMBDA_BASE  = 0.1      # sera scale selon sqrt(ratio_positifs)
RANDOM_STATE = 42


def score_model(y_true, y_pred):
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
    }


def aggregate(scores):
    keys = scores[0].keys()
    return {
        k: {"mean": float(np.mean([s[k] for s in scores])),
            "std":  float(np.std([s[k] for s in scores]))}
        for k in keys
    }


def print_row(name, agg, elapsed):
    a, f = agg["acc"], agg["f1"]
    p, r = agg["prec"], agg["rec"]
    print(
        f"  {name:<35} | "
        f"ACC {a['mean']*100:6.2f}%+/-{a['std']*100:.2f}% | "
        f"F1  {f['mean']*100:6.2f}%+/-{f['std']*100:.2f}% | "
        f"P:{p['mean']*100:5.1f}% R:{r['mean']*100:5.1f}% | "
        f"{elapsed:5.0f}s"
    )


def run_final_cv():
    print("=" * 100)
    print("  TEST 2 (v3 FINAL) : API NATIVE pos_weight — FaNet.fit(pos_weight=...)")
    print("  Protocole : SMOTE + pos_weight + lambda_fa adaptatifs, zero monkey-patch")
    print("=" * 100)

    # Chargement
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data_poc", "agri_dataset.csv"
    )
    df = pd.read_csv(csv_path).dropna(axis=1, how="all")
    df["is_target"] = (df["label"] == TARGET_CROP).astype(int)
    X = df.drop(columns=["label", "is_target"]).values.astype(np.float64)
    y = df["is_target"].values

    n_pos   = int(y.sum())
    n_neg   = int(len(y) - n_pos)
    r_pos   = n_pos / len(y)
    print(f"\n[INFO] {len(y)} echantillons | positifs: {n_pos} ({r_pos*100:.1f}%) | negatifs: {n_neg}")
    print(f"[INFO] pos_weight global : {n_neg/n_pos:.1f}  |  lambda adaptatif : {LAMBDA_BASE*np.sqrt(r_pos):.4f}\n")

    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    scores = {"rf": [], "gbm": [], "ffa": []}
    totals = {k: 0.0 for k in scores}

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Standardisation (uniquement sur train)
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc  = sc.transform(X_te)

        # SMOTE (uniquement sur train standardise)
        X_sm, y_sm = smote.fit_resample(X_tr_sc, y_tr)

        # Parametres adaptatifs calcules depuis y_sm (POST-SMOTE)
        # SMOTE equilibre a ~50/50 => pos_weight post-SMOTE ~= 1.0
        # On applique un leger boost (sqrt du ratio pre-SMOTE) pour garder
        # une sensibilite superieure aux faux negatifs sans saturer la BCE
        n_pos_fold   = int(y_tr.sum())
        n_neg_fold   = int(len(y_tr) - n_pos_fold)
        pre_ratio    = float(n_neg_fold) / float(max(n_pos_fold, 1))
        pw           = float(np.sqrt(pre_ratio))  # sqrt(21) ~= 4.6 au lieu de 21
        lam          = LAMBDA_BASE * float(np.sqrt(n_pos_fold / len(y_tr)))

        print(f"  [Fold {fold_idx:02d}/{N_SPLITS}] "
              f"train_SMOTE={len(y_sm)} | test={len(y_te)} | "
              f"pos_w={pw:.1f} | lambda={lam:.4f}")

        # Random Forest (balanced)
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=fold_idx, n_jobs=-1)
        rf.fit(X_sm, y_sm)
        scores["rf"].append(score_model(y_te, rf.predict(X_te_sc)))
        totals["rf"] += time.time() - t0

        # Gradient Boosting
        t0 = time.time()
        gbm = GradientBoostingClassifier(n_estimators=100, subsample=0.8,
                                          random_state=fold_idx)
        gbm.fit(X_sm, y_sm)
        scores["gbm"].append(score_model(y_te, gbm.predict(X_te_sc)))
        totals["gbm"] += time.time() - t0

        # FaNet v3.1 — API NATIVE pos_weight (zero hack)
        t0 = time.time()
        ffa = TorchFaNet(n_features=X_tr.shape[1], hidden_dim=32, n_outputs=1, seed=fold_idx)
        ffa.fit(
            X_sm, y_sm,
            epochs=EPOCHS, lr=LR,
            batch_size=64, optimizer="adamw",
            verbose=False,
            lambda_fa=lam,
            pos_weight=pw           # <-- API NATIVE, pas de monkey-patch
        )
        p_ffa = (ffa.predict_proba(X_te_sc.astype(np.float32)) > 0.5).astype(int).flatten()
        scores["ffa"].append(score_model(y_te, p_ffa))
        totals["ffa"] += time.time() - t0

    # Resultats
    agg = {k: aggregate(v) for k, v in scores.items()}
    hist = {
        "v1_f1": 72.39,   # Test 2 v1 sans correction
        "v2_f1": 78.56,   # Test 2 v2 avec monkey-patch
    }

    print("\n" + "=" * 100)
    print("  RESULTATS DEFINITIFS K=10 — API NATIVE pos_weight")
    print("=" * 100)
    print_row("Random Forest (balanced)",           agg["rf"],  totals["rf"])
    print_row("Gradient Boosting (SMOTE)",          agg["gbm"], totals["gbm"])
    print_row("FaNet v3.1 (API native pos_weight)", agg["ffa"], totals["ffa"])
    print("=" * 100)

    ffa_f1 = agg["ffa"]["f1"]["mean"] * 100
    ffa_acc = agg["ffa"]["acc"]["mean"] * 100
    ffa_rec = agg["ffa"]["rec"]["mean"] * 100

    print("\n[PROGRESSION F1 FaNet]")
    print(f"  v1 (sans correction)  : {hist['v1_f1']:.2f}%")
    print(f"  v2 (monkey-patch)     : {hist['v2_f1']:.2f}%  (+{hist['v2_f1']-hist['v1_f1']:.2f}%)")
    print(f"  v3 (API native)       : {ffa_f1:.2f}%  ({ffa_f1-hist['v2_f1']:+.2f}% vs v2)")

    print(f"\n[CONCLUSION ACADEMIQUE]")
    print(f"  - Accuracy  : {ffa_acc:.2f}% (reproductible sur 10 folds)")
    print(f"  - F1-Score  : {ffa_f1:.2f}% (progression totale depuis v1 : {ffa_f1-hist['v1_f1']:+.2f}%)")
    print(f"  - Rappel    : {ffa_rec:.1f}% (metrique critique en usage medical / agricole)")

    if ffa_f1 >= 80.0:
        print("  [VERDICT] F1 >= 80%. FaNet v3.1 est pret pour soumission academique.")
    else:
        gap = 80.0 - ffa_f1
        print(f"  [VERDICT] F1 = {ffa_f1:.2f}%. Il manque {gap:.2f}% pour atteindre le seuil 80%.")
        print("  Piste : augmenter EPOCHS (60+) ou tenter hidden_dim=32.")


if __name__ == "__main__":
    run_final_cv()
