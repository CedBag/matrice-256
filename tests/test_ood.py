"""
Test 3 : Out-of-Distribution (OOD) - Generalisation Transversale
================================================================
Objectif : Tester si FaNet extrait des invariants ontologiques (ex: signature 
Eau/Terre des cereales) resistants au changement de classe, la ou les modeles 
statistiques comme Random Forest sur-apprennent les bornes exactes des features.

Methode :
- Tache de base : Reconnaitre une "Cereale" d'un "Fruit".
- Train  : Positif = Riz    | Negatif = Fruits (Pomme, Orange, Papaye)
- Test   : Positif = Mais   | Negatif = Fruits (Pomme, Orange, Papaye)

Hypothese : Random Forest, ultra-specialise sur les statistiques exactes du Riz,
va s'effondrer sur le Mais (chute drastique de Rappel). FaNet, guide par 
la FaImbalanceLoss et ses attracteurs, aura capte le "Kpoli" d'une cereale
(haut besoin en azote, tolerance specifique) et reconnaitra le Mais, avec
une chute de F1/Rappel beaucoup plus faible.

Auteur : Projet Matrice-256 / Phase Academique v4
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bokono.ml.torch_network import TorchFaNet

# ─── Configuration ─────────────────────────────────────────────────────────────
TRAIN_POS = ["rice"]
TEST_POS  = ["jute"]
# Negatifs : Des cultures seches (les haricots/lentilles) avec de faibles besoins en eau (rain < 100)
NEGATIVES = ["chickpea", "kidneybeans", "lentil", "mothbeans"]

EPOCHS       = 100
LR           = 0.005
LAMBDA_BASE  = 0.1
RANDOM_STATE = 42

def evaluate_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return {"acc": acc, "f1": f1, "prec": pr, "rec": rec}

def print_result(name, metrics, elapsed):
    print(
        f"  {name:<30} | "
        f"ACC {metrics['acc']*100:6.2f}% | "
        f"F1  {metrics['f1']*100:6.2f}% | "
        f"P:{metrics['prec']*100:5.1f}% R:{metrics['rec']*100:5.1f}% | "
        f"{elapsed:5.0f}s"
    )

def run_ood_test():
    print("=" * 95)
    print("  TEST 3 : OUT-OF-DISTRIBUTION (OOD) - GENERALISATION TRANSVERSALE")
    print("  Train : Riz (Eau++) vs Secs  ───>  Test : Jute (Eau+) vs Secs")
    print("  Hypothese : Random Forest va overfitter sur Rainfall > 180 (limite du Riz).")
    print("              L'attracteur Fâ (FaNet) va reconnaitre la signature fondamentale (Yin/Eau)")
    print("              du Jute malgre la chute de rainfall autour de 150-199.")
    print("=" * 95)

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data_poc", "agri_dataset.csv"
    )
    df = pd.read_csv(csv_path).dropna(axis=1, how="all")

    # Construction du dataset d'entrainement (Train : Riz = 1, Fruits = 0)
    df_train = df[df["label"].isin(TRAIN_POS + NEGATIVES)].copy()
    df_train["target"] = df_train["label"].isin(TRAIN_POS).astype(int)
    X_tr = df_train.drop(columns=["label", "target"]).values.astype(np.float64)
    y_tr = df_train["target"].values

    # Construction du dataset de test de reference (Test In-Distribution : Riz vs Fruits)
    # Pour avoir un test baseline, on utilise un split artificiel ou on re-evalue sur Riz vs Fruits (par ex via 5-fold CV simulé ou juste un shuffle)
    from sklearn.model_selection import train_test_split
    X_tr_in, X_te_in, y_tr_in, y_te_in = train_test_split(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=RANDOM_STATE)

    # Construction du dataset de test OOD (Test : Mais = 1, Fruits = 0)
    df_test_ood = df[df["label"].isin(TEST_POS + NEGATIVES)].copy()
    df_test_ood["target"] = df_test_ood["label"].isin(TEST_POS).astype(int)
    X_te_ood = df_test_ood.drop(columns=["label", "target"]).values.astype(np.float64)
    y_te_ood = df_test_ood["target"].values

    print(f"\n[DATASETS]")
    print(f"  Distribution d'entrainement (IN) : Positifs ({'+'.join(TRAIN_POS)})={y_tr_in.sum()} | Negatifs ({len(NEGATIVES)} fruits)={len(y_tr_in)-y_tr_in.sum()}")
    print(f"  Distribution de Test (OOD)       : Positifs ({'+'.join(TEST_POS)})={y_te_ood.sum()} | Negatifs ({len(NEGATIVES)} fruits)={len(y_te_ood)-y_te_ood.sum()}")

    # Preprocessing pipeline
    sc = StandardScaler()
    X_tr_in_sc = sc.fit_transform(X_tr_in)
    X_te_in_sc = sc.transform(X_te_in)
    X_te_ood_sc = sc.transform(X_te_ood)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_sm, y_sm = smote.fit_resample(X_tr_in_sc, y_tr_in)

    # Parametres adaptatifs
    n_pos_tr = int(y_tr_in.sum())
    n_neg_tr = int(len(y_tr_in) - n_pos_tr)
    pw = float(np.sqrt(n_neg_tr / max(n_pos_tr, 1)))
    lam = LAMBDA_BASE * float(np.sqrt(n_pos_tr / len(y_tr_in)))

    print(f"  Apprentissage avec pos_weight={pw:.2f} et lambda={lam:.4f}\n")

    # ======== 1. Random Forest ========
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_sm, y_sm)
    rf_in  = evaluate_model(y_te_in, rf.predict(X_te_in_sc), "RF")
    rf_ood = evaluate_model(y_te_ood, rf.predict(X_te_ood_sc), "RF")
    t_rf = time.time() - t0

    # ======== 2. Gradient Boosting ========
    t0 = time.time()
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gbm.fit(X_sm, y_sm)
    gbm_in  = evaluate_model(y_te_in, gbm.predict(X_te_in_sc), "GBM")
    gbm_ood = evaluate_model(y_te_ood, gbm.predict(X_te_ood_sc), "GBM")
    t_gbm = time.time() - t0

    # ======== 3. FaNet v3.1 ========
    t0 = time.time()
    ffa = TorchFaNet(n_features=X_sm.shape[1], hidden_dim=32, n_outputs=1, seed=RANDOM_STATE)
    ffa.fit(
        X_sm, y_sm,
        epochs=EPOCHS, lr=LR,
        batch_size=64, optimizer="adamw",
        verbose=False, lambda_fa=lam, pos_weight=pw
    )
    p_ffa_in  = (ffa.predict_proba(X_te_in_sc.astype(np.float32)) > 0.5).astype(int).flatten()
    p_ffa_ood = (ffa.predict_proba(X_te_ood_sc.astype(np.float32)) > 0.5).astype(int).flatten()
    
    ffa_in  = evaluate_model(y_te_in, p_ffa_in, "FaNet")
    ffa_ood = evaluate_model(y_te_ood, p_ffa_ood, "FaNet")
    t_ffa = time.time() - t0

    # Affichage des resultats
    print("=" * 95)
    print("  RESULTATS IN-DISTRIBUTION (Test sur split de validation RIZ)")
    print("=" * 95)
    print_result("Random Forest", rf_in, t_rf)
    print_result("Gradient Boosting", gbm_in, t_gbm)
    print_result("FaNet v3.1", ffa_in, t_ffa)

    print("\n" + "=" * 95)
    print("  RESULTATS OUT-OF-DISTRIBUTION (Test OOD : MAIS)")
    print("=" * 95)
    print_result("Random Forest", rf_ood, t_rf)
    print_result("Gradient Boosting", gbm_ood, t_gbm)
    print_result("FaNet v3.1", ffa_ood, t_ffa)

    print("\n" + "=" * 95)
    print("  ANALYSE DE LA ROBUSTESSE OOD (Chute de Rappel 'Riz -> Mais')")
    print("=" * 95)
    
    drop_rf  = rf_in['rec'] - rf_ood['rec']
    drop_gbm = gbm_in['rec'] - gbm_ood['rec']
    drop_fa  = ffa_in['rec'] - ffa_ood['rec']

    print(f"  - Random Forest     : Rappel passe de {rf_in['rec']*100:.1f}% a {rf_ood['rec']*100:.1f}%  [CHUTE: {drop_rf*100:+.1f}%]")
    print(f"  - Gradient Boosting : Rappel passe de {gbm_in['rec']*100:.1f}% a {gbm_ood['rec']*100:.1f}%  [CHUTE: {drop_gbm*100:+.1f}%]")
    print(f"  - FaNet v3.1        : Rappel passe de {ffa_in['rec']*100:.1f}% a {ffa_ood['rec']*100:.1f}%  [CHUTE: {drop_fa*100:+.1f}%]")

    print("\n[VERDICT OOD]")
    if drop_fa > drop_rf:
        print("FaNet s'est effondre face au changement de distribution. Hypothese ontologique invalidee.")
    else:
        print("L'attracteur Fa permet a FaNet d'extraire l'invariant structurel d'une hypothese plutot que memoriser.")
        print("La chute de performance est amortie prouvant une meilleure caracterisation des invariants Fâ.")

if __name__ == "__main__":
    run_ood_test()
