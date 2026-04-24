"""
Test 6 : Benchmark Temporel (IoT Series Chronologiques) — v4.1
================================================================
Probleme resolue : Classe "Danger" rare (13.6%) + seuil fixe 0.5 aveugle.
Solution en deux etapes chirurgicales :

  1. Temporal Oversampling : Duplication des fenetres "Danger" uniquement dans
     l'ensemble TRAIN (pas dans TEST). Preserve la causalite temporelle.
     On ne shufflé pas le temps. On clone les patterns de crise jusqu'a
     atteindre un ratio ~35% de positifs pour que l'LSTM s'entraîne sur des
     evenements suffisamment representes.

  2. Seuil Adaptatif : Calibration du seuil optimal via maximisation du F1-Score
     sur un sous-ensemble de validation extrait UNIQUEMENT du Train (jamais du Test).
     Elimine le biais du seuil fixe 0.5 qui penalise les classes rares.

Auteur : Projet Matrice-256 / Phase Academique v4.1
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bokono.ml.torch_network import TorchFaNet

# ─── Configuration ─────────────────────────────────────────────────────────────
N_SENSORS       = 3      # temp, hum, soil
LOOKBACK_WINDOW = 12     # 12 heures de contexte
FORECAST_HORIZON = 6     # Predire un risque dans les 6 prochaines heures
TARGET_POS_RATIO = 0.35  # Ratio cible de positifs apres oversampling temporel
EPOCHS           = 100
LR               = 0.005
LAMBDA_BASE      = 0.1
RANDOM_STATE     = 42

def print_result(name, acc, f1, pr, rec, elapsed, threshold=None):
    thr_str = f" [seuil={threshold:.2f}]" if threshold is not None else ""
    print(
        f"  {name:<32} | "
        f"ACC {acc*100:6.2f}% | "
        f"F1  {f1*100:6.2f}% | "
        f"P:{pr*100:5.1f}% R:{rec*100:5.1f}%{thr_str}"
    )

def generate_iot_time_series(days=60):
    """Serie temporelle realiste : cycles jour/nuit + evenements de pluie aleatoires."""
    np.random.seed(RANDOM_STATE)
    hours = days * 24
    t = np.arange(hours)

    temperature   = 25 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, hours)
    humidity      = 70 + 20 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 2, hours)
    soil_moisture = np.zeros(hours)
    soil_moisture[0] = 50.0
    for i in range(1, hours):
        drop = 0.5 + 0.02 * max(0, temperature[i] - 25)
        val  = soil_moisture[i-1] - drop + np.random.normal(0, 0.5)
        if np.random.rand() < 0.02:
            val += 30.0
        soil_moisture[i] = np.clip(val, 0, 80)

    df = pd.DataFrame({'temp': temperature, 'hum': humidity, 'soil': soil_moisture})
    df['danger'] = (df['soil'] < 15.0).astype(int)
    return df

def create_sliding_window(df, lookback_h, forecast_h):
    """Fenetre glissante -> vecteur aplati (lookback_h * n_sensors)."""
    X, y = [], []
    features = ['temp', 'hum', 'soil']
    for i in range(len(df) - lookback_h - forecast_h):
        window = df.iloc[i : i + lookback_h][features].values.flatten()
        X.append(window)
        y.append(df.iloc[i + lookback_h : i + lookback_h + forecast_h]['danger'].max())
    return np.array(X), np.array(y)

def temporal_oversample(X, y, target_ratio=0.35, seed=42):
    """
    Duplication des fenetres "Danger" dans l'ensemble Train.
    REGLE : On ne touchera jamais au Test. On ne shufflé pas le temps.
    On clone les exemples minoritaires (danger=1) jusqu'a target_ratio.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos_orig = len(pos_idx)
    n_total    = len(y)

    # Combien de doublons faut-il pour atteindre target_ratio ?
    # target_ratio = (n_pos_orig + n_dup) / (n_total + n_dup)
    # => n_dup = (target_ratio * n_total - n_pos_orig) / (1 - target_ratio)
    n_dup = int(np.ceil(
        (target_ratio * n_total - n_pos_orig) / max(1 - target_ratio, 1e-6)
    ))
    n_dup = max(0, n_dup)

    if n_dup == 0:
        return X, y

    # Tirage avec remise parmi les positifs uniquement
    dup_idx = rng.choice(pos_idx, size=n_dup, replace=True)
    X_dup   = X[dup_idx]
    y_dup   = y[dup_idx]

    # Petit bruit gaussien (sigma = 1% de l'ecart-type feature) pour ne pas
    # coller exactement les memes vecteurs -> evite le sur-apprentissage de copie.
    noise = rng.normal(0, 0.01 * X.std(axis=0, keepdims=True), X_dup.shape)
    X_dup = X_dup + noise

    X_aug = np.vstack([X, X_dup])
    y_aug = np.concatenate([y, y_dup])
    return X_aug, y_aug

def find_best_threshold(probas, y_true, floor=0.20, ceil=0.90):
    """
    Calibration du seuil optimal maximisant le F1-Score sur le jeu de validation.
    Parcours de la courbe Precision-Recall.
    - floor : seuil minimum pour eviter les over-alertes
    - ceil  : seuil maximum pour eviter les sous-alertes totales
    """
    if y_true.sum() == 0:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, probas)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0
    )
    # On masque les seuils hors [floor, ceil] pour forcer un seuil raisonnable
    valid_mask = np.ones(len(thresholds), dtype=bool)
    for i, t in enumerate(thresholds):
        if t < floor or t > ceil:
            valid_mask[i] = False
    if valid_mask.sum() == 0:
        return 0.5, 0.0  # Fallback neutre si la zone valide est vide
    masked_f1 = np.where(valid_mask, f1_scores[:-1], 0.0)
    best_idx   = np.argmax(masked_f1)
    return float(thresholds[best_idx]), float(masked_f1[best_idx])

def run_iot_test():
    print("=" * 95)
    print("  TEST 6 v4.1 : BENCHMARK TEMPOREL (IoT PREDICTIF — LSTM Fâ + Calibration)")
    print("  Contexte    : Suivi horaire (Temp/Hum/Sol) sur 60 jours continus")
    print("  Corrections : Temporal Oversampling + Seuil Adaptatif sur Precision-Recall")
    print("=" * 95)

    df_iot  = generate_iot_time_series(days=60)
    X, y    = create_sliding_window(df_iot, LOOKBACK_WINDOW, FORECAST_HORIZON)

    # ── Decoupage TEMPOREL strict : Train 60%, Val 15%, Test 25% ──────────────
    # Jamais de shuffle. L'ordre chronologique est la loi.
    n = len(X)
    split_tr  = int(n * 0.60)
    split_val = int(n * 0.75)
    X_tr_raw, y_tr_raw = X[:split_tr], y[:split_tr]
    X_val, y_val       = X[split_tr:split_val], y[split_tr:split_val]
    X_te, y_te         = X[split_val:], y[split_val:]

    print(f"\n[DATASET IoT]")
    print(f"  Series totales  : {n} fenetres de {LOOKBACK_WINDOW}h ({N_SENSORS} capteurs)")
    print(f"  Train brut      : {len(y_tr_raw)} | positifs : {y_tr_raw.sum()} ({y_tr_raw.mean()*100:.1f}%)")
    print(f"  Validation      : {len(y_val)} | positifs : {y_val.sum()} ({y_val.mean()*100:.1f}%)")
    print(f"  Test            : {len(y_te)}  | positifs : {y_te.sum()} ({y_te.mean()*100:.1f}%)")

    # ── Temporal Oversampling sur TRAIN uniquement ────────────────────────────
    X_tr_aug, y_tr_aug = temporal_oversample(X_tr_raw, y_tr_raw, TARGET_POS_RATIO, RANDOM_STATE)
    print(f"\n  Apres Temporal Oversampling (cible {TARGET_POS_RATIO*100:.0f}%) :")
    print(f"  Train augmente  : {len(y_tr_aug)} sequences | positifs : {y_tr_aug.sum()} ({y_tr_aug.mean()*100:.1f}%)")

    # ── Standardisation (fit sur Train aug, transform sur Val/Test) ───────────
    sc = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_tr_aug)
    X_val_sc = sc.transform(X_val)
    X_te_sc  = sc.transform(X_te)

    n_pos_tr = int(y_tr_aug.sum())
    n_neg_tr = int(len(y_tr_aug) - n_pos_tr)
    pw  = float(np.sqrt(n_neg_tr / max(n_pos_tr, 1)))
    lam = LAMBDA_BASE * float(np.sqrt(n_pos_tr / len(y_tr_aug)))
    print(f"  pos_weight={pw:.2f} | lambda_fa={lam:.4f}\n")

    # ======== 1. Random Forest ========
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_tr_sc, y_tr_aug)
    rf_probas_val  = rf.predict_proba(X_val_sc)[:, 1]
    rf_probas_te   = rf.predict_proba(X_te_sc)[:, 1]
    thr_rf, _      = find_best_threshold(rf_probas_val, y_val)
    y_pred_rf      = (rf_probas_te > thr_rf).astype(int)
    acc_r  = accuracy_score(y_te, y_pred_rf)
    f1_r   = f1_score(y_te, y_pred_rf, zero_division=0)
    pr_r   = precision_score(y_te, y_pred_rf, zero_division=0)
    rec_r  = recall_score(y_te, y_pred_rf, zero_division=0)
    t_rf   = time.time() - t0

    # ======== 2. Gradient Boosting ========
    t0 = time.time()
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gbm.fit(X_tr_sc, y_tr_aug)
    gbm_probas_val = gbm.predict_proba(X_val_sc)[:, 1]
    gbm_probas_te  = gbm.predict_proba(X_te_sc)[:, 1]
    thr_gbm, _     = find_best_threshold(gbm_probas_val, y_val)
    y_pred_g       = (gbm_probas_te > thr_gbm).astype(int)
    acc_g  = accuracy_score(y_te, y_pred_g)
    f1_g   = f1_score(y_te, y_pred_g, zero_division=0)
    pr_g   = precision_score(y_te, y_pred_g, zero_division=0)
    rec_g  = recall_score(y_te, y_pred_g, zero_division=0)
    t_gbm  = time.time() - t0

    # ======== 3. FaNet v4.0 LSTM + Seuil Adaptatif ========
    t0 = time.time()
    n_feat_per_step = X_tr_aug.shape[1] // LOOKBACK_WINDOW  # = 3 (capteurs)
    ffa = TorchFaNet(
        n_features=n_feat_per_step,
        hidden_dim=32,
        n_outputs=1,
        seed=RANDOM_STATE,
        seq_len=LOOKBACK_WINDOW,
    )
    ffa.fit(
        X_tr_sc, y_tr_aug,
        epochs=EPOCHS, lr=LR, batch_size=64, optimizer="adamw",
        verbose=False, lambda_fa=lam, pos_weight=pw, task="classification"
    )
    # Calibration du seuil sur le split de validation
    fa_probas_val = ffa.predict_proba(X_val_sc.astype(np.float32)).flatten()
    thr_fa, best_f1_val = find_best_threshold(fa_probas_val, y_val)

    # Application du seuil calibre sur le Test
    fa_probas_te = ffa.predict_proba(X_te_sc.astype(np.float32)).flatten()
    y_pred_fa    = (fa_probas_te > thr_fa).astype(int)
    acc_f  = accuracy_score(y_te, y_pred_fa)
    f1_f   = f1_score(y_te, y_pred_fa, zero_division=0)
    pr_f   = precision_score(y_te, y_pred_fa, zero_division=0)
    rec_f  = recall_score(y_te, y_pred_fa, zero_division=0)
    t_fa   = time.time() - t0

    print("=" * 95)
    print("  RESULTATS DEFINITIFS (Prediction Risque Hydrique +6h / Seuil Calibre)")
    print("=" * 95)
    print_result("Random Forest", acc_r, f1_r, pr_r, rec_r, t_rf, thr_rf)
    print_result("Gradient Boosting", acc_g, f1_g, pr_g, rec_g, t_gbm, thr_gbm)
    print_result("FaNet v4.0 LSTM (Calibre)", acc_f, f1_f, pr_f, rec_f, t_fa, thr_fa)

    print(f"\n  [CALIBRATION FaNet] Meilleur F1 sur Validation = {best_f1_val*100:.2f}% | Seuil retenu = {thr_fa:.3f}")

    print("\n[VERDICT TIME-SERIES IOT v4.1]")
    if f1_f > f1_r and f1_f > f1_g:
        print("  SUCCES ABSOLU: FaNet LSTM depasse Random Forest et GBM sur l'IoT temporel calibre.")
    elif f1_f > 0.70 and rec_f > 0.80:
        print("  SUCCES MAJEUR: FaNet LSTM calibre atteint F1 > 70% avec un Rappel clinique fort.")
        print("  Systeme pret pour un deploiement en alerte preventive sur parcelles agricoles.")
    elif f1_f > 0.50:
        print("  PROGRES SIGNIFICATIF: L'oversampling + calibration ameliorent substantiellement le F1.")
        print(f"  FaNet IoT passe de ~43% a {f1_f*100:.1f}% de F1. Prêt pour une iteration architecturale.")
    else:
        print(f"  F1={f1_f*100:.1f}%. L'architecture LSTM seule ne capte pas encore l'entropie temporelle.")
        print("  Prochaine etape architecturale : Bidirectionnel LSTM ou Transformer Temporel.")

if __name__ == "__main__":
    run_iot_test()
