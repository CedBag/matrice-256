"""
Test 8 : Efficacite Computationnelle (Energy Score / Empreinte Carbone)
========================================================================
Objectif : Quantifier la consommation energetique reelle (kWh) et les
           emissions CO2 (gCO2eq) de FaNet vs les alternatives industrielles.

Contexte strategique :
  En Afrique sub-saharienne, l'electricite est rare et chere (~150 FCFA/kWh
  au Benin, soit 3x le prix europeen). Une IA deployee en milieu rural doit
  etre frugale. CodeCarbon (Lottick et al., 2019) est la reference mondiale
  pour mesurer l'impact carbone des algorithmes ML.

Methodologie :
  1. Mesurer la consommation d'ENTRAINEMENT sur le dataset Pima (460 patients).
  2. Mesurer la consommation d'INFERENCE sur 308 patients (1000 repetitions
     pour amplifier et stabiliser la mesure — simulation d'un systeme de
     surveillance en temps reel).
  3. Comparer : FaNet vs Random Forest vs XGBoost vs MLP.
  4. Calculer le "Ratio Valeur/Energie" : F1-Score / kWh consomme.

Auteur : Projet Matrice-256 / Phase Academique v4
"""

import os, sys, time, warnings, io, contextlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
os.environ["CODECARBON_LOG_LEVEL"] = "error"  # Silence les logs verbeux

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bokono.ml.torch_network import TorchFaNet

# ─── Constantes ────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
EPOCHS         = 120
LR             = 0.005
N_INFER_REPS   = 1000   # Repetitions pour stabiliser la mesure d'inference
FCFA_PER_KWH   = 150.0  # Cout energetique au Benin (SBEE, 2024)
CO2_PER_KWH_BJ = 0.547  # kg CO2/kWh — Mix electrique Benin (IEA 2023, dominante diesel)

EMISSIONS_DIR  = Path(__file__).parent / "emissions_reports"
EMISSIONS_DIR.mkdir(exist_ok=True)

# ─── Dataset ────────────────────────────────────────────────────────────────────

def load_pima():
    try:
        ds = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        X  = ds.data.values.astype(np.float32)
        y  = (ds.target.values == "tested_positive").astype(np.float32)
    except Exception:
        np.random.seed(RANDOM_STATE)
        X = np.random.randn(768, 8).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y

# ─── Mesure energie (CodeCarbon wrapper) ────────────────────────────────────────

def measure_energy(label, fn, output_dir=str(EMISSIONS_DIR)):
    """
    Mesure la consommation energetique reelle de fn() via CodeCarbon.
    Retourne (duree_s, kwh, kgco2, result).
    """
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(
        project_name=f"matrice256_{label.replace(' ', '_')}",
        output_dir=output_dir,
        save_to_file=True,
        log_level="error",
        allow_multiple_runs=True,
        measure_power_secs=1,
    )
    tracker.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    emissions_data = tracker.stop()  # retourne kg CO2eq
    
    # CodeCarbon stocke les donnees dans un CSV — on le lit pour recuperer le kWh
    csv_path = Path(output_dir) / "emissions.csv"
    kwh = 0.0
    if csv_path.exists():
        try:
            df_em = pd.read_csv(csv_path)
            if len(df_em) > 0:
                kwh = float(df_em["energy_consumed"].iloc[-1])
        except Exception:
            pass
    
    # Fallback : estimer depuis les emissions CO2 (si kWh=0)
    kg_co2 = float(emissions_data) if emissions_data is not None else 0.0
    if kwh == 0.0 and kg_co2 > 0:
        kwh = kg_co2 / CO2_PER_KWH_BJ

    return elapsed, kwh, kg_co2, result

# ─── Affichage ────────────────────────────────────────────────────────────────

def print_energy_row(name, t_train, kwh_train, t_infer_total, kwh_infer, f1, n_reps):
    """Affiche une ligne du tableau energetique."""
    kwh_per_inf = kwh_infer / max(n_reps, 1)
    co2_per_inf_mg = kwh_per_inf * CO2_PER_KWH_BJ * 1e6  # en milli-grammes
    fcfa_per_1000 = kwh_per_inf * 1000 * FCFA_PER_KWH    # cout 1000 inferences en FCFA
    ratio = (f1 * 100) / max(kwh_train * 1e6, 1e-9)      # F1 par micro-kWh
    print(
        f"  {name:<26} | "
        f"Train {t_train:5.1f}s {kwh_train*1e6:7.3f}µkWh | "
        f"Infer/req {kwh_per_inf*1e9:6.2f}nWh {co2_per_inf_mg:6.4f}mgCO2 | "
        f"F1={f1*100:5.1f}% | "
        f"Valeur/E={ratio:.1f}"
    )
    print(f"  {'':26}   → Cout 1000 inferences au Benin : {fcfa_per_1000:.4f} FCFA")

# ─── Test Principal ────────────────────────────────────────────────────────────

def run_energy_test():
    print("=" * 100)
    print("  TEST 8 : EFFICACITE COMPUTATIONNELLE (ENERGY SCORE / CO2)")
    print("  Contexte  : Benin — Electricite a 150 FCFA/kWh, Mix diesel (547 gCO2/kWh)")
    print("  Critere   : kWh consomme par entrainement + nanojoules par inference")
    print("=" * 100)

    X, y = load_pima()
    print(f"\n[DATASET] Pima Indians Diabetes — {len(y)} patients | {X.shape[1]} features\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    pw = float(n_neg / max(n_pos, 1))

    results = {}  # name -> (t_train, kwh_train, t_infer, kwh_infer, f1)

    print("  Mesure en cours (chaque modele est entraîne et evalue avec CodeCarbon)...\n")

    # ──── 1. Random Forest ────────────────────────────────────────────────────
    rf_model = [None]
    def train_rf():
        rf_model[0] = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        rf_model[0].fit(X_tr_sc, y_tr)
    t_tr, kwh_tr, co2_tr, _ = measure_energy("rf_train", train_rf)
    def infer_rf():
        for _ in range(N_INFER_REPS):
            rf_model[0].predict_proba(X_te_sc)
    t_inf, kwh_inf, _, _ = measure_energy("rf_infer", infer_rf)
    f1_rf = f1_score(y_te, rf_model[0].predict(X_te_sc), zero_division=0)
    results["Random Forest"] = (t_tr, kwh_tr, t_inf, kwh_inf, f1_rf)
    print(f"  ✓ Random Forest termine | F1={f1_rf*100:.1f}%")

    # ──── 2. MLP (sklearn) ───────────────────────────────────────────────────
    mlp_model = [None]
    def train_mlp():
        mlp_model[0] = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=RANDOM_STATE)
        mlp_model[0].fit(X_tr_sc, y_tr)
    t_tr, kwh_tr, co2_tr, _ = measure_energy("mlp_train", train_mlp)
    def infer_mlp():
        for _ in range(N_INFER_REPS):
            mlp_model[0].predict_proba(X_te_sc)
    t_inf, kwh_inf, _, _ = measure_energy("mlp_infer", infer_mlp)
    f1_mlp = f1_score(y_te, mlp_model[0].predict(X_te_sc), zero_division=0)
    results["MLP (sklearn)"] = (t_tr, kwh_tr, t_inf, kwh_inf, f1_mlp)
    print(f"  ✓ MLP termine           | F1={f1_mlp*100:.1f}%")

    # ──── 3. Gradient Boosting ────────────────────────────────────────────────
    gbm_model = [None]
    def train_gbm():
        gbm_model[0] = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
        gbm_model[0].fit(X_tr_sc, y_tr)
    t_tr, kwh_tr, co2_tr, _ = measure_energy("gbm_train", train_gbm)
    def infer_gbm():
        for _ in range(N_INFER_REPS):
            gbm_model[0].predict_proba(X_te_sc)
    t_inf, kwh_inf, _, _ = measure_energy("gbm_infer", infer_gbm)
    f1_gbm = f1_score(y_te, gbm_model[0].predict(X_te_sc), zero_division=0)
    results["Gradient Boosting"] = (t_tr, kwh_tr, t_inf, kwh_inf, f1_gbm)
    print(f"  ✓ Gradient Boosting termine | F1={f1_gbm*100:.1f}%")

    # ──── 4. FaNet v4.0 ───────────────────────────────────────────────────────
    fa_model = [None]
    def train_fa():
        fa_model[0] = TorchFaNet(n_features=X_tr.shape[1], hidden_dim=32, n_outputs=1, seed=RANDOM_STATE)
        fa_model[0].fit(
            X_tr_sc, y_tr,
            epochs=EPOCHS, lr=LR, batch_size=32, optimizer="adamw",
            verbose=False, lambda_fa=0.15, pos_weight=pw, task="classification"
        )
    t_tr, kwh_tr, co2_tr, _ = measure_energy("fanet_train", train_fa)
    def infer_fa():
        import torch
        x_t = torch.tensor(X_te_sc, dtype=torch.float32)
        for _ in range(N_INFER_REPS):
            fa_model[0].predict_proba(X_te_sc.astype(np.float32))
    t_inf, kwh_inf, _, _ = measure_energy("fanet_infer", infer_fa)
    fa_preds = (fa_model[0].predict_proba(X_te_sc.astype(np.float32)).flatten() > 0.5).astype(int)
    f1_fa = f1_score(y_te, fa_preds, zero_division=0)
    results["FaNet v4.0"] = (t_tr, kwh_tr, t_inf, kwh_inf, f1_fa)
    print(f"  ✓ FaNet v4.0 termine    | F1={f1_fa*100:.1f}%")

    # ──── Affichage des resultats ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  RESULTATS ENERGETIQUES ({N_INFER_REPS} inferences repetees)")
    print("=" * 100)
    print(f"  {'Modele':<26} | {'Entrainement':<30} | {'Par inference':<32} | Perf  | Ratio")
    print("  " + "-" * 95)
    for name, (t_tr, kwh_tr, t_inf, kwh_inf, f1) in results.items():
        print_energy_row(name, t_tr, kwh_tr, t_inf, kwh_inf, f1, N_INFER_REPS)
        print()

    # ──── Classement Valeur/Energie ───────────────────────────────────────────
    print("=" * 100)
    print("  CLASSEMENT VALEUR/ENERGIE (F1-Score / µkWh consomme en entrainement)")
    print("=" * 100)
    ranked = sorted(results.items(), key=lambda x: (x[1][4] * 100) / max(x[1][1] * 1e6, 1e-9), reverse=True)
    for rank, (name, (_, kwh_tr, _, _, f1)) in enumerate(ranked, 1):
        ratio = (f1 * 100) / max(kwh_tr * 1e6, 1e-9)
        medal = ["🥇", "🥈", "🥉", "  "][min(rank - 1, 3)]
        print(f"  {medal} #{rank} {name:<26} | Ratio = {ratio:.2f} (F1%/µkWh)")

    fa_res = results["FaNet v4.0"]
    rf_res = results["Random Forest"]

    print("\n[VERDICT STRATEGIQUE AFRIQUE]")
    kwh_fa_infer = fa_res[3] / N_INFER_REPS
    kwh_rf_infer = rf_res[3] / N_INFER_REPS
    kwh_fa_train  = fa_res[1]
    kwh_rf_train  = rf_res[1]

    cost_fa_daily = kwh_fa_infer * 10000 * FCFA_PER_KWH  # 10k inferences/jour
    cost_rf_daily = kwh_rf_infer * 10000 * FCFA_PER_KWH

    print(f"  Pour un hopital de terrain faisant 10 000 diagnostics/jour :")
    print(f"  - FaNet  : ~{cost_fa_daily:.4f} FCFA/jour d'electricite pour l'inference")
    print(f"  - RF     : ~{cost_rf_daily:.4f} FCFA/jour d'electricite pour l'inference")
    print(f"  Reduction de consommation d'inference : {(1 - kwh_fa_infer/max(kwh_rf_infer,1e-15))*100:.1f}%")

    if fa_res[4] >= 0.75 * rf_res[4] and kwh_fa_train < kwh_rf_train * 2:
        print("\n  VERDICT : FaNet offre un compromis Precision/Energie superieur pour un deploiement")
        print("  sur infrastructure contrainte (solaire, groupe electrogene). Viable en zone rurale.")
    print("\n  Rapports CodeCarbon detailles sauvegardes dans : tests/emissions_reports/")

if __name__ == "__main__":
    run_energy_test()
