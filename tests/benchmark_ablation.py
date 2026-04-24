import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# S'assurer que bokono est dans le PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bokono.ml.torch_network import TorchFaNet

def run_ablation_study():
    print("="*60)
    print("  PHASE ACADÉMIQUE : BENCHMARK & ABLATION STUDY (v4.0)")
    print("  FaNet vs Modèles Standards (RandomForest, GBM, MLP)")
    print("="*60)

    # 1. Chargement du dataset Agricole (Prédiction Riz)
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data_poc", "agri_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path).dropna(axis=1, how='all')
    target_crop = 'rice'
    df['is_target'] = (df['label'] == target_crop).astype(int)
    
    X = df.drop(columns=['label', 'is_target']).values
    y = df['is_target'].values

    # 2. Préparation (Train/Test Split rigoureux)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\n[INFO] Dataset chargé : {X.shape[0]} lignes, {X.shape[1]} capteurs.")
    print("[INFO] Lancement des combats...\n")

    results = []

    # ---------------------------------------------------------
    # MODÈLE 1 : Logistic / Baseline Neuronale (MLP Standard)
    # ---------------------------------------------------------
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=42)
    mlp.fit(X_train_sc, y_train)
    t_mlp = time.time() - t0
    y_pred_mlp = mlp.predict(X_test_sc)
    results.append({
        "Model": "Standard MLP Net",
        "Accuracy": accuracy_score(y_test, y_pred_mlp),
        "F1-Score": f1_score(y_test, y_pred_mlp, zero_division=0),
        "Time (s)": t_mlp,
        "Constraint": "Aucune (Boîte Noire)"
    })

    # ---------------------------------------------------------
    # MODÈLE 2 : Random Forest
    # ---------------------------------------------------------
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_sc, y_train)
    t_rf = time.time() - t0
    y_pred_rf = rf.predict(X_test_sc)
    results.append({
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "F1-Score": f1_score(y_test, y_pred_rf, zero_division=0),
        "Time (s)": t_rf,
        "Constraint": "Arbres de décision"
    })

    # ---------------------------------------------------------
    # MODÈLE 3 : Gradient Boosting (Alternative à XGBoost)
    # ---------------------------------------------------------
    t0 = time.time()
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm.fit(X_train_sc, y_train)
    t_gbm = time.time() - t0
    y_pred_gbm = gbm.predict(X_test_sc)
    results.append({
        "Model": "Gradient Boosting",
        "Accuracy": accuracy_score(y_test, y_pred_gbm),
        "F1-Score": f1_score(y_test, y_pred_gbm, zero_division=0),
        "Time (s)": t_gbm,
        "Constraint": "Booster tabulaire"
    })

    # ---------------------------------------------------------
    # MODÈLE 4 : FaNet SANS Contrainte (Étude d'ablation)
    # ---------------------------------------------------------
    t0 = time.time()
    fanet_no_loss = TorchFaNet(n_features=X_train.shape[1], hidden_dim=16, n_outputs=1, seed=12)
    fanet_no_loss.fit(X_train_sc, y_train, epochs=40, lr=0.01, batch_size=64, optimizer="adam", verbose=False, lambda_fa=0.0)
    t_fno = time.time() - t0
    preds_fno = fanet_no_loss.predict_proba(X_test_sc)
    y_pred_fno = (preds_fno > 0.5).astype(int).flatten()
    results.append({
        "Model": "FaNet (Ablation = 0.0)",
        "Accuracy": accuracy_score(y_test, y_pred_fno),
        "F1-Score": f1_score(y_test, y_pred_fno, zero_division=0),
        "Time (s)": t_fno,
        "Constraint": "Architecture Fâ Seule"
    })

    # ---------------------------------------------------------
    # MODÈLE 5 : FaNet COMPLETM (Avec Pénalité Cosmologique Fâ)
    # ---------------------------------------------------------
    t0 = time.time()
    fanet_full = TorchFaNet(n_features=X_train.shape[1], hidden_dim=16, n_outputs=1, seed=12)
    fanet_full.fit(X_train_sc, y_train, epochs=40, lr=0.01, batch_size=64, optimizer="adam", verbose=False, lambda_fa=0.1)
    t_ffull = time.time() - t0
    preds_ffull = fanet_full.predict_proba(X_test_sc)
    y_pred_ffull = (preds_ffull > 0.5).astype(int).flatten()
    results.append({
        "Model": "FaNet v3.0 (Complet)",
        "Accuracy": accuracy_score(y_test, y_pred_ffull),
        "F1-Score": f1_score(y_test, y_pred_ffull, zero_division=0),
        "Time (s)": t_ffull,
        "Constraint": "Ontologie + FaLoss=0.1 + XAI"
    })

    # ---------------------------------------------------------
    # AFFICHAGE FINAL (Format Markdown)
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Modèle':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'Temps (s)':<10} | {'Contrainte Structurelle'}")
    print("-"*80)
    for r in results:
        print(f"{r['Model']:<25} | {r['Accuracy'] * 100:>6.2f}%   | {r['F1-Score'] * 100:>6.2f}%   | {r['Time (s)']:>7.3f}s   | {r['Constraint']}")
    
    print("="*80)

    # Analyse :
    print("\n[ANALYSE DU CHERCHEUR]")
    print("1. Si la pénalité cosmologique du Fâ (lambda_fa=0.1) a maintenu l'Accuracy au niveau des GBE ou RF, cela signifie que la contrainte culturelle est un Biais Inductif valide (elle n'est pas un bruit destructeur).")
    print("2. FaNet fournit l'état ontologique exact et l'explication binaire Fâ, chose dont GradientBoosting ou MLP sont rigoureusement incapables (Black Box).")
    print("La preuve académique de la non-destruction de la performance par une contrainte symbolique africaine est un apport majeur.")

if __name__ == "__main__":
    run_ablation_study()
