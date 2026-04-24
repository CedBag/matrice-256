import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# S'assurer que bokono est dans le PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bokono.ml.torch_network import TorchFaNet

def run_agri_poc():
    print("="*60)
    print("  POC 2 : SMART-AGRI (INTELLIGENCE AGRICOLE RÉELLE)")
    print("  Matrice-256 (FaNet v4.0) vs Analyse de Sols IoT (Crop)")
    print("="*60)
    
    # 1. Chargement des données
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poc", "agri_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return

    # Nettoyage des colonnes fantômes du CSV (ex: ',,')
    df = pd.read_csv(csv_path).dropna(axis=1, how='all')
    
    # On isole une culture exigeante en eau pour le "Test Fâ" : Le Riz (Rice)
    # Cela permet de tester si l'ontologie détecte l'équilibre "Humidité/Précipitation"
    print(f"Cultures disponibles : {df['label'].unique()[:5]}...")
    target_crop = 'rice'
    print(f"-> Modélisation Neuro-Symbolique d'Aptitude pour le : {target_crop.upper()}")
    
    df['is_target'] = (df['label'] == target_crop).astype(int)
    
    # Features : N, P, K, temperature, humidity, ph, rainfall
    X = df.drop(columns=['label', 'is_target']).values
    y = df['is_target'].values

    # 2. Séparation et Standardisation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[INFO] Capteurs IoT analysés : {X.shape[1]} variables (N, P, K, Temp, ...)")
    print(f"[INFO] Points de donnée Sols: {X_train.shape[0]} entraînement, {X_test.shape[0]} validation")

    # 3. Modèle Fâ
    model = TorchFaNet(n_features=X_train.shape[1], hidden_dim=16, n_outputs=1, seed=12)

    # 4. Apprentissage
    print("[INFO] Synthèse de l'apprentissage (Epochs: 40, FaLoss Lambda: 0.1)...")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=40, 
        lr=0.01, 
        batch_size=64, 
        optimizer="adam",
        verbose=False,
        lambda_fa=0.1
    )

    # Évaluation Basique
    preds_test = model.predict_proba(X_test_scaled)
    y_pred_bin = (preds_test > 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred_bin == y_test)
    
    print(f"\n[RÉSULTAT] Convergence finale. Loss = {history['loss'][-1]:.4f}")
    print(f"[RÉSULTAT] Précision de la recommandation agricole = {accuracy * 100:.2f}%")

    # 5. Diagnostic Terrain (Zéro Bullshit)
    print("\n" + "="*40)
    print("  ANALYSE D'UN CHAMP - PARCELLE #5 (Test Set)")
    print("="*40)
    
    # On va chercher un exemple qui EST vraiment la culture ciblée pour voir comment le Fâ réagit
    idx = np.where(y_test == 1)[0][0]
    parcelle_x = X_test_scaled[idx:idx+1]
    
    pred_val, interpretations = model.predict(parcelle_x)
    diag_du = interpretations[0]

    prob = float(pred_val[0][0]) * 100
    
    print(f"Données brutes de la parcelle   : {np.round(X_test[idx], 1)}")
    print(f"Vérité (Était-ce du {target_crop} ?) : {'OUI' if y_test[idx] == 1 else 'NON'}")
    print(f"Probabilité du réseau Neuronal  : {prob:.1f}%\n")
    
    print("--- ONTOLOGIE ALGORITHMIQUE DU SOL ---")
    print(f"-> Node Combinatoire       : {diag_du['state_id']}")
    print(f"-> Empreinte Binaire       : {diag_du['binary_matrix']}")
    print(f"-> État Sémantique Calculé : {diag_du['ontological_state']}")
    print(f"-> Stabilité du Système    : {diag_du['system_polarity']}")
    
    print("\n[RECOMMANDATION AGRICULTEUR]")
    if prob > 80.0:
        print("Conditions climato-édaphiques IDÉALES. Le modèle confirme la viabilité.")
    elif prob > 50.0:
        print("Conditions ACCEPTABLES, mais un déséquilibre ontologique mineur est détecté.")
    else:
        print("SOL HOSTILE à cette culture. Risque élevé de perte.")

if __name__ == "__main__":
    run_agri_poc()
