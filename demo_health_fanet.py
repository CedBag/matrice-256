import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# S'assurer que bokono est dans le PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bokono.ml.torch_network import TorchFaNet

def run_health_poc():
    print("="*60)
    print("  POC 1 : HEALTH-TECH (DIAGNOSTIC MÉDICAL HYBRIDE)")
    print("  Matrice-256 (FaNet v4.0) vs Pima Indians Diabetes")
    print("="*60)
    
    # 1. Chargement des données brutes
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poc", "sante_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values

    # 2. Préparation des données (Standardisation critique pour PyTorch)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[INFO] Echantillons d'apprentissage : {X_train.shape[0]}")
    print(f"[INFO] Echantillons de test : {X_test.shape[0]}")
    print("[INFO] Initialisation du réseau Neuro-Symbolique : TorchFaNet")

    # 3. Architecture Mathématique Inédite
    # n_features = 8 (Grossesses, Glucose, Pression, etc.)
    model = TorchFaNet(n_features=X_train.shape[1], hidden_dim=32, n_outputs=1, seed=42)

    # 4. Apprentissage guidé par la FaLoss (lambda_fa=0.2)
    print("[INFO] Lancement de l'apprentissage (Epochs: 20, Optim: AdamW, AMP: Auto)...")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=30, 
        lr=0.01, 
        batch_size=32, 
        optimizer="adamw",
        verbose=False,
        lambda_fa=0.2  # Pénalité philosophique de l'équilibre Fâ activée
    )

    # 5. Évaluation Industrielle
    preds_test = model.predict_proba(X_test_scaled)
    y_pred_bin = (preds_test > 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred_bin == y_test)
    
    print(f"\n[RÉSULTAT] Moteur entraîné. Loss Finale = {history['loss'][-1]:.4f}")
    print(f"[RÉSULTAT] Précision (Accuracy) sur Test Set = {accuracy * 100:.2f}%")

    # 6. Explicabilité Mathématique (XAI Fâ) - Le "Zero Bullshit"
    print("\n" + "="*40)
    print("  RAPPORT DE DIAGNOSTIC - PATIENT #0 (Test Set)")
    print("="*40)
    
    patient_0_x = X_test_scaled[0:1]
    
    # Inférence en mode Dual (Explicabilité totale)
    pred_val, interpretations = model.predict(patient_0_x)
    diag_du = interpretations[0]

    prob_maladie = float(pred_val[0][0]) * 100
    
    print(f"Données biométriques standardisées : {np.round(patient_0_x[0], 2)}")
    print(f"Vérité Terrain (Malade ?) : {'OUI' if y_test[0] == 1 else 'NON'}")
    print(f"Probabilité Prédite FaNet : {prob_maladie:.1f}%\n")
    
    print("--- ONTOLOGIE ALGORITHMIQUE (Justification du cerveau) ---")
    print(f"-> État Tensoriel Mapping  : {diag_du['state_id']}")
    print(f"-> Signature Combinatoire  : {diag_du['binary_matrix']}")
    print(f"-> Node Ontologique Actif  : {diag_du['ontological_state']}")
    print(f"-> Polarité Sémantique     : {diag_du['system_polarity']}")
    print("\n[CONCLUSION]")
    if prob_maladie > 50.0:
        print("La machine recommande un diagnostic POSITIF fondé sur la brisure d'équilibre biométrique (Polarité alignée au risque).")
    else:
        print("La machine recommande un diagnostic NÉGATIF (Système stable).")


if __name__ == "__main__":
    run_health_poc()
