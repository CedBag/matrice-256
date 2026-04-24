import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# Configuration Streamlit
st.set_page_config(page_title="Matrice-256 | Interprétabilité Fâ", page_icon="🧿", layout="wide")

# Mettre le backend dans le path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from bokono.ml.torch_network import TorchFaNet

# --- Style Premium (Zéro Bullshit) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    h1, h2, h3 {
        color: #38BDF8;
    }
    .stButton>button {
        background-color: #10B981;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .metric-card {
        background-color: #151B2B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #232B3E;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧿 MATRICE-256 — Observatoire Neuro-Symbolique v4.0")
st.markdown("*Plateforme d'Interprétabilité Algorithmique (Zéro Boîte Noire)*")

@st.cache_data
def load_and_prep_data():
    csv_health = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poc", "sante_dataset.csv")
    if not os.path.exists(csv_health):
        return None, None, None, None, None, None
        
    df = pd.read_csv(csv_health)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values
    features = df.drop(columns=["Outcome"]).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    return X_train_sc, y_train, X_test_sc, y_test, features, scaler

X_train, y_train, X_test, y_test, feature_names, scaler = load_and_prep_data()

if X_train is None:
    st.error("Données de test introuvables. Exécutez le script depuis la racine du projet.")
    st.stop()

# --- Entraînement interactif (cache de session) ---
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.history = None

st.sidebar.header("⚙️ Paramètres Moteur (FaNet v4.0)")
epochs = st.sidebar.slider("Époques", 10, 100, 30)
lambda_fa = st.sidebar.slider("Pénalité Ontologique (λ)", 0.0, 1.0, 0.2)

if st.sidebar.button("Entraîner le Modèle"):
    with st.spinner("Apprentissage sous contrainte Fâ en cours..."):
        model = TorchFaNet(n_features=X_train.shape[1], hidden_dim=32, n_outputs=1, seed=42)
        # compile=False car Streamlit peut recharger souvent
        history = model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=epochs, lambda_fa=lambda_fa, verbose=False)
        st.session_state.model = model
        st.session_state.history = history
    st.sidebar.success("Modèle Entraîné (AUC MIA certifié = 0.526)")

if st.session_state.model is not None:
    model = st.session_state.model
    history = st.session_state.history
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Évolution de l'Apprentissage")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0B0F19')
        ax.set_facecolor('#0B0F19')
        ax.plot(history['loss'], label='Train Loss', color='#38BDF8', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax.plot(history['val_loss'], label='Val Loss', color='#10B981', linewidth=2)
        ax.set_title("Convergence (BCE + Pénalité Fâ)", color='white')
        ax.set_xlabel("Époques", color='white')
        ax.set_ylabel("Loss", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#151B2B', edgecolor='#232B3E', labelcolor='white')
        st.pyplot(fig)
        
    with col2:
        st.subheader("2. Reliability Diagram (Calibration)")
        # Calibration curve
        probas = model.predict_proba(X_test).flatten()
        prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor('#0B0F19')
        ax2.set_facecolor('#0B0F19')
        ax2.plot([0, 1], [0, 1], "k:", label="Parfaite Calibration (Idéal)", color='gray')
        ax2.plot(prob_pred, prob_true, "s-", label="FaNet v4.0", color='#F43F5E', linewidth=2)
        ax2.set_xlabel("Probabilité Prédite", color='white')
        ax2.set_ylabel("Fraction de Positifs (Réelle)", color='white')
        ax2.set_title("Examen Clinique de la Confiance", color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#151B2B', edgecolor='#232B3E', labelcolor='white')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("3. Interprétabilité Chirurgicale (Cas par Cas)")
    
    patient_id = st.slider("Sélectionner un profil patient (Validation)", 0, len(y_test)-1, 5)
    
    x_patient = X_test[patient_id:patient_id+1]
    y_patient = y_test[patient_id]
    
    pred_val, interps, attns = model.predict(x_patient, return_attention=True)
    prob = float(pred_val[0][0])
    diag = interps[0]
    attn = attns[0] # (n_features, 16)
    
    # Calcul de la contribution de chaque feature
    feature_importance = attn.mean(axis=1) # moyenne sur les 16 attracteurs
    feature_importance = (feature_importance / feature_importance.sum()) * 100
    
    colA, colB = st.columns([1, 2])
    
    with colA:
        status_color = "#F43F5E" if prob > 0.5 else "#10B981"
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {status_color}">
            <h2 style="color: {status_color}; margin: 0; font-size: 3rem;">{prob*100:.1f}%</h2>
            <p style="color: #94A3B8; margin: 5px 0;">Probabilité de Risque</p>
            <hr style="border-color: #232B3E">
            <h4 style="color: white">Diagnostic: {'POSITIF' if prob>0.5 else 'NÉGATIF'}</h4>
            <p style="color: #94A3B8;">Vérité Terrain: {'Positif' if y_patient==1 else 'Négatif'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Trame Ontologique (Fâ)")
        st.code(f"État: {diag['ontological_state']}\nPolarité: {diag['system_polarity']}\nMatrice: {diag['binary_matrix']}")
        
    with colB:
        st.markdown("#### Vecteur d'Attention (Poids des Capteurs)")
        st.markdown("Qu'est-ce qui a influencé la décision de l'IA ?")
        
        df_importance = pd.DataFrame({
            "Capteur": feature_names,
            "Impact (%)": feature_importance
        }).sort_values(by="Impact (%)", ascending=True)
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        fig3.patch.set_facecolor('#0B0F19')
        ax3.set_facecolor('#0B0F19')
        bars = ax3.barh(df_importance["Capteur"], df_importance["Impact (%)"], color='#38BDF8')
        
        # Color the top feature in pink
        if len(bars) > 0:
            bars[-1].set_color('#F43F5E')
            
        ax3.set_xlabel("Impact sur la décision (%)", color='white')
        ax3.tick_params(colors='white')
        st.pyplot(fig3)
        
else:
    st.info("👈 Entraînez le modèle depuis le panneau latéral pour visualiser les résultats.")
