import os
import sys
import threading
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import dash
from dash import dcc, html, Input, Output, State
import dash.exceptions

# S'assurer que bokono est dans le PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from bokono.ml.torch_network import TorchFaNet

# ==============================================================================
# INITIALISATION DU MOTEUR (Training en Arrière-plan pour éviter les latences UI)
# ==============================================================================
print("[INITIALISATION] Entraînement des Modèles FaNet v4.0 pour l'Interface...")
health_model = None
health_scaler = None
health_X_test = None
health_y_test = None

agri_model = None
agri_scaler = None
agri_X_test = None
agri_y_test = None
agri_target = 'rice'

# 1. Santé
csv_health = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poc", "sante_dataset.csv")
if os.path.exists(csv_health):
    df_h = pd.read_csv(csv_health)
    X_h = df_h.drop(columns=["Outcome"]).values
    y_h = df_h["Outcome"].values
    X_train_h, health_X_test, y_train_h, health_y_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    health_scaler = StandardScaler()
    X_train_h_sc = health_scaler.fit_transform(X_train_h)
    health_model = TorchFaNet(n_features=X_train_h.shape[1], hidden_dim=32, n_outputs=1, seed=42)
    health_model.fit(X_train_h_sc, y_train_h, epochs=30, lr=0.01, verbose=False, lambda_fa=0.2)

# 2. Agri
csv_agri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poc", "agri_dataset.csv")
if os.path.exists(csv_agri):
    df_a = pd.read_csv(csv_agri).dropna(axis=1, how='all')
    df_a['is_target'] = (df_a['label'] == agri_target).astype(int)
    X_a = df_a.drop(columns=['label', 'is_target']).values
    y_a = df_a['is_target'].values
    X_train_a, agri_X_test, y_train_a, agri_y_test = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
    agri_scaler = StandardScaler()
    X_train_a_sc = agri_scaler.fit_transform(X_train_a)
    agri_model = TorchFaNet(n_features=X_train_a.shape[1], hidden_dim=16, n_outputs=1, seed=12)
    agri_model.fit(X_train_a_sc, y_train_a, epochs=40, lr=0.01, verbose=False, lambda_fa=0.1)

print("[INITIALISATION] Moteurs Synchronisés. Démarrage de l'Application Dash...")

# ==============================================================================
# INTERFACE UTILISATEUR (DASH) - STYLE "Zero Bullshit" PREMIUM
# ==============================================================================
app = dash.Dash(__name__, update_title=None)
app.title = "Matrice-256 | Observatoire"

CSS_COLORS = {
    "bg": "#0B0F19",          # Deep space dark
    "card": "#151B2B",        # Slightly lighter dark
    "border": "#232B3E",      # Subtle borders
    "text": "#E2E8F0",        # Readable white
    "neon": "#38BDF8",        # Electric Blue
    "success": "#10B981",     # Neon Green
    "danger": "#F43F5E",      # Neon Red
    "accent": "#8B5CF6",      # Pinkish purple
}

app.layout = html.Div(style={
    "backgroundColor": CSS_COLORS["bg"],
    "color": CSS_COLORS["text"],
    "fontFamily": "Inter, 'Segoe UI', Roboto, sans-serif",
    "minHeight": "100vh",
    "padding": "40px",
    "boxSizing": "border-box"
}, children=[
    
    # HEADER
    html.Div(style={"textAlign": "center", "marginBottom": "50px", "animation": "fadeIn 1s"}, children=[
        html.H1("MATRICE-256", style={"color": CSS_COLORS["neon"], "letterSpacing": "4px", "margin": "0", "fontSize": "3rem"}),
        html.P("Réseau Neuro-Symbolique d'Architecture Fâ (v4.0)", style={"color": "#94A3B8", "fontSize": "1.2rem", "marginTop": "10px"})
    ]),

    # TABS
    dcc.Tabs(id="tabs-main", value="tab-health", 
             style={"maxWidth": "800px", "margin": "0 auto 30px auto", "backgroundColor": CSS_COLORS["bg"]}, 
             parent_style={"backgroundColor": CSS_COLORS["bg"]},
             children=[
        dcc.Tab(label="🩺 Diagnostic Médical (Health-Tech)", value="tab-health",
                style={"backgroundColor": CSS_COLORS["card"], "border": "none", "borderBottom": f"2px solid {CSS_COLORS['border']}", "padding": "15px", "color": CSS_COLORS["text"], "fontWeight": "bold"},
                selected_style={"backgroundColor": CSS_COLORS["border"], "border": "none", "borderBottom": f"3px solid {CSS_COLORS['success']}", "padding": "15px", "color": CSS_COLORS["success"], "fontWeight": "bold"}),
        dcc.Tab(label="🌾 Capteurs Sols (Smart-Agri)", value="tab-agri",
                style={"backgroundColor": CSS_COLORS["card"], "border": "none", "borderBottom": f"2px solid {CSS_COLORS['border']}", "padding": "15px", "color": CSS_COLORS["text"], "fontWeight": "bold"},
                selected_style={"backgroundColor": CSS_COLORS["border"], "border": "none", "borderBottom": f"3px solid {CSS_COLORS['neon']}", "padding": "15px", "color": CSS_COLORS["neon"], "fontWeight": "bold"}),
    ]),
    
    # CONTENT CONTAINER
    html.Div(id="tab-content", style={"maxWidth": "800px", "margin": "0 auto"})
])

@app.callback(
    Output("tab-content", "children"),
    Input("tabs-main", "value")
)
def render_tab(tab_name):
    if tab_name == "tab-health":
        if health_model is None:
            return html.H3("Fichier de données data_poc/sante_dataset.csv manquant", style={"color": CSS_COLORS["danger"]})
        return html.Div(style={"backgroundColor": CSS_COLORS["card"], "borderRadius": "12px", "padding": "30px", "boxShadow": "0 10px 30px rgba(0,0,0,0.5)"}, children=[
            html.H3("Simulation de Diagnostic Hybride (Diabète)", style={"margin": "0 0 20px 0"}),
            html.P("Entrez l'ID du patient (0 à 153) pris dans le dataset de validation pour forcer la machine à raisonner :"),
            dcc.Input(id="patient-id", type="number", min=0, max=len(health_y_test)-1, value=0, 
                      style={"padding": "15px", "borderRadius": "8px", "border": f"1px solid {CSS_COLORS['border']}", "backgroundColor": CSS_COLORS["bg"], "color": "#FFF", "width": "100%", "fontSize": "1.1rem"}),
            html.Button("Exécuter Matrice Fâ", id="btn-run-health", n_clicks=0,
                        style={"marginTop": "20px", "width": "100%", "padding": "15px", "backgroundColor": CSS_COLORS["success"], "color": "#FFF", "border": "none", "borderRadius": "8px", "fontSize": "1.1rem", "fontWeight": "bold", "cursor": "pointer"}),
            html.Div(id="health-result", style={"marginTop": "30px"})
        ])
    
    elif tab_name == "tab-agri":
        if agri_model is None:
            return html.H3("Fichier de données data_poc/agri_dataset.csv manquant", style={"color": CSS_COLORS["danger"]})
        return html.Div(style={"backgroundColor": CSS_COLORS["card"], "borderRadius": "12px", "padding": "30px", "boxShadow": "0 10px 30px rgba(0,0,0,0.5)"}, children=[
            html.H3(f"Aptitude Édaphique (Culture: {agri_target.upper()})", style={"margin": "0 0 20px 0"}),
            html.P("Entrez l'ID du sol (0 à 439) capté par l'IoT :"),
            dcc.Input(id="agri-id", type="number", min=0, max=len(agri_y_test)-1, value=5, 
                      style={"padding": "15px", "borderRadius": "8px", "border": f"1px solid {CSS_COLORS['border']}", "backgroundColor": CSS_COLORS["bg"], "color": "#FFF", "width": "100%", "fontSize": "1.1rem"}),
            html.Button("Analyser le Sol", id="btn-run-agri", n_clicks=0,
                        style={"marginTop": "20px", "width": "100%", "padding": "15px", "backgroundColor": CSS_COLORS["neon"], "color": "#111", "border": "none", "borderRadius": "8px", "fontSize": "1.1rem", "fontWeight": "bold", "cursor": "pointer"}),
            html.Div(id="agri-result", style={"marginTop": "30px"})
        ])

@app.callback(
    Output("health-result", "children"),
    Input("btn-run-health", "n_clicks"),
    State("patient-id", "value"),
    prevent_initial_call=True
)
def execute_health(n_clicks, p_id):
    if p_id is None or p_id >= len(health_y_test) or p_id < 0:
        return html.Div("ID invalide.")
    
    X_raw = health_X_test[p_id:p_id+1]
    X_scaled = health_scaler.transform(X_raw)
    y_true = health_y_test[p_id]
    
    pred_val, interpretations = health_model.predict(X_scaled)
    prob = float(pred_val[0][0]) * 100
    diag = interpretations[0]
    
    color = CSS_COLORS["danger"] if prob > 50 else CSS_COLORS["success"]
    verdict = "POSITIF (Risque détecté)" if prob > 50 else "NÉGATIF (Sain)"
    
    return html.Div(style={"padding": "20px", "borderLeft": f"4px solid {color}", "backgroundColor": CSS_COLORS["border"], "borderRadius": "4px"}, children=[
        html.H4(f"Probabilité Générée : {prob:.1f}%", style={"color": color, "margin": "0 0 10px 0"}),
        html.P(f"Vérité terrain : {'Malade' if y_true == 1 else 'Sain'}"),
        html.Hr(style={"borderColor": CSS_COLORS["bg"]}),
        html.H5("Explicabilité Algorithmique (Logique Mathématique Fâ) :", style={"color": "#FFF"}),
        html.Code(f"État Tensionnel : {diag['state_id']}", style={"display": "block", "color": CSS_COLORS["accent"]}),
        html.Code(f"Empreinte Binaire : {diag['binary_matrix']}", style={"display": "block", "color": CSS_COLORS["accent"]}),
        html.Code(f"Node Ontologique  : {diag['ontological_state']}", style={"display": "block", "color": CSS_COLORS["accent"]}),
        html.Code(f"Polarité Système  : {diag['system_polarity']}", style={"display": "block", "color": CSS_COLORS["accent"]}),
        html.H4(f"Verdict : {verdict}", style={"marginTop": "20px"})
    ])

@app.callback(
    Output("agri-result", "children"),
    Input("btn-run-agri", "n_clicks"),
    State("agri-id", "value"),
    prevent_initial_call=True
)
def execute_agri(n_clicks, a_id):
    if a_id is None or a_id >= len(agri_y_test) or a_id < 0:
        return html.Div("ID invalide.")
    
    X_raw = agri_X_test[a_id:a_id+1]
    X_scaled = agri_scaler.transform(X_raw)
    y_true = agri_y_test[a_id]
    
    pred_val, interpretations = agri_model.predict(X_scaled)
    prob = float(pred_val[0][0]) * 100
    diag = interpretations[0]
    
    color = CSS_COLORS["success"] if prob > 50 else CSS_COLORS["danger"]
    verdict = "FAVORABLE (Engrais & Eau alignés)" if prob > 50 else "HOSTILE (Risque critique)"
    
    return html.Div(style={"padding": "20px", "borderLeft": f"4px solid {color}", "backgroundColor": CSS_COLORS["border"], "borderRadius": "4px"}, children=[
        html.H4(f"Viabilité de culture : {prob:.1f}%", style={"color": color, "margin": "0 0 10px 0"}),
        html.P(f"Vérité terrain : {'Riz valide' if y_true == 1 else 'Autre culture'}"),
        html.Hr(style={"borderColor": CSS_COLORS["bg"]}),
        html.H5("Explicabilité Algorithmique (Logique Fâ de l'Élément Eau/Terre) :", style={"color": "#FFF"}),
        html.Code(f"Node Combinatoire : {diag['state_id']}", style={"display": "block", "color": CSS_COLORS["neon"]}),
        html.Code(f"Empreinte Binaire : {diag['binary_matrix']}", style={"display": "block", "color": CSS_COLORS["neon"]}),
        html.Code(f"État Ontologique  : {diag['ontological_state']}", style={"display": "block", "color": CSS_COLORS["neon"]}),
        html.Code(f"Stabilité Causal  : {diag['system_polarity']}", style={"display": "block", "color": CSS_COLORS["accent"]}),
        html.H4(f"Verdict : {verdict}", style={"marginTop": "20px"})
    ])

if __name__ == '__main__':
    print("--------------------------------------------------")
    print("  DÉMONSTRATEUR MLOps PRÊT. OUVREZ VOTRE NAVIGATEUR")
    print("  SUR: http://127.0.0.1:8050/")
    print("--------------------------------------------------")
    app.run(debug=False, port=8050)
