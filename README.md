# MATRICE-256 (FaNet v4.0 / bokono)

**Framework d'Intelligence Artificielle Neuro-Symbolique inspiré de la combinatoire binaire du Fâ (Bénin)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FOCAC 2026](https://img.shields.io/badge/FOCAC%20%2F%20China--BRICS%202026-Cas%20Repr%C3%A9sentatif-1e40af.svg)](#reconnaissance-internationale)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-red.svg)](LICENSE)

> **English Summary:** Matrice-256 (FaNet / `bokono`) is a lightweight PyTorch neuro-symbolic framework exploring a structured inductive bias inspired by the 8-bit binary combinatorics of the Beninese Fa system (16 fundamental Kpoli keys, 256 composite states). Selected in October 2026 as a representative case in the *AI Case Innovation Competition for African Youth* (FOCAC / China-BRICS AI Center, 114 selected out of 738 submissions across 47 African countries, Technological Progress track).

---

## Reconnaissance internationale

En **octobre 2026**, **Matrice-256 (FaNet)** a été sélectionné comme **cas représentatif** au *Concours créatif sur les cas d'application de l'intelligence artificielle à l'intention des jeunes des pays africains* (*AI Case Innovation Competition for African Youth*), organisé par le **Secrétariat du Comité de suivi chinois du Forum sur la Coopération Sino-Africaine (FOCAC)** et mis en oeuvre par le **China-BRICS Artificial Intelligence Development and Cooperation Center** :

- **Sélection globale** : 114 cas retenus sur 738 dossiers soumis dans 47 pays africains (taux de sélection d'environ 15,4 %).
- **Catégorie** : **Progrès technologique** (13 cas retenus sur 58 soumissions), publié aux pages 165 à 167 du recueil officiel (*Cas d'innovation en IA de la jeunesse africaine*).
- **Représentation nationale** : 1 des 7 projets béninois retenus dans le recueil.

---

## Vue d'ensemble du projet

**Matrice-256** explore l'intégration d'un système formel endogène africain, la combinatoire binaire du **Fâ** (16 signes fondamentaux sur 4 bits donnant 256 combinaisons sur 8 bits), comme **biais inductif structurel** au sein d'un réseau de neurones PyTorch (`TorchFaNet`) :

1. **Architecture combinatoire (`TorchFaEmbedding`)** : deux projections de requête (Kpoli gauche et Kpoli droit) calculent une attention croisée sur les 16 clés de base et forment une distribution sur les 256 états par produit de Kronecker (`16 x 16`).
2. **Perte à contrainte ontologique (`FaImbalanceLoss`)** : combine une perte statistique (`BCEWithLogitsLoss` ou `MSELoss`) avec un terme de régularisation qui pénalise l'écart entre la prédiction du modèle et la polarité composite de l'espace latent.
3. **Activations paramétrées et mode temporel** : supporte le mode tabulaire (`TorchFaHidden` avec `TorchLearnableFaSigma` ou `FaReLU`) et le mode séquentiel pour capteurs IoT (`nn.LSTM`, optionnellement bidirectionnel et multicouche).
4. **Exécution locale et frugale** : modèle léger (~1 100 paramètres en mode tabulaire sur Pima Diabetes, ~6 000 en mode LSTM IoT), fonctionnant entièrement hors-ligne (air-gapped) avec export ONNX intégré.

Documentation scientifique :
- [Livre blanc technique (FR)](docs/whitepaper_matrice256.md)
- [Article académique (EN)](docs/academic_paper_en.md)
- [Article académique (FR)](docs/academic_paper_fr.md)

---

## Note sur la base de connaissances publique et la reproductibilité

Afin de protéger les textes traditionnels et les annotations privées de la base de connaissances complète (`fa_knowledge_private.json`, exclue du dépôt via `.gitignore`), ce dépôt public distribue un fichier épuré (`bokono/core/fa_knowledge_public.json`) où toutes les polarités sont neutralisées à `"neutre"` (`0.5`).

Conséquence importante pour l'évaluation scientifique : dans la configuration publique actuelle (`v4.0.0`), la pénalité `FaImbalanceLoss` se comporte comme une régularisation de confiance vers `0.5`. L'intégration d'une table de polarités numériques publiques reproductibles (dérivée de propriétés combinatoires binaires sans exposer les textes privés) fait partie des priorités de la prochaine itération (`v4.1 / v5.0`).

---

## Résultats expérimentaux (Bancs d'essai v4.0)

Les scripts de validation du dépôt couvrent 8 axes expérimentaux. Le tableau ci-dessous résume les mesures obtenues en phase v4.0 ainsi que leurs limites connues :

| Test | Script | Protocole et données | Résultat principal (v4.0) | Notes et limites identifiées |
|---|---|---|---|---|
| **T1 : Robustesse au bruit** | `tests/test_noise_robustness.py` | Bruit gaussien (0 % à 50 %) sur test, `agri_dataset.csv` (cible `rice`, 4,5 % positifs) | Accuracy 96,14 % à 50 % de bruit (RF : 95,68 %) | Split unique et classe très déséquilibrée : ré-évaluation multi-seeds sur F1 / PR-AUC prévue |
| **T2 : Cross-Validation K=10** | `tests/test_cv_final.py` | Stratified 10-Fold + SMOTE sur train, `agri_dataset.csv` | Accuracy 97,50 % (+/- 1,21 %), F1 79,26 %, Rappel 100 % | `pos_weight` appliqué en plus de SMOTE : comparaison symétrique avec MLP en cours |
| **T3 : Généralisation OOD** | `tests/test_ood.py` | Entraînement sur `rice`, test sur `jute` vs cultures sèches | Rappel OOD 100 % (RF : 92 %, GBM : 72 %) | Fuite sur les négatifs et baseline MLP à intégrer dans le protocole v5.0 |
| **T4 : Calibration probabiliste** | `tests/test_calibration.py` | Split 60/40 sur `agri_dataset.csv` (`rice, jute, maize`) | Brier Score 0,0094 ; ECE non pondérée 14,71 % | Formule ECE à pondérer par taille de bin et à étendre aux données cliniques |
| **T5 : Régression (Fintech)** | `tests/test_regression.py` | Données climatiques/prix synthétiques (FCFA) | MAE 142,6 FCFA, R2 0,32 (MLP : 70,6 FCFA, R2 0,84) | Limite liée à la saturation de la tête de sortie en régression |
| **T6 : IoT temporel (LSTM)** | `tests/test_iot_timeseries.py` | Split chronologique strict (60/15/25) + seuil calibré sur Val | F1 54,01 %, Rappel 97,4 % (RF : F1 91,57 %) | Limite documentée ; baseline LSTM standard et purge temporelle à ajouter |
| **T7 : Confidentialité (MIA)** | `tests/test_mia.py` | Attaque par seuil de confiance sur Pima Indians Diabetes (768 patients) | AUC d'attaque 0,5261 (RF : 0,7530 ; GBM : 0,6617) | Indicateur empirique de faible mémorisation (esprit Privacy-by-Design, non une certification juridique) ; compromis Privacy/Utility à compléter |
| **T8 : Empreinte énergétique** | `tests/test_energy.py` | Mesure CodeCarbon sur Pima (1 000 batchs de 308 patients) | 40,0 nWh/batch d'inférence (-67,8 % vs RF 124,1 nWh ; MLP : 5,1 nWh ; GBM : 15,8 nWh) | Coût d'entraînement plus élevé (95,4 uWh) ; benchmark d'inférence ONNX prévu |

---

## Démarrage rapide

### 1. Installation locale

```bash
git clone https://github.com/CedBag/matrice-256.git
cd matrice-256
python -m venv .venv
# Linux / macOS : source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Exemple minimal (API PyTorch)

```python
import numpy as np
from bokono.ml.torch_network import TorchFaNet

# Données d'exemple normalisées (ex. 8 variables biomédicales)
X_train = np.random.randn(200, 8).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.float32)
X_test = np.random.randn(5, 8).astype(np.float32)

# Initialisation du réseau neuro-symbolique (mode tabulaire)
model = TorchFaNet(n_features=8, hidden_dim=32, n_outputs=1, seed=42)

# Entraînement supervisé avec pénalité ontologique (lambda_fa)
model.fit(X_train, y_train, epochs=30, lr=0.01, lambda_fa=0.1, verbose=False)

# Inférence avec état symbolique associé
probabilities, interpretations = model.predict(X_test)
print(probabilities[0])
print(interpretations[0]["ontological_state"])  # ex: STATE_128
print(interpretations[0]["system_polarity"])    # ex: NEUTRE (avec KB publique)
```

### 3. Démonstrations incluses

```bash
python demo_health_fanet.py
python demo_agri_fanet.py
```

---

## Structure du dépôt

- `bokono/core/` : encodeur binaire Fâ (`fa_encoder.py`), matrice 16x16 (`fa_matrix.py`) et base publique (`fa_knowledge_public.json`).
- `bokono/ml/` : couches PyTorch (`TorchFaEmbedding`, `TorchFaHidden`, `TorchFaDecoder`), `FaImbalanceLoss`, `TorchFaNet` et implémentations NumPy historiques.
- `data_poc/` : jeux de données PoC (`sante_dataset.csv`, `agri_dataset.csv`).
- `docs/` : livre blanc, brouillons d'articles académiques et rapports d'audit.
- `scripts/` : utilitaires de parsing, sécurisation de la KB et génération de figures.
- `tests/` : suite de tests unitaires et bancs d'essai expérimentaux (T1 à T8).
- `demo_health_fanet.py` / `demo_agri_fanet.py` : démonstrations santé et agriculture.
- `LICENSE` / `pyproject.toml` : licence de recherche académique et configuration du package `bokono`.

---

## Feuille de route scientifique (v4.1 / v5.0)

Les travaux en cours visent à consolider la rigueur scientifique et la reproductibilité du framework avant publication académique (preprint arXiv / Zenodo, Deep Learning Indaba, workshops neuro-symboliques) :

1. **Correctifs architecturaux** : alignement strict entraînement/inférence sur le décodeur, correction du mapping de la matrice 16x16 et de la gestion GPU dans `FaImbalanceLoss`, continuité des fonctions d'activation `FaReLU` et `FaSigma`.
2. **Étude d'ablation complète multi-seeds** : comparaison à capacité égale contre Régression Logistique, Random Forest régularisé, Gradient Boosting, MLP PyTorch, `FaNet (lambda_fa=0)`, contrainte constante (`0.5`) et contrôle aléatoire (polarités permutées) avec tests statistiques appariés.
3. **Standardisation et Edge AI** : publication stabilisée du package `bokono` sur PyPI, benchmarks d'inférence ONNX Runtime sur matériel contraint (CPU / Raspberry Pi) et validation pilote sur données terrain réelles.

---

## Citation

Si vous évaluez ou citez ce travail dans un contexte académique :

```bibtex
@software{bagbonon2026matrice256,
  author       = {Bagbonon, Cedoque},
  title        = {Matrice-256 (FaNet / bokono): A Neuro-Symbolic AI Framework Inspired by Beninese Fa Binary Combinatorics},
  year         = {2026},
  version      = {4.0.0},
  url          = {https://github.com/CedBag/matrice-256},
  note         = {Selected as a representative case in the AI Case Innovation Competition for African Youth (FOCAC / China-BRICS AI Center, October 2026)}
}
```

---

## Licence et contact

Ce dépôt est distribué sous **Licence de Recherche Académique et d'Évaluation** (voir [LICENSE](LICENSE)) : la lecture, l'audit, la vérification scientifique par les pairs et l'expérimentation locale non commerciale sont autorisés. Toute utilisation commerciale, redistribution ou exploitation industrielle requiert un accord écrit préalable.

- **Auteur** : Cedoque BAGBONON (Projet Matrice-256, Bénin)
- **Contact** : `projet.matrice256@protonmail.com`
