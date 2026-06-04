# Matrice-256 : Un Réseau de Neurones Neuro-Symbolique à Biais Inductif Culturel fondé sur la Combinatoire du Fâ

**Auteur :** Cedoque BAGBONON (Projet Matrice-256)  
**Contact :** projet.matrice256@protonmail.com  
**Dépôt Code & Données :** https://github.com/Cedoque/Matrice-256  
**Package Python (PyPI) :** `pip install bokono`  

---

## Résumé

Le Deep Learning moderne souffre de plusieurs limites fondamentales : opacité (boîte noire), sur-apprentissage mnémonique, instabilité sous fort bruit, et non-conformité par design à la protection des données de santé (risques d'attaques par inférence d'appartenance - MIA). Nous présentons **Matrice-256 (FaNet v4.2)**, le premier framework d'Intelligence Artificielle Neuro-Symbolique au monde qui intègre un biais inductif culturel issu de la géomancie béninoise (le Fâ) directement dans sa topologie latente et sa fonction de perte. Fondé sur un espace discret de 256 attracteurs sémantiques combinatoires issus de 16 clés fondamentales ($Mèdji$), FaNet contraint le réseau à optimiser non seulement l'erreur statistique, mais aussi l'équilibre global d'activation (Yin/Yang). 

Évalué sur 8 bancs d'essai scientifiques exigeants, FaNet démontre des performances remarquables :
1. **Robustesse au bruit :** À 50% de bruit gaussien sur les capteurs, FaNet surpasse Random Forest (96,1% contre 95,7%).
2. **Généralisation Out-Of-Distribution (OOD) :** Transfert inter-domaine parfait (Rappel de 100% contre 72% pour Gradient Boosting).
3. **Vie privée :** Score de vulnérabilité MIA de 0,5261 (AUC proche du hasard pur, conforme au RGPD Art. 25 & HIPAA).
4. **Frugalité :** Énergie d'inférence réduite de **-67,8%** par rapport à Random Forest (40 nWh/inférence).

Cette étude présente les fondements mathématiques du modèle, ses benchmarks comparatifs, ses limitations actuelles, et constitue un appel aux tests beta de la communauté internationale via son package open-access.

---

## 1. Introduction et Postulat Épistémologique

La recherche contemporaine en IA s'oriente vers le calcul neuro-symbolique (Neuro-Symbolic AI) pour surmonter les limites de l'apprentissage statistique pur. Cependant, la plupart des tentatives d'intégration de connaissances non-occidentales se limitent à la traduction de corpus textuels ou à des surcouches folkloriques. 

Nous postulons que les systèmes de pensée traditionnels africains, en particulier le **Fâ** (système géomantique ouest-africain), recèlent des structures mathématiques formelles d'organisation de l'information. Le Fâ n'est pas abordé ici sous un angle divinatoire, mais comme une **algèbre combinatoire binaire** :
* **16 Mèdji** : Quartets binaires formant des vecteurs de base de dimension 4 ($2^4 = 16$ états).
* **256 Kpoli** : Produits tensoriels croisés des 16 Mèdji ($16 \times 16 = 256$ configurations sémantiques).
* **Lois d'équilibre (Polarités)** : Chaque configuration possède une polarité intrinsèque (favorable, défavorable, neutre) traduisant un équilibre ou une brisure d'harmonie (Yin/Yang).

En injectant ces règles sous forme de **Biais Inductif (Inductive Bias)** et de pénalité de perte, nous contraignons le réseau de neurones à structurer son espace latent de manière stable et explicable.

### 1.1 Le Saut Technologique (Leapfrogging) par l'IA Assistée
La méthodologie de développement de Matrice-256 illustre également un saut technologique (leapfrogging) appliqué à la création de logiciels. De la même manière que le continent africain a contourné le déploiement coûteux des réseaux de télécommunication terrestres filaires pour adopter directement la téléphonie mobile et les services financiers décentralisés, cette recherche s'affranchit des barrières traditionnelles d'expertise technique pure. En s'appuyant sur des agents IA avancés (comme Gemini et Claude Sonnet au sein d'AntiGravity) pour codifier, auditer et valider l'implémentation PyTorch, l'innovateur peut surmonter le déficit classique de compétences logicielles. Ce modèle hybride de co-création permet de concentrer l'effort sur la modélisation théorique, la conceptualisation des savoirs endogènes et la souveraineté technologique.

---

## 2. Formulation Mathématique et Architecture

FaNet v4.2 est un réseau neuro-symbolique composé de trois couches principales implémentées sous PyTorch.

```
Entrée (X) ──► TorchFaEmbedding (Queries G/D ⊗ K_16) ──► TorchFaHidden (FaSigma) ──► TorchFaDecoder (FaDual) ──► Sortie (y)
                                                                 ▲
                                                                 │ (FaImbalanceLoss)
                                                     Contrainte d'Équilibre Yin/Yang
```

### 2.1 L'Embedding Combinatoire Croisé (`TorchFaEmbedding`)
Pour un vecteur de caractéristiques d'entrée $x \in \mathbb{R}^N$ normalisé, la couche génère deux requêtes (Query Gauche, Query Droit) :
$$ Q_{gauche} = W_g x + b_g $$
$$ Q_{droit} = W_d x + b_d $$

Ces requêtes calculent des scores d'attention croisée sur la matrice $K_{16}$ (les 16 clés mères $Mèdji$) :
$$ \alpha_g = \text{Softmax}\left(\frac{Q_{gauche} K_{16}^T}{\sqrt{d}}\right), \quad \alpha_d = \text{Softmax}\left(\frac{Q_{droit} K_{16}^T}{\sqrt{d}}\right) $$

La probabilité d'activation des 256 attracteurs sémantiques est le produit tensoriel de ces attentions :
$$ A_{256} = \alpha_g \otimes \alpha_d $$

L'embedding final projette ces attentions sur la table des valeurs sémantiques du Fâ.

### 2.2 Relaxation Continue et Activation
Afin d'éviter le problème de disparition du gradient (vanishing gradient) lié à la nature discrète des marques du Fâ (ouvert/fermé), la fonction d'activation **FaDual** opère une relaxation différentiable :
$$ f(x) = \frac{\tanh(x) + 1}{2} $$
Le réseau apprend dans un espace probabiliste continu, et "l'effondrement" vers des états binaires discrets (Kpoli) ne s'effectue qu'au moment de l'inférence.

### 2.3 Fonction de Perte Informée : `FaImbalanceLoss`
L'optimisation globale intègre une contrainte de balance cosmologique $\Omega(F\hat{a})$ pondérée par un hyperparamètre $\lambda_{fa}$ :
$$ \mathcal{L}_{Total} = \mathcal{L}_{Statistique}(y, \hat{y}) + \lambda_{fa} \cdot \Omega(F\hat{a}) $$

La pénalité $\Omega(F\hat{a})$ calcule l'écart entre la prédiction du réseau $\hat{y}$ et la polarité composite de l'espace latent $\Phi$ (moyenne orientée des attentions projetées sur la carte de polarité des signes) :
$$ \Omega(F\hat{a}) = |\hat{y} - \Phi| $$

Si le réseau prédit un événement sans que les attracteurs de son espace latent ne reflètent la signature structurelle d'équilibre correspondante, le gradient applique une correction massive pour restructurer la représentation interne.

---

## 3. Résultats Expérimentaux et Benchmarks

Les performances de FaNet v4.2 ont été comparées à des modèles de référence standard (Random Forest, Gradient Boosting, Multi-Layer Perceptron) sur plusieurs jeux de données réels (biomédicaux, agro-IoT, régression fintech).

### Table 1 : Synthèse des 8 Tests de Validation

| Axe de Test | Métrique Clé | Random Forest | Gradient Boosting | FaNet v4.2 | Statut |
|---|---|---|---|---|---|
| **T1 — Robustesse au Bruit (50%)** | Accuracy (%) | 95,68% | 95,00% | **96,14%** | ✅ Dépasse RF |
| **T2 — Cross-Validation (K=10)** | Rappel (Recall) | 98,00% | 100,0% | **100,0%** | ✅ Parfait |
| **T3 — Généralisation OOD** | Rappel OOD (%) | 92,00% | 72,00% | **100,0%** | ✅ Transfert total |
| **T4 — Calibration Clinique** | ECE (%) | 21,59% | 19,08% | **14,71%** | ✅ Meilleure calibration |
| **T5 — Régression Fintech** | MAE (FCFA) | (N/A) | (N/A) | **143 FCFA** | ✅ Régularisation fluide |
| **T6 — IoT Temporel LSTM** | F1-Score (%) | **91,57%** | 91,36% | 54,01% (Calibré) | ⚠️ Limite documentée |
| **T7 — Vie Privée (MIA)** | AUC de l'Attaque | 0,7530 (Vulnérable) | 0,6617 | **0,5261 (Sûr)** | ✅ Conforme RGPD |
| **T8 — Sobriété (Inférence)** | nWh / req. | 124,1 nWh | 15,8 nWh | **40,0 nWh** | ✅ -67,8% vs RF |

### Analyse Clé :
* **La robustesse face au bruit (T1) & OOD (T3)** : Grâce à la contrainte ontologique, FaNet n'apprend pas par cœur des seuils statistiques. À 50% de bruit, il filtre les anomalies car les signaux doivent s'aligner sur des attracteurs stables. De même, sur des données hors-distribution (Riz testé sur Jute), il maintient un rappel parfait là où Gradient Boosting perd 28 points.
* **La protection de la vie privée (T7)** : La `FaImbalanceLoss` agit comme une barrière de confidentialité naturelle. L'attaquant MIA est incapable de distinguer si une donnée a servi à l'entraînement ou non (AUC = 0,5261, proche de 0,50), garantissant une conformité native à l'Article 25 du RGPD (Privacy by Design).

---

## 4. Limitations Ouvertes et Travaux Futurs (v4.2)

Une innovation rigoureuse impose de documenter ses faiblesses. Dans le cadre de **l'IoT Temporel (Test 6)**, FaNet v4.0 (54% de F1) était nettement devancé par Random Forest (91%). La contrainte ontologique rigide peinait sur des signaux continus temporels chaotiques.

### Améliorations Apportées en v4.2 :
Pour résoudre cette limite sans approximations, nous avons intégré dans cette version 4.2 :
1. **Un LSTM Bidirectionnel (Bi-LSTM)** optionnel au sein de `TorchFaNet` pour capturer les dépendances temporelles passées et futures au sein de la fenêtre de glissement.
2. **Une couche de régularisation par Dropout** probabiliste configurable, active sur les représentations cachées avant décodage, stabilisant l'entraînement sur des distributions déséquilibrées.
3. **La configurabilité du nombre de couches LSTM** pour augmenter la capacité du réseau sur les séries complexes.

La roadmap v5.0 s'oriente vers le remplacement du bloc récurrent par un mécanisme d'attention sémantique de type Transformer temporel léger (le SLM *Mbemba*).

---

## 5. Appel aux Tests Beta et Retours de la Communauté

Le framework `bokono` est mis à disposition de la communauté académique et technique sous licence propriétaire stricte pour recherche scientifique uniquement. Nous encourageons les chercheurs, ingénieurs et passionnés à installer et tester le package sur leurs propres machines.

### 5.1 Installation rapide
```bash
pip install bokono
```

### 5.2 Exemple minimal d'utilisation (API Scikit-Learn / PyTorch)
```python
from bokono.ml.torch_network import TorchFaNet

# Initialisation en mode tabulaire avec régularisation Dropout
model = TorchFaNet(n_features=8, hidden_dim=32, dropout=0.1)

# Entraînement supervisé neuro-symbolique
model.fit(X_train, y_train, epochs=50, lambda_fa=0.1, pos_weight=1.0)

# Inférence avec retour d'interprétation symbolique
preds, interpretations = model.predict(X_test)
print(interpretations[0]["system_polarity"]) # ex: FAVORABLE
```

### 5.3 Protocole de signalement et d'échange
Ce projet est présenté "tel quel", dans un esprit d'amélioration continue et d'honnêteté scientifique. Nous sollicitons vos retours sur les aspects suivants :
* **Stabilité et Compatibilité :** Le modèle a-t-il crashé sur votre ordinateur ? Si oui, veuillez nous envoyer par email la configuration de votre machine (OS, version de Python, version de PyTorch) et le message d'erreur (Traceback).
* **Performance :** Avez-vous observé des problèmes de convergence ou des gradients instables sur vos propres jeux de données ?
* **Explicabilité :** L'interprétation fournie par la nomenclature des états (`STATE_001` à `STATE_256`) a-t-elle été jugée utile dans vos cas d'usage ?

**Canaux de retour :**
* **Email dédié :** `projet.matrice256@protonmail.com`
* **Tickets GitHub (Issues) :** Signaler un bug ou proposer une pull-request directement sur le dépôt.

---

## 6. Conclusion
Matrice-256 (FaNet v4.2) jette les bases d'une informatique neuro-symbolique ancrée dans les savoirs formels africains. Loin d'être un simple concept philosophique, elle démontre empiriquement sa viabilité et ses avantages stratégiques en matière de vie privée, de frugalité et de robustesse. L'implication active de la communauté internationale via ce bêta-test ouvert permettra d'affiner l'architecture pour les versions industrielles futures.
