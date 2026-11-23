# Matrice-256 (FaNet v4.2) : Une architecture neuro-symbolique fondée sur l'hypercube binaire 4 bits du Fâ béninois

**Auteur :** Cedoque BAGBONON (Projet Matrice-256, Bénin)
**Contact :** `projet.matrice256@protonmail.com`
**Code et benchmarks reproductibles :** `https://github.com/CedBag/matrice-256`
**Reconnaissance :** Cas représentatif sélectionné au *Concours créatif sur les cas d'application de l'IA à l'intention des jeunes des pays africains* (FOCAC / China-BRICS AI Center, octobre 2026 ; catégorie Progrès technologique, pages 165 à 167).

---

## Résumé

Les modèles d'apprentissage automatique standards sur données tabulaires et cliniques restent confrontés à des compromis difficiles entre pouvoir prédictif, calibration probabiliste, vulnérabilité aux attaques par inférence d'appartenance (MIA), robustesse au bruit de capteurs et frugalité en périphérie (Edge AI). Nous présentons **Matrice-256 (FaNet v4.2 / `bokono`)**, une architecture neuro-symbolique PyTorch compacte (~2 361 paramètres entraînables, export ONNX de 30,2 Ko) fondée sur l'algèbre binaire formelle sur 8 bits du **Fâ** béninois. FaNet projette les entrées par requêtes cross-features sur l'hypercube binaire exact 4 bits `K_16 in {-1, +1}^4` (16 clés Kpoli), forme une distribution sur les 256 Du par produit de Kronecker (`16 x 16`), traite les représentations via un bloc résiduel hybride (`FaReLU` continue + `FaSigma` lisse), et optimise une pénalité d'équilibre ontologique factorisée (`FaImbalanceLoss`) en complexité `O(16)`.

Évalué sur **GPU Kaggle Tesla T4** sur 8 bancs d'essai sans fuite de données (`5-Fold x 5 seeds = 25 runs` appariés avec tests de Wilcoxon face à Régression Logistique, Random Forest régularisé, HistGradientBoosting, MLP PyTorch résiduel iso-paramètres et 4 contrôles d'ablation), FaNet v4.2 démontre :
1. **Performance clinique :** Surpasse significativement HistGradientBoosting et le MLP iso-paramètres en ROC-AUC et PR-AUC sur Breast Cancer Wisconsin (`ROC-AUC 0,9949, p = 0,0037`) et Pima Indians Diabetes (`ROC-AUC 0,8318, p = 0,0045` ; variance F1 2,5x plus faible que le MLP), avec la meilleure ECE clinique pondérée sur Breast Cancer (`1,27 %`).
2. **Compromis Confidentialité / Utilité :** Présente une faible vulnérabilité aux attaques MIA par perte (`AUC 0,5133 à 0,5229`, `TPR@1%FPR = 1,28 %`) tout en conservant une utilité clinique élevée, devant `MLP` (`0,5337`), `RF` régularisé (`0,5895`) et `RF` non régularisé (`0,7674`).
3. **Transfert OOD sur hypercube et IoT temporel :** Conserve `100 %` F1 OOD sur Jute et `60,5 %` F1 (`ROC-AUC 1,000`) sur Maïs là où les arbres tombent à `0 %` et où la permutation aléatoire des polarités sur l'hypercube 4 bits fait chuter le F1 de `-38,8` points (`21,7 %`). En séries temporelles IoT (purge chronologique de 18h), `FaNet-BiLSTM` atteint `84,88 %` F1 (`93,2 %` Rappel), dépassant un BiLSTM PyTorch standard (`69,56 %`) de `+15,3` points.
4. **Frugalité Edge :** S'exécute 7,3x plus vite avec `-85,9 %` d'énergie d'inférence par patient (`124,4 nWh`) que Random Forest et `-48,7 %` que HistGradientBoosting.

---

## 1. Introduction et formalisation algébrique

Dans cette étude, le **Fâ** béninois n'est pas abordé sous un angle divinatoire, mais comme une **algèbre combinatoire binaire formelle sur `{0, 1}^4 x {0, 1}^4`** :
- **16 Kpoli** : mots binaires de 4 bits représentant les 16 sommets de l'hypercube `K_16 in {-1, +1}^4` (`0` = marque ouverte `+1`, `1` = marque fermée `-1`).
- **256 Du** : paires ordonnées `(i, j)` encodées sur 8 bits (`1 octet`), dont les 16 éléments diagonaux (`i = j`, **Mèdji**) maximisent la symétrie structurelle (`1 - d_H(i, j)/4 = 1,0`).
- **Polarité canonique d'ouverture** : chaque Kpoli porte une ouverture binaire exacte `p_k = (nombre de bits ouverts) / 4` dans `{0,0 ; 0,25 ; 0,50 ; 0,75 ; 1,0}`, engendrant une polarité symétrique `p_{i,j} = (p_i + p_j) / 2` sur les 256 Du (93 favorables, 70 neutres, 93 défavorables).

Cette formulation est 100 % mathématique, ouverte et reproductible sans dépendre de corpus privés.

---

## 2. Architecture et formulation mathématique (FaNet v4.2)

### 2.1 Embedding combinatoire sur hypercube 4 bits (`TorchFaEmbedding`)

Les entrées `x in R^n` sont normalisées de façon lisse par `x_norm = sigmoid((x - mu) / sigma)` (statistiques ajustées uniquement sur `X_train`). Pour chaque variable `f`, les requêtes Gauche et Droite combinent une projection locale (`PerFeatureLinear`) et un contexte global cross-features (`x_c = 2 * x_norm - 1`) :

$$Q_g(f) = w_{g,f} x_{\text{norm}, f} + b_{g,f} + W_g^{\text{ctx}} x_c + b_g^{\text{ctx}}, \qquad Q_d(f) = w_{d,f} x_{\text{norm}, f} + b_{d,f} + W_d^{\text{ctx}} x_c + b_d^{\text{ctx}}$$

L'attention croisée sur l'hypercube `K_16 in {-1, +1}^{16 x 4}` et son produit de Kronecker produisent une distribution sur les 256 Du :

$$\alpha_g = \text{Softmax}\left(\frac{Q_g K_{16}^T}{2}\right), \qquad \alpha_d = \text{Softmax}\left(\frac{Q_d K_{16}^T}{2}\right), \qquad A_{256} = \alpha_g \otimes \alpha_d$$

### 2.2 Bloc résiduel hybride et cohérence train / inférence

- **`FaReLU` continue par morceaux** : raccordée sans saut aux seuils `1/16`, `8/16`, `15/16` (pentes `0,00 ; 0,25 ; 0,50 ; 1,00 ; 0,10`).
- **`TorchLearnableFaSigma` lisse** : interpolation par noyau gaussien sur les 16 centres Kpoli (`C-infini`, gains entraînables dans `(0,5 ; 1,5)`).
- **Bloc tabulaire résiduel** : `h_1 = FaReLU(W_1 e + b_1)`, puis `h = h_1 + FaSigma(W_2 h_1 + b_2)` avec initialisation plein rang (85 % Xavier + 15 % prior Matrice-256 à pas premier `17`).
- **Décodeur aligné** : `sigmoid(z)` + `BCEWithLogitsLoss` en classification standard ; sortie linéaire non saturée + `MSELoss` en régression.

### 2.3 Perte factorisée `FaImbalanceLoss`

Grâce à la relation `p_{i,j} = (p_i + p_j) / 2`, l'espérance de polarité sur les 256 Du du produit de Kronecker se factorise exactement en `O(16)` :

$$\Phi = \sum_{i,j=0}^{15} \alpha_g(i)\alpha_d(j) p_{i,j} = \frac{1}{2}\left(\sum_{i=0}^{15}\alpha_g(i)p_i + \sum_{j=0}^{15}\alpha_d(j)p_j\right)$$

et la fonction de perte minimise `L_total = L_stat(y, y_hat) + lambda_fa * mean(|p_hat - Phi|)`.

---

## 3. Synthèse des résultats expérimentaux (Kaggle Tesla T4)

Exécution complète via `tests/run_benchmark_v5.py` sur **Tesla T4** (`6 267 s`, `5-Fold x 5 seeds = 25 runs` appariés, 79/79 tests unitaires `pytest`) :

| Domaine / Expérience | Métrique principale | Baselines de référence | FaNet v4.2 (`F1_FaNet_Full`) | Constat clé |
|---|---|---|---|---|
| **Exp 1 : Breast Cancer (25 runs)** | F1 / ROC-AUC / PR-AUC | `RF 0,946 / 0,991 / 0,989` ; `HistGB 0,951 / 0,993 / 0,990` ; `MLP 0,964 / 0,991 / 0,990` | **0,961 / 0,995 / 0,994** | Bat significativement `RF`, `HistGB` (5/5 métriques) et `MLP` en ROC-AUC (`p=0,0037`) et PR-AUC (`p=0,0046`) |
| **Exp 1 : Pima Diabetes (25 runs)** | F1 / ROC-AUC / Brier | `HistGB 0,620 / 0,815 / 0,169` ; `MLP 0,605 / 0,821 / 0,164` | **0,631 / 0,832 / 0,159** | Bat significativement `HistGB` (`p=0,0001`) et `MLP` (`p=0,0045`), variance F1 2,5x plus faible |
| **Exp 1 : Agri High-Water (25 runs)** | F1 / ROC-AUC / ECE | `RF 1,000 / 1,000 / 0,83 %` ; `MLP 0,990 / 1,000 / 0,28 %` | **0,988 / 1,000 / 0,38 %** | +12,4 pts F1 vs v4.1 ; ECE meilleure que `RF` (`p<0,0001`) |
| **Exp 2 : Bruit 50 % (Agri)** | F1 / ROC-AUC | `RF 0,540 / 0,907` ; `HistGB 0,579 / 0,878` ; `MLP 0,730 / 0,926` | **0,694 / 0,945** (`NoLoss`: **0,706 / 0,955**) | Meilleur ROC-AUC sous fort bruit ; largement devant les arbres |
| **Exp 3 : OOD Maïs (sans fuite)** | Rappel / F1 / ROC-AUC | `RF 0 %` ; `HistGB 0 %` ; `LogReg 28,4 % / 44,2 %` ; `A3_Permuted 13,6 % / 21,7 %` | **47,0 % / 60,5 % / 1,000** | L'alignement de l'hypercube bat la permutation aléatoire de **+38,8 points F1** |
| **Exp 4 : Calibration clinique** | ECE pondérée (OOF) | `Breast Cancer` : `LogReg 1,50 %`, `MLP 2,91 %`, `RF 3,59 %` | **1,27 % (1er/6)** ; Pima : `3,96 %` (`FaDual`: **1,73 %**) | Meilleure fiabilité probabiliste sur Breast Cancer |
| **Exp 5 : Régression** | MAE / R2 | `Prix` : `RF 43,41 / 0,947` ; `Diabetes` : `RF 46,94 / 0,430` | **42,90 FCFA / 0,948** ; **45,32 / 0,456** | Dépasse `RF` sur les deux tâches de régression |
| **Exp 6 : IoT temporel (purge 18h)** | F1 / Rappel / PR-AUC | `M0_BiLSTM_Iso 69,6 % / 62,1 % / 0,692` ; `RF 92,5 %` | **84,9 % / 93,2 % / 0,807** | Surpasse un BiLSTM standard de **+15,3 points F1** (+4,7 pts vs `NoLoss`) |
| **Exp 7 : MIA Confidentialité/Utilité** | Loss-AUC / Test AUC / TPR@1% | `RF_Unreg 0,767 / 0,810` ; `RF_Reg 0,590 / 0,817` ; `MLP 0,534 / 0,815` | **0,523 / 0,819 / 1,28 %** (`NoLoss`: **0,513 / 0,825**) | Plus faible `TPR@1%FPR` et meilleur compromis que `MLP` et `RF` |
| **Exp 8 : Efficacité CPU / ONNX** | ms/1k / nWh/patient / F1 | `RF 47,4 ms / 880,5 nWh / 0,577` ; `MLP 0,61 ms / 12,6 nWh / 0,567` | **6,52 ms / 124,4 nWh / 0,645** (30,2 Ko ONNX) | 7,3x plus rapide et `-85,9 %` d'énergie vs `RF` avec le meilleur F1 |

---

## 4. Limites documentées et conclusion

1. **Ablation en distribution vs hors distribution :** En distribution (Exp 1), `F1_FaNet_Full` et `A3_FaNet_Permuted` obtiennent des scores équivalents (`p > 0,05`), montrant que l'avantage biomédical en distribution tient avant tout à l'architecture factorisée 16x16 et au bloc résiduel `FaReLU/FaSigma`. L'apport propre de l'alignement géométrique des polarités sur l'hypercube apparaît nettement en décalage OOD (`OOD_maize` : `60,5 %` vs `21,7 %`) et en IoT temporel (`84,9 %` vs `80,2 %`).
2. **Classes tabulaires très rares :** Sur `agri_rice` (4,5 % positifs), les forêts et arbres boostés (`RF 0,954`, `HistGB 0,948`) restent supérieurs aux réseaux compacts (`FaNet 0,834`, `MLP 0,880`).
3. **Débit CPU brut face au MLP minimal :** Un MLP dense pur reste plus rapide en latence brute (`0,61 ms` vs `6,52 ms` pour 1 000 patients), en contrepartie d'une discrimination et d'une calibration cliniques moindres.

**Conclusion :** Matrice-256 (FaNet v4.2) apporte une preuve reproductible qu'une formalisation algébrique issue du patrimoine béninois du Fâ peut structurer un framework PyTorch léger, calibré, respectueux de la vie privée et prêt pour l'Edge AI (`30,2 Ko ONNX`).
