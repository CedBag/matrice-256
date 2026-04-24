# AUDIT 2/3 — Éthique Technologique et des Données (EthiTec)
## Projet bokono / Matrice-256
## Conformité Vision Bénin 2060
## Date : 29 mars 2026

---

## 1. Périmètre de l'audit

Cet audit évalue la conformité éthique et légale de l'utilisation des données dans le cadre de la validation empirique (v1.0) du réseau de neurones `FaNet`. Il s'assure en particulier que les obligations de protection des données personnelles (confidentialité, anonymisation) exigées par les standards internationaux et la **Vision Bénin 2060** sont respectées.

---

## 2. Analyse des Datasets v1.0

Le module `bokono/ml/benchmark.py` utilise trois jeux de données réels pour la validation de l'algorithme :

### 2.1 Dataset 1 : Breast Cancer Wisconsin (Diagnostic)
- **Source** : `sklearn.datasets.load_breast_cancer` (UCI Machine Learning Repository).
- **Domaine** : Santé (Classification binaire malin/bénin).
- **Statut Éthique** : **✅ CONFORME**. Ce jeu de données public est mondialement utilisé pour la recherche médicale. Toutes les caractéristiques (rayons, textures, périmètres des tumeurs) proviennent d'images numérisées d'aspirations à l'aiguille fine (FNA). **Aucune donnée d'identification du patient (nom, identifiant hospitalier, date de naissance) n'y figure.** L'anonymisation est totale.

### 2.2 Dataset 2 : Diabetes Dataset
- **Source** : `sklearn.datasets.load_diabetes` (Utilisé initialement dans l'algorithme LARS).
- **Domaine** : Santé (Régression de la progression de la maladie).
- **Statut Éthique** : **✅ CONFORME**. Les variables (âge, sexe, IMC, pression artérielle, 6 mesures sériques) de 442 patients sont standardisées. Les données ont été collectées avec consentement pour la recherche et sont **totalement anonymisées**. Aucun risque d'ingénierie inverse pour ré-identifier un patient.

### 2.3 Dataset 3 : Heart Disease (Statlog) / Wine
- **Source** : OpenML (`fetch_openml`) ou `sklearn.datasets.load_wine`.
- **Domaine** : Santé (Cardiologie) / Agriculture.
- **Statut Éthique** : **✅ CONFORME**. Données historiques et anonymisées du domaine public.

---

## 3. Conformité Vision Bénin 2060

La Vision Bénin 2060 met un accent fort sur la souveraineté numérique et la protection des citoyens.

### 3.1 Ce qui est validé aujourd'hui
- La publication de notre benchmark v1.0 **ne viole aucune loi** sur la protection des données (ni le code du numérique du Bénin, ni le RGPD européen). 
- Nous pouvons publier les résultats d'Accuracy et de R² obtenus sur ces jeux de données de façon ouverte, transparente, et sans risque juridique, car il s'agit de **données ouvertes (Open Data)**.

### 3.2 Défis pour le déploiement local (Phase Future)
Cependant, l'objectif de `FaNet` est d'être déployé sur des **données locales béninoises et africaines**. Dès que nous commencerons à collecter nos propres données (ex: centres de santé ruraux, hôpitaux locaux), les règles changeront drastiquement.

> [!WARNING]
> FaNet est capable d'extraire des correspondances médicales endogènes. L'association entre le dossier médical d'un patient et son "interprétation Fâ" crée une donnée ultra-sensible (Donnée de Santé + Donnée de Croyance/Culture), nécessitant un traitement classé "Haut Risque".

---

## 4. Recommandations EthiTec (Action Plan)

Pour préparer le passage à des données ouest-africaines réelles, les actions suivantes sont requises :

| Action | Criticité | Description |
|--------|-----------|-------------|
| **Protocole de Consentement** | 🔴 Haute | Rédiger un formulaire de consentement éclairé spécifiant que les données alimenteront une IA "endogène" (FaNet). Le patient doit comprendre qu'une interprétation issue de la Matrice-256 sera générée. |
| **Pipeline d'Anonymisation Local** | 🔴 Haute | Avant toute ingestion de données locales par `TorchFaNet`, implémenter un script de hachage irréversible des clés primaires (NPI, noms). |
| **Biais de Représentation** | 🟡 Moyenne | Actuellement, FaNet apprend sur des données occidentales (Wisconsin). Il faudra qualifier la validité des modèles d'inférence appliqués à une démographie différente pour éviter des biais de diagnostic métabolique. |
| **Déclaration APDP** | 🔴 Haute | Toute base de données locales de santé utilisée pour entraîner FaNet devra être déclarée à l'Autorité de Protection des Données Personnelles (APDP) du Bénin. |

---
**Conclusion de l'audit** : Les fondations empiriques actuelles (v1.0) sont exemptes de tout risque de fuite de données personnelles ou de violation éthique. La voie est libre pour une publication scientifique avec cas d'usage sur les jeux de données `sklearn`. Il faut impérativement encadrer la phase d'expérimentation locale à venir.

---
*Audit réalisé le 29 mars 2026*
*Document interne — Projet Matrice-256 / bokono*
