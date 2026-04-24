# AUDIT 1/3 — Propriété Intellectuelle (PI) et Juridique
## Projet bokono / Matrice-256
## Date : 29 mars 2026

---

## 1. Périmètre de l'audit

Cet audit examine la protection juridique du code source Python de la bibliothèque `bokono` et la protection du secret industriel autour de la base de données originelle du Fâ.

### Fichiers audités
| Fichier | Lignes | Objet |
|---------|--------|-------|
| `bokono/__init__.py` | 12 | Package principal |
| `bokono/core/fa_encoder.py` | ~300 | Encodeur Fâ↔Binaire |
| `bokono/core/fa_matrix.py` | ~210 | Matrice 16×16 |
| `bokono/ml/activations.py` | ~685 | Fonctions d'activation (FaReLU, FaSigma, LearnableFaSigma) |
| `bokono/ml/layers.py` | ~400 | Couches réseau (Embedding, Hidden, Decoder) |
| `bokono/ml/network.py` | ~400 | FaNet v0.2 (NumPy) |
| `bokono/ml/torch_*.py` | ~500 | FaNet v0.3 (PyTorch) |
| `bokono/ml/benchmark.py` | ~280 | Moteur de benchmark v1.0 |
| `data/knowledge_base/fa_knowledge.json` | 6526 | Base de connaissance Fâ (256 signes) |

---

## 2. Protection du code source par le droit d'auteur

### 2.1 Constats

> [!CAUTION]
> **AUCUN en-tête de copyright n'existe dans aucun fichier source.**
> La mention `__license__ = "MIT"` dans `__init__.py` et `pyproject.toml` existe mais est INSUFFISANTE.

| Élément | État actuel | Recommandation |
|---------|-------------|----------------|
| Header copyright dans les fichiers `.py` | ❌ **ABSENT** | Ajouter `# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256` |
| Fichier `LICENSE` à la racine | ❌ **ABSENT** | Créer le fichier avec le texte complet de la licence choisie |
| `__author__` dans `__init__.py` | ⚠️ "Projet Matrice-256" (trop vague) | Spécifier le nom légal de l'auteur/entité |
| Clause d'attribution dans la licence | ❌ **ABSENT** | Intégrer une clause de reconnaissance culturelle |
| `pyproject.toml` email | ⚠️ `contact@matrice256.bj` | Vérifier que ce domaine est enregistré |

### 2.2 Problème critique : Choix de la licence

> [!WARNING]
> **La licence MIT est TROP PERMISSIVE pour ce projet.**
> 
> La MIT permet à quiconque de :
> - Copier l'intégralité du code
> - Le modifier sans attribution culturelle
> - Le commercialiser sans reverser quoi que ce soit
> - Le breveter dans un autre pays
> 
> **Pour un projet d'innovation endogène béninoise visant la protection IP, c'est une erreur stratégique.**

### 2.3 Recommandations de licence

| Licence | Permet copie | Oblige source ouverte | Protège commercialement | Recommandation |
|---------|-------------|----------------------|------------------------|----------------|
| MIT | ✅ | ❌ | ❌ | ❌ **Non recommandée** |
| Apache 2.0 | ✅ | ❌ | ⚠️ (protection brevets) | ⚠️ Meilleure que MIT |
| AGPL-3.0 | ✅ | ✅ (même en SaaS) | ⚠️ | ✅ Pour la recherche |
| BSL 1.1 (Business Source License) | ⚠️ (délai) | ✅ (après délai) | ✅ | ✅ **Recommandée** |
| Licence propriétaire custom | ❌ | ❌ | ✅ | ✅ Pour la commercialisation |

**Recommandation** : 
- **Phase recherche/publication** → AGPL-3.0 (oblige à partager toute amélioration, même en SaaS)
- **Phase commerciale** → BSL 1.1 ou licence dual (open source + commercial)

---

## 3. Secret industriel — Base de données Fâ

### 3.1 Analyse d'exposition

> [!CAUTION]
> **La base de connaissance Fâ est ENTIÈREMENT EXPOSÉE dans le dépôt.**

Le fichier `data/knowledge_base/fa_knowledge.json` (401 Ko, 6 526 lignes) contient pour chacun des 256 signes :

| Champ | Contenu | Niveau de sensibilité |
|-------|---------|----------------------|
| `nom_fon`, `nom_canonique` | Noms en langue Fon | 🟢 Public (littérature existante) |
| `code_binaire`, `valeur_decimale` | Encodage mathématique | 🟡 Notre innovation |
| `polarite`, `intensite` | Caractérisations numériques | 🔴 **SECRET — notre mapping propriétaire** |
| `interpretation` | Texte complet d'interprétation | 🔴 **SECRET — texte source des PDF** |
| `prescriptions`, `interdictions` | Règles rituelles | 🔴 **SECRET — connaissance traditionnelle** |
| `texte_source` | Texte brut extrait des PDF | 🔴 **ULTRA-SENSIBLE — reproduction directe** |
| `correspondance_medicale` | (null actuellement) | 🟡 Futur secret |

### 3.2 Risques identifiés

1. **Publication sur GitHub** → N'importe qui peut copier l'intégralité de la KB Fâ
2. **Reproduction des textes sources** → Les PDF originaux sont reproduits verbatim dans le JSON
3. **Mapping polarité/intensité** → Notre innovation algorithmique (comment on traduit un signe en nombre) est visible en clair
4. **Poids de polarité par Kpoli** → Les 16 valeurs codées en dur dans `activations.py` (lignes 55-72) révèlent notre traduction numérique du savoir Fâ

### 3.3 Recommandations de protection

| Action | Priorité | Détail |
|--------|----------|--------|
| **Séparer la KB du dépôt public** | 🔴 URGENT | `fa_knowledge.json` ne doit JAMAIS être poussé sur GitHub public |
| **Créer un `.gitignore` pour `data/`** | 🔴 URGENT | `data/knowledge_base/` doit être exclu du VCS |
| **Chiffrer la KB** | 🟡 Important | AES-256 avec clé séparée pour les déploiements |
| **version publique réduite** | 🟡 Important | Créer `fa_knowledge_public.json` avec seulement : `id`, `nom_canonique`, `code_binaire`, `est_meji` |
| **Supprimer `texte_source`** | 🔴 URGENT | Ce champ est la reproduction brute des PDF — risque de plagiat |
| **Externaliser les poids de polarité** | 🟡 Important | Les 16 valeurs dans `activations.py` révèlent la formule secrète |

---

## 4. Recommandations pour la publication scientifique

### Ce qu'on PEUT publier :
- L'architecture FaNet (couches, dimensions, fonctions d'activation)
- La mathématique de FaSigma, FaReLU, FaDual
- Les résultats de benchmark (accuracy, F1, AUC-ROC)
- Le concept d'embedding Fâ (sans les valeurs exactes)
- Le principe de la Matrice 16×16

### Ce qu'on ne DOIT PAS publier :
- Le contenu de `fa_knowledge.json`
- Les 16 poids de polarité par Kpoli (valeurs exactes)
- Les textes d'interprétation des 256 signes
- Les prescriptions et interdictions rituelles
- Le mapping exact `nom_fon → polarité → intensité`

---

## 5. Actions immédiates requises

- [ ] **A1** : Ajouter un en-tête copyright dans tous les fichiers `.py`
- [ ] **A2** : Créer le fichier `LICENSE` avec la licence choisie (AGPL-3.0 recommandée)
- [ ] **A3** : Ajouter `data/knowledge_base/` et `data/*.pdf` au `.gitignore`
- [ ] **A4** : Changer `__license__` de `"MIT"` à la licence choisie dans `__init__.py` et `pyproject.toml`
- [ ] **A5** : Supprimer le champ `texte_source` de la KB ou le déplacer dans un fichier séparé non versionné
- [ ] **A6** : Créer une version publique réduite de la KB pour les tests open-source
- [ ] **A7** : Enregistrer officiellement le copyright auprès de l'OAPI (Organisation Africaine de la Propriété Intellectuelle)
- [ ] **A8** : Rédiger une clause d'attribution culturelle à inclure dans la licence

---

**Conclusion** : Le code est fonctionnel mais juridiquement **non protégé**. La base de connaissance Fâ est **entièrement exposée**. Avant toute publication (GitHub, preprint, ou demande de financement), les actions A1-A6 sont **impératives**.

---
*Audit réalisé le 29 mars 2026*
*Document interne — Projet Matrice-256 / bokono*
