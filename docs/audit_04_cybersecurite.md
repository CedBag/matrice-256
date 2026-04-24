# AUDIT 1/2 — Cybersécurité & Vulnérabilité (Pen-testing Approfondi)
**Projet** : bokono / Matrice-256 — FaNet v3.0  
**Date** : 2026-03-29  
**Auditeur** : Antigravity — Mode Innovateur Réaliste  
**Périmètre** : Code source Python complet (`bokono/`), configuration projet, données, persistance modèle  

---

## Tableau de Synthèse

| # | Vulnérabilité | Sévérité | Fichier | Statut |
|---|---|---|---|---|
| V-01 | `torch.load(weights_only=False)` — Exécution de code arbitraire | CRITIQUE | torch_network.py:287 | À CORRIGER |
| V-02 | `np.load(allow_pickle=True)` — Désérialisation pickle dangereuse | CRITIQUE | network.py:375 | À CORRIGER |
| V-03 | Aucune validation des entrées `.fit()` / `.predict()` | HAUTE | torch_network.py | À CORRIGER |
| V-04 | PDFs sources présents dans `data/` et potentiellement commités | HAUTE | .gitignore | À VÉRIFIER |
| V-05 | Aucun système de logging (erreurs silencieuses possibles) | MOYENNE | Tout le code | À CORRIGER |
| V-06 | Exception broad `except Exception:` (masque les vrais bugs) | MOYENNE | activations.py:86 | À CORRIGER |
| V-07 | Seed cryptographique faible (seed=42 par défaut partout) | MOYENNE | Tout le code ML | INFO |
| V-08 | Pas de hash d'intégrité sur la KB JSON | MOYENNE | fa_encoder.py | À CORRIGER |
| V-09 | Variable globale mutable `_KB` (cache en mémoire) | BASSE | fa_encoder.py:92 | INFO |
| V-10 | Pas de `requirements.txt` / lock file de dépendances | MOYENNE | Racine projet | À CORRIGER |
| V-11 | Email de contact fictif dans pyproject.toml | BASSE | pyproject.toml:13 | INFO |

---

## Analyse Détaillée des Vulnérabilités

### V-01 : `torch.load(weights_only=False)` — CRITIQUE (RCE)

**Fichier** : torch_network.py ligne 287

```python
data = torch.load(path, weights_only=False)  # DANGER
```

**Vecteur d'attaque** : Un attaquant qui fournit un fichier `.pth` piégé peut exécuter du code Python arbitraire sur la machine qui charge le modèle. `torch.load` utilise `pickle` en interne. Avec `weights_only=False`, il n'y a aucune protection.

**Impact** : Exécution de code arbitraire sur le serveur ou le poste de l'utilisateur. Vol de la KB privée, destruction de données, ransomware.

**Correction** :
```python
# AVANT (vulnérable)
data = torch.load(path, weights_only=False)

# APRÈS (sécurisé)
data = torch.load(path, weights_only=True)
```
Nécessite de sauvegarder uniquement des types sûrs (dict, tenseurs) — ce qui est déjà le cas dans `save()`.

---

### V-02 : `np.load(allow_pickle=True)` — CRITIQUE (RCE)

**Fichier** : network.py ligne 375

```python
data = np.load(path, allow_pickle=True)  # DANGER
```

**Vecteur d'attaque** : Identique à V-01 mais via NumPy. Un fichier `.npz` piégé peut contenir des objets pickle malicieux.

**Correction** :
```python
# APRÈS (sécurisé) — ne charger que des arrays purs
data = np.load(path, allow_pickle=False)
```
Impact : nécessite de ne plus sauvegarder de types non-array dans le `.npz`. Le champ `learnable_sigma` (booléen) devra être converti en `np.array([True])`.

---

### V-03 : Aucune Validation des Entrées

**Fichier** : torch_network.py lignes 92-130

Les méthodes `.fit()`, `.predict()`, `.evaluate()` acceptent directement les arrays NumPy de l'utilisateur sans aucune vérification :
- Pas de contrôle de type (`X` pourrait être une string)
- Pas de contrôle de dimension (`X` à 3D passerait silencieusement)
- Pas de contrôle de NaN/Inf (corrompt l'entraînement silencieusement)
- Pas de contrôle de cohérence `X.shape[1] == n_features`

**Correction** : Ajouter une fonction `_validate_input(X, y=None)` centralisée.

---

### V-04 : PDFs Sources Présents et Non-Gitignored Correctement

**Répertoire** : `data/`

```
data/Fâ.pdf                          (554 KB)
data/Les-256-Signe-de-Fa-Fon.pdf     (468 KB)
data/Les-256-Signes-Du-Fa-Officielpdf.pdf (3.6 MB)
```

Le `.gitignore` couvre `data/*.pdf` mais ces fichiers sont peut-être déjà commités dans l'historique Git. Si le repo est poussé sur GitHub, les documents sources originaux du Fâ (votre Secret Industriel) seront exposés publiquement.

**Correction** :
1. Vérifier si déjà commités : `git log --all --full-history -- "data/*.pdf"`
2. Si oui, purger l'historique Git avec `git filter-repo`
3. Confirmer que `data/*.pdf` est dans `.gitignore` (déjà fait)

---

### V-05 : Aucun Système de Logging

**Constat** : Le grep sur `logging|logger` renvoie zéro résultat. Le code n'a aucun système de journalisation. En production :
- Les erreurs de chargement KB sont silencieuses
- Les échecs d'entraînement ne laissent aucune trace
- Impossible de faire du forensics en cas d'incident

**Correction** : Implémenter `logging` avec niveaux (DEBUG/INFO/WARNING/ERROR).

---

### V-06 : `except Exception:` — Exceptions Trop Larges

**Fichiers** : activations.py ligne 86, benchmark.py ligne 183

```python
except Exception:  # Masque TOUTES les erreurs
    return _DEFAULT_POLARITY_WEIGHTS.copy()
```

Un attaquant pourrait provoquer une erreur critique (PermissionError, MemoryError) qui serait silencieusement avalée, faisant tourner le modèle avec des poids par défaut incorrects.

**Correction** : Remplacer par des exceptions spécifiques (`FileNotFoundError`, `json.JSONDecodeError`, `KeyError`).

---

### V-08 : Pas de Hash d'Intégrité sur la KB JSON

**Fichier** : fa_encoder.py lignes 73-90

La base de connaissance est chargée via `json.load()` sans aucune vérification d'intégrité. Un attaquant avec accès au système de fichiers pourrait modifier `fa_knowledge_private.json` pour corrompre les prédictions du modèle (data poisoning).

**Correction** : Calculer un SHA-256 à la première utilisation, le stocker, et vérifier à chaque chargement.

---

### V-10 : Pas de Lock File de Dépendances

**Constat** : Le `pyproject.toml` spécifie `numpy>=1.24` sans plafond. Un `pip install` futur pourrait installer une version incompatible ou vulnérable. Aucun `requirements.txt` figé n'existe.

**Correction** : Générer `pip freeze > requirements.lock` et le versionner.

---

## Posture de Sécurité Globale

| Catégorie | Score |
|---|---|
| Surface d'attaque réseau | NULLE (pas de serveur web, pas d'API, pas de sockets) |
| Injection de code (RCE) | 2 vulnérabilités critiques (torch.load + np.load) |
| Validation d'entrées | Absente — risque de crash et corruption silencieuse |
| Gestion des secrets | Correcte — gitignore, scission pub/priv |
| Journalisation / Audit trail | Inexistante |
| Intégrité des données | Aucun contrôle de hash |
| Dépendances | Non verrouillées |

---

## Plan de Remédiation Recommandé

### Actions immédiates (à faire maintenant)
1. **V-01** : Passer `torch.load` à `weights_only=True`
2. **V-02** : Passer `np.load` à `allow_pickle=False` + adapter la sauvegarde
3. **V-03** : Créer `_validate_input()` dans torch_network.py
4. **V-04** : Vérifier et purger les PDFs de l'historique Git

### Actions court terme
5. **V-05** : Injecter `import logging` dans les modules critiques
6. **V-06** : Rétrécir les `except` à des exceptions spécifiques
7. **V-08** : Ajouter vérification SHA-256 de la KB
8. **V-10** : Générer `requirements.lock`
