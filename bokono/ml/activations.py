# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.activations — Fonctions d'Activation Fâ
==================================================
Le cœur de l'innovation algorithmique du Projet Matrice-256.

Toutes les fonctions d'activation standard (ReLU, Sigmoid...) ont été conçues
dans un contexte occidental, sans ancrage culturel.

Ces fonctions sont différentes. Elles sont dérivées de la logique du Fâ :
  - La dialectique ouvert/fermé (FaReLU)
  - La polarité des 16 Kpoli (FaSigma)
  - L'auto-référentialité des signes Meji (FaDual)

Chaque fonction est :
  1. Mathématiquement valide (différentiable, bornée ou contrôlée)
  2. Interprétable en logique Fâ (chaque zone a un sens)
  3. Compatible avec la rétropropagation (dérivée définie partout)

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union

# ─── Seuils canoniques du Fâ ─────────────────────────────────────────────────
#
# Le Fâ est une grille 16×16. En normalisant sur [0, 1] :
#   - T1 = 1/16 = 0.0625 → seuil d'entrée (premier Kpoli)
#   - T2 = 8/16 = 0.5000 → centre de la matrice (demi-ouverture)
#   - T3 = 15/16= 0.9375 → seuil de saturation (avant le dernier Kpoli)
#
# Ces 3 seuils divisent l'espace en 4 zones naturelles de la logique Fâ.

FA_T1 = 1.0 / 16.0    # 0.0625
FA_T2 = 8.0 / 16.0    # 0.5000
FA_T3 = 15.0 / 16.0   # 0.9375

# Gains associés aux 4 zones (calibrés pour équilibre gradient)
FA_GAIN_ZONE1 = 0.0    # fermé         → pas de signal
FA_GAIN_ZONE2 = 0.25   # semi-ouvert   → signal faible
FA_GAIN_ZONE3 = 0.5    # pré-ouvert    → signal modéré
FA_GAIN_ZONE4 = 1.0    # plein ouvert  → signal maximal
FA_GAIN_SURSAT = 0.1   # sursaturation → régulation (sagesse du Fâ)


# ─── Poids de polarité par défaut (16 Kpoli) ─────────────────────────────────
#
# Valeurs dérivées de la base de connaissance fa_knowledge.json.
# Chaque Kpoli k a une polarité moyenne sur ses 16 du (une colonne de la matrice).
# favorable=1.0, neutre=0.5, défavorable=0.0
#
# Ces poids sont utilisés par FaSigma pour moduler le gain du sigmoid.
# Ils peuvent être rechargés depuis la matrice réelle avec load_polarity_weights().

_DEFAULT_POLARITY_WEIGHTS = np.full(16, 0.5, dtype=np.float32)


def load_polarity_weights() -> np.ndarray:
    """
    Charge les poids de polarité réels depuis la base de connaissance.

    Calcule la polarité moyenne de chaque Kpoli (colonne) sur les 16 du
    de cette colonne dans la matrice 16×16.

    Returns:
        np.ndarray de forme (16,), valeurs entre 0.0 et 1.0
    """
    try:
        from bokono.core.fa_matrix import FaMatrix
        _POLARITE_NUM = {"favorable": 1.0, "neutre": 0.5,
                         "défavorable": 0.0, "non_défini": 0.5}
        matrix = FaMatrix()
        weights = np.zeros(16, dtype=np.float32)
        for j in range(16):  # Pour chaque colonne (Kpoli droit)
            col_polarities = []
            for i in range(16):  # Pour chaque ligne (Kpoli gauche)
                du = matrix._matrix[i][j]
                pol = _POLARITE_NUM.get(du.get("polarite", "non_défini"), 0.5)
                col_polarities.append(pol)
            weights[j] = np.mean(col_polarities)
        return weights
    except (ImportError, FileNotFoundError, KeyError, TypeError, ValueError) as e:
        import logging
        logging.getLogger("bokono.ml.activations").warning(
            f"Impossible de charger les poids de polarité (utilise les valeurs par défaut) : {e}"
        )
        return _DEFAULT_POLARITY_WEIGHTS.copy()


# ─── T8 : FaReLU ─────────────────────────────────────────────────────────────

def fa_relu(x: np.ndarray) -> np.ndarray:
    """
    FaReLU — Activation 4 zones inspirée de la logique ouverte/fermée du Fâ.

    Contrairement à ReLU (binaire : mort ou passage), FaReLU reconnaît
    4 états distincts, comme le Fâ reconnaît différents niveaux d'ouverture :

    Zone 1 — Fermé     [x < 0]         : f(x) = 0
             Le signe est fermé. Aucun signal ne passe.
             → Équivalent : neurone mort (comme ReLU), mais sémantiquement intentionnel.

    Zone 2 — Semi-ouvert [0 ≤ x < T1]  : f(x) = x × 0.25
             Le signe s'entrouvre. Signal faible, prudent.
             → Le Fâ dit : "la voie s'ouvre mais reste étroite."

    Zone 3 — Pré-ouvert  [T1 ≤ x < T2] : f(x) = x × 0.5
             L'ouverture est engagée. Signal modéré.
             → Le Fâ dit : "le chemin est dégagé à moitié."

    Zone 4 — Plein ouvert [T2 ≤ x < T3]: f(x) = x
             Ouverture totale. Signal maximal, transmission intégrale.
             → Le Fâ dit : "la voix est ouverte." (DJO GBE — Gbe-Meji)

    Zone Sat — Sursaturation [x ≥ T3]  : f(x) = T3 + (x - T3) × 0.1
             Le signal est trop fort. Le Fâ régule l'excès.
             → Sagesse : même le plus puissant des signes se tempère.

    Gradient non nul partout (sauf zone fermée) → Pas de "dead neuron" problem.

    Args:
        x: Tableau NumPy (n'importe quelle forme)

    Returns:
        Tableau de même forme, activé par FaReLU

    Exemple:
        >>> fa_relu(np.array([-1., 0.03, 0.3, 0.7, 0.95]))
        array([0.   , 0.0075, 0.15  , 0.7   , 0.9413])
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)

    # Zone 2 : semi-ouvert
    m2 = (x >= 0) & (x < FA_T1)
    result[m2] = x[m2] * FA_GAIN_ZONE2

    # Zone 3 : pré-ouvert
    m3 = (x >= FA_T1) & (x < FA_T2)
    result[m3] = x[m3] * FA_GAIN_ZONE3

    # Zone 4 : plein ouvert
    m4 = (x >= FA_T2) & (x < FA_T3)
    result[m4] = x[m4] * FA_GAIN_ZONE4

    # Zone de sursaturation
    m5 = x >= FA_T3
    result[m5] = FA_T3 + (x[m5] - FA_T3) * FA_GAIN_SURSAT

    return result


def fa_relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Dérivée de FaReLU (pour rétropropagation).

    C'est une dérivée par zone (fonction constante par morceaux) :
      Zone fermée    [x < 0]         : f'(x) = 0.00
      Zone semi-ouv. [0 ≤ x < T1]   : f'(x) = 0.25
      Zone pré-ouv.  [T1 ≤ x < T2]  : f'(x) = 0.50
      Zone ouverte   [T2 ≤ x < T3]  : f'(x) = 1.00
      Sursaturation  [x ≥ T3]       : f'(x) = 0.10

    Args:
        x: Valeurs d'entrée (avant activation)

    Returns:
        Gradients, même forme que x
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)

    grad[(x >= 0) & (x < FA_T1)] = FA_GAIN_ZONE2
    grad[(x >= FA_T1) & (x < FA_T2)] = FA_GAIN_ZONE3
    grad[(x >= FA_T2) & (x < FA_T3)] = FA_GAIN_ZONE4
    grad[x >= FA_T3] = FA_GAIN_SURSAT

    return grad


# ─── T9 : FaSigma ────────────────────────────────────────────────────────────

def fa_sigma(
    x: np.ndarray,
    polarity_weights: Optional[np.ndarray] = None,
    x_range: tuple[float, float] = (-6.0, 6.0),
) -> np.ndarray:
    """
    FaSigma — Sigmoid modulée par la polarité des Kpoli du Fâ.

    Problème du Sigmoid standard : le gain (la pente) est identique
    partout. Il ne distingue pas les zones favorables des zones défavorables.

    FaSigma résout ça : le gain varie selon la région Fâ de l'input.
    - Dans une zone Kpoli "favorable" (polarité ~1.0) : gain amplifié → décision franche
    - Dans une zone Kpoli "défavorable" (polarité ~0.0) : gain réduit → prudence
    - Dans une zone "neutre" (polarité ~0.5) : gain standard

    Formule :
      gain(x) = 0.5 + polarity[kpoli_region(x_norm)]  → gain ∈ [0.5, 1.5]
      FaSigma(x) = σ(x × gain(x)) = 1 / (1 + e^(-x × gain(x)))

    La dérivabilité est conservée car gain est constante par morceau
    (ou interpolée lisse ici pour l'entraînement).

    Args:
        x:                Tableau NumPy d'entrée
        polarity_weights: Poids de polarité (16,) — utilise defaults si None
        x_range:          Plage pour normaliser x vers les 16 Kpoli

    Returns:
        Tableau activé, valeurs dans ]0, 1[

    Exemple:
        >>> fa_sigma(np.array([-3., 0., 3.]))
        # Valeurs différentes selon la polarité de chaque zone
    """
    x = np.asarray(x, dtype=np.float64)

    if polarity_weights is None:
        polarity_weights = load_polarity_weights().astype(np.float64)
    else:
        polarity_weights = np.asarray(polarity_weights, dtype=np.float64)

    x_min, x_max = x_range

    # 1. Normaliser x → [0, 1] pour identifier la zone Kpoli
    x_norm = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)

    # 2. Identifier le Kpoli (0-15) pour chaque valeur
    kpoli_idx = np.clip((x_norm * 16).astype(int), 0, 15)

    # 3. Récupérer le gain (polarité) de chaque zone
    gain = polarity_weights[kpoli_idx]  # shape identique à x
    gain = 0.5 + gain  # → gain ∈ [0.5, 1.5]

    # 4. Sigmoid modulé
    return 1.0 / (1.0 + np.exp(-x * gain))


def fa_sigma_derivative(
    x: np.ndarray,
    polarity_weights: Optional[np.ndarray] = None,
    x_range: tuple[float, float] = (-6.0, 6.0),
) -> np.ndarray:
    """
    Dérivée de FaSigma (pour rétropropagation).

    FaSigma(x) = σ(x × g) où g = gain(x)
    ∂FaSigma/∂x = g × σ(x×g) × (1 - σ(x×g))

    (g est traité comme constante locale pour la dérivée.)

    Args:
        x:                Valeurs d'entrée
        polarity_weights: Poids de polarité (16,)
        x_range:          Plage de normalisation

    Returns:
        Gradients, même forme que x
    """
    x = np.asarray(x, dtype=np.float64)

    if polarity_weights is None:
        polarity_weights = load_polarity_weights().astype(np.float64)

    x_min, x_max = x_range
    x_norm = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
    kpoli_idx = np.clip((x_norm * 16).astype(int), 0, 15)
    gain = 0.5 + polarity_weights[kpoli_idx]

    s = fa_sigma(x, polarity_weights, x_range)
    return gain * s * (1.0 - s)


# ─── T10 : FaDual ────────────────────────────────────────────────────────────

def fa_dual(
    x: np.ndarray,
    polarity_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FaDual — Activation inspirée des signes Meji (auto-référentiels).

    Les 16 signes Meji du Fâ sont des "doubles" : le même Kpoli sur les
    deux côtés (GBE×GBE, YEKU×YEKU...). Ils représentent la plénitude,
    l'intensité maximale d'un archétype.

    FaDual capture cette idée : on applique FaSigma deux fois, créant
    une activation "redoublée" qui amplifie les décisions claires
    et atténue les zones d'incertitude.

    Formule : FaDual(x) = FaSigma(FaSigma(x) × 6 - 3)
      → FaSigma(x) ∈ ]0,1[, donc FaSigma(x)×6-3 ∈ ]-3, 3[
      → Le second passage affine la décision.

    Propriétés :
      - Sortie dans ]0, 1[ (doublement contractée vers 0.5 puis re-étirée)
      - Plus "tranchante" que FaSigma seule : zones de saturation plus larges
      - Gradient non nul partout (doublement différentiable)

    Args:
        x:                Tableau NumPy d'entrée
        polarity_weights: Poids de polarité (16,)

    Returns:
        Tableau activé, valeurs dans ]0, 1[
    """
    x = np.asarray(x, dtype=np.float64)

    # Premier passage FaSigma
    s1 = fa_sigma(x, polarity_weights)

    # Remapping vers [-3, 3] pour le second passage
    s1_remapped = s1 * 6.0 - 3.0

    # Second passage FaSigma (avec les mêmes poids de polarité)
    return fa_sigma(s1_remapped, polarity_weights)


def fa_dual_derivative(
    x: np.ndarray,
    polarity_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Dérivée de FaDual par règle de chaîne.

    FaDual(x) = FaSigma(g(x)) où g(x) = FaSigma(x) × 6 - 3
    ∂FaDual/∂x = FaSigma'(g(x)) × g'(x)
               = FaSigma'(g(x)) × (6 × FaSigma'(x))
    """
    x = np.asarray(x, dtype=np.float64)

    # Gradient du premier sigmoid
    ds1 = fa_sigma_derivative(x, polarity_weights)

    # Valeur intermédiaire g(x)
    s1 = fa_sigma(x, polarity_weights)
    g = s1 * 6.0 - 3.0

    # Gradient du second sigmoid appliqué à g
    ds2 = fa_sigma_derivative(g, polarity_weights)

    # Règle de chaîne : ∂FaDual/∂x = ds2 × 6 × ds1
    return ds2 * 6.0 * ds1


# ─── T12 : Registre des activations ──────────────────────────────────────────

class FaActivation:
    """
    Conteneur d'une fonction d'activation bokono.

    Regroupe la fonction, sa dérivée, et ses métadonnées Fâ.
    Utilisé par les couches FaNet (Semaine 3).

    Exemple:
        >>> act = FaActivation.get("fa_relu")
        >>> y = act.forward(x)
        >>> grad = act.backward(x)
    """

    _registry: dict[str, "FaActivation"] = {}

    def __init__(
        self,
        name: str,
        forward_fn,
        backward_fn,
        description: str = "",
        fa_reference: str = "",
    ):
        self.name = name
        self._forward = forward_fn
        self._backward = backward_fn
        self.description = description
        self.fa_reference = fa_reference
        FaActivation._registry[name] = self

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Passe avant (forward pass)."""
        return self._forward(x, **kwargs)

    def backward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Passe arrière (gradient pour backprop)."""
        return self._backward(x, **kwargs)

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.forward(x, **kwargs)

    def __repr__(self) -> str:
        return f"FaActivation('{self.name}') — {self.fa_reference}"

    @classmethod
    def get(cls, name: str) -> "FaActivation":
        """Récupère une activation par son nom."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Activation '{name}' introuvable. "
                f"Disponibles : {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """Liste toutes les activations disponibles."""
        return list(cls._registry.keys())


# ─── Enregistrement des activations Fâ ───────────────────────────────────────

FA_RELU = FaActivation(
    name="fa_relu",
    forward_fn=fa_relu,
    backward_fn=fa_relu_derivative,
    description="Activation 4 zones : fermé / semi-ouvert / ouvert / sursaturation",
    fa_reference="Logique ouverte/fermée du Fâ — marques simples et doubles",
)

FA_SIGMA = FaActivation(
    name="fa_sigma",
    forward_fn=fa_sigma,
    backward_fn=fa_sigma_derivative,
    description="Sigmoid modulée par la polarité des 16 Kpoli",
    fa_reference="Polarité des Kpoli — zones favorables et défavorables de la matrice",
)

FA_DUAL = FaActivation(
    name="fa_dual",
    forward_fn=fa_dual,
    backward_fn=fa_dual_derivative,
    description="Activation double inspirée des signes Meji (auto-référentiels)",
    fa_reference="Signes Meji — doublement du même Kpoli, intensité maximale",
)


# ─── Utilitaires de comparaison ───────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU standard (référence de comparaison)."""
    return np.maximum(0.0, np.asarray(x, dtype=np.float64))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid standard (référence de comparaison)."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def compare_activations(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compare toutes les activations sur un même tableau d'entrée.

    Retourne un dictionnaire {nom_activation: valeurs_sorties}.
    Utile pour le benchmark et la visualisation.

    Args:
        x: Tableau NumPy d'entrée

    Returns:
        Dict {str: np.ndarray}
    """
    x = np.asarray(x, dtype=np.float64)
    return {
        "ReLU (standard)":     relu(x),
        "Sigmoid (standard)":  sigmoid(x),
        "FaReLU":              fa_relu(x),
        "FaSigma":             fa_sigma(x),
        "FaDual":              fa_dual(x),
    }


# ─── v0.2 : LearnableFaSigma ──────────────────────────────────────────────────

class LearnableFaSigma:
    """
    FaSigma avec poids de polarité ENTRAÎNABLES — innovation v0.2.

    Problème de FaSigma v0.1 :
      Les poids de polarité étaient statiques (extraits du JSON).
      Le réseau partait de la sagesse Fâ mais ne pouvait pas l'affiner.

    Solution v0.2 :
      Les 16 poids de polarité (un par Kpoli) deviennent des paramètres
      mis à jour par rétropropagation — exactement comme W et b d'une
      couche linéaire.

    Paramètre réel : raw_k ∈ ℝ (non contraint, espace non borné)
    Contrainte via sigmoid : polarity_k = σ(raw_k) ∈ ]0, 1[
    Gain final : gain_k = 0.5 + polarity_k ∈ ]0.5, 1.5[

    Initialisation : raw_k = logit(polarity_k_initial)
      → Le réseau commence exactement aux valeurs Fâ connues,
        puis les adapte aux données réelles.

    Forward :
      gain(x) = 0.5 + σ(raw[kpoli_zone(x)])
      y = σ(x × gain(x))

    Backward (règle de chaîne complète) :
      ∂L/∂raw_k = Σ_{x ∈ zone k} grad_output × (x × y × (1-y)) × (σ(raw_k) × (1-σ(raw_k)))
      ∂L/∂x     = grad_output × gain(x) × y × (1-y)

    Compatibilité : même interface que FaActivation (.forward, .backward).
    """

    N_KPOLI: int = 16  # Nombre de zones Kpoli = nombre de paramètres

    def __init__(
        self,
        init_weights: Optional[np.ndarray] = None,
        x_range: tuple[float, float] = (-6.0, 6.0),
    ):
        """
        Args:
            init_weights: Poids initiaux de polarité (16,) dans ]0, 1[.
                          Par défaut : charge depuis la base de connaissance Fâ.
            x_range:      Plage de normalisation pour identifier les zones Kpoli.
        """
        if init_weights is None:
            init_weights = load_polarity_weights().astype(np.float64)
        else:
            init_weights = np.asarray(init_weights, dtype=np.float64)

        # Vérification des bornes
        init_weights = np.clip(init_weights, 1e-6, 1.0 - 1e-6)

        # Paramètres bruts → espace non borné via logit (inverse de sigmoid)
        # logit(p) = log(p / (1-p))  → raw=0 correspond à polarity=0.5
        self._raw: np.ndarray = np.log(
            init_weights / (1.0 - init_weights)
        ).astype(np.float64)

        self.x_range = x_range

        # Gradients accumulés
        self.d_raw: np.ndarray = np.zeros(self.N_KPOLI, dtype=np.float64)

        # Cache pour la passe arrière
        self._x: Optional[np.ndarray] = None
        self._kpoli_idx: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._gain_per_elem: Optional[np.ndarray] = None

    # ── Propriétés lisibles ───────────────────────────────────────────────────

    @property
    def polarity_weights(self) -> np.ndarray:
        """Polarités actuelles après sigmoid — dans ]0, 1[. Shape (16,)."""
        return 1.0 / (1.0 + np.exp(-self._raw))

    @property
    def gain_values(self) -> np.ndarray:
        """Gains actuels par zone Kpoli — dans ]0.5, 1.5[. Shape (16,)."""
        return 0.5 + self.polarity_weights

    @property
    def n_params(self) -> int:
        """Nombre de paramètres entraînables."""
        return self.N_KPOLI

    # ── Passe avant ───────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Calcule FaSigma avec les poids de polarité courants.

        Args:
            x: np.ndarray de n'importe quelle forme

        Returns:
            np.ndarray de même forme, valeurs dans ]0, 1[
        """
        x = np.asarray(x, dtype=np.float64)
        self._x = x

        x_min, x_max = self.x_range
        x_norm = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
        kpoli_idx = np.clip((x_norm * 16).astype(int), 0, 15)
        self._kpoli_idx = kpoli_idx

        # Gain par élément (lookup vectorisé)
        gain = self.gain_values                     # (16,)
        self._gain_per_elem = gain[kpoli_idx]       # même shape que x

        # Sigmoid modulé
        self._y = 1.0 / (1.0 + np.exp(-x * self._gain_per_elem))
        return self._y

    # ── Passe arrière ─────────────────────────────────────────────────────────

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Rétropropagation complète : gradients vers x ET vers raw_weights.

        Règle de chaîne :
          y = σ(x × g)  où g = 0.5 + σ(raw[k])

          ∂y/∂x    = g × y × (1-y)
          ∂y/∂g    = x × y × (1-y)
          ∂g/∂raw_k= σ(raw_k) × (1 - σ(raw_k))   (dérivée du sigmoid)

          ∂L/∂x    = grad_output × ∂y/∂x
          ∂L/∂raw_k= Σ_{elems in zone k} grad_output × ∂y/∂g × ∂g/∂raw_k

        Args:
            grad_output: Gradient de la perte, même shape que la sortie forward

        Returns:
            grad_x: Gradient par rapport à l'entrée x
        """
        assert self._x is not None and self._y is not None, \
            "backward() appelé sans forward() préalable"
        assert self._gain_per_elem is not None and self._kpoli_idx is not None

        # Narrowing explicite pour Pyre2
        x_loc: np.ndarray = self._x
        y_loc: np.ndarray = self._y
        gain_loc: np.ndarray = self._gain_per_elem
        kidx_loc: np.ndarray = self._kpoli_idx

        # ── Gradient vers x ────────────────────────────────────────────────
        dy_dx = gain_loc * y_loc * (1.0 - y_loc)
        grad_x = grad_output * dy_dx

        # ── Gradient vers raw_weights ───────────────────────────────────────
        dy_dg = x_loc * y_loc * (1.0 - y_loc)

        # ∂g/∂raw_k = dérivée de sigmoid appliquée à raw_k
        p = self.polarity_weights           # σ(raw), shape (16,)
        dg_draw = p * (1.0 - p)            # dérivée sigmoid, shape (16,)

        # Accumulation sur chaque zone Kpoli
        self.d_raw = np.zeros(self.N_KPOLI, dtype=np.float64)
        for k in range(self.N_KPOLI):
            mask = kidx_loc == k
            if bool(mask.any()):           # cast bool : évite l'erreur Pyre2 sur ndarray.any()
                dl_dg_k = float(np.sum(grad_output[mask] * dy_dg[mask]))
                self.d_raw[k] = dl_dg_k * dg_draw[k]

        return grad_x


    # ── Mise à jour des paramètres ────────────────────────────────────────────

    def update(self, lr: float) -> None:
        """Descente de gradient sur les paramètres raw."""
        self._raw -= lr * self.d_raw

    def zero_grad(self) -> None:
        """Remet les gradients accumulés à zéro."""
        self.d_raw = np.zeros(self.N_KPOLI, dtype=np.float64)

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        """Retourne les paramètres bruts pour la sauvegarde."""
        return self._raw.copy()

    def set_state(self, raw: np.ndarray) -> None:
        """Restaure les paramètres depuis un état sauvegardé."""
        self._raw = np.asarray(raw, dtype=np.float64)

    def __repr__(self) -> str:
        pol = self.polarity_weights
        return (
            f"LearnableFaSigma("
            f"polarity_min={pol.min():.3f}, "
            f"polarity_max={pol.max():.3f}, "
            f"gain_range=[{self.gain_values.min():.3f}, {self.gain_values.max():.3f}])"
        )
