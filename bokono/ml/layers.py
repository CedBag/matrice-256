# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.layers — Couches du réseau FaNet
==========================================
Implémentation des 3 couches fondamentales du réseau de neurones béninois.

Architecture :
  FaEmbeddingLayer → FaHiddenLayer → FaDecoderLayer

Chaque couche a un ancrage direct dans la logique du Fâ :

  FaEmbeddingLayer (Couche d'entrée)
    Traduit les données brutes en "espace Fâ" — trouve quel du correspond
    à chaque valeur d'entrée. Chaque feature devient un vecteur de 4 dimensions
    extrait de la Matrice-256. Analogie : le Bokonon "lit" d'abord les nukets
    avant d'interpréter.

  FaHiddenLayer (Couche cachée)
    Traitement au cœur de la logique Fâ. Poids initialisés depuis la Matrice-256.
    Activation FaReLU (logique ouverte/fermée). Analogie : le Bokonon "pense"
    en croisant les signes entre eux.

  FaDecoderLayer (Couche de sortie)
    Décode la décision finale et la traduit en du Fâ + valeur numérique.
    Activation FaDual (Meji — confirmation double). Analogie : le Bokonon
    "prononce" son verdict, qui est à la fois un chiffre et un signe.

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union

from bokono.ml.activations import (
    fa_relu, fa_relu_derivative,
    fa_sigma, fa_sigma_derivative,
    fa_dual, fa_dual_derivative,
    FaActivation,
    LearnableFaSigma,
    load_polarity_weights,
)
from bokono.core.fa_encoder import value_to_du, _get_kb


# ─── Utilitaires internes ──────────────────────────────────────────────────────

def _xavier_init(shape: tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
    """Initialisation Xavier/Glorot — stable pour sigmoid et tanh."""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return rng.uniform(-limit, limit, shape).astype(np.float64)


def _fa_weight_init(
    shape: tuple[int, int],
    fa_tensor: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Initialisation basée sur le tenseur de la Matrice-256.

    Au lieu de poids aléatoires (Xavier/He), on initialise W depuis la
    Matrice-256 : les poids reflètent la "sagesse" du Fâ dès le départ.

    Si la forme ne correspond pas, on tile/tronque le tenseur Fâ et on
    ajoute un bruit gaussien minimal pour briser la symétrie.

    Args:
        shape:     (n_out, n_in) de la couche
        fa_tensor: Tenseur (16, 16, 4) de la Matrice-256 (optionnel)
        seed:      Graine aléatoire pour la reproductibilité

    Returns:
        np.ndarray de forme `shape`, dtype float64
    """
    if fa_tensor is not None:
        # Flatten le tenseur Fâ → vecteur 1D de 16×16×4 = 1024 valeurs
        flat = fa_tensor.flatten().astype(np.float64)
        n_needed = shape[0] * shape[1]

        # Tile ou tronque pour atteindre la taille nécessaire
        if n_needed <= len(flat):
            base = flat[:n_needed]
        else:
            repeats = (n_needed // len(flat)) + 1
            base = np.tile(flat, repeats)[:n_needed]

        W = base.reshape(shape)

        # Normaliser dans [-0.5, 0.5] + bruit minimal pour briser la symétrie
        W = (W - W.mean()) / (W.std() + 1e-8)
        rng = np.random.default_rng(seed)
        W += rng.normal(0, 0.01, shape)
        return W

    # Fallback : Xavier si pas de tenseur Fâ
    return _xavier_init(shape, seed)


# ─── T15 : FaEmbeddingLayer ───────────────────────────────────────────────────

class FaEmbeddingLayer:
    """
    Couche d'entrée : traduit les données brutes en espace Fâ.

    Pour chaque feature d'entrée, on calcule son vecteur Fâ de 4 dimensions :
      [0] val_norm    : valeur normalisée dans [0, 1]
      [1] polarite    : polarité du du correspondant (0.0=déf, 0.5=neutre, 1.0=fav)
      [2] intensite   : intensité/force du signe
      [3] est_meji    : 1.0 si c'est un signe Meji, 0.0 sinon

    Output shape : (batch_size, n_features × 4)

    Exemple :
        layer = FaEmbeddingLayer(n_features=3)
        x = np.array([[0.2, 0.7, 0.5]])  # 1 échantillon, 3 features
        emb = layer.forward(x)
        print(emb.shape)  # (1, 12)
    """

    FA_EMBED_DIM: int = 4  # Dimensions du vecteur Fâ par feature

    def __init__(self, n_features: int, x_min: float = 0.0, x_max: float = 1.0):
        """
        Args:
            n_features: Nombre de features en entrée
            x_min:      Valeur minimale attendue (pour normalisation)
            x_max:      Valeur maximale attendue (pour normalisation)
        """
        self.n_features = n_features
        self.x_min = x_min
        self.x_max = x_max
        self.output_dim = n_features * self.FA_EMBED_DIM

        # Lazy-load de la KB pour les vecteurs Fâ
        self._kb: Optional[list[dict]] = None

        # Cache des entrées pour la passe arrière
        self._last_input: Optional[np.ndarray] = None

    def _get_kb(self) -> list[dict]:
        if self._kb is None:
            self._kb = _get_kb()
        return self._kb

    def _embed_value(self, v: float) -> np.ndarray:
        """
        Convertit une valeur scalaire en vecteur Fâ de 4 dimensions.

        Stratégie :
        1. Normaliser v → [0, 1]
        2. Trouver le du correspondant (index 0-255)
        3. Retourner son vecteur de caractéristiques

        Args:
            v: Valeur scalaire (dans [x_min, x_max])

        Returns:
            np.ndarray de forme (4,)
        """
        kb = self._get_kb()
        x_range = self.x_max - self.x_min
        normalized = np.clip((v - self.x_min) / (x_range + 1e-10), 0.0, 1.0)
        du_idx = int(normalized * 255)
        du = kb[du_idx]

        # Mapping polarité → float
        pol_map = {"favorable": 1.0, "neutre": 0.5,
                   "défavorable": 0.0, "non_défini": 0.5}
        polarite = pol_map.get(du.get("polarite", "non_défini"), 0.5)
        intensite = float(du.get("intensite", 0.5))
        est_meji = 1.0 if du.get("est_meji", False) else 0.0

        return np.array([float(normalized), polarite, intensite, est_meji],
                        dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passe avant : encode x en vecteurs Fâ.

        Args:
            x: np.ndarray de forme (batch_size, n_features)
               ou (n_features,) pour un seul échantillon

        Returns:
            np.ndarray de forme (batch_size, n_features × 4)
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self._last_input = x
        batch_size, n_feat = x.shape

        if n_feat != self.n_features:
            raise ValueError(
                f"FaEmbeddingLayer : attendu {self.n_features} features, "
                f"reçu {n_feat}"
            )

        # Construire le tableau d'embeddings
        embedded = np.zeros((batch_size, self.output_dim), dtype=np.float64)
        for b in range(batch_size):
            for f in range(n_feat):
                vec = self._embed_value(x[b, f])
                embedded[b, f * self.FA_EMBED_DIM:(f + 1) * self.FA_EMBED_DIM] = vec

        return embedded

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Passe arrière simplifiée.

        L'embedding Fâ est une transformation déterministe (lookup table),
        donc son gradient par rapport à l'entrée brute est approximé par
        la composante val_norm (dim 0 du vecteur Fâ) — la seule dimension
        continue en x.

        Retourne le gradient par rapport à x original (batch, n_features).
        """
        batch_size = grad_output.shape[0]
        grad_x = np.zeros((batch_size, self.n_features), dtype=np.float64)

        for f in range(self.n_features):
            # Gradient passe uniquement par la dimension val_norm (index 0)
            col = f * self.FA_EMBED_DIM  # index de val_norm dans l'embedding
            scale = 1.0 / (self.x_max - self.x_min + 1e-10)
            grad_x[:, f] = grad_output[:, col] * scale

        return grad_x

    @property
    def params(self) -> dict:
        """FaEmbeddingLayer n'a pas de paramètres entraînables (lookup table)."""
        return {}


# ─── T16 : FaHiddenLayer ──────────────────────────────────────────────────────

class FaHiddenLayer:
    """
    Couche cachée du FaNet — traitement au cœur de la logique Fâ.

    y = FaReLU(x @ W.T + b)

    Initialisation des poids W depuis la Matrice-256 au lieu de l'aléatoire
    pure (Xavier). Cela ancre le réseau dans la topologie du Fâ dès le départ.

    Exemple :
        layer = FaHiddenLayer(input_dim=12, hidden_dim=32)
        h = layer.forward(x_embedded)
        print(h.shape)  # (batch, 32)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "fa_relu",
        seed: int = 42,
        fa_tensor: Optional[np.ndarray] = None,
        learnable_sigma: bool = False,
    ):
        """
        Args:
            input_dim:       Dimension d'entrée (= n_features × 4 pour FaEmbedding)
            hidden_dim:      Nombre de neurones cachés
            activation:      Nom de l'activation ('fa_relu', 'fa_sigma', 'fa_dual')
            seed:            Graine pour la reproductibilité
            fa_tensor:       Tenseur Matrice-256 (16,16,4) pour l'initialisation
            learnable_sigma: Si True et activation='fa_sigma', utilise
                             LearnableFaSigma (poids de polarité entraînable) [v0.2]
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed

        # Poids et biais — initialisés depuis la Matrice-256
        self.W = _fa_weight_init((hidden_dim, input_dim), fa_tensor, seed)
        self.b = np.zeros(hidden_dim, dtype=np.float64)

        # Activation Fa — v0.2 : support de LearnableFaSigma
        self._learnable_sigma: Optional[LearnableFaSigma] = None
        act: object
        if learnable_sigma and activation == "fa_sigma":
            init_w = load_polarity_weights()
            lfs = LearnableFaSigma(init_weights=init_w)
            self._learnable_sigma = lfs
            act = lfs
        else:
            act = FaActivation.get(activation)
        # Union pour que Pyre2 connaisse les méthodes forward/backward
        self._act: Union[FaActivation, LearnableFaSigma] = act  # type: ignore[assignment]

        # Cache pour la rétropropagation
        self._z: Optional[np.ndarray] = None
        self._a: Optional[np.ndarray] = None
        self._x_in: Optional[np.ndarray] = None

        # Gradients accumulés
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passe avant : transforme l'embedding en représentation cachée.

        z = x @ W.T + b   (combinaison linéaire)
        a = Activation(z) (activation Fâ)

        Args:
            x: np.ndarray de forme (batch_size, input_dim)

        Returns:
            np.ndarray de forme (batch_size, hidden_dim)
        """
        x = np.asarray(x, dtype=np.float64)
        self._x_in = x
        self._z = x @ self.W.T + self.b
        self._a = self._act.forward(self._z)
        return self._a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Passe arrière : calcule les gradients.

        Pour LearnableFaSigma : backward() accumule aussi d_raw.
        Pour les activations statiques : backward retourne f'(z).

        Args:
            grad_output: Gradient venant de la couche suivante (batch, hidden_dim)

        Returns:
            Gradient par rapport à l'entrée x (batch, input_dim)
        """
        assert self._z is not None and self._x_in is not None, \
            "backward() appelé sans forward() préalable"

        # Gradient à travers l'activation
        dz = grad_output * self._act.backward(self._z)

        # Gradients des paramètres
        self.dW = dz.T @ self._x_in
        self.db = dz.sum(axis=0)

        return dz @ self.W

    def update(self, lr: float = 0.01) -> None:
        """Mise à jour des poids. Inclut les poids de polarité si LearnableFaSigma."""
        self.W -= lr * self.dW
        self.b -= lr * self.db
        lfs = self._learnable_sigma
        if lfs is not None:
            lfs.update(lr)

    def zero_grad(self) -> None:
        """Remet les gradients à zéro (W, b, et poids de polarité si appris)."""
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        lfs = self._learnable_sigma
        if lfs is not None:
            lfs.zero_grad()

    @property
    def n_learnable_params(self) -> int:
        """Nombre total de paramètres entraînables (W + b + polarités si actif)."""
        base = self.W.size + self.b.size
        lfs = self._learnable_sigma
        extra = lfs.n_params if lfs is not None else 0
        return base + extra

    @property
    def params(self) -> dict:
        p: dict = {"W": self.W, "b": self.b}
        lfs = self._learnable_sigma
        if lfs is not None:
            p["polarity_raw"] = lfs.get_state()
        return p


# ─── T17 : FaDecoderLayer ─────────────────────────────────────────────────────

class FaDecoderLayer:
    """
    Couche de sortie : décode la décision finale avec double ancrage.

    Produit deux sorties simultanées :
      1. Valeur numérique  (prédiction classique)
      2. Interprétation Fâ (quel du correspond à cette prédiction ?)

    L'activation FaDual (Meji) est utilisée : la décision est "redoublée"
    comme un signe Meji, garantissant une sortie dans ]0, 1[.

    Exemple :
        layer = FaDecoderLayer(hidden_dim=32, n_outputs=1)
        pred, du = layer.forward_with_interpretation(h)
        print(pred, du["nom_fon"])
    """

    def __init__(
        self,
        hidden_dim: int,
        n_outputs: int = 1,
        seed: int = 42,
    ):
        """
        Args:
            hidden_dim: Dimension de la couche cachée (entrée du décodeur)
            n_outputs:  Nombre de sorties (1 = régression, N = classification N-classes)
            seed:       Graine
        """
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs

        # Poids et biais
        self.W = _xavier_init((n_outputs, hidden_dim), seed)
        self.b = np.zeros(n_outputs, dtype=np.float64)

        # Activation : FaDual (Meji — confirmation double)
        self._act = FaActivation.get("fa_dual")

        # Cache backprop
        self._z: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passe avant : produit la sortie numérique.

        z = x @ W.T + b
        out = FaDual(z)  → dans ]0, 1[

        Args:
            x: (batch, hidden_dim)

        Returns:
            (batch, n_outputs), valeurs dans ]0, 1[
        """
        x = np.asarray(x, dtype=np.float64)
        self._x = x
        self._z = x @ self.W.T + self.b
        return self._act.forward(self._z)

    def forward_with_interpretation(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Passe avant avec interprétation Fâ de chaque sortie.

        Retourne la valeur numérique + le du correspondant pour chaque
        élément du batch.

        Args:
            x: (batch, hidden_dim)

        Returns:
            (predictions: ndarray (batch, n_outputs),
             interpretations: list[dict] — un du par élément du batch)
        """
        predictions = self.forward(x)

        interpretations = []
        for b in range(predictions.shape[0]):
            # Prendre la première sortie (ou moyenne si multi-sorties)
            val = float(predictions[b, 0]) if predictions.ndim > 1 \
                  else float(predictions[b])
            du = value_to_du(val)
            interpretations.append(du)

        return predictions, interpretations

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Passe arrière.

        Args:
            grad_output: Gradient de la loss (batch, n_outputs)

        Returns:
            Gradient vers la couche cachée (batch, hidden_dim)
        """
        assert self._z is not None and self._x is not None

        dz = grad_output * self._act.backward(self._z)
        self.dW = dz.T @ self._x
        self.db = dz.sum(axis=0)
        return dz @ self.W

    def update(self, lr: float = 0.01) -> None:
        """Mise à jour des poids."""
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def zero_grad(self) -> None:
        """Remet les gradients à zéro."""
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    @property
    def params(self) -> dict:
        return {"W": self.W, "b": self.b}
