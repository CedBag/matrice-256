# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.network — FaNet v0.1
================================
Le premier réseau de neurones entièrement béninois.

FaNet assemble les 3 couches Fâ en un pipeline d'apprentissage automatique complet :

  Données brutes
      ↓ FaEmbeddingLayer    (traduit en espace Fâ)
      ↓ FaHiddenLayer       (traitement — logique Fâ)
      ↓ FaDecoderLayer      (prédiction + interprétation du)
  Sortie numérique + du Fâ

Le réseau dispose de :
  - forward()  : passe avant
  - train()    : apprentissage supervisé (SGD)
  - predict()  : inférence avec interprétation Fâ
  - evaluate() : métriques de performance
  - summary()  : résumé de l'architecture

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Callable

from bokono.ml.layers import FaEmbeddingLayer, FaHiddenLayer, FaDecoderLayer
from bokono.core.fa_matrix import FaMatrix


# ─── Fonctions de perte ───────────────────────────────────────────────────────

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error — perte pour régression."""
    return float(np.mean((y_pred - y_true) ** 2))


def mse_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient de MSE par rapport à y_pred."""
    return 2.0 * (y_pred - y_true) / y_pred.size


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary Cross-Entropy — perte pour classification binaire."""
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def bce_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient de BCE par rapport à y_pred."""
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_pred.size)


LOSS_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "mse": (mse_loss, mse_gradient),
    "bce": (binary_cross_entropy, bce_gradient),
}


# ─── T18 : FaNet ──────────────────────────────────────────────────────────────

class FaNet:
    """
    FaNet v0.1 — Réseau de neurones béninois.

    Architecture :
      Input(n_features) → Embedding(n_features×4) → Hidden(hidden_dim) → Output(n_outputs)

    Initialisation unique : les poids de la couche cachée sont initialisés
    depuis la Matrice-256 (tenseur 16×16×4), ancrant le réseau dans la
    topologie du Fâ dès le départ.

    Exemple :
        net = FaNet(n_features=4, hidden_dim=32, n_outputs=1)
        net.summary()

        # Entraînement
        history = net.train(X_train, y_train, epochs=100, lr=0.01)

        # Prédiction avec interprétation Fâ
        pred, du = net.predict(X_test)
        print(f"Prédiction: {pred}, Signe Fâ: {du['nom_fon']}")
    """

    VERSION = "0.2.0"

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        n_outputs: int = 1,
        activation: str = "fa_sigma",
        loss: str = "mse",
        x_min: float = 0.0,
        x_max: float = 1.0,
        learnable_sigma: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            n_features:      Nombre de features d'entrée
            hidden_dim:      Nombre de neurones dans la couche cachée
            n_outputs:       Nombre de sorties
            activation:      Activation de la couche cachée ('fa_relu', 'fa_sigma', 'fa_dual')
            loss:            Fonction de perte ('mse' ou 'bce')
            x_min:           Valeur minimale des features (pour l'embedding)
            x_max:           Valeur maximale des features (pour l'embedding)
            learnable_sigma: [v0.2] Si True et activation='fa_sigma', les 16 poids
                             de polarité Kpoli deviennent des paramètres entraînable.
            seed:            Graine pour la reproductibilité
        """
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.activation = activation
        self.loss_name = loss
        self.learnable_sigma = learnable_sigma
        self.seed = seed

        # Charger le tenseur Matrice-256 pour l'initialisation des poids
        self._fa_matrix = FaMatrix()
        fa_tensor = self._fa_matrix.to_weight_tensor()

        # Construction des couches
        embedding_dim = n_features * FaEmbeddingLayer.FA_EMBED_DIM

        self.embedding = FaEmbeddingLayer(n_features, x_min, x_max)
        self.hidden    = FaHiddenLayer(
            embedding_dim, hidden_dim, activation, seed, fa_tensor,
            learnable_sigma=learnable_sigma,
        )
        self.decoder   = FaDecoderLayer(hidden_dim, n_outputs, seed)

        # Fonctions de perte
        if loss not in LOSS_REGISTRY:
            raise ValueError(f"Perte inconnue : '{loss}'. Disponibles : {list(LOSS_REGISTRY)}")
        self._loss_fn, self._loss_grad = LOSS_REGISTRY[loss]

        # Historique d'entraînement
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    # ─── Passe avant ──────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passe avant complète : x → prédiction.

        Args:
            x: np.ndarray de forme (batch_size, n_features)

        Returns:
            np.ndarray de forme (batch_size, n_outputs)
        """
        emb  = self.embedding.forward(x)
        hidden = self.hidden.forward(emb)
        out  = self.decoder.forward(hidden)
        return out

    def backward(self, grad_loss: np.ndarray) -> None:
        """
        Rétropropagation complète.

        Args:
            grad_loss: Gradient de la perte par rapport à la sortie (batch, n_outputs)
        """
        # Decoder → Hidden → Embedding
        grad = self.decoder.backward(grad_loss)
        grad = self.hidden.backward(grad)
        # L'embedding n'a pas de poids entraînables (lookup table)

    def zero_grad(self) -> None:
        """Remet tous les gradients à zéro."""
        self.hidden.zero_grad()
        self.decoder.zero_grad()

    def update(self, lr: float) -> None:
        """Met à jour tous les poids."""
        self.hidden.update(lr)
        self.decoder.update(lr)

    # ─── Entraînement ─────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        verbose_every: int = 10,
    ) -> dict[str, list[float]]:
        """
        Entraînement supervisé par mini-batch SGD.

        Args:
            X:            Données d'entraînement (n_samples, n_features)
            y:            Cibles (n_samples,) ou (n_samples, n_outputs)
            epochs:       Nombre d'époques
            lr:           Taux d'apprentissage
            batch_size:   Taille des mini-batches
            X_val:        Données de validation (optionnel)
            y_val:        Cibles de validation (optionnel)
            verbose:      Afficher la progression
            verbose_every: Afficher tous les N epochs

        Returns:
            Historique : {"loss": [...], "val_loss": [...]}
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.seed)

        self.history = {"loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            # Mélanger les données
            idx = rng.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch SGD
            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                # Forward
                pred = self.forward(X_batch)
                loss = self._loss_fn(pred, y_batch)
                epoch_loss += loss
                n_batches += 1

                # Backward
                self.zero_grad()
                grad = self._loss_grad(pred, y_batch)
                self.backward(grad)
                self.update(lr)

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["loss"].append(avg_loss)

            # Validation
            val_loss = 0.0
            if X_val is not None and y_val is not None:
                y_val_arr = np.asarray(y_val, dtype=np.float64)
                if y_val_arr.ndim == 1:
                    y_val_arr = y_val_arr.reshape(-1, 1)
                val_pred = self.forward(np.asarray(X_val, dtype=np.float64))
                val_loss = self._loss_fn(val_pred, y_val_arr)
                self.history["val_loss"].append(val_loss)

            if verbose and (epoch % verbose_every == 0 or epoch == 1):
                val_str = f" | val_loss={val_loss:.5f}" if X_val is not None else ""
                print(f"  Époque {epoch:4d}/{epochs} | loss={avg_loss:.5f}{val_str}")

        return self.history

    # ─── Inférence ────────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Prédiction avec interprétation Fâ.

        Pour chaque échantillon, retourne :
          - La prédiction numérique
          - Le du Fâ correspondant (nom, polarité, interprétation...)

        Args:
            x: (batch_size, n_features) ou (n_features,) pour 1 échantillon

        Returns:
            (predictions: ndarray, interpretations: list[dict])

        Exemple :
            pred, dus = net.predict(X_test)
            for i, du in enumerate(dus):
                print(f"Sample {i}: {pred[i]} → {du['nom_fon']} ({du['polarite']})")
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        emb = self.embedding.forward(x)
        hidden = self.hidden.forward(emb)
        preds, dus = self.decoder.forward_with_interpretation(hidden)
        return preds, dus

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Prédiction numérique seule (sans interprétation Fâ)."""
        return self.forward(np.atleast_2d(np.asarray(x, dtype=np.float64)))

    # ─── Évaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """
        Évalue le réseau sur un jeu de données.

        Returns:
            Dict avec les métriques : loss, mae, rmse, accuracy (si binaire)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        pred = self.forward(X)
        loss = self._loss_fn(pred, y)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

        metrics: dict[str, float] = {
            "loss": loss,
            "mae": mae,
            "rmse": rmse,
        }

        # Accuracy pour classification binaire
        if self.n_outputs == 1:
            y_class = (pred >= 0.5).astype(int)
            y_true_class = (y >= 0.5).astype(int)
            metrics["accuracy"] = float(np.mean(y_class == y_true_class))

        return metrics

    # ─── Persistance ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Sauvegarde les poids du réseau en format NumPy (.npz).
        [v0.2] Inclut les poids de polarité si LearnableFaSigma est actif.
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        save_dict: dict = {
            "hidden_W":  self.hidden.W,
            "hidden_b":  self.hidden.b,
            "decoder_W": self.decoder.W,
            "decoder_b": self.decoder.b,
            "n_features":     np.array(self.n_features),
            "hidden_dim":     np.array(self.hidden_dim),
            "n_outputs":      np.array(self.n_outputs),
            "version":        np.array([self.VERSION]),
            "learnable_sigma": np.array(self.learnable_sigma),
        }

        # Sauvegarder les poids de polarité appris si présents
        lfs = self.hidden._learnable_sigma
        if lfs is not None:
            save_dict["polarity_raw"] = lfs.get_state()

        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str, **init_kwargs) -> "FaNet":
        """
        Charge un FaNet depuis un fichier .npz.
        [v0.2] Restaure les poids de polarité si présents.
        """
        # SÉCURITÉ : allow_pickle=False empêche l'exécution de code arbitraire
        # via pickle deserialization dans des fichiers .npz piégés.
        data = np.load(path, allow_pickle=False)
        n_features = int(data["n_features"])
        hidden_dim = int(data["hidden_dim"])
        n_outputs  = int(data["n_outputs"])
        learnable  = bool(data["learnable_sigma"]) if "learnable_sigma" in data else False

        net = cls(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_outputs=n_outputs,
            learnable_sigma=learnable,
            **init_kwargs,
        )
        net.hidden.W  = data["hidden_W"]
        net.hidden.b  = data["hidden_b"]
        net.decoder.W = data["decoder_W"]
        net.decoder.b = data["decoder_b"]

        # Restaurer les poids de polarité si présents
        if "polarity_raw" in data and net.hidden._learnable_sigma is not None:
            net.hidden._learnable_sigma.set_state(data["polarity_raw"])

        return net

    # ─── Affichage ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Affiche le résumé de l'architecture FaNet v0.2."""
        emb_dim = self.n_features * FaEmbeddingLayer.FA_EMBED_DIM
        hidden_params = self.hidden_dim * emb_dim + self.hidden_dim
        polarity_params = 16 if (self.learnable_sigma and self.activation == "fa_sigma") else 0
        decoder_params = self.n_outputs * self.hidden_dim + self.n_outputs
        total_params = hidden_params + polarity_params + decoder_params

        print("=" * 65)
        print(f"  FaNet v{self.VERSION} — Réseau de neurones béninois")
        print("=" * 65)
        print(f"  Perte               : {self.loss_name.upper()}")
        print(f"  Activation cachée   : {self.activation}")
        learnable_str = " [ENTRAÎNABLE v0.2]" if polarity_params > 0 else " [statique]"
        print(f"  Poids de polarité   : {polarity_params} params{learnable_str}")
        print()
        print(f"  {'Couche':<25} {'Forme sortie':<20} {'Paramètres':>12}")
        print("  " + "-" * 60)
        print(f"  {'FaEmbedding (Fâ)':<25} {'(' + str(emb_dim) + ',)':<20} {'0':>12}")
        print(f"  {'FaHidden (Fâ)':<25} {'(' + str(self.hidden_dim) + ',)':<20} {hidden_params + polarity_params:>12,}")
        print(f"  {'FaDecoder (Meji)':<25} {'(' + str(self.n_outputs) + ',)':<20} {decoder_params:>12,}")
        print("  " + "-" * 60)
        print(f"  {'TOTAL':<25} {'':<20} {total_params:>12,}")
        print("=" * 65)
        print(f"  Initialisation      : Matrice-256 (tenseur 16×16×4)")
        if polarity_params:
            print(f"  Adaptation          : Poids Kpoli affinés par les données")
        print(f"  Interprétation      : chaque prédiction → du Fâ")
        print("=" * 65)
