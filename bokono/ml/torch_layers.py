# Copyright (c) 2026 Cedoque BAGBONON, Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_layers : Couches FaNet en PyTorch (v4.2)
========================================================
Architecture PyTorch v4.2 :
  TorchFaEmbedding (Hypercube binaire 4 bits + requêtes cross-features)
  -> TorchFaHidden (FaReLU) -> TorchFaHidden (FaSigma + résiduel)
  -> TorchFaDecoder (aligné train/inférence)

Améliorations scientifiques v4.2 :
  1. Clés K_16 définies sur l'hypercube binaire exact 4 bits {-1, +1}^4 du Fâ
     (GBE = [+1,+1,+1,+1], ..., IKA = [-1,-1,-1,-1]), où le produit scalaire
     encode directement l'alignement de Hamming avec les 4 marques du Kpoli.
  2. Requêtes combinées locale + globale cross-features dans TorchFaEmbedding
     afin que chaque Kpoli synthétise l'interaction entre toutes les variables.
  3. Initialisation plein rang dans _fa_weight_tensor : évite la colinéarité
     de pas 4 observée quand input_dim = 4 * n_features.

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from bokono.core.fa_encoder import value_to_algorithmic_state, _get_kb
from bokono.ml.activations import load_polarity_weights
from bokono.ml.torch_activations import (
    TorchFaDual,
    get_torch_activation,
)


def _fa_weight_tensor(
    shape: tuple[int, int],
    fa_tensor: Optional[np.ndarray] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Initialisation plein rang guidée par la Matrice-256 (ou Xavier standard).

    En v4.2, au lieu d'aplatir fa_tensor directement en pas de 4 (ce qui alignait
    les colonnes sur la même périodicité que 4 * n_features), on combine une base
    Xavier orthogonale (plein rang garanti) avec une modulation douce issue de
    la Matrice-256.
    """
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    base_xavier = rng.uniform(-limit, limit, shape).astype(np.float32)

    if fa_tensor is not None:
        flat = fa_tensor.flatten().astype(np.float32)
        n_needed = shape[0] * shape[1]
        # Échantillonnage à pas premier (17, copremier avec 4 et 256) pour éviter toute colinéarité
        idx = (np.arange(n_needed) * 17) % len(flat)
        prior = flat[idx].reshape(shape)
        prior = (prior - prior.mean()) / (prior.std() + 1e-8)
        prior_scaled = prior * (limit * 0.35)
        W = 0.85 * base_xavier + 0.15 * prior_scaled
        return torch.tensor(W, dtype=torch.float32)

    return torch.tensor(base_xavier, dtype=torch.float32)


def _build_hypercube_keys() -> torch.Tensor:
    """
    Construit les 16 vecteurs clés exacts de l'hypercube binaire 4 bits du Fâ :
      bit '0' (ouvert, GBE) -> +1.0
      bit '1' (fermé,  IKA) -> -1.0
    Forme : (16, 4), norme euclidienne constante = 2.0.
    """
    keys = torch.zeros(16, 4, dtype=torch.float32)
    for k in range(16):
        bits = f"{k:04b}"
        for b_idx, ch in enumerate(bits):
            keys[k, b_idx] = 1.0 if ch == "0" else -1.0
    return keys


class PerFeatureLinear(nn.Module):
    """
    Projection affine indépendante par variable : (batch, n_features, 1) -> (batch, n_features, out_dim).
    """

    def __init__(self, n_features: int, out_dim: int, seed: int = 42):
        super().__init__()
        self.n_features = n_features
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (1 + out_dim))
        w_init = rng.uniform(-limit, limit, (n_features, out_dim)).astype(np.float32)
        self.weight = nn.Parameter(torch.tensor(w_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n_features, out_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TorchFaEmbedding(nn.Module):
    """
    Couche d'embedding Fâ combinatoire native sur hypercube 4 bits (v4.2).

    Clés K_16 : hypercube binaire {-1, +1}^4 (16 Kpoli).
    Requêtes : combinaison d'une requête locale par variable (query_gauche / query_droit)
               et d'une requête contextuelle globale cross-features (global_gauche / global_droit).
    Produit de Kronecker : 16 x 16 -> 256 Du.
    """

    FA_EMBED_DIM: int = 4

    def __init__(
        self,
        n_features: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        polarity_mode: str = "fa",
        seed: int = 42,
        per_feature_queries: bool = True,
        auto_normalize: bool = True,
        cross_feature: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.output_dim = n_features * self.FA_EMBED_DIM
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.polarity_mode = polarity_mode
        self.seed = seed
        self.per_feature_queries = per_feature_queries
        self.auto_normalize = auto_normalize
        self.cross_feature = cross_feature

        if per_feature_queries:
            self.query_gauche: nn.Module = PerFeatureLinear(n_features, self.FA_EMBED_DIM, seed=seed)
            self.query_droit: nn.Module = PerFeatureLinear(n_features, self.FA_EMBED_DIM, seed=seed + 1000)
        else:
            self.query_gauche = nn.Linear(1, self.FA_EMBED_DIM, dtype=torch.float32)
            self.query_droit = nn.Linear(1, self.FA_EMBED_DIM, dtype=torch.float32)

        # Contexte global cross-features : chaque variable observe l'état global du vecteur d'entrée
        if cross_feature:
            torch.manual_seed(seed + 2000)
            self.global_gauche: Optional[nn.Linear] = nn.Linear(n_features, self.FA_EMBED_DIM, dtype=torch.float32)
            self.global_droit: Optional[nn.Linear] = nn.Linear(n_features, self.FA_EMBED_DIM, dtype=torch.float32)
        else:
            self.global_gauche = None
            self.global_droit = None

        # Buffers de normalisation lisse (ajustés sur X_train uniquement dans fit())
        default_center = 0.5 * (self.x_min + self.x_max)
        default_scale = max((self.x_max - self.x_min) / 4.0, 1e-6)
        self.register_buffer("input_mean", torch.full((n_features,), default_center, dtype=torch.float32))
        self.register_buffer("input_std", torch.full((n_features,), default_scale, dtype=torch.float32))
        self.register_buffer("norm_fitted", torch.tensor(False, dtype=torch.bool))

        kpoli_table, ref_table, kpoli_polarities = self._build_reference_tables()
        self.register_buffer("kpoli_table", kpoli_table)            # (16, 4) Hypercube {-1, +1}^4
        self.register_buffer("ref_table", ref_table)                # (256, 4) Valeurs symboliques
        self.register_buffer("kpoli_polarities", kpoli_polarities)  # (16,) Polarités Kpoli

    def set_normalization(self, X: np.ndarray) -> None:
        """
        Ajuste la moyenne et l'écart-type par variable strictement sur X_train.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 2 and X_arr.shape[1] != self.n_features:
            X_arr = X_arr.reshape(-1, self.n_features)
        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        device = getattr(self, "input_mean").device
        self.input_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=device))
        self.input_std.copy_(torch.tensor(std, dtype=torch.float32, device=device))
        self.norm_fitted.fill_(True)

    def _build_reference_tables(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kb = _get_kb()
        k_pols = load_polarity_weights(polarity_mode=self.polarity_mode, seed=self.seed)

        # Clés K_16 sur l'hypercube binaire exact 4 bits {-1, +1}^4
        # En mode "fa", k_pols[i] = (4 + sum(K_16[i])) / 8 est géométriquement aligné sur l'hypercube.
        # En mode "permuted", k_pols est mélangé sur l'hypercube fixe, brisant cet alignement géométrique.
        kpoli_table = _build_hypercube_keys()

        # Table de valeurs des 256 Du : (256, 4) = [val_norm, polarite_num, intensite, est_meji]
        table = torch.zeros(256, 4, dtype=torch.float32)
        for i in range(16):
            for j in range(16):
                du_id = i * 16 + j
                du = kb[du_id]
                table[du_id, 0] = du_id / 255.0
                table[du_id, 1] = float(0.5 * (k_pols[i] + k_pols[j]))
                table[du_id, 2] = float(du.get("intensite", 0.5))
                table[du_id, 3] = 1.0 if (i == j) else 0.0

        kpoli_polarities = torch.tensor(k_pols, dtype=torch.float32)
        return kpoli_table, table, kpoli_polarities

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_dual: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]

        if self.auto_normalize:
            mean = getattr(self, "input_mean").to(dtype=x.dtype, device=x.device)
            std = getattr(self, "input_std").to(dtype=x.dtype, device=x.device)
            x_norm = torch.sigmoid((x - mean) / std)
        else:
            x_range = self.x_max - self.x_min
            x_norm = torch.clamp((x - self.x_min) / (x_range + 1e-10), 0.0, 1.0)

        x_in = x_norm.unsqueeze(-1)  # (batch, n_features, 1)

        # 1. Requêtes locales + contexte global cross-features
        Q_gauche = self.query_gauche(x_in)  # (b, n, 4)
        Q_droit = self.query_droit(x_in)    # (b, n, 4)

        if self.global_gauche is not None and self.global_droit is not None:
            x_centered = 2.0 * x_norm - 1.0  # [-1, +1]
            Q_gauche = Q_gauche + self.global_gauche(x_centered).unsqueeze(1)
            Q_droit = Q_droit + self.global_droit(x_centered).unsqueeze(1)

        # 2. Clés hypercube K_16 (16, 4) et Valeurs V (256, 4)
        K_16: torch.Tensor = getattr(self, "kpoli_table").to(dtype=x.dtype, device=x.device)
        V: torch.Tensor = getattr(self, "ref_table").to(dtype=x.dtype, device=x.device)

        # 3. Attention croisée sur l'hypercube 4 bits
        scale = float(self.FA_EMBED_DIM) ** 0.5
        scores_gauche = torch.matmul(Q_gauche, K_16.transpose(0, 1)) / scale
        scores_droit = torch.matmul(Q_droit, K_16.transpose(0, 1)) / scale

        attn_gauche = torch.softmax(scores_gauche, dim=-1)  # (b, n, 16)
        attn_droit = torch.softmax(scores_droit, dim=-1)    # (b, n, 16)

        # 4. Produit de Kronecker (16x16 -> 256)
        attn_matrix = attn_gauche.unsqueeze(-1) * attn_droit.unsqueeze(-2)
        attn_weights = attn_matrix.reshape(batch_size, self.n_features, 256)

        # 5. Agrégation symbolique sans écriture in-place
        embedded_raw = torch.matmul(attn_weights, V)  # (b, n, 4)
        embedded = torch.cat([x_norm.unsqueeze(-1), embedded_raw[:, :, 1:]], dim=-1)

        out = embedded.reshape(batch_size, -1)

        if return_dual:
            return out, attn_weights, attn_gauche, attn_droit
        if return_attention:
            return out, attn_weights
        return out


class TorchFaHidden(nn.Module):
    """
    Couche cachée de FaNet (v4.2).

    y = Activation(x @ W.T + b)
    Supporte l'initialisation plein rang Matrice-256 (fa_tensor) ou Xavier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "fa_sigma",
        seed: int = 42,
        fa_tensor: Optional[np.ndarray] = None,
        polarity_mode: str = "fa",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        W = _fa_weight_tensor((hidden_dim, input_dim), fa_tensor, seed)
        self.weight = nn.Parameter(W)
        self.bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        self.activation = get_torch_activation(
            activation,
            polarity_mode=polarity_mode,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.linear(x.to(torch.float32), self.weight, self.bias)
        return self.activation(z)


class TorchFaDecoder(nn.Module):
    """
    Couche de sortie de FaNet (v4.2) avec cohérence stricte entraînement / inférence.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_outputs: int = 1,
        seed: int = 42,
        task: str = "classification",
        output_mode: str = "standard",
        polarity_mode: str = "fa",
    ):
        super().__init__()
        self.task = task
        self.output_mode = output_mode

        limit = np.sqrt(6.0 / (n_outputs + hidden_dim))
        rng = np.random.default_rng(seed)
        W = rng.uniform(-limit, limit, (n_outputs, hidden_dim))

        self.weight = nn.Parameter(torch.tensor(W, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n_outputs, dtype=torch.float32))
        self.activation: Optional[nn.Module] = (
            TorchFaDual(polarity_mode=polarity_mode, seed=seed)
            if (task == "classification" and output_mode == "fa_dual")
            else None
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les logits bruts avant activation."""
        return torch.nn.functional.linear(x.to(torch.float32), self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_logits(x)
        if self.task == "regression":
            return z
        if self.output_mode == "fa_dual" and self.activation is not None:
            return self.activation(z)
        return torch.sigmoid(z)

    @torch.no_grad()
    def forward_with_interpretation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict]]:
        """Prédiction numérique + état symbolique Fâ pour chaque élément du batch."""
        preds = self.forward(x)
        interpretations = []
        for b in range(preds.shape[0]):
            raw_val = float(preds[b, 0]) if preds.ndim > 1 else float(preds[b])
            if self.task == "regression":
                mapped_val = float(1.0 / (1.0 + np.exp(-np.clip(raw_val, -20.0, 20.0))))
            else:
                mapped_val = raw_val
            du = value_to_algorithmic_state(mapped_val)
            interpretations.append(du)
        return preds, interpretations
