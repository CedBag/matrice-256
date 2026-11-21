# Copyright (c) 2026 Cedoque BAGBONON, Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_layers : Couches FaNet en PyTorch (v4.1)
========================================================
Architecture PyTorch corrigée :
  TorchFaEmbedding -> TorchFaHidden -> TorchFaDecoder

Correctifs scientifiques v4.1 :
  1. kpoli_table extraite depuis la vraie diagonale Meji (indices 0, 17, 34, ..., 255).
  2. Support des modes de polarité ("fa", "constant", "permuted") pour l'étude d'ablation.
  3. Normalisation lisse par sigmoïde (buffers input_mean / input_std) évitant la perte
     des valeurs négatives sur des données standardisées (StandardScaler).
  4. Projections de requête par variable (PerFeatureLinear) pour éviter qu'une même
     projection linéaire traite de manière identique des capteurs hétérogènes.
  5. Suppression de la modification en place sur embedded[:, :, 0].
  6. Alignement strict entraînement / inférence dans TorchFaDecoder (classification et régression).

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
    Initialisation des poids depuis la Matrice-256 (version Tensor) ou Xavier.
    """
    rng = np.random.default_rng(seed)
    if fa_tensor is not None:
        flat = fa_tensor.flatten().astype(np.float32)
        n_needed = shape[0] * shape[1]

        if n_needed <= len(flat):
            base = flat[:n_needed]
        else:
            repeats = (n_needed // len(flat)) + 1
            base = np.tile(flat, repeats)[:n_needed]

        W = base.reshape(shape)
        W = (W - W.mean()) / (W.std() + 1e-8)
        # Mise à l'échelle type Xavier pour stabilité des gradients en début d'entraînement
        scale = np.sqrt(2.0 / (shape[0] + shape[1]))
        W = W * scale + rng.normal(0.0, 0.01, shape)
        return torch.tensor(W, dtype=torch.float32)

    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    W = rng.uniform(-limit, limit, shape)
    return torch.tensor(W, dtype=torch.float32)


class PerFeatureLinear(nn.Module):
    """
    Projection affine indépendante par variable d'entrée : (batch, n_features, 1) -> (batch, n_features, out_dim).

    Chaque capteur ou biomarqueur apprend ses propres poids de requête vers les 16 Kpoli,
    tout en exposant .weight et .bias comme nn.Linear.
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
    Couche d'embedding Fâ combinatoire native (16x16 -> 256) v4.1.

    Génère deux distributions d'attention sur les 16 Kpoli (gauche et droite)
    et forme la distribution conjointe sur les 256 Du par produit de Kronecker.
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

        if per_feature_queries:
            self.query_gauche: nn.Module = PerFeatureLinear(n_features, self.FA_EMBED_DIM, seed=seed)
            self.query_droit: nn.Module = PerFeatureLinear(n_features, self.FA_EMBED_DIM, seed=seed + 1000)
        else:
            self.query_gauche = nn.Linear(1, self.FA_EMBED_DIM, dtype=torch.float32)
            self.query_droit = nn.Linear(1, self.FA_EMBED_DIM, dtype=torch.float32)

        # Buffers de normalisation lisse (ajustés sur X_train uniquement dans fit())
        default_center = 0.5 * (self.x_min + self.x_max)
        default_scale = max((self.x_max - self.x_min) / 4.0, 1e-6)
        self.register_buffer("input_mean", torch.full((n_features,), default_center, dtype=torch.float32))
        self.register_buffer("input_std", torch.full((n_features,), default_scale, dtype=torch.float32))
        self.register_buffer("norm_fitted", torch.tensor(False, dtype=torch.bool))

        kpoli_table, ref_table, kpoli_polarities = self._build_reference_tables()
        self.register_buffer("kpoli_table", kpoli_table)            # (16, 4) Keys
        self.register_buffer("ref_table", ref_table)                # (256, 4) Values
        self.register_buffer("kpoli_polarities", kpoli_polarities)  # (16,) Polarités Kpoli

    def set_normalization(self, X: np.ndarray) -> None:
        """
        Ajuste la moyenne et l'écart-type par variable strictement sur les données
        d'entraînement (zéro fuite de données).
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

        # Table des 256 Du : (256, 4) = [val_norm, polarite_num, intensite, est_meji]
        table = torch.zeros(256, 4, dtype=torch.float32)
        for i in range(16):
            for j in range(16):
                du_id = i * 16 + j
                du = kb[du_id]
                val_norm = du_id / 255.0
                # Cohérence algébrique exacte : pol(i, j) = (pol(i) + pol(j)) / 2
                pol_val = float(0.5 * (k_pols[i] + k_pols[j]))
                intensite = float(du.get("intensite", 0.5))
                est_meji = 1.0 if (i == j) else 0.0

                table[du_id, 0] = val_norm
                table[du_id, 1] = pol_val
                table[du_id, 2] = intensite
                table[du_id, 3] = est_meji

        # Vraie diagonale des 16 Meji : indices 0, 17, 34, ..., 255
        meji_indices = torch.tensor([k * 16 + k for k in range(16)], dtype=torch.long)
        kpoli_table = table[meji_indices].clone()
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

        # 1. Requêtes Kpoli gauche et droit
        Q_gauche = self.query_gauche(x_in)  # (b, n, 4)
        Q_droit = self.query_droit(x_in)    # (b, n, 4)

        # 2. Clés K_16 (16 Meji sur la diagonale) et Valeurs V (256 Du)
        K_16: torch.Tensor = getattr(self, "kpoli_table").to(dtype=x.dtype, device=x.device)
        V: torch.Tensor = getattr(self, "ref_table").to(dtype=x.dtype, device=x.device)

        # 3. Attention croisée sur les 16 Kpoli
        scale = float(self.FA_EMBED_DIM) ** 0.5
        scores_gauche = torch.matmul(Q_gauche, K_16.transpose(0, 1)) / scale
        scores_droit = torch.matmul(Q_droit, K_16.transpose(0, 1)) / scale

        attn_gauche = torch.softmax(scores_gauche, dim=-1)  # (b, n, 16)
        attn_droit = torch.softmax(scores_droit, dim=-1)    # (b, n, 16)

        # 4. Produit de Kronecker (16x16 -> 256)
        attn_matrix = attn_gauche.unsqueeze(-1) * attn_droit.unsqueeze(-2)
        attn_weights = attn_matrix.reshape(batch_size, self.n_features, 256)

        # 5. Agrégation symbolique + conservation sans écriture in-place
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
    Couche cachée de FaNet (v4.1).

    y = Activation(x @ W.T + b)
    Supporte l'initialisation Matrice-256 (init_mode="fa") ou Xavier (init_mode="xavier").
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
    Couche de sortie de FaNet (v4.1) avec cohérence stricte entraînement / inférence.

    Modes :
      - task="classification", output_mode="standard" : forward() applique torch.sigmoid(z),
        exactement cohérent avec BCEWithLogitsLoss sur forward_logits().
      - task="classification", output_mode="fa_dual"  : forward() applique TorchFaDual(z),
        entraîné directement avec BCELoss (ablation).
      - task="regression"                             : forward() retourne la projection
        linéaire non bornée z (pas de saturation artificielle sur [0.047, 0.953]).
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
