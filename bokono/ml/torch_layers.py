# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_layers — Couches FaNet en PyTorch (v0.3)
=========================================================
Port natif des 3 couches du réseau de neurones béninois.

Architecture PyTorch :
  TorchFaEmbedding → TorchFaHidden → TorchFaDecoder

Avantages de v0.3 sur v0.2 :
  - Autograd complet → zéro backprop manuel, zéro bug de gradient
  - FaEmbedding DIFFÉRENTIABLE → gradient complet sur toute l'architecture
  - GPU-ready → .to("cuda") natif, scalabilité industrielle
  - Composable → nn.Sequential, nn.ModuleList, etc.

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from bokono.ml.torch_activations import (
    FaReLU, TorchLearnableFaSigma, TorchFaDual,
    get_torch_activation,
)
from bokono.core.fa_encoder import value_to_algorithmic_state, _get_kb


# ─── Utilitaires internes ──────────────────────────────────────────────────────

def _fa_weight_tensor(
    shape: tuple[int, int],
    fa_tensor: Optional[np.ndarray] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Initialisation des poids depuis la Matrice-256 (version Tensor).

    Même logique que la version NumPy (_fa_weight_init) de v0.2,
    sauf que le résultat est directement un torch.Tensor.
    """
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

        rng = np.random.default_rng(seed)
        W += rng.normal(0, 0.01, shape)
        return torch.tensor(W, dtype=torch.float32)

    # Fallback : Xavier
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    rng = np.random.default_rng(seed)
    W = rng.uniform(-limit, limit, shape)
    return torch.tensor(W, dtype=torch.float32)


# ─── TorchFaEmbedding ─────────────────────────────────────────────────────────

class TorchFaEmbedding(nn.Module):
    """
    Couche d'embedding Fâ COMBINATOIRE NATIVE — v3.0

    Innovation majeure : L'architecture mime le vrai tirage binaire du Fâ (16x16).
    Le réseau apprend 2 requêtes (Query Gauche et Query Droit) pour cibler les 
    16 Kpoli majeurs (Meji), et génère la matrice d'attention sur les 256 signes 
    par produit tensoriel (Kronecker).
    
    Plus de "Soft-Lookup" plat, mais un raisonnement généalogique croisé.
    """

    FA_EMBED_DIM: int = 4

    def __init__(self, n_features: int, x_min: float = 0.0, x_max: float = 1.0):
        super().__init__()
        self.n_features = n_features
        self.output_dim = n_features * self.FA_EMBED_DIM
        self.x_min = x_min
        self.x_max = x_max

        # Projections de requête (Query) : 2 têtes (Kpoli Gauche, Kpoli Droit)
        self.query_gauche = nn.Linear(1, self.FA_EMBED_DIM, dtype=torch.float32)
        self.query_droit = nn.Linear(1, self.FA_EMBED_DIM, dtype=torch.float32)

        # Tables Fâ
        kpoli_table, ref_table = self._build_reference_tables()
        self.register_buffer("kpoli_table", kpoli_table)  # (16, 4) - Keys
        self.register_buffer("ref_table", ref_table)      # (256, 4) - Values

    def _build_reference_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        kb = _get_kb()
        pol_map = {"favorable": 1.0, "neutre": 0.5,
                   "défavorable": 0.0, "non_défini": 0.5}

        # Value table (256, 4)
        table = torch.zeros(256, 4, dtype=torch.float32)
        # Key table (16, 4) pour les 16 Kpoli
        kpoli_table = torch.zeros(16, 4, dtype=torch.float32)
        
        for i, du in enumerate(kb):
            table[i, 0] = i / 255.0  # val_norm
            table[i, 1] = pol_map.get(du.get("polarite", "non_défini"), 0.5)
            table[i, 2] = float(du.get("intensite", 0.5))
            table[i, 3] = 1.0 if du.get("est_meji", False) else 0.0
            
            # Les 16 premiers éléments sont les Meji fondamentaux
            if i < 16:
                kpoli_table[i] = table[i]

        return kpoli_table, table

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_dual: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Architecture Combinatoire Fâ (16x16).
        """
        x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]

        x_range = self.x_max - self.x_min
        x_norm = torch.clamp((x - self.x_min) / (x_range + 1e-10), 0.0, 1.0)

        x_in = x_norm.unsqueeze(-1)  # (batch, n_features, 1)

        # 1. Génération des Queries Binaires (Gauche et Droit)
        Q_gauche = self.query_gauche(x_in)  # (b, n, 4)
        Q_droit = self.query_droit(x_in)    # (b, n, 4)

        # 2. Récupérer K_16 (Les 16 Meji générateurs) et V (Les 256 valeurs)
        K_16: torch.Tensor = self.kpoli_table  # type: ignore[assignment]
        V: torch.Tensor = self.ref_table       # type: ignore[assignment]

        # 3. Attention Gauche (sur 16) et Droite (sur 16)
        scores_gauche = torch.matmul(Q_gauche, K_16.transpose(0, 1)) / (self.FA_EMBED_DIM ** 0.5)
        scores_droit  = torch.matmul(Q_droit,  K_16.transpose(0, 1)) / (self.FA_EMBED_DIM ** 0.5)

        attn_gauche = torch.softmax(scores_gauche, dim=-1)  # (b, n, 16)
        attn_droit  = torch.softmax(scores_droit, dim=-1)   # (b, n, 16)

        # 4. Magie Combinatoire (Kronecker Product temporel)
        # attn_gauche.unsqueeze(-1) => (b, n, 16, 1)
        # attn_droit.unsqueeze(-2)  => (b, n, 1, 16)
        # Multiplication => (b, n, 16, 16)
        attn_matrix = attn_gauche.unsqueeze(-1) * attn_droit.unsqueeze(-2)
        
        # Le Fâ suit exactement ID = gauche_idx * 16 + droit_idx
        # Un flattened (.view) produit nativement l'alignement sur 256.
        attn_weights = attn_matrix.view(batch_size, self.n_features, 256)

        # 5. Calcul de l'embedding final
        embedded = torch.matmul(attn_weights, V)  # (b, n, 4)
        embedded[:, :, 0] = x_norm

        out = embedded.reshape(batch_size, -1)
        
        if return_dual:
            return out, attn_weights, attn_gauche, attn_droit 
        if return_attention:
            return out, attn_weights
            
        return out


# ─── TorchFaHidden ─────────────────────────────────────────────────────────────

class TorchFaHidden(nn.Module):
    """
    Couche cachée du FaNet — version PyTorch.

    y = Activation(x @ W.T + b)

    W initialisé depuis la Matrice-256. Activation = FaReLU ou LearnableFaSigma.
    Tout le backprop est géré par autograd — zéro calcul manuel.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "fa_sigma",
        seed: int = 42,
        fa_tensor: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Poids initialisés depuis la Matrice-256
        W = _fa_weight_tensor((hidden_dim, input_dim), fa_tensor, seed)
        self.weight = nn.Parameter(W)
        self.bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        # Activation Fâ (nn.Module → autograd natif)
        self.activation = get_torch_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.linear(x.to(torch.float32), self.weight, self.bias)
        return self.activation(z)


# ─── TorchFaDecoder ────────────────────────────────────────────────────────────

class TorchFaDecoder(nn.Module):
    """
    Couche de sortie FaNet — prédiction + interprétation Fâ.

    Activation FaDual (Meji) → sortie dans ]0, 1[.
    forward_with_interpretation() retourne la prédiction + le du correspondant.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_outputs: int = 1,
        seed: int = 42,
    ):
        super().__init__()

        limit = np.sqrt(6.0 / (n_outputs + hidden_dim))
        rng = np.random.default_rng(seed)
        W = rng.uniform(-limit, limit, (n_outputs, hidden_dim))

        self.weight = nn.Parameter(torch.tensor(W, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n_outputs, dtype=torch.float32))
        self.activation = TorchFaDual()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.linear(x.to(torch.float32), self.weight, self.bias)
        return self.activation(z)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les logits bruts AVANT activation (TorchFaDual/Sigmoid).
        Usage exclusif : boucle d'entrainement avec BCEWithLogitsLoss.
        BCEWithLogitsLoss applique sigmoid en interne (log-sum-exp trick)
        pour une stabilite numerique maximale."""
        return torch.nn.functional.linear(x.to(torch.float32), self.weight, self.bias)

    @torch.no_grad()
    def forward_with_interpretation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict]]:
        """Prédiction + interprétation Fâ pour chaque élément du batch."""
        preds = self.forward(x)
        interpretations = []
        for b in range(preds.shape[0]):
            val = float(preds[b, 0]) if preds.ndim > 1 else float(preds[b])
            du = value_to_algorithmic_state(val)
            interpretations.append(du)
        return preds, interpretations
