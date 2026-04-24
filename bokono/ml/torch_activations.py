# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_activations — Activations Fâ en PyTorch (v0.3)
================================================================
Port natif des fonctions d'activation bokono vers PyTorch.

Avantages par rapport à la version NumPy (v0.1/v0.2) :
  1. Autograd   — gradient calculé automatiquement, zéro backprop manuel
  2. GPU-ready  — .to("cuda") natif
  3. Composable — empilable dans nn.Sequential

Chaque activation est un nn.Module avec forward() standard.
LearnableFaSigma utilise nn.Parameter pour les poids de polarité.

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ─── Seuils canoniques du Fâ ──────────────────────────────────────────────────
FA_T1 = 1.0 / 16.0    # 0.0625
FA_T2 = 8.0 / 16.0    # 0.5000
FA_T3 = 15.0 / 16.0   # 0.9375

FA_GAIN_ZONE2 = 0.25
FA_GAIN_ZONE3 = 0.50
FA_GAIN_ZONE4 = 1.00
FA_GAIN_SURSAT = 0.10


# ─── FaReLU ───────────────────────────────────────────────────────────────────

class FaReLUFunction(torch.autograd.Function):
    """
    Autograd Function pour FaReLU — activation 4 zones du Fâ.

    Le forward est identique à la version NumPy.
    Le backward est calculé automatiquement par PyTorch via ce custom Function.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result = torch.zeros_like(x)

        m2 = (x >= 0) & (x < FA_T1)
        m3 = (x >= FA_T1) & (x < FA_T2)
        m4 = (x >= FA_T2) & (x < FA_T3)
        m5 = x >= FA_T3

        result[m2] = x[m2] * FA_GAIN_ZONE2
        result[m3] = x[m3] * FA_GAIN_ZONE3
        result[m4] = x[m4] * FA_GAIN_ZONE4
        result[m5] = FA_T3 + (x[m5] - FA_T3) * FA_GAIN_SURSAT

        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x, = ctx.saved_tensors
        grad = torch.zeros_like(x)

        grad[(x >= 0) & (x < FA_T1)] = FA_GAIN_ZONE2
        grad[(x >= FA_T1) & (x < FA_T2)] = FA_GAIN_ZONE3
        grad[(x >= FA_T2) & (x < FA_T3)] = FA_GAIN_ZONE4
        grad[x >= FA_T3] = FA_GAIN_SURSAT

        return grad_output * grad


class FaReLU(nn.Module):
    """FaReLU comme nn.Module — drop-in replacement pour nn.ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FaReLUFunction.apply(x)


# ─── LearnableFaSigma (nn.Module) ─────────────────────────────────────────────

class TorchLearnableFaSigma(nn.Module):
    """
    FaSigma avec poids de polarité entraînables — version PyTorch native.

    Les 16 raw_weights sont des nn.Parameter → autograd les met à jour
    automatiquement. Plus de backprop manuel.

    Paramètre : raw ∈ ℝ^16 (espace non borné)
    Polarité  : σ(raw) ∈ ]0, 1[
    Gain      : 0.5 + σ(raw) ∈ ]0.5, 1.5[

    Forward : y = σ(x × gain[zone(x)])
    """

    N_KPOLI: int = 16

    def __init__(
        self,
        init_weights: Optional[np.ndarray] = None,
        x_range: tuple[float, float] = (-6.0, 6.0),
    ):
        super().__init__()
        self.x_min = x_range[0]
        self.x_max = x_range[1]

        # Initialisation depuis la base de connaissance Fâ
        if init_weights is None:
            from bokono.ml.activations import load_polarity_weights
            init_weights = load_polarity_weights().astype(np.float32)

        init_weights = np.clip(init_weights, 1e-6, 1.0 - 1e-6)
        raw_init = np.log(init_weights / (1.0 - init_weights))

        # nn.Parameter → autograd gère le gradient automatiquement
        self.raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    @property
    def polarity_weights(self) -> torch.Tensor:
        """Polarités actuelles — dans ]0, 1[."""
        return torch.sigmoid(self.raw)

    @property
    def gain_values(self) -> torch.Tensor:
        """Gains par zone Kpoli — dans ]0.5, 1.5[."""
        return 0.5 + self.polarity_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identifier la zone Kpoli de chaque élément
        x_norm = torch.clamp((x - self.x_min) / (self.x_max - self.x_min), 0.0, 1.0)
        kpoli_idx = torch.clamp((x_norm * 16).long(), 0, 15)

        # Gain par élément (lookup vectorisé)
        gain_per_elem = self.gain_values[kpoli_idx]

        # Sigmoid modulé — tout est différentiable via autograd
        return torch.sigmoid(x * gain_per_elem)


# ─── FaDual ───────────────────────────────────────────────────────────────────

class TorchFaDual(nn.Module):
    """
    FaDual — Activation Meji (double passage FaSigma).

    Utilise un LearnableFaSigma interne — les poids de polarité
    sont partagés entre les deux passages.
    """

    def __init__(self, init_weights: Optional[np.ndarray] = None):
        super().__init__()
        self.sigma = TorchLearnableFaSigma(init_weights=init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.sigma(x)
        s1_remapped = s1 * 6.0 - 3.0
        return self.sigma(s1_remapped)


# ─── Registre ─────────────────────────────────────────────────────────────────

TORCH_ACTIVATIONS = {
    "fa_relu": FaReLU,
    "fa_sigma": TorchLearnableFaSigma,
    "fa_dual": TorchFaDual,
}


def get_torch_activation(name: str, **kwargs) -> nn.Module:
    """Récupère une activation Fâ par son nom."""
    if name not in TORCH_ACTIVATIONS:
        raise KeyError(f"Activation '{name}' inconnue. Disponibles : {list(TORCH_ACTIVATIONS)}")
    return TORCH_ACTIVATIONS[name](**kwargs)
