# Copyright (c) 2026 Cedoque BAGBONON, Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_activations : Activations Fâ en PyTorch (v4.1)
==============================================================
Fonctions d'activation différentiables inspirées de la combinatoire du Fâ :
  - FaReLU : activation continue par morceaux à 4 zones + sursaturation
  - TorchLearnableFaSigma : sigmoïde modulée de manière lisse (C-infini) par les 16 Kpoli
  - TorchFaDual : double passage FaSigma (conservé pour compatibilité et ablation)

Toutes les activations sont 100 % natives PyTorch (compatibles CUDA AMP,
torch.compile et export ONNX).

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


# Seuils canoniques du Fâ sur [0, 1]
FA_T1 = 1.0 / 16.0    # 0.0625
FA_T2 = 8.0 / 16.0    # 0.5000
FA_T3 = 15.0 / 16.0   # 0.9375

FA_GAIN_ZONE2 = 0.25
FA_GAIN_ZONE3 = 0.50
FA_GAIN_ZONE4 = 1.00
FA_GAIN_SURSAT = 0.10


class FaReLU(nn.Module):
    """
    FaReLU continue par morceaux (v4.1).

    Corrigée par rapport à v0.3/v4.0 : les zones sont raccordées sans
    discontinuité aux seuils T1, T2 et T3 grâce à la décomposition en
    incréments bornés (torch.clamp), ce qui rend l'activation strictement
    continue, monotone, compatible AMP GPU et exportable en ONNX.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z2 = torch.clamp(x, min=0.0, max=FA_T1)
        z3 = torch.clamp(x - FA_T1, min=0.0, max=FA_T2 - FA_T1)
        z4 = torch.clamp(x - FA_T2, min=0.0, max=FA_T3 - FA_T2)
        z5 = torch.clamp(x - FA_T3, min=0.0)
        return (
            z2 * FA_GAIN_ZONE2
            + z3 * FA_GAIN_ZONE3
            + z4 * FA_GAIN_ZONE4
            + z5 * FA_GAIN_SURSAT
        )


class TorchLearnableFaSigma(nn.Module):
    """
    FaSigma à polarités entraînables avec interpolation lisse sur les 16 Kpoli (v4.1).

    Contrairement à l'indexation discrète (kpoli_idx = long()) qui introduisait
    15 sauts discontinus, cette version interpole les 16 gains Kpoli par un noyau
    gaussien doux (softmax sur distances quadratiques aux 16 centres).

    Paramètre : raw in R^16
    Polarité  : sigmoid(raw) in (0, 1)
    Gain      : 0.5 + sigmoid(raw) in (0.5, 1.5)
    Forward   : y = sigmoid(x * gain_smooth(x))
    """

    N_KPOLI: int = 16

    def __init__(
        self,
        init_weights: Optional[np.ndarray] = None,
        x_range: tuple[float, float] = (-6.0, 6.0),
        polarity_mode: str = "fa",
        seed: int = 42,
    ):
        super().__init__()
        self.x_min = float(x_range[0])
        self.x_max = float(x_range[1])
        self.kernel_width = 1.0 / float(self.N_KPOLI)

        if init_weights is None:
            from bokono.ml.activations import load_polarity_weights
            init_weights = load_polarity_weights(
                polarity_mode=polarity_mode, seed=seed
            ).astype(np.float32)

        init_arr = np.clip(np.asarray(init_weights, dtype=np.float32), 1e-4, 1.0 - 1e-4)
        raw_init = np.log(init_arr / (1.0 - init_arr))

        self.raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        centers = torch.linspace(0.5 / 16.0, 15.5 / 16.0, self.N_KPOLI, dtype=torch.float32)
        self.register_buffer("kpoli_centers", centers)

    @property
    def polarity_weights(self) -> torch.Tensor:
        """Polarités actuelles dans ]0, 1[."""
        return torch.sigmoid(self.raw)

    @property
    def gain_values(self) -> torch.Tensor:
        """Gains par Kpoli dans ]0.5, 1.5[."""
        return 0.5 + self.polarity_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centers = getattr(self, "kpoli_centers").to(dtype=x.dtype, device=x.device)
        gains = self.gain_values.to(dtype=x.dtype, device=x.device)

        x_norm = torch.clamp(
            (x - self.x_min) / (self.x_max - self.x_min), 0.0, 1.0
        )
        dist_sq = (x_norm.unsqueeze(-1) - centers) ** 2
        attn = torch.softmax(-dist_sq / (2.0 * (self.kernel_width ** 2)), dim=-1)
        gain_per_elem = torch.sum(attn * gains, dim=-1)

        return torch.sigmoid(x * gain_per_elem)


class TorchFaDual(nn.Module):
    """
    FaDual : double passage FaSigma (conservé pour étude d'ablation et rétro-compatibilité).

    Note scientifique : la composition s1_remapped = 6 * s1 - 3 borne la sortie
    dans environ [0.047, 0.953] pour un gain unitaire. En classification calibrée,
    privilégier output_mode="standard" (sigmoid aligné sur BCEWithLogitsLoss).
    """

    def __init__(
        self,
        init_weights: Optional[np.ndarray] = None,
        polarity_mode: str = "fa",
        seed: int = 42,
    ):
        super().__init__()
        self.sigma = TorchLearnableFaSigma(
            init_weights=init_weights,
            polarity_mode=polarity_mode,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.sigma(x)
        s1_remapped = s1 * 6.0 - 3.0
        return self.sigma(s1_remapped)


TORCH_ACTIVATIONS = {
    "fa_relu": FaReLU,
    "fa_sigma": TorchLearnableFaSigma,
    "fa_dual": TorchFaDual,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def get_torch_activation(name: str, **kwargs) -> nn.Module:
    """Récupère une activation par son nom."""
    if name not in TORCH_ACTIVATIONS:
        raise KeyError(
            f"Activation '{name}' inconnue. Disponibles : {list(TORCH_ACTIVATIONS)}"
        )
    cls = TORCH_ACTIVATIONS[name]
    if name in ("fa_relu", "relu", "gelu"):
        return cls()
    return cls(**kwargs)
