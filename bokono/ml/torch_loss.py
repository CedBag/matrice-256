# Copyright (c) 2026 Cedoque BAGBONON, Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_loss : Fonctions de perte Fâ (v4.1)
===================================================
Correctifs scientifiques v4.1 :
  - Gestion complète CPU / GPU CUDA (les buffers polarities et pos_w suivent pred.device).
  - Support de polarity_weights personnalisé (cohérence exacte avec polarity_mode="fa",
    "constant" ou "permuted" dans l'étude d'ablation).
  - Support from_logits=True (BCEWithLogitsLoss) et from_logits=False (BCELoss pour
    l'ablation TorchFaDual) afin de garantir train == inference.
  - Identité algébrique vérifiée : sous le produit de Kronecker A = attn_g x attn_d,
    composite_polarity = (pol_g + pol_d) / 2 est exactement égale à la polarité attendue
    sur les 256 Du.

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from bokono.ml.activations import load_polarity_weights


class FaImbalanceLoss(nn.Module):
    """
    Fonction de perte neuro-symbolique FaImbalanceLoss v4.1.

    Calcule :
      Total_Loss = Base_Loss(pred, target) + lambda_fa * mean(|pred_prob - composite_polarity|)
    """

    def __init__(
        self,
        lambda_fa: float = 0.1,
        pos_weight: float = 1.0,
        task: str = "classification",
        from_logits: bool = True,
        polarity_weights: Optional[np.ndarray] = None,
        polarity_mode: str = "fa",
        seed: int = 42,
    ):
        super().__init__()
        self.lambda_fa = float(lambda_fa)
        self.task = task
        self.from_logits = bool(from_logits)

        pw_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
        self.register_buffer("pos_w", pw_tensor)

        if self.task == "classification":
            self.bce_logits = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
            self.bce_probs = nn.BCELoss(reduction="none")
        else:
            self.mse = nn.MSELoss()

        if polarity_weights is None:
            p_weights = load_polarity_weights(
                polarity_mode=polarity_mode, seed=seed
            ).astype(np.float32)
        else:
            p_weights = np.asarray(polarity_weights, dtype=np.float32)

        self.register_buffer("polarities", torch.tensor(p_weights, dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attn_g: torch.Tensor,
        attn_d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred   : logits bruts (from_logits=True) ou probabilités/valeurs (from_logits=False)
            target : cible (batch_size, 1)
            attn_g : attention Kpoli gauche (batch_size, n_features, 16)
            attn_d : attention Kpoli droit  (batch_size, n_features, 16)
        """
        device = pred.device
        dtype = pred.dtype

        if self.task == "classification":
            pos_w = getattr(self, "pos_w").to(device=device, dtype=dtype)
            if self.from_logits:
                loss_base = nn.functional.binary_cross_entropy_with_logits(
                    pred, target, pos_weight=pos_w
                )
                pred_prob = torch.sigmoid(pred)
            else:
                p_safe = torch.clamp(pred.float(), 1e-6, 1.0 - 1e-6)
                t_f32 = target.float()
                pos_w_f32 = getattr(self, "pos_w").to(device=device, dtype=torch.float32)
                weights = torch.where(t_f32 > 0.5, pos_w_f32, torch.ones_like(t_f32))
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    loss_base = (self.bce_probs(p_safe, t_f32) * weights).mean()
                pred_prob = p_safe.to(dtype=dtype)
        else:
            loss_base = self.mse(pred, target)
            pred_prob = torch.sigmoid(pred)

        if self.lambda_fa <= 0.0:
            return loss_base

        polarities = getattr(self, "polarities").to(device=device, dtype=dtype)
        mean_attn_g = attn_g.mean(dim=1)  # (batch, 16)
        mean_attn_d = attn_d.mean(dim=1)  # (batch, 16)

        pol_g = torch.sum(mean_attn_g * polarities, dim=1, keepdim=True)
        pol_d = torch.sum(mean_attn_d * polarities, dim=1, keepdim=True)
        composite_polarity = 0.5 * (pol_g + pol_d)

        fa_discrepancy = torch.abs(pred_prob - composite_polarity)
        penalty = fa_discrepancy.mean()

        return loss_base + self.lambda_fa * penalty
