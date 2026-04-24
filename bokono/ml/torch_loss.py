# Copyright (c) 2026 Cedoque BAGBONON -- Projet Matrice-256.
# Tous droits reserves. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_loss -- Fonctions de perte purement Fa (v3.1)
=============================================================
v3.1 -- Correction du probleme des classes desequilibrees.

Améliorations :
  - Remplacement de MSELoss par BCELoss (Binary Cross-Entropy) avec
    un pos_weight adaptatif pour penaliser fortement les faux negatifs
    sur les classes minoritaires (ex: 4.5% de positifs).
  - lambda_fa adaptatif : plus le desequilibre est fort, plus on reduit
    la penalite ontologique pour ne pas creer de competition de gradients.

Auteur : Projet Matrice-256 / bokono
"""

import torch
import torch.nn as nn
from bokono.ml.activations import load_polarity_weights


class FaImbalanceLoss(nn.Module):
    """
    Fonction de Perte de Desequilibre Fa v3.1.
    
    Calcule :
      Total_Loss = BCE(pred, y, pos_weight) + lambda_fa * Penalty(Fa_Equilibrium)
    
    Parametres :
      lambda_fa  : force de la penalite ontologique Fa (0 = desactivee)
      pos_weight : poids de la classe positive (ratio negatifs/positifs)  
                   Exemple : si 4.5% de positifs => pos_weight = 95.5/4.5 ~= 21
                   Un pos_weight eleve force le modele a ne pas ignorer la classe minoritaire.
    """

    def __init__(self, lambda_fa: float = 0.1, pos_weight: float = 1.0, task: str = "classification"):
        super().__init__()
        self.lambda_fa = lambda_fa
        self.task = task
        
        if self.task == "classification":
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
            self.register_buffer("pos_w", pw_tensor)
        else:
            self.mse = nn.MSELoss()
        
        # Charger les polarites de la KB (16 Kpolis)
        p_weights = load_polarity_weights().astype('float32')
        self.register_buffer("polarities", torch.tensor(p_weights))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, attn_g: torch.Tensor, attn_d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : Logits bruts du FaDecoder (batch_size, 1) -- NON passes par sigmoid
            target : Cible Y binaire (batch_size, 1)
            attn_g : Probabilites d'attention du Kpoli Gauche (batch_size, n_features, 16)
            attn_d : Probabilites d'attention du Kpoli Droit (batch_size, n_features, 16)
        """
        # 1. Perte Classique
        if self.task == "classification":
            self.bce.pos_weight = getattr(self, "pos_w").to(pred.device)
            loss_base = self.bce(pred, target)
        else:
            loss_base = self.mse(pred, target)

        if self.lambda_fa <= 0.0:
            return loss_base

        # 2. Penalite Ontologique (Fa Imbalance Penalty)
        # 2. Penalite Ontologique (Fa Imbalance Penalty)
        # En classification, pred est un logit => on sigmoid_
        # En regression, on s'attend a ce que pred soit deja borne [0,1]
        if self.task == "classification":
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = torch.clamp(pred, 0.0, 1.0)
            
        mean_attn_g = attn_g.mean(dim=1)
        mean_attn_d = attn_d.mean(dim=1)
        
        pol_g = torch.sum(mean_attn_g * getattr(self, "polarities"), dim=1, keepdim=True)
        pol_d = torch.sum(mean_attn_d * getattr(self, "polarities"), dim=1, keepdim=True)
        composite_polarity = (pol_g + pol_d) / 2.0
        
        fa_discrepancy = torch.abs(pred_prob - composite_polarity)
        penalty = fa_discrepancy.mean()

        return loss_base + self.lambda_fa * penalty
