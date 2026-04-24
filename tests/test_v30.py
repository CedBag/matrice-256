# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests pour la version 3.0.0 de FaNet (Architecture Fâ-Embedding Combinatoire Native).

Vérifie que :
1. Les Têtes d'Attention (Gauche et Droite) génèrent l'Attention 16x16.
2. Le Produit Tensoriel croisé reproduit la distribution sur les 256 signes.
3. L'extraction des poids d'Attention duale (XAI) fonctionne.
4. La propagation des gradients passe bien par les deux projections (Gauche et Droite).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from bokono.ml.torch_layers import TorchFaEmbedding
from bokono.ml.torch_network import TorchFaNet


class TestCombinatorialEmbedding:
    """Validation mathématique de la couche d'Embedding Combinatoire v3.0."""

    @pytest.fixture
    def embedding_layer(self):
        # 3 features d'entrée
        return TorchFaEmbedding(n_features=3, x_min=0.0, x_max=1.0)

    def test_combinatorial_softmax_distribution(self, embedding_layer):
        """Vérifie que l'Attention produit bien une distribution de probabilité via Kronecker sur 256 signes."""
        x = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)  # batch_size=1, n_features=3
        
        out, attn_weights, attn_g, attn_d = embedding_layer.forward(x, return_dual=True)
        
        # Les tenseurs d'attention gauche et droite doivent faire (1, 3, 16)
        assert attn_g.shape == (1, 3, 16)
        assert attn_d.shape == (1, 3, 16)
        
        # La somme des attentions individuelles doit valoir 1
        torch.testing.assert_close(torch.sum(attn_g, dim=-1), torch.ones(1, 3, dtype=torch.float32), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(torch.sum(attn_d, dim=-1), torch.ones(1, 3, dtype=torch.float32), rtol=1e-4, atol=1e-4)

        # Le tenseur d'attention combiné doit avoir la shape (batch, n_features, 256)
        assert attn_weights.shape == (1, 3, 256)
        
        # La somme de l'Outer Product doit valoir 1.0
        torch.testing.assert_close(torch.sum(attn_weights, dim=-1), torch.ones(1, 3, dtype=torch.float32), rtol=1e-4, atol=1e-4)
        
        # L'embedding de sortie doit faire 12
        assert out.shape == (1, 12)

    def test_dual_differentiability(self, embedding_layer):
        """Vérifie que les deux matrices de projection Q (Gauche, Droit) apprennent."""
        x = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        
        emb = embedding_layer(x)
        loss = emb.sum()
        loss.backward()
        
        # Les têtes Gauche et Droite doivent recevoir des gradients
        assert embedding_layer.query_gauche.weight.grad is not None
        assert embedding_layer.query_droit.weight.grad is not None
        
        grad_g = torch.sum(torch.abs(embedding_layer.query_gauche.weight.grad)).item()
        grad_d = torch.sum(torch.abs(embedding_layer.query_droit.weight.grad)).item()
        
        assert grad_g > 0.0, "Le gradient GAUCHE ne s'est pas propagé."
        assert grad_d > 0.0, "Le gradient DROIT ne s'est pas propagé."


class TestFaNetV30Network:
    """Validation de l'intégration dans FaNet v3.0."""

    @pytest.fixture
    def fanet_v3(self):
        return TorchFaNet(n_features=5, hidden_dim=16, n_outputs=1, seed=42)

    def test_version_bump(self, fanet_v3):
        assert fanet_v3.VERSION == "3.0.0"
        
    def test_end_to_end_training_combinatorial(self, fanet_v3):
        """Test rapide d'apprentissage."""
        np.random.seed(42)
        X = np.random.rand(10, 5)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        
        pred_before = fanet_v3.predict_proba(X)
        history = fanet_v3.fit(X, y, epochs=5, lr=0.1, optimizer="adam", verbose=False)
        pred_after = fanet_v3.predict_proba(X)
        
        assert not np.allclose(pred_before, pred_after)
        assert len(history["loss"]) == 5

    def test_xai_dual_extraction(self, fanet_v3):
        """L'interfaçage `predict(return_dual=True)` expose toute la chaîne causale Fâ."""
        X = np.array([[0.1, 0.9, 0.5, 0.0, 1.0]])
        
        preds, dus, attn_g, attn_d = fanet_v3.predict(X, return_dual=True)
        
        assert isinstance(preds, np.ndarray)
        assert isinstance(dus, list)
        
        # Validation géométrique
        assert attn_g.shape == (1, 5, 16)
        assert attn_d.shape == (1, 5, 16)
        np.testing.assert_allclose(np.sum(attn_g, axis=-1), np.ones((1, 5)), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.sum(attn_d, axis=-1), np.ones((1, 5)), rtol=1e-4, atol=1e-4)
