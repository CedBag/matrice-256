# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_network — FaNet v4.0 (PyTorch)
================================================
Le premier réseau de neurones béninois, avec Architecture Combinatoire Native.

Capacités v4.0 :
  1. Autograd    — gradients exacts via autograd PyTorch
  2. GPU-ready   — .to("cuda") natif + AMP FP16
  3. Embedding   — Combinatoire binaire native (16x16 -> 256)
  4. LSTM Natif  — Mode temporel pour séries IoT (seq_len > 0)
  5. Compilation — torch.compile() pour accélération 3-10× en inférence
  6. Optimiseurs — Adam, AdamW, SGD momentum
  7. Sauvegarde  — state_dict natif PyTorch

L'interface utilisateur reste simple :
  net = TorchFaNet(n_features=4, hidden_dim=32)
  history = net.fit(X, y, epochs=100)
  pred, du = net.predict(X_test)

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

logger = logging.getLogger("bokono.ml.torch_network")


def _validate_input(
    X: np.ndarray,
    y: np.ndarray | None = None,
    expected_features: int | None = None,
) -> None:
    """Validation centralisée des entrées — empêche les corruptions silencieuses."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X doit être un np.ndarray, reçu {type(X).__name__}")
    if X.ndim not in (1, 2):
        raise ValueError(f"X doit être 1D ou 2D, reçu {X.ndim}D (shape={X.shape})")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des NaN ou Inf — données corrompues.")
    if expected_features is not None and X.ndim == 2 and X.shape[1] != expected_features:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != n_features={expected_features}"
        )
    if y is not None:
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y doit être un np.ndarray, reçu {type(y).__name__}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contient des NaN ou Inf — cibles corrompues.")

from bokono.ml.torch_layers import TorchFaEmbedding, TorchFaHidden, TorchFaDecoder
from bokono.core.fa_matrix import FaMatrix


class TorchFaNet(nn.Module):
    """
    FaNet v0.3 — Réseau de neurones béninois en PyTorch.

    Architecture :
      Input(n_features) → Embedding(n_features×4) → Hidden(hidden_dim) → Output(n_outputs)

    Poids initialisés depuis la Matrice-256.
    Poids de polarité Kpoli entraînables (LearnableFaSigma).
    Embedding Fâ différentiable (Combinatoire Native 16x16).
    """

    VERSION = "4.0.0"

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        n_outputs: int = 1,
        activation: str = "fa_sigma",
        x_min: float = 0.0,
        x_max: float = 1.0,
        seed: int = 42,
        seq_len: int = 0,
        bidirectional: bool = False,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.activation_name = activation
        self.x_min = x_min
        self.x_max = x_max
        self.seed = seed
        self.seq_len = seq_len  # 0 = mode Tabulaire (retro-compat), >0 = mode LSTM Temporel
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout_val = dropout

        # Tenseur Matrice-256 pour l'initialisation
        fa_tensor = FaMatrix().to_weight_tensor()

        embedding_dim = n_features * TorchFaEmbedding.FA_EMBED_DIM

        self.embedding = TorchFaEmbedding(n_features, x_min, x_max)

        if seq_len > 0:
            # Mode LSTM : L'Embedding Fa extrait les Kpolis par instant T,
            # puis le LSTM lit la dynamique temporelle de la sequence.
            # TorchFaHidden (Dense) est bypasse pour ne pas confondre espace et temps.
            self.lstm_layer = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.hidden = None  # Non utilise en mode LSTM
        else:
            # Mode Tabulaire (Classification / Regression instantanee)
            self.hidden = TorchFaHidden(embedding_dim, hidden_dim, activation, seed, fa_tensor)
            self.lstm_layer = None  # Non utilise en mode Dense

        # Dimension d'entrée du décodeur : doublée si LSTM bidirectionnel
        decoder_input_dim = hidden_dim * 2 if (seq_len > 0 and bidirectional) else hidden_dim

        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

        self.decoder = TorchFaDecoder(decoder_input_dim, n_outputs, seed)
        self._compiled = False

    def compile_model(self, mode: str = "reduce-overhead"):
        """Compile le forward pass via torch.compile() (PyTorch 2.0+).
        Accélère l'inférence de 3× à 10× après le premier appel (warmup JIT).
        
        Args:
            mode: 'default', 'reduce-overhead' (recommandé), ou 'max-autotune'
        """
        if self._compiled:
            return self
        try:
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if torch_version >= (2, 0):
                self.forward = torch.compile(self.forward, mode=mode)
                self._compiled = True
                logging.info(f"FaNet compilé avec torch.compile(mode='{mode}')")
            else:
                logging.warning(f"torch.compile() requiert PyTorch >= 2.0 (actuel: {torch.__version__})")
        except Exception as e:
            logging.warning(f"torch.compile() non disponible: {e}. Mode eager conservé.")
        return self

    def _embed_temporal(self, x: torch.Tensor):
        """
        Mode LSTM : x est un vecteur aplati (batch, seq_len * n_features).
        On restructure, on passe chaque pas de temps dans le FaEmbedding,
        puis dans le LSTM pour extraire la memoire dynamique de la sequence.
        Retourne (last_hidden, attn_g_mean, attn_d_mean).
        """
        B = x.shape[0]
        # 1. Restructurer : (batch, seq_len, n_features)
        x_seq = x.view(B, self.seq_len, self.n_features)
        # 2. Aplatir temporellement : (batch*seq_len, n_features)
        x_flat = x_seq.reshape(B * self.seq_len, self.n_features)
        # 3. FaEmbedding sur chaque instant T (extraction Kpolis parallele)
        emb_flat, attn_flat, attn_g_flat, attn_d_flat = self.embedding(x_flat, return_dual=True)
        embed_dim = emb_flat.shape[-1]
        # 4. Restructurer en sequence pour LSTM : (batch, seq_len, embed_dim)
        emb_seq = emb_flat.view(B, self.seq_len, embed_dim)
        # 5. LSTM : sortie (batch, seq_len, hidden_dim * num_directions), dernier etat cache (num_layers * num_directions, batch, hidden_dim)
        _, (h_n, _) = self.lstm_layer(emb_seq)
        
        if self.bidirectional:
            # Concaténer le dernier état caché forward (h_n[-2]) et backward (h_n[-1]) de la dernière couche LSTM
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            last_hidden = torch.cat((h_forward, h_backward), dim=-1)  # (batch, hidden_dim * 2)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_dim)

        if self.dropout_layer is not None:
            last_hidden = self.dropout_layer(last_hidden)
            
        # 6. Attention ontologique moyenne sur la fenetre temporelle (pour FaImbalanceLoss)
        attn_g_mean = attn_g_flat.view(B, self.seq_len, self.n_features, -1).mean(dim=1)  # (batch, n_features, 16)
        attn_d_mean = attn_d_flat.view(B, self.seq_len, self.n_features, -1).mean(dim=1)
        return last_hidden, attn_g_mean, attn_d_mean

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_dual: bool = False):
        """Passe avant : x -> prediction dans ]0, 1[.
        Mode Tabulaire (seq_len=0) : Dense classique.
        Mode LSTM (seq_len>0)      : Sequence temporelle via LSTM.
        """
        if self.seq_len > 0:
            last_h, attn_g, attn_d = self._embed_temporal(x)
            out = self.decoder(last_h)
            if return_dual:
                return out, attn_g, attn_d
            return out

        # Mode Tabulaire
        if return_dual:
            emb, attn, attn_gauche, attn_droit = self.embedding(x, return_dual=True)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            return self.decoder(h), attn_gauche, attn_droit
        elif return_attention:
            emb, attn = self.embedding(x, return_attention=True)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            return self.decoder(h), attn
        else:
            emb = self.embedding(x)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            return self.decoder(h)

    def forward_logits(self, x: torch.Tensor, return_dual: bool = False):
        """Logits bruts (avant sigmoid) pour BCEWithLogitsLoss.
        Mode LSTM ou Tabulaire detecte automatiquement via self.seq_len.
        """
        if self.seq_len > 0:
            last_h, attn_g, attn_d = self._embed_temporal(x)
            out = self.decoder.forward_logits(last_h)
            if return_dual:
                return out, attn_g, attn_d
            return out

        # Mode Tabulaire
        if return_dual:
            emb, attn, attn_gauche, attn_droit = self.embedding(x, return_dual=True)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            return self.decoder.forward_logits(h), attn_gauche, attn_droit
        else:
            emb = self.embedding(x)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            return self.decoder.forward_logits(h)

    # ─── Entraînement ─────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        optimizer: str = "adam",
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        verbose_every: int = 10,
        lambda_fa: float = 0.1,
        pos_weight: float = 1.0,
        task: str = "classification",
        compile: bool = False,
    ) -> dict[str, list[float]]:
        """
        Entrainement supervise.

        Utilise l'optimiseur PyTorch (Adam par defaut) au lieu du SGD manuel.

        Args:
            X:            Donnees (n_samples, n_features) - NumPy
            y:            Cibles (n_samples,) - NumPy
            epochs:       Nombre d'epoques
            lr:           Learning rate
            batch_size:   Taille des mini-batches
            optimizer:    'adam', 'adamw', ou 'sgd'
            X_val:        Donnees de validation (optionnel)
            y_val:        Cibles de validation (optionnel)
            verbose:      Afficher la progression
            verbose_every: Afficher tous les N epochs
            lambda_fa:    Force de la penalite ontologique Fa (0 = desactivee)
            pos_weight:   Poids de la classe positive pour BCEWithLogitsLoss.
                          Valeur recommandee : n_negatifs / n_positifs.
                          Ex : 4.5%% positifs => pos_weight = 95.5/4.5 = 21.0
                          Defaut : 1.0 (classes equilibrees, comportement classique)

        Returns:
            {"loss": [...], "val_loss": [...]}
        """
        # En mode LSTM, X est aplati : (batch, seq_len * n_features)
        # => la validation doit porter sur seq_len * n_features, pas n_features seul.
        expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
        _validate_input(X, y, expected_w)
        if X_val is not None and y_val is not None:
            _validate_input(X_val, y_val, expected_w)

        device = next(self.parameters()).device
        is_cuda = (device.type == "cuda")

        # 1. DataLoader (Big Data OOM Protection)
        from torch.utils.data import TensorDataset, DataLoader
        
        # Les données restent en CPU RAM au début, on force float32
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.float32))
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
            
        dataset = TensorDataset(X_t, y_t)
        # DataLoader streamera les batchs en VRAM un par un
        # generator permet reproductibilité du shuffle
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=rng
        )

        # Optimiseur PyTorch
        opt_map = {
            "adam": lambda: torch.optim.Adam(self.parameters(), lr=lr),
            "adamw": lambda: torch.optim.AdamW(self.parameters(), lr=lr),
            "sgd": lambda: torch.optim.SGD(self.parameters(), lr=lr),
        }
        if optimizer not in opt_map:
            raise ValueError(f"Optimiseur inconnu : '{optimizer}'. Disponibles : {list(opt_map)}")
        opt = opt_map[optimizer]()

        # FaImbalanceLoss v3.1 — BCE (ou MSE) + penalite ontologique + pos_weight
        from bokono.ml.torch_loss import FaImbalanceLoss
        loss_fn = FaImbalanceLoss(lambda_fa=lambda_fa, pos_weight=pos_weight, task=task)
        scaler = torch.amp.GradScaler('cuda') if is_cuda else None

        n_samples = X_t.shape[0]
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        # Compilation JIT optionnelle (PyTorch 2.0+)
        if compile:
            self.compile_model()

        self.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_b in loader:
                # Streaming en VRAM
                X_b, y_b = X_b.to(device), y_b.to(device)

                opt.zero_grad()
                
                # Choix de la passe avant selon la tache
                def _forward_pass(X_batch):
                    if task == "classification":
                        return self.forward_logits(X_batch, return_dual=True)
                    else:
                        return self.forward(X_batch, return_dual=True)

                # Forward pass avec AMP si GPU
                if is_cuda and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        pred, attn_g, attn_d = _forward_pass(X_b)
                        loss = loss_fn(pred, y_b, attn_g, attn_d)
                    # Backward pass scalé (pour éviter l'underflow en FP16)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred, attn_g, attn_d = _forward_pass(X_b)
                    loss = loss_fn(pred, y_b, attn_g, attn_d)
                    loss.backward()
                    opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            # Validation
            val_loss_val = 0.0
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_v = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
                    y_v = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)
                    if y_v.ndim == 1:
                        y_v = y_v.unsqueeze(1)
                        
                    if is_cuda:
                        with torch.amp.autocast('cuda'):
                            if task == "classification":
                                val_pred, v_attn_g, v_attn_d = self.forward_logits(X_v, return_dual=True)
                            else:
                                val_pred, v_attn_g, v_attn_d = self.forward(X_v, return_dual=True)
                            val_loss_val = loss_fn(val_pred, y_v, v_attn_g, v_attn_d).item()
                    else:
                        if task == "classification":
                            val_pred, v_attn_g, v_attn_d = self.forward_logits(X_v, return_dual=True)
                        else:
                            val_pred, v_attn_g, v_attn_d = self.forward(X_v, return_dual=True)
                        val_loss_val = loss_fn(val_pred, y_v, v_attn_g, v_attn_d).item()
                        
                history["val_loss"].append(val_loss_val)
                self.train()

            if verbose and (epoch % verbose_every == 0 or epoch == 1):
                val_str = f" | val_loss={val_loss_val:.5f}" if X_val is not None else ""
                print(f"  Époque {epoch:4d}/{epochs} | loss={avg_loss:.5f}{val_str}")

        self.eval()
        return history

    # ─── Inférence ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: np.ndarray, return_attention: bool = False, return_dual: bool = False):
        """
        Prédiction avec interprétation Fâ.

        Args:
            x: (batch_size, n_features) ou (n_features,) — NumPy
            return_attention: Si True, retourne la matrice d'attention (256)
            return_dual: Si True, retourne (attn_gauche(16), attn_droit(16))

        Returns:
            Si normal: (predictions: np.ndarray, interpretations: list[dict])
            Si return_attention: (preds, dus, attn(batch, n_features, 256))
            Si return_dual: (preds, dus, attn_g, attn_d)
        """
        expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
        _validate_input(x, None, expected_w)
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(np.atleast_2d(np.asarray(x, dtype=np.float32)), device=device)

        if self.seq_len > 0:
            last_h, attn_g, attn_d = self._embed_temporal(x_t)
            preds, dus = self.decoder.forward_with_interpretation(last_h)
            if return_dual:
                return preds.cpu().numpy(), dus, attn_g.cpu().numpy(), attn_d.cpu().numpy()
            elif return_attention:
                attn_matrix = attn_g.unsqueeze(-1) * attn_d.unsqueeze(-2)
                attn_weights = attn_matrix.view(x_t.shape[0], self.n_features, 256)
                return preds.cpu().numpy(), dus, attn_weights.cpu().numpy()
            else:
                return preds.cpu().numpy(), dus

        # Mode Tabulaire
        if return_dual:
            emb, attn, attn_g, attn_d = self.embedding(x_t, return_dual=True)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            preds, dus = self.decoder.forward_with_interpretation(h)
            return preds.cpu().numpy(), dus, attn_g.cpu().numpy(), attn_d.cpu().numpy()
        elif return_attention:
            emb, attn = self.embedding(x_t, return_attention=True)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            preds, dus = self.decoder.forward_with_interpretation(h)
            return preds.cpu().numpy(), dus, attn.cpu().numpy()
        else:
            emb = self.embedding(x_t)
            h = self.hidden(emb)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
            preds, dus = self.decoder.forward_with_interpretation(h)
            return preds.cpu().numpy(), dus

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Prediction numerique seule."""
        expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
        _validate_input(x, None, expected_w)
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(np.atleast_2d(np.asarray(x, dtype=np.float32)), device=device)
        return self(x_t).cpu().numpy()

    # ─── Évaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evalue le reseau sur un jeu de donnees."""
        expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
        _validate_input(X, y, expected_w)
        self.eval()
        device = next(self.parameters()).device

        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

        pred = self(X_t)
        mse = float(torch.mean((pred - y_t) ** 2).item())
        mae = float(torch.mean(torch.abs(pred - y_t)).item())
        rmse = float(torch.sqrt(torch.tensor(mse)).item())

        metrics: dict[str, float] = {"loss": mse, "mae": mae, "rmse": rmse}

        if self.n_outputs == 1:
            y_cls = (pred >= 0.5).int()
            y_true_cls = (y_t >= 0.5).int()
            metrics["accuracy"] = float(torch.mean((y_cls == y_true_cls).float()).item())

        return metrics

    # ─── Persistance ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Sauvegarde le modèle complet (state_dict + métadonnées)."""
        import os
        from bokono.core.fa_encoder import get_kb_hash
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "n_outputs": self.n_outputs,
            "activation": self.activation_name,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "seq_len": self.seq_len,
            "bidirectional": self.bidirectional,
            "num_layers": self.num_layers,
            "dropout": self.dropout_val,
            "version": self.VERSION,
            "kb_hash": get_kb_hash(),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TorchFaNet":
        """Charge un modèle sauvegardé."""
        from bokono.core.fa_encoder import get_kb_hash
        
        # SÉCURITÉ : weights_only=True empêche l'exécution de code arbitraire
        # via pickle deserialization (CVE connue sur torch.load).
        data = torch.load(path, weights_only=True)
        
        # Vérification d'intégrité de la KB
        current_hash = get_kb_hash()
        saved_hash = data.get("kb_hash")
        if saved_hash and saved_hash != current_hash:
            logger.warning(
                "La Data Poisoning Protection signale que la base de connaissance (KB) "
                "a été modifiée depuis l'entraînement du modèle. Les interprétations peuvent diverger."
            )
        net = cls(
            n_features=data["n_features"],
            hidden_dim=data["hidden_dim"],
            n_outputs=data["n_outputs"],
            activation=data.get("activation", "fa_sigma"),
            x_min=data.get("x_min", 0.0),
            x_max=data.get("x_max", 1.0),
            seq_len=data.get("seq_len", 0),
            bidirectional=data.get("bidirectional", False),
            num_layers=data.get("num_layers", 1),
            dropout=data.get("dropout", 0.0),
            **kwargs,
        )
        net.load_state_dict(data["state_dict"])
        net.eval()
        return net

    # ─── Affichage ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Affiche le résumé de l'architecture FaNet v0.3."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total_params - trainable

        emb_dim = self.n_features * TorchFaEmbedding.FA_EMBED_DIM

        print("=" * 65)
        print(f"  FaNet v{self.VERSION} — Réseau de neurones béninois (PyTorch)")
        print("=" * 65)
        print(f"  Backend            : PyTorch {torch.__version__}")
        device = next(self.parameters()).device
        print(f"  Device             : {device}")
        print(f"  Activation cachée  : {self.activation_name}")
        print()
        print(f"  {'Couche':<25} {'Forme sortie':<20} {'Paramètres':>12}")
        print("  " + "-" * 60)

        # Embedding : 2 projections Q_gauche, Q_droit
        emb_p = sum(p.numel() for p in self.embedding.parameters())
        print(f"  {'FaEmbed Combinatory v3':<25} {'(' + str(emb_dim) + ',)':<20} {emb_p:>12,}")

        hidden_p = sum(p.numel() for p in self.hidden.parameters())
        print(f"  {'FaHidden (Fâ)':<25} {'(' + str(self.hidden_dim) + ',)':<20} {hidden_p:>12,}")

        dec_p = sum(p.numel() for p in self.decoder.parameters())
        print(f"  {'FaDecoder (Meji)':<25} {'(' + str(self.n_outputs) + ',)':<20} {dec_p:>12,}")

        print("  " + "-" * 60)
        print(f"  {'TOTAL':<25} {'':<20} {total_params:>12,}")
        print(f"  {'  entraînables':<25} {'':<20} {trainable:>12,}")
        print(f"  {'  non-entraînables':<25} {'':<20} {non_trainable:>12,}")
        print("=" * 65)
        print(f"  Initialisation     : Matrice-256 (tenseur 16×16×4)")
        print(f"  Embedding          : COMBINATOIRE (Kronecker 16x16)")
        print(f"  Polarité Kpoli     : entraînable (16 params)")
        print(f"  Interprétation     : chaque prédiction → du Fâ")
        print("=" * 65)

    # ─── Intégration Scikit-Learn (MLOps) ─────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Compatibilité native avec Scikit-Learn (GridSearchCV, Pipelines)."""
        return {
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "n_outputs": self.n_outputs,
            "activation": self.activation_name,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "bidirectional": self.bidirectional,
            "num_layers": self.num_layers,
            "dropout": self.dropout_val,
        }

    def set_params(self, **params) -> "TorchFaNet":
        """Compatibilité native avec Scikit-Learn."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ─── Déploiement Edge (Industriel) ────────────────────────────────────────

    def export_onnx(self, path: str, input_shape: tuple[int, ...] | None = None) -> None:
        """
        Exporte le cœur tensoriel Fâ en fichier ONNX pour l'inférence C++/Go Edge.
        
        Nécessite : `pip install onnx onnxscript`
        
        Args:
            path: Chemin du fichier de destination (ex: 'model.onnx')
            input_shape: Forme du tenseur d'entrée factice (par défaut: (1, n_features))
        """
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            raise ImportError("L'export ONNX nécessite 'onnxscript'. Exécutez : pip install onnx onnxscript")
        self.eval()
        device = next(self.parameters()).device
        
        # Le tenseur factice (Dummy Tensor) sert à tracer le graphe PyTorch
        if input_shape is None:
            input_shape = (1, self.n_features)
            
        dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)
        
        # On exporte UNIQUEMENT la passe "forward numeric" car l'analyse JSON 
        # (return_dual=True) de forward_with_interpretation n'est pas "traçable" en ONNX.
        
        class _OnnxForwardWrapper(nn.Module):
            def __init__(self, core_net):
                super().__init__()
                self.core = core_net
            def forward(self, x):
                return self.core(x)
                
        wrapper = _OnnxForwardWrapper(self)
        
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        torch.onnx.export(
            wrapper, 
            dummy_input, 
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Modèle Fâ (v{self.VERSION}) exporté avec succès en ONNX vers {path}")
