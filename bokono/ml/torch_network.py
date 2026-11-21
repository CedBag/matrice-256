# Copyright (c) 2026 Cedoque BAGBONON, Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.ml.torch_network : FaNet v4.2 (PyTorch)
==============================================
Réseau neuro-symbolique fondé sur la combinatoire binaire du Fâ (16x16 -> 256).

Évolutions architecturales v4.2 (issues de l'analyse des benchmarks Kaggle v5.0) :
  1. Clés K_16 sur l'hypercube binaire exact 4 bits {-1, +1}^4 + requêtes
     cross-features dans TorchFaEmbedding.
  2. Bloc caché neuro-symbolique à 2 étages (mode tabulaire) :
     mixer FaReLU (frontières nettes par morceaux, capture des clusters multi-variables)
     -> hidden FaSigma + connexion résiduelle (lissage biomédical, calibration et MIA).
  3. Initialisation plein rang sans colinéarité de pas 4.
  4. Boucle d'entraînement rapide sur GPU (mini-batchs par indexation directe sur device
     pour les datasets <= 50 000 lignes, réduisant l'overhead CPU-GPU par 4x à 6x).

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import copy
import logging
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from bokono.core.fa_matrix import FaMatrix
from bokono.ml.torch_layers import TorchFaEmbedding, TorchFaHidden, TorchFaDecoder
from bokono.ml.torch_loss import FaImbalanceLoss

logger = logging.getLogger("bokono.ml.torch_network")


def _validate_input(
    X: np.ndarray,
    y: np.ndarray | None = None,
    expected_features: int | None = None,
) -> None:
    """Validation centralisée des entrées."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X doit être un np.ndarray, reçu {type(X).__name__}")
    if X.ndim not in (1, 2):
        raise ValueError(f"X doit être 1D ou 2D, reçu {X.ndim}D (shape={X.shape})")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contient des NaN ou Inf (données corrompues).")
    if expected_features is not None and X.ndim == 2 and X.shape[1] != expected_features:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} != attendu={expected_features}"
        )
    if y is not None:
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y doit être un np.ndarray, reçu {type(y).__name__}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contient des NaN ou Inf (cibles corrompues).")


class TorchFaNet(nn.Module):
    """
    FaNet v4.2 : réseau de neurones neuro-symbolique en PyTorch.

    Architecture tabulaire (seq_len=0) :
      Input(n_features)
        -> TorchFaEmbedding(n_features * 4, hypercube 4 bits + cross-features)
        -> mixer TorchFaHidden(hidden_dim, FaReLU) [si use_residual_mixer=True]
        -> hidden TorchFaHidden(hidden_dim, FaSigma) + connexion résiduelle
        -> TorchFaDecoder(n_outputs)

    Architecture temporelle (seq_len>0) :
      Input(seq_len * n_features) -> TorchFaEmbedding par instant T -> LSTM -> TorchFaDecoder(n_outputs)
    """

    VERSION = "4.2.0"

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
        task: str = "classification",
        output_mode: str = "standard",
        polarity_mode: str = "fa",
        init_mode: str = "fa",
        per_feature_queries: bool = True,
        auto_normalize: bool = True,
        cross_feature: bool = True,
        use_residual_mixer: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.activation_name = activation
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.seed = int(seed)
        self.seq_len = int(seq_len)
        self.bidirectional = bool(bidirectional)
        self.num_layers = int(num_layers)
        self.dropout_val = float(dropout)
        self.task = task
        self.output_mode = output_mode
        self.polarity_mode = polarity_mode
        self.init_mode = init_mode
        self.per_feature_queries = bool(per_feature_queries)
        self.auto_normalize = bool(auto_normalize)
        self.cross_feature = bool(cross_feature)
        self.use_residual_mixer = bool(use_residual_mixer)

        torch.manual_seed(self.seed)

        fa_tensor = FaMatrix().to_weight_tensor() if init_mode == "fa" else None
        embedding_dim = n_features * TorchFaEmbedding.FA_EMBED_DIM

        self.embedding = TorchFaEmbedding(
            n_features=n_features,
            x_min=x_min,
            x_max=x_max,
            polarity_mode=polarity_mode,
            seed=seed,
            per_feature_queries=per_feature_queries,
            auto_normalize=auto_normalize,
            cross_feature=cross_feature,
        )

        if self.seq_len > 0:
            self.lstm_layer: Optional[nn.LSTM] = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            self.mixer: Optional[TorchFaHidden] = None
            self.hidden: Optional[TorchFaHidden] = None
        else:
            if self.use_residual_mixer:
                # Étage 1 : mélange multi-variables par FaReLU continue (frontières nettes)
                self.mixer = TorchFaHidden(
                    input_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    activation="fa_relu",
                    seed=seed + 500,
                    fa_tensor=fa_tensor,
                    polarity_mode=polarity_mode,
                )
                hidden_in = hidden_dim
            else:
                self.mixer = None
                hidden_in = embedding_dim

            # Étage 2 : régularisation douce par FaSigma (avec connexion résiduelle si mixer actif)
            self.hidden = TorchFaHidden(
                input_dim=hidden_in,
                hidden_dim=hidden_dim,
                activation=activation,
                seed=seed,
                fa_tensor=fa_tensor,
                polarity_mode=polarity_mode,
            )
            self.lstm_layer = None

        decoder_input_dim = hidden_dim * 2 if (self.seq_len > 0 and bidirectional) else hidden_dim
        self.dropout_layer: Optional[nn.Dropout] = nn.Dropout(p=dropout) if dropout > 0.0 else None

        self.decoder = TorchFaDecoder(
            hidden_dim=decoder_input_dim,
            n_outputs=n_outputs,
            seed=seed,
            task=task,
            output_mode=output_mode,
            polarity_mode=polarity_mode,
        )
        self._compiled = False

    def compile_model(self, mode: str = "reduce-overhead") -> "TorchFaNet":
        """Compile le forward pass via torch.compile() (PyTorch 2.0+)."""
        if self._compiled:
            return self
        try:
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_version >= (2, 0):
                self.forward = torch.compile(self.forward, mode=mode)  # type: ignore[method-assign]
                self._compiled = True
                logger.info(f"FaNet compilé avec torch.compile(mode='{mode}')")
            else:
                logger.warning(f"torch.compile() requiert PyTorch >= 2.0 (actuel: {torch.__version__})")
        except Exception as e:
            logger.warning(f"torch.compile() non disponible: {e}. Mode eager conservé.")
        return self

    def _embed_temporal(self, x: torch.Tensor):
        """
        Mode LSTM : x est un vecteur aplati (batch, seq_len * n_features).
        """
        assert self.lstm_layer is not None
        B = x.shape[0]
        x_seq = x.reshape(B, self.seq_len, self.n_features)
        x_flat = x_seq.reshape(B * self.seq_len, self.n_features)

        emb_flat, _, attn_g_flat, attn_d_flat = self.embedding(x_flat, return_dual=True)
        embed_dim = emb_flat.shape[-1]
        emb_seq = emb_flat.reshape(B, self.seq_len, embed_dim)

        _, (h_n, _) = self.lstm_layer(emb_seq)
        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            last_hidden = h_n[-1]

        if self.dropout_layer is not None:
            last_hidden = self.dropout_layer(last_hidden)

        attn_g_mean = attn_g_flat.reshape(B, self.seq_len, self.n_features, -1).mean(dim=1)
        attn_d_mean = attn_d_flat.reshape(B, self.seq_len, self.n_features, -1).mean(dim=1)
        return last_hidden, attn_g_mean, attn_d_mean

    def _apply_tabular_hidden(self, emb: torch.Tensor) -> torch.Tensor:
        assert self.hidden is not None
        if self.mixer is not None:
            h1 = self.mixer(emb)
            h2 = self.hidden(h1)
            h = h1 + h2
        else:
            h = self.hidden(emb)
        if self.dropout_layer is not None:
            h = self.dropout_layer(h)
        return h

    def _encode_tabular(self, x: torch.Tensor, return_attention: bool = False, return_dual: bool = False):
        if return_dual:
            emb, attn, attn_g, attn_d = self.embedding(x, return_dual=True)
            return self._apply_tabular_hidden(emb), attn, attn_g, attn_d
        if return_attention:
            emb, attn = self.embedding(x, return_attention=True)
            return self._apply_tabular_hidden(emb), attn
        emb = self.embedding(x)
        return self._apply_tabular_hidden(emb)

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_dual: bool = False):
        """Passe avant cohérente avec la tâche (classification ou régression)."""
        if self.seq_len > 0:
            last_h, attn_g, attn_d = self._embed_temporal(x)
            out = self.decoder(last_h)
            if return_dual:
                return out, attn_g, attn_d
            if return_attention:
                attn_matrix = attn_g.unsqueeze(-1) * attn_d.unsqueeze(-2)
                attn_weights = attn_matrix.reshape(x.shape[0], self.n_features, 256)
                return out, attn_weights
            return out

        if return_dual:
            h, _, attn_g, attn_d = self._encode_tabular(x, return_dual=True)
            return self.decoder(h), attn_g, attn_d
        if return_attention:
            h, attn = self._encode_tabular(x, return_attention=True)
            return self.decoder(h), attn
        h = self._encode_tabular(x)
        return self.decoder(h)

    def forward_logits(self, x: torch.Tensor, return_dual: bool = False):
        """Logits bruts avant activation de sortie (pour BCEWithLogitsLoss)."""
        if self.seq_len > 0:
            last_h, attn_g, attn_d = self._embed_temporal(x)
            out = self.decoder.forward_logits(last_h)
            if return_dual:
                return out, attn_g, attn_d
            return out

        if return_dual:
            h, _, attn_g, attn_d = self._encode_tabular(x, return_dual=True)
            return self.decoder.forward_logits(h), attn_g, attn_d
        h = self._encode_tabular(x)
        return self.decoder.forward_logits(h)

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
        task: Optional[str] = None,
        compile: bool = False,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 0,
        use_amp: bool = False,
    ) -> dict[str, list[float]]:
        """
        Entraînement supervisé avec alignement strict train/inférence et mini-batching GPU rapide.
        """
        active_task = task if task is not None else self.task
        self.task = active_task
        self.decoder.task = active_task

        expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
        _validate_input(X, y, expected_w)
        if X_val is not None and y_val is not None:
            _validate_input(X_val, y_val, expected_w)

        if self.auto_normalize:
            self.embedding.set_normalization(X)

        device = next(self.parameters()).device
        amp_active = (device.type == "cuda") and bool(use_amp)

        X_np_f32 = np.asarray(X, dtype=np.float32)
        y_np_f32 = np.asarray(y, dtype=np.float32)
        n_samples = X_np_f32.shape[0]
        use_fast_device_batches = (n_samples <= 50_000)

        if use_fast_device_batches:
            X_dev = torch.tensor(X_np_f32, dtype=torch.float32, device=device)
            y_dev = torch.tensor(y_np_f32, dtype=torch.float32, device=device)
            if y_dev.ndim == 1:
                y_dev = y_dev.unsqueeze(1)
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            loader = None
        else:
            from torch.utils.data import TensorDataset, DataLoader
            X_t = torch.tensor(X_np_f32, dtype=torch.float32)
            y_t = torch.tensor(y_np_f32, dtype=torch.float32)
            if y_t.ndim == 1:
                y_t = y_t.unsqueeze(1)
            gen_cpu = torch.Generator()
            gen_cpu.manual_seed(self.seed)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, generator=gen_cpu)
            X_dev = None
            y_dev = None
            gen = None

        opt_map = {
            "adam": lambda: torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),
            "adamw": lambda: torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay),
            "sgd": lambda: torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
        }
        if optimizer not in opt_map:
            raise ValueError(f"Optimiseur inconnu : '{optimizer}'. Disponibles : {list(opt_map)}")
        opt = opt_map[optimizer]()

        use_logits = (active_task == "classification" and self.output_mode == "standard")
        kpoli_pols = getattr(self.embedding, "kpoli_polarities").detach().cpu().numpy()
        loss_fn = FaImbalanceLoss(
            lambda_fa=lambda_fa,
            pos_weight=pos_weight,
            task=active_task,
            from_logits=use_logits,
            polarity_weights=kpoli_pols,
            polarity_mode=self.polarity_mode,
            seed=self.seed,
        ).to(device)

        scaler = torch.amp.GradScaler("cuda") if amp_active else None
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        if compile:
            self.compile_model()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        X_v_tensor = None
        y_v_tensor = None
        if X_val is not None and y_val is not None:
            X_v_tensor = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
            y_v_tensor = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)
            if y_v_tensor.ndim == 1:
                y_v_tensor = y_v_tensor.unsqueeze(1)

        self.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            if use_fast_device_batches:
                assert X_dev is not None and y_dev is not None and gen is not None
                perm = torch.randperm(n_samples, generator=gen, device=device)
                batch_iter = (
                    (X_dev[perm[i : i + batch_size]], y_dev[perm[i : i + batch_size]])
                    for i in range(0, n_samples, batch_size)
                )
            else:
                assert loader is not None
                batch_iter = ((xb.to(device), yb.to(device)) for xb, yb in loader)

            for X_b, y_b in batch_iter:
                opt.zero_grad(set_to_none=True)

                if amp_active and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        if use_logits:
                            pred, attn_g, attn_d = self.forward_logits(X_b, return_dual=True)
                        else:
                            pred, attn_g, attn_d = self.forward(X_b, return_dual=True)
                        loss = loss_fn(pred, y_b, attn_g, attn_d)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    if use_logits:
                        pred, attn_g, attn_d = self.forward_logits(X_b, return_dual=True)
                    else:
                        pred, attn_g, attn_d = self.forward(X_b, return_dual=True)
                    loss = loss_fn(pred, y_b, attn_g, attn_d)
                    loss.backward()
                    opt.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            val_loss_val = 0.0
            if X_v_tensor is not None and y_v_tensor is not None:
                self.eval()
                with torch.no_grad():
                    if use_logits:
                        val_pred, v_attn_g, v_attn_d = self.forward_logits(X_v_tensor, return_dual=True)
                    else:
                        val_pred, v_attn_g, v_attn_d = self.forward(X_v_tensor, return_dual=True)
                    val_loss_val = float(loss_fn(val_pred, y_v_tensor, v_attn_g, v_attn_d).item())
                history["val_loss"].append(val_loss_val)

                if early_stopping_patience > 0:
                    if val_loss_val < best_val_loss - 1e-5:
                        best_val_loss = val_loss_val
                        best_state = copy.deepcopy(self.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_stopping_patience:
                            if verbose:
                                print(f"  Early stopping à l'époque {epoch}/{epochs} (best val_loss={best_val_loss:.5f})")
                            break
                self.train()

            if verbose and (epoch % verbose_every == 0 or epoch == 1):
                val_str = f" | val_loss={val_loss_val:.5f}" if X_v_tensor is not None else ""
                print(f"  Époque {epoch:4d}/{epochs} | loss={avg_loss:.5f}{val_str}")

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        return history

    @torch.no_grad()
    def predict(self, x: np.ndarray, return_attention: bool = False, return_dual: bool = False):
        """Prédiction numérique accompagnée de l'interprétation symbolique Fâ."""
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
            if return_attention:
                attn_matrix = attn_g.unsqueeze(-1) * attn_d.unsqueeze(-2)
                attn_weights = attn_matrix.reshape(x_t.shape[0], self.n_features, 256)
                return preds.cpu().numpy(), dus, attn_weights.cpu().numpy()
            return preds.cpu().numpy(), dus

        if return_dual:
            h, _, attn_g, attn_d = self._encode_tabular(x_t, return_dual=True)
            preds, dus = self.decoder.forward_with_interpretation(h)
            return preds.cpu().numpy(), dus, attn_g.cpu().numpy(), attn_d.cpu().numpy()
        if return_attention:
            h, attn = self._encode_tabular(x_t, return_attention=True)
            preds, dus = self.decoder.forward_with_interpretation(h)
            return preds.cpu().numpy(), dus, attn.cpu().numpy()
        h = self._encode_tabular(x_t)
        preds, dus = self.decoder.forward_with_interpretation(h)
        return preds.cpu().numpy(), dus

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Prédiction numérique seule."""
        expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
        _validate_input(x, None, expected_w)
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(np.atleast_2d(np.asarray(x, dtype=np.float32)), device=device)
        return self(x_t).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Évalue le réseau sur un jeu de données."""
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
        rmse = float(np.sqrt(mse))

        metrics: dict[str, float] = {"loss": mse, "mae": mae, "rmse": rmse}

        if self.n_outputs == 1 and self.task == "classification":
            y_cls = (pred >= 0.5).int()
            y_true_cls = (y_t >= 0.5).int()
            metrics["accuracy"] = float(torch.mean((y_cls == y_true_cls).float()).item())

        return metrics

    def count_parameters(self) -> int:
        """Retourne le nombre exact de paramètres entraînables du modèle."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def save(self, path: str) -> None:
        """Sauvegarde le modèle complet (state_dict + métadonnées)."""
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
            "task": self.task,
            "output_mode": self.output_mode,
            "polarity_mode": self.polarity_mode,
            "init_mode": self.init_mode,
            "per_feature_queries": self.per_feature_queries,
            "auto_normalize": self.auto_normalize,
            "cross_feature": self.cross_feature,
            "use_residual_mixer": self.use_residual_mixer,
            "version": self.VERSION,
            "kb_hash": get_kb_hash(),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TorchFaNet":
        """Charge un modèle sauvegardé (weights_only=True)."""
        from bokono.core.fa_encoder import get_kb_hash

        data = torch.load(path, weights_only=True)
        current_hash = get_kb_hash()
        saved_hash = data.get("kb_hash")
        if saved_hash and saved_hash != current_hash:
            logger.warning(
                "La base de connaissances active diffère de celle utilisée à l'entraînement."
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
            task=data.get("task", "classification"),
            output_mode=data.get("output_mode", "standard"),
            polarity_mode=data.get("polarity_mode", "fa"),
            init_mode=data.get("init_mode", "fa"),
            per_feature_queries=data.get("per_feature_queries", True),
            auto_normalize=data.get("auto_normalize", True),
            cross_feature=data.get("cross_feature", True),
            use_residual_mixer=data.get("use_residual_mixer", True),
            **kwargs,
        )
        net.load_state_dict(data["state_dict"])
        net.eval()
        return net

    def summary(self) -> None:
        """Affiche le résumé exact de l'architecture (mode tabulaire ou LSTM)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total_params - trainable
        emb_dim = self.n_features * TorchFaEmbedding.FA_EMBED_DIM
        device = next(self.parameters()).device

        print("=" * 65)
        print(f"  FaNet v{self.VERSION} : Réseau neuro-symbolique Fâ (PyTorch)")
        print("=" * 65)
        print(f"  Backend            : PyTorch {torch.__version__} ({device})")
        print(f"  Mode               : {'LSTM Temporel (seq=' + str(self.seq_len) + ')' if self.seq_len > 0 else 'Tabulaire'}")
        print(f"  Tâche / Sortie     : {self.task} / {self.output_mode}")
        print(f"  Polarité / Init    : {self.polarity_mode} / {self.init_mode}")
        print()
        print(f"  {'Couche':<25} {'Forme sortie':<20} {'Paramètres':>12}")
        print("  " + "-" * 60)

        emb_p = sum(p.numel() for p in self.embedding.parameters())
        print(f"  {'TorchFaEmbedding (16x16)':<25} {'(' + str(emb_dim) + ',)':<20} {emb_p:>12,}")

        if self.seq_len > 0 and self.lstm_layer is not None:
            lstm_p = sum(p.numel() for p in self.lstm_layer.parameters())
            out_dim = self.hidden_dim * (2 if self.bidirectional else 1)
            print(f"  {'LSTM Temporel':<25} {'(' + str(out_dim) + ',)':<20} {lstm_p:>12,}")
        else:
            if self.mixer is not None:
                mix_p = sum(p.numel() for p in self.mixer.parameters())
                print(f"  {'FaMixer (FaReLU)':<25} {'(' + str(self.hidden_dim) + ',)':<20} {mix_p:>12,}")
            if self.hidden is not None:
                hidden_p = sum(p.numel() for p in self.hidden.parameters())
                print(f"  {'FaHidden (Résiduel)':<25} {'(' + str(self.hidden_dim) + ',)':<20} {hidden_p:>12,}")

        dec_p = sum(p.numel() for p in self.decoder.parameters())
        print(f"  {'TorchFaDecoder':<25} {'(' + str(self.n_outputs) + ',)':<20} {dec_p:>12,}")

        print("  " + "-" * 60)
        print(f"  {'TOTAL':<25} {'':<20} {total_params:>12,}")
        print(f"  {'  entraînables':<25} {'':<20} {trainable:>12,}")
        print(f"  {'  non-entraînables':<25} {'':<20} {non_trainable:>12,}")
        print("=" * 65)

    def get_params(self, deep: bool = True) -> dict:
        """Retourne les hyperparamètres du modèle."""
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
            "task": self.task,
            "output_mode": self.output_mode,
            "polarity_mode": self.polarity_mode,
            "init_mode": self.init_mode,
            "per_feature_queries": self.per_feature_queries,
            "auto_normalize": self.auto_normalize,
            "cross_feature": self.cross_feature,
            "use_residual_mixer": self.use_residual_mixer,
        }

    def export_onnx(self, path: str, input_shape: tuple[int, ...] | None = None) -> None:
        """
        Exporte la passe avant numérique en ONNX (compatible mode tabulaire et mode LSTM).
        """
        self.eval()
        device = next(self.parameters()).device

        if input_shape is None:
            expected_w = self.seq_len * self.n_features if self.seq_len > 0 else self.n_features
            input_shape = (1, expected_w)

        dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)

        class _OnnxForwardWrapper(nn.Module):
            def __init__(self, core_net: TorchFaNet):
                super().__init__()
                self.core = core_net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.core(x)

        wrapper = _OnnxForwardWrapper(self)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        export_kwargs = dict(
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        try:
            torch.onnx.export(wrapper, dummy_input, path, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(wrapper, dummy_input, path, **export_kwargs)
        logger.info(f"Modèle FaNet v{self.VERSION} exporté en ONNX vers {path}")
